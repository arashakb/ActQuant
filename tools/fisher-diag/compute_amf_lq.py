#!/usr/bin/env python3
"""
Action Fisher (AF) — full VLA forward, action-head loss, optional NLL hybrid.

Computes per-weight Fisher diagonal:

    F_ii = (1/N) Σ_d (∂L/∂w_i)²

over LLM 2D weight matrices (attn q/k/v/o, mlp gate/up/down; lm_head only when α<1).

Two independent knobs control the action loss:

  --action-loss   per-timestep error metric D_t (summed over 7 action dims)
    l1          D_t = Σ_k |a_pred[t,k] - a_gt[t,k]|
    mahalanobis D_t = Σ_k (1/σ_k) |a_pred[t,k] - a_gt[t,k]|   (σ from calib GT)

  --q             aggregation of D_1..D_8 → scalar L_action
    1     L = Σ_t D_t
    2     L = (Σ_t D_t²)^(1/2)
    inf   L = max_t D_t

Hybrid blend with NLL on the 57 action+EOS token IDs:

    L_total = (1 - α) * L_nll  +  α * L_action

  α=1.0 → pure action Fisher (lm_head excluded)
  α=0.0 → pure NLL Fisher restricted to action positions
  α<1.0 → hybrid (lm_head added to target params)

Calibration data: directory produced by get_vla_calib_data.py
  pixel_values.npy / input_ids.npy / attention_mask.npy / proprio.npy /
  targets.npy / oft_gt.npy / metadata.json

Usage:
    python compute_amf_lq.py \\
        --checkpoint /path/to/openvla-oft-checkpoints/oft_combined \\
        --calib-dir  /path/to/openvla-oft/calib_data \\
        --action-loss l1 --q 1 --alpha 1.0 \\
        --spatial-episodes 10 --object-episodes 10 --goal-episodes 10 --long-episodes 30 \\
        --num-gpus 8 --batch-size 2 \\
        --output af_action_only.gguf
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

# gguf-py + openvla-oft imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "gguf-py"))
sys.path.insert(0, str(project_root.parent / "openvla-oft"))

# Local helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_fisher import (
    load_calib_dir,
    load_vla,
    build_batch,
    stratified_frame_indices,
    hf_name_to_gguf_name,
)
from gguf import GGUFWriter


IGNORE_INDEX = -100
DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


# ─── Vision tensor name mapping ───────────────────────────────────────────
# Mirrors tools/openvla-oft/export_openvla_oft.py: HF timm names → vision GGUF
# names (`v.blk.<i>.<role>.weight`). The fused QKV weight is split 3-ways along
# dim 0 to match the exported layout.

_VISION_PATTERNS = [
    (re.compile(r"^blocks\.(\d+)\.attn\.qkv\.weight$"),  "qkv"),
    (re.compile(r"^blocks\.(\d+)\.attn\.proj\.weight$"), "attn_out"),
    (re.compile(r"^blocks\.(\d+)\.mlp\.fc1\.weight$"),   "ffn_up"),
    (re.compile(r"^blocks\.(\d+)\.mlp\.fc2\.weight$"),   "ffn_down"),
]


def hf_to_vision_gguf(hf_name: str):
    """Returns a list of (gguf_name, slicer) pairs for one HF vision param,
    or None if the param isn't a quantizable 2D weight in the vision GGUF.

    `slicer` takes a tensor with the param's full shape and returns the slice
    that should be stored under `gguf_name`. For QKV this returns 3 chunks
    along dim 0; for the rest it's the identity.
    """
    for rx, kind in _VISION_PATTERNS:
        m = rx.match(hf_name)
        if not m:
            continue
        b = m.group(1)
        if kind == "qkv":
            return [
                (f"v.blk.{b}.attn_q.weight", lambda t: t.chunk(3, dim=0)[0].contiguous()),
                (f"v.blk.{b}.attn_k.weight", lambda t: t.chunk(3, dim=0)[1].contiguous()),
                (f"v.blk.{b}.attn_v.weight", lambda t: t.chunk(3, dim=0)[2].contiguous()),
            ]
        return [(f"v.blk.{b}.{kind}.weight", lambda t: t)]
    return None


def get_vision_targets(vla):
    """Returns {'dinov2': [(hf_name, param), ...], 'siglip': [...]}."""
    targets = {}
    for branch, mod in [("dinov2", vla.vision_backbone.featurizer),
                        ("siglip", vla.vision_backbone.fused_featurizer)]:
        params = []
        for name, p in mod.named_parameters():
            if p.ndim != 2: continue
            if hf_to_vision_gguf(name) is None: continue
            params.append((name, p))
        targets[branch] = params
        n = sum(p.numel() for _, p in params)
        print(f"Vision target {branch}: {len(params)} tensors ({n/1e6:.1f}M params)")
    return targets


def init_vision_accumulators(vision_targets):
    accs = {}
    for branch, params in vision_targets.items():
        accs[branch] = {}
        for name, p in params:
            for gguf_name, slicer in hf_to_vision_gguf(name):
                shape = slicer(torch.zeros(p.shape, dtype=torch.float32)).shape
                accs[branch][gguf_name] = torch.zeros(shape, dtype=torch.float32)
    return accs


def accumulate_vision_grads(vision_targets, vision_accs):
    """Add param.grad² (sliced for QKV) to per-branch CPU accumulators."""
    for branch, params in vision_targets.items():
        accs = vision_accs[branch]
        for name, p in params:
            if p.grad is None:
                continue
            sq = p.grad.float().square().cpu()
            for gguf_name, slicer in hf_to_vision_gguf(name):
                accs[gguf_name] += slicer(sq)


# ─── Action Head Loading ─────────────────────────────────────────────────

def load_action_head(checkpoint_dir: str, llm_hidden_dim: int, device: str):
    from prismatic.models.action_heads import L1RegressionActionHead

    candidates = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if "action_head" in f and "checkpoint" in f and f.endswith(".pt")
    ]
    if not candidates:
        raise FileNotFoundError(f"No action_head checkpoint in {checkpoint_dir}")
    ckpt_path = candidates[0]

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    elif isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    fc1_key = next((k for k in sd if k.endswith("model.fc1.weight")), None)
    hidden_dim = sd[fc1_key].shape[0] if fc1_key else 4096

    head = L1RegressionActionHead(input_dim=llm_hidden_dim, hidden_dim=hidden_dim, action_dim=7)
    head.load_state_dict(sd)
    head = head.to(device).eval()  # keep float32 — predict_action gets float32 input
    for p in head.parameters():
        p.requires_grad_(False)
    print(f"Loaded action head ({sum(p.numel() for p in head.parameters())/1e6:.1f}M params, frozen)")
    return head


def get_target_params(vla, include_lm_head: bool = False):
    """LLM 2D weights. lm_head included only when NLL grad needs to flow through it (α<1)."""
    params = []
    for name, param in vla.language_model.named_parameters():
        if param.ndim != 2: continue
        if "embed_tokens" in name: continue
        if "norm" in name: continue
        if not include_lm_head and "lm_head" in name: continue
        params.append((name, param))
    total = sum(p.numel() for _, p in params)
    print(f"Target: {len(params)} LLM tensors ({total/1e9:.2f}B params) "
          f"[lm_head={'included' if include_lm_head else 'excluded'}]")
    return params


def freeze_except_targets(vla, proprio_projector, action_head, target_params, vision_targets=None):
    target_ids = set(id(p) for _, p in target_params)
    if vision_targets:
        for ps in vision_targets.values():
            for _, p in ps:
                target_ids.add(id(p))
    for p in vla.parameters():
        p.requires_grad_(id(p) in target_ids)
    if proprio_projector is not None:
        for p in proprio_projector.parameters():
            p.requires_grad_(False)
    for p in action_head.parameters():
        p.requires_grad_(False)


# ─── Σ_task for Mahalanobis ───────────────────────────────────────────────

def compute_sigma_inv(action_targets: np.ndarray) -> np.ndarray:
    """Per-action-dim std → 1/σ weights. action_targets: (N, 8, 7)."""
    flat = action_targets.reshape(-1, 7)
    sigma = flat.std(axis=0) + 1e-8
    sigma_inv = (1.0 / sigma).astype(np.float32)
    print("\nΣ_task (1/σ weights for Mahalanobis):")
    print(f"  {'dim':<10} {'σ':>10} {'1/σ':>10}")
    for k, nm in enumerate(DIM_NAMES):
        print(f"  {nm:<10} {sigma[k]:>10.4f} {sigma_inv[k]:>10.4f}")
    return sigma_inv


# ─── Loss Computations ────────────────────────────────────────────────────

def compute_action_loss(a_pred, a_gt, action_loss, q, sigma_inv_t):
    """a_pred, a_gt: (B, 8, 7). Returns per-frame mean scalar.

    action_loss:
      l1          D_t = Σ_k |a_pred[t,k] - a_gt[t,k]|
      l2          D_t = Σ_k (a_pred[t,k] - a_gt[t,k])²   (squared L2, no sqrt — Gauss-Newton form)
      mahalanobis D_t = Σ_k (1/σ_k) · |a_pred[t,k] - a_gt[t,k]|
    """
    B = a_pred.shape[0]
    if action_loss == "l2":
        D = (a_pred - a_gt).pow(2).sum(dim=-1)              # (B, 8)
    elif action_loss == "mahalanobis":
        D = ((a_pred - a_gt).abs() * sigma_inv_t).sum(dim=-1)
    else:  # l1
        D = (a_pred - a_gt).abs().sum(dim=-1)
    if q == float("inf"):
        raw = D.max(dim=1).values.sum()
    elif q == 1.0:
        raw = D.sum()
    else:
        raw = D.pow(q).sum(dim=1).pow(1.0 / q).sum()
    return raw / B


def compute_nll_loss(lm_head, nll_h, tok_tgts):
    """nll_h: (B, 57, hd), tok_tgts: (B, 57). Returns per-token mean CE."""
    logits = lm_head(nll_h.bfloat16()).float()  # (B, 57, V)
    V = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, V), tok_tgts.reshape(-1))


# ─── Full VLA Forward → hidden states ─────────────────────────────────────

def vla_hidden_states(vla, proprio_projector, pixel_values, input_ids, attention_mask, proprio, labels, n_vision_tokens):
    """Run vision + projector via calibration_mode, then LLM base → last_hidden_state.

    Returns:
        hs:           (B, mm_seq_len, hd) bf16  — Llama base output
        real_mm_lens: (B,) long              — real (non-padded) mm length per row
    """
    out_calib = vla(
        input_ids=input_ids, attention_mask=attention_mask,
        pixel_values=pixel_values, proprio=proprio,
        proprio_projector=proprio_projector,
        labels=labels, calibration_mode=True,
    )
    mm_emb = out_calib["multimodal_embeddings"]  # (B, mm_seq_len, hd)
    B, mm_seq_len, _ = mm_emb.shape

    real_input_lens = attention_mask.sum(dim=1)             # (B,)
    real_mm_lens = (n_vision_tokens + real_input_lens).long()
    mm_attn = torch.zeros(B, mm_seq_len, dtype=torch.long, device=mm_emb.device)
    for i in range(B):
        mm_attn[i, : real_mm_lens[i]] = 1

    hs = vla.language_model.model(
        inputs_embeds=mm_emb,
        attention_mask=mm_attn,
        use_cache=False,
    ).last_hidden_state  # (B, mm_seq_len, hd)
    return hs, real_mm_lens


def get_n_vision_tokens(vla, num_images_in_input: int, use_proprio: bool) -> int:
    n = vla.vision_backbone.get_num_patches() * num_images_in_input
    if use_proprio:
        n += 1
    return n


# ─── Per-Batch Forward + Backward + Accumulate ────────────────────────────

def per_batch_step_multi(
    vla, proprio_projector, action_head, target_params,
    arrs, batch_indices, device,
    n_vision_tokens, n_action_tokens,
    accumulators,
    vision_targets=None, vision_accs=None,
):
    """Multi-loss Fisher sum: F_total = F_L1 + F_L2 + F_NLL.

    Per batch: 1 forward, 3 separate backwards (with retain_graph), each accumulates
    its own squared grads into the SAME accumulator dict → final Fisher is the sum.
    Returns dict with per-loss scalar values.
    """
    pixel_values, input_ids, attention_mask, labels, proprio = build_batch(arrs, batch_indices, device)
    gt = torch.from_numpy(np.ascontiguousarray(arrs["oft_gt"][batch_indices])).to(device, dtype=torch.float32)
    nll_tgts = torch.from_numpy(np.ascontiguousarray(arrs["targets"][batch_indices])).to(device, dtype=torch.long)

    vla.zero_grad(set_to_none=True)
    hs, real_mm_lens = vla_hidden_states(
        vla, proprio_projector,
        pixel_values, input_ids, attention_mask, proprio, labels,
        n_vision_tokens,
    )
    B = hs.shape[0]
    act_h = torch.stack([
        hs[i, real_mm_lens[i] - n_action_tokens - 1 : real_mm_lens[i] - 1]
        for i in range(B)
    ])
    nll_h = torch.stack([
        hs[i, real_mm_lens[i] - n_action_tokens - 2 : real_mm_lens[i] - 1]
        for i in range(B)
    ])
    a_pred = action_head.predict_action(act_h.float())

    diff = a_pred - gt
    losses = [
        ("l1",  diff.abs().sum() / B),
        ("l2",  diff.pow(2).sum() / B),
        ("nll", compute_nll_loss(vla.language_model.lm_head, nll_h, nll_tgts)),
    ]
    loss_vals = {}
    for i, (name, loss) in enumerate(losses):
        loss_vals[name] = loss.item()
        is_last = (i == len(losses) - 1)
        loss.backward(retain_graph=not is_last)
        for pname, p in target_params:
            if p.grad is not None:
                accumulators[hf_name_to_gguf_name(pname)] += p.grad.float().square().cpu()
        if vision_targets is not None and vision_accs is not None:
            accumulate_vision_grads(vision_targets, vision_accs)
        vla.zero_grad(set_to_none=True)

    del pixel_values, input_ids, attention_mask, labels, proprio, gt, nll_tgts
    del hs, act_h, nll_h, a_pred, diff
    torch.cuda.empty_cache()
    return loss_vals


def per_batch_step(
    vla, proprio_projector, action_head, target_params,
    arrs, batch_indices, device,
    n_vision_tokens, n_action_tokens,
    action_loss, q, alpha, sigma_inv_t,
    accumulators,
    vision_targets=None, vision_accs=None,
):
    """One forward+backward over a batch, accumulate squared grads on CPU.

    Returns (total_loss, action_loss, nll_loss) as floats.
    """
    pixel_values, input_ids, attention_mask, labels, proprio = build_batch(arrs, batch_indices, device)

    # Ground-truth continuous actions for this batch
    gt_np = arrs["oft_gt"][batch_indices]
    gt = torch.from_numpy(np.ascontiguousarray(gt_np)).to(device, dtype=torch.float32)

    # Token IDs for NLL (only needed when α<1)
    nll_tgts = None
    if alpha < 1.0:
        tgt_np = arrs["targets"][batch_indices]  # (B, 57) int64
        nll_tgts = torch.from_numpy(np.ascontiguousarray(tgt_np)).to(device, dtype=torch.long)

    vla.zero_grad(set_to_none=True)
    hs, real_mm_lens = vla_hidden_states(
        vla, proprio_projector,
        pixel_values, input_ids, attention_mask, proprio, labels,
        n_vision_tokens,
    )
    B = hs.shape[0]

    # Action positions: last 56 valid before EOS
    act_h = torch.stack([
        hs[i, real_mm_lens[i] - n_action_tokens - 1 : real_mm_lens[i] - 1]
        for i in range(B)
    ])  # (B, 56, hd)
    a_pred = action_head.predict_action(act_h.float())  # (B, 8, 7)
    a_loss = compute_action_loss(a_pred, gt, action_loss, q, sigma_inv_t)

    if alpha < 1.0:
        nll_h = torch.stack([
            hs[i, real_mm_lens[i] - n_action_tokens - 2 : real_mm_lens[i] - 1]
            for i in range(B)
        ])  # (B, 57, hd)
        n_loss = compute_nll_loss(vla.language_model.lm_head, nll_h, nll_tgts)
    else:
        n_loss = torch.tensor(0.0, device=device)

    loss = (1.0 - alpha) * n_loss + alpha * a_loss
    a_val = a_loss.item()
    n_val = n_loss.item()
    total = loss.item()

    loss.backward()

    for name, param in target_params:
        if param.grad is not None:
            gguf_name = hf_name_to_gguf_name(name)
            accumulators[gguf_name] += param.grad.float().square().cpu()

    if vision_targets is not None and vision_accs is not None:
        accumulate_vision_grads(vision_targets, vision_accs)

    vla.zero_grad(set_to_none=True)
    del pixel_values, input_ids, attention_mask, labels, proprio, gt, nll_tgts
    del hs, act_h, a_pred, a_loss, n_loss, loss
    torch.cuda.empty_cache()
    return total, a_val, n_val


def init_accumulators(target_params):
    return {
        hf_name_to_gguf_name(name): torch.zeros(p.shape, dtype=torch.float32)
        for name, p in target_params
    }


def finalize_fisher(accumulators, num_samples):
    fisher = {gn: (acc / num_samples).numpy() for gn, acc in accumulators.items()}
    return fisher


def print_fisher_summary(fisher, target_params):
    print(f"\n{'Tensor':<45s} {'Shape':>18s} {'Mean':>12s} {'Max':>12s}")
    print("-" * 90)
    for name, _ in target_params:
        gn = hf_name_to_gguf_name(name)
        fd = fisher[gn]
        print(f"{gn:<45s} {str(tuple(fd.shape)):>18s} "
              f"{fd.mean():>12.4e} {fd.max():>12.4e}")


# ─── Single-GPU Loop ──────────────────────────────────────────────────────

def compute_af_single_gpu(
    checkpoint_path, calib_dir, batch_size, num_samples, base_seed, device,
    num_images_in_input, action_loss, q, alpha, episode_filter, multi_loss=False,
):
    calib = load_calib_dir(calib_dir)
    calib["arrs"]["oft_gt"]  = np.load(calib_dir / "oft_gt.npy",  mmap_mode="r")
    total = calib["num_frames"]

    if episode_filter:
        print(f"Stratified episode filter: {episode_filter}")
        indices = np.sort(stratified_frame_indices(calib["meta"], episode_filter))
        num_samples = len(indices)
    elif num_samples > 0 and num_samples < total:
        rng = np.random.RandomState(base_seed)
        indices = np.sort(rng.choice(total, num_samples, replace=False))
    else:
        indices = np.arange(total)
        num_samples = total
    print(f"Using {num_samples}/{total} frames")

    # Σ_task only matters for action loss = mahalanobis
    sigma_inv = None
    sigma_inv_t = None
    if action_loss == "mahalanobis":
        sigma_inv = compute_sigma_inv(calib["arrs"]["oft_gt"][indices])
        sigma_inv_t = torch.tensor(sigma_inv, dtype=torch.float32, device=device)

    vla, proprio_projector = load_vla(checkpoint_path, device, num_images_in_input)
    action_head = load_action_head(checkpoint_path, vla.llm_dim, device)
    # Multi-loss mode always needs lm_head (NLL term flows through it)
    include_lm_head = (alpha < 1.0) or multi_loss
    target_params = get_target_params(vla, include_lm_head=include_lm_head)
    vision_targets = get_vision_targets(vla)
    freeze_except_targets(vla, proprio_projector, action_head, target_params, vision_targets)

    n_vision = get_n_vision_tokens(vla, num_images_in_input, use_proprio=(proprio_projector is not None))
    print(f"n_vision_tokens (incl. proprio) = {n_vision}")

    accumulators = init_accumulators(target_params)
    vision_accs = init_vision_accumulators(vision_targets)
    losses, action_losses, nll_losses = [], [], []
    total_batches = (num_samples + batch_size - 1) // batch_size
    t0 = time.time()

    if multi_loss:
        print(f"\n{'Batch':>10s} {'L1':>10s} {'L2':>10s} {'NLL':>10s} {'Rate':>12s} {'ETA':>9s}")
    else:
        print(f"\n{'Batch':>10s} {'Total':>10s} {'Action':>10s} {'NLL':>10s} {'Rate':>12s} {'ETA':>9s}")
    print("-" * 70)

    for bi, start in enumerate(range(0, num_samples, batch_size)):
        end = min(start + batch_size, num_samples)
        batch_indices = np.sort(indices[start:end])
        if multi_loss:
            lv = per_batch_step_multi(
                vla, proprio_projector, action_head, target_params,
                calib["arrs"], batch_indices, device,
                n_vision, 56, accumulators,
                vision_targets=vision_targets, vision_accs=vision_accs,
            )
            total_l = lv["l1"] + lv["l2"] + lv["nll"]
            a_l = lv["l1"]
            n_l = lv["nll"]
        else:
            total_l, a_l, n_l = per_batch_step(
                vla, proprio_projector, action_head, target_params,
                calib["arrs"], batch_indices, device,
                n_vision, 56, action_loss, q, alpha, sigma_inv_t,
                accumulators,
                vision_targets=vision_targets, vision_accs=vision_accs,
            )
        losses.append(total_l); action_losses.append(a_l); nll_losses.append(n_l)
        elapsed = time.time() - t0
        rate = (bi + 1) / elapsed * 60 if elapsed > 0 else 0
        eta = (total_batches - bi - 1) / (rate / 60) if rate > 0 else 0
        print(f"{bi+1:>4d}/{total_batches:<4d}  {total_l:>10.4f} {a_l:>10.4f} {n_l:>10.4f} "
              f"{rate:>8.1f} b/m {eta:>8.0f}s")

    fisher = finalize_fisher(accumulators, num_samples)
    vision_fisher = {
        branch: {gn: (acc / num_samples).numpy() for gn, acc in accs.items()}
        for branch, accs in vision_accs.items()
    }
    print_fisher_summary(fisher, target_params)
    stats = {
        "num_samples": num_samples,
        "loss_mean":    float(np.mean(losses)),
        "loss_std":     float(np.std(losses)),
        "action_mean":  float(np.mean(action_losses)),
        "nll_mean":     float(np.mean(nll_losses)),
        "total_time_sec": time.time() - t0,
    }
    return fisher, vision_fisher, stats, sigma_inv


# ─── Multi-GPU Worker ─────────────────────────────────────────────────────

def _gpu_worker(
    gpu_id, checkpoint_path, calib_dir, result_dir, batch_size,
    num_images_in_input, action_loss, q, alpha, multi_loss=False,
):
    import pickle
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"

    with open(os.path.join(result_dir, f"gpu_{gpu_id}_data.pkl"), "rb") as f:
        data = pickle.load(f)
    indices = data["indices"]
    sigma_inv = data["sigma_inv"]

    calib = load_calib_dir(Path(calib_dir))
    calib["arrs"]["oft_gt"] = np.load(Path(calib_dir) / "oft_gt.npy", mmap_mode="r")
    print(f"  [GPU {gpu_id}] Processing {len(indices)} frames  multi_loss={multi_loss}")

    sigma_inv_t = (torch.tensor(sigma_inv, dtype=torch.float32, device=device)
                   if sigma_inv is not None else None)

    vla, proprio_projector = load_vla(checkpoint_path, device, num_images_in_input)
    action_head = load_action_head(checkpoint_path, vla.llm_dim, device)
    include_lm_head = (alpha < 1.0) or multi_loss
    target_params = get_target_params(vla, include_lm_head=include_lm_head)
    vision_targets = get_vision_targets(vla)
    freeze_except_targets(vla, proprio_projector, action_head, target_params, vision_targets)
    n_vision = get_n_vision_tokens(vla, num_images_in_input, proprio_projector is not None)

    accumulators = init_accumulators(target_params)
    vision_accs = init_vision_accumulators(vision_targets)
    losses, action_losses, nll_losses = [], [], []
    total_batches = (len(indices) + batch_size - 1) // batch_size
    t0 = time.time()

    for bi, start in enumerate(range(0, len(indices), batch_size)):
        end = min(start + batch_size, len(indices))
        batch_indices = np.sort(indices[start:end])
        if multi_loss:
            lv = per_batch_step_multi(
                vla, proprio_projector, action_head, target_params,
                calib["arrs"], batch_indices, device,
                n_vision, 56, accumulators,
                vision_targets=vision_targets, vision_accs=vision_accs,
            )
            total_l = lv["l1"] + lv["l2"] + lv["nll"]
            a_l = lv["l1"]
            n_l = lv["nll"]
        else:
            total_l, a_l, n_l = per_batch_step(
                vla, proprio_projector, action_head, target_params,
                calib["arrs"], batch_indices, device,
                n_vision, 56, action_loss, q, alpha, sigma_inv_t,
                accumulators,
                vision_targets=vision_targets, vision_accs=vision_accs,
            )
        losses.append(total_l); action_losses.append(a_l); nll_losses.append(n_l)
        elapsed = time.time() - t0
        rate = (bi + 1) / elapsed * 60 if elapsed > 0 else 0
        eta = (total_batches - bi - 1) / (rate / 60) if rate > 0 else 0
        print(f"  [GPU {gpu_id}] {bi+1}/{total_batches} total={total_l:.4f} act={a_l:.4f} "
              f"nll={n_l:.4f} {rate:.1f} b/m ETA {eta:.0f}s", flush=True)

    print(f"  [GPU {gpu_id}] Done {len(indices)} frames in {time.time()-t0:.1f}s")

    gpu_dir = os.path.join(result_dir, f"gpu_{gpu_id}")
    os.makedirs(gpu_dir, exist_ok=True)
    np.save(os.path.join(gpu_dir, "num_frames.npy"),     np.array(len(indices)))
    np.save(os.path.join(gpu_dir, "losses.npy"),         np.array(losses))
    np.save(os.path.join(gpu_dir, "action_losses.npy"),  np.array(action_losses))
    np.save(os.path.join(gpu_dir, "nll_losses.npy"),     np.array(nll_losses))
    name_map = {}
    for gguf_name, acc in accumulators.items():
        safe = gguf_name.replace(".", "_").replace("/", "_")
        name_map[safe] = gguf_name
        np.save(os.path.join(gpu_dir, f"acc_{safe}.npy"), acc.numpy())
    with open(os.path.join(gpu_dir, "name_map.json"), "w") as f:
        json.dump(name_map, f)

    for branch, accs in vision_accs.items():
        vdir = os.path.join(gpu_dir, f"vision_{branch}")
        os.makedirs(vdir, exist_ok=True)
        vmap = {}
        for gguf_name, acc in accs.items():
            safe = gguf_name.replace(".", "_").replace("/", "_")
            vmap[safe] = gguf_name
            np.save(os.path.join(vdir, f"acc_{safe}.npy"), acc.numpy())
        with open(os.path.join(vdir, "name_map.json"), "w") as f:
            json.dump(vmap, f)

    del vla, proprio_projector, action_head, target_params, accumulators
    del vision_targets, vision_accs
    torch.cuda.empty_cache()
    return gpu_dir


def compute_af_multi_gpu(
    num_gpus, gpu_ids, checkpoint_path, calib_dir, batch_size, num_samples, base_seed,
    output_path, num_images_in_input, action_loss, q, alpha, episode_filter, multi_loss=False,
):
    import pickle, shutil
    result_dir = output_path.replace(".gguf", "_gpu_results")
    os.makedirs(result_dir, exist_ok=True)

    calib = load_calib_dir(Path(calib_dir))
    calib["arrs"]["oft_gt"] = np.load(Path(calib_dir) / "oft_gt.npy", mmap_mode="r")
    total = calib["num_frames"]

    if episode_filter:
        print(f"Stratified episode filter: {episode_filter}")
        all_indices = np.sort(stratified_frame_indices(calib["meta"], episode_filter))
        num_samples = len(all_indices)
    elif num_samples > 0 and num_samples < total:
        rng = np.random.RandomState(base_seed)
        all_indices = np.sort(rng.choice(total, num_samples, replace=False))
    else:
        all_indices = np.arange(total)
        num_samples = total
    print(f"Selected {num_samples}/{total} frames")

    sigma_inv = (compute_sigma_inv(calib["arrs"]["oft_gt"][all_indices])
                 if action_loss == "mahalanobis" else None)

    chunks = np.array_split(all_indices, num_gpus)
    for i, gid in enumerate(gpu_ids):
        with open(os.path.join(result_dir, f"gpu_{gid}_data.pkl"), "wb") as f:
            pickle.dump({"indices": chunks[i], "sigma_inv": sigma_inv}, f)
        print(f"  GPU {gid}: {len(chunks[i])} frames")

    print(f"\nLaunching {num_gpus} GPU workers...")
    t0 = time.time()
    mp.set_start_method("spawn", force=True)
    with mp.Pool(num_gpus) as pool:
        futures = [
            pool.apply_async(
                _gpu_worker,
                args=(gpu_ids[i], checkpoint_path, str(calib_dir), result_dir,
                      batch_size, num_images_in_input, action_loss, q, alpha, multi_loss),
            )
            for i in range(num_gpus)
        ]
        result_paths = [f.get(timeout=36000) for f in futures]

    total_time = time.time() - t0
    print(f"\nAll GPUs done in {total_time:.1f}s ({total_time/60:.1f}min)")

    print("Merging...")
    merged = {}
    vision_merged = {"dinov2": {}, "siglip": {}}
    total_count = 0
    all_losses, all_action, all_nll = [], [], []
    for gpu_dir in result_paths:
        total_count += int(np.load(os.path.join(gpu_dir, "num_frames.npy")))
        all_losses.extend(np.load(os.path.join(gpu_dir, "losses.npy")).tolist())
        all_action.extend(np.load(os.path.join(gpu_dir, "action_losses.npy")).tolist())
        all_nll.extend(np.load(os.path.join(gpu_dir, "nll_losses.npy")).tolist())
        with open(os.path.join(gpu_dir, "name_map.json")) as f:
            name_map = json.load(f)
        for safe, gn in name_map.items():
            acc = np.load(os.path.join(gpu_dir, f"acc_{safe}.npy"))
            if gn not in merged:
                merged[gn] = acc.astype(np.float64)
            else:
                merged[gn] += acc.astype(np.float64)
        for branch in vision_merged:
            vdir = os.path.join(gpu_dir, f"vision_{branch}")
            map_path = os.path.join(vdir, "name_map.json")
            if not os.path.exists(map_path):
                continue
            with open(map_path) as f:
                vmap = json.load(f)
            for safe, gn in vmap.items():
                acc = np.load(os.path.join(vdir, f"acc_{safe}.npy"))
                if gn not in vision_merged[branch]:
                    vision_merged[branch][gn] = acc.astype(np.float64)
                else:
                    vision_merged[branch][gn] += acc.astype(np.float64)

    fisher = {gn: (m / total_count).astype(np.float32) for gn, m in merged.items()}
    vision_fisher = {
        branch: {gn: (m / total_count).astype(np.float32) for gn, m in d.items()}
        for branch, d in vision_merged.items()
    }
    print(f"\n{'Tensor':<45s} {'Shape':>18s} {'Mean':>12s} {'Max':>12s}")
    print("-" * 90)
    for gn in sorted(fisher.keys()):
        fd = fisher[gn]
        print(f"{gn:<45s} {str(tuple(fd.shape)):>18s} "
              f"{fd.mean():>12.4e} {fd.max():>12.4e}")

    stats = {
        "num_samples":    total_count,
        "loss_mean":      float(np.mean(all_losses)),
        "loss_std":       float(np.std(all_losses)),
        "action_mean":    float(np.mean(all_action)),
        "nll_mean":       float(np.mean(all_nll)),
        "total_time_sec": total_time,
        "num_gpus":       num_gpus,
    }
    print(f"\nLoss: total={stats['loss_mean']:.4f}  action={stats['action_mean']:.4f}  nll={stats['nll_mean']:.4f}")
    print(f"Time: {total_time:.1f}s across {num_gpus} GPUs")

    shutil.rmtree(result_dir, ignore_errors=True)
    return fisher, vision_fisher, stats, sigma_inv


# ─── GGUF Output ──────────────────────────────────────────────────────────

def save_vision_amf_gguf(fisher, output_path, num_samples, calib_source, alpha, branch):
    """Write a vision-branch AMF GGUF in the imatrix-compatible layout.

    Per tensor: <gguf_name>.in_sum2 (float32, [d_out, d_in]) and .counts ([[1.0]]).
    """
    if not fisher:
        print(f"  [{branch}] no Fisher entries — skipping {output_path}")
        return
    writer = GGUFWriter(output_path, arch="")
    writer.add_string("general.type",   "af_diag_vision")
    writer.add_string("af.branch",      branch)
    writer.add_float32("af.alpha",      alpha)
    writer.add_array("imatrix.datasets", [calib_source])
    writer.add_uint32("imatrix.chunk_count", num_samples)
    writer.add_uint32("imatrix.chunk_size",  1)
    writer.add_uint32("af.num_samples",      num_samples)

    for name in sorted(fisher):
        fd = np.maximum(fisher[name], 0.0).astype(np.float32)
        writer.add_tensor(f"{name}.in_sum2", fd)
        writer.add_tensor(f"{name}.counts",  np.array([[1.0]], dtype=np.float32))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"Saved {output_path} ({size_mb:.1f} MB) branch={branch} entries={len(fisher)} α={alpha}")


def save_af_gguf(fisher, output_path, num_samples, calib_source, action_loss, q, alpha, sigma_inv):
    writer = GGUFWriter(output_path, arch="")
    q_label = "inf" if q == float("inf") else str(q)
    mode = "hybrid" if 0 < alpha < 1 else ("nll" if alpha == 0 else "action")

    writer.add_string("general.type",   "af_diag")
    writer.add_string("af.mode",        mode)
    writer.add_string("af.action_loss", action_loss)
    writer.add_string("af.q",           q_label)
    writer.add_float32("af.alpha",      alpha)
    writer.add_array("imatrix.datasets", [calib_source])
    writer.add_uint32("imatrix.chunk_count", num_samples)
    writer.add_uint32("imatrix.chunk_size",  1)
    writer.add_uint32("af.num_samples",      num_samples)
    if sigma_inv is not None:
        writer.add_array("af.sigma_inv", sigma_inv.tolist())

    for name in sorted(fisher):
        fd = np.maximum(fisher[name], 0.0).astype(np.float32)
        writer.add_tensor(f"{name}.in_sum2", fd)
        writer.add_tensor(f"{name}.counts",  np.array([[1.0]], dtype=np.float32))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    size_gb = os.path.getsize(output_path) / (1024 ** 3)
    print(f"Saved {output_path} ({size_gb:.2f} GB)  mode={mode} action={action_loss} q={q_label} α={alpha}")


# ─── CLI ──────────────────────────────────────────────────────────────────

def parse_q(s: str) -> float:
    if s.lower() in ("inf", "infinity"):
        return float("inf")
    v = float(s)
    if v < 1.0:
        raise argparse.ArgumentTypeError("q must be >= 1 or 'inf'")
    return v


def main():
    parser = argparse.ArgumentParser(description="Action Fisher (full VLA forward)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calib-dir",  required=True)
    parser.add_argument("--output",     default="af.gguf")
    parser.add_argument("--action-loss", choices=["l1", "l2", "mahalanobis"], default="l1")
    parser.add_argument("--q", type=parse_q, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="Action weight: 1.0=pure action, 0.0=pure NLL, 0.6=hybrid (default)")
    parser.add_argument("--num-samples", type=int, default=0,
                        help="Frames to use (0 = all). Ignored if episode filter set.")
    parser.add_argument("--batch-size",  type=int, default=1)
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--base-seed",   type=int, default=42)
    parser.add_argument("--num-gpus",    type=int, default=1)
    parser.add_argument("--gpus",        default=None)
    parser.add_argument("--num-images-in-input", type=int, default=2)
    parser.add_argument("--spatial-episodes", type=int, default=-1)
    parser.add_argument("--object-episodes",  type=int, default=-1)
    parser.add_argument("--goal-episodes",    type=int, default=-1)
    parser.add_argument("--long-episodes",    type=int, default=-1)
    parser.add_argument("--multi-loss", action="store_true",
                        help="Multi-loss Fisher sum: F = F_L1 + F_L2 + F_NLL "
                             "(ignores --action-loss/--q/--alpha; always includes lm_head)")
    args = parser.parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError(f"--alpha must be in [0, 1], got {args.alpha}")

    if args.gpus:
        gpu_ids = [int(x) for x in args.gpus.split(",")]
        args.num_gpus = len(gpu_ids)
    else:
        gpu_ids = list(range(args.num_gpus))

    episode_filter = {}
    if args.spatial_episodes > 0: episode_filter["spatial"] = args.spatial_episodes
    if args.object_episodes  > 0: episode_filter["object"]  = args.object_episodes
    if args.goal_episodes    > 0: episode_filter["goal"]    = args.goal_episodes
    if args.long_episodes    > 0: episode_filter["long"]    = args.long_episodes

    q_label = "inf" if args.q == float("inf") else str(args.q)
    if args.multi_loss:
        mode = "multi_loss_sum"
    else:
        mode = "hybrid" if 0 < args.alpha < 1 else ("nll" if args.alpha == 0 else "action")
    print("=" * 70)
    print("Action Fisher (full VLA forward)")
    print("=" * 70)
    print(f"  mode         = {mode}")
    print(f"  action_loss  = {args.action_loss}")
    print(f"  q            = {q_label}")
    print(f"  alpha        = {args.alpha}")
    print(f"  calib_dir    = {args.calib_dir}")
    print(f"  episode_filter = {episode_filter or 'none (use all)'}")
    print(f"  batch_size   = {args.batch_size}  num_gpus = {args.num_gpus}  gpu_ids = {gpu_ids}")
    print()

    calib_dir = Path(args.calib_dir).expanduser()
    t_start = time.time()

    if args.num_gpus > 1:
        fisher, vision_fisher, stats, sigma_inv = compute_af_multi_gpu(
            num_gpus=args.num_gpus, gpu_ids=gpu_ids,
            checkpoint_path=args.checkpoint, calib_dir=calib_dir,
            batch_size=args.batch_size, num_samples=args.num_samples,
            base_seed=args.base_seed, output_path=args.output,
            num_images_in_input=args.num_images_in_input,
            action_loss=args.action_loss, q=args.q, alpha=args.alpha,
            episode_filter=episode_filter,
            multi_loss=args.multi_loss,
        )
    else:
        fisher, vision_fisher, stats, sigma_inv = compute_af_single_gpu(
            checkpoint_path=args.checkpoint, calib_dir=calib_dir,
            batch_size=args.batch_size, num_samples=args.num_samples,
            base_seed=args.base_seed, device=args.device,
            num_images_in_input=args.num_images_in_input,
            action_loss=args.action_loss, q=args.q, alpha=args.alpha,
            episode_filter=episode_filter,
            multi_loss=args.multi_loss,
        )
    stats["total_time_sec"] = time.time() - t_start

    print(f"\n--- Saving GGUF (LLM) ---")
    save_af_gguf(fisher, args.output, stats["num_samples"], str(calib_dir),
                 args.action_loss, args.q, args.alpha, sigma_inv)

    print(f"\n--- Saving GGUF (vision) ---")
    if args.output.endswith(".gguf"):
        stem = args.output[: -len(".gguf")]
    else:
        stem = args.output
    for branch in ("dinov2", "siglip"):
        vpath = f"{stem}_{branch}.gguf"
        save_vision_amf_gguf(vision_fisher.get(branch, {}), vpath,
                             stats["num_samples"], str(calib_dir),
                             args.alpha, branch)

    config = {
        "method":           "action_fisher_full_vla",
        "mode":             mode,
        "action_loss":      args.action_loss,
        "q":                q_label,
        "alpha":            args.alpha,
        "checkpoint":       args.checkpoint,
        "calib_dir":        str(calib_dir),
        "episode_filter":   episode_filter,
        "num_samples":      stats["num_samples"],
        "batch_size":       args.batch_size,
        "base_seed":        args.base_seed,
        "num_gpus":         args.num_gpus,
        "loss_mean":        stats["loss_mean"],
        "action_mean":      stats["action_mean"],
        "nll_mean":         stats["nll_mean"],
        "total_time_sec":   stats["total_time_sec"],
    }
    cfg_path = args.output.replace(".gguf", "_config.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {cfg_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
