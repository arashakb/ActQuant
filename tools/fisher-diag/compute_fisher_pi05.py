#!/usr/bin/env python3
"""
compute_fisher_pi05.py

Compute the empirical Fisher diagonal for the PaliGemma LLM weights of Pi 0.5,
using flow-matching loss as the natural NLL analog.

Loss: L_flow = ||v_theta(x_t, t, prefix) - (x_0 - x_1)||^2
  with x_0 ~ N(0, I), t ~ Beta(1.5, 1.0), x_t = t*x_0 + (1-t)*x_1

Per-batch:
    1. Single forward through PI0Pytorch (vision + LLM + action expert)
    2. Single backward via standard autograd
    3. Accumulate (param.grad)^2 for each target LLM weight tensor

Output: GGUF file in llama-imatrix format (drop-in for `llama-quantize --imatrix`).

The output tensor names match llama-arch / export_pi05_llm.py:
    blk.{i}.attn_q.weight       blk.{i}.ffn_gate.weight
    blk.{i}.attn_k.weight       blk.{i}.ffn_up.weight
    blk.{i}.attn_v.weight       blk.{i}.ffn_down.weight
    blk.{i}.attn_output.weight

Usage:
    ${HOME}/miniconda3/envs/openpi-server/bin/python \
        tools/fisher-diag/compute_fisher_pi05.py \
        --checkpoint /path/to/openpi/pi05_libero_base_pytorch \
        --calib-dir  /path/to/openpi/calib_data_raw \
        --output     /path/to/openpi/pi05_libero_base_gguf/fisher_flow.gguf \
        --num-gpus 8 --batch-size 1 --num-samples 0
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp


# ─── Path bootstrap ───────────────────────────────────────────────────────────
THIS = Path(__file__).resolve()
REPO = THIS.parent.parent.parent
sys.path.insert(0, str(REPO / "gguf-py"))


# ─── HF -> GGUF name mapping (matches export_pi05_llm.py) ─────────────────────
# Runtime PyTorch path: paligemma_with_expert.paligemma.language_model.layers.{i}.<...>
# (note: NOT "model.language_model" — PaliGemma unwraps the inner "model" at runtime)

def hf_to_gguf_llm(name: str) -> str | None:
    """Map a PI0Pytorch language_model parameter name to llama-imatrix tensor name.
    Returns None if the parameter is not a 2-D LLM weight we want to track."""
    prefix = "paligemma_with_expert.paligemma.model.language_model.layers."
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix):]
    # rest is "{i}.<...>"
    try:
        i_str, sub = rest.split(".", 1)
        i = int(i_str)
    except ValueError:
        return None

    SUFFIX_MAP = {
        "self_attn.q_proj.weight":   f"blk.{i}.attn_q.weight",
        "self_attn.k_proj.weight":   f"blk.{i}.attn_k.weight",
        "self_attn.v_proj.weight":   f"blk.{i}.attn_v.weight",
        "self_attn.o_proj.weight":   f"blk.{i}.attn_output.weight",
        "mlp.gate_proj.weight":      f"blk.{i}.ffn_gate.weight",
        "mlp.up_proj.weight":        f"blk.{i}.ffn_up.weight",
        "mlp.down_proj.weight":      f"blk.{i}.ffn_down.weight",
    }
    return SUFFIX_MAP.get(sub)


def hf_to_gguf_vision(name: str) -> str | None:
    """Map a PI0Pytorch SigLIP vision param name to the GGUF tensor name used
    by export_pi05.py (v.blk.{i}.*). Returns None for non-2-D / non-tracked
    weights. The projector (mm.*) and action head are intentionally NOT tracked
    — they stay unquantized.

    PyTorch:  paligemma_with_expert.paligemma.model.vision_tower.vision_model
              .encoder.layers.{i}.{self_attn.{q,k,v,out}_proj | mlp.{fc1,fc2}}.weight
    GGUF:     v.blk.{i}.{attn_q,attn_k,attn_v,attn_out,ffn_up,ffn_down}.weight
    """
    prefix = ("paligemma_with_expert.paligemma.model.vision_tower."
              "vision_model.encoder.layers.")
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix):]              # "{i}.<...>"
    try:
        i_str, sub = rest.split(".", 1)
        i = int(i_str)
    except ValueError:
        return None
    SUFFIX_MAP = {
        "self_attn.q_proj.weight":   f"v.blk.{i}.attn_q.weight",
        "self_attn.k_proj.weight":   f"v.blk.{i}.attn_k.weight",
        "self_attn.v_proj.weight":   f"v.blk.{i}.attn_v.weight",
        "self_attn.out_proj.weight": f"v.blk.{i}.attn_out.weight",
        "mlp.fc1.weight":            f"v.blk.{i}.ffn_up.weight",
        "mlp.fc2.weight":            f"v.blk.{i}.ffn_down.weight",
    }
    return SUFFIX_MAP.get(sub)


def get_target_params(model: torch.nn.Module):
    """Return list of (hf_name, gguf_name, param, group) for each 2-D weight to
    track. group ∈ {"llm","vision"}. Vision gguf names start with "v.".
    """
    out = []
    for name, p in model.named_parameters():
        if p.ndim != 2:
            continue
        gname = hf_to_gguf_llm(name)
        if gname is not None:
            out.append((name, gname, p, "llm"))
            continue
        vname = hf_to_gguf_vision(name)
        if vname is not None:
            out.append((name, vname, p, "vision"))
    return out


# ─── Calib data loading ──────────────────────────────────────────────────────

def load_calib(calib_dir: Path) -> Dict:
    pv = np.load(calib_dir / "pixel_values.npy", mmap_mode="r")  # (N,3,224,224,3) uint8
    state = np.load(calib_dir / "state.npy")                      # (N,8) float32
    actions = np.load(calib_dir / "gt_actions.npy")               # (N,H,7) float32
    with open(calib_dir / "prompts.json") as f:
        prompts = json.load(f)
    with open(calib_dir / "metadata.json") as f:
        meta = json.load(f)
    return {"pixel_values": pv, "state": state, "gt_actions": actions,
            "prompts": prompts, "meta": meta}


def build_observation_and_actions(
    calib: Dict,
    indices: np.ndarray,
    sp,                       # sentencepiece processor
    norm_stats: Dict | None,
    device: torch.device,
    action_horizon: int,
    action_dim: int,
    state_dim: int,
    max_token_len: int,
):
    """Construct (observation, actions) tensors for a batch of frame indices."""
    B = len(indices)

    # ── Images: (B, 3cams, 224, 224, 3rgb) uint8 → 3 × (B, 3, 224, 224) float32 [-1,1] BCHW ──
    pv = np.ascontiguousarray(calib["pixel_values"][indices])      # (B,3,224,224,3) uint8 numpy
    pv_t = torch.from_numpy(pv).to(device, dtype=torch.float32)    # (B,3,224,224,3) f32
    pv_t = pv_t.div_(255.0).mul_(2.0).sub_(1.0)                    # → [-1, 1]
    pv_t = pv_t.permute(0, 1, 4, 2, 3).contiguous()                # → (B, 3cams, 3, 224, 224)
    images = {
        "base_0_rgb":         pv_t[:, 0],   # (B, 3, 224, 224) — channels-first
        "left_wrist_0_rgb":   pv_t[:, 1],
        "right_wrist_0_rgb":  pv_t[:, 2],
    }
    image_masks = {
        "base_0_rgb":         torch.ones(B, dtype=torch.bool, device=device),
        "left_wrist_0_rgb":   torch.ones(B, dtype=torch.bool, device=device),
        "right_wrist_0_rgb":  torch.zeros(B, dtype=torch.bool, device=device),  # placeholder
    }

    # ── State: (B, 8) raw → normalize via q01/q99 → pad to state_dim with zeros ──
    raw_state = np.ascontiguousarray(calib["state"][indices])
    if norm_stats is not None:
        q01 = norm_stats["q01"]; q99 = norm_stats["q99"]
        denom = np.where(np.abs(q99 - q01) < 1e-8, 1.0, q99 - q01)
        nstate = np.clip((raw_state - q01) / denom * 2.0 - 1.0, -1.0, 1.0)
    else:
        nstate = np.clip(raw_state, -1.0, 1.0)
    nstate = nstate.astype(np.float32)
    if nstate.shape[1] < state_dim:
        pad = np.zeros((B, state_dim - nstate.shape[1]), dtype=np.float32)
        nstate = np.concatenate([nstate, pad], axis=1)
    state_t = torch.from_numpy(nstate).to(device)

    # ── Tokenize prompts (Pi 0.5 prompt template: "Task: <task>\n") ──
    # NOTE: state is provided to the model via the suffix path (state-conditioning
    # in embed_suffix), so we don't bin it into the prompt here — match openpi's
    # default LIBERO prompt format ("<task>\n").
    token_ids = np.zeros((B, max_token_len), dtype=np.int64)
    token_mask = np.zeros((B, max_token_len), dtype=np.bool_)
    for b in range(B):
        prompt = calib["prompts"][int(indices[b])] + "\n"
        ids = sp.encode(prompt, out_type=int)
        L = min(len(ids), max_token_len)
        token_ids[b, :L] = ids[:L]
        token_mask[b, :L] = True
    tokens_t = torch.from_numpy(token_ids).to(device)
    token_mask_t = torch.from_numpy(token_mask).to(device)

    # ── Actions: (B, H, 7) → slice/pad to (B, action_horizon, action_dim) ──
    raw_actions = np.ascontiguousarray(calib["gt_actions"][indices])  # (B, H_calib, 7)
    H_calib = raw_actions.shape[1]
    if H_calib >= action_horizon:
        a = raw_actions[:, :action_horizon, :]                      # (B, AH, 7)
    else:
        # repeat last action to fill (shouldn't happen with calib H=50 >= AH=10)
        pad = np.tile(raw_actions[:, -1:, :], (1, action_horizon - H_calib, 1))
        a = np.concatenate([raw_actions, pad], axis=1)
    if a.shape[2] < action_dim:
        zeros = np.zeros((B, action_horizon, action_dim - a.shape[2]), dtype=np.float32)
        a = np.concatenate([a.astype(np.float32), zeros], axis=2)
    actions_t = torch.from_numpy(a).to(device)

    # Build observation object compatible with PI0Pytorch._preprocess_observation
    class Obs:
        pass
    obs = Obs()
    obs.images = images
    obs.image_masks = image_masks
    obs.state = state_t
    obs.tokenized_prompt = tokens_t
    obs.tokenized_prompt_mask = token_mask_t
    obs.token_ar_mask = None
    obs.token_loss_mask = None
    return obs, actions_t


# ─── Norm stats loader ───────────────────────────────────────────────────────

def load_norm_stats(checkpoint_dir: Path) -> Dict | None:
    """Find state q01/q99 in either norm_stats.json or assets/.../norm_stats.json."""
    candidates = [
        checkpoint_dir / "norm_stats.json",
        *checkpoint_dir.glob("assets/*/*/norm_stats.json"),
        *checkpoint_dir.glob("assets/**/norm_stats.json"),
    ]
    for p in candidates:
        if not p.exists():
            continue
        with open(p) as f:
            data = json.load(f)
        if "norm_stats" in data:
            data = data["norm_stats"]
        s = data.get("state", {})
        q01 = s.get("q01") or data.get("state_q01")
        q99 = s.get("q99") or data.get("state_q99")
        if q01 and q99:
            print(f"  Loaded state q01/q99 from {p.name}")
            return {"q01": np.asarray(q01, dtype=np.float32),
                    "q99": np.asarray(q99, dtype=np.float32)}
    print("  WARNING: no state norm_stats found — falling back to clip(state, -1, 1)")
    return None


# ─── Model loader ────────────────────────────────────────────────────────────

def load_model(checkpoint_dir: Path, device: torch.device, max_token_len: int):
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from safetensors.torch import load_file as st_load

    cfg_path = checkpoint_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            ckcfg = json.load(f)
    else:
        ckcfg = {}
    action_dim     = ckcfg.get("action_dim", 32)
    action_horizon = ckcfg.get("action_horizon", 10)
    pali_variant   = ckcfg.get("paligemma_variant", "gemma_2b")
    expert_variant = ckcfg.get("action_expert_variant", "gemma_300m")

    cfg = Pi0Config(
        pi05=True,
        action_horizon=action_horizon,
        paligemma_variant=pali_variant,
        action_expert_variant=expert_variant,
        action_dim=action_dim,
        max_token_len=max_token_len,
        pytorch_compile_mode=None,
    )
    print(f"  Pi0Config: action_horizon={action_horizon} action_dim={action_dim} "
          f"pali={pali_variant} expert={expert_variant} max_token_len={max_token_len}")

    print("  Building PI0Pytorch...")
    model = PI0Pytorch(config=cfg)

    print(f"  Loading weights from {checkpoint_dir / 'model.safetensors'}")
    sd = st_load(str(checkpoint_dir / "model.safetensors"))
    if all(k.startswith("model.") for k in sd):
        sd = {k[len("model."):]: v for k, v in sd.items()}
    # tied embedding
    embed_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    lm_head_key = "paligemma_with_expert.paligemma.lm_head.weight"
    if embed_key not in sd and lm_head_key in sd:
        sd[embed_key] = sd[lm_head_key]
    model.load_state_dict(sd, strict=True)
    model = model.to(device)
    model.train(False)  # disable dropout (no batchnorm anyway)
    print(f"  Loaded model on {device}, dtype={next(model.parameters()).dtype}")
    return model, cfg


# ─── Per-batch Fisher step ───────────────────────────────────────────────────

def fisher_step(
    model,
    target_params: List[Tuple[str, str, torch.nn.Parameter]],
    accumulators: Dict[str, torch.Tensor],
    obs,
    actions,
):
    """One forward + selective backward via torch.autograd.grad — computes
    gradients ONLY for target_params (no .grad allocated on non-target params),
    drastically reducing memory vs loss.backward()."""
    # PI0Pytorch.forward returns per-element MSE (B, AH, AD) — mean to scalar
    per_elem = model(obs, actions)
    loss = per_elem.mean()
    grads = torch.autograd.grad(
        loss,
        [p for _, _, p, _ in target_params],
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    for (_, gname, _, _), g in zip(target_params, grads):
        if g is None:
            continue
        accumulators[gname] += g.detach().float().pow(2).cpu()
    return float(loss.item())


def freeze_non_target(model, target_params):
    """Keep requires_grad=True everywhere so autograd builds a complete graph
    through every layer. torch.autograd.grad() in fisher_step computes grads
    ONLY for target_params, so non-target params don't allocate .grad buffers."""
    for p in model.parameters():
        p.requires_grad_(True)


def init_accumulators(target_params):
    return {gname: torch.zeros(p.shape, dtype=torch.float32)
            for _, gname, p, _ in target_params}


# ─── Per-GPU worker ──────────────────────────────────────────────────────────

def gpu_worker(
    gpu_id: int,
    visible_idx: int,        # which CUDA device this worker should use (0..N-1 after restriction)
    checkpoint_dir: str,
    calib_dir: str,
    result_dir: str,
    batch_size: int,
    max_token_len: int,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_idx)
    device = torch.device("cuda:0")
    import sentencepiece as spm

    with open(os.path.join(result_dir, f"gpu_{gpu_id}_indices.pkl"), "rb") as f:
        indices = pickle.load(f)

    print(f"[GPU {gpu_id}] Loading model on {device} (visible={visible_idx})...")
    model, cfg = load_model(Path(checkpoint_dir), device, max_token_len)

    sp = spm.SentencePieceProcessor()
    sp.Load(str(Path(checkpoint_dir) / "tokenizer.model"))
    print(f"[GPU {gpu_id}] tokenizer vocab={sp.GetPieceSize()}")

    norm_stats = load_norm_stats(Path(checkpoint_dir))

    target_params = get_target_params(model)
    n_llm = sum(1 for *_, g in target_params if g == "llm")
    n_vis = sum(1 for *_, g in target_params if g == "vision")
    print(f"[GPU {gpu_id}] Target tensors: {len(target_params)} "
          f"(llm={n_llm} vision={n_vis}, "
          f"total params: {sum(p.numel() for _, _, p, _ in target_params)/1e9:.2f}B)")
    freeze_non_target(model, target_params)

    calib = load_calib(Path(calib_dir))
    accumulators = init_accumulators(target_params)

    n_batches = (len(indices) + batch_size - 1) // batch_size
    losses = []
    t0 = time.time()
    for b, start in enumerate(range(0, len(indices), batch_size)):
        end = min(start + batch_size, len(indices))
        batch_idx = np.sort(indices[start:end])

        obs, actions = build_observation_and_actions(
            calib, batch_idx, sp, norm_stats, device,
            action_horizon=cfg.action_horizon,
            action_dim=cfg.action_dim,
            state_dim=cfg.action_dim,   # Pi0 uses same dim for state padding
            max_token_len=max_token_len,
        )
        loss_val = fisher_step(model, target_params, accumulators, obs, actions)
        losses.append(loss_val)

        if (b + 1) % 25 == 0 or b == n_batches - 1:
            dt = time.time() - t0
            rate = (b + 1) / dt * 60 if dt > 0 else 0
            eta = (n_batches - b - 1) / (rate / 60) if rate > 0 else 0
            print(f"[GPU {gpu_id}] {b+1:4d}/{n_batches}  loss={loss_val:.5f}  "
                  f"rate={rate:.1f}b/m  eta={eta:.0f}s", flush=True)

        del obs, actions
        torch.cuda.empty_cache()

    # Save per-GPU result
    out_dir = os.path.join(result_dir, f"gpu_{gpu_id}")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "num_samples.npy"), np.array(len(indices), dtype=np.int64))
    np.save(os.path.join(out_dir, "losses.npy"), np.array(losses, dtype=np.float32))
    name_map = {}
    group_map = {}
    for _, gname, _, grp in target_params:
        safe = gname.replace(".", "_").replace("/", "_")
        name_map[safe] = gname
        group_map[gname] = grp
        np.save(os.path.join(out_dir, f"acc_{safe}.npy"), accumulators[gname].numpy())
    with open(os.path.join(out_dir, "name_map.json"), "w") as f:
        json.dump(name_map, f)
    with open(os.path.join(out_dir, "group_map.json"), "w") as f:
        json.dump(group_map, f)
    print(f"[GPU {gpu_id}] Done — {len(indices)} frames in {time.time()-t0:.1f}s")


# ─── Main driver ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Pi 0.5 Fisher diagonal (flow-matching loss)")
    ap.add_argument("--checkpoint", required=True,
                    help="Path to pi05 PyTorch checkpoint dir (config.json + model.safetensors)")
    ap.add_argument("--calib-dir", required=True,
                    help="Calibration data dir produced by get_pi05_calib_data.py")
    ap.add_argument("--output", required=True,
                    help="Output GGUF imatrix file path (LLM)")
    ap.add_argument("--vision-output", default=None,
                    help="Output GGUF imatrix path for the vision tower. "
                         "Default: <output> with '.gguf'→'_vision.gguf'. "
                         "fc1/fc2 are zero-padded to match export_pi05.py's "
                         "K-quant vision padding so the per-element imatrix "
                         "aligns with the exported GGUF tensors.")
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--gpus", default=None,
                    help="Comma-separated GPU ids (overrides --num-gpus)")
    ap.add_argument("--batch-size", type=int, default=6,
                    help="Per-GPU batch size. B=6 uses ~36GB peak on a 45GB GPU.")
    ap.add_argument("--num-samples", type=int, default=0,
                    help="Limit total calibration samples (0 = all)")
    ap.add_argument("--max-token-len", type=int, default=200,
                    help="Tokenized prompt budget (must match pi05 training config)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Validate paths
    ckpt_dir = Path(args.checkpoint)
    calib_dir = Path(args.calib_dir)
    if not (ckpt_dir / "model.safetensors").exists():
        raise FileNotFoundError(f"Missing {ckpt_dir / 'model.safetensors'}")
    if not (calib_dir / "pixel_values.npy").exists():
        raise FileNotFoundError(f"Missing {calib_dir / 'pixel_values.npy'}")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # GPU selection
    if args.gpus:
        gpu_ids = [int(x) for x in args.gpus.split(",")]
    else:
        gpu_ids = list(range(args.num_gpus))
    n_gpus = len(gpu_ids)

    # Sample selection
    with open(calib_dir / "metadata.json") as f:
        meta = json.load(f)
    total = meta["total_frames"]
    if 0 < args.num_samples < total:
        rng = np.random.RandomState(args.seed)
        all_indices = np.sort(rng.choice(total, args.num_samples, replace=False))
    else:
        all_indices = np.arange(total)

    print("=" * 70)
    print("Pi 0.5 Fisher Diagonal — Flow-Matching Loss")
    print("=" * 70)
    print(f"  checkpoint:    {ckpt_dir}")
    print(f"  calib_dir:     {calib_dir}")
    print(f"  output:        {out_path}")
    print(f"  total frames:  {total}, using {len(all_indices)}")
    print(f"  num_gpus:      {n_gpus}, ids={gpu_ids}")
    print(f"  batch_size:    {args.batch_size}")
    print()

    # Sharding + result dir
    chunks = np.array_split(all_indices, n_gpus)
    result_dir = str(out_path) + "_workdir"
    os.makedirs(result_dir, exist_ok=True)
    for i, gid in enumerate(gpu_ids):
        with open(os.path.join(result_dir, f"gpu_{gid}_indices.pkl"), "wb") as f:
            pickle.dump(chunks[i], f)
        print(f"  GPU {gid}: {len(chunks[i])} frames")

    # Spawn workers
    print(f"\nSpawning {n_gpus} GPU workers...")
    t_start = time.time()
    mp.set_start_method("spawn", force=True)
    with mp.Pool(n_gpus) as pool:
        futs = [
            pool.apply_async(
                gpu_worker,
                args=(gid, gid, str(ckpt_dir), str(calib_dir), result_dir,
                      args.batch_size, args.max_token_len),
            )
            for gid in gpu_ids
        ]
        for f in futs:
            f.get(timeout=36000)
    t_dt = time.time() - t_start
    print(f"\nAll GPU workers complete in {t_dt:.1f}s ({t_dt/60:.1f}min)")

    # Merge accumulators across GPUs
    print("Merging per-GPU Fisher accumulators...")
    merged: Dict[str, np.ndarray] = {}
    total_samples = 0
    all_losses: List[float] = []
    for gid in gpu_ids:
        d = os.path.join(result_dir, f"gpu_{gid}")
        total_samples += int(np.load(os.path.join(d, "num_samples.npy")))
        all_losses.extend(np.load(os.path.join(d, "losses.npy")).tolist())
        with open(os.path.join(d, "name_map.json")) as f:
            name_map = json.load(f)
        group_map = {}
        gm_path = os.path.join(d, "group_map.json")
        if os.path.exists(gm_path):
            with open(gm_path) as f:
                group_map = json.load(f)
        for safe, gname in name_map.items():
            arr = np.load(os.path.join(d, f"acc_{safe}.npy"))
            if gname not in merged:
                merged[gname] = arr.astype(np.float64)
            else:
                merged[gname] += arr.astype(np.float64)

    fisher = {n: (m / total_samples).astype(np.float32) for n, m in merged.items()}

    # Split LLM vs vision by gguf-name prefix ("v." → vision tower).
    fisher_llm = {n: a for n, a in fisher.items() if not n.startswith("v.")}
    fisher_vis = {n: a for n, a in fisher.items() if n.startswith("v.")}

    # Pad vision fc1/fc2 (ffn_up/ffn_down) Fisher exactly like export_pi05.py
    # pads the weights for K-quant vision, so the per-element imatrix aligns
    # with the exported GGUF tensor shapes:
    #   ffn_down (fc2): (1152, 4304) → (1152, 4352)        pad +48 cols
    #   ffn_up   (fc1): (4304, 1152) → (4352, 1280)        pad +48 rows, +128 cols
    def _pad_vision_fisher(name: str, arr: np.ndarray) -> np.ndarray:
        if name.endswith("ffn_down.weight"):
            return np.pad(arr, ((0, 0), (0, 48)), mode="constant")
        if name.endswith("ffn_up.weight"):
            return np.pad(arr, ((0, 48), (0, 128)), mode="constant")
        return arr
    fisher_vis = {n: _pad_vision_fisher(n, a) for n, a in fisher_vis.items()}

    # Print Fisher summary
    print(f"\n{'Tensor':<40s} {'Shape':>15s} {'Mean':>12s} {'Max':>12s}")
    print("-" * 82)
    for n in sorted(fisher.keys()):
        f = fisher[n]
        print(f"{n:<40s} {str(tuple(f.shape)):>15s} {f.mean():>12.4e} {f.max():>12.4e}")

    print(f"\nLoss mean: {np.mean(all_losses):.5f}  std: {np.std(all_losses):.5f}")
    print(f"Total samples used: {total_samples}")
    print(f"LLM tensors: {len(fisher_llm)}   Vision tensors: {len(fisher_vis)}")

    # Write LLM GGUF imatrix
    print(f"\nWriting LLM GGUF imatrix to {out_path} ...")
    write_imatrix_gguf(fisher_llm, out_path, total_samples, str(calib_dir))
    print(f"  Done. {out_path} ({out_path.stat().st_size/1024**2:.1f} MB)")

    # Write Vision GGUF imatrix
    if fisher_vis:
        if args.vision_output:
            vis_path = Path(args.vision_output)
        else:
            vis_path = Path(str(out_path).replace(".gguf", "_vision.gguf"))
        print(f"\nWriting Vision GGUF imatrix to {vis_path} ...")
        write_imatrix_gguf(fisher_vis, vis_path, total_samples, str(calib_dir))
        print(f"  Done. {vis_path} ({vis_path.stat().st_size/1024**2:.1f} MB)")

    print("\nNext steps:")
    print(f"  llama-quantize --imatrix {out_path} \\")
    print(f"      <pali_llm_bf16.gguf> <pali_llm_iq2xs.gguf> IQ2_XS")

    # Cleanup work dir
    shutil.rmtree(result_dir, ignore_errors=True)


def write_imatrix_gguf(fisher: Dict[str, np.ndarray], out_path: Path,
                       num_samples: int, calib_source: str):
    """Write per-element Fisher diagonal as a GGUF file.

    llama-quant.cpp:953-967 detects per-weight Fisher by tensor size:
      - if imatrix.size == ne[0]*ne[2]            → per-column (legacy imatrix)
      - if imatrix.size == ne[0]*ne[1]*ne[2]      → per-weight Fisher (logs "(per-weight)")

    For Fisher we save the full per-element gradient² (n_out × n_in) so each weight
    gets its own importance (which is what Fisher actually measures), not a
    column-aggregate.

    Tensor layout:
      <name>.in_sum2 : shape (n_out, n_in) float32  — mean_n (∂L/∂W[i,j])²_n
      <name>.counts  : shape (1, 1) float32         — 1.0  (in_sum2 already mean-per-sample)
    """
    from gguf import GGUFWriter

    writer = GGUFWriter(str(out_path), arch="")
    writer.add_string("general.type", "imatrix")
    writer.add_array("imatrix.datasets", [calib_source])
    writer.add_uint32("imatrix.chunk_count", int(num_samples))
    writer.add_uint32("imatrix.chunk_size", 1)
    writer.add_string("fisher.method", "flow_matching_per_weight_pi05")
    writer.add_uint32("fisher.num_samples", int(num_samples))

    n_floored_total = 0
    for name in sorted(fisher.keys()):
        F = fisher[name].astype(np.float32, copy=True)        # (n_out, n_in) mean_n grad²
        F = np.maximum(F, 0.0)
        # Element-wise floor at 1e-6 * mean(nonzero) to avoid degenerate 32-weight
        # blocks that crash IQ2_XS / IQ1_S quantizers ("point not on grid" abort
        # when scale = sumqx/sumq2 collapses to 0). At per-weight granularity
        # most entries are non-zero so the floor barely shifts magnitudes.
        nz = F[F > 0]
        floor = float(nz.mean() * 1e-6) if nz.size else 1e-20
        n_floored_total += int((F < floor).sum())
        F = np.maximum(F, floor).astype(np.float32)
        # Save full per-element Fisher: gguf-py reverses to ggml ne=[n_in, n_out, 1, 1]
        # so total elements = ne[0]*ne[1] = n_in*n_out → llama-quant sets
        # imatrix_is_per_weight=true.
        writer.add_tensor(f"{name}.in_sum2", F)
        writer.add_tensor(f"{name}.counts",  np.array([[1.0]], dtype=np.float32))

    print(f"  Floored {n_floored_total} per-element entries below 1e-6×mean (avoid IQ-quant crashes)")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


if __name__ == "__main__":
    main()
