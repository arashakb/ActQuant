#!/usr/bin/env python3
"""
HSIC Per-Tensor Sensitivity Scores for Pi 0.5

Mirror of tools/hsic/compute_hsic_tensor.py but adapted to Pi 0.5:
  - Hook the 126 PaliGemma linear modules (18 layers × 7 weights each)
  - Forward via PI0Pytorch.forward(observation, actions) (full Pi 0.5 stack:
    SigLIP → projector → PaliGemma + Action Expert flow-matching), under
    torch.no_grad() — we only need activations, not gradients
  - Mean-pool each linear layer's input and output across the PREFIX tokens
    (vision + text), which is the natural analog of "action token positions"
    in OFT (Pi 0.5 has no action token positions in the LLM)
  - X = input to first PaliGemma layer
  - Y = ground-truth continuous actions (B, action_horizon, action_dim) flattened
  - Compute HSIC-IB:  F(Z) = -lx·HSIC(X, Z) + ly·HSIC(Z, Y)

Output: JSON file with per-tensor scores in the same format as
compute_hsic_tensor.py — directly consumable by allocate_pi05.py.

Tensor names match llama-arch (`blk.{i}.{type}.weight`) so they line up
with what export_pi05_llm.py produces and what llama-quantize expects.

Usage (8 GPU):
    ${HOME}/miniconda3/envs/openpi-server/bin/python \
        tools/hsic/compute_hsic_pi05.py \
        --checkpoint /path/to/openpi/pi05_libero_base_pytorch \
        --calib-dir  /path/to/openpi/calib_data_raw \
        --output     tools/hsic/scores/hsic_pi05_combined.json \
        --num-gpus 8 --batch-size 8 --use-gpu-hsic
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm


# ─── Path bootstrap ───────────────────────────────────────────────────────────
THIS = Path(__file__).resolve()
REPO = THIS.parent.parent.parent
sys.path.insert(0, str(REPO / "tools" / "hsic"))
sys.path.insert(0, str(REPO / "tools" / "fisher-diag"))

from hsic_utils import rbf_kernel, linear_kernel, centering_matrix, hsic, distmat  # noqa: E402

# Reuse Pi 0.5 model loader / data builder from the Fisher pipeline.
from compute_fisher_pi05 import (  # noqa: E402
    load_model as load_pi05_model,
    load_calib as load_pi05_calib,
    load_norm_stats as load_pi05_norm_stats,
    build_observation_and_actions,
    hf_to_gguf_llm,
)


PROJ_NAMES = [
    "self_attn.q_proj",   "self_attn.k_proj",
    "self_attn.v_proj",   "self_attn.o_proj",
    "mlp.gate_proj",      "mlp.up_proj",      "mlp.down_proj",
]
GGUF_TYPE_MAP = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj":    "ffn_gate",
    "mlp.up_proj":      "ffn_up",
    "mlp.down_proj":    "ffn_down",
}


def gguf_name(layer_idx: int, proj_name: str) -> str:
    return f"blk.{layer_idx}.{GGUF_TYPE_MAP[proj_name]}.weight"


# ─── Activation collection (single GPU) ───────────────────────────────────────

def collect_activations_pi05(
    model: torch.nn.Module,
    calib: Dict,
    indices: np.ndarray,
    norm_stats,
    sp,
    device: torch.device,
    action_horizon: int,
    action_dim: int,
    state_dim: int,
    max_token_len: int,
    batch_size: int,
    pool_token_count: int,
):
    """Run the forward pass for `indices` calibration frames, capture
    mean-pooled inputs and outputs of each PaliGemma linear layer.

    `pool_token_count` controls how many of the LAST prefix tokens to average
    over (analog of OFT's `n_action_tokens`). Default 0 → average over ALL
    prefix tokens (vision + text), matching --pool-all in the OFT script.

    Returns:
        X_acts:   (n, d_model)             — input to first PaliGemma layer
        Y_acts:   (n, action_horizon*7)    — standardized ground-truth actions
        in_acts:  {gguf_name: (n, d_in)}   — mean-pooled inputs of each linear
        out_acts: {gguf_name: (n, d_out)}  — mean-pooled outputs of each linear
    """
    layers = model.paligemma_with_expert.paligemma.language_model.layers
    n_layers = len(layers)

    # Storage (per linear, per frame)
    in_buckets:  Dict[str, list] = {}
    out_buckets: Dict[str, list] = {}
    X_buckets:   list = []

    handles = []

    def pool(tensor):
        # tensor: (B, n_prefix, d). Mean-pool over the last `pool_token_count`
        # tokens if >0, else over all prefix tokens.
        if pool_token_count > 0 and tensor.shape[1] >= pool_token_count:
            t = tensor[:, -pool_token_count:, :]
        else:
            t = tensor
        return t.mean(dim=1).float().cpu().numpy()  # (B, d)

    # X hook: pi05 inlines the layer forward (paligemma_with_expert.forward in
    # gemma_pytorch.py) so layer.__call__ is never invoked. But `layer.input_layernorm`
    # IS called as a module — its forward_pre_hook fires with the raw hidden_states
    # entering layer 0, which is exactly X.
    def x_pre_hook(module, args):
        if args:
            X_buckets.append(pool(args[0]))
    handles.append(layers[0].input_layernorm.register_forward_pre_hook(x_pre_hook))

    # Per-linear hooks
    def make_hook(name):
        def hook_fn(module, inp, out):
            # inp: tuple containing (B, n_prefix, d_in); out: (B, n_prefix, d_out)
            in_buckets[name].append(pool(inp[0]))
            out_buckets[name].append(pool(out))
        return hook_fn

    for li in range(n_layers):
        layer = layers[li]
        for proj in PROJ_NAMES:
            name = gguf_name(li, proj)
            in_buckets[name] = []
            out_buckets[name] = []
            sub = layer
            for part in proj.split("."):
                sub = getattr(sub, part)
            handles.append(sub.register_forward_hook(make_hook(name)))

    # Forward in batches
    Y_list: list = []
    n = len(indices)
    pbar = tqdm(range(0, n, batch_size), desc="HSIC fwd")
    for start in pbar:
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        obs, actions = build_observation_and_actions(
            calib, batch_idx, sp, norm_stats, device,
            action_horizon=action_horizon, action_dim=action_dim,
            state_dim=state_dim, max_token_len=max_token_len,
        )
        with torch.no_grad():
            _ = model(obs, actions)

        # Y: standardized ground-truth actions (B, action_horizon, 7) → flatten
        gt = np.ascontiguousarray(calib["gt_actions"][batch_idx][:, :action_horizon, :])
        Y_list.append(gt.reshape(len(batch_idx), -1).astype(np.float32))

        del obs, actions
        torch.cuda.empty_cache()

    for h in handles:
        h.remove()

    # Concatenate per-frame batches
    X_acts = np.concatenate(X_buckets, axis=0).astype(np.float32)   # (n, d_model)
    Y_raw  = np.concatenate(Y_list, axis=0).astype(np.float32)      # (n, action_horizon*7)
    in_acts  = {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in in_buckets.items()}
    out_acts = {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in out_buckets.items()}

    # Standardize Y
    Y_acts = (Y_raw - Y_raw.mean(axis=0)) / (Y_raw.std(axis=0) + 1e-8)

    return X_acts, Y_acts, in_acts, out_acts


# ─── GPU HSIC computation (re-implemented locally; matches compute_hsic_tensor.py) ──

def _rbf_kernel_torch(X: torch.Tensor, sigma=None) -> torch.Tensor:
    """Memory-efficient RBF kernel: builds D, applies median, computes K in
    one (n,n) buffer with in-place ops. Avoids the 4×(n,n) intermediates
    that the textbook expression `sq[:,None] + sq[None,:] - 2*(X@X.T)`
    creates — at n≈47k each intermediate is 9 GB."""
    n = X.shape[0]
    d = X.shape[1]
    sq = (X * X).sum(dim=1)                # (n,)
    D = X @ X.T                            # (n,n) — sole big allocation
    D.mul_(-2.0).add_(sq.view(-1, 1)).add_(sq.view(1, -1)).clamp_min_(0.0)

    if sigma is not None:
        variance = 2.0 * float(sigma) * float(sigma) * d
    else:
        # Sample-based median of upper triangle. Full triu_indices would
        # allocate 17 GB of int64 for n≈47k.
        n_samples = min(200_000, n * (n - 1) // 2)
        i = torch.randint(0, n, (n_samples * 2,), device=X.device)
        j = torch.randint(0, n, (n_samples * 2,), device=X.device)
        mask = i < j
        flat = D[i[mask][:n_samples], j[mask][:n_samples]]
        sigma_est = float(flat.median().item())
        if sigma_est <= 0:
            sigma_est = float(flat.mean().item())
        if sigma_est < 1e-2:
            sigma_est = 1e-2
        variance = 2.0 * sigma_est * sigma_est
        del flat, i, j, mask

    D.div_(-variance).exp_()                # K = exp(-D / variance), in-place
    return D                                 # D is now K


def compute_tensor_hsic_gpu(
    X_acts: np.ndarray,
    Y_acts: np.ndarray,
    in_acts: Dict[str, np.ndarray],
    out_acts: Dict[str, np.ndarray],
    kernel_hidden: str = "rbf",
    kernel_y: str = "linear",
    sigma: float = None,
    lx: float = 1.0,
    ly: float = 1.0,
    device: str = "cuda:0",
) -> Dict:
    n = X_acts.shape[0]
    print(f"[GPU HSIC] n={n} kernel_hidden={kernel_hidden} kernel_y={kernel_y} "
          f"sigma={sigma} lx={lx} ly={ly}")

    # In-place mean-centering trick: HK = (I - 1/n J) @ K  is equivalent to
    # K - K.mean(dim=0, keepdim=True). Avoids materializing the n×n H matrix
    # (~9 GB for n≈47k) and the H @ K intermediate.
    def _center(K):
        K.sub_(K.mean(dim=0, keepdim=True))
        return K  # K is now HK in place

    # Chunked trace: tr(A @ B) = sum_i (row_i(A) ⋅ col_i(B)).
    # The naive (A * B.T).sum() allocates a full n×n tensor (9 GB at n≈47k).
    # Chunking rows of A keeps the temp at chunk×n.
    def _trace_AB(A, B, chunk=2048):
        n_local = A.shape[0]
        total = 0.0
        for ii in range(0, n_local, chunk):
            sl = slice(ii, min(ii + chunk, n_local))
            total += float((A[sl] * B.T[sl]).sum().item())
        return total

    X_t = torch.from_numpy(X_acts.astype(np.float32)).to(device)
    Y_t = torch.from_numpy(Y_acts.astype(np.float32)).to(device)
    K_X = _rbf_kernel_torch(X_t, sigma=sigma) if kernel_hidden == "rbf" else (X_t @ X_t.T)
    del X_t
    K_Y = _rbf_kernel_torch(Y_t, sigma=sigma) if kernel_y      == "rbf" else (Y_t @ Y_t.T)
    del Y_t
    HK_X = _center(K_X)
    HK_Y = _center(K_Y)
    n2 = (n - 1) ** 2

    results = {"n_samples": n, "n_tensors": len(out_acts), "tensors": {}}
    in_cache: Dict[bytes, Tuple[float, float, float]] = {}

    for name in tqdm(sorted(out_acts.keys()), desc="[GPU HSIC] per tensor"):
        Z_out_np = out_acts[name]; Z_in_np = in_acts[name]

        Z_out = torch.from_numpy(Z_out_np.astype(np.float32)).to(device)
        K_out = _rbf_kernel_torch(Z_out, sigma=sigma) if kernel_hidden == "rbf" else (Z_out @ Z_out.T)
        del Z_out
        HK_out = _center(K_out)
        hsic_x_out = _trace_AB(HK_X, HK_out) / n2
        hsic_y_out = _trace_AB(HK_out, HK_Y) / n2
        F_out = -lx * hsic_x_out + ly * hsic_y_out
        del HK_out
        torch.cuda.empty_cache()

        in_key = Z_in_np.tobytes()[:64]
        if in_key not in in_cache:
            Z_in = torch.from_numpy(Z_in_np.astype(np.float32)).to(device)
            K_in = _rbf_kernel_torch(Z_in, sigma=sigma) if kernel_hidden == "rbf" else (Z_in @ Z_in.T)
            del Z_in
            HK_in = _center(K_in)
            hsic_x_in = _trace_AB(HK_X, HK_in) / n2
            hsic_y_in = _trace_AB(HK_in, HK_Y) / n2
            F_in = -lx * hsic_x_in + ly * hsic_y_in
            del HK_in
            torch.cuda.empty_cache()
            in_cache[in_key] = (hsic_x_in, hsic_y_in, F_in)
        hsic_x_in, hsic_y_in, F_in = in_cache[in_key]

        parts = name.split(".")
        results["tensors"][name] = {
            "hsic_x_out": hsic_x_out, "hsic_y_out": hsic_y_out, "F_out": F_out,
            "hsic_x_in":  hsic_x_in,  "hsic_y_in":  hsic_y_in,  "F_in": F_in,
            "sens": F_out - F_in,
            "d_in":  int(Z_in_np.shape[1]), "d_out": int(Z_out_np.shape[1]),
            "layer": int(parts[1]), "type": parts[2],
        }
        torch.cuda.empty_cache()

    del HK_X, HK_Y
    torch.cuda.empty_cache()
    return results


def compute_tensor_hsic_cpu(
    X_acts, Y_acts, in_acts, out_acts,
    kernel_hidden="rbf", kernel_y="linear",
    sigma=None, lx=1.0, ly=1.0,
):
    n = X_acts.shape[0]
    hidden_kf = (lambda Z: rbf_kernel(Z, sigma=sigma)) if kernel_hidden == "rbf" else linear_kernel
    y_kf      = (lambda Z: rbf_kernel(Z, sigma=sigma)) if kernel_y      == "rbf" else linear_kernel
    K_X = hidden_kf(X_acts); K_Y = y_kf(Y_acts); H = centering_matrix(n)

    results = {"n_samples": n, "n_tensors": len(out_acts), "tensors": {}}
    in_cache = {}
    for name in tqdm(sorted(out_acts.keys()), desc="HSIC (cpu)"):
        Z_out = out_acts[name]; Z_in = in_acts[name]
        K_out = hidden_kf(Z_out)
        hsic_x_out = hsic(K_X, K_out, H); hsic_y_out = hsic(K_out, K_Y, H)
        F_out = -lx * hsic_x_out + ly * hsic_y_out
        in_key = Z_in.tobytes()[:64]
        if in_key not in in_cache:
            K_in = hidden_kf(Z_in)
            hsic_x_in = hsic(K_X, K_in, H); hsic_y_in = hsic(K_in, K_Y, H)
            F_in = -lx * hsic_x_in + ly * hsic_y_in
            in_cache[in_key] = (hsic_x_in, hsic_y_in, F_in)
        hsic_x_in, hsic_y_in, F_in = in_cache[in_key]
        parts = name.split(".")
        results["tensors"][name] = {
            "hsic_x_out": hsic_x_out, "hsic_y_out": hsic_y_out, "F_out": F_out,
            "hsic_x_in":  hsic_x_in,  "hsic_y_in":  hsic_y_in,  "F_in": F_in,
            "sens": F_out - F_in,
            "d_in":  int(Z_in.shape[1]), "d_out": int(Z_out.shape[1]),
            "layer": int(parts[1]), "type": parts[2],
        }
    return results


# ─── Multi-GPU activation collection ─────────────────────────────────────────

def _gpu_worker(
    worker_id: int,
    visible_idx: int,
    checkpoint_dir: str,
    calib_dir: str,
    indices_path: str,
    output_path: str,
    batch_size: int,
    pool_token_count: int,
    max_token_len: int,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(visible_idx)
    device = torch.device("cuda:0")
    import sentencepiece as spm

    indices = np.load(indices_path)
    print(f"[W{worker_id}/GPU{visible_idx}] {len(indices)} frames", flush=True)

    model, cfg = load_pi05_model(Path(checkpoint_dir), device, max_token_len)
    sp = spm.SentencePieceProcessor()
    sp.Load(str(Path(checkpoint_dir) / "tokenizer.model"))
    norm_stats = load_pi05_norm_stats(Path(checkpoint_dir))
    calib = load_pi05_calib(Path(calib_dir))
    # Disable grad globally — saves memory
    for p in model.parameters(): p.requires_grad_(False)

    X_acts, Y_acts, in_acts, out_acts = collect_activations_pi05(
        model, calib, indices, norm_stats, sp, device,
        action_horizon=cfg.action_horizon, action_dim=cfg.action_dim,
        state_dim=cfg.action_dim, max_token_len=max_token_len,
        batch_size=batch_size, pool_token_count=pool_token_count,
    )

    np.savez(   # per-GPU shard, uncompressed for speed; main process re-saves merged
        output_path,
        X_acts=X_acts.astype(np.float16),
        Y_acts=Y_acts.astype(np.float16),
        in_acts=np.array({k: v.astype(np.float16) for k, v in in_acts.items()}, dtype=object),
        out_acts=np.array({k: v.astype(np.float16) for k, v in out_acts.items()}, dtype=object),
    )
    print(f"[W{worker_id}/GPU{visible_idx}] Saved {output_path}", flush=True)


def collect_acts_multi_gpu(
    checkpoint_dir: str,
    calib_dir: str,
    indices: np.ndarray,
    num_gpus: int,
    batch_size: int,
    pool_token_count: int,
    max_token_len: int,
    work_dir: str,
):
    n = len(indices)
    chunk = (n + num_gpus - 1) // num_gpus
    chunks = [(i * chunk, min((i + 1) * chunk, n)) for i in range(num_gpus)]
    chunks = [(s, e) for s, e in chunks if s < e]

    indices_paths = []
    for i, (s, e) in enumerate(chunks):
        p = os.path.join(work_dir, f"indices_gpu{i}.npy")
        np.save(p, np.asarray(indices[s:e]))
        indices_paths.append(p)
    out_paths = [os.path.join(work_dir, f"acts_gpu{i}.npz") for i in range(len(chunks))]

    print(f"Spawning {len(chunks)} GPU workers...")
    ctx = mp.get_context("spawn")
    procs = []
    for i, (idx_path, out_path) in enumerate(zip(indices_paths, out_paths)):
        p = ctx.Process(target=_gpu_worker, args=(
            i, i, checkpoint_dir, calib_dir, idx_path, out_path,
            batch_size, pool_token_count, max_token_len,
        ))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"GPU worker exited with code {p.exitcode}")

    # Merge
    X_list, Y_list = [], []
    in_lists, out_lists = {}, {}
    for path in out_paths:
        data = np.load(path, allow_pickle=True)
        X_list.append(data["X_acts"])
        Y_list.append(data["Y_acts"])
        for k, v in data["in_acts"].item().items():
            in_lists.setdefault(k, []).append(v)
        for k, v in data["out_acts"].item().items():
            out_lists.setdefault(k, []).append(v)
    X_acts = np.concatenate(X_list, axis=0)
    Y_acts = np.concatenate(Y_list, axis=0)
    in_acts  = {k: np.concatenate(v, axis=0) for k, v in in_lists.items()}
    out_acts = {k: np.concatenate(v, axis=0) for k, v in out_lists.items()}
    return X_acts, Y_acts, in_acts, out_acts


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Pi 0.5 HSIC per-tensor sensitivity")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--calib-dir",  required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Forward batch size (no_grad mode, fits comfortably at B=8 on 45GB GPU)")
    ap.add_argument("--n-frames", type=int, default=0,
                    help="Limit calib samples (0 = all)")
    ap.add_argument("--max-token-len", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-acts", default=None,
                    help="Path to .npz activation cache; reuse to skip re-running model")
    ap.add_argument("--pool-tokens", type=int, default=0,
                    help="If >0, mean-pool over the last K prefix tokens; 0 = mean over all")
    ap.add_argument("--kernel-hidden", choices=["rbf", "linear"], default="rbf")
    ap.add_argument("--kernel-y",      choices=["rbf", "linear"], default="linear")
    ap.add_argument("--sigma", type=float, default=None,
                    help="RBF bandwidth; None = median heuristic")
    ap.add_argument("--lx", type=float, default=1.0)
    ap.add_argument("--ly", type=float, default=1.0)
    ap.add_argument("--use-gpu-hsic", action="store_true")
    ap.add_argument("--hsic-device", default="cuda:0")
    args = ap.parse_args()

    ckpt_dir = Path(args.checkpoint)
    calib_dir = Path(args.calib_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(calib_dir / "metadata.json") as f:
        meta = json.load(f)
    total = meta["total_frames"]
    if 0 < args.n_frames < total:
        rng = np.random.RandomState(args.seed)
        all_indices = np.sort(rng.choice(total, args.n_frames, replace=False))
    else:
        all_indices = np.arange(total)
    use_frames = len(all_indices)

    print("=" * 70)
    print("Pi 0.5 HSIC Per-Tensor Sensitivity")
    print("=" * 70)
    print(f"  checkpoint:    {ckpt_dir}")
    print(f"  calib_dir:     {calib_dir}  ({use_frames}/{total} frames)")
    print(f"  output:        {out_path}")
    print(f"  num_gpus:      {args.num_gpus}")
    print(f"  batch_size:    {args.batch_size}")
    print(f"  cache:         {args.cache_acts}")
    print(f"  kernels:       hidden={args.kernel_hidden} y={args.kernel_y} sigma={args.sigma}")
    print(f"  lx/ly:         {args.lx}/{args.ly}")

    # ── Activation collection or cache load ──
    cache = args.cache_acts
    t0 = time.time()
    if cache and os.path.exists(cache):
        print(f"\n[1/2] Loading activation cache from {cache}...")
        data = np.load(cache, allow_pickle=True)
        X_acts = data["X_acts"].astype(np.float32)
        Y_acts = data["Y_acts"].astype(np.float32)
        in_acts  = {k: v.astype(np.float32) for k, v in data["in_acts"].item().items()}
        out_acts = {k: v.astype(np.float32) for k, v in data["out_acts"].item().items()}
        n = X_acts.shape[0]
        print(f"  Loaded n={n}, {len(out_acts)} tensors")
    else:
        print(f"\n[1/2] Collecting activations across {args.num_gpus} GPU(s)...")
        if args.num_gpus > 1:
            work_dir = str(out_path) + "_workdir"
            os.makedirs(work_dir, exist_ok=True)
            mp.set_start_method("spawn", force=True)
            X_acts, Y_acts, in_acts, out_acts = collect_acts_multi_gpu(
                str(ckpt_dir), str(calib_dir), all_indices, args.num_gpus,
                args.batch_size, args.pool_tokens, args.max_token_len, work_dir,
            )
            import shutil; shutil.rmtree(work_dir, ignore_errors=True)
        else:
            import sentencepiece as spm
            device = torch.device("cuda:0")
            model, cfg = load_pi05_model(ckpt_dir, device, args.max_token_len)
            for p in model.parameters(): p.requires_grad_(False)
            sp = spm.SentencePieceProcessor()
            sp.Load(str(ckpt_dir / "tokenizer.model"))
            norm = load_pi05_norm_stats(ckpt_dir)
            calib = load_pi05_calib(calib_dir)
            X_acts, Y_acts, in_acts, out_acts = collect_activations_pi05(
                model, calib, all_indices, norm, sp, device,
                action_horizon=cfg.action_horizon, action_dim=cfg.action_dim,
                state_dim=cfg.action_dim, max_token_len=args.max_token_len,
                batch_size=args.batch_size, pool_token_count=args.pool_tokens,
            )

        if cache:
            print(f"  Saving cache to {cache} (uncompressed; compression on 24GB is slow)...")
            np.savez(   # uncompressed: ~10x faster than savez_compressed for large arrays
                cache,
                X_acts=X_acts.astype(np.float16),
                Y_acts=Y_acts.astype(np.float16),
                in_acts=np.array({k: v.astype(np.float16) for k, v in in_acts.items()}, dtype=object),
                out_acts=np.array({k: v.astype(np.float16) for k, v in out_acts.items()}, dtype=object),
            )
    t_collect = time.time() - t0
    print(f"  Collected: X={X_acts.shape} Y={Y_acts.shape} tensors={len(out_acts)} ({t_collect:.1f}s)")

    # ── Sigma diagnostics ──
    sample = sorted(out_acts.keys())[len(out_acts) // 2]
    Zs = out_acts[sample]
    n = X_acts.shape[0]
    triu = np.triu_indices(n, k=1)
    med_X = float(np.median(distmat(X_acts.astype(np.float32))[triu]))
    med_Z = float(np.median(distmat(Zs.astype(np.float32))[triu]))
    print(f"  Median pairwise sq distances: X={med_X:.2f} (good σ≈{np.sqrt(med_X / (2 * X_acts.shape[1])):.3f})")
    print(f"                                Z_mid={med_Z:.2f} (good σ≈{np.sqrt(med_Z / (2 * Zs.shape[1])):.3f})")

    # ── HSIC computation ──
    print(f"\n[2/2] Computing per-tensor HSIC scores...")
    t0 = time.time()
    if args.use_gpu_hsic:
        results = compute_tensor_hsic_gpu(
            X_acts, Y_acts, in_acts, out_acts,
            kernel_hidden=args.kernel_hidden, kernel_y=args.kernel_y,
            sigma=args.sigma, lx=args.lx, ly=args.ly,
            device=args.hsic_device,
        )
    else:
        results = compute_tensor_hsic_cpu(
            X_acts, Y_acts, in_acts, out_acts,
            kernel_hidden=args.kernel_hidden, kernel_y=args.kernel_y,
            sigma=args.sigma, lx=args.lx, ly=args.ly,
        )
    t_hsic = time.time() - t0
    print(f"  HSIC: {t_hsic:.1f}s")

    results["metadata"] = {
        "checkpoint": str(ckpt_dir),
        "calib_dir":  str(calib_dir),
        "n_frames": int(use_frames),
        "kernel_hidden": args.kernel_hidden,
        "kernel_y": args.kernel_y,
        "sigma": args.sigma, "lx": args.lx, "ly": args.ly,
        "pool_tokens": args.pool_tokens,
        "t_collect_s": t_collect, "t_hsic_s": t_hsic,
    }

    # Quick summary
    f_outs = [info["F_out"] for info in results["tensors"].values()]
    sens   = [info["sens"]  for info in results["tensors"].values()]
    print(f"\n  F_out:  min={min(f_outs):.3e} max={max(f_outs):.3e} mean={np.mean(f_outs):.3e}")
    print(f"  sens:   min={min(sens):.3e} max={max(sens):.3e} mean={np.mean(sens):.3e}")
    n_neg_f = sum(1 for v in f_outs if v < 0)
    print(f"  negative F_out: {n_neg_f}/{len(f_outs)}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved scores to {out_path}")


if __name__ == "__main__":
    main()
