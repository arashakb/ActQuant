#!/usr/bin/env python3
"""
HSIC Per-Tensor Sensitivity Scores

Computes HSIC(T_output, Y) for each of the 224 linear layers in the LLM,
where T_output is the raw linear output (W × input) mean-pooled over
action token positions, and Y is the ground truth actions.

This gives a direct per-tensor task-relevance score: tensors whose outputs
are more statistically dependent on actions are more sensitive to quantization.

Usage:
    python tools/hsic/compute_hsic_tensor.py \
        --checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
        --calib-data ~/openvla-oft/calib_data/spatial_10.bin \
        --calib-targets ~/openvla-oft/calib_data/spatial_10_targets.npy \
        --output hsic_tensor_scores.json \
        --n-frames 200
"""

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "tools" / "hsic"))
sys.path.insert(0, str(project_root / "tools" / "fisher-diag"))

from hsic_utils import rbf_kernel, linear_kernel, centering_matrix, hsic, hsic_normalized_cca

# New VLA-input calib helpers (optional; only used when --calib-dir is supplied)
try:
    from compute_fisher import load_calib_dir, load_vla, stratified_frame_indices
    from compute_amf_lq import vla_hidden_states, get_n_vision_tokens
    _HAS_VLA_CALIB = True
except Exception as _e:
    _HAS_VLA_CALIB = False
    _vla_calib_import_error = str(_e)


# ─── Calibration Data Loading (same as compute_hsic_ib.py) ──────────────

CALIB_MAGIC = b"OPENVLA_CALIB\0\0\0"


def load_calibration_data(bin_path: str):
    with open(bin_path, "rb") as f:
        magic = f.read(16)
        assert magic == CALIB_MAGIC, f"Bad magic: {magic}"
        version = struct.unpack("<I", f.read(4))[0]
        num_frames = struct.unpack("<I", f.read(4))[0]
        hidden_dim = struct.unpack("<I", f.read(4))[0]
        _reserved = struct.unpack("<IIII", f.read(16))
        _padding = struct.unpack("<I", f.read(4))
        seq_lengths = [struct.unpack("<I", f.read(4))[0] for _ in range(num_frames)]
        offsets = [struct.unpack("<Q", f.read(8))[0] for _ in range(num_frames)]
        data_start = f.tell()
        embeddings = []
        for i in range(num_frames):
            f.seek(data_start + offsets[i])
            n_floats = seq_lengths[i] * hidden_dim
            raw = f.read(n_floats * 4)
            emb = np.frombuffer(raw, dtype=np.float32).reshape(seq_lengths[i], hidden_dim)
            embeddings.append(emb)
    metadata = {
        "version": version, "num_frames": num_frames,
        "hidden_dim": hidden_dim, "seq_lengths": seq_lengths,
    }
    print(f"Loaded {num_frames} frames, hidden_dim={hidden_dim}, "
          f"seq_len range=[{min(seq_lengths)}, {max(seq_lengths)}]")
    return embeddings, metadata


def load_multiple_calibration_data(bin_paths: list):
    """Load and concatenate calibration data from multiple .bin files."""
    all_embeddings = []
    combined_seq_lengths = []
    hidden_dim = None
    for path in bin_paths:
        embeddings, meta = load_calibration_data(path)
        if hidden_dim is None:
            hidden_dim = meta["hidden_dim"]
        else:
            assert hidden_dim == meta["hidden_dim"], \
                f"Hidden dim mismatch: {hidden_dim} vs {meta['hidden_dim']} in {path}"
        all_embeddings.extend(embeddings)
        combined_seq_lengths.extend(meta["seq_lengths"])
    total = len(all_embeddings)
    print(f"Combined: {total} frames from {len(bin_paths)} files, hidden_dim={hidden_dim}")
    metadata = {
        "num_frames": total, "hidden_dim": hidden_dim, "seq_lengths": combined_seq_lengths,
    }
    return all_embeddings, metadata


def load_targets(npy_path: str):
    targets = np.load(npy_path)
    print(f"Loaded targets: shape={targets.shape}, dtype={targets.dtype}")
    return targets


def load_multiple_targets(npy_paths: list):
    """Load and concatenate targets from multiple .npy files."""
    parts = [np.load(p) for p in npy_paths]
    targets = np.concatenate(parts, axis=0)
    print(f"Combined targets: shape={targets.shape} from {len(npy_paths)} files")
    return targets


def targets_to_actions(targets: np.ndarray) -> np.ndarray:
    actions = targets.astype(np.float32)
    actions = (actions - actions.mean(axis=0)) / (actions.std(axis=0) + 1e-8)
    return actions


def derive_gt_paths(bin_paths: list) -> list:
    """Derive *_oft_gt.npy paths from calibration .bin paths (same directory)."""
    return [str(Path(p).with_suffix("").as_posix() + "_oft_gt.npy") for p in bin_paths]


def load_gt_actions(npy_paths: list, n_frames: int) -> np.ndarray:
    """Load ground-truth continuous actions from *_oft_gt.npy files.

    Each file has shape (N, 8, 7) — N frames, 8-step chunk, 7-DoF actions.
    Files are concatenated, truncated to n_frames, reshaped to (N, 56),
    and standardized per-dimension.

    Returns: (n_frames, 56) float32 array — standardized continuous actions.
    """
    parts = [np.load(p) for p in npy_paths]
    raw = np.concatenate(parts, axis=0)  # (N_total, 8, 7)
    raw = raw[:n_frames]
    print(f"GT actions loaded: raw shape={raw.shape} from {len(npy_paths)} files")

    flat = raw.reshape(len(raw), -1).astype(np.float32)  # (N, 56)
    flat = (flat - flat.mean(axis=0)) / (flat.std(axis=0) + 1e-8)
    print(f"GT actions as Y: shape={flat.shape}  "
          f"range=[{flat.min():.3f}, {flat.max():.3f}]")
    return flat


# ─── Model Loading ───────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str = "cuda"):
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    sys.path.insert(0, str(project_root.parent / "openvla-oft"))
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    print(f"Loading model from {checkpoint_path}...")
    vla = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    llm = vla.language_model
    llm = llm.to(device)
    llm.eval()
    n_layers = llm.config.num_hidden_layers
    hidden_dim = llm.config.hidden_size
    del vla
    torch.cuda.empty_cache()
    total_params = sum(p.numel() for p in llm.parameters())
    print(f"LLM loaded: {total_params / 1e9:.2f}B params, {n_layers} layers, "
          f"d_model={hidden_dim}, device={device}")
    return llm, n_layers, hidden_dim


# ─── GGML Name Mapping ──────────────────────────────────────────────────

PROJ_MAP = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}


def get_ggml_name(layer_idx: int, proj_name: str) -> str:
    """Convert layer index + projection name to GGML tensor name."""
    ggml_proj = PROJ_MAP.get(proj_name)
    if ggml_proj is None:
        return None
    return f"blk.{layer_idx}.{ggml_proj}.weight"


# ─── Activation Collection ───────────────────────────────────────────────

def collect_tensor_activations(
    llm,
    embeddings: list,
    n_action_tokens: int,
    n_layers: int,
    device: str,
    n_frames: int,
    batch_layers: int = 4,
    pool_all: bool = False,
) -> tuple:
    """Collect mean-pooled action token inputs AND outputs for every linear layer,
    plus X (first layer input) for the IB functional.

    For each linear layer, captures:
        - input:  the activation going INTO the layer (before W × input)
        - output: the raw linear output (W × input)
    Both mean-pooled over action token positions.

    Also captures X = input to the first transformer layer (before any processing),
    mean-pooled over action token positions.

    Returns:
        X_acts:         (n_frames, d_model) numpy array — first layer input
        tensor_inputs:  {ggml_name: (n_frames, d_in)} numpy arrays
        tensor_outputs: {ggml_name: (n_frames, d_out)} numpy arrays
    """
    use_frames = min(n_frames, len(embeddings))
    layers = llm.model.layers

    tensor_inputs = {}
    tensor_outputs = {}
    X_list = []  # first layer input, mean-pooled over all tokens
    Y_list = []  # last layer output, mean-pooled over all tokens

    print(f"\nCollecting per-tensor input+output activations for {use_frames} frames, {n_layers} layers...")

    for batch_start in range(0, n_layers, batch_layers):
        batch_end = min(batch_start + batch_layers, n_layers)
        batch_range = range(batch_start, batch_end)
        print(f"  Processing layers {batch_start}-{batch_end-1}...")

        # Storage for this batch
        batch_in = {}   # ggml_name -> list of (d_in,) arrays
        batch_out = {}  # ggml_name -> list of (d_out,) arrays
        handles = []

        # Hook for X (first layer input) — only on first batch
        if batch_start == 0:
            def x_hook_fn(module, input, output):
                # input[0] = hidden states entering first layer = X
                x = input[0]
                x_pool = x[0].float() if pool_all else x[0, -n_action_tokens:, :].float()
                X_list.append(x_pool.mean(dim=0).cpu().numpy())

            h_x = layers[0].register_forward_pre_hook(
                lambda module, input: x_hook_fn(module, input, None)
            )
            handles.append(h_x)

        # Hook for Y (last layer output) — only on last batch
        if batch_end >= n_layers:
            def y_hook_fn(module, input, output):
                y = output[0]
                y_pool = y[0].float() if pool_all else y[0, -n_action_tokens:, :].float()
                Y_list.append(y_pool.mean(dim=0).cpu().numpy())

            h_y = layers[n_layers - 1].register_forward_hook(y_hook_fn)
            handles.append(h_y)

        for l in batch_range:
            layer_module = layers[l]

            # Hook every Linear module in this layer
            for proj_name, ggml_suffix in PROJ_MAP.items():
                ggml_name = f"blk.{l}.{ggml_suffix}.weight"
                batch_in[ggml_name] = []
                batch_out[ggml_name] = []

                # Navigate to the submodule
                parts = proj_name.split(".")
                submodule = layer_module
                for part in parts:
                    submodule = getattr(submodule, part)

                def make_hook(name):
                    def hook_fn(module, input, output):
                        # input[0] is the activation before linear: (batch, seq_len, d_in)
                        inp = input[0]
                        inp_pool = inp[0].float() if pool_all else inp[0, -n_action_tokens:, :].float()
                        batch_in[name].append(inp_pool.mean(dim=0).cpu().numpy())

                        # output is the raw linear output: (batch, seq_len, d_out)
                        out_pool = output[0].float() if pool_all else output[0, -n_action_tokens:, :].float()
                        batch_out[name].append(out_pool.mean(dim=0).cpu().numpy())
                    return hook_fn

                h = submodule.register_forward_hook(make_hook(ggml_name))
                handles.append(h)

        # Forward passes
        with torch.no_grad():
            for i in tqdm(range(use_frames), desc=f"  Layers {batch_start}-{batch_end-1}"):
                emb = embeddings[i]
                seq_len = emb.shape[0]
                input_embeds = torch.from_numpy(emb).unsqueeze(0).to(
                    dtype=torch.bfloat16, device=device
                )
                attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)
                llm(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    use_cache=False,
                )

        # Remove hooks
        for h in handles:
            h.remove()

        # Stack and store
        for name in batch_in:
            tensor_inputs[name] = np.stack(batch_in[name])    # (n_frames, d_in)
            tensor_outputs[name] = np.stack(batch_out[name])  # (n_frames, d_out)

        del batch_in, batch_out
        torch.cuda.empty_cache()

    X_acts = np.stack(X_list)  # (n_frames, d_model)
    Y_acts = np.stack(Y_list) if Y_list else None  # (n_frames, d_model)
    return X_acts, Y_acts, tensor_inputs, tensor_outputs


# ─── GPU-Accelerated HSIC Computation ────────────────────────────────────

def _rbf_kernel_torch(X: torch.Tensor, sigma=None) -> torch.Tensor:
    """RBF kernel on GPU with dimension-scaled sigma (matches numpy rbf_kernel)."""
    sq = (X * X).sum(dim=1)
    D = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    D = D.clamp_min_(0.0)
    d = X.shape[1]
    if sigma is not None:
        variance = 2.0 * float(sigma) * float(sigma) * d
    else:
        n = X.shape[0]
        idx = torch.triu_indices(n, n, offset=1, device=X.device)
        flat = D[idx[0], idx[1]]
        sigma_est = float(flat.median().item())
        if sigma_est <= 0:
            sigma_est = float(flat.mean().item())
        if sigma_est < 1e-2:
            sigma_est = 1e-2
        variance = 2.0 * sigma_est * sigma_est
        del flat, idx
    return torch.exp(-D / variance)


def compute_tensor_hsic_scores_gpu(
    X_acts: np.ndarray,
    Y_acts: np.ndarray,
    tensor_inputs: dict,
    tensor_outputs: dict,
    kernel_hidden: str = "rbf",
    kernel_y: str = "linear",
    sigma: float = None,
    lx: float = 1.0,
    ly: float = 1.0,
    device: str = "cuda",
) -> dict:
    """Same per-tensor HSIC computation as compute_tensor_hsic_scores but
    runs all kernel and trace operations on a CUDA device. ~50-100x faster
    than the CPU NumPy path at n>10k.

    Memory: keeps K_X (1.8 GB at n=21k fp32), HK_X, K_Y, HK_Y resident, plus
    one transient tensor K_Z + HK_Z per iteration (~3.6 GB). Fits comfortably
    on a 45 GB card.
    """
    n = X_acts.shape[0]
    print(f"\n[GPU HSIC] n={n}, device={device}, kernel_hidden={kernel_hidden}, "
          f"kernel_y={kernel_y}, sigma={sigma}, lx={lx}, ly={ly}")

    X_t = torch.from_numpy(X_acts.astype(np.float32)).to(device)
    Y_t = torch.from_numpy(Y_acts.astype(np.float32)).to(device)

    print("[GPU HSIC] Precomputing K_X, K_Y, H...")
    if kernel_hidden == "rbf":
        K_X = _rbf_kernel_torch(X_t, sigma=sigma)
    else:
        K_X = X_t @ X_t.T
    if kernel_y == "rbf":
        K_Y = _rbf_kernel_torch(Y_t, sigma=sigma)
    else:
        K_Y = Y_t @ Y_t.T
    del X_t, Y_t

    H = torch.eye(n, device=device, dtype=torch.float32) - 1.0 / n

    HK_X = H @ K_X       # (n, n)
    HK_Y = H @ K_Y       # (n, n)
    del K_X, K_Y

    n2 = (n - 1) ** 2
    results = {"n_samples": n, "n_tensors": len(tensor_outputs), "tensors": {}}
    input_F_cache: dict = {}

    for name in tqdm(sorted(tensor_outputs.keys()), desc="[GPU HSIC] per tensor"):
        Z_out_np = tensor_outputs[name]
        Z_in_np  = tensor_inputs[name]

        Z_out = torch.from_numpy(Z_out_np.astype(np.float32)).to(device)
        if kernel_hidden == "rbf":
            K_out = _rbf_kernel_torch(Z_out, sigma=sigma)
        else:
            K_out = Z_out @ Z_out.T
        del Z_out
        HK_out = H @ K_out
        del K_out

        hsic_x_out = float((HK_X * HK_out.T).sum().item()) / n2
        hsic_y_out = float((HK_out * HK_Y.T).sum().item()) / n2
        F_out = -lx * hsic_x_out + ly * hsic_y_out
        del HK_out

        in_key = Z_in_np.tobytes()[:64]
        if in_key not in input_F_cache:
            Z_in = torch.from_numpy(Z_in_np.astype(np.float32)).to(device)
            if kernel_hidden == "rbf":
                K_in = _rbf_kernel_torch(Z_in, sigma=sigma)
            else:
                K_in = Z_in @ Z_in.T
            del Z_in
            HK_in = H @ K_in
            del K_in
            hsic_x_in = float((HK_X * HK_in.T).sum().item()) / n2
            hsic_y_in = float((HK_in * HK_Y.T).sum().item()) / n2
            F_in = -lx * hsic_x_in + ly * hsic_y_in
            del HK_in
            input_F_cache[in_key] = (hsic_x_in, hsic_y_in, F_in)
        hsic_x_in, hsic_y_in, F_in = input_F_cache[in_key]

        sens = F_out - F_in
        parts = name.split(".")
        results["tensors"][name] = {
            "hsic_x_out": hsic_x_out,
            "hsic_y_out": hsic_y_out,
            "F_out": F_out,
            "hsic_x_in":  hsic_x_in,
            "hsic_y_in":  hsic_y_in,
            "F_in": F_in,
            "sens": sens,
            "d_in":  int(Z_in_np.shape[1]),
            "d_out": int(Z_out_np.shape[1]),
            "layer": int(parts[1]),
            "type":  parts[2],
        }
        torch.cuda.empty_cache()

    del HK_X, HK_Y, H
    torch.cuda.empty_cache()
    return results


# ─── HSIC Computation ────────────────────────────────────────────────────

def compute_tensor_hsic_scores(
    X_acts: np.ndarray,
    Y_acts: np.ndarray,
    tensor_inputs: dict,
    tensor_outputs: dict,
    kernel_hidden: str = "rbf",
    kernel_y: str = "linear",
    hsic_mode: str = "standard",
    sigma: float = 5.0,
    lx: float = 1.0,
    ly: float = 1.0,
) -> dict:
    """Compute HSIC-IB functional for each tensor's input and output.

    F(Z) = -lx * HSIC(X, Z) + ly * HSIC(Z, Y)
    Sens(T) = F(T_output) - F(T_input)

    Raw HSIC terms (hsic_x_out, hsic_y_out, etc.) are always stored so
    callers can compute any (lx, ly) combination in post-processing.

    Supports two HSIC modes:
        - "standard": biased HSIC estimator tr(KHLH)/(n-1)^2
        - "cca": CCA-normalized HSIC from HBaR (scale-invariant)

    Args:
        X_acts:         (n, d_model) — first layer input (X in IB)
        Y_acts:         (n, d_model or action_dim) — target (actions or last hidden)
        tensor_inputs:  {ggml_name: (n, d_in)} arrays
        tensor_outputs: {ggml_name: (n, d_out)} arrays
        hsic_mode:      "standard" or "cca"
        sigma:          bandwidth for RBF kernel (HBaR default: 5.0)
        lx:             compression weight  (default 1.0)
        ly:             relevance weight     (default 1.0)

    Returns:
        results dict with per-tensor HSIC-IB scores
    """
    n = X_acts.shape[0]

    if hsic_mode == "cca":
        # HBaR-style: CCA-normalized HSIC
        # Z kernels: gaussian with dimension-scaled sigma
        # Y kernel: linear (following HBaR config)
        # X kernel: gaussian with dimension-scaled sigma
        k_type_x = kernel_hidden  # "gaussian" or "linear"
        k_type_y = kernel_y       # "linear" or "gaussian"
        if k_type_x == "rbf":
            k_type_x = "gaussian"
        if k_type_y == "rbf":
            k_type_y = "gaussian"

        print(f"\nComputing per-tensor HSIC-CCA ({n} samples, {len(tensor_outputs)} tensors, "
              f"sigma={sigma}, k_hidden={k_type_x}, k_y={k_type_y})...")
        print(f"  X: first layer input, shape={X_acts.shape}")
        print(f"  Y: actions/last hidden, shape={Y_acts.shape}")

        # Cache for shared inputs
        input_F_cache = {}

        results = {
            "n_samples": n,
            "n_tensors": len(tensor_outputs),
            "tensors": {},
        }

        for name in tqdm(sorted(tensor_outputs.keys()), desc="  HSIC-CCA per tensor"):
            Z_out = tensor_outputs[name]
            Z_in = tensor_inputs[name]

            # HSIC(X, Z_out) and HSIC(Z_out, Y) via CCA
            hsic_x_out = hsic_normalized_cca(X_acts, Z_out, sigma=sigma,
                                              k_type_x=k_type_x, k_type_y=k_type_x)
            hsic_y_out = hsic_normalized_cca(Z_out, Y_acts, sigma=sigma,
                                              k_type_x=k_type_x, k_type_y=k_type_y)
            F_out = -lx * hsic_x_out + ly * hsic_y_out

            in_key = Z_in.tobytes()[:64]
            if in_key not in input_F_cache:
                hsic_x_in = hsic_normalized_cca(X_acts, Z_in, sigma=sigma,
                                                 k_type_x=k_type_x, k_type_y=k_type_x)
                hsic_y_in = hsic_normalized_cca(Z_in, Y_acts, sigma=sigma,
                                                 k_type_x=k_type_x, k_type_y=k_type_y)
                F_in = -lx * hsic_x_in + ly * hsic_y_in
                input_F_cache[in_key] = (hsic_x_in, hsic_y_in, F_in)
            hsic_x_in, hsic_y_in, F_in = input_F_cache[in_key]

            sens = F_out - F_in
            parts = name.split(".")

            results["tensors"][name] = {
                "hsic_x_out": hsic_x_out,
                "hsic_y_out": hsic_y_out,
                "F_out": F_out,
                "hsic_x_in": hsic_x_in,
                "hsic_y_in": hsic_y_in,
                "F_in": F_in,
                "sens": sens,
                "d_in": int(Z_in.shape[1]),
                "d_out": int(Z_out.shape[1]),
                "layer": int(parts[1]),
                "type": parts[2],
            }

        return results

    else:
        # Standard HSIC mode
        # Bake sigma into closures so kernel calls are uniform regardless of type.
        if kernel_hidden == "linear":
            hidden_kern_fn = linear_kernel
        else:
            hidden_kern_fn = lambda X, _sigma=sigma: rbf_kernel(X, sigma=_sigma)

        if kernel_y == "linear":
            y_kern_fn = linear_kernel
        else:
            y_kern_fn = lambda X, _sigma=sigma: rbf_kernel(X, sigma=_sigma)

        print(f"\nComputing per-tensor HSIC ({n} samples, {len(tensor_outputs)} tensors, "
              f"kernel_hidden={kernel_hidden}, kernel_y={kernel_y}, sigma={sigma})...")
        print(f"  X: first layer input, shape={X_acts.shape}")
        print(f"  Y: last layer output, shape={Y_acts.shape}")

        print("  Precomputing K_X and K_Y...")
        K_X = hidden_kern_fn(X_acts)
        K_Y = y_kern_fn(Y_acts)
        H = centering_matrix(n)

        input_F_cache = {}

        results = {
            "n_samples": n,
            "n_tensors": len(tensor_outputs),
            "tensors": {},
        }

        for name in tqdm(sorted(tensor_outputs.keys()), desc="  HSIC per tensor"):
            Z_out = tensor_outputs[name]
            Z_in = tensor_inputs[name]

            K_out = hidden_kern_fn(Z_out)
            hsic_x_out = hsic(K_X, K_out, H)
            hsic_y_out = hsic(K_out, K_Y, H)
            F_out = -lx * hsic_x_out + ly * hsic_y_out

            in_key = Z_in.tobytes()[:64]
            if in_key not in input_F_cache:
                K_in = hidden_kern_fn(Z_in)
                hsic_x_in = hsic(K_X, K_in, H)
                hsic_y_in = hsic(K_in, K_Y, H)
                F_in = -lx * hsic_x_in + ly * hsic_y_in
                input_F_cache[in_key] = (hsic_x_in, hsic_y_in, F_in)
            hsic_x_in, hsic_y_in, F_in = input_F_cache[in_key]

            sens = F_out - F_in
            parts = name.split(".")

            results["tensors"][name] = {
                "hsic_x_out": hsic_x_out,
                "hsic_y_out": hsic_y_out,
                "F_out": F_out,
                "hsic_x_in": hsic_x_in,
                "hsic_y_in": hsic_y_in,
                "F_in": F_in,
                "sens": sens,
                "d_in": int(Z_in.shape[1]),
                "d_out": int(Z_out.shape[1]),
                "layer": int(parts[1]),
                "type": parts[2],
            }

        return results


# ─── Output ──────────────────────────────────────────────────────────────

_IQ2XS_HEURISTIC_UPGRADES = {"attn_v", "ffn_down_early"}  # used only for display

def _iq2xs_preset_type(ttype: str, layer: int, n_layers: int = 32) -> str:
    """Return the IQ2_XS hardcoded quant type for a given tensor type and layer."""
    if ttype == "attn_v":
        return "Q2_K"
    if ttype == "ffn_down" and layer < n_layers // 8:
        return "Q2_K"
    return "IQ2_XS"


def print_summary(results: dict):
    n_layers = 32  # assumed
    tensors = results["tensors"]

    print("\n" + "=" * 110)
    print("HSIC PER-TENSOR SENSITIVITY SCORES")
    print("=" * 110)
    print(f"Samples: {results['n_samples']}, Tensors: {results['n_tensors']}")

    # ── Three Sens(T) formulations ──────────────────────────────────────────
    # Formulation 1: F_out = -HSIC(X, Z_out) + HSIC(Z_out, Y)
    # Formulation 2: sens  = F_out - F_in  (marginal contribution of tensor)
    # Formulation 3: hsic_y_out only  (ignores compression term entirely)

    f_out_vals  = [info["F_out"]       for info in tensors.values()]
    sens_vals   = [info["sens"]        for info in tensors.values()]
    hyout_vals  = [info["hsic_y_out"]  for info in tensors.values()]
    hxout_vals  = [info["hsic_x_out"]  for info in tensors.values()]

    n_neg_fout  = sum(1 for v in f_out_vals  if v < 0)
    n_neg_sens  = sum(1 for v in sens_vals   if v < 0)
    n_neg_hyout = sum(1 for v in hyout_vals  if v < 0)
    n_neg_hxout = sum(1 for v in hxout_vals  if v < 0)
    n = len(tensors)

    print(f"\nNegative-score counts (out of {n} tensors):")
    print(f"  F_out  (= -HSIC(X,Z) + HSIC(Z,Y)):  {n_neg_fout:>3} ({100*n_neg_fout/n:.1f}%)")
    print(f"  sens   (= F_out - F_in):             {n_neg_sens:>3} ({100*n_neg_sens/n:.1f}%)")
    print(f"  HSIC(Z,Y) only:                      {n_neg_hyout:>3} ({100*n_neg_hyout/n:.1f}%)")
    print(f"  HSIC(X,Z) only:                      {n_neg_hxout:>3} ({100*n_neg_hxout/n:.1f}%)")

    print(f"\nScore ranges:")
    for label, vals in [("F_out", f_out_vals), ("sens", sens_vals),
                        ("HSIC(Z,Y)", hyout_vals), ("HSIC(X,Z)", hxout_vals)]:
        print(f"  {label:<12}: min={min(vals):.4e}  max={max(vals):.4e}  "
              f"mean={np.mean(vals):.4e}  std={np.std(vals):.4e}")

    # ── IQ2_XS heuristic comparison ─────────────────────────────────────────
    print(f"\n{'─'*110}")
    print("IQ2_XS HEURISTIC vs HSIC SENSITIVITY RANKING")
    print(f"{'─'*110}")
    print("IQ2_XS hardcoded upgrades: attn_v → Q2_K, ffn_down (L0-3) → Q2_K, all else → IQ2_XS")

    # Rank tensors by each Sens(T) formulation and check if upgrades match heuristic
    for label, key in [("F_out", "F_out"), ("sens (F_out-F_in)", "sens"), ("HSIC(Z,Y)", "hsic_y_out")]:
        ranked = sorted(tensors.items(), key=lambda x: x[1][key], reverse=True)
        # Top 32 tensors (number upgraded in heuristic: 32 attn_v + 4 ffn_down_early = 36)
        top36 = set(name for name, _ in ranked[:36])
        heuristic_upgrades = {
            name for name, info in tensors.items()
            if _iq2xs_preset_type(
                info.get("type", name.split(".")[2]),
                info.get("layer", int(name.split(".")[1])),
            ) != "IQ2_XS"
        }
        overlap = top36 & heuristic_upgrades
        print(f"\n  Sens(T) = {label}:")
        print(f"    Heuristic upgrades: {len(heuristic_upgrades)} tensors  |  "
              f"Top-36 by score: {len(top36)}  |  Overlap: {len(overlap)}")
        print(f"    Top-10 most sensitive:")
        for name, info in ranked[:10]:
            h = _iq2xs_preset_type(info["type"], info.get("layer", int(name.split(".")[1])))
            marker = " ★" if h != "IQ2_XS" else ""
            print(f"      {name:<35} {key}={info[key]:>10.4e}{marker}")

    # ── Per tensor-type summary ──────────────────────────────────────────────
    print(f"\n{'─'*110}")
    print("PER TENSOR-TYPE AVERAGES")
    type_sens = {}
    type_F_out = {}
    type_F_in = {}
    type_hyout = {}
    for name, info in tensors.items():
        ttype = name.split(".")[2]
        type_sens.setdefault(ttype, []).append(info["sens"])
        type_F_out.setdefault(ttype, []).append(info["F_out"])
        type_F_in.setdefault(ttype, []).append(info["F_in"])
        type_hyout.setdefault(ttype, []).append(info["hsic_y_out"])

    print(f"{'Type':<15} {'F_out':>12} {'F_in':>12} {'sens':>12} {'HSIC(Z,Y)':>12}  IQ2_XS_default")
    print("-" * 90)
    for ttype in sorted(type_F_out.keys(), key=lambda t: np.mean(type_F_out[t]), reverse=True):
        h_default = _iq2xs_preset_type(ttype, 10)  # mid-layer representative
        print(f"  {ttype:<13}: {np.mean(type_F_out[ttype]):>12.4e} "
              f"{np.mean(type_F_in[ttype]):>12.4e} "
              f"{np.mean(type_sens[ttype]):>12.4e} "
              f"{np.mean(type_hyout[ttype]):>12.4e}  → {h_default}")

    # ── Per-layer table ──────────────────────────────────────────────────────
    print(f"\nPer-layer sensitivity (sens = F_out - F_in) by tensor type:")
    print(f"{'Layer':>5} {'attn_q':>10} {'attn_k':>10} {'attn_v':>10} {'attn_out':>10} "
          f"{'ffn_gate':>10} {'ffn_up':>10} {'ffn_down':>10}")
    print("-" * 85)
    for l in range(n_layers):
        vals = {}
        for ttype in ['attn_q', 'attn_k', 'attn_v', 'attn_output',
                       'ffn_gate', 'ffn_up', 'ffn_down']:
            name = f'blk.{l}.{ttype}.weight'
            vals[ttype] = tensors.get(name, {}).get('sens', 0)
        print(f'{l:>5} {vals["attn_q"]:>10.4e} {vals["attn_k"]:>10.4e} '
              f'{vals["attn_v"]:>10.4e} {vals["attn_output"]:>10.4e} '
              f'{vals["ffn_gate"]:>10.4e} {vals["ffn_up"]:>10.4e} '
              f'{vals["ffn_down"]:>10.4e}')


# ─── Multi-GPU activation collection ─────────────────────────────────────

def _gpu_worker(gpu_id, checkpoint, embeddings_path,
                n_action_tokens, batch_layers, pool_all, output_path):
    """Subprocess: load model on one GPU, collect activations for a frame slice."""
    import pickle, time as _time
    device = f"cuda:{gpu_id}"
    # Stagger model loading to avoid all workers hitting CPU RAM simultaneously
    _time.sleep(gpu_id * 20)
    print(f"[GPU {gpu_id}] Loading model...", flush=True)
    llm, n_layers, _ = load_model(checkpoint, device)

    with open(embeddings_path, "rb") as f:
        embeddings_slice = pickle.load(f)
    print(f"[GPU {gpu_id}] Processing {len(embeddings_slice)} frames...", flush=True)

    X_acts, Y_hidden, tensor_inputs, tensor_outputs = collect_tensor_activations(
        llm, embeddings_slice,
        n_action_tokens=n_action_tokens,
        n_layers=n_layers,
        device=device,
        n_frames=len(embeddings_slice),
        batch_layers=batch_layers,
        pool_all=pool_all,
    )
    del llm
    torch.cuda.empty_cache()

    np.savez_compressed(
        output_path,
        X_acts=X_acts.astype(np.float16),
        Y_hidden=Y_hidden.astype(np.float16) if Y_hidden is not None else np.array([]),
        tensor_inputs=np.array({k: v.astype(np.float16) for k, v in tensor_inputs.items()}, dtype=object),
        tensor_outputs=np.array({k: v.astype(np.float16) for k, v in tensor_outputs.items()}, dtype=object),
    )
    print(f"[GPU {gpu_id}] Done. Saved to {output_path}", flush=True)


def collect_activations_multi_gpu(checkpoint, embeddings, num_gpus,
                                   n_action_tokens, batch_layers, pool_all):
    import pickle
    import multiprocessing as mp

    n = len(embeddings)
    chunk_size = (n + num_gpus - 1) // num_gpus
    chunks = [(i * chunk_size, min((i + 1) * chunk_size, n)) for i in range(num_gpus)]
    chunks = [(s, e) for s, e in chunks if s < e]  # drop empty

    # Save per-GPU slice pickles so each worker only loads its own frames (not all N)
    print(f"Saving {len(chunks)} per-GPU embedding slices...")
    emb_paths = []
    for i, (start, end) in enumerate(chunks):
        path = f"/tmp/hsic_emb_gpu{i}.pkl"
        with open(path, "wb") as f:
            pickle.dump(embeddings[start:end], f)
        emb_paths.append(path)

    output_paths = [f"/tmp/hsic_acts_gpu{i}.npz" for i in range(len(chunks))]

    print(f"Launching {len(chunks)} GPU workers...")
    ctx = mp.get_context("spawn")
    processes = []
    for i, emb_path in enumerate(emb_paths):
        p = ctx.Process(target=_gpu_worker, args=(
            i, checkpoint, emb_path,
            n_action_tokens, batch_layers, pool_all, output_paths[i],
        ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"GPU worker exited with code {p.exitcode}")

    print("Merging results from all GPUs...")
    X_list, Y_list = [], []
    tensor_inputs_lists, tensor_outputs_lists = {}, {}
    for path in output_paths:
        data = np.load(path, allow_pickle=True)
        X_list.append(data["X_acts"])
        if data["Y_hidden"].shape != ():
            Y_list.append(data["Y_hidden"])
        for k, v in data["tensor_inputs"].item().items():
            tensor_inputs_lists.setdefault(k, []).append(v)
        for k, v in data["tensor_outputs"].item().items():
            tensor_outputs_lists.setdefault(k, []).append(v)

    X_acts = np.concatenate(X_list, axis=0)
    Y_hidden = np.concatenate(Y_list, axis=0) if Y_list else None
    tensor_inputs  = {k: np.concatenate(v, axis=0) for k, v in tensor_inputs_lists.items()}
    tensor_outputs = {k: np.concatenate(v, axis=0) for k, v in tensor_outputs_lists.items()}
    return X_acts, Y_hidden, tensor_inputs, tensor_outputs


# ─── VLA-Input Calibration Path (full VLA forward) ────────────────────────

def _build_single_frame_inputs(arrs, idx, device):
    """Build unpadded single-frame VLA inputs from the VLA-input calib arrays.

    Returns (pixel_values, input_ids, attention_mask, proprio, labels) for
    a SINGLE frame, trimmed to the real (non-pad) length so the multimodal
    sequence has no trailing padding.
    """
    am = arrs["attention_mask"][idx]
    real_len = int(am.sum())
    pv = arrs["pixel_values"][idx]                 # (C*num_images, H, W)
    ids = arrs["input_ids"][idx, :real_len]
    proprio = arrs["proprio"][idx]
    tgt = arrs["targets"][idx]                     # (57,)

    pixel_values   = torch.from_numpy(np.ascontiguousarray(pv)).to(device, dtype=torch.bfloat16).unsqueeze(0)
    input_ids      = torch.from_numpy(np.ascontiguousarray(ids)).to(device, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones(1, real_len, dtype=torch.long, device=device)
    proprio_t      = torch.from_numpy(np.ascontiguousarray(proprio)).to(device, dtype=torch.bfloat16).unsqueeze(0)

    n_action_tokens = 57
    labels_np = np.full((1, real_len), -100, dtype=np.int64)
    labels_np[0, real_len - n_action_tokens:real_len] = tgt
    labels = torch.from_numpy(labels_np).to(device)
    return pixel_values, input_ids, attention_mask, proprio_t, labels


def collect_tensor_activations_vla_input(
    vla, proprio_projector, calib_arrs, indices,
    n_action_tokens, n_layers, device,
    batch_layers=4, pool_all=False, num_images_in_input=2,
):
    """Same shape as collect_tensor_activations(...) but per frame runs the full
    VLA forward (vision encoders + multimodal projector + proprio projector +
    LLM) instead of starting from a precomputed multimodal embedding.

    For each batch of `batch_layers` LLM layers, hooks are installed on the
    Linear sub-modules; for each frame we run the full VLA forward; hooks
    capture the per-tensor inputs and outputs.

    Returns: X_acts, Y_hidden, tensor_inputs, tensor_outputs (same as old).
    """
    use_proprio = proprio_projector is not None
    n_vision_tokens = get_n_vision_tokens(vla, num_images_in_input, use_proprio)

    llm = vla.language_model
    layers = llm.model.layers

    n_frames = len(indices)
    tensor_inputs = {}
    tensor_outputs = {}
    X_list = []
    Y_list = []

    print(f"\nCollecting per-tensor activations for {n_frames} frames, {n_layers} layers, "
          f"n_vision_tokens={n_vision_tokens}, pool_all={pool_all} (VLA-input mode)...")

    for batch_start in range(0, n_layers, batch_layers):
        batch_end = min(batch_start + batch_layers, n_layers)
        batch_range = range(batch_start, batch_end)
        print(f"  Processing layers {batch_start}-{batch_end-1}...")

        batch_in = {}
        batch_out = {}
        handles = []

        if batch_start == 0:
            def x_hook_fn(module, input, output):
                x = input[0]
                x_pool = x[0].float() if pool_all else x[0, -n_action_tokens:, :].float()
                X_list.append(x_pool.mean(dim=0).cpu().numpy())
            handles.append(layers[0].register_forward_pre_hook(
                lambda module, input: x_hook_fn(module, input, None)
            ))

        if batch_end >= n_layers:
            def y_hook_fn(module, input, output):
                y = output[0]
                y_pool = y[0].float() if pool_all else y[0, -n_action_tokens:, :].float()
                Y_list.append(y_pool.mean(dim=0).cpu().numpy())
            handles.append(layers[n_layers - 1].register_forward_hook(y_hook_fn))

        for l in batch_range:
            layer_module = layers[l]
            for proj_name, ggml_suffix in PROJ_MAP.items():
                ggml_name = f"blk.{l}.{ggml_suffix}.weight"
                batch_in[ggml_name]  = []
                batch_out[ggml_name] = []
                parts = proj_name.split(".")
                submodule = layer_module
                for part in parts:
                    submodule = getattr(submodule, part)

                def make_hook(name):
                    def hook_fn(module, input, output):
                        inp = input[0]
                        inp_pool = inp[0].float() if pool_all else inp[0, -n_action_tokens:, :].float()
                        batch_in[name].append(inp_pool.mean(dim=0).cpu().numpy())
                        out_pool = output[0].float() if pool_all else output[0, -n_action_tokens:, :].float()
                        batch_out[name].append(out_pool.mean(dim=0).cpu().numpy())
                    return hook_fn

                handles.append(submodule.register_forward_hook(make_hook(ggml_name)))

        with torch.no_grad():
            for i in tqdm(range(n_frames), desc=f"  Layers {batch_start}-{batch_end-1}"):
                idx = int(indices[i])
                pv, ids, am, pr, lbl = _build_single_frame_inputs(calib_arrs, idx, device)
                # vla_hidden_states runs vision+projector via calibration_mode, then
                # forwards through llm.model so the registered hooks fire on every
                # Linear sub-module.
                _ = vla_hidden_states(vla, proprio_projector, pv, ids, am, pr, lbl, n_vision_tokens)
                del pv, ids, am, pr, lbl

        for h in handles:
            h.remove()

        for name in batch_in:
            tensor_inputs[name]  = np.stack(batch_in[name])
            tensor_outputs[name] = np.stack(batch_out[name])
        del batch_in, batch_out
        torch.cuda.empty_cache()

    X_acts   = np.stack(X_list)
    Y_hidden = np.stack(Y_list) if Y_list else None
    return X_acts, Y_hidden, tensor_inputs, tensor_outputs


def _gpu_worker_vla_input(worker_id, gpu_id, checkpoint, calib_dir, indices_path, output_path,
                          n_action_tokens, batch_layers, pool_all, num_images_in_input):
    import time as _time
    device = f"cuda:{gpu_id}"
    _time.sleep(worker_id * 15)
    print(f"[W{worker_id}/GPU{gpu_id}] Loading full VLA + proprio projector...", flush=True)
    vla, proprio_projector = load_vla(checkpoint, device, num_images_in_input)
    n_layers = vla.language_model.config.num_hidden_layers

    calib = load_calib_dir(Path(calib_dir))
    indices = np.load(indices_path)
    print(f"[W{worker_id}/GPU{gpu_id}] Processing {len(indices)} frames...", flush=True)

    X_acts, Y_hidden, tensor_inputs, tensor_outputs = collect_tensor_activations_vla_input(
        vla, proprio_projector, calib["arrs"], indices,
        n_action_tokens=n_action_tokens, n_layers=n_layers, device=device,
        batch_layers=batch_layers, pool_all=pool_all, num_images_in_input=num_images_in_input,
    )
    del vla, proprio_projector
    torch.cuda.empty_cache()

    np.savez_compressed(
        output_path,
        X_acts=X_acts.astype(np.float16),
        Y_hidden=Y_hidden.astype(np.float16) if Y_hidden is not None else np.array([]),
        tensor_inputs=np.array({k: v.astype(np.float16) for k, v in tensor_inputs.items()}, dtype=object),
        tensor_outputs=np.array({k: v.astype(np.float16) for k, v in tensor_outputs.items()}, dtype=object),
    )
    print(f"[W{worker_id}/GPU{gpu_id}] Done. Saved to {output_path}", flush=True)


def collect_activations_multi_gpu_vla_input(checkpoint, calib_dir, indices, num_gpus,
                                            n_action_tokens, batch_layers, pool_all,
                                            num_images_in_input, workers_per_gpu=1):
    import multiprocessing as mp
    n = len(indices)
    total_workers = num_gpus * workers_per_gpu
    chunk_size = (n + total_workers - 1) // total_workers
    chunks = [(i * chunk_size, min((i + 1) * chunk_size, n)) for i in range(total_workers)]
    chunks = [(s, e) for s, e in chunks if s < e]

    indices_paths = []
    for i, (s, e) in enumerate(chunks):
        path = f"/tmp/hsic_vla_indices_gpu{i}.npy"
        np.save(path, np.asarray(indices[s:e]))
        indices_paths.append(path)
    output_paths = [f"/tmp/hsic_vla_acts_gpu{i}.npz" for i in range(len(chunks))]

    print(f"Launching {len(chunks)} workers ({workers_per_gpu}/GPU x {num_gpus} GPUs, VLA-input)...")
    ctx = mp.get_context("spawn")
    procs = []
    for i, idx_path in enumerate(indices_paths):
        gpu_id = i % num_gpus
        p = ctx.Process(target=_gpu_worker_vla_input, args=(
            i, gpu_id, checkpoint, calib_dir, idx_path, output_paths[i],
            n_action_tokens, batch_layers, pool_all, num_images_in_input,
        ))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"GPU worker exited with code {p.exitcode}")

    X_list, Y_list = [], []
    tensor_inputs_lists, tensor_outputs_lists = {}, {}
    for path in output_paths:
        data = np.load(path, allow_pickle=True)
        X_list.append(data["X_acts"])
        if data["Y_hidden"].shape != ():
            Y_list.append(data["Y_hidden"])
        for k, v in data["tensor_inputs"].item().items():
            tensor_inputs_lists.setdefault(k, []).append(v)
        for k, v in data["tensor_outputs"].item().items():
            tensor_outputs_lists.setdefault(k, []).append(v)

    X_acts   = np.concatenate(X_list, axis=0)
    Y_hidden = np.concatenate(Y_list, axis=0) if Y_list else None
    tensor_inputs  = {k: np.concatenate(v, axis=0) for k, v in tensor_inputs_lists.items()}
    tensor_outputs = {k: np.concatenate(v, axis=0) for k, v in tensor_outputs_lists.items()}
    return X_acts, Y_hidden, tensor_inputs, tensor_outputs


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HSIC per-tensor sensitivity for mixed-precision quantization",
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--calib-data", type=str, nargs="+", default=None,
                        help="(Legacy) one or more pre-embedded calibration .bin files. "
                             "Mutually exclusive with --calib-dir.")
    parser.add_argument("--calib-dir", type=str, default=None,
                        help="VLA-input calib directory (new format with pixel_values.npy etc). "
                             "If set, runs full VLA forward (vision + projector + LLM) per frame.")
    parser.add_argument("--spatial-episodes", type=int, default=-1)
    parser.add_argument("--object-episodes",  type=int, default=-1)
    parser.add_argument("--goal-episodes",    type=int, default=-1)
    parser.add_argument("--long-episodes",    type=int, default=-1)
    parser.add_argument("--num-images-in-input", type=int, default=2)
    parser.add_argument("--calib-gt", type=str, nargs="+", default=None,
                        help="Ground-truth action .npy files (shape [N,8,7], *_oft_gt.npy). "
                             "Default: auto-derived from --calib-data by replacing .bin → _oft_gt.npy")
    parser.add_argument("--calib-targets", type=str, nargs="+", default=None,
                        help="Legacy token-ID .npy files. Only used if --calib-gt is absent "
                             "and no *_oft_gt.npy files exist next to --calib-data.")
    parser.add_argument("--output", type=str, default="hsic_tensor_scores.json")
    parser.add_argument("--n-frames", type=int, default=200)
    parser.add_argument("--n-action-tokens", type=int, default=57)
    parser.add_argument("--batch-layers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--y-mode", choices=["actions", "last_hidden", "targets"],
                        default="actions",
                        help="Y source: actions=continuous oft_gt (default), "
                             "targets=action-token IDs from targets.npy (vocabulary IDs), "
                             "last_hidden=final-layer hidden state")
    parser.add_argument("--pool-all", action="store_true",
                        help="Average over all tokens instead of action tokens only")
    parser.add_argument("--kernel-hidden", choices=["rbf", "linear"],
                        default="rbf",
                        help="Kernel for hidden states (X, Z): rbf (HBaR default, continuous activations)")
    parser.add_argument("--kernel-y", choices=["rbf", "linear"],
                        default="linear",
                        help="Kernel for Y (actions): linear (HBaR default for discrete/binned targets)")
    parser.add_argument("--hsic-mode", choices=["standard", "cca"],
                        default="cca",
                        help="HSIC estimator: standard (biased) or cca (HBaR CCA-normalized, default per HSIC-bottleneck paper)")
    parser.add_argument("--sigma", type=float, default=None,
                        help="Sigma for RBF kernel. None (default) = median heuristic auto-calibration")
    parser.add_argument("--lx", type=float, default=1.0,
                        help="Compression weight: F(Z) = -lx*HSIC(X,Z) + ly*HSIC(Z,Y)")
    parser.add_argument("--ly", type=float, default=1.0,
                        help="Relevance weight:   F(Z) = -lx*HSIC(X,Z) + ly*HSIC(Z,Y)")
    parser.add_argument("--cache-acts", type=str, default=None,
                        help="Path to save/load activation cache (.npz). "
                             "If file exists, skip model inference and load from cache. "
                             "Enables fast multi-sigma sweeps without re-running the model.")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs to use for activation collection (default: 1)")
    parser.add_argument("--workers-per-gpu", type=int, default=1,
                        help="VLA replicas per GPU for activation collection (default: 1)")
    parser.add_argument("--use-gpu-hsic", action="store_true",
                        help="Run the per-tensor HSIC kernel/trace computation on GPU "
                             "(50-100x faster at n>10k). Requires CUDA.")
    parser.add_argument("--hsic-device", type=str, default="cuda:0",
                        help="Device for GPU HSIC (default: cuda:0)")

    args = parser.parse_args()

    # Validate input mode: must provide exactly one of --calib-dir / --calib-data
    if (args.calib_dir is None) == (args.calib_data is None):
        raise ValueError("Provide exactly one of --calib-dir (new VLA-input format) "
                         "or --calib-data (legacy .bin files).")
    if args.calib_dir is not None and not _HAS_VLA_CALIB:
        raise ImportError(f"--calib-dir requires fisher-diag helpers: {_vla_calib_import_error}")

    # Resolve GT paths only for the legacy .bin mode
    if args.calib_data is not None and args.calib_gt is None:
        derived = derive_gt_paths(args.calib_data)
        if all(os.path.exists(p) for p in derived):
            args.calib_gt = derived
            print(f"Auto-detected GT files: {args.calib_gt}")
        elif args.calib_targets is None:
            raise ValueError(
                "No *_oft_gt.npy files found next to --calib-data and neither "
                "--calib-gt nor --calib-targets was provided."
            )

    # ── Activation collection or cache load ───────────────────────────────
    cache_path = args.cache_acts
    vla_indices = None  # set below if using --calib-dir
    if cache_path and os.path.exists(cache_path):
        print(f"Loading activation cache from {cache_path}...")
        cache = np.load(cache_path, allow_pickle=True)
        X_acts  = cache["X_acts"]
        tensor_inputs  = cache["tensor_inputs"].item()
        tensor_outputs = cache["tensor_outputs"].item()
        Y_hidden = cache["Y_hidden"] if "Y_hidden" in cache.files and cache["Y_hidden"].shape != () else None
        use_frames = X_acts.shape[0]
        hidden_dim = X_acts.shape[1]
        n_layers   = 32
        t_collect  = 0.0
        if args.calib_dir is not None:
            vla_indices = cache["vla_indices"] if "vla_indices" in cache.files and cache["vla_indices"].shape != () else None
        print(f"Loaded: {use_frames} frames, hidden_dim={hidden_dim}")
    elif args.calib_dir is not None:
        # NEW VLA-input calib path
        calib = load_calib_dir(Path(args.calib_dir))
        episode_filter = {}
        if args.spatial_episodes > 0: episode_filter["spatial"] = args.spatial_episodes
        if args.object_episodes  > 0: episode_filter["object"]  = args.object_episodes
        if args.goal_episodes    > 0: episode_filter["goal"]    = args.goal_episodes
        if args.long_episodes    > 0: episode_filter["long"]    = args.long_episodes
        if episode_filter:
            print(f"Stratified episode filter: {episode_filter}")
            vla_indices = np.sort(stratified_frame_indices(calib["meta"], episode_filter))
        else:
            vla_indices = np.arange(calib["num_frames"])
        if 0 < args.n_frames < len(vla_indices):
            vla_indices = vla_indices[:args.n_frames]
        use_frames = len(vla_indices)
        print(f"Using {use_frames} frames from VLA-input calib")

        t0 = time.time()
        if args.num_gpus > 1:
            print(f"Using {args.num_gpus} GPUs for activation collection (VLA-input)...")
            X_acts, Y_hidden, tensor_inputs, tensor_outputs = collect_activations_multi_gpu_vla_input(
                args.checkpoint, args.calib_dir, vla_indices, args.num_gpus,
                args.n_action_tokens, args.batch_layers, args.pool_all,
                args.num_images_in_input,
                workers_per_gpu=args.workers_per_gpu,
            )
            n_layers   = max(int(k.split(".")[1]) for k in tensor_outputs) + 1
            hidden_dim = X_acts.shape[1]
        else:
            vla, proprio_projector = load_vla(args.checkpoint, args.device, args.num_images_in_input)
            n_layers   = vla.language_model.config.num_hidden_layers
            hidden_dim = vla.language_model.config.hidden_size
            X_acts, Y_hidden, tensor_inputs, tensor_outputs = collect_tensor_activations_vla_input(
                vla, proprio_projector, calib["arrs"], vla_indices,
                n_action_tokens=args.n_action_tokens, n_layers=n_layers,
                device=args.device, batch_layers=args.batch_layers,
                pool_all=args.pool_all, num_images_in_input=args.num_images_in_input,
            )
            del vla, proprio_projector
            torch.cuda.empty_cache()
        t_collect = time.time() - t0
        print(f"Activation collection: {t_collect:.1f}s")
        print(f"X (first layer input): shape={X_acts.shape}")
    else:
        # LEGACY .bin path
        if len(args.calib_data) == 1:
            embeddings, metadata = load_calibration_data(args.calib_data[0])
        else:
            embeddings, metadata = load_multiple_calibration_data(args.calib_data)

        n_actions = 0
        action_count_paths = args.calib_gt if args.calib_gt else args.calib_targets
        for p in action_count_paths:
            n_actions += np.load(p, mmap_mode="r").shape[0]

        use_frames = min(args.n_frames, len(embeddings), n_actions)
        embeddings = embeddings[:use_frames]
        print(f"Using {use_frames} frames")

        t0 = time.time()
        if args.num_gpus > 1:
            print(f"Using {args.num_gpus} GPUs for activation collection...")
            X_acts, Y_hidden, tensor_inputs, tensor_outputs = collect_activations_multi_gpu(
                args.checkpoint, embeddings, args.num_gpus,
                args.n_action_tokens, args.batch_layers, args.pool_all,
            )
            n_layers   = max(int(k.split(".")[1]) for k in tensor_outputs) + 1
            hidden_dim = X_acts.shape[1]
        else:
            llm, n_layers, hidden_dim = load_model(args.checkpoint, args.device)
            X_acts, Y_hidden, tensor_inputs, tensor_outputs = collect_tensor_activations(
                llm, embeddings,
                n_action_tokens=args.n_action_tokens,
                n_layers=n_layers,
                device=args.device,
                n_frames=use_frames,
                batch_layers=args.batch_layers,
                pool_all=args.pool_all,
            )
            del llm
            torch.cuda.empty_cache()
        t_collect = time.time() - t0
        print(f"Activation collection: {t_collect:.1f}s")
        print(f"X (first layer input): shape={X_acts.shape}")

    # ── Build Y ───────────────────────────────────────────────────────────
    if args.calib_dir is not None:
        if vla_indices is None:
            raise RuntimeError("Cache missing vla_indices and --calib-dir was supplied")
        if args.y_mode == "targets":
            # Y = action-token IDs (vocabulary IDs), per position, standardized.
            tgt_path = Path(args.calib_dir) / "targets.npy"
            tgt = np.load(tgt_path, mmap_mode="r")
            Y_raw = np.array(tgt[vla_indices])              # (N, 57) int64
            Y_acts = Y_raw.astype(np.float32)
            Y_acts = (Y_acts - Y_acts.mean(axis=0)) / (Y_acts.std(axis=0) + 1e-8)
            y_source = "targets_calib_dir"
            print(f"Y = standardized action-token IDs (from {tgt_path}): shape={Y_acts.shape}  "
                  f"raw range=[{Y_raw.min()}, {Y_raw.max()}]")
        elif args.y_mode == "last_hidden":
            if Y_hidden is None:
                raise RuntimeError("Y_hidden not available; cannot use --y-mode last_hidden with --calib-dir")
            Y_acts = Y_hidden.astype(np.float32)
            y_source = "last_hidden_calib_dir"
            print(f"Y = last hidden state: shape={Y_acts.shape}")
        else:
            # Default: continuous oft_gt actions
            oft_gt_path = Path(args.calib_dir) / "oft_gt.npy"
            oft_gt = np.load(oft_gt_path, mmap_mode="r")
            Y_raw = np.array(oft_gt[vla_indices])           # (N, 8, 7)
            Y_acts = Y_raw.reshape(len(Y_raw), -1).astype(np.float32)
            Y_acts = (Y_acts - Y_acts.mean(axis=0)) / (Y_acts.std(axis=0) + 1e-8)
            y_source = "oft_gt_calib_dir"
            print(f"Y = ground-truth continuous actions (from {oft_gt_path}): shape={Y_acts.shape}")
    elif args.calib_gt:
        Y_acts = load_gt_actions(args.calib_gt, use_frames)
        y_source = "oft_gt"
        print(f"Y = ground-truth continuous actions: shape={Y_acts.shape}")
    else:
        if len(args.calib_targets) == 1:
            targets = load_targets(args.calib_targets[0])
        else:
            targets = load_multiple_targets(args.calib_targets)
        targets = targets[:use_frames]
        if args.y_mode == "actions":
            Y_acts = targets_to_actions(targets)
            y_source = "token_ids_standardized"
            print(f"Y = standardized token IDs (legacy): shape={Y_acts.shape}")
        else:
            Y_acts = Y_hidden
            y_source = "last_hidden"
            print(f"Y = last hidden state: shape={Y_acts.shape}")

    if cache_path and not os.path.exists(cache_path):
        print(f"Saving activation cache to {cache_path}...")
        cache_dict = dict(
            X_acts=X_acts.astype(np.float16),
            Y_hidden=Y_hidden.astype(np.float16) if Y_hidden is not None else np.array([]),
            tensor_inputs=np.array(
                {k: v.astype(np.float16) for k, v in tensor_inputs.items()},
                dtype=object,
            ),
            tensor_outputs=np.array(
                {k: v.astype(np.float16) for k, v in tensor_outputs.items()},
                dtype=object,
            ),
        )
        if vla_indices is not None:
            cache_dict["vla_indices"] = np.asarray(vla_indices)
        np.savez_compressed(cache_path, **cache_dict)
        print(f"Cache saved (Y not cached — reloaded from oft_gt each run).")

    # Print median pairwise distances for sigma calibration
    sample_name = sorted(tensor_outputs.keys())[len(tensor_outputs)//2]
    Z_sample = tensor_outputs[sample_name].astype(np.float32)
    X_f32    = X_acts.astype(np.float32)
    from hsic_utils import distmat
    D_X = distmat(X_f32)
    D_Z = distmat(Z_sample)
    n = X_f32.shape[0]
    triu = np.triu_indices(n, k=1)
    med_X = float(np.median(D_X[triu]))
    med_Z = float(np.median(D_Z[triu]))
    d_X   = X_f32.shape[1]
    d_Z   = Z_sample.shape[1]
    print(f"\nMedian pairwise squared distances:")
    print(f"  X (d={d_X}): {med_X:.2f}  →  good sigma ≈ {np.sqrt(med_X / (2*d_X)):.3f}")
    print(f"  Z_mid (d={d_Z}, {sample_name}): {med_Z:.2f}  →  good sigma ≈ {np.sqrt(med_Z / (2*d_Z)):.3f}")
    print(f"  (Good sigma gives exp(-D_median/variance)≈0.5, i.e. variance≈D_median/ln2)")
    if args.sigma is not None:
        var_X = 2 * args.sigma**2 * d_X
        print(f"  With sigma={args.sigma}: variance={var_X:.0f}, "
              f"K_median(X)={np.exp(-med_X/var_X):.4f} (want ~0.5)")

    # ── HSIC computation ──────────────────────────────────────────────────
    t0 = time.time()
    if args.use_gpu_hsic:
        results = compute_tensor_hsic_scores_gpu(
            X_acts.astype(np.float32), Y_acts.astype(np.float32),
            {k: v.astype(np.float32) for k, v in tensor_inputs.items()},
            {k: v.astype(np.float32) for k, v in tensor_outputs.items()},
            kernel_hidden=args.kernel_hidden, kernel_y=args.kernel_y,
            sigma=args.sigma, lx=args.lx, ly=args.ly,
            device=args.hsic_device,
        )
    else:
        results = compute_tensor_hsic_scores(
            X_acts.astype(np.float32), Y_acts.astype(np.float32),
            {k: v.astype(np.float32) for k, v in tensor_inputs.items()},
            {k: v.astype(np.float32) for k, v in tensor_outputs.items()},
            kernel_hidden=args.kernel_hidden, kernel_y=args.kernel_y,
            hsic_mode=args.hsic_mode, sigma=args.sigma,
            lx=args.lx, ly=args.ly,
        )
    t_hsic = time.time() - t0
    print(f"HSIC computation: {t_hsic:.1f}s")

    results["metadata"] = {
        "checkpoint": args.checkpoint,
        "calib_data": args.calib_data,
        "calib_dir":  args.calib_dir,
        "episode_filter": {"spatial": args.spatial_episodes, "object": args.object_episodes,
                           "goal": args.goal_episodes, "long": args.long_episodes}
                          if args.calib_dir else None,
        "calib_gt": args.calib_gt,
        "n_frames": use_frames,
        "n_action_tokens": args.n_action_tokens,
        "hidden_dim": hidden_dim,
        "y_source": y_source,
        "kernel_hidden": args.kernel_hidden,
        "kernel_y": args.kernel_y,
        "pool_all": args.pool_all,
        "hsic_mode": args.hsic_mode,
        "sigma": args.sigma,
        "lx": args.lx,
        "ly": args.ly,
        "t_collect_s": t_collect,
        "t_hsic_s": t_hsic,
    }

    print_summary(results)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
