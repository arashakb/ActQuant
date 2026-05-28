#!/usr/bin/env python3
"""
Action-Mahalanobis Fisher (AMF) for OpenVLA-OFT Quantization

Computes per-weight AMF diagonal as a drop-in replacement for imatrix in llama-quantize.

Unlike standard diagonal Fisher (compute_fisher.py, SqueezeLLM-style NLL gradients),
AMF differentiates through the OFT regression action head:

    F_ii^AMF = (1/N) Σ_d (∂L_amf/∂w_i)²

    L_amf = ||Σ_task^{-1/2} (a_pred(w) - a_gt)||_F²

where:
  - a_pred = L1RegressionActionHead(hidden_states at action positions)  [shape: (8, 7)]
  - a_gt   = ground-truth continuous actions decoded from action token IDs [shape: (8, 7)]
  - Σ_task = per-action-dimension variance (diagonal), estimated from calibration data
  - hidden_states come from the LLM backbone (not lm_head)

This is VLA-specific because:
  1. Gradient flows through the actual OFT action head, not vocabulary logits
  2. Σ_task weights by per-task action precision requirements
  3. Measures "which LLM weights affect the physical robot action"

Gradient flows:  LLM weights → LLM hidden states → action head → actions → L_amf
Action head parameters are FROZEN (no_grad). Only LLM weights accumulate grad².

--calib-targets accepts the same _targets.npy files used by compute_fisher.py
(action token ID targets, shape [N, >=56]). These are decoded to continuous actions
using ActionTokenizer — so the targets are always perfectly aligned with the .bin file
since both were produced by the same get_llm_calib_data.py run.

Usage (single GPU):
    python tools/fisher-diag/compute_amf.py \
        --checkpoint ~/arash/openvla-oft-checkpoints/oft_spatial \
        --action-head-checkpoint ~/arash/openvla-oft-checkpoints/oft_spatial/action_head--300000_checkpoint.pt \
        --calib-data ~/arash/openvla-oft/calib_data/spatial_20.bin \
        --calib-targets ~/arash/openvla-oft/calib_data/spatial_20_targets.npy \
        --num-samples 500 \
        --output amf_spatial.gguf

Usage (multi GPU):
    python tools/fisher-diag/compute_amf.py \
        --checkpoint ~/arash/openvla-oft-checkpoints/oft_combined \
        --action-head-checkpoint ~/arash/openvla-oft-checkpoints/oft_combined/action_head--300000_checkpoint.pt \
        --calib-data ~/arash/openvla-oft/calib_data/spatial_20.bin \
                     ~/arash/openvla-oft/calib_data/object_20.bin \
                     ~/arash/openvla-oft/calib_data/goal_20.bin \
                     ~/arash/openvla-oft/calib_data/long_20.bin \
        --calib-targets ~/arash/openvla-oft/calib_data/spatial_20_targets.npy \
                        ~/arash/openvla-oft/calib_data/object_20_targets.npy \
                        ~/arash/openvla-oft/calib_data/goal_20_targets.npy \
                        ~/arash/openvla-oft/calib_data/long_20_targets.npy \
        --num-gpus 8 \
        --output amf_combined.gguf
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
import torch.multiprocessing as mp

# Add gguf-py to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "gguf-py"))
# Add openvla-oft to path (for L1RegressionActionHead)
_OPENVLA_PATH = project_root.parent / "openvla-oft"
sys.path.insert(0, str(_OPENVLA_PATH))

from gguf import GGUFWriter


# ─── Calibration Data Loading (identical to compute_fisher.py) ───────────────

CALIB_MAGIC = b"OPENVLA_CALIB\0\0\0"


def load_calibration_data(bin_path: str):
    """Load pre-computed multimodal embeddings from binary calibration file."""
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


def load_calibration_metadata(bin_path: str) -> dict:
    """Read .bin header only — no embeddings loaded."""
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
    return {"num_frames": num_frames, "hidden_dim": hidden_dim,
            "seq_lengths": seq_lengths, "offsets": offsets, "data_start": data_start}


def load_embeddings_for_indices(bin_paths: list, global_indices: np.ndarray, frame_counts: list) -> list:
    """Load only specific frames from .bin files using direct seeks.

    global_indices: sorted array of frame indices (global, spanning all bin files)
    frame_counts:   number of frames per bin file (e.g. [2344, 2830, 2126, 5198])

    Workers call this directly so the main process never loads embeddings into RAM.
    """
    # Build cumulative offsets to map global index → (file_idx, local_idx)
    cum = np.cumsum([0] + frame_counts)

    # Group global indices by file
    file_to_local: dict = {i: [] for i in range(len(bin_paths))}
    index_to_file_local = {}
    for gi in global_indices:
        fi = int(np.searchsorted(cum[1:], gi, side='right'))
        li = int(gi - cum[fi])
        file_to_local[fi].append(li)
        index_to_file_local[int(gi)] = (fi, li)

    # Load metadata per file (header only — fast)
    metas = [load_calibration_metadata(p) for p in bin_paths]

    # Seek-read only the needed frames from each file
    file_embeddings: dict = {}  # (fi, li) -> embedding
    for fi, local_indices in file_to_local.items():
        if not local_indices:
            continue
        meta = metas[fi]
        with open(bin_paths[fi], "rb") as f:
            for li in local_indices:
                f.seek(meta["data_start"] + meta["offsets"][li])
                n_floats = meta["seq_lengths"][li] * meta["hidden_dim"]
                raw = f.read(n_floats * 4)
                emb = np.frombuffer(raw, dtype=np.float32).copy().reshape(
                    meta["seq_lengths"][li], meta["hidden_dim"])
                file_embeddings[(fi, li)] = emb

    # Return in original global index order
    return [file_embeddings[index_to_file_local[int(gi)]] for gi in global_indices]


def load_action_tokenizer(checkpoint_path: str):
    """Load ActionTokenizer from VLA checkpoint for decoding token ID targets."""
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from prismatic.vla.action_tokenizer import ActionTokenizer

    try:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    except ValueError:
        pass  # already registered by load_llm

    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
    return ActionTokenizer(processor.tokenizer)


def decode_action_targets(token_id_targets: np.ndarray, action_tokenizer, n_action_tokens: int = 56) -> np.ndarray:
    """Decode action token IDs to continuous actions.

    token_id_targets: (N, >=n_action_tokens) — first n_action_tokens columns are action tokens
                      (same format as _targets.npy produced by get_llm_calib_data.py)
    Returns: (N, 8, 7) float32 continuous actions in normalized space (~[-1, 1])
    """
    act_tokens = token_id_targets[:, :n_action_tokens]  # (N, 56) — drop EOS/pad columns
    N = act_tokens.shape[0]
    decoded = np.stack(
        [action_tokenizer.decode_token_ids_to_actions(act_tokens[i]) for i in range(N)],
        axis=0,
    )  # (N, 56)
    return decoded.reshape(N, -1, 7).astype(np.float32)  # (N, 8, 7)


def load_all_calibration_data(bin_paths: list, calib_targets_paths: list, action_tokenizer):
    """Load and concatenate multiple calibration data files.

    calib_targets_paths: paths to _targets.npy files (action token IDs, shape [N, >=56]).
    Targets are decoded to continuous actions using ActionTokenizer — guaranteed aligned
    with embeddings since both come from the same get_llm_calib_data.py run.
    """
    from concurrent.futures import ThreadPoolExecutor

    def _load_one(args):
        bin_path, tgt_path = args
        print(f"--- Loading {bin_path} ---")
        embeddings, metadata = load_calibration_data(bin_path)
        token_ids = np.load(tgt_path)  # (N, >=56) int64 action token IDs
        targets = decode_action_targets(token_ids, action_tokenizer)  # (N, 8, 7) float32
        print(f"    targets decoded: token_ids={token_ids.shape} → actions={targets.shape}")
        if len(embeddings) != len(targets):
            n = min(len(embeddings), len(targets))
            print(f"WARNING: embeddings ({len(embeddings)}) / targets ({len(targets)}) "
                  f"mismatch in {bin_path}, using {n}")
            embeddings = embeddings[:n]
            targets = targets[:n]
        return embeddings, targets, metadata

    with ThreadPoolExecutor(max_workers=len(bin_paths)) as ex:
        results = list(ex.map(_load_one, zip(bin_paths, calib_targets_paths)))

    all_embeddings, all_targets, last_metadata = [], [], None
    for embeddings, targets, metadata in results:
        all_embeddings.extend(embeddings)
        all_targets.append(targets)
        last_metadata = metadata

    combined_targets = np.concatenate(all_targets, axis=0)  # (N_total, 8, 7)
    print(f"\nCombined: {len(all_embeddings)} frames from {len(bin_paths)} files")
    return all_embeddings, combined_targets, last_metadata


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_llm(checkpoint_path: str, device: str = "cuda"):
    """Load OpenVLA-OFT and return the LLM backbone component."""
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
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
    llm = vla.language_model.to(device)
    llm.eval()
    del vla
    torch.cuda.empty_cache()

    total_params = sum(p.numel() for p in llm.parameters())
    llm_hidden_dim = llm.config.hidden_size
    print(f"LLM loaded: {total_params/1e9:.2f}B parameters, hidden_dim={llm_hidden_dim}, device={device}")
    return llm, llm_hidden_dim


def load_action_head(checkpoint_path: str, llm_hidden_dim: int, device: str = "cuda"):
    """Load OFT L1RegressionActionHead from checkpoint and freeze it.

    The action head is separate from the HF model — stored in action_head--*.pt.
    We freeze it: only LLM weights accumulate AMF gradients.
    """
    from prismatic.models.action_heads import L1RegressionActionHead

    print(f"Loading action head from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")

    # Strip DataParallel "module." prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Detect hidden_dim from fc1 weight (shape: [hidden_dim, input_dim*ACTION_DIM])
    fc1_key = next((k for k in state_dict if k.endswith("model.fc1.weight")), None)
    hidden_dim = state_dict[fc1_key].shape[0] if fc1_key else 4096

    action_head = L1RegressionActionHead(
        input_dim=llm_hidden_dim, hidden_dim=hidden_dim, action_dim=7
    )
    action_head.load_state_dict(state_dict)
    action_head = action_head.to(device)
    action_head.eval()

    # Freeze: grad flows through the computation graph but not into action head weights
    for p in action_head.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in action_head.parameters())
    print(f"Action head loaded: {n_params/1e6:.2f}M parameters, hidden_dim={hidden_dim}, FROZEN")
    return action_head


# ─── Target Weight Selection (same as compute_fisher.py) ─────────────────────

def get_target_params(llm):
    """Get list of (name, param) for LLM weight tensors to accumulate AMF diagonal."""
    params = []
    for name, param in llm.named_parameters():
        if param.ndim != 2:
            continue
        if "embed_tokens" in name:
            continue
        if "norm" in name:
            continue
        if "lm_head" in name:
            # lm_head is off the OFT loss path (action head replaces it),
            # so its grad is always None — produces a zero tensor in the GGUF.
            continue
        params.append((name, param))
    total = sum(p.numel() for _, p in params)
    print(f"Target: {len(params)} tensors ({total/1e9:.2f}B parameters)")
    return params


def hf_name_to_gguf_name(hf_name: str) -> str:
    """Convert HuggingFace LLM parameter name to GGUF tensor name."""
    name = hf_name
    if name.startswith("model."):
        name = name[len("model."):]
    name = name.replace("layers.", "blk.")
    name = name.replace("self_attn.q_proj", "attn_q")
    name = name.replace("self_attn.k_proj", "attn_k")
    name = name.replace("self_attn.v_proj", "attn_v")
    name = name.replace("self_attn.o_proj", "attn_output")
    name = name.replace("mlp.gate_proj", "ffn_gate")
    name = name.replace("mlp.up_proj", "ffn_up")
    name = name.replace("mlp.down_proj", "ffn_down")
    name = name.replace("lm_head", "output")
    return name


# ─── Σ_task Computation ───────────────────────────────────────────────────────

def compute_sigma_task(action_targets: np.ndarray) -> np.ndarray:
    """Compute per-action-dimension std from calibration targets.

    action_targets: (N, 8, 7)
    Returns sigma_inv_sq: (7,) — Mahalanobis weights (1/sigma²)
    """
    a_flat = action_targets.reshape(-1, 7)  # (N*8, 7)
    sigma = a_flat.std(axis=0) + 1e-8       # (7,)
    sigma_inv_sq = 1.0 / (sigma ** 2)       # (7,)

    dim_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    print(f"\nΣ_task (from {len(a_flat)} action samples):")
    print(f"  {'dim':<10} {'sigma':>10} {'weight (1/σ²)':>15}")
    print(f"  {'-'*38}")
    for k, name in enumerate(dim_names):
        print(f"  {name:<10} {sigma[k]:>10.4f} {sigma_inv_sq[k]:>15.4f}")

    return sigma_inv_sq.astype(np.float32)


# ─── Single-GPU AMF Diagonal Computation ─────────────────────────────────────

def compute_amf_diagonal(
    llm,
    action_head,
    target_params: list,
    embeddings: list,
    action_targets: np.ndarray,
    sigma_inv_sq: np.ndarray,
    n_action_tokens: int,
    batch_size: int,
    device: str,
    num_samples: int,
    base_seed: int,
):
    """Compute per-weight AMF diagonal on a single GPU.

    F_ii^AMF = (1/N) Σ_d (∂L_amf/∂w_i)²
    L_amf = Σ_k sigma_inv_sq[k] * Σ_{t=0}^{7} (a_pred[t,k] - a_gt[t,k])²

    For each calibration frame:
      1. Forward LLM backbone → hidden states (no lm_head, no NLL)
      2. Extract hidden states at last n_action_tokens positions
      3. Forward action head (frozen) → predicted continuous actions (8, 7)
      4. Compute Mahalanobis loss w.r.t. ground-truth actions
      5. Backward → accumulate grad² for LLM weights only
    """
    num_frames = len(embeddings)
    sigma_inv_sq_t = torch.tensor(sigma_inv_sq, dtype=torch.float32, device=device)

    # Sample subset if requested
    if num_samples > 0 and num_samples < num_frames:
        rng = np.random.RandomState(base_seed)
        indices = rng.choice(num_frames, num_samples, replace=False)
        indices.sort()
        embeddings = [embeddings[i] for i in indices]
        action_targets = action_targets[indices]
        print(f"Using {num_samples}/{num_frames} calibration frames")
    else:
        num_samples = num_frames
        print(f"Using all {num_frames} calibration frames")

    # Initialize accumulators on CPU
    accumulators = {}
    for name, param in target_params:
        gguf_name = hf_name_to_gguf_name(name)
        accumulators[gguf_name] = torch.zeros(param.shape, dtype=torch.float32)

    # Only LLM target params require grad
    all_params_set = set(id(p) for _, p in target_params)
    for param in llm.parameters():
        param.requires_grad_(id(param) in all_params_set)

    total_batches = (num_samples + batch_size - 1) // batch_size
    losses = []
    t_start = time.time()

    print(f"\n{'Batch':>6s} {'AMF loss':>12s} {'Rate':>12s} {'ETA':>8s}")
    print("-" * 46)

    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_embs = embeddings[batch_start:batch_end]
        batch_tgt = action_targets[batch_start:batch_end]  # (B, 8, 7)
        batch_idx = batch_start // batch_size

        # Build padded input tensors
        B = len(batch_embs)
        max_len = max(e.shape[0] for e in batch_embs)
        hidden_dim = batch_embs[0].shape[1]

        input_embeds = torch.zeros(B, max_len, hidden_dim, dtype=torch.bfloat16, device=device)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)

        for i, emb in enumerate(batch_embs):
            seq_len = emb.shape[0]
            input_embeds[i, :seq_len] = torch.from_numpy(emb.copy()).to(
                dtype=torch.bfloat16, device=device
            )
            attention_mask[i, :seq_len] = 1

        # ── Forward LLM backbone (no lm_head) ────────────────────────────────
        llm.zero_grad()
        backbone_out = llm.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        # last_hidden_state: (B, seq_len, hidden_dim)
        hidden_states = backbone_out.last_hidden_state

        # ── Extract action token hidden states at correct positions ──────────
        # Sequence: [...instruction..., action_1..action_56, EOS]
        # action tokens are at [sl-57 : sl-1] per sample (not [-56:] which is off-by-one + EOS)
        actual_seq_lens = attention_mask.sum(dim=1)  # (B,)
        action_hiddens = []
        for i in range(B):
            sl = actual_seq_lens[i].item()
            action_hiddens.append(hidden_states[i, sl - n_action_tokens - 1 : sl - 1, :])
        action_hidden = torch.stack(action_hiddens, dim=0)  # (B, 56, 4096)

        # ── Action head forward (frozen) ──────────────────────────────────────
        # Cast bfloat16 → float32 (LLM backbone outputs bf16, action head weights are f32).
        # .float() is differentiable: grad flows back through the cast into LLM weights.
        # predict_action: (B, 56, 4096) → (B, 8, 7) normalized actions
        a_pred = action_head.predict_action(action_hidden.float())  # (B, 8, 7)

        # ── Mahalanobis loss ──────────────────────────────────────────────────
        a_gt = torch.tensor(batch_tgt, dtype=torch.float32, device=device)  # (B, 8, 7)
        diff = (a_pred - a_gt)                                                # (B, 8, 7)
        # Weight each action dimension by 1/sigma²
        loss = (diff.pow(2) * sigma_inv_sq_t).sum()

        losses.append(loss.float().item())

        # ── Backward: grad flows through action head into LLM weights ─────────
        loss.backward()

        # Accumulate grad² on CPU
        for name, param in target_params:
            if param.grad is not None:
                gguf_name = hf_name_to_gguf_name(name)
                accumulators[gguf_name] += param.grad.float().square().cpu()

        llm.zero_grad(set_to_none=True)
        del input_embeds, attention_mask, backbone_out, hidden_states, action_hidden
        del a_pred, a_gt, diff, loss

        # Progress
        elapsed = time.time() - t_start
        rate = (batch_idx + 1) / elapsed * 60
        remaining_batches = total_batches - batch_idx - 1
        eta = remaining_batches / (rate / 60) if rate > 0 else 0
        print(f"{batch_idx+1:>4d}/{total_batches:<2d} {losses[-1]:>12.4f} "
              f"{rate:>8.1f} b/min {eta:>7.1f}s")

    total_time = time.time() - t_start

    # Normalize: F_ii^AMF = (1/N) Σ grad²
    amf_diags = {}
    print(f"\n{'Tensor':<45s} {'Shape':>15s} {'Mean':>12s} {'Max':>12s}")
    print("-" * 90)
    for name, param in target_params:
        gguf_name = hf_name_to_gguf_name(name)
        f_diag = (accumulators[gguf_name] / num_samples).numpy()
        amf_diags[gguf_name] = f_diag
        print(f"{gguf_name:<45s} {str(f_diag.shape):>15s} "
              f"{f_diag.mean():>12.4e} {f_diag.max():>12.4e}")

    stats = {
        "num_samples": num_samples,
        "losses": losses,
        "loss_mean": float(np.mean(losses)),
        "loss_std": float(np.std(losses)),
        "total_time_sec": total_time,
    }
    print(f"\nLoss: mean={stats['loss_mean']:.4f}, std={stats['loss_std']:.4f}")
    print(f"Time: {total_time:.1f}s ({total_time/60:.1f}min)")
    return amf_diags, stats


# ─── Multi-GPU Worker ─────────────────────────────────────────────────────────

def _gpu_worker(
    gpu_id: int,
    checkpoint_path: str,
    action_head_checkpoint: str,
    n_action_tokens: int,
    result_dir: str,
    batch_size: int,
):
    """Worker function: runs on a single GPU, loads its own data slice directly from .bin files."""
    import pickle

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"

    # Tiny pkl — just indices + paths + sigma (kilobytes, not gigabytes)
    slice_path = os.path.join(result_dir, f"gpu_{gpu_id}_data.pkl")
    with open(slice_path, "rb") as f:
        data = pickle.load(f)
    global_indices = data["global_indices"]   # (N_gpu,) int array
    bin_paths      = data["bin_paths"]        # list of str
    frame_counts   = data["frame_counts"]     # list of int
    targets_path   = data["targets_path"]     # path to all_targets.npy
    sigma_inv_sq   = data["sigma_inv_sq"]     # (7,)
    del data

    # Load only this GPU's embeddings directly from .bin files (seek-based, no deserialization)
    print(f"  [GPU {gpu_id}] Loading {len(global_indices)} frames from disk...")
    worker_embeddings = load_embeddings_for_indices(bin_paths, global_indices, frame_counts)

    # Load targets (tiny file) and select this GPU's slice
    all_targets = np.load(targets_path)       # (N_total, 8, 7)
    worker_targets = all_targets[global_indices]  # (N_gpu, 8, 7)
    del all_targets

    num_frames = len(worker_embeddings)

    llm, llm_hidden_dim = load_llm(checkpoint_path, device=device)
    action_head = load_action_head(action_head_checkpoint, llm_hidden_dim, device=device)
    target_params = get_target_params(llm)

    accumulators = {}
    for name, param in target_params:
        gguf_name = hf_name_to_gguf_name(name)
        accumulators[gguf_name] = torch.zeros(param.shape, dtype=torch.float32)

    all_params_set = set(id(p) for _, p in target_params)
    for param in llm.parameters():
        param.requires_grad_(id(param) in all_params_set)

    sigma_inv_sq_t = torch.tensor(sigma_inv_sq, dtype=torch.float32, device=device)
    losses = []
    t_start = time.time()
    total_batches = (num_frames + batch_size - 1) // batch_size

    for start in range(0, num_frames, batch_size):
        end = min(start + batch_size, num_frames)
        batch_embs = worker_embeddings[start:end]
        batch_tgt = worker_targets[start:end]
        batch_idx = start // batch_size

        B = len(batch_embs)
        max_len = max(e.shape[0] for e in batch_embs)
        hidden_dim = batch_embs[0].shape[1]

        input_embeds = torch.zeros(B, max_len, hidden_dim, dtype=torch.bfloat16, device=device)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
        for i, emb in enumerate(batch_embs):
            seq_len = emb.shape[0]
            input_embeds[i, :seq_len] = torch.from_numpy(emb.copy()).to(
                dtype=torch.bfloat16, device=device
            )
            attention_mask[i, :seq_len] = 1

        llm.zero_grad()
        backbone_out = llm.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden_states = backbone_out.last_hidden_state
        actual_seq_lens = attention_mask.sum(dim=1)  # (B,)
        action_hiddens_list = []
        for i in range(B):
            sl = actual_seq_lens[i].item()
            action_hiddens_list.append(hidden_states[i, sl - n_action_tokens - 1 : sl - 1, :])
        action_hidden = torch.stack(action_hiddens_list, dim=0)  # (B, 56, 4096)
        a_pred = action_head.predict_action(action_hidden.float())  # bf16→f32 cast is differentiable
        a_gt = torch.tensor(batch_tgt, dtype=torch.float32, device=device)
        diff = (a_pred - a_gt)
        loss = (diff.pow(2) * sigma_inv_sq_t).sum()
        losses.append(loss.float().item())
        loss.backward()

        for name, param in target_params:
            if param.grad is not None:
                gguf_name = hf_name_to_gguf_name(name)
                accumulators[gguf_name] += param.grad.float().square().cpu()

        llm.zero_grad(set_to_none=True)
        del input_embeds, attention_mask, backbone_out, hidden_states, action_hidden
        del a_pred, a_gt, diff, loss

        elapsed = time.time() - t_start
        rate = (batch_idx + 1) / elapsed * 60
        eta = (total_batches - batch_idx - 1) / (rate / 60) if rate > 0 else 0
        print(f"  [GPU {gpu_id}] {batch_idx+1}/{total_batches} "
              f"loss={losses[-1]:.4f} {rate:.1f} b/min ETA {eta:.0f}s", flush=True)

    elapsed = time.time() - t_start
    print(f"  [GPU {gpu_id}] Done: {num_frames} frames in {elapsed:.1f}s")

    gpu_dir = os.path.join(result_dir, f"gpu_{gpu_id}")
    os.makedirs(gpu_dir, exist_ok=True)
    np.save(os.path.join(gpu_dir, "num_frames.npy"), np.array(num_frames))
    np.save(os.path.join(gpu_dir, "losses.npy"), np.array(losses))
    for gguf_name, acc in accumulators.items():
        safe_name = gguf_name.replace(".", "_").replace("/", "_")
        np.save(os.path.join(gpu_dir, f"acc_{safe_name}.npy"), acc.numpy())
    with open(os.path.join(gpu_dir, "name_map.json"), "w") as _f:
        json.dump(
            {gguf_name.replace(".", "_").replace("/", "_"): gguf_name
             for gguf_name in accumulators},
            _f,
        )

    del llm, action_head, target_params, accumulators
    torch.cuda.empty_cache()
    return gpu_dir


def compute_amf_multi_gpu(
    num_gpus: int,
    gpu_ids: list,
    checkpoint_path: str,
    action_head_checkpoint: str,
    calib_data_paths: list,
    calib_targets_paths: list,
    action_tokenizer,
    n_action_tokens: int,
    batch_size: int,
    num_samples: int,
    base_seed: int,
    output_path: str,
):
    """Distribute AMF diagonal computation across multiple GPUs."""
    import pickle

    result_dir = output_path.replace(".gguf", "_gpu_results")
    os.makedirs(result_dir, exist_ok=True)

    # Load targets only in main process (tiny: N×8×7 float32, a few MB)
    # Embeddings are NOT loaded here — workers load their own slice from disk.
    print("[*] Loading targets and computing Σ_task (no embeddings loaded in main process)...")
    frame_counts = []
    for bp in calib_data_paths:
        meta = load_calibration_metadata(bp)
        frame_counts.append(meta["num_frames"])
        print(f"    {bp}: {meta['num_frames']} frames")
    total_frames = sum(frame_counts)

    # Decode all targets (fast: token IDs → continuous actions, a few MB)
    all_token_ids = np.concatenate([np.load(p) for p in calib_targets_paths], axis=0)
    action_targets = decode_action_targets(all_token_ids, action_tokenizer)  # (N, 8, 7)
    sigma_inv_sq = compute_sigma_task(action_targets)

    # Save decoded targets as a small .npy for workers to mmap
    targets_path = os.path.join(result_dir, "all_targets.npy")
    np.save(targets_path, action_targets)
    del action_targets, all_token_ids

    if num_samples > 0 and num_samples < total_frames:
        rng = np.random.RandomState(base_seed)
        all_indices = rng.choice(total_frames, num_samples, replace=False)
        all_indices.sort()
        print(f"Selected {num_samples}/{total_frames} calibration frames")
    else:
        all_indices = np.arange(total_frames)
        num_samples = total_frames
        print(f"Using all {total_frames} calibration frames")

    chunks = np.array_split(all_indices, num_gpus)

    # Save tiny pkl per worker: just indices + paths (kilobytes, not gigabytes)
    print("Saving per-GPU index slices (tiny)...")
    for i, gpu_id in enumerate(gpu_ids):
        slice_path = os.path.join(result_dir, f"gpu_{gpu_id}_data.pkl")
        with open(slice_path, "wb") as f:
            pickle.dump({
                "global_indices": chunks[i],
                "bin_paths": calib_data_paths,
                "frame_counts": frame_counts,
                "targets_path": targets_path,
                "sigma_inv_sq": sigma_inv_sq,
            }, f)
        print(f"  GPU {gpu_id}: {len(chunks[i])} frames")

    print(f"\nLaunching {num_gpus} GPU workers...")
    t_start = time.time()
    mp.set_start_method("spawn", force=True)
    with mp.Pool(num_gpus) as pool:
        futures = [
            pool.apply_async(
                _gpu_worker,
                args=(gpu_ids[i], checkpoint_path, action_head_checkpoint,
                      n_action_tokens, result_dir, batch_size),
            )
            for i in range(num_gpus)
        ]
        result_paths = []
        for i, f in enumerate(futures):
            try:
                result_paths.append(f.get(timeout=36000))
            except Exception as e:
                print(f"  [GPU {i}] FAILED: {e}")
                raise

    total_time = time.time() - t_start
    print(f"\nAll GPUs done in {total_time:.1f}s ({total_time/60:.1f}min)")

    # Merge per-GPU results
    merged = {}
    total_sample_count = 0
    all_losses = []
    for gpu_dir in result_paths:
        gpu_frames = int(np.load(os.path.join(gpu_dir, "num_frames.npy")))
        total_sample_count += gpu_frames
        all_losses.extend(np.load(os.path.join(gpu_dir, "losses.npy")).tolist())
        with open(os.path.join(gpu_dir, "name_map.json")) as _f:
            name_map = json.load(_f)
        for safe_name, gguf_name in name_map.items():
            acc = np.load(os.path.join(gpu_dir, f"acc_{safe_name}.npy"))
            if gguf_name not in merged:
                merged[gguf_name] = acc.astype(np.float64)
            else:
                merged[gguf_name] += acc.astype(np.float64)

    amf_diags = {}
    print(f"\n{'Tensor':<45s} {'Shape':>15s} {'Mean':>12s} {'Max':>12s}")
    print("-" * 90)
    for gguf_name in sorted(merged):
        f_diag = (merged[gguf_name] / total_sample_count).astype(np.float32)
        amf_diags[gguf_name] = f_diag
        print(f"{gguf_name:<45s} {str(f_diag.shape):>15s} "
              f"{f_diag.mean():>12.4e} {f_diag.max():>12.4e}")

    stats = {
        "num_samples": total_sample_count,
        "losses": all_losses,
        "loss_mean": float(np.mean(all_losses)),
        "loss_std": float(np.std(all_losses)),
        "total_time_sec": total_time,
        "num_gpus": num_gpus,
    }

    import shutil
    shutil.rmtree(result_dir, ignore_errors=True)
    return amf_diags, stats, sigma_inv_sq


# ─── GGUF Output ──────────────────────────────────────────────────────────────

def save_amf_gguf(
    amf_diags: dict,
    output_path: str,
    num_samples: int,
    calib_data_path: str,
    sigma_inv_sq: np.ndarray,
):
    """Save AMF diagonal in GGUF format — drop-in replacement for imatrix.

    Format is compatible with compute_fisher.py output:
        {name}.in_sum2 -> [d_out, d_in] per-weight AMF values (float32)
        {name}.counts  -> [1, 1] with value 1.0
    """
    writer = GGUFWriter(output_path, arch="")

    # Metadata
    writer.add_string("general.type", "amf_diag")
    writer.add_string("amf.method", "action_mahalanobis_fisher")
    writer.add_array("imatrix.datasets", [calib_data_path])
    writer.add_uint32("imatrix.chunk_count", num_samples)
    writer.add_uint32("imatrix.chunk_size", 1)
    writer.add_uint32("amf.num_samples", num_samples)
    writer.add_array("amf.sigma_inv_sq", sigma_inv_sq.tolist())

    for name in sorted(amf_diags):
        f_diag = np.maximum(amf_diags[name], 0.0).astype(np.float32)
        writer.add_tensor(f"{name}.in_sum2", f_diag)
        writer.add_tensor(f"{name}.counts", np.array([[1.0]], dtype=np.float32))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    file_size = os.path.getsize(output_path) / (1024 ** 3)
    print(f"Saved AMF diagonal to {output_path} ({file_size:.2f} GB)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Action-Mahalanobis Fisher (AMF) for OpenVLA-OFT quantization"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to OpenVLA-OFT checkpoint")
    parser.add_argument("--action-head-checkpoint", type=str, required=True,
                        help="Path to action head checkpoint (action_head--*.pt)")
    parser.add_argument("--calib-data", type=str, nargs="+", required=True,
                        help="Path(s) to calibration .bin file(s)")
    parser.add_argument("--calib-targets", type=str, nargs="+", required=True,
                        help="Path(s) to action token ID targets .npy (shape [N, >=56]). "
                             "Same files used by compute_fisher.py — produced by get_llm_calib_data.py. "
                             "Must match --calib-data order.")
    parser.add_argument("--output", type=str, default="amf_diag.gguf",
                        help="Output GGUF file path")
    parser.add_argument("--n-action-tokens", type=int, default=56,
                        help="Number of action tokens (= action_dim × num_actions_chunk = 7×8=56)")
    parser.add_argument("--num-samples", type=int, default=0,
                        help="Number of calibration frames to use (0 = use all)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Frames per forward/backward pass (default: 1)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for single-GPU mode")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs (>1 enables multi-GPU)")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated physical GPU IDs (e.g. '0,1,3,7')")
    args = parser.parse_args()

    if args.gpus is not None:
        args.gpu_ids = [int(x) for x in args.gpus.split(",")]
        args.num_gpus = len(args.gpu_ids)
    else:
        args.gpu_ids = list(range(args.num_gpus))

    print("=" * 70)
    print("Action-Mahalanobis Fisher (AMF) Estimator")
    print("=" * 70)
    print(f"  method          = AMF (grad through OFT action head, Mahalanobis loss)")
    print(f"  n_action_tokens = {args.n_action_tokens}")
    print(f"  num_samples     = {args.num_samples if args.num_samples > 0 else 'all'}")
    print(f"  batch_size      = {args.batch_size}")
    print(f"  num_gpus        = {args.num_gpus}")
    print(f"  gpu_ids         = {args.gpu_ids}")
    print()

    print("[*] Loading action tokenizer...")
    action_tokenizer = load_action_tokenizer(args.checkpoint)

    if args.num_gpus > 1:
        amf_diags, stats, sigma_inv_sq = compute_amf_multi_gpu(
            num_gpus=args.num_gpus,
            gpu_ids=args.gpu_ids,
            checkpoint_path=args.checkpoint,
            action_head_checkpoint=args.action_head_checkpoint,
            calib_data_paths=args.calib_data,
            calib_targets_paths=args.calib_targets,
            action_tokenizer=action_tokenizer,
            n_action_tokens=args.n_action_tokens,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            base_seed=args.base_seed,
            output_path=args.output,
        )
    else:
        embeddings, action_targets, _ = load_all_calibration_data(
            args.calib_data, args.calib_targets, action_tokenizer
        )
        sigma_inv_sq = compute_sigma_task(action_targets)

        llm, llm_hidden_dim = load_llm(args.checkpoint, device=args.device)
        action_head = load_action_head(
            args.action_head_checkpoint, llm_hidden_dim, device=args.device
        )
        target_params = get_target_params(llm)

        print("\n--- Computing AMF Diagonal ---")
        amf_diags, stats = compute_amf_diagonal(
            llm, action_head, target_params,
            embeddings, action_targets, sigma_inv_sq,
            args.n_action_tokens, args.batch_size,
            args.device, args.num_samples, args.base_seed,
        )

    print(f"\n--- Saving to GGUF ---")
    save_amf_gguf(
        amf_diags, args.output, stats["num_samples"],
        ", ".join(args.calib_data), sigma_inv_sq,
    )

    config = {
        "method": "action_mahalanobis_fisher",
        "checkpoint": args.checkpoint,
        "action_head_checkpoint": args.action_head_checkpoint,
        "calib_data": args.calib_data,
        "calib_targets": args.calib_targets,
        "n_action_tokens": args.n_action_tokens,
        "num_samples": stats["num_samples"],
        "batch_size": args.batch_size,
        "base_seed": args.base_seed,
        "num_gpus": args.num_gpus,
        "loss_mean": stats["loss_mean"],
        "loss_std": stats["loss_std"],
        "total_time_sec": stats["total_time_sec"],
        "sigma_inv_sq": sigma_inv_sq.tolist(),
    }
    config_path = args.output.replace(".gguf", "_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
