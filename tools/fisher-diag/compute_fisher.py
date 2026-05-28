#!/usr/bin/env python3
"""
Fisher Information Diagonal for OpenVLA-OFT LLM Quantization (SqueezeLLM-style)

Runs the FULL VLA forward pass (vision backbone + projector + proprio projector + LLM)
on raw calibration inputs, then backpropagates CE loss on action tokens to obtain
per-weight squared gradients of the LLM's 2D weight matrices:

    F_ii = (1/N) * sum_{d=1}^{N} (dL/dw_i)^2

Only the LLM's transformer weights + lm_head accumulate gradients; vision/projector
paths run in eval() mode with frozen parameters but are executed so the LLM sees
realistic multimodal embeddings.

Calibration data format (from vla-scripts/get_vla_calib_data.py):
    pixel_values.npy     [N, C*num_images, H, W]  float16
    input_ids.npy        [N, max_seq_len]          int32
    attention_mask.npy   [N, max_seq_len]          bool
    proprio.npy          [N, proprio_dim]          float32
    targets.npy          [N, 57]                   int64   (action token IDs + EOS)
    metadata.json

Usage (single GPU):
    python compute_fisher.py \
        --checkpoint ~/arash/openvla-oft-checkpoints/oft_combined \
        --calib-dir  ~/arash/openvla-oft/calib_data \
        --output     fisher_diag_combined.gguf

Usage (multi GPU):
    python compute_fisher.py \
        --checkpoint ~/arash/openvla-oft-checkpoints/oft_combined \
        --calib-dir  ~/arash/openvla-oft/calib_data \
        --num-gpus 8 --batch-size 2 \
        --output     fisher_diag_combined.gguf
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

# Add gguf-py to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "gguf-py"))

from gguf import GGUFWriter


IGNORE_INDEX = -100


# ─── Calibration Data Loading ────────────────────────────────────────────

def load_calib_dir(calib_dir: Path) -> dict:
    """Memory-map all arrays in the calibration directory."""
    calib_dir = Path(calib_dir)
    with open(calib_dir / "metadata.json") as f:
        meta = json.load(f)

    arrs = {
        "pixel_values":   np.load(calib_dir / "pixel_values.npy",   mmap_mode="r"),
        "input_ids":      np.load(calib_dir / "input_ids.npy",      mmap_mode="r"),
        "attention_mask": np.load(calib_dir / "attention_mask.npy", mmap_mode="r"),
        "targets":        np.load(calib_dir / "targets.npy",        mmap_mode="r"),
    }
    if meta.get("has_proprio", False):
        arrs["proprio"] = np.load(calib_dir / "proprio.npy", mmap_mode="r")
    else:
        arrs["proprio"] = None

    N = meta["total_frames"]
    for k, v in arrs.items():
        if v is not None:
            assert v.shape[0] == N, f"{k} has {v.shape[0]} frames, expected {N}"
    return {"meta": meta, "arrs": arrs, "num_frames": N}


# ─── Stratified Episode Subsampling ──────────────────────────────────────

# Map friendly CLI names to dataset names in metadata["suites"][i]["dataset"]
SUITE_NAME_MAP = {
    "spatial": "libero_spatial_no_noops",
    "object":  "libero_object_no_noops",
    "goal":    "libero_goal_no_noops",
    "long":    "libero_10_no_noops",
}


def stratified_frame_indices(meta: dict, requested: dict) -> np.ndarray:
    """Build a global frame index mask that keeps only a stratified subset of episodes.

    `requested` is a dict like {"spatial": 10, "object": 10, "goal": 10, "long": 30}.
    For each suite, groups episodes by task, picks first `requested[suite] / num_tasks`
    per task (deterministic — matches the original stratified order), then translates
    to global frame indices using per-episode frame counts stored in metadata.json.

    If a suite is missing from `requested` (or set to -1), ALL its episodes are used.

    Requires metadata.json with per-suite "episodes" field (list of {task, num_frames}).
    """
    selected_global_frames: List[np.ndarray] = []
    suite_frame_offset = 0

    for suite_info in meta["suites"]:
        dataset_name = suite_info["dataset"]
        episodes     = suite_info.get("episodes")
        if episodes is None:
            raise RuntimeError(
                f"metadata.json has no per-episode info for {dataset_name}. "
                f"Regenerate calib_data with the updated get_vla_calib_data.py."
            )
        # Cumulative frame offsets within this suite
        ep_offsets = np.cumsum([0] + [e["num_frames"] for e in episodes])

        # Find the CLI key (spatial/object/goal/long) for this dataset
        friendly = next((k for k, v in SUITE_NAME_MAP.items() if v == dataset_name), None)
        want = requested.get(friendly, -1) if friendly else -1

        if want <= 0 or want >= len(episodes):
            # Keep all frames from this suite
            suite_frames = np.arange(suite_frame_offset,
                                     suite_frame_offset + suite_info["num_frames"])
            selected_global_frames.append(suite_frames)
            suite_frame_offset += suite_info["num_frames"]
            continue

        # Stratified: group by task, keep first K per task
        task_to_ep_indices: Dict[str, List[int]] = {}
        for ep_idx, ep in enumerate(episodes):
            task_to_ep_indices.setdefault(ep["task"], []).append(ep_idx)

        num_tasks = len(task_to_ep_indices)
        if want % num_tasks != 0:
            raise ValueError(
                f"{dataset_name}: requested {want} episodes is not divisible by "
                f"{num_tasks} tasks — can't be evenly stratified."
            )
        per_task = want // num_tasks
        kept_ep_ids: List[int] = []
        for task in sorted(task_to_ep_indices.keys()):
            eps = task_to_ep_indices[task]
            if per_task > len(eps):
                raise ValueError(
                    f"{dataset_name}/{task}: requested {per_task} episodes but only "
                    f"{len(eps)} available."
                )
            kept_ep_ids.extend(eps[:per_task])

        # Translate kept episode ids to global frame indices
        kept_frames = []
        for ep_idx in kept_ep_ids:
            start = suite_frame_offset + ep_offsets[ep_idx]
            end   = suite_frame_offset + ep_offsets[ep_idx + 1]
            kept_frames.append(np.arange(start, end))
        selected_global_frames.append(np.concatenate(kept_frames))
        print(f"  {dataset_name}: kept {len(kept_ep_ids)} / {len(episodes)} episodes "
              f"({per_task}/task × {num_tasks} tasks), "
              f"{selected_global_frames[-1].size} frames")
        suite_frame_offset += suite_info["num_frames"]

    return np.concatenate(selected_global_frames)


# ─── Model Loading ───────────────────────────────────────────────────────

def load_vla(checkpoint_path: str, device: str, num_images_in_input: int = 2):
    """Load the full VLA (vision + projector + LLM) and the proprio projector.

    Only LLM 2D weights will have requires_grad=True. Vision/projector params
    are frozen but run forward so the LLM sees realistic multimodal inputs.
    """
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

    # Make prismatic modules importable
    openvla_root = Path(__file__).resolve().parent.parent.parent.parent / "openvla-oft"
    sys.path.insert(0, str(openvla_root))
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    print(f"Loading VLA from {checkpoint_path}...")
    vla = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    vla.vision_backbone.set_num_images_in_input(num_images_in_input)
    vla.eval()

    # Load proprio projector if available
    proprio_projector = None
    proprio_pattern = "proprio_projector"
    candidates = [
        os.path.join(checkpoint_path, f)
        for f in os.listdir(checkpoint_path)
        if proprio_pattern in f and "checkpoint" in f and f.endswith(".pt")
    ]
    if candidates:
        from prismatic.models.projectors import ProprioProjector
        from prismatic.vla.constants import PROPRIO_DIM

        llm_dim = vla.llm_dim
        proprio_projector = ProprioProjector(llm_dim=llm_dim, proprio_dim=PROPRIO_DIM).to(device)
        proprio_projector = proprio_projector.to(torch.bfloat16)
        state_dict = torch.load(candidates[0], weights_only=True)
        # Strip DDP prefix if present
        state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
        proprio_projector.load_state_dict(state_dict)
        proprio_projector.eval()
        for p in proprio_projector.parameters():
            p.requires_grad_(False)
        print(f"  Loaded proprio_projector from {candidates[0]}")
    else:
        print("  No proprio_projector checkpoint found (proprio will not be used)")

    return vla, proprio_projector


def get_target_params(vla):
    """Get list of (hf_name, param) for LLM 2D weights (attn + mlp).

    Skip embeddings, norms, biases, and lm_head (lm_head grad is a huge [vocab, hidden]
    tensor and is typically kept at higher bits anyway).
    """
    params = []
    for name, param in vla.language_model.named_parameters():
        if param.ndim != 2:
            continue
        if "embed_tokens" in name:
            continue
        if "norm" in name:
            continue
        if "lm_head" in name:
            continue
        params.append((name, param))

    total = sum(p.numel() for _, p in params)
    print(f"Target: {len(params)} LLM tensors ({total / 1e9:.2f}B parameters)")
    return params


def freeze_except(vla, target_params, proprio_projector):
    """Set requires_grad=True only on the target LLM weights; freeze everything else."""
    target_ids = set(id(p) for _, p in target_params)
    for param in vla.parameters():
        param.requires_grad_(id(param) in target_ids)
    if proprio_projector is not None:
        for param in proprio_projector.parameters():
            param.requires_grad_(False)


def hf_name_to_gguf_name(hf_name: str) -> str:
    """Convert HF LLM parameter name to GGUF tensor name."""
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


# ─── Batch Building ──────────────────────────────────────────────────────

def build_batch(arrs: dict, indices: np.ndarray, device: str):
    """Materialize a batch from mmap'd arrays and move to device.

    Builds labels: IGNORE_INDEX everywhere except the last `num_actions` positions
    of each row, where targets are placed. Matches how HF LlamaForCausalLM computes
    CE loss with shifting.
    """
    pv  = arrs["pixel_values"][indices]        # (B, C, H, W) float16
    ids = arrs["input_ids"][indices]           # (B, L) int32
    am  = arrs["attention_mask"][indices]      # (B, L) bool
    tgt = arrs["targets"][indices]             # (B, 57) int64
    pr  = arrs["proprio"][indices] if arrs["proprio"] is not None else None

    B, L = ids.shape
    num_actions = tgt.shape[1]

    # Full sequence length per row = number of real tokens (attn sum)
    seq_lens = am.sum(axis=1)  # (B,)

    labels_np = np.full((B, L), IGNORE_INDEX, dtype=np.int64)
    for i in range(B):
        end = int(seq_lens[i])
        labels_np[i, end - num_actions:end] = tgt[i]

    pixel_values   = torch.from_numpy(np.ascontiguousarray(pv)).to(device, dtype=torch.bfloat16, non_blocking=True)
    input_ids      = torch.from_numpy(np.ascontiguousarray(ids)).to(device, dtype=torch.long, non_blocking=True)
    attention_mask = torch.from_numpy(np.ascontiguousarray(am)).to(device, dtype=torch.long, non_blocking=True)
    labels         = torch.from_numpy(labels_np).to(device, non_blocking=True)
    proprio        = (
        torch.from_numpy(np.ascontiguousarray(pr)).to(device, dtype=torch.bfloat16, non_blocking=True)
        if pr is not None else None
    )

    return pixel_values, input_ids, attention_mask, labels, proprio


# ─── Single-GPU Fisher Diagonal ──────────────────────────────────────────

def _run_fisher_on_indices(
    vla,
    proprio_projector,
    target_params,
    arrs,
    indices: np.ndarray,
    batch_size: int,
    device: str,
    progress_prefix: str = "",
):
    """Run forward + backward + accumulate for a given index list.

    Returns:
        accumulators: dict gguf_name -> CPU float32 tensor [d_out, d_in]
        losses: list of per-batch mean losses
        num_samples: number of frames processed
    """
    accumulators = {
        hf_name_to_gguf_name(name): torch.zeros(param.shape, dtype=torch.float32)
        for name, param in target_params
    }

    total_batches = (len(indices) + batch_size - 1) // batch_size
    losses = []
    t_start = time.time()

    for batch_idx, start in enumerate(range(0, len(indices), batch_size)):
        end = min(start + batch_size, len(indices))
        batch_indices = np.sort(indices[start:end])

        pixel_values, input_ids, attention_mask, labels, proprio = build_batch(
            arrs, batch_indices, device
        )

        vla.zero_grad(set_to_none=True)
        output = vla(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            proprio=proprio,
            proprio_projector=proprio_projector,
        )
        loss = output.loss
        losses.append(loss.float().item())

        loss.backward()

        for name, param in target_params:
            if param.grad is not None:
                gguf_name = hf_name_to_gguf_name(name)
                accumulators[gguf_name] += param.grad.float().square().cpu()

        vla.zero_grad(set_to_none=True)
        del pixel_values, input_ids, attention_mask, labels, proprio, output, loss
        torch.cuda.empty_cache()

        elapsed = time.time() - t_start
        rate_per_min = (batch_idx + 1) / elapsed * 60 if elapsed > 0 else 0
        eta = (total_batches - batch_idx - 1) / (rate_per_min / 60) if rate_per_min > 0 else 0
        print(f"{progress_prefix}{batch_idx+1:>4d}/{total_batches:<4d} "
              f"loss={losses[-1]:.4f} {rate_per_min:>6.1f} b/min ETA {eta:>6.0f}s")

    return accumulators, losses, len(indices)


def compute_fisher_single_gpu(
    checkpoint_path: str,
    calib_dir: Path,
    batch_size: int,
    num_samples: int,
    base_seed: int,
    device: str,
    num_images_in_input: int,
    episode_filter: dict = None,
):
    """Single-GPU Fisher diagonal: load everything, select frames, run loop, normalize."""
    calib = load_calib_dir(calib_dir)
    total = calib["num_frames"]
    print(f"Calibration frames: {total}")

    if episode_filter:
        print(f"Stratified episode filter: {episode_filter}")
        indices = np.sort(stratified_frame_indices(calib["meta"], episode_filter))
        num_samples = len(indices)
        print(f"Using {num_samples}/{total} calibration frames after episode filter")
    elif num_samples > 0 and num_samples < total:
        rng = np.random.RandomState(base_seed)
        indices = np.sort(rng.choice(total, num_samples, replace=False))
        print(f"Using {num_samples}/{total} calibration frames (random subsample)")
    else:
        indices = np.arange(total)
        num_samples = total
        print(f"Using all {total} calibration frames")

    vla, proprio_projector = load_vla(checkpoint_path, device, num_images_in_input)
    target_params = get_target_params(vla)
    freeze_except(vla, target_params, proprio_projector)

    print(f"\n{'Batch':>10s}  {'Loss':>7s} {'Rate':>14s} {'ETA':>9s}")
    print("-" * 50)

    accumulators, losses, sample_count = _run_fisher_on_indices(
        vla, proprio_projector, target_params, calib["arrs"],
        indices, batch_size, device,
    )

    total_time = time.time() - (time.time() - sum(1 for _ in losses) * 0)  # no-op, keep structure

    # Normalize
    fisher_diags = {}
    print(f"\n{'Tensor':<45s} {'Shape':>18s} {'Mean F_ii':>12s} {'Max F_ii':>12s} {'Std F_ii':>12s}")
    print("-" * 102)
    for name, param in target_params:
        gguf_name = hf_name_to_gguf_name(name)
        f_diag = (accumulators[gguf_name] / sample_count).numpy()
        fisher_diags[gguf_name] = f_diag
        print(f"{gguf_name:<45s} {str(tuple(f_diag.shape)):>18s} "
              f"{f_diag.mean():>12.4e} {f_diag.max():>12.4e} {f_diag.std():>12.4e}")

    stats = {
        "num_samples":  sample_count,
        "losses":       losses,
        "loss_mean":    float(np.mean(losses)),
        "loss_std":     float(np.std(losses)),
    }
    print(f"\nLoss: mean={stats['loss_mean']:.4f}, std={stats['loss_std']:.4f}")
    return fisher_diags, stats


# ─── Multi-GPU Worker ────────────────────────────────────────────────────

def _gpu_worker(
    gpu_id: int,
    checkpoint_path: str,
    calib_dir: str,
    result_dir: str,
    batch_size: int,
    num_images_in_input: int,
):
    import pickle

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"

    with open(os.path.join(result_dir, f"gpu_{gpu_id}_indices.pkl"), "rb") as f:
        indices = pickle.load(f)

    calib = load_calib_dir(Path(calib_dir))
    print(f"  [GPU {gpu_id}] Processing {len(indices)} frames")

    vla, proprio_projector = load_vla(checkpoint_path, device, num_images_in_input)
    target_params = get_target_params(vla)
    freeze_except(vla, target_params, proprio_projector)

    accumulators, losses, num_frames = _run_fisher_on_indices(
        vla, proprio_projector, target_params, calib["arrs"],
        indices, batch_size, device,
        progress_prefix=f"  [GPU {gpu_id}] ",
    )

    gpu_dir = os.path.join(result_dir, f"gpu_{gpu_id}")
    os.makedirs(gpu_dir, exist_ok=True)
    np.save(os.path.join(gpu_dir, "num_frames.npy"), np.array(num_frames))
    np.save(os.path.join(gpu_dir, "losses.npy"),     np.array(losses))
    name_map = {}
    for gguf_name, acc in accumulators.items():
        safe = gguf_name.replace(".", "_").replace("/", "_")
        name_map[safe] = gguf_name
        np.save(os.path.join(gpu_dir, f"acc_{safe}.npy"), acc.numpy())
    with open(os.path.join(gpu_dir, "name_map.json"), "w") as f:
        json.dump(name_map, f)

    del vla, proprio_projector, target_params, accumulators
    torch.cuda.empty_cache()
    return gpu_dir


def compute_fisher_multi_gpu(
    num_gpus: int,
    gpu_ids: list,
    checkpoint_path: str,
    calib_dir: Path,
    batch_size: int,
    num_samples: int,
    base_seed: int,
    output_path: str,
    num_images_in_input: int,
    episode_filter: dict = None,
):
    import pickle

    result_dir = output_path.replace(".gguf", "_gpu_results")
    os.makedirs(result_dir, exist_ok=True)

    calib = load_calib_dir(calib_dir)
    total = calib["num_frames"]

    if episode_filter:
        print(f"Stratified episode filter: {episode_filter}")
        all_indices = np.sort(stratified_frame_indices(calib["meta"], episode_filter))
        num_samples = len(all_indices)
        print(f"Selected {num_samples}/{total} calibration frames after episode filter")
    elif num_samples > 0 and num_samples < total:
        rng = np.random.RandomState(base_seed)
        all_indices = np.sort(rng.choice(total, num_samples, replace=False))
        print(f"Selected {num_samples}/{total} calibration frames (random subsample)")
    else:
        all_indices = np.arange(total)
        num_samples = total
        print(f"Using all {total} calibration frames")

    chunks = np.array_split(all_indices, num_gpus)
    for i, gid in enumerate(gpu_ids):
        with open(os.path.join(result_dir, f"gpu_{gid}_indices.pkl"), "wb") as f:
            pickle.dump(chunks[i], f)
        print(f"  GPU {gid}: {len(chunks[i])} frames assigned")

    print(f"\nLaunching {num_gpus} GPU workers...")
    t_start = time.time()
    mp.set_start_method("spawn", force=True)
    with mp.Pool(num_gpus) as pool:
        futures = [
            pool.apply_async(
                _gpu_worker,
                args=(gpu_ids[i], checkpoint_path, str(calib_dir), result_dir,
                      batch_size, num_images_in_input),
            )
            for i in range(num_gpus)
        ]
        result_paths = [f.get(timeout=36000) for f in futures]

    total_time = time.time() - t_start
    print(f"\nAll GPUs done in {total_time:.1f}s ({total_time/60:.1f}min)")

    # Merge
    print("Merging results...")
    merged = {}
    total_count = 0
    all_losses = []
    for gpu_dir in result_paths:
        total_count += int(np.load(os.path.join(gpu_dir, "num_frames.npy")))
        all_losses.extend(np.load(os.path.join(gpu_dir, "losses.npy")).tolist())
        with open(os.path.join(gpu_dir, "name_map.json")) as f:
            name_map = json.load(f)
        for safe, gguf_name in name_map.items():
            acc = np.load(os.path.join(gpu_dir, f"acc_{safe}.npy"))
            if gguf_name not in merged:
                merged[gguf_name] = acc.astype(np.float64)
            else:
                merged[gguf_name] += acc.astype(np.float64)

    fisher_diags = {}
    print(f"\n{'Tensor':<45s} {'Shape':>18s} {'Mean F_ii':>12s} {'Max F_ii':>12s} {'Std F_ii':>12s}")
    print("-" * 102)
    for gguf_name in sorted(merged.keys()):
        f_diag = (merged[gguf_name] / total_count).astype(np.float32)
        fisher_diags[gguf_name] = f_diag
        print(f"{gguf_name:<45s} {str(tuple(f_diag.shape)):>18s} "
              f"{f_diag.mean():>12.4e} {f_diag.max():>12.4e} {f_diag.std():>12.4e}")

    stats = {
        "num_samples":     total_count,
        "losses":          all_losses,
        "loss_mean":       float(np.mean(all_losses)),
        "loss_std":        float(np.std(all_losses)),
        "total_time_sec":  total_time,
        "num_gpus":        num_gpus,
    }
    print(f"\nLoss: mean={stats['loss_mean']:.4f}, std={stats['loss_std']:.4f}")
    print(f"Time: {total_time:.1f}s across {num_gpus} GPUs")

    import shutil
    shutil.rmtree(result_dir, ignore_errors=True)
    return fisher_diags, stats


# ─── GGUF Output ────────────────────────────────────────────────────────

def save_fisher_gguf(fisher_diags: dict, output_path: str, num_samples: int, calib_source: str):
    writer = GGUFWriter(output_path, arch="")
    writer.add_string("general.type", "fisher_diag")
    writer.add_array("imatrix.datasets", [calib_source])
    writer.add_uint32("imatrix.chunk_count", num_samples)
    writer.add_uint32("imatrix.chunk_size", 1)
    writer.add_uint32("fisher.num_samples", num_samples)

    for name in sorted(fisher_diags.keys()):
        f_diag = np.maximum(fisher_diags[name], 0.0).astype(np.float32)
        writer.add_tensor(f"{name}.in_sum2", f_diag)
        writer.add_tensor(f"{name}.counts",  np.array([[1.0]], dtype=np.float32))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    size_gb = os.path.getsize(output_path) / (1024 ** 3)
    print(f"Saved Fisher to {output_path} ({size_gb:.2f} GB)")


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fisher Information Diagonal for OpenVLA-OFT")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to OpenVLA-OFT checkpoint directory")
    parser.add_argument("--calib-dir", type=str, required=True,
                        help="Directory produced by get_vla_calib_data.py (contains *.npy + metadata.json)")
    parser.add_argument("--output", type=str, default="fisher_diag.gguf")
    parser.add_argument("--num-samples", type=int, default=0,
                        help="Number of frames to use (0 = all)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs (overrides --num-gpus)")
    parser.add_argument("--num-images-in-input", type=int, default=2)
    parser.add_argument("--spatial-episodes", type=int, default=-1,
                        help="Use only first N episodes from libero_spatial_no_noops (stratified by task; -1 = all)")
    parser.add_argument("--object-episodes", type=int, default=-1,
                        help="Use only first N episodes from libero_object_no_noops (stratified by task; -1 = all)")
    parser.add_argument("--goal-episodes", type=int, default=-1,
                        help="Use only first N episodes from libero_goal_no_noops (stratified by task; -1 = all)")
    parser.add_argument("--long-episodes", type=int, default=-1,
                        help="Use only first N episodes from libero_10_no_noops (stratified by task; -1 = all)")
    args = parser.parse_args()

    episode_filter = {}
    if args.spatial_episodes > 0: episode_filter["spatial"] = args.spatial_episodes
    if args.object_episodes  > 0: episode_filter["object"]  = args.object_episodes
    if args.goal_episodes    > 0: episode_filter["goal"]    = args.goal_episodes
    if args.long_episodes    > 0: episode_filter["long"]    = args.long_episodes

    if args.gpus is not None:
        gpu_ids = [int(x) for x in args.gpus.split(",")]
        args.num_gpus = len(gpu_ids)
    else:
        gpu_ids = list(range(args.num_gpus))

    calib_dir = Path(args.calib_dir).expanduser()
    print("=" * 70)
    print("Fisher Information Diagonal (full VLA forward)")
    print("=" * 70)
    print(f"  checkpoint         = {args.checkpoint}")
    print(f"  calib_dir          = {calib_dir}")
    print(f"  num_samples        = {args.num_samples if args.num_samples > 0 else 'all'}")
    print(f"  batch_size         = {args.batch_size}")
    print(f"  num_gpus           = {args.num_gpus}")
    print(f"  gpu_ids            = {gpu_ids}")
    print(f"  num_images_in_input= {args.num_images_in_input}")
    print()

    t_start = time.time()
    if args.num_gpus > 1:
        fisher_diags, stats = compute_fisher_multi_gpu(
            num_gpus=args.num_gpus, gpu_ids=gpu_ids,
            checkpoint_path=args.checkpoint, calib_dir=calib_dir,
            batch_size=args.batch_size, num_samples=args.num_samples,
            base_seed=args.base_seed, output_path=args.output,
            num_images_in_input=args.num_images_in_input,
            episode_filter=episode_filter,
        )
    else:
        fisher_diags, stats = compute_fisher_single_gpu(
            checkpoint_path=args.checkpoint, calib_dir=calib_dir,
            batch_size=args.batch_size, num_samples=args.num_samples,
            base_seed=args.base_seed, device=args.device,
            num_images_in_input=args.num_images_in_input,
            episode_filter=episode_filter,
        )
    stats["total_time_sec"] = time.time() - t_start

    print(f"\n--- Saving to GGUF ---")
    save_fisher_gguf(fisher_diags, args.output, stats["num_samples"], str(calib_dir))

    config = {
        "method":         "fisher_diagonal_full_vla",
        "checkpoint":     args.checkpoint,
        "calib_dir":      str(calib_dir),
        "num_samples":    stats["num_samples"],
        "batch_size":     args.batch_size,
        "base_seed":      args.base_seed,
        "num_gpus":       args.num_gpus,
        "loss_mean":      stats["loss_mean"],
        "loss_std":       stats["loss_std"],
        "total_time_sec": stats["total_time_sec"],
    }
    with open(args.output.replace(".gguf", "_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {args.output.replace('.gguf', '_config.json')}")
    print("\nDone!")


if __name__ == "__main__":
    main()
