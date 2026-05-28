#!/usr/bin/env python3
"""
Export Continuous OFT Action Predictions from Calibration Embeddings

Runs pre-computed calibration embeddings (.bin files) through the combined
LLM backbone + frozen action head and saves predicted continuous actions.

Output shape: (N, 8, 7) — N frames, 8-step chunk, 7-DoF actions (normalized).
Output files: <stem>_oft_actions.npy written to --output-dir.

Used to produce ground-truth Y for HSIC per-tensor sensitivity, replacing
the discrete action token IDs (_targets.npy) with real continuous actions.

Usage:
    python tools/fisher-diag/export_oft_actions.py \
        --checkpoint ~/arash/openvla-oft-checkpoints/oft_combined \
        --action-head-checkpoint ~/arash/openvla-oft-checkpoints/oft_combined/action_head--300000_checkpoint.pt \
        --calib-data \
            ~/arash/openvla-oft/calib_data/spatial_20.bin \
            ~/arash/openvla-oft/calib_data/object_20.bin \
            ~/arash/openvla-oft/calib_data/goal_20.bin \
            ~/arash/openvla-oft/calib_data/long_20.bin \
            ~/arash/openvla-oft/calib_data/long_50.bin \
        --output-dir ~/arash/openvla-oft/calib_data \
        --device cuda
"""

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root.parent / "openvla-oft"))

CALIB_MAGIC = b"OPENVLA_CALIB\0\0\0"
N_ACTION_TOKENS = 56  # 8 chunks × 7 DoF


def load_calibration_data(bin_path: str):
    with open(bin_path, "rb") as f:
        magic = f.read(16)
        assert magic == CALIB_MAGIC, f"Bad magic in {bin_path}: {magic}"
        _version = struct.unpack("<I", f.read(4))[0]
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
    print(f"  {num_frames} frames, hidden_dim={hidden_dim}, "
          f"seq_len=[{min(seq_lengths)}, {max(seq_lengths)}]")
    return embeddings


def load_llm(checkpoint_path: str, device: str):
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    print(f"Loading LLM from {checkpoint_path}...")
    vla = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    llm = vla.language_model.to(device)
    llm.eval()
    hidden_dim = llm.config.hidden_size
    del vla
    torch.cuda.empty_cache()
    print(f"  LLM loaded: hidden_dim={hidden_dim}")
    return llm, hidden_dim


def load_action_head(checkpoint_path: str, hidden_dim: int, device: str):
    from prismatic.models.action_heads import L1RegressionActionHead

    action_head = L1RegressionActionHead(input_dim=hidden_dim, hidden_dim=4096, action_dim=7)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    action_head.load_state_dict(state_dict)
    action_head = action_head.to(device)
    action_head.eval()
    for p in action_head.parameters():
        p.requires_grad_(False)
    print(f"  Action head loaded from {Path(checkpoint_path).name}")
    return action_head


def process_bin(bin_path: str, output_dir: str, llm, action_head, device: str):
    stem = Path(bin_path).stem                    # e.g. "spatial_20"
    out_path = os.path.join(output_dir, f"{stem}_oft_actions.npy")

    if os.path.exists(out_path):
        print(f"\nSkipping (already exists): {out_path}")
        return

    print(f"\n{'='*60}\nProcessing: {bin_path}")
    embeddings = load_calibration_data(bin_path)
    N = len(embeddings)
    all_actions = np.zeros((N, 8, 7), dtype=np.float32)

    with torch.no_grad():
        for i in tqdm(range(N), desc=f"  {stem}"):
            emb = embeddings[i]
            sl = emb.shape[0]

            input_embeds = torch.from_numpy(emb).unsqueeze(0).to(
                dtype=torch.bfloat16, device=device
            )
            attention_mask = torch.ones(1, sl, dtype=torch.long, device=device)

            out = llm.model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            )
            # Extract hidden states at action token positions [sl-57 : sl-1]
            # (excludes EOS at position sl-1)
            action_hidden = out.last_hidden_state[:, sl - N_ACTION_TOKENS - 1 : sl - 1, :]
            action_hidden = action_hidden.to(dtype=next(action_head.parameters()).dtype)
            # shape: (1, 56, 4096)

            a_pred = action_head.predict_action(action_hidden)  # (1, 8, 7)
            all_actions[i] = a_pred[0].float().cpu().numpy()

    np.save(out_path, all_actions)

    a_flat = all_actions.reshape(-1, 7)
    dim_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    print(f"  Saved → {out_path}  shape={all_actions.shape}")
    print(f"  {'dim':<10} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
    for d, name in enumerate(dim_names):
        v = a_flat[:, d]
        print(f"  {name:<10} {v.mean():>8.4f} {v.std():>8.4f} {v.min():>8.4f} {v.max():>8.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--action-head-checkpoint", required=True)
    parser.add_argument("--calib-data", nargs="+", required=True,
                        help="Calibration .bin files to process")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write *_oft_actions.npy files")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    llm, hidden_dim = load_llm(args.checkpoint, args.device)
    action_head = load_action_head(args.action_head_checkpoint, hidden_dim, args.device)

    for bin_path in args.calib_data:
        process_bin(bin_path, args.output_dir, llm, action_head, args.device)

    print(f"\n{'='*60}\nAll done.")


if __name__ == "__main__":
    main()
