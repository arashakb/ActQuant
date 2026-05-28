#!/usr/bin/env python3
from __future__ import annotations
"""
Pi0.5 to GGUF conversion script.
Converts lerobot/pi05 model weights to a single unified GGUF file for VLA.cpp inference.

This script exports ALL model components to a single GGUF file:
1. Vision (SigLIP) - v.*
2. Projector - mm.*
3. Action Expert - action.*
4. PaliGemma layers - pali.*
5. Embedding table - embed.*

Usage:
    python export_pi05.py -d /path/to/pi05_model -o /path/to/output

Prerequisites:
    - tokenizer.model from google/paligemma-3b-pt-224 must be in the model directory
"""

import os
import argparse
import shutil
import json
import numpy as np
import gguf
from gguf import GGMLQuantizationType
from gguf.quants import quantize
from ggml_quant import KQUANT_TYPES
import torch
from safetensors import safe_open
from safetensors.torch import load_file as torch_load_file


import os as _os
OUTPUT_PRECISION = _os.environ.get("PI05_OUTPUT_PRECISION", "fp16")  # Options: "fp16", "f32"

# Quantization type mapping
QUANT_MAP = {
    "q8":  GGMLQuantizationType.Q8_0,
    "q4":  GGMLQuantizationType.Q4_0,
    "q2k": GGMLQuantizationType.Q2_K,
    "q3k": GGMLQuantizationType.Q3_K,
    "q4k": GGMLQuantizationType.Q4_K,
    "q5k": GGMLQuantizationType.Q5_K,
    "q6k": GGMLQuantizationType.Q6_K,
}

# K-quant type strings (need ctypes path + 256-alignment padding)
KQUANT_STRS = {"q2k", "q3k", "q4k", "q5k", "q6k"}


def get_tensor_quant_type(name: str, shape: tuple,
                          quant_llm: str, quant_vision: str,
                          quant_embedding: str, quant_action: str = None) -> GGMLQuantizationType | None:
    """
    Determine quantization type for a tensor.
    Returns None if tensor should not be quantized.
    """
    # 1D tensors (norm, bias) are not quantized
    if len(shape) == 1:
        return None

    # Action Expert (independently controlled)
    if name.startswith("action."):
        if quant_action:
            # Skip precision-sensitive layers
            if any(x in name for x in ["norm", "timestep", "action_in", "action_out"]):
                return None
            return QUANT_MAP[quant_action]
        return None

    # Projector is not quantized
    if name.startswith("mm."):
        return None

    # Normalization stats are not quantized
    if name.startswith("norm."):
        return None

    # Embedding table (independently controlled)
    if name.startswith("embed."):
        if quant_embedding:
            return QUANT_MAP[quant_embedding]
        return None

    # LLM part (PaliGemma)
    if name.startswith("pali."):
        if quant_llm:
            # Skip norm layers
            if "norm" in name:
                return None
            return QUANT_MAP[quant_llm]
        return None

    # Vision part (SigLIP)
    if name.startswith("v."):
        if quant_vision:
            # Skip special layers
            if any(x in name for x in ["norm", "ln1", "ln2", "position_embd", "patch_embd", "post_ln"]):
                return None
            return QUANT_MAP[quant_vision]
        return None

    return None


def print_quant_plan(tensor_names_shapes: list, args):
    """Print quantization decision preview (dry run)."""
    stats = {"Q8_0": 0, "Q4_0": 0, "Q2_K": 0, "Q3_K": 0, "Q4_K": 0, "Q5_K": 0, "Q6_K": 0, "FP16": 0, "FP32": 0}
    errors = []

    print("\n" + "=" * 80)
    print("Quantization Plan (Dry Run)")
    print("=" * 80)
    print(f"  --quant_llm:       {args.quant_llm or 'none'}")
    print(f"  --quant_vision:    {args.quant_vision or 'none'}")
    print(f"  --quant_embedding: {args.quant_embedding or 'none'}")
    print(f"  --quant_action:    {args.quant_action or 'none'}")
    print("-" * 80)

    for name, shape in tensor_names_shapes:
        qtype = get_tensor_quant_type(
            name, shape,
            args.quant_llm, args.quant_vision, args.quant_embedding, args.quant_action
        )

        # Determine component category
        if name.startswith("embed."):
            component = "Embedding"
        elif name.startswith("pali."):
            component = "LLM"
        elif name.startswith("v."):
            component = "Vision"
        elif name.startswith("action."):
            component = "Action"
        elif name.startswith("mm."):
            component = "Projector"
        elif name.startswith("norm."):
            component = "Norm"
        else:
            component = "Other"

        # Determine output type and reason
        if qtype is not None:
            dtype_str = qtype.name
            reason = ""
            # Check if shape meets quantization requirements (last dim must be multiple of 32)
            if shape[-1] % 32 != 0:
                errors.append(f"  ERROR: {name} shape {shape} last dim not divisible by 32")
        elif len(shape) == 1:
            dtype_str = "FP32"
            reason = "(1D, skip)"
        else:
            dtype_str = "FP16"
            if name.startswith("action."):
                if args.quant_action:
                    reason = "(protected)"
                else:
                    reason = "(no quant flag)"
            elif name.startswith("mm."):
                reason = "(excluded)"
            elif name.startswith("norm."):
                reason = "(excluded)"
            else:
                reason = "(no quant flag)"

        stats[dtype_str] = stats.get(dtype_str, 0) + 1
        print(f"  [{component:9}] {name:50} {str(shape):25} -> {dtype_str} {reason}")

    # Print errors
    if errors:
        print("\n" + "-" * 80)
        print("Errors:")
        for e in errors:
            print(e)

    # Print summary
    print("\n" + "-" * 80)
    print("Summary:")
    for dtype, count in stats.items():
        if count > 0:
            print(f"  {dtype}: {count} tensors")
    print("=" * 80)


def convert_and_write_tensor(gguf_writer: gguf.GGUFWriter, name: str,
                              tensor: np.ndarray, args=None, quantizer=None) -> None:
    """Convert and write tensor with optional quantization."""
    quant_llm = getattr(args, 'quant_llm', None) if args else None
    quant_vision = getattr(args, 'quant_vision', None) if args else None
    quant_embedding = getattr(args, 'quant_embedding', None) if args else None
    quant_action = getattr(args, 'quant_action', None) if args else None

    qtype = get_tensor_quant_type(
        name, tensor.shape,
        quant_llm, quant_vision, quant_embedding, quant_action
    )

    if qtype is not None:
        data = tensor.astype(np.float32)
        if qtype in KQUANT_TYPES:
            # K-quants: use ctypes quantizer (supports Q2_K–Q6_K).
            # For vision weights, optionally guide with the per-element Fisher
            # imatrix (aligned to the final, padded tensor shape here so we
            # don't have to mirror export's padding in the Fisher writer).
            imat = None
            vim = getattr(args, "_vision_imatrix", None) if args else None
            if (vim is not None and name.startswith("v.blk.")
                    and name.endswith(".weight") and data.ndim == 2 and name in vim):
                M = vim[name]
                R, C = data.shape
                if M.shape == (C, R):
                    M = M.T
                aligned = np.full((R, C), 0.0, dtype=np.float32)
                rr, cc = min(R, M.shape[0]), min(C, M.shape[1])
                aligned[:rr, :cc] = M[:rr, :cc]
                # floor non-positive entries (padded region + zeros) so ggml
                # k-quant never sees a degenerate all-zero importance row.
                pos = aligned[aligned > 0]
                floor = float(pos.mean() * 1e-6) if pos.size else 1e-20
                np.maximum(aligned, floor, out=aligned)
                imat = np.ascontiguousarray(aligned, dtype=np.float32)
            quantized_bytes = quantizer.quantize(data, qtype, imatrix=imat)
            n_per_row = tensor.shape[-1]
            row_size = len(quantized_bytes) // (int(np.prod(tensor.shape[:-1])) if tensor.ndim > 1 else 1)
            if tensor.ndim == 1:
                byte_shape = (len(quantized_bytes),)
            else:
                nrows = int(np.prod(tensor.shape[:-1]))
                byte_shape = (nrows, row_size)
            quantized_data = np.frombuffer(quantized_bytes, dtype=np.uint8).reshape(byte_shape)
            gguf_writer.add_tensor(name, quantized_data, raw_dtype=qtype)
        else:
            # Q8_0/Q4_0: use gguf.quants.quantize
            data = quantize(data, qtype)
            gguf_writer.add_tensor(name, data, raw_dtype=qtype)
    else:
        # No quantization, use original logic
        is_1d = len(tensor.shape) == 1
        if is_1d:
            data = tensor.astype(np.float32)
        else:
            data = tensor.astype(np.float16 if OUTPUT_PRECISION == "fp16" else np.float32)
        gguf_writer.add_tensor(name, data)


def add_config(gguf_writer: gguf.GGUFWriter, model_cfg: dict):
    """Add configuration key-value pairs to GGUF."""
    for k, v in model_cfg.items():
        if isinstance(v, bool):
            gguf_writer.add_bool(k, v)
        elif isinstance(v, float):
            gguf_writer.add_float32(k, v)
        elif isinstance(v, int):
            gguf_writer.add_uint32(k, v)
        elif isinstance(v, str):
            gguf_writer.add_string(k, v)
        elif isinstance(v, list):
            gguf_writer.add_array(k, v)
        else:
            raise ValueError(f"Unsupported type: {type(v)}")


def get_tensor(st, key: str, prefix: str = "") -> np.ndarray:
    """Get tensor from safetensors file or state dict.

    Args:
        st: safetensors file handle or PyTorch state dict
        key: tensor name (without model prefix)
        prefix: optional prefix (e.g., "model.") for new model format
    """
    full_key = prefix + key
    if isinstance(st, dict):
        # PyTorch state dict (for bfloat16 support)
        tensor = st[full_key]
        return tensor.float().numpy()
    else:
        # safetensors file handle
        return st.get_tensor(full_key)


def detect_tensor_prefix(st) -> str:
    """Detect if tensor names have 'model.' prefix (new LeRobot format)."""
    if isinstance(st, dict):
        keys = list(st.keys())
    else:
        keys = list(st.keys())

    # Check if keys start with 'model.'
    sample_key = keys[0] if keys else ""
    if sample_key.startswith("model."):
        return "model."
    return ""


def convert_tensor(tensor: np.ndarray, is_bias_or_norm: bool = False) -> np.ndarray:
    """Convert tensor to target precision."""
    if is_bias_or_norm:
        return tensor.astype(np.float32)
    elif OUTPUT_PRECISION == "fp16":
        return tensor.astype(np.float16)
    else:
        return tensor.astype(np.float32)


def load_norm_stats(model_dir: str) -> dict:
    """Load normalization statistics from preprocessor/postprocessor safetensors files or assets JSON."""
    norm_stats = {}

    # Try to load from policy_preprocessor (for state normalization) - LeRobot format
    pre_path = os.path.join(model_dir, "policy_preprocessor_step_2_normalizer_processor.safetensors")
    if os.path.exists(pre_path):
        pre_tensors = torch_load_file(pre_path)
        # Extract state normalization stats
        if "observation.state.mean" in pre_tensors:
            norm_stats["state_mean"] = pre_tensors["observation.state.mean"].float().numpy()
            norm_stats["state_std"] = pre_tensors["observation.state.std"].float().numpy()
            print(f"  Loaded state norm stats: dim={len(norm_stats['state_mean'])}")
        # Extract action normalization stats (for training reference)
        if "action.mean" in pre_tensors:
            norm_stats["action_mean"] = pre_tensors["action.mean"].float().numpy()
            norm_stats["action_std"] = pre_tensors["action.std"].float().numpy()
            print(f"  Loaded action norm stats: dim={len(norm_stats['action_mean'])}")

    # Try to load from policy_postprocessor (for action unnormalization) - LeRobot format
    post_path = os.path.join(model_dir, "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
    if os.path.exists(post_path) and "action_mean" not in norm_stats:
        post_tensors = torch_load_file(post_path)
        if "action.mean" in post_tensors:
            norm_stats["action_mean"] = post_tensors["action.mean"].float().numpy()
            norm_stats["action_std"] = post_tensors["action.std"].float().numpy()
            print(f"  Loaded action norm stats from postprocessor: dim={len(norm_stats['action_mean'])}")

    # Try to load from assets directory (JAX/OpenPI format)
    # Path: assets/physical-intelligence/<task>/norm_stats.json
    if not norm_stats:
        assets_dir = os.path.join(model_dir, "assets")
        if os.path.exists(assets_dir):
            # Search for norm_stats.json in assets subdirectories
            for root, dirs, files in os.walk(assets_dir):
                if "norm_stats.json" in files:
                    json_path = os.path.join(root, "norm_stats.json")
                    print(f"  Found norm_stats.json at: {json_path}")
                    with open(json_path, "r") as f:
                        stats_json = json.load(f)

                    # Handle nested "norm_stats" wrapper (OpenPI format)
                    if "norm_stats" in stats_json:
                        stats_json = stats_json["norm_stats"]

                    # Extract action stats (try both "actions" and "action")
                    action_key = "actions" if "actions" in stats_json else "action"
                    if action_key in stats_json:
                        action_stats = stats_json[action_key]
                        # Load mean/std (for compatibility)
                        if "mean" in action_stats and "std" in action_stats:
                            norm_stats["action_mean"] = np.array(action_stats["mean"], dtype=np.float32)
                            norm_stats["action_std"] = np.array(action_stats["std"], dtype=np.float32)
                            print(f"  Loaded action mean/std from JSON: dim={len(norm_stats['action_mean'])}")
                        # Load q01/q99 (for OpenPI Pi0.5 quantile normalization)
                        if "q01" in action_stats and "q99" in action_stats:
                            norm_stats["action_q01"] = np.array(action_stats["q01"], dtype=np.float32)
                            norm_stats["action_q99"] = np.array(action_stats["q99"], dtype=np.float32)
                            print(f"  Loaded action q01/q99 from JSON: dim={len(norm_stats['action_q01'])}")

                    # Extract state stats if present
                    if "state" in stats_json:
                        state_stats = stats_json["state"]
                        # Load mean/std
                        if "mean" in state_stats and "std" in state_stats:
                            norm_stats["state_mean"] = np.array(state_stats["mean"], dtype=np.float32)
                            norm_stats["state_std"] = np.array(state_stats["std"], dtype=np.float32)
                            print(f"  Loaded state mean/std from JSON: dim={len(norm_stats['state_mean'])}")
                        # Load q01/q99 (for OpenPI Pi0.5 quantile normalization)
                        if "q01" in state_stats and "q99" in state_stats:
                            norm_stats["state_q01"] = np.array(state_stats["q01"], dtype=np.float32)
                            norm_stats["state_q99"] = np.array(state_stats["q99"], dtype=np.float32)
                            print(f"  Loaded state q01/q99 from JSON: dim={len(norm_stats['state_q01'])}")
                    break

    return norm_stats


def get_output_filename(args) -> str:
    """Generate output filename (fixed name for consistency)."""
    return "pi05.gguf"


def export_unified(st, output_dir: str, model_dir: str, model_config: dict = None, args=None, quantizer=None):
    """Export all model components to a single unified GGUF file."""
    print("\n" + "=" * 60)
    print("Exporting Pi0.5 model to unified GGUF file...")
    print("=" * 60)

    # Detect tensor prefix (new LeRobot format uses 'model.' prefix)
    tensor_prefix = detect_tensor_prefix(st)
    if tensor_prefix:
        print(f"  Detected tensor prefix: '{tensor_prefix}'")

    # Helper function to get tensor with prefix
    def get(key: str) -> np.ndarray:
        return get_tensor(st, key, tensor_prefix)

    # Read action_horizon from config (action_horizon, chunk_size, or n_action_steps)
    action_horizon = 50  # Default for base model
    if model_config:
        # JAX format uses "action_horizon", LeRobot uses "chunk_size" or "n_action_steps"
        action_horizon = model_config.get("action_horizon",
                            model_config.get("chunk_size",
                                model_config.get("n_action_steps", 50)))
    print(f"  action_horizon: {action_horizon}")

    # Load normalization statistics
    print("\n  Loading normalization statistics...")
    norm_stats = load_norm_stats(model_dir)

    # Generate output filename based on quantization config
    output_filename = get_output_filename(args) if args else "pi05.gguf"
    gguf_path = os.path.join(output_dir, output_filename)
    gguf_writer = gguf.GGUFWriter(gguf_path, "pi05")

    # ========================================================================
    # Configuration
    # ========================================================================
    # Determine actual action/state dimensions from norm stats
    actual_action_dim = len(norm_stats.get("action_mean", [])) or 7  # Default LIBERO
    actual_state_dim = len(norm_stats.get("state_mean", [])) or 8   # Default LIBERO

    cfg = {
        # Model type
        "pi05.unified": True,

        # Vision (SigLIP) configuration
        "clip.projector_type": "pi0",
        "clip.has_vision_encoder": True,
        "clip.vision.embedding_length": 1152,
        "clip.vision.attention.head_count": 16,
        "clip.vision.feed_forward_length": 4304,
        "clip.vision.block_count": 27,
        "clip.vision.projection_dim": 2048,
        "clip.vision.attention.layer_norm_epsilon": 1e-6,
        "clip.vision.image_size": 224,
        "clip.vision.patch_size": 14,
        "clip.use_gelu": True,
        "clip.vision.image_mean": [0.5, 0.5, 0.5],
        "clip.vision.image_std": [0.5, 0.5, 0.5],

        # Projector config
        "projector.input_dim": 1152,
        "projector.output_dim": 2048,

        # Action Expert configuration
        "pi05_action.hidden_size": 1024,
        "pi05_action.intermediate_size": 4096,
        "pi05_action.block_count": 18,
        "pi05_action.attention.head_count": 8,
        "pi05_action.attention.head_count_kv": 1,
        "pi05_action.attention.head_dim": 256,
        "pi05_action.action_dim": 32,
        "pi05_action.state_dim": 32,
        "pi05_action.actual_action_dim": actual_action_dim,
        "pi05_action.actual_state_dim": actual_state_dim,
        "pi05_action.action_horizon": action_horizon,
        "pi05_action.attention.layer_norm_rms_epsilon": 1e-6,
        "pi05_action.timestep_sinusoidal_dim": 1024,
        "pi05_action.use_adaln": True,

        # PaliGemma config for cross-attention
        "pi05_action.pali_hidden_size": 2048,
        "pi05_action.pali_intermediate_size": 16384,

        # Embedding config
        "embed.vocab_size": 257152,
        "embed.hidden_size": 2048,
    }
    add_config(gguf_writer, cfg)

    # Add quantization metadata
    if args:
        gguf_writer.add_string("pi05.quant_llm", args.quant_llm or "none")
        gguf_writer.add_string("pi05.quant_vision", args.quant_vision or "none")
        gguf_writer.add_string("pi05.quant_embedding", args.quant_embedding or "none")

    # ========================================================================
    # 1. Vision Encoder (SigLIP)
    # ========================================================================
    print("\n[1/6] Exporting Vision Encoder (SigLIP)...")

    vision_prefix = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"

    # Embeddings
    pos_emb = get(f"{vision_prefix}.embeddings.position_embedding.weight")
    patch_emb = get(f"{vision_prefix}.embeddings.patch_embedding.weight")
    patch_bias = get(f"{vision_prefix}.embeddings.patch_embedding.bias")

    # Reshape patch embedding: (hidden, C, H, W) format expected
    patch_emb = patch_emb.reshape(1152, 3, 14, 14)

    convert_and_write_tensor(gguf_writer, "v.position_embd.weight", pos_emb, args, quantizer)
    convert_and_write_tensor(gguf_writer, "v.patch_embd.weight", patch_emb, args, quantizer)
    convert_and_write_tensor(gguf_writer, "v.patch_embd.bias", patch_bias, args, quantizer)

    # Post layer norm
    post_ln_w = get(f"{vision_prefix}.post_layernorm.weight")
    post_ln_b = get(f"{vision_prefix}.post_layernorm.bias")
    convert_and_write_tensor(gguf_writer, "v.post_ln.weight", post_ln_w, args, quantizer)
    convert_and_write_tensor(gguf_writer, "v.post_ln.bias", post_ln_b, args, quantizer)

    # Encoder layers
    n_vision_layers = 27
    for i in range(n_vision_layers):
        layer_prefix = f"{vision_prefix}.encoder.layers.{i}"

        # Attention
        q_w = get(f"{layer_prefix}.self_attn.q_proj.weight")
        q_b = get(f"{layer_prefix}.self_attn.q_proj.bias")
        k_w = get(f"{layer_prefix}.self_attn.k_proj.weight")
        k_b = get(f"{layer_prefix}.self_attn.k_proj.bias")
        v_w = get(f"{layer_prefix}.self_attn.v_proj.weight")
        v_b = get(f"{layer_prefix}.self_attn.v_proj.bias")
        out_w = get(f"{layer_prefix}.self_attn.out_proj.weight")
        out_b = get(f"{layer_prefix}.self_attn.out_proj.bias")

        # K-quants need 256-alignment: pad attn weights input dim 1152 -> 1280 (+128)
        if args.quant_vision in KQUANT_STRS:
            q_w = np.pad(q_w, ((0, 0), (0, 128)), mode='constant')
            k_w = np.pad(k_w, ((0, 0), (0, 128)), mode='constant')
            v_w = np.pad(v_w, ((0, 0), (0, 128)), mode='constant')
            out_w = np.pad(out_w, ((0, 0), (0, 128)), mode='constant')

        convert_and_write_tensor(gguf_writer, f"v.blk.{i}.attn_q.weight", q_w, args, quantizer)
        convert_and_write_tensor(gguf_writer, f"v.blk.{i}.attn_q.bias", q_b, args, quantizer)
        convert_and_write_tensor(gguf_writer, f"v.blk.{i}.attn_k.weight", k_w, args, quantizer)
        convert_and_write_tensor(gguf_writer, f"v.blk.{i}.attn_k.bias", k_b, args, quantizer)
        convert_and_write_tensor(gguf_writer, f"v.blk.{i}.attn_v.weight", v_w, args, quantizer)
        convert_and_write_tensor(gguf_writer, f"v.blk.{i}.attn_v.bias", v_b, args, quantizer)
        convert_and_write_tensor(gguf_writer, f"v.blk.{i}.attn_out.weight", out_w, args, quantizer)
        convert_and_write_tensor(gguf_writer, f"v.blk.{i}.attn_out.bias", out_b, args, quantizer)

        # Layer norms
        ln1_w = get(f"{layer_prefix}.layer_norm1.weight")
        ln1_b = get(f"{layer_prefix}.layer_norm1.bias")
        ln2_w = get(f"{layer_prefix}.layer_norm2.weight")
        ln2_b = get(f"{layer_prefix}.layer_norm2.bias")

        convert_and_write_tensor(gguf_writer, f"v.blk.{i}.ln1.weight", ln1_w, args, quantizer)
        convert_and_write_tensor(gguf_writer, f"v.blk.{i}.ln1.bias", ln1_b, args, quantizer)
        convert_and_write_tensor(gguf_writer, f"v.blk.{i}.ln2.weight", ln2_w, args, quantizer)
        convert_and_write_tensor(gguf_writer, f"v.blk.{i}.ln2.bias", ln2_b, args, quantizer)

        # FFN
        fc1_w = get(f"{layer_prefix}.mlp.fc1.weight")
        fc1_b = get(f"{layer_prefix}.mlp.fc1.bias")
        fc2_w = get(f"{layer_prefix}.mlp.fc2.weight")
        fc2_b = get(f"{layer_prefix}.mlp.fc2.bias")

        # Pad ffn_down: (1152, 4304) -> (1152, 4352) for all quant types (+48)
        fc2_w = np.pad(fc2_w, ((0, 0), (0, 48)), mode='constant')

        # Pad ffn_up: rows +48 (4304->4352), cols +128 for K-quants (256-align 1152->1280)
        if args.quant_vision in KQUANT_STRS:
            fc1_w = np.pad(fc1_w, ((0, 48), (0, 128)), mode='constant')
        else:
            fc1_w = np.pad(fc1_w, ((0, 48), (0, 0)), mode='constant')
        fc1_b = np.pad(fc1_b, (0, 48), mode='constant')

        convert_and_write_tensor(gguf_writer, f"v.blk.{i}.ffn_up.weight", fc1_w, args, quantizer)
        convert_and_write_tensor(gguf_writer, f"v.blk.{i}.ffn_up.bias", fc1_b, args, quantizer)
        convert_and_write_tensor(gguf_writer, f"v.blk.{i}.ffn_down.weight", fc2_w, args, quantizer)
        convert_and_write_tensor(gguf_writer, f"v.blk.{i}.ffn_down.bias", fc2_b, args, quantizer)

    print(f"  Exported {n_vision_layers} vision layers")

    # ========================================================================
    # 2. Projector
    # ========================================================================
    print("\n[2/6] Exporting Projector...")

    proj_prefix = "paligemma_with_expert.paligemma.model.multi_modal_projector"
    linear_w = get(f"{proj_prefix}.linear.weight")
    linear_b = get(f"{proj_prefix}.linear.bias")

    convert_and_write_tensor(gguf_writer, "mm.0.weight", linear_w, args, quantizer)
    convert_and_write_tensor(gguf_writer, "mm.0.bias", linear_b, args, quantizer)

    print("  Exported projector")

    # ========================================================================
    # 3. Embedding Table
    # ========================================================================
    print("\n[3/6] Exporting Embedding Table...")

    embed_weight = get("paligemma_with_expert.paligemma.lm_head.weight")
    print(f"  Embedding shape: {embed_weight.shape}")

    convert_and_write_tensor(gguf_writer, "embed.weight", embed_weight, args, quantizer)

    print(f"  Exported embedding table: vocab_size={embed_weight.shape[0]}, hidden_size={embed_weight.shape[1]}")

    # ========================================================================
    # 4. Action Expert
    # ========================================================================
    print("\n[4/6] Exporting Action Expert...")

    # Flow matching components
    action_in_w = get("action_in_proj.weight")
    action_in_b = get("action_in_proj.bias")
    action_out_w = get("action_out_proj.weight")
    action_out_b = get("action_out_proj.bias")

    convert_and_write_tensor(gguf_writer, "action.action_in.weight", action_in_w, args, quantizer)
    convert_and_write_tensor(gguf_writer, "action.action_in.bias", action_in_b, args, quantizer)
    convert_and_write_tensor(gguf_writer, "action.action_out.weight", action_out_w, args, quantizer)
    convert_and_write_tensor(gguf_writer, "action.action_out.bias", action_out_b, args, quantizer)

    # Timestep MLP
    time_in_w = get("time_mlp_in.weight")
    time_in_b = get("time_mlp_in.bias")
    time_out_w = get("time_mlp_out.weight")
    time_out_b = get("time_mlp_out.bias")

    convert_and_write_tensor(gguf_writer, "action.timestep_mlp.in.weight", time_in_w, args, quantizer)
    convert_and_write_tensor(gguf_writer, "action.timestep_mlp.in.bias", time_in_b, args, quantizer)
    convert_and_write_tensor(gguf_writer, "action.timestep_mlp.out.weight", time_out_w, args, quantizer)
    convert_and_write_tensor(gguf_writer, "action.timestep_mlp.out.bias", time_out_b, args, quantizer)

    # Gemma expert transformer layers
    expert_prefix = "paligemma_with_expert.gemma_expert.model"
    n_layers = 18

    # Output norm (AdaLN)
    norm_w = get(f"{expert_prefix}.norm.dense.weight")
    norm_b = get(f"{expert_prefix}.norm.dense.bias")
    convert_and_write_tensor(gguf_writer, "action.output_norm.weight", norm_w, args, quantizer)
    convert_and_write_tensor(gguf_writer, "action.output_norm.bias", norm_b, args, quantizer)

    for i in range(n_layers):
        layer_prefix = f"{expert_prefix}.layers.{i}"

        # Attention
        q_w = get(f"{layer_prefix}.self_attn.q_proj.weight")
        k_w = get(f"{layer_prefix}.self_attn.k_proj.weight")
        v_w = get(f"{layer_prefix}.self_attn.v_proj.weight")
        o_w = get(f"{layer_prefix}.self_attn.o_proj.weight")

        convert_and_write_tensor(gguf_writer, f"action.blk.{i}.attn_q.weight", q_w, args, quantizer)
        convert_and_write_tensor(gguf_writer, f"action.blk.{i}.attn_k.weight", k_w, args, quantizer)
        convert_and_write_tensor(gguf_writer, f"action.blk.{i}.attn_v.weight", v_w, args, quantizer)
        convert_and_write_tensor(gguf_writer, f"action.blk.{i}.attn_output.weight", o_w, args, quantizer)

        # Adaptive LayerNorm
        attn_norm_w = get(f"{layer_prefix}.input_layernorm.dense.weight")
        attn_norm_b = get(f"{layer_prefix}.input_layernorm.dense.bias")
        ffn_norm_w = get(f"{layer_prefix}.post_attention_layernorm.dense.weight")
        ffn_norm_b = get(f"{layer_prefix}.post_attention_layernorm.dense.bias")

        convert_and_write_tensor(gguf_writer, f"action.blk.{i}.attn_norm.weight", attn_norm_w, args, quantizer)
        convert_and_write_tensor(gguf_writer, f"action.blk.{i}.attn_norm.bias", attn_norm_b, args, quantizer)
        convert_and_write_tensor(gguf_writer, f"action.blk.{i}.ffn_norm.weight", ffn_norm_w, args, quantizer)
        convert_and_write_tensor(gguf_writer, f"action.blk.{i}.ffn_norm.bias", ffn_norm_b, args, quantizer)

        # FFN
        gate_w = get(f"{layer_prefix}.mlp.gate_proj.weight")
        up_w = get(f"{layer_prefix}.mlp.up_proj.weight")
        down_w = get(f"{layer_prefix}.mlp.down_proj.weight")

        convert_and_write_tensor(gguf_writer, f"action.blk.{i}.ffn_gate.weight", gate_w, args, quantizer)
        convert_and_write_tensor(gguf_writer, f"action.blk.{i}.ffn_up.weight", up_w, args, quantizer)
        convert_and_write_tensor(gguf_writer, f"action.blk.{i}.ffn_down.weight", down_w, args, quantizer)

    print(f"  Exported {n_layers} action expert layers")

    # ========================================================================
    # 5. PaliGemma Layers (for cross-attention)
    # ========================================================================
    print("\n[5/6] Exporting PaliGemma Layers...")

    pali_prefix = "paligemma_with_expert.paligemma.model.language_model"

    for i in range(n_layers):
        pali_layer_prefix = f"{pali_prefix}.layers.{i}"

        # Input LayerNorm (RMSNorm weight)
        input_norm_w = get(f"{pali_layer_prefix}.input_layernorm.weight")
        convert_and_write_tensor(gguf_writer, f"pali.blk.{i}.attn_norm.weight", input_norm_w, args, quantizer)

        # Full attention projections Q/K/V/O
        q_w = get(f"{pali_layer_prefix}.self_attn.q_proj.weight")
        k_w = get(f"{pali_layer_prefix}.self_attn.k_proj.weight")
        v_w = get(f"{pali_layer_prefix}.self_attn.v_proj.weight")
        o_w = get(f"{pali_layer_prefix}.self_attn.o_proj.weight")
        convert_and_write_tensor(gguf_writer, f"pali.blk.{i}.attn_q.weight", q_w, args, quantizer)
        convert_and_write_tensor(gguf_writer, f"pali.blk.{i}.attn_k.weight", k_w, args, quantizer)
        convert_and_write_tensor(gguf_writer, f"pali.blk.{i}.attn_v.weight", v_w, args, quantizer)
        convert_and_write_tensor(gguf_writer, f"pali.blk.{i}.attn_o.weight", o_w, args, quantizer)

        # Post attention LayerNorm (RMSNorm weight)
        post_norm_w = get(f"{pali_layer_prefix}.post_attention_layernorm.weight")
        convert_and_write_tensor(gguf_writer, f"pali.blk.{i}.ffn_norm.weight", post_norm_w, args, quantizer)

        # MLP
        gate_w = get(f"{pali_layer_prefix}.mlp.gate_proj.weight")
        up_w = get(f"{pali_layer_prefix}.mlp.up_proj.weight")
        down_w = get(f"{pali_layer_prefix}.mlp.down_proj.weight")
        convert_and_write_tensor(gguf_writer, f"pali.blk.{i}.ffn_gate.weight", gate_w, args, quantizer)
        convert_and_write_tensor(gguf_writer, f"pali.blk.{i}.ffn_up.weight", up_w, args, quantizer)
        convert_and_write_tensor(gguf_writer, f"pali.blk.{i}.ffn_down.weight", down_w, args, quantizer)

    print(f"  Exported {n_layers} PaliGemma layers")

    # ========================================================================
    # 6. Normalization Statistics (for action/state unnormalization)
    # ========================================================================
    print("\n[6/6] Exporting Normalization Statistics...")

    # Export action norm stats to GGUF
    if "action_mean" in norm_stats:
        gguf_writer.add_tensor("norm.action_mean", norm_stats["action_mean"].astype(np.float32))
        gguf_writer.add_tensor("norm.action_std", norm_stats["action_std"].astype(np.float32))
        print(f"  Exported action mean/std: dim={len(norm_stats['action_mean'])}")

    if "action_q01" in norm_stats:
        gguf_writer.add_tensor("norm.action_q01", norm_stats["action_q01"].astype(np.float32))
        gguf_writer.add_tensor("norm.action_q99", norm_stats["action_q99"].astype(np.float32))
        print(f"  Exported action q01/q99: dim={len(norm_stats['action_q01'])}")

    if "action_mean" not in norm_stats and "action_q01" not in norm_stats:
        print("  WARNING: No action norm stats found, unnormalization will use defaults")

    # Export state norm stats to GGUF
    if "state_mean" in norm_stats:
        gguf_writer.add_tensor("norm.state_mean", norm_stats["state_mean"].astype(np.float32))
        gguf_writer.add_tensor("norm.state_std", norm_stats["state_std"].astype(np.float32))
        print(f"  Exported state mean/std: dim={len(norm_stats['state_mean'])}")

    if "state_q01" in norm_stats:
        gguf_writer.add_tensor("norm.state_q01", norm_stats["state_q01"].astype(np.float32))
        gguf_writer.add_tensor("norm.state_q99", norm_stats["state_q99"].astype(np.float32))
        print(f"  Exported state q01/q99: dim={len(norm_stats['state_q01'])}")

    if "state_mean" not in norm_stats and "state_q01" not in norm_stats:
        print("  WARNING: No state norm stats found, normalization will use defaults")

    # Save norm stats as JSON for serve_policy.py to read
    norm_stats_json_path = os.path.join(output_dir, "norm_stats.json")
    norm_stats_json = {}

    # Action stats (mean/std for LeRobot, q01/q99 for OpenPI)
    if "action_mean" in norm_stats:
        norm_stats_json["action_mean"] = norm_stats["action_mean"].tolist()
        norm_stats_json["action_std"] = norm_stats["action_std"].tolist()
    if "action_q01" in norm_stats:
        norm_stats_json["action_q01"] = norm_stats["action_q01"].tolist()
        norm_stats_json["action_q99"] = norm_stats["action_q99"].tolist()

    # State stats
    if "state_mean" in norm_stats:
        norm_stats_json["state_mean"] = norm_stats["state_mean"].tolist()
        norm_stats_json["state_std"] = norm_stats["state_std"].tolist()
    if "state_q01" in norm_stats:
        norm_stats_json["state_q01"] = norm_stats["state_q01"].tolist()
        norm_stats_json["state_q99"] = norm_stats["state_q99"].tolist()

    if norm_stats_json:
        with open(norm_stats_json_path, "w") as f:
            json.dump(norm_stats_json, f, indent=2)
        print(f"  Saved norm_stats.json for inference")

    # ========================================================================
    # Write GGUF file
    # ========================================================================
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()

    # Get file size
    file_size = os.path.getsize(gguf_path)
    print(f"\n  Written to {gguf_path}")
    print(f"  File size: {file_size / (1024**3):.2f} GB")

    # Copy tokenizer
    print("\n  Copying tokenizer files...")
    tokenizer_src = os.path.join(model_dir, "tokenizer.model")
    tokenizer_dst = os.path.join(output_dir, "tokenizer.model")
    if os.path.exists(tokenizer_src):
        shutil.copy(tokenizer_src, tokenizer_dst)
        print(f"  Copied tokenizer.model")
    else:
        print(f"  WARNING: tokenizer.model not found at {tokenizer_src}")

    return gguf_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Pi0.5 model to unified GGUF format")
    parser.add_argument("-d", "--dir-model", required=True, help="Path to pi05 model directory")
    parser.add_argument("-o", "--output-dir", default=None, help="Output directory (default: same as model dir)")
    _quant_choices = ["q8", "q4", "q2k", "q3k", "q4k", "q5k", "q6k"]
    parser.add_argument("--quant_llm", choices=_quant_choices, default=None,
                        help="Quantize LLM (PaliGemma) weights: q8/q4 or q2k–q6k (K-quants)")
    parser.add_argument("--quant_vision", choices=_quant_choices, default=None,
                        help="Quantize Vision (SigLIP) weights: q8/q4 or q2k–q6k (K-quants)")
    parser.add_argument("--quant_embedding", choices=_quant_choices, default=None,
                        help="Quantize embedding table (conservative, off by default)")
    parser.add_argument("--quant_action", choices=_quant_choices, default=None,
                        help="Quantize Action Expert weights (action.blk.* QKV and MLP)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Preview quantization decisions without actual export")
    parser.add_argument("--vision-imatrix", default=None,
                        help="Per-element Fisher imatrix GGUF for the vision tower "
                             "(from compute_fisher_pi05.py --vision-output). When set, "
                             "K-quant vision weights are quantized with per-weight "
                             "importance instead of the blind round.")
    args = parser.parse_args()

    model_dir = args.dir_model
    output_dir = args.output_dir or model_dir

    os.makedirs(output_dir, exist_ok=True)

    # Check for tokenizer files
    tokenizer_model = os.path.join(model_dir, "tokenizer.model")
    if not os.path.exists(tokenizer_model):
        print("WARNING: tokenizer.model not found!")
        print("Please download from google/paligemma-3b-pt-224:")
        print("  huggingface-cli download google/paligemma-3b-pt-224 tokenizer.model \\")
        print(f"    --local-dir {model_dir}")
        print()

    # Load config.json
    config_path = os.path.join(model_dir, "config.json")
    model_config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            model_config = json.load(f)
        print(f"Loaded config from {config_path}")
        print(f"  chunk_size/n_action_steps: {model_config.get('chunk_size', model_config.get('n_action_steps', 'not found'))}")
    else:
        print(f"WARNING: config.json not found, using defaults")

    # Load safetensors - prefer PyTorch loader for bfloat16 support
    model_path = os.path.join(model_dir, "model.safetensors")
    print(f"Loading model from {model_path}...")

    # Try PyTorch loader first (handles bfloat16 automatically)
    try:
        st = torch_load_file(model_path)
        print("  Loaded with PyTorch loader (supports bfloat16)")
    except Exception as e:
        print(f"  PyTorch loader failed: {e}")
        print("  Trying safetensors numpy loader...")
        st = safe_open(model_path, framework="numpy")

    # Dry run mode: preview quantization decisions without export
    if args.dry_run:
        # Collect all tensor names and shapes for preview
        tensor_names_shapes = []

        # Get tensor prefix
        tensor_prefix = detect_tensor_prefix(st)

        def collect_tensor_info(gguf_name: str, src_key: str, pad_rows: int = 0, pad_cols: int = 0):
            full_key = tensor_prefix + src_key
            if isinstance(st, dict):
                shape = list(st[full_key].shape)
            else:
                shape = list(st.get_tensor(full_key).shape)
            if len(shape) == 2:
                shape[0] += pad_rows
                shape[1] += pad_cols
            elif len(shape) == 1 and pad_rows > 0:
                shape[0] += pad_rows
            tensor_names_shapes.append((gguf_name, tuple(shape)))

        # Collect all tensors (same order as export_unified)
        vision_prefix = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
        collect_tensor_info("v.position_embd.weight", f"{vision_prefix}.embeddings.position_embedding.weight")
        collect_tensor_info("v.patch_embd.weight", f"{vision_prefix}.embeddings.patch_embedding.weight")
        collect_tensor_info("v.patch_embd.bias", f"{vision_prefix}.embeddings.patch_embedding.bias")
        collect_tensor_info("v.post_ln.weight", f"{vision_prefix}.post_layernorm.weight")
        collect_tensor_info("v.post_ln.bias", f"{vision_prefix}.post_layernorm.bias")

        # Vision layer padding: K-quants need 256-alignment for dim 1152 -> 1280 (+128)
        use_kquant_vision = args.quant_vision in KQUANT_STRS
        attn_col_pad = 128 if use_kquant_vision else 0
        ffn_up_col_pad = 128 if use_kquant_vision else 0

        for i in range(27):
            layer_prefix = f"{vision_prefix}.encoder.layers.{i}"
            # Attn weights: Q4_K pads input dim +128
            collect_tensor_info(f"v.blk.{i}.attn_q.weight", f"{layer_prefix}.self_attn.q_proj.weight", 0, attn_col_pad)
            collect_tensor_info(f"v.blk.{i}.attn_q.bias", f"{layer_prefix}.self_attn.q_proj.bias")
            collect_tensor_info(f"v.blk.{i}.attn_k.weight", f"{layer_prefix}.self_attn.k_proj.weight", 0, attn_col_pad)
            collect_tensor_info(f"v.blk.{i}.attn_k.bias", f"{layer_prefix}.self_attn.k_proj.bias")
            collect_tensor_info(f"v.blk.{i}.attn_v.weight", f"{layer_prefix}.self_attn.v_proj.weight", 0, attn_col_pad)
            collect_tensor_info(f"v.blk.{i}.attn_v.bias", f"{layer_prefix}.self_attn.v_proj.bias")
            collect_tensor_info(f"v.blk.{i}.attn_out.weight", f"{layer_prefix}.self_attn.out_proj.weight", 0, attn_col_pad)
            collect_tensor_info(f"v.blk.{i}.attn_out.bias", f"{layer_prefix}.self_attn.out_proj.bias")
            collect_tensor_info(f"v.blk.{i}.ln1.weight", f"{layer_prefix}.layer_norm1.weight")
            collect_tensor_info(f"v.blk.{i}.ln1.bias", f"{layer_prefix}.layer_norm1.bias")
            collect_tensor_info(f"v.blk.{i}.ln2.weight", f"{layer_prefix}.layer_norm2.weight")
            collect_tensor_info(f"v.blk.{i}.ln2.bias", f"{layer_prefix}.layer_norm2.bias")
            # ffn_up: rows +48, cols +128 for Q4_K
            collect_tensor_info(f"v.blk.{i}.ffn_up.weight", f"{layer_prefix}.mlp.fc1.weight", 48, ffn_up_col_pad)
            collect_tensor_info(f"v.blk.{i}.ffn_up.bias", f"{layer_prefix}.mlp.fc1.bias", 48)
            # ffn_down: cols +48 (unified)
            collect_tensor_info(f"v.blk.{i}.ffn_down.weight", f"{layer_prefix}.mlp.fc2.weight", 0, 48)
            collect_tensor_info(f"v.blk.{i}.ffn_down.bias", f"{layer_prefix}.mlp.fc2.bias")

        # Projector
        proj_prefix = "paligemma_with_expert.paligemma.model.multi_modal_projector"
        collect_tensor_info("mm.0.weight", f"{proj_prefix}.linear.weight")
        collect_tensor_info("mm.0.bias", f"{proj_prefix}.linear.bias")

        # Embedding
        collect_tensor_info("embed.weight", "paligemma_with_expert.paligemma.lm_head.weight")

        # Action Expert
        collect_tensor_info("action.action_in.weight", "action_in_proj.weight")
        collect_tensor_info("action.action_in.bias", "action_in_proj.bias")
        collect_tensor_info("action.action_out.weight", "action_out_proj.weight")
        collect_tensor_info("action.action_out.bias", "action_out_proj.bias")
        collect_tensor_info("action.timestep_mlp.in.weight", "time_mlp_in.weight")
        collect_tensor_info("action.timestep_mlp.in.bias", "time_mlp_in.bias")
        collect_tensor_info("action.timestep_mlp.out.weight", "time_mlp_out.weight")
        collect_tensor_info("action.timestep_mlp.out.bias", "time_mlp_out.bias")

        expert_prefix = "paligemma_with_expert.gemma_expert.model"
        collect_tensor_info("action.output_norm.weight", f"{expert_prefix}.norm.dense.weight")
        collect_tensor_info("action.output_norm.bias", f"{expert_prefix}.norm.dense.bias")

        for i in range(18):
            layer_prefix = f"{expert_prefix}.layers.{i}"
            collect_tensor_info(f"action.blk.{i}.attn_q.weight", f"{layer_prefix}.self_attn.q_proj.weight")
            collect_tensor_info(f"action.blk.{i}.attn_k.weight", f"{layer_prefix}.self_attn.k_proj.weight")
            collect_tensor_info(f"action.blk.{i}.attn_v.weight", f"{layer_prefix}.self_attn.v_proj.weight")
            collect_tensor_info(f"action.blk.{i}.attn_output.weight", f"{layer_prefix}.self_attn.o_proj.weight")
            collect_tensor_info(f"action.blk.{i}.attn_norm.weight", f"{layer_prefix}.input_layernorm.dense.weight")
            collect_tensor_info(f"action.blk.{i}.attn_norm.bias", f"{layer_prefix}.input_layernorm.dense.bias")
            collect_tensor_info(f"action.blk.{i}.ffn_norm.weight", f"{layer_prefix}.post_attention_layernorm.dense.weight")
            collect_tensor_info(f"action.blk.{i}.ffn_norm.bias", f"{layer_prefix}.post_attention_layernorm.dense.bias")
            collect_tensor_info(f"action.blk.{i}.ffn_gate.weight", f"{layer_prefix}.mlp.gate_proj.weight")
            collect_tensor_info(f"action.blk.{i}.ffn_up.weight", f"{layer_prefix}.mlp.up_proj.weight")
            collect_tensor_info(f"action.blk.{i}.ffn_down.weight", f"{layer_prefix}.mlp.down_proj.weight")

        # PaliGemma
        pali_prefix = "paligemma_with_expert.paligemma.model.language_model"
        for i in range(18):
            pali_layer_prefix = f"{pali_prefix}.layers.{i}"
            collect_tensor_info(f"pali.blk.{i}.attn_norm.weight", f"{pali_layer_prefix}.input_layernorm.weight")
            collect_tensor_info(f"pali.blk.{i}.attn_q.weight", f"{pali_layer_prefix}.self_attn.q_proj.weight")
            collect_tensor_info(f"pali.blk.{i}.attn_k.weight", f"{pali_layer_prefix}.self_attn.k_proj.weight")
            collect_tensor_info(f"pali.blk.{i}.attn_v.weight", f"{pali_layer_prefix}.self_attn.v_proj.weight")
            collect_tensor_info(f"pali.blk.{i}.attn_o.weight", f"{pali_layer_prefix}.self_attn.o_proj.weight")
            collect_tensor_info(f"pali.blk.{i}.ffn_norm.weight", f"{pali_layer_prefix}.post_attention_layernorm.weight")
            collect_tensor_info(f"pali.blk.{i}.ffn_gate.weight", f"{pali_layer_prefix}.mlp.gate_proj.weight")
            collect_tensor_info(f"pali.blk.{i}.ffn_up.weight", f"{pali_layer_prefix}.mlp.up_proj.weight")
            collect_tensor_info(f"pali.blk.{i}.ffn_down.weight", f"{pali_layer_prefix}.mlp.down_proj.weight")

        print_quant_plan(tensor_names_shapes, args)
        print("\nDry run complete. No files were written.")
        exit(0)

    # Initialize K-quant quantizer if any K-quant type is selected
    quantizer = None
    all_quant_args = [args.quant_llm, args.quant_vision, args.quant_embedding, args.quant_action]
    if any(q in KQUANT_STRS for q in all_quant_args if q):
        from ggml_quant import GGMLQuantizer
        quantizer = GGMLQuantizer()
        print("  Initialized K-quant quantizer (ctypes)")

    # Load the per-element vision Fisher imatrix (if provided).
    args._vision_imatrix = None
    if getattr(args, "vision_imatrix", None):
        from gguf import GGUFReader
        rdr = GGUFReader(args.vision_imatrix)
        vim = {}
        for t in rdr.tensors:
            if t.name.endswith(".in_sum2"):
                base = t.name[: -len(".in_sum2")]
                # gguf-py returns numpy with the original (n_out, n_in) shape.
                vim[base] = np.asarray(t.data, dtype=np.float32)
        args._vision_imatrix = vim
        print(f"  Loaded vision imatrix: {len(vim)} tensors from {args.vision_imatrix}")

    # Export unified GGUF
    gguf_path = export_unified(st, output_dir, model_dir, model_config, args, quantizer)

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)
    output_filename = get_output_filename(args)
    print("\nGenerated files:")
    print(f"  {output_dir}/{output_filename}  - Unified model")
    print(f"  {output_dir}/tokenizer.model    - SentencePiece tokenizer")
    print(f"  {output_dir}/norm_stats.json    - Normalization statistics")
    print("\nUsage:")
    print(f"  ./bin/pi05 -m {output_dir} -i image.png -p \"pick up the object\"")
