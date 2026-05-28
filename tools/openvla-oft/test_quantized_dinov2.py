#!/usr/bin/env python3
"""
Test script to verify quantized DINOv2 models work with the OpenVLA-OFT pipeline.
Compares outputs between original F16 DINOv2 and various quantized versions.
"""
import sys
import os
import numpy as np
from PIL import Image

# Add the build directory to path
sys.path.insert(0, "/path/to/openvla.cpp/oft_spatial/bin")
import openvla_oft_spatial

CHECKPOINT_DIR = "/path/to/openvla-oft-checkpoints/oft_spatial_gguf"
TEST_IMAGE = "/path/to/vote/vote-gguf/2.png"

def run_model(dinov2_path, label, verbose=True, siglip_path=None):
    """Run the model with specified DINOv2 model path."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing with {label}")
        print(f"{'='*60}")

    if siglip_path is None:
        siglip_path = f"{CHECKPOINT_DIR}/siglip.gguf"

    model = openvla_oft_spatial.OpenvlaOFTPipelineWithProprio(
        dinov2_model_path=dinov2_path,
        siglip_model_path=siglip_path,
        proj_model_path=f"{CHECKPOINT_DIR}/proj.gguf",
        llm_model_path=f"{CHECKPOINT_DIR}/llm_q4_k_m.gguf",
        action_head_path=f"{CHECKPOINT_DIR}/action_head.gguf",
        proprio_proj_path=f"{CHECKPOINT_DIR}/proprio_projector.gguf",
        tokenizer_path=f"{CHECKPOINT_DIR}/language_model/tokenizer.json",
        device_name="CUDA0",
        n_threads=4,
        max_nodes=4096,
        ngl=999,
        n_ctx=768
    )

    # Test input
    proprio = [0.1, 0.05, 1.0, 3.0, -0.5, 0.2, 0.02, -0.02]
    instruction = "pick up the black bowl"

    if verbose:
        print(f"Running inference...")
    actions = model.run(TEST_IMAGE, instruction, proprio)
    if verbose:
        print(f"Actions shape: {actions.shape}")
        print(f"First 7 action values: {actions[:7]}")

    return actions

def get_file_size_mb(path):
    """Get file size in MB."""
    return os.path.getsize(path) / (1024 * 1024)

def main():
    print("="*70)
    print("DINOv2 Quantization Comparison Test")
    print("="*70)

    # Define models to test (DINOv2 variations)
    models = [
        ("dinov2.gguf", "Original (F16)"),
        ("dinov2_q4_0.gguf", "Q4_0"),
        ("dinov2_q4_k_m.gguf", "Q4_K_M"),
        ("dinov2_q2_k.gguf", "Q2_K"),
    ]

    # Also test SigLIP if quantized version exists
    siglip_original = f"{CHECKPOINT_DIR}/siglip.gguf"
    siglip_q4_0 = f"{CHECKPOINT_DIR}/siglip_q4_0.gguf"
    if os.path.exists(siglip_q4_0):
        print("\n" + "="*70)
        print("SigLIP Quantization Info")
        print("="*70)
        orig_size = get_file_size_mb(siglip_original)
        q4_size = get_file_size_mb(siglip_q4_0)
        print(f"SigLIP Original: {orig_size:.1f} MB")
        print(f"SigLIP Q4_0:     {q4_size:.1f} MB ({(1-q4_size/orig_size)*100:.1f}% reduction)")
        print(f"Note: ffn_down weights (4304 dim) cannot be quantized with block-32 methods")

    # Filter to only existing models
    available_models = []
    for filename, label in models:
        path = f"{CHECKPOINT_DIR}/{filename}"
        if os.path.exists(path):
            available_models.append((path, label, filename))
        else:
            print(f"Skipping {label}: {filename} not found")

    if len(available_models) < 2:
        print("Error: Need at least original and one quantized model")
        return 1

    # Run inference for all models
    results = {}
    for path, label, filename in available_models:
        print(f"\nLoading {label}...")
        actions = run_model(path, label, verbose=False)
        size_mb = get_file_size_mb(path)
        results[label] = {
            "actions": actions,
            "size_mb": size_mb,
            "path": path,
        }
        print(f"  Size: {size_mb:.1f} MB, Actions: {actions[:3]}...")

    # Get original as baseline
    original = results["Original (F16)"]["actions"]
    original_size = results["Original (F16)"]["size_mb"]

    # Print comparison table
    print("\n" + "="*70)
    print("QUANTIZATION COMPARISON RESULTS")
    print("="*70)
    print(f"{'Model':<15} {'Size (MB)':<12} {'Reduction':<12} {'Max Diff':<12} {'Mean Diff':<12} {'RMSE':<12}")
    print("-"*70)

    for label, data in results.items():
        actions = data["actions"]
        size_mb = data["size_mb"]

        if label == "Original (F16)":
            reduction = "-"
            max_diff = "-"
            mean_diff = "-"
            rmse = "-"
        else:
            diff = np.abs(original - actions)
            max_diff = f"{np.max(diff):.6f}"
            mean_diff = f"{np.mean(diff):.6f}"
            rmse = f"{np.sqrt(np.mean((original - actions)**2)):.6f}"
            reduction = f"{(1 - size_mb/original_size)*100:.1f}%"

        print(f"{label:<15} {size_mb:<12.1f} {reduction:<12} {max_diff:<12} {mean_diff:<12} {rmse:<12}")

    # Detailed action comparison
    print("\n" + "="*70)
    print("DETAILED ACTION VALUES (first 7 dimensions)")
    print("="*70)

    for label, data in results.items():
        actions = data["actions"][:7]
        print(f"{label:<15}: [{', '.join([f'{a:8.5f}' for a in actions])}]")

    # Per-dimension error analysis
    print("\n" + "="*70)
    print("PER-DIMENSION ERROR (vs Original)")
    print("="*70)
    print(f"{'Dim':<6}", end="")
    for label in results.keys():
        if label != "Original (F16)":
            print(f"{label:<15}", end="")
    print()
    print("-"*70)

    for i in range(min(7, len(original))):
        print(f"{i:<6}", end="")
        for label, data in results.items():
            if label != "Original (F16)":
                err = abs(original[i] - data["actions"][i])
                print(f"{err:<15.6f}", end="")
        print()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for label, data in results.items():
        if label == "Original (F16)":
            continue
        actions = data["actions"]
        diff = np.abs(original - actions)
        max_diff = np.max(diff)

        if max_diff < 0.05:
            quality = "Excellent"
        elif max_diff < 0.1:
            quality = "Good"
        elif max_diff < 0.2:
            quality = "Acceptable"
        else:
            quality = "Significant loss"

        print(f"{label}: {quality} (max error: {max_diff:.4f})")

    # Test combined quantization (both DINOv2 and SigLIP quantized)
    siglip_q4_0 = f"{CHECKPOINT_DIR}/siglip_q4_0.gguf"
    dinov2_q4_k_m = f"{CHECKPOINT_DIR}/dinov2_q4_k_m.gguf"

    if os.path.exists(siglip_q4_0) and os.path.exists(dinov2_q4_k_m):
        print("\n" + "="*70)
        print("COMBINED QUANTIZATION TEST (DINOv2 Q4_K_M + SigLIP Q4_0)")
        print("="*70)

        combined_actions = run_model(dinov2_q4_k_m, "Combined Q4", verbose=False,
                                     siglip_path=siglip_q4_0)

        diff = np.abs(original - combined_actions)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        rmse = np.sqrt(np.mean((original - combined_actions)**2))

        # Calculate total size savings
        orig_dinov2 = get_file_size_mb(f"{CHECKPOINT_DIR}/dinov2.gguf")
        orig_siglip = get_file_size_mb(f"{CHECKPOINT_DIR}/siglip.gguf")
        q_dinov2 = get_file_size_mb(dinov2_q4_k_m)
        q_siglip = get_file_size_mb(siglip_q4_0)

        total_orig = orig_dinov2 + orig_siglip
        total_quant = q_dinov2 + q_siglip

        print(f"Original vision encoders:   {total_orig:.1f} MB (DINOv2: {orig_dinov2:.1f} + SigLIP: {orig_siglip:.1f})")
        print(f"Quantized vision encoders:  {total_quant:.1f} MB (DINOv2: {q_dinov2:.1f} + SigLIP: {q_siglip:.1f})")
        print(f"Total reduction:            {(1-total_quant/total_orig)*100:.1f}%")
        print(f"\nAccuracy vs Original:")
        print(f"  Max error:  {max_diff:.6f}")
        print(f"  Mean error: {mean_diff:.6f}")
        print(f"  RMSE:       {rmse:.6f}")

        print(f"\nAction comparison:")
        print(f"  Original: {original[:7]}")
        print(f"  Combined: {combined_actions[:7]}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
