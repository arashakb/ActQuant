#!/usr/bin/env python3
"""
Test script for SigLIP quantization with padding support.

Compares different quantization types (Q4_0, Q4_K_M, Q2_K, etc.) against F16 baseline.
"""

import os
import subprocess
import sys
from pathlib import Path

# Paths
QUANTIZE_TOOL = "./build/bin/quantize-vision"
SIGLIP_F16 = "/path/to/openvla-oft-checkpoints/oft_spatial_gguf/siglip.gguf"
OUTPUT_DIR = "/path/to/openvla-oft-checkpoints/oft_spatial_gguf"

# Quantization types to test
QUANT_TYPES = [
    # Legacy quants (block size 32)
    ("Q4_0", "--pad"),
    ("Q5_0", "--pad"),
    ("Q8_0", "--pad"),

    # K-quants (block size 256) - require padding for SigLIP
    ("Q2_K", "--pad"),
    ("Q4_K_M", "--pad"),
    ("Q4_K_S", "--pad"),
    ("Q5_K_M", "--pad"),

    # Test without padding (should skip incompatible tensors)
    ("Q4_K_M", "--no-pad"),
]


def get_file_size_mb(filepath):
    """Get file size in MB."""
    return os.path.getsize(filepath) / (1024 * 1024)


def run_quantization(input_path, output_path, quant_type, pad_flag):
    """Run quantization and return success status."""
    cmd = [QUANTIZE_TOOL, input_path, output_path, quant_type]
    if pad_flag:
        cmd.append(pad_flag)

    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Quantization failed!")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def main():
    # Check if quantize tool exists
    if not os.path.exists(QUANTIZE_TOOL):
        print(f"Error: Quantization tool not found at {QUANTIZE_TOOL}")
        print("Please build the project first:")
        print("  cd /path/to/openvla.cpp")
        print("  cmake -B build -DOPENVLA_OFT_SPATIAL_MODULE=ON")
        print("  cmake --build build --target quantize-vision")
        return 1

    # Check if input file exists
    if not os.path.exists(SIGLIP_F16):
        print(f"Error: Input file not found: {SIGLIP_F16}")
        return 1

    # Get baseline size
    baseline_size = get_file_size_mb(SIGLIP_F16)
    print(f"\n{'='*80}")
    print(f"BASELINE: {SIGLIP_F16}")
    print(f"Size: {baseline_size:.2f} MB")
    print(f"{'='*80}\n")

    # Create output directory if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Results table
    results = []

    # Run quantizations
    for quant_type, pad_flag in QUANT_TYPES:
        # Generate output filename
        pad_suffix = "_padded" if pad_flag == "--pad" else "_nopad"
        output_name = f"siglip_{quant_type.lower()}{pad_suffix}.gguf"
        output_path = os.path.join(OUTPUT_DIR, output_name)

        # Run quantization
        success = run_quantization(SIGLIP_F16, output_path, quant_type, pad_flag)

        if success and os.path.exists(output_path):
            output_size = get_file_size_mb(output_path)
            reduction = (1 - output_size / baseline_size) * 100
            results.append({
                'type': quant_type,
                'pad': pad_flag,
                'size_mb': output_size,
                'reduction': reduction,
                'path': output_path
            })
        else:
            results.append({
                'type': quant_type,
                'pad': pad_flag,
                'size_mb': None,
                'reduction': None,
                'path': None
            })

    # Print summary table
    print(f"\n{'='*80}")
    print("QUANTIZATION SUMMARY")
    print(f"{'='*80}")
    print(f"Baseline (F16): {baseline_size:.2f} MB")
    print(f"\n{' Quant Type ':<15} {'Padding':<10} {'Size (MB)':<12} {'Reduction':<12} {'Status':<10}")
    print("-" * 80)

    for r in results:
        pad_str = "Yes" if r['pad'] == "--pad" else "No"
        if r['size_mb'] is not None:
            status = "✓ Success"
            print(f"{r['type']:<15} {pad_str:<10} {r['size_mb']:<12.2f} {r['reduction']:<11.1f}% {status:<10}")
        else:
            status = "✗ Failed"
            print(f"{r['type']:<15} {pad_str:<10} {'N/A':<12} {'N/A':<12} {status:<10}")

    print(f"{'='*80}\n")

    # Print recommendations
    print("RECOMMENDATIONS:")
    print("-" * 80)

    best_q4 = None
    best_q2 = None

    for r in results:
        if r['size_mb'] and r['type'].startswith('Q4'):
            if best_q4 is None or r['size_mb'] < best_q4['size_mb']:
                best_q4 = r
        if r['size_mb'] and r['type'].startswith('Q2'):
            if best_q2 is None or r['size_mb'] < best_q2['size_mb']:
                best_q2 = r

    if best_q4:
        print(f"Best Q4 variant: {best_q4['type']} ({best_q4['reduction']:.1f}% reduction)")
        print(f"  Size: {best_q4['size_mb']:.2f} MB")
        print(f"  File: {best_q4['path']}")

    if best_q2:
        print(f"\nBest Q2 variant: {best_q2['type']} ({best_q2['reduction']:.1f}% reduction)")
        print(f"  Size: {best_q2['size_mb']:.2f} MB")
        print(f"  File: {best_q2['path']}")

    print(f"\n{'='*80}")
    print("NEXT STEPS:")
    print("-" * 80)
    print("1. Test inference with quantized models to verify accuracy")
    print("2. Compare output differences vs F16 baseline")
    print("3. Measure inference speed for each quantization type")
    print(f"{'='*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
