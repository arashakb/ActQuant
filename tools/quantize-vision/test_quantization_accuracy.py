#!/usr/bin/env python3
"""
Test quantization accuracy by comparing vision encoder outputs.

This script:
1. Loads the original F16 SigLIP model
2. Loads a quantized version
3. Runs inference on test images
4. Compares outputs and computes error metrics
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "openvla-oft"))

try:
    import openvla_oft
    HAS_OPENVLA = True
except ImportError:
    HAS_OPENVLA = False
    print("Warning: openvla_oft module not found. Please build with BUILD_PYTHON=ON")


def compute_error_metrics(output_f16, output_quant):
    """Compute error metrics between two outputs."""
    # Flatten arrays for comparison
    f16_flat = output_f16.flatten()
    quant_flat = output_quant.flatten()

    # Compute errors
    abs_diff = np.abs(f16_flat - quant_flat)
    rel_diff = abs_diff / (np.abs(f16_flat) + 1e-8)

    metrics = {
        'max_abs_error': np.max(abs_diff),
        'mean_abs_error': np.mean(abs_diff),
        'rmse': np.sqrt(np.mean((f16_flat - quant_flat) ** 2)),
        'max_rel_error': np.max(rel_diff),
        'mean_rel_error': np.mean(rel_diff),
        'cosine_similarity': np.dot(f16_flat, quant_flat) / (
            np.linalg.norm(f16_flat) * np.linalg.norm(quant_flat)
        ),
    }

    return metrics


def assess_quality(metrics):
    """Assess quantization quality based on metrics."""
    max_error = metrics['max_abs_error']
    rmse = metrics['rmse']
    cosine_sim = metrics['cosine_similarity']

    if max_error < 0.01 and rmse < 0.005 and cosine_sim > 0.999:
        return "Excellent"
    elif max_error < 0.05 and rmse < 0.02 and cosine_sim > 0.99:
        return "Good"
    elif max_error < 0.1 and rmse < 0.05 and cosine_sim > 0.95:
        return "Acceptable"
    elif max_error < 0.2 and rmse < 0.1 and cosine_sim > 0.90:
        return "Fair"
    else:
        return "Poor"


def test_quantized_model(f16_path, quant_path, device="CPU"):
    """Test a quantized model against F16 baseline."""
    if not HAS_OPENVLA:
        print("Error: openvla_oft module not available")
        return None

    print(f"\nTesting: {Path(quant_path).name}")
    print("-" * 80)

    try:
        # Load models
        print("Loading F16 baseline...")
        # TODO: Add actual model loading code here
        # For now, just return dummy metrics
        print("Loading quantized model...")

        # TODO: Generate test input
        # For now, use dummy data
        dummy_output_f16 = np.random.randn(256, 1152).astype(np.float32)
        dummy_output_quant = dummy_output_f16 + np.random.randn(256, 1152).astype(np.float32) * 0.01

        # Compute metrics
        metrics = compute_error_metrics(dummy_output_f16, dummy_output_quant)
        quality = assess_quality(metrics)

        print(f"Quality Assessment: {quality}")
        print(f"Max Absolute Error: {metrics['max_abs_error']:.6f}")
        print(f"Mean Absolute Error: {metrics['mean_abs_error']:.6f}")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"Cosine Similarity: {metrics['cosine_similarity']:.6f}")

        return metrics

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main test function."""
    print("=" * 80)
    print("SigLIP Quantization Accuracy Test")
    print("=" * 80)

    # Paths
    f16_path = "/path/to/openvla-oft-checkpoints/oft_spatial_gguf/siglip.gguf"
    test_files = [
        "/path/to/openvla-oft-checkpoints/oft_spatial_gguf/siglip_q4_0_padded.gguf",
        "/path/to/openvla-oft-checkpoints/oft_spatial_gguf/siglip_q4_k_m_padded.gguf",
        "/path/to/openvla-oft-checkpoints/oft_spatial_gguf/siglip_q2_k_padded.gguf",
    ]

    results = []

    for quant_path in test_files:
        if Path(quant_path).exists():
            metrics = test_quantized_model(f16_path, quant_path)
            if metrics:
                results.append({
                    'path': quant_path,
                    'metrics': metrics,
                    'quality': assess_quality(metrics)
                })
        else:
            print(f"\nSkipping {Path(quant_path).name} (file not found)")

    # Print summary
    if results:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"{'Model':<40} {'Quality':<12} {'Max Error':<12} {'RMSE':<12}")
        print("-" * 80)

        for r in results:
            model_name = Path(r['path']).name
            print(f"{model_name:<40} {r['quality']:<12} "
                  f"{r['metrics']['max_abs_error']:<12.6f} "
                  f"{r['metrics']['rmse']:<12.6f}")

        print("=" * 80)
    else:
        print("\nNo quantized models found to test.")
        print("Please run test_siglip_quantization.py first to generate quantized models.")

    print("\nNOTE: This is a placeholder test script.")
    print("Full accuracy testing requires:")
    print("  1. Building the Python bindings (BUILD_PYTHON=ON)")
    print("  2. Loading actual vision encoder models")
    print("  3. Running inference on test images")
    print("  4. Comparing feature outputs")

    return 0


if __name__ == "__main__":
    sys.exit(main())
