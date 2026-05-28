# SigLIP Quantization with Padding - Implementation Summary

## Overview

Implemented padding support for vision encoder quantization to enable K-quant (Q4_K_M, Q2_K, etc.) for SigLIP, which has dimensions incompatible with block size 256.

## The Challenge

### SigLIP Dimension Issues

**Problem**: SigLIP has dimensions that prevent K-quant quantization:
- `embedding_length`: **1152** (1152 % 256 = 128) ❌
- `feed_forward_length`: **4304** (4304 % 256 = 208) ❌

**Impact**: Without padding, ~34% of model weights (all `ffn_down` layers) cannot be quantized with K-quants, limiting compression to only ~48% instead of ~75%.

### Tensor Breakdown

| Tensor | Shape | ne[0] | Block 32 | Block 256 | % of Model |
|--------|-------|-------|----------|-----------|------------|
| `attn_q/k/v/out.weight` | [1152, 1152] | 1152 | ✓ | ❌ | 36% |
| `ffn_up.weight` | [1152, 4304] | 1152 | ✓ | ❌ | 30% |
| `ffn_down.weight` | [4304, 1152] | 4304 | ❌ | ❌ | 34% |

**With padding**:
- 1152 → **1280** (next multiple of 256)
- 4304 → **4320** (next multiple of 32) or **4352** (next multiple of 256)

## Solution: Zero-Padding with Metadata

### Design Principles

1. **Transparent**: Padding happens automatically with `--pad` flag
2. **Lossless**: Zero-padding doesn't affect computation (matrix mult with zeros = no change)
3. **Reversible**: Original dimensions stored in GGUF metadata
4. **Efficient**: <5% overhead after quantization

### Implementation

#### Phase 1: Quantization Tool (`quantize-vision.cpp`)

**Added Functions**:
```cpp
int64_t calc_padded_dim(int64_t dim, int block_size)
  // Rounds up to next multiple of block_size

std::vector<float> pad_tensor_data(const float* src, int64_t ne0, int64_t ne1, int64_t padded_ne0)
  // Zero-pads tensor data to padded dimensions
```

**Quantization Flow**:
1. Check if tensor needs quantization
2. If dimension not divisible by block size:
   - Calculate padded dimension
   - Pad data with zeros
   - Quantize padded data
   - Store original dimension in metadata
3. Write to GGUF with metadata

**Metadata Stored**:
```
quantize.padded_tensor_count = <number of padded tensors>
quantize.original_ne0.<tensor_name> = <original first dimension>
```

**Example**:
```
quantize.padded_tensor_count = 135
quantize.original_ne0.v.blk.0.ffn_down.weight = 4304
quantize.original_ne0.v.blk.0.attn_q.weight = 1152
...
```

#### Phase 2: Model Loader (`model_defs.cpp`)

**Loading Flow**:
1. Read GGUF metadata to detect padded tensors
2. Load tensor data (with padding) from file
3. After loading, adjust `ne[0]` to original dimension
4. GGML operations will use original dimension

**Key Code**:
```cpp
// Read padding metadata
int padded_count_idx = gguf_find_key(ctx_gguf.get(), "quantize.padded_tensor_count");
if (padded_count_idx >= 0) {
    // Load original ne[0] for each padded tensor
    std::string meta_key = "quantize.original_ne0." + tensor_name;
    int64_t original_ne0 = gguf_get_val_u32(ctx_gguf.get(), meta_idx);

    // After loading tensor data
    tensor->ne[0] = original_ne0;  // Adjust to original dimension
}
```

**How It Works**:
- Padded quantized data is loaded from file
- Tensor metadata (`ne[0]`) is updated to original dimension
- GGML matrix operations respect `ne[0]`, only using first N rows
- Padded rows are ignored during computation

## Files Modified

### 1. `tools/quantize-vision/quantize-vision.cpp`
- Added `calc_padded_dim()` and `pad_tensor_data()` functions
- Modified main() to accept `--pad` / `--no-pad` flags
- Updated tensor processing loop to detect and pad incompatible tensors
- Added metadata writing for padded tensors
- Updated help text and summary output

**Lines changed**: ~150 additions

### 2. `tools/openvla-oft/model_defs.cpp`
- Modified `BaseModel::load_tensors()` to read padding metadata
- Added dimension adjustment after tensor loading
- Added debug output for padded tensors

**Lines changed**: ~40 additions

### 3. Documentation & Tests
- **README.md**: Complete user guide
- **IMPLEMENTATION_SUMMARY.md**: This file
- **test_siglip_quantization.py**: Automated quantization testing
- **test_quantization_accuracy.py**: Accuracy testing framework

## Testing & Validation

### Build the Tool

```bash
cd /path/to/openvla.cpp
cmake -B build -DOPENVLA_OFT_SPATIAL_MODULE=ON
cmake --build build --target quantize-vision
```

### Test SigLIP Quantization

```bash
# Test basic Q4_K_M with padding
./build/bin/quantize-vision \
    /path/to/openvla-oft-checkpoints/oft_spatial_gguf/siglip.gguf \
    /path/to/openvla-oft-checkpoints/oft_spatial_gguf/siglip_q4_k_m_test.gguf \
    Q4_K_M --pad

# Run automated tests
cd tools/quantize-vision
python3 test_siglip_quantization.py
```

### Expected Output

```
Input:   siglip.gguf
Output:  siglip_q4_k_m.gguf
Type:    Q4_K_M (block size: 256)
Padding: enabled

Processing tensors...
  v.blk.0.attn_q.weight: [1152, 1152] F16 -> Q4_K_M (padded 1152->1280, 2.8x smaller)
  v.blk.0.ffn_down.weight: [4304, 1152] F16 -> Q4_K_M (padded 4304->4352, 2.9x smaller)
  ...

Storing padding metadata for 135 tensors...

========================================
Quantization complete!
========================================
Tensors quantized: 162
Tensors padded:    135
Tensors kept:      273
Input size:        788.0 MB
Output size:       200.5 MB
Reduction:         74.6%

Note: 135 tensor(s) were padded to align with block size 256.
      Original dimensions are stored in metadata.
```

## Expected Results

### Size Comparison

| Model | Type | Padding | Size (MB) | Reduction | Compression Ratio |
|-------|------|---------|-----------|-----------|-------------------|
| SigLIP | F16 | - | 788 | 0% | 1.0x |
| SigLIP | Q4_0 | No | 408 | 48% | 1.9x |
| SigLIP | Q4_0 | Yes | 285 | 64% | 2.8x |
| SigLIP | Q4_K_M | No | N/A | - | - |
| SigLIP | Q4_K_M | Yes | **200** | **75%** | **3.9x** |
| SigLIP | Q2_K | Yes | **130** | **84%** | **6.1x** |

### Quality Assessment

Based on DINOv2 quantization results:
- **Q4_K_M**: Excellent quality, <0.005 RMSE
- **Q2_K**: Good quality, <0.02 RMSE
- **Q8_0**: Near-lossless, <0.001 RMSE

## Padding Overhead Analysis

### For Q4_K_M (block size 256):

**Before quantization**:
- Attention weights (108 tensors): 128 × 1152 × 108 ≈ 31 MB padding
- FFN up (27 tensors): 128 × 4304 × 27 ≈ 29 MB padding
- FFN down (27 tensors): 48 × 1152 × 27 ≈ 3 MB padding
- **Total**: ~63 MB padding (8% of 788 MB)

**After quantization** (4 bits per element):
- ~63 MB × (4/32) ≈ **8 MB overhead**
- Final size: 200 MB (vs 788 MB original)
- Overhead: **4%** of final size

**Cost-Benefit**:
- Overhead: 8 MB
- Benefit: Enable quantization of 34% of weights that were previously unquantizable
- Net gain: ~200 MB savings (75% vs 48% reduction)

## Potential Issues & Solutions

### Issue 1: Inference Errors with Padded Tensors

**Symptom**: Segfault or incorrect results during inference

**Cause**: Quantized tensor layout incompatible with dimension adjustment

**Solution**: Add dequantization path in model loader:
```cpp
if (tensor_is_padded && tensor_is_quantized) {
    // Dequantize to F32
    float* dequant_data = dequantize_ggml_tensor(tensor);

    // Slice to original dimensions
    float* sliced_data = slice_tensor(dequant_data, original_ne0, ne1);

    // Store as F16
    tensor = convert_to_f16(sliced_data);
}
```

This keeps padded tensors unquantized in memory (trades size for correctness).

### Issue 2: GGML Operations Don't Respect ne[0]

**Symptom**: Outputs differ from F16 baseline

**Cause**: GGML might compute strides based on actual data layout, not `ne[0]`

**Solution**: Use `ggml_view` to create sliced views:
```cpp
// In build_graph
if (weight_is_padded) {
    // Create view with original dimensions
    weight = ggml_view_2d(ctx, weight_padded, original_ne0, ne1, nb1, 0);
}
```

### Issue 3: Large Memory Usage

**Symptom**: Increased RAM usage during inference

**Cause**: Padded tensors take more memory

**Solution**: Expected and acceptable. Padding adds <5% to quantized model size.

## Next Steps

### Immediate Testing

1. **Build & run basic test**:
   ```bash
   cmake --build build --target quantize-vision
   ./build/bin/quantize-vision siglip.gguf siglip_q4_k_m.gguf Q4_K_M --pad
   ```

2. **Verify file size**:
   ```bash
   ls -lh siglip*.gguf
   # Should see ~200 MB for Q4_K_M
   ```

3. **Test inference**:
   ```bash
   # Use openvla-oft tool to run inference
   # Compare outputs with F16 baseline
   ```

### Accuracy Validation

1. Build Python bindings:
   ```bash
   cmake -B build -DOPENVLA_OFT_SPATIAL_MODULE=ON -DBUILD_PYTHON=ON
   cmake --build build
   ```

2. Run quantization comparison:
   ```bash
   python3 tools/quantize-vision/test_quantization_accuracy.py
   ```

3. Test on LIBERO benchmark:
   - Load quantized SigLIP
   - Run full pipeline
   - Compare success rates vs F16

### Performance Benchmarks

1. **Inference speed**: Measure latency for each quant type
2. **Memory usage**: Monitor peak RAM during inference
3. **Accuracy**: Compute feature similarity vs F16 baseline

## Implementation Notes

### Why This Approach?

**Alternative 1**: Dequantize during loading
- ✅ Always correct
- ❌ Loses compression benefit
- ❌ More complex

**Alternative 2**: Pad during inference (ggml_view)
- ✅ Flexible
- ❌ Requires graph modifications
- ❌ Per-layer overhead

**Our approach**: Adjust dimensions after loading
- ✅ Simple implementation
- ✅ Maintains compression
- ✅ Minimal overhead
- ⚠️ May need fallback to dequantization if issues arise

### Assumptions

1. GGML matrix operations respect `tensor->ne[0]` for dimension checking
2. Quantized data layout allows using subset of rows (first `ne[0]` out of `padded_ne0`)
3. Zero-padding doesn't affect numerical stability

If any assumption fails, we have documented fallback approaches.

## Summary

✅ **Implemented**: Padding support for quantize-vision tool
✅ **Implemented**: Metadata storage for original dimensions
✅ **Implemented**: Model loader adjustments for padded tensors
✅ **Created**: Test scripts for validation
✅ **Documented**: Complete usage guide

📝 **TODO**:
- Test with actual SigLIP inference
- Validate accuracy vs F16 baseline
- Benchmark performance
- Add dequantization fallback if needed

**Expected Outcome**:
- SigLIP quantizable with K-quants (Q4_K_M, Q2_K)
- 75-84% size reduction (vs 48% without padding)
- Minimal accuracy loss (<1% typical for K-quants)
- <5% padding overhead in final quantized model
