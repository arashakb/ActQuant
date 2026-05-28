/**
 * Vision Encoder Quantization Tool
 *
 * Quantizes vision encoder GGUF files (DINOv2, SigLIP, CLIP, etc.) to lower precision.
 *
 * Usage:
 *   quantize-vision input.gguf output.gguf <type> [--pad|--no-pad]
 *
 * Supported types:
 *   Legacy:  Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 (block size 32)
 *   K-quant: Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K (block size 256)
 *   Float:   F16, F32
 *
 * Options:
 *   --pad     Enable padding for incompatible dimensions (default)
 *   --no-pad  Disable padding, skip incompatible tensors
 *
 * Example:
 *   quantize-vision siglip.gguf siglip_q4_k_m.gguf Q4_K_M --pad
 */

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// Quantization type mapping
struct QuantType {
    const char *name;
    ggml_type type;
    int block_size;
};

static const QuantType QUANT_TYPES[] = {
    // Legacy quantization (block size 32)
    {"Q4_0", GGML_TYPE_Q4_0, 32},
    {"Q4_1", GGML_TYPE_Q4_1, 32},
    {"Q5_0", GGML_TYPE_Q5_0, 32},
    {"Q5_1", GGML_TYPE_Q5_1, 32},
    {"Q8_0", GGML_TYPE_Q8_0, 32},
    // K-quant (block size 256, better quality)
    {"Q2_K",   GGML_TYPE_Q2_K,   256},
    {"Q3_K_S", GGML_TYPE_Q3_K,   256},
    {"Q3_K_M", GGML_TYPE_Q3_K,   256},  // Same type, different name for compatibility
    {"Q3_K_L", GGML_TYPE_Q3_K,   256},
    {"Q4_K_S", GGML_TYPE_Q4_K,   256},
    {"Q4_K_M", GGML_TYPE_Q4_K,   256},  // Same type, different name for compatibility
    {"Q5_K_S", GGML_TYPE_Q5_K,   256},
    {"Q5_K_M", GGML_TYPE_Q5_K,   256},
    {"Q6_K",   GGML_TYPE_Q6_K,   256},
    // Float types
    {"F16",  GGML_TYPE_F16,  1},
    {"F32",  GGML_TYPE_F32,  1},
};

static const int NUM_QUANT_TYPES = sizeof(QUANT_TYPES) / sizeof(QUANT_TYPES[0]);

// Tensors that should NOT be quantized
static const char *SKIP_PATTERNS[] = {
    "patch_embd",     // Patch embedding has shape [14,14,3,N] - 14 not divisible by 32
    "position_embd",  // Position embeddings - keep full precision
    "class_embd",     // Class token
    "reg_embd",       // Register tokens
    "ln1", "ln2",     // Layer norms
    "ls1", "ls2",     // Layer scales
    ".bias",          // Biases
};

static const int NUM_SKIP_PATTERNS = sizeof(SKIP_PATTERNS) / sizeof(SKIP_PATTERNS[0]);

// Per-tensor imatrix entry. `data` is the raw Fisher diagonal already divided
// by `counts` so callers can use it as `quant_weights` directly.
//   per_weight = false: data has length ne0 (one weight per input column)
//   per_weight = true:  data has length ne0 * ne1 (full per-weight Fisher)
struct ImatrixEntry {
    std::vector<float> data;
    bool per_weight = false;
    int64_t ne0 = 0;
    int64_t ne1 = 0;
};

// Load AMF / imatrix GGUF (mirrors src/llama-quant.cpp:217 load_imatrix).
// Returns a map keyed by tensor base name (no `.in_sum2` / `.counts` suffix).
static bool load_imatrix_gguf(const char *path,
                              std::unordered_map<std::string, ImatrixEntry> &out) {
    struct ggml_context *ctx_data = nullptr;
    struct gguf_init_params params = {
        /*.no_alloc =*/ false,
        /*.ctx      =*/ &ctx_data,
    };
    struct gguf_context *ctx = gguf_init_from_file(path, params);
    if (!ctx) {
        fprintf(stderr, "Error: failed to open imatrix file '%s'\n", path);
        return false;
    }

    const std::string sums_suffix   = ".in_sum2";
    const std::string counts_suffix = ".counts";

    std::map<std::string, std::pair<struct ggml_tensor *, struct ggml_tensor *>> sums_counts_for;
    for (struct ggml_tensor *cur = ggml_get_first_tensor(ctx_data); cur;
         cur = ggml_get_next_tensor(ctx_data, cur)) {
        std::string name = cur->name;
        if (name.empty()) continue;
        if (name.size() >= sums_suffix.size() &&
            name.compare(name.size() - sums_suffix.size(), sums_suffix.size(), sums_suffix) == 0) {
            sums_counts_for[name.substr(0, name.size() - sums_suffix.size())].first = cur;
        } else if (name.size() >= counts_suffix.size() &&
                   name.compare(name.size() - counts_suffix.size(), counts_suffix.size(), counts_suffix) == 0) {
            sums_counts_for[name.substr(0, name.size() - counts_suffix.size())].second = cur;
        }
    }

    for (auto &kv : sums_counts_for) {
        const std::string &name = kv.first;
        const struct ggml_tensor *sums   = kv.second.first;
        const struct ggml_tensor *counts = kv.second.second;
        if (!sums || !counts) {
            fprintf(stderr, "Warning: mismatched sums/counts for '%s' in imatrix\n", name.c_str());
            continue;
        }
        const int64_t ne0 = sums->ne[0];
        const int64_t ne1 = sums->ne[1];
        ImatrixEntry e;
        e.ne0 = ne0;
        e.ne1 = ne1;
        e.per_weight = (ne1 > 1);  // [ne0,ne1] = [d_in,d_out] per-weight Fisher
        e.data.resize((size_t)ne0 * ne1);
        const float *src = (const float *)sums->data;
        for (int64_t j = 0; j < ne1; ++j) {
            const int64_t count_idx = (ggml_nelements(counts) > 1) ? j : 0;
            const float c = ((const float *)counts->data)[count_idx];
            const float inv = (c > 0.0f) ? 1.0f / c : 0.0f;
            for (int64_t i = 0; i < ne0; ++i) {
                e.data[j * ne0 + i] = (c > 0.0f) ? src[j * ne0 + i] * inv : 1.0f;
            }
        }
        out[name] = std::move(e);
    }

    gguf_free(ctx);
    ggml_free(ctx_data);
    printf("Loaded %zu imatrix entries from %s\n", out.size(), path);
    return true;
}

// Quantize a 2D tensor using ggml_quantize_chunk, optionally guided by an
// imatrix. Mirrors src/llama-quant.cpp:494 llama_tensor_quantize_impl: for
// per-weight Fisher, walk row-by-row and pass each row's slice; otherwise
// pass the per-column vector to a single chunk call.
static size_t quantize_tensor_with_imatrix(
    enum ggml_type new_type,
    const float *f32_data, void *new_data,
    int64_t nrows, int64_t n_per_row,
    const float *imatrix, bool imatrix_is_per_weight) {
    if (imatrix_is_per_weight) {
        size_t total = 0;
        for (int64_t row = 0; row < nrows; ++row) {
            const float *row_imatrix = imatrix + row * n_per_row;
            total += ggml_quantize_chunk(new_type, f32_data, new_data,
                                         row * n_per_row, 1, n_per_row,
                                         row_imatrix);
        }
        return total;
    }
    return ggml_quantize_chunk(new_type, f32_data, new_data,
                               0, nrows, n_per_row, imatrix);
}

// Calculate padded dimension to align with block size
static int64_t calc_padded_dim(int64_t dim, int block_size) {
    if (block_size <= 1) return dim;  // No padding for F16/F32
    if (dim % block_size == 0) return dim;  // Already aligned

    // Round up to next multiple of block_size
    return ((dim + block_size - 1) / block_size) * block_size;
}

// Pad tensor data with zeros to align with block size
static std::vector<float> pad_tensor_data(
    const float* src,
    int64_t ne0, int64_t ne1,  // Original dimensions
    int64_t padded_ne0          // Padded first dimension
) {
    size_t padded_size = padded_ne0 * ne1;
    std::vector<float> padded(padded_size, 0.0f);  // Zero-initialize

    // Copy original data row-by-row
    for (int64_t i1 = 0; i1 < ne1; i1++) {
        for (int64_t i0 = 0; i0 < ne0; i0++) {
            padded[i1 * padded_ne0 + i0] = src[i1 * ne0 + i0];
        }
        // Rows [ne0, padded_ne0) remain zero-padded
    }

    return padded;
}

static void print_usage(const char *prog) {
    printf("Usage: %s <input.gguf> <output.gguf> <type> [--pad|--no-pad] [--imatrix <path>]\n\n", prog);
    printf("Quantizes vision encoder GGUF files to lower precision.\n\n");
    printf("Supported quantization types:\n");
    printf("  Legacy (block size 32):\n");
    printf("    Q4_0, Q4_1, Q5_0, Q5_1, Q8_0\n");
    printf("  K-quant (block size 256, recommended):\n");
    printf("    Q2_K          - 2-bit, smallest size\n");
    printf("    Q3_K_S/M/L    - 3-bit variants\n");
    printf("    Q4_K_S, Q4_K_M - 4-bit, good quality/size balance\n");
    printf("    Q5_K_S, Q5_K_M - 5-bit, better quality\n");
    printf("    Q6_K          - 6-bit, near-lossless\n");
    printf("  Float:\n");
    printf("    F16, F32\n");
    printf("\nOptions:\n");
    printf("  --pad             Enable padding for incompatible dimensions (default)\n");
    printf("  --no-pad          Disable padding, skip incompatible tensors\n");
    printf("  --imatrix <path>  Use AMF/imatrix GGUF as quantization weights\n");
    printf("                    (per-tensor `<name>.in_sum2` + `.counts`).\n");
    printf("\nNote: Padding adds zero rows/cols to make dimensions divisible by block size.\n");
    printf("      The original dimensions are stored in metadata for inference.\n");
    printf("\nExamples:\n");
    printf("  %s dinov2.gguf dinov2_q4_k_m.gguf Q4_K_M\n", prog);
    printf("  %s siglip.gguf siglip_q4_k_m.gguf Q4_K_M --pad\n", prog);
}

static const QuantType *find_quant_type(const char *name) {
    for (int i = 0; i < NUM_QUANT_TYPES; i++) {
        if (strcasecmp(QUANT_TYPES[i].name, name) == 0) {
            return &QUANT_TYPES[i];
        }
    }
    return nullptr;
}

static bool should_skip_tensor(const char *name) {
    for (int i = 0; i < NUM_SKIP_PATTERNS; i++) {
        if (strstr(name, SKIP_PATTERNS[i]) != nullptr) {
            return true;
        }
    }
    return false;
}

static bool can_quantize_shape(const int64_t *ne, int n_dims, int block_size) {
    // Only quantize 2D+ tensors
    if (n_dims < 2) {
        return false;
    }

    // Check if dimensions are divisible by block size
    // Only check the innermost dimension for quantization
    if (ne[0] % block_size != 0) {
        return false;
    }

    return true;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    const char *input_path = argv[1];
    const char *output_path = argv[2];
    const char *quant_type_name = argv[3];

    // Parse remaining flags (--pad / --no-pad / --imatrix <path>)
    bool enable_padding = true;
    const char *imatrix_path = nullptr;
    for (int ai = 4; ai < argc; ++ai) {
        if (strcmp(argv[ai], "--no-pad") == 0) {
            enable_padding = false;
        } else if (strcmp(argv[ai], "--pad") == 0) {
            enable_padding = true;
        } else if (strcmp(argv[ai], "--imatrix") == 0) {
            if (ai + 1 >= argc) {
                fprintf(stderr, "Error: --imatrix requires a path argument\n\n");
                print_usage(argv[0]);
                return 1;
            }
            imatrix_path = argv[++ai];
        } else {
            fprintf(stderr, "Error: Unknown option '%s'\n\n", argv[ai]);
            print_usage(argv[0]);
            return 1;
        }
    }

    const QuantType *quant_type = find_quant_type(quant_type_name);
    if (!quant_type) {
        fprintf(stderr, "Error: Unknown quantization type '%s'\n\n", quant_type_name);
        print_usage(argv[0]);
        return 1;
    }

    printf("Input:   %s\n", input_path);
    printf("Output:  %s\n", output_path);
    printf("Type:    %s (block size: %d)\n", quant_type->name, quant_type->block_size);
    printf("Padding: %s\n", enable_padding ? "enabled" : "disabled");
    printf("Imatrix: %s\n", imatrix_path ? imatrix_path : "(none)");
    printf("\n");

    std::unordered_map<std::string, ImatrixEntry> imatrix_map;
    if (imatrix_path) {
        if (!load_imatrix_gguf(imatrix_path, imatrix_map)) {
            return 1;
        }
    }

    // Open input file
    struct ggml_context *ctx_meta = nullptr;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &ctx_meta,
    };

    struct gguf_context *ctx_in = gguf_init_from_file(input_path, params);
    if (!ctx_in) {
        fprintf(stderr, "Error: Failed to open input file '%s'\n", input_path);
        return 1;
    }

    // Get metadata
    int n_tensors = gguf_get_n_tensors(ctx_in);
    int n_kv = gguf_get_n_kv(ctx_in);

    printf("Metadata entries: %d\n", n_kv);
    printf("Tensors: %d\n", n_tensors);

    // Open input file for reading tensor data
    std::ifstream fin(input_path, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "Error: Failed to open input file for reading\n");
        gguf_free(ctx_in);
        ggml_free(ctx_meta);
        return 1;
    }

    size_t data_offset = gguf_get_data_offset(ctx_in);

    // Process tensors and collect data
    printf("\nProcessing tensors...\n");

    int n_quantized = 0;
    int n_skipped = 0;
    size_t input_size = 0;
    size_t output_size = 0;

    struct TensorInfo {
        std::string name;
        ggml_type type;
        int n_dims;
        int64_t ne[GGML_MAX_DIMS];
        int64_t original_ne0;  // Original ne[0] before padding (0 if not padded)
        std::vector<uint8_t> data;
    };

    std::vector<TensorInfo> tensor_infos;
    int n_padded = 0;

    for (int i = 0; i < n_tensors; i++) {
        const char *name = gguf_get_tensor_name(ctx_in, i);
        struct ggml_tensor *tensor_in = ggml_get_tensor(ctx_meta, name);

        if (!tensor_in) {
            fprintf(stderr, "Error: Tensor '%s' not found in context\n", name);
            continue;
        }

        enum ggml_type type_in = tensor_in->type;
        int n_dims = ggml_n_dims(tensor_in);
        size_t tensor_size_in = ggml_nbytes(tensor_in);

        input_size += tensor_size_in;

        // Determine output type and if we need padding
        enum ggml_type type_out = type_in;
        bool do_quantize = false;
        bool needs_padding = false;
        int64_t padded_ne0 = tensor_in->ne[0];

        // Check if we should skip this tensor
        bool skip = should_skip_tensor(name);

        if (!skip && (type_in == GGML_TYPE_F32 || type_in == GGML_TYPE_F16)) {
            // Check if dimensions are compatible with block size
            bool can_quantize_native = can_quantize_shape(tensor_in->ne, n_dims, quant_type->block_size);

            if (can_quantize_native) {
                // Can quantize without padding
                type_out = quant_type->type;
                do_quantize = true;
            } else if (enable_padding && n_dims >= 2 && tensor_in->ne[0] > quant_type->block_size) {
                // Check if we can pad to make it compatible
                padded_ne0 = calc_padded_dim(tensor_in->ne[0], quant_type->block_size);
                if (padded_ne0 != tensor_in->ne[0]) {
                    needs_padding = true;
                    type_out = quant_type->type;
                    do_quantize = true;
                }
            }
        }

        // Read input tensor data
        size_t tensor_offset = gguf_get_tensor_offset(ctx_in, i);
        fin.seekg(data_offset + tensor_offset);

        std::vector<uint8_t> data_in(tensor_size_in);
        fin.read(reinterpret_cast<char *>(data_in.data()), tensor_size_in);

        // Prepare output data
        TensorInfo info;
        info.name = name;
        info.n_dims = n_dims;
        info.original_ne0 = 0;  // No padding by default

        // Set dimensions (will be padded dimensions if padding is used)
        for (int d = 0; d < GGML_MAX_DIMS; d++) {
            info.ne[d] = tensor_in->ne[d];
        }

        if (do_quantize && type_out != type_in) {
            // Convert to float32 first if needed
            std::vector<float> data_f32;
            size_t n_elements = ggml_nelements(tensor_in);

            if (type_in == GGML_TYPE_F16) {
                data_f32.resize(n_elements);
                const ggml_fp16_t *src = reinterpret_cast<const ggml_fp16_t *>(data_in.data());
                for (size_t j = 0; j < n_elements; j++) {
                    data_f32[j] = ggml_fp16_to_fp32(src[j]);
                }
            } else {
                data_f32.resize(n_elements);
                memcpy(data_f32.data(), data_in.data(), n_elements * sizeof(float));
            }

            // Apply padding if needed
            std::vector<float> *data_to_quantize = &data_f32;
            std::vector<float> padded_data;

            if (needs_padding) {
                // Calculate dimensions for padding
                int64_t ne0 = tensor_in->ne[0];
                int64_t ne1 = (n_dims >= 2) ? tensor_in->ne[1] : 1;

                padded_data = pad_tensor_data(data_f32.data(), ne0, ne1, padded_ne0);
                data_to_quantize = &padded_data;

                // Store original dimension for metadata
                info.original_ne0 = ne0;
                info.ne[0] = padded_ne0;  // Update to padded dimension
                n_padded++;
            }
            (void)n_elements;

            // Quantize
            int64_t q_nrows    = (n_dims >= 2) ? (needs_padding ? tensor_in->ne[1] : tensor_in->ne[1]) : 1;
            int64_t q_n_per_row = needs_padding ? padded_ne0 : tensor_in->ne[0];

            size_t bytes_out = ggml_row_size(type_out, q_n_per_row) * q_nrows;
            info.data.resize(bytes_out);
            info.type = type_out;

            // Look up imatrix entry for this tensor (after padding-aware sizing)
            const float *imatrix_ptr = nullptr;
            bool imatrix_per_weight = false;
            std::vector<float> padded_imatrix;
            auto it = imatrix_map.find(name);
            if (it != imatrix_map.end()) {
                const ImatrixEntry &ie = it->second;
                const int64_t orig_ne0 = tensor_in->ne[0];
                const int64_t orig_ne1 = (n_dims >= 2) ? tensor_in->ne[1] : 1;
                bool size_ok = false;
                if (ie.per_weight && ie.ne0 == orig_ne0 && ie.ne1 == orig_ne1) {
                    imatrix_per_weight = true;
                    size_ok = true;
                } else if (!ie.per_weight && (int64_t)ie.data.size() == orig_ne0) {
                    size_ok = true;
                }
                if (!size_ok) {
                    fprintf(stderr, "  warn: imatrix shape [%lld,%lld] doesn't match tensor [%lld,%lld] for '%s' — ignoring\n",
                            (long long)ie.ne0, (long long)ie.ne1,
                            (long long)orig_ne0, (long long)orig_ne1, name);
                } else if (needs_padding) {
                    // Pad imatrix with 1.0 along the padded columns so quantization
                    // doesn't underweight them (the tensor data itself is zero-padded).
                    if (imatrix_per_weight) {
                        padded_imatrix.assign((size_t)q_n_per_row * q_nrows, 1.0f);
                        for (int64_t r = 0; r < orig_ne1; ++r) {
                            for (int64_t c = 0; c < orig_ne0; ++c) {
                                padded_imatrix[r * q_n_per_row + c] = ie.data[r * orig_ne0 + c];
                            }
                        }
                    } else {
                        padded_imatrix.assign((size_t)q_n_per_row, 1.0f);
                        for (int64_t c = 0; c < orig_ne0; ++c) {
                            padded_imatrix[c] = ie.data[c];
                        }
                    }
                    imatrix_ptr = padded_imatrix.data();
                } else {
                    imatrix_ptr = ie.data.data();
                }
            }

            quantize_tensor_with_imatrix(type_out, data_to_quantize->data(), info.data.data(),
                                         q_nrows, q_n_per_row, imatrix_ptr, imatrix_per_weight);

            printf("  %s: [", name);
            for (int d = 0; d < n_dims; d++) {
                printf("%lld%s", (long long)tensor_in->ne[d], d < n_dims - 1 ? ", " : "");
            }
            printf("] %s -> %s", ggml_type_name(type_in), ggml_type_name(type_out));

            if (needs_padding) {
                printf(" (padded %lld->%lld, %.1fx smaller)",
                       (long long)tensor_in->ne[0], (long long)padded_ne0,
                       (float)tensor_size_in / bytes_out);
            } else {
                printf(" (%.1fx smaller)", (float)tensor_size_in / bytes_out);
            }
            if (imatrix_ptr) {
                printf(" [imatrix: %s]", imatrix_per_weight ? "per-weight" : "per-column");
            } else if (!imatrix_map.empty()) {
                printf(" [imatrix: missing]");
            }
            printf("\n");

            n_quantized++;
            output_size += bytes_out;
        } else {
            // Keep original data
            info.data = std::move(data_in);
            info.type = type_in;

            if (skip) {
                printf("  %s: [", name);
                for (int d = 0; d < n_dims; d++) {
                    printf("%lld%s", (long long)tensor_in->ne[d], d < n_dims - 1 ? ", " : "");
                }
                printf("] %s (kept - skip pattern)\n", ggml_type_name(type_in));
            } else if (!do_quantize && !enable_padding) {
                printf("  %s: [", name);
                for (int d = 0; d < n_dims; d++) {
                    printf("%lld%s", (long long)tensor_in->ne[d], d < n_dims - 1 ? ", " : "");
                }
                printf("] %s (kept - incompatible dimensions, use --pad)\n", ggml_type_name(type_in));
            }

            n_skipped++;
            output_size += tensor_size_in;
        }

        tensor_infos.push_back(std::move(info));
    }

    fin.close();

    // Create output context with proper memory allocation
    printf("\nCreating output file...\n");

    // Calculate memory needed for all tensor data
    size_t total_data_size = 0;
    for (const auto &info : tensor_infos) {
        total_data_size += info.data.size();
        // Add padding for alignment
        total_data_size += 32;
    }

    // Create ggml context with enough memory for tensor metadata AND data
    size_t ctx_size = ggml_tensor_overhead() * tensor_infos.size() + total_data_size + 16 * 1024 * 1024;
    printf("Allocating %.1f MB for tensor context...\n", ctx_size / (1024.0 * 1024.0));

    struct ggml_init_params ctx_params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,  // We need allocation
    };
    struct ggml_context *ctx_data = ggml_init(ctx_params);

    // Create output GGUF context
    struct gguf_context *ctx_out = gguf_init_empty();

    // Copy metadata
    for (int i = 0; i < n_kv; i++) {
        const char *key = gguf_get_key(ctx_in, i);
        enum gguf_type type = gguf_get_kv_type(ctx_in, i);

        switch (type) {
            case GGUF_TYPE_UINT8:
                gguf_set_val_u8(ctx_out, key, gguf_get_val_u8(ctx_in, i));
                break;
            case GGUF_TYPE_INT8:
                gguf_set_val_i8(ctx_out, key, gguf_get_val_i8(ctx_in, i));
                break;
            case GGUF_TYPE_UINT16:
                gguf_set_val_u16(ctx_out, key, gguf_get_val_u16(ctx_in, i));
                break;
            case GGUF_TYPE_INT16:
                gguf_set_val_i16(ctx_out, key, gguf_get_val_i16(ctx_in, i));
                break;
            case GGUF_TYPE_UINT32:
                gguf_set_val_u32(ctx_out, key, gguf_get_val_u32(ctx_in, i));
                break;
            case GGUF_TYPE_INT32:
                gguf_set_val_i32(ctx_out, key, gguf_get_val_i32(ctx_in, i));
                break;
            case GGUF_TYPE_FLOAT32:
                gguf_set_val_f32(ctx_out, key, gguf_get_val_f32(ctx_in, i));
                break;
            case GGUF_TYPE_BOOL:
                gguf_set_val_bool(ctx_out, key, gguf_get_val_bool(ctx_in, i));
                break;
            case GGUF_TYPE_STRING:
                gguf_set_val_str(ctx_out, key, gguf_get_val_str(ctx_in, i));
                break;
            case GGUF_TYPE_ARRAY: {
                enum gguf_type arr_type = gguf_get_arr_type(ctx_in, i);
                int arr_n = gguf_get_arr_n(ctx_in, i);

                if (arr_type == GGUF_TYPE_FLOAT32) {
                    const float *arr = (const float *)gguf_get_arr_data(ctx_in, i);
                    gguf_set_arr_data(ctx_out, key, arr_type, arr, arr_n);
                } else if (arr_type == GGUF_TYPE_INT32) {
                    const int32_t *arr = (const int32_t *)gguf_get_arr_data(ctx_in, i);
                    gguf_set_arr_data(ctx_out, key, arr_type, arr, arr_n);
                }
                break;
            }
            default:
                break;
        }
    }

    // Add padding metadata if any tensors were padded
    if (n_padded > 0) {
        printf("\nStoring padding metadata for %d tensors...\n", n_padded);
        gguf_set_val_u32(ctx_out, "quantize.padded_tensor_count", n_padded);

        // Store original ne[0] for each padded tensor
        for (const auto &info : tensor_infos) {
            if (info.original_ne0 > 0) {
                std::string key = "quantize.original_ne0." + info.name;
                gguf_set_val_u32(ctx_out, key.c_str(), (uint32_t)info.original_ne0);
            }
        }
    }

    // Create tensors and add to output context
    for (auto &info : tensor_infos) {
        struct ggml_tensor *tensor = ggml_new_tensor(ctx_data, info.type, info.n_dims, info.ne);
        ggml_set_name(tensor, info.name.c_str());

        // Copy data to tensor
        memcpy(tensor->data, info.data.data(), info.data.size());

        gguf_add_tensor(ctx_out, tensor);
    }

    // Write to file
    printf("Writing output file...\n");
    gguf_write_to_file(ctx_out, output_path, false);

    // Get actual output file size
    std::ifstream check_file(output_path, std::ios::binary | std::ios::ate);
    size_t actual_output_size = check_file.tellg();
    check_file.close();

    // Summary
    printf("\n");
    printf("========================================\n");
    printf("Quantization complete!\n");
    printf("========================================\n");
    printf("Tensors quantized: %d\n", n_quantized);
    printf("Tensors padded:    %d\n", n_padded);
    printf("Tensors kept:      %d\n", n_skipped);
    printf("Input size:        %.1f MB\n", input_size / (1024.0 * 1024.0));
    printf("Output size:       %.1f MB\n", actual_output_size / (1024.0 * 1024.0));
    printf("Reduction:         %.1f%%\n", (1.0 - (double)actual_output_size / input_size) * 100.0);

    if (n_padded > 0) {
        printf("\nNote: %d tensor(s) were padded to align with block size %d.\n", n_padded, quant_type->block_size);
        printf("      Original dimensions are stored in metadata.\n");
        printf("      Inference code must unpad these tensors after loading.\n");
    }

    // Cleanup
    gguf_free(ctx_out);
    gguf_free(ctx_in);
    ggml_free(ctx_data);
    ggml_free(ctx_meta);

    return 0;
}
