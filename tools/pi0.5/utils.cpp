#include "utils.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION

#include "stb/stb_image.h"
#include "stb/stb_image_resize2.h"
#include "stb/stb_image_write.h"

#include <algorithm>
#include <cstring>

// Helper function to resize with padding (preserves aspect ratio, pads with black)
// Same as OpenPI's resize_with_pad function
static void resize_with_pad(const uint8_t *src, int src_w, int src_h,
                            uint8_t *dst, int dst_w, int dst_h) {
    // Calculate resize ratio to fit within target while preserving aspect ratio
    float ratio = std::max((float)src_w / dst_w, (float)src_h / dst_h);
    int resized_w = (int)(src_w / ratio);
    int resized_h = (int)(src_h / ratio);

    // Ensure resized dimensions don't exceed target
    resized_w = std::min(resized_w, dst_w);
    resized_h = std::min(resized_h, dst_h);

    // Allocate temporary buffer for resized image
    std::vector<uint8_t> resized(resized_w * resized_h * 3);
    stbir_resize_uint8_linear(src, src_w, src_h, 0,
                              resized.data(), resized_w, resized_h, 0, STBIR_RGB);

    // Calculate padding
    int pad_h0 = (dst_h - resized_h) / 2;
    int pad_w0 = (dst_w - resized_w) / 2;

    // Fill destination with black (padding)
    std::memset(dst, 0, dst_w * dst_h * 3);

    // Copy resized image to center of destination
    for (int y = 0; y < resized_h; y++) {
        for (int x = 0; x < resized_w; x++) {
            int src_idx = (y * resized_w + x) * 3;
            int dst_idx = ((y + pad_h0) * dst_w + (x + pad_w0)) * 3;
            dst[dst_idx + 0] = resized[src_idx + 0];
            dst[dst_idx + 1] = resized[src_idx + 1];
            dst[dst_idx + 2] = resized[src_idx + 2];
        }
    }
}

bool resize_normalize(const std::string &img_path, int target_h, int target_w,
                      const std::vector<float> &mean, const std::vector<float> &std,
                      std::vector<float> &out, bool to_chw) {
    int width, height, channels;
    uint8_t *img_data = stbi_load(img_path.c_str(), &width, &height, &channels, 3);
    if (!img_data) {
        std::string error_msg = stbi_failure_reason();
        printf("%s: Failed to decode image %s: %s\n", __func__, img_path.c_str(), error_msg.c_str());
        return false;
    }

    // Use resize_with_pad to preserve aspect ratio (same as OpenPI)
    std::vector<uint8_t> resized(target_w * target_h * 3);
    resize_with_pad(img_data, width, height, resized.data(), target_w, target_h);
    stbi_image_free(img_data);

    out.resize(target_w * target_h * 3);
    if (to_chw) {
        for (int i = 0; i < target_h; ++i) {
            for (int j = 0; j < target_w; ++j) {
                for (int c = 0; c < 3; ++c) {
                    int dst_idx = c * (target_w * target_h) + i * target_w + j;
                    int idx = i * target_w * 3 + j * 3 + c;
                    float val = resized[idx] / 255.0f;
                    out[dst_idx] = (val - mean[c]) / std[c];
                }
            }
        }
    } else {
        for (int i = 0; i < target_w * target_h; ++i) {
            for (int c = 0; c < 3; c++) {
                int idx = i * 3 + c;
                float val = resized[idx] / 255.0f;
                out[idx] = (val - mean[c]) / std[c];
            }
        }
    }
    return true;
}

bool resize_normalize(const uint8_t *img_data, int width, int height,
                      int target_h, int target_w, const std::vector<float> &mean,
                      const std::vector<float> &std, std::vector<float> &out, bool to_chw) {
    if (!img_data) {
        printf("%s: Invalid image data pointer\n", __func__);
        return false;
    }

    // Use resize_with_pad to preserve aspect ratio (same as OpenPI)
    std::vector<uint8_t> resized(target_w * target_h * 3);
    resize_with_pad(img_data, width, height, resized.data(), target_w, target_h);

    out.resize(target_w * target_h * 3);
    if (to_chw) {
        for (int i = 0; i < target_h; ++i) {
            for (int j = 0; j < target_w; ++j) {
                for (int c = 0; c < 3; ++c) {
                    int dst_idx = c * (target_w * target_h) + i * target_w + j;
                    int idx = i * target_w * 3 + j * 3 + c;
                    float val = resized[idx] / 255.0f;
                    out[dst_idx] = (val - mean[c]) / std[c];
                }
            }
        }
    } else {
        for (int i = 0; i < target_w * target_h; ++i) {
            for (int c = 0; c < 3; c++) {
                int idx = i * 3 + c;
                float val = resized[idx] / 255.0f;
                out[idx] = (val - mean[c]) / std[c];
            }
        }
    }
    return true;
}

std::vector<std::string> split_text(const std::string &input, const std::string &delimiter) {
    std::vector<std::string> result;
    if (input.empty()) {
        return result;
    }
    size_t start = 0;
    size_t pos = 0;
    while ((pos = input.find(delimiter, start)) != std::string::npos) {
        if (pos > start) {
            result.push_back(input.substr(start, pos - start));
        }
        result.push_back(delimiter);
        start = pos + delimiter.length();
    }
    if (start < input.length()) {
        result.push_back(input.substr(start));
    }
    return result;
}

std::string string_format(const char *fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX);
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), buf.size());
}

ggml_tensor *get_inp_tensor(ggml_cgraph *gf, const char *name) {
    ggml_tensor *inp = ggml_graph_get_tensor(gf, name);
    if (inp == nullptr) {
        GGML_ABORT("Failed to get tensor %s", name);
    }
    if (!(inp->flags & GGML_TENSOR_FLAG_INPUT)) {
        GGML_ABORT("Tensor %s is not an input tensor", name);
    }
    return inp;
}

void set_input_f32(ggml_cgraph *gf, const char *name, const std::vector<float> &values) {
    ggml_tensor *cur = get_inp_tensor(gf, name);
    GGML_ASSERT(cur->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(cur) == (int64_t)values.size());
    ggml_backend_tensor_set(cur, values.data(), 0, ggml_nbytes(cur));
}

void set_input_i32(ggml_cgraph *gf, const char *name, const std::vector<int32_t> &values) {
    ggml_tensor *cur = get_inp_tensor(gf, name);
    GGML_ASSERT(cur->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_nelements(cur) == (int64_t)values.size());
    ggml_backend_tensor_set(cur, values.data(), 0, ggml_nbytes(cur));
}

std::vector<ggml_fp16_t> float_to_fp16(const std::vector<float> &values) {
    std::vector<ggml_fp16_t> result(values.size());
    ggml_fp32_to_fp16_row(values.data(), result.data(), values.size());
    return result;
}

std::vector<float> fp16_to_float(const std::vector<ggml_fp16_t> &values) {
    std::vector<float> result(values.size());
    ggml_fp16_to_fp32_row(values.data(), result.data(), values.size());
    return result;
}

void set_input_f16(ggml_cgraph *gf, const char *name, const std::vector<float> &values) {
    ggml_tensor *cur = get_inp_tensor(gf, name);
    GGML_ASSERT(cur->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_nelements(cur) == (int64_t)values.size());
    std::vector<ggml_fp16_t> fp16_values = float_to_fp16(values);
    ggml_backend_tensor_set(cur, fp16_values.data(), 0, ggml_nbytes(cur));
}

void get_output_f16_to_float(ggml_tensor *tensor, std::vector<float> &out) {
    GGML_ASSERT(tensor->type == GGML_TYPE_F16);
    std::vector<ggml_fp16_t> fp16_data(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, fp16_data.data(), 0, ggml_nbytes(tensor));
    out = fp16_to_float(fp16_data);
}

bool set_backend_threads(ggml_backend_t backend, int n_threads) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (reg) {
        auto ggml_backend_set_n_threads_fn =
            (ggml_backend_set_n_threads_t)ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_set_n_threads");
        if (ggml_backend_set_n_threads_fn) {
            ggml_backend_set_n_threads_fn(backend, n_threads);
        }
    }
    return true;
}

ggml_tensor *get_tensor(ggml_context *ctx_meta, const std::string &name,
                        std::vector<ggml_tensor *> &tensors, bool required, bool save) {
    ggml_tensor *cur = ggml_get_tensor(ctx_meta, name.c_str());
    if (!cur && required) {
        throw std::runtime_error(string_format("%s: unable to find tensor %s\n",
                                               __func__, name.c_str()));
    }
    if (cur && save) {
        tensors.push_back(cur);
    }
    return cur;
}
