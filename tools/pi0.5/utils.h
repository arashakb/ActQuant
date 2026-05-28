#pragma once

#include "ggml-backend.h"
#include "ggml.h"
#include <fstream>
#include <iostream>
#include <vector>

struct ContextParams {
    std::string device_name = "CPU";
    int n_threads = 1;
    int max_nodes = GGML_DEFAULT_GRAPH_SIZE;
    enum ggml_log_level verbosity = GGML_LOG_LEVEL_INFO;
};

template <typename T>
bool load_file_to_vector(std::vector<T> &vec, const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        file.seekg(0, std::ios::end);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::size_t num_elements = size / sizeof(T);
        vec.clear();
        vec.resize(num_elements);
        if (!file.read(reinterpret_cast<char *>(vec.data()), size)) {
            return false;
        }
        return true;
    }
    return false;
}

template <typename T>
bool save_vector_to_file(const std::vector<T> &vec, const std::string &filename) {
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char *>(vec.data()), vec.size() * sizeof(T));
        file.close();
        return true;
    }
    return false;
}

std::vector<std::string> split_text(const std::string &input, const std::string &delimiter);
std::string string_format(const char *fmt, ...);

ggml_tensor *get_inp_tensor(ggml_cgraph *gf, const char *name);
void set_input_f32(ggml_cgraph *gf, const char *name, const std::vector<float> &values);
void set_input_f16(ggml_cgraph *gf, const char *name, const std::vector<float> &values);
void set_input_i32(ggml_cgraph *gf, const char *name, const std::vector<int32_t> &values);

std::vector<ggml_fp16_t> float_to_fp16(const std::vector<float> &values);
std::vector<float> fp16_to_float(const std::vector<ggml_fp16_t> &values);
void get_output_f16_to_float(ggml_tensor *tensor, std::vector<float> &out);

bool set_backend_threads(ggml_backend_t backend, int n_threads);

ggml_tensor *get_tensor(ggml_context *ctx_meta, const std::string &name,
                        std::vector<ggml_tensor *> &tensors,
                        bool required = true, bool save = true);

bool resize_normalize(const std::string &img_path, int target_h, int target_w,
                      const std::vector<float> &mean, const std::vector<float> &std,
                      std::vector<float> &out, bool to_chw = true);

bool resize_normalize(const uint8_t *img_data, int width, int height,
                      int target_h, int target_w, const std::vector<float> &mean,
                      const std::vector<float> &std, std::vector<float> &out,
                      bool to_chw = true);
