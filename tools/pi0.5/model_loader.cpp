#include "model_loader.h"
#include "ggml.h"
#include "gguf.h"
#include "utils.h"

ModelLoader::ModelLoader(const std::string &fname) : fname_(fname) {
    ggml_context *meta = nullptr;
    gguf_init_params params = {
        .no_alloc = true,
        .ctx = &meta,
    };
    ctx_gguf_.reset(gguf_init_from_file(fname.c_str(), params));
    if (!ctx_gguf_) {
        throw std::runtime_error(string_format(
            "%s: failed to load model from %s\n", __func__, fname.c_str()));
    }
    ctx_meta_.reset(meta);
}

void ModelLoader::get_bool(const std::string &key, bool &output, bool required) const {
    const int i = gguf_find_key(ctx_gguf_.get(), key.c_str());
    if (i < 0) {
        if (required) throw std::runtime_error("Key not found: " + key);
        return;
    }
    output = gguf_get_val_bool(ctx_gguf_.get(), i);
}

void ModelLoader::get_i32(const std::string &key, int &output, bool required) const {
    const int i = gguf_find_key(ctx_gguf_.get(), key.c_str());
    if (i < 0) {
        if (required) throw std::runtime_error("Key not found: " + key);
        return;
    }
    output = gguf_get_val_i32(ctx_gguf_.get(), i);
}

void ModelLoader::get_u32(const std::string &key, int &output, bool required) const {
    const int i = gguf_find_key(ctx_gguf_.get(), key.c_str());
    if (i < 0) {
        if (required) throw std::runtime_error("Key not found: " + key);
        return;
    }
    output = gguf_get_val_u32(ctx_gguf_.get(), i);
}

void ModelLoader::get_f32(const std::string &key, float &output, bool required) const {
    const int i = gguf_find_key(ctx_gguf_.get(), key.c_str());
    if (i < 0) {
        if (required) throw std::runtime_error("Key not found: " + key);
        return;
    }
    output = gguf_get_val_f32(ctx_gguf_.get(), i);
}

void ModelLoader::get_string(const std::string &key, std::string &output, bool required) const {
    const int i = gguf_find_key(ctx_gguf_.get(), key.c_str());
    if (i < 0) {
        if (required) throw std::runtime_error("Key not found: " + key);
        return;
    }
    output = std::string(gguf_get_val_str(ctx_gguf_.get(), i));
}

void ModelLoader::get_arr_int(const std::string &key, std::vector<int> &output, bool required) const {
    const int i = gguf_find_key(ctx_gguf_.get(), key.c_str());
    if (i < 0) {
        if (required) throw std::runtime_error("Key not found: " + key);
        return;
    }
    int n = gguf_get_arr_n(ctx_gguf_.get(), i);
    output.resize(n);
    const int32_t *values = (const int32_t *)gguf_get_arr_data(ctx_gguf_.get(), i);
    for (int j = 0; j < n; ++j) {
        output[j] = values[j];
    }
}
