#include "model_loader.h"
#include "ggml.h"
#include "gguf.h"
#include "utils.h"
#include <inttypes.h>

ModelLoader::ModelLoader(const std::string &fname) : fname_(fname) {
  // 初始化gguf_context和对应的meta ggml_context
  ggml_context *meta = nullptr;
  gguf_init_params params = {
      .no_alloc = true,
      .ctx = &meta,
  };
  ctx_gguf_.reset(gguf_init_from_file(fname.c_str(), params));
  if (!ctx_gguf_) {
    throw std::runtime_error(string_format(
        "%s: failed to load model from %s. Does this file exist?\n", __func__,
        fname.c_str()));
  }
  ctx_meta_.reset(meta);

  // const int n_tensors = gguf_get_n_tensors(ctx_gguf_.get());
  // tensors
  // {
  //   for (int i = 0; i < n_tensors; ++i) {
  //     const char *name = gguf_get_tensor_name(ctx_gguf_.get(), i);
  //     const size_t offset = gguf_get_tensor_offset(ctx_gguf_.get(), i);
  //     enum ggml_type type = gguf_get_tensor_type(ctx_gguf_.get(), i);
  //     ggml_tensor *cur = ggml_get_tensor(meta, name);
  //     size_t tensor_size = ggml_nbytes(cur);
  //     model_size_ += tensor_size;
  //     printf("%s: tensor[%d]: n_dims = %d, name = %s, tensor_size=%zu, "
  //            "offset=%zu, shape:[%" PRIu64 ", %" PRIu64 ", %" PRIu64
  //            ", %" PRIu64 "], type = %s\n",
  //            __func__, i, ggml_n_dims(cur), cur->name, tensor_size, offset,
  //            cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3],
  //            ggml_type_name(type));
  //   }
  // }
}

void ModelLoader::get_bool(const std::string &key, bool &output,
                           bool required) const {
  const int i = gguf_find_key(ctx_gguf_.get(), key.c_str());
  if (i < 0) {
    if (required) {
      throw std::runtime_error("Key not found: " + key);
    }
    return;
  }
  output = gguf_get_val_bool(ctx_gguf_.get(), i);
}

void ModelLoader::get_i32(const std::string &key, int32_t &output,
                          bool required) const {
  const int key_idx = gguf_find_key(ctx_gguf_.get(), key.c_str());
  if (key_idx < 0) {
    if (required) {
      throw std::runtime_error("Key not found: " + key);
    }
    return;
  }
  output = gguf_get_val_i32(ctx_gguf_.get(), key_idx);
}

void ModelLoader::get_u32(const std::string &key, uint32_t &output,
                          bool required) const {
  const int key_idx = gguf_find_key(ctx_gguf_.get(), key.c_str());
  if (key_idx < 0) {
    if (required) {
      throw std::runtime_error("Key not found: " + key);
    }
    return;
  }
  output = gguf_get_val_u32(ctx_gguf_.get(), key_idx);
}

void ModelLoader::get_f32(const std::string &key, float &output,
                          bool required) const {
  const int i = gguf_find_key(ctx_gguf_.get(), key.c_str());
  if (i < 0) {
    if (required) {
      throw std::runtime_error("Key not found: " + key);
    }
    return;
  }
  output = gguf_get_val_f32(ctx_gguf_.get(), i);
}

void ModelLoader::get_string(const std::string &key, std::string &output,
                             bool required) const {
  const int i = gguf_find_key(ctx_gguf_.get(), key.c_str());
  if (i < 0) {
    if (required) {
      throw std::runtime_error("Key not found: " + key);
    }
    return;
  }
  output = std::string(gguf_get_val_str(ctx_gguf_.get(), i));
}

void ModelLoader::get_arr_int(const std::string &key, std::vector<int> &output,
                              bool required) const {
  const int key_idx = gguf_find_key(ctx_gguf_.get(), key.c_str());
  if (key_idx < 0) {
    if (required) {
      throw std::runtime_error("Key not found: " + key);
    }
    return;
  }
  int n = gguf_get_arr_n(ctx_gguf_.get(), key_idx);
  output.resize(n);
  const int32_t *values =
      (const int32_t *)gguf_get_arr_data(ctx_gguf_.get(), key_idx);
  for (int idx = 0; idx < n; ++idx) {
    output[idx] = values[idx];
  }
}