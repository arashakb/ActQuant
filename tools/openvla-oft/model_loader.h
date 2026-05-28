#pragma once

#include "ggml-cpp.h"
#include "utils.h"
#include <string>

class ModelLoader {
public:
  ModelLoader(const std::string &fname);
  void get_bool(const std::string &key, bool &output,
                bool required = true) const;
  void get_i32(const std::string &key, int32_t &output, bool required = true) const;
  void get_u32(const std::string &key, uint32_t &output, bool required = true) const;

  void get_f32(const std::string &key, float &output,
               bool required = true) const;
  void get_string(const std::string &key, std::string &output,
                  bool required = true) const;
  void get_arr_int(const std::string &key, std::vector<int> &output,
                   bool required = true) const;

  // private:
  std::string fname_;
  size_t model_size_ = 0; // in bytes
  gguf_context_ptr ctx_gguf_ = nullptr;
  ggml_context_ptr ctx_meta_ = nullptr;
};
