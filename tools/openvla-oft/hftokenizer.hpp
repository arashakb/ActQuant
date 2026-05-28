#pragma once

#include "tokenizers_cpp.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
// #include <sys/_types/_int32_t.h>

class HfTokenizer {

public:
  HfTokenizer() = default;
  ~HfTokenizer() { tokenizer_.reset(); }

  bool init(const std::string &tokenizer_path, uint32_t max_length) {
    try {
      max_length_ = max_length;
      auto blob = LoadBytesFromFile(tokenizer_path);
      tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(blob);
      if (!tokenizer_) {
        std::cerr << "Failed to load tokenizer from " << tokenizer_path
                  << std::endl;
        return false;
      }
    } catch (std::exception &e) {
      std::cerr << "Failed to load tokenizer from " << tokenizer_path
                << std::endl;
      std::cerr << e.what() << std::endl;
      return false;
    }
    set_eos("</s>");
    set_pad("<PAD>");
    return true;
  }

  void set_eos(const std::string &content = "</s>") {
    set_eos(tokenizer_->TokenToId(content));
  }

  void set_eos(int32_t eos) { eos_ = eos; }

  void set_pad(const std::string &content = "<PAD>") {
    set_pad(tokenizer_->TokenToId(content));
  }
  void set_pad(int32_t pad) { pad_ = pad; }

  inline int32_t eos() { return eos_; }
  inline int32_t pad() { return pad_; }

  std::vector<int32_t>
  encode(const std::string &prompt_in,
         bool add_special = false) { //, bool pad_max = false) {
    std::vector<int32_t> ids = tokenizer_->Encode(prompt_in, add_special);
    if (ids.size() > max_length_ - 1)
      ids.resize(max_length_ - 1);
    // if (pad_max)
    //   ids.resize(max_length_, pad_);
    return ids;
  }

  std::string decode(const std::vector<int> &ids) {
    return tokenizer_->Decode(ids);
  }

private:
  std::shared_ptr<tokenizers::Tokenizer> tokenizer_{nullptr};
  int32_t eos_, pad_;
  uint32_t max_length_;
  inline std::string LoadBytesFromFile(const std::string &path) {
    std::ifstream fs(path, std::ios::in | std::ios::binary);
    if (fs.fail()) {
      std::cerr << "Cannot open " << path << std::endl;
      throw std::runtime_error("Failed to open file: " + path);
    }
    std::string data;
    fs.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(fs.tellg());
    fs.seekg(0, std::ios::beg);
    data.resize(size);
    fs.read(data.data(), size);
    return data;
  }
};

inline std::unique_ptr<HfTokenizer> create_tokenizer(const std::string &tokenizer_path,
                                              uint32_t max_length = 1024) {
  std::unique_ptr<HfTokenizer> tokenizer = std::make_unique<HfTokenizer>();
  if (!tokenizer->init(tokenizer_path, max_length)) {
    std::cerr << "Failed to initialize tokenizer" << std::endl;
    return nullptr;
  }
  return tokenizer;
}