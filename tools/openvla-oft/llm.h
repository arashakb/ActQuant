#pragma once

#include "hftokenizer.hpp"
#include "llama.h"
#include "utils.h"
#include <string>

class Llm {
public:
  Llm();

  bool load_model(const std::string &model_path, LlmParam params);

  virtual std::string format_prompt(const std::string &prompt,
                                    const std::string &system_prompt = "");

  bool encode_text(const std::string &prompt,
                   std::vector<llama_token> &prompt_tokens, bool add_special);
  bool eval_chunk(llama_token *tokens, float *embd, int n_tokens, bool is_last);
  virtual std::string generate(const std::string &prompt,
                               const std::string &system_prompt,
                               std::vector<llama_token> &generated_tokens,
                               std::vector<float> &embd,
                               bool use_history = true);
  // const float *get_last_hidden_state(int &n_embd) const;
  bool get_last_hidden_state(std::vector<float>& output) const;
  bool get_last_hidden_state(std::vector<float>& output, int32_t i) const;
  bool get_hidden_state_at(int32_t token_index, std::vector<float>& output) const;
  bool get_last_logit(std::vector<float>& output) const;
  virtual ~Llm();

  bool init_tokenizer(const std::string &tokenizer_path);
  bool encode_text_by_tokenizer_cpp(const std::string &prompt,
                                    std::vector<llama_token> &prompt_tokens,
                                    bool add_special);

protected:
  llama_model *model_ = nullptr;
  const llama_vocab *vocab_ = nullptr;
  llama_context *ctx_ = nullptr;
  llama_sampler *smpl_ = nullptr;
  int n_ctx_ = 2048;
  std::unique_ptr<HfTokenizer> tokenizer_;
  bool require_embeddings_ = false;

  Llm(const Llm &) = delete;
  Llm &operator=(const Llm &) = delete;
};

class OpenvlaLlm : public Llm {
public:
  std::string format_prompt(const std::string &prompt,
                            const std::string &system_prompt = "") override;
  bool generate(const std::string &prompt, float *img_emb, int32_t n_img_tokens,
                std::vector<llama_token> &generated_tokens,
                std::vector<float> &output, bool use_history = false);

  inline void set_empty_token(llama_token empty_token) {
    empty_token_ = empty_token;
  }

  llama_token empty_token_ = 29871; //29871 for llama2 and 220 for llama3.2; // default
  int pad_to_multiple_of_ = 64;
};

// OpenVLA-OFT specific LLM class
// Key differences from OpenvlaLlm:
// 1. No token generation - only prefill with placeholder action tokens
// 2. Extracts hidden states from ALL action token positions
// 3. Uses non-causal (bidirectional) attention for action tokens
class OpenvlaOFTLlm : public Llm {
public:
  std::string format_prompt(const std::string &prompt,
                            const std::string &system_prompt = "") override;

  // OFT-specific forward pass (legacy - uses segmented decode with causal attention)
  // Processes prompt + vision + action placeholders and extracts hidden states
  // from all action token positions
  // Returns: hidden states flattened as (num_actions_chunk * action_dim * hidden_dim)
  bool forward_oft(const std::string &prompt, float *img_emb,
                   int32_t n_img_tokens, int32_t num_action_tokens,
                   std::vector<float> &output, bool use_history = false);

  // OFT-specific forward pass v2 - single batch with non-causal attention
  // Like Python's approach: build all embeddings, single forward pass, non-causal
  // Requires token embedding table to be loaded first via load_token_embeddings()
  //
  // Parameters:
  //   prompt: task description
  //   img_emb: vision embeddings from projector (n_img_tokens * n_embd)
  //   n_img_tokens: number of vision tokens (256 for single image, 512 for dual)
  //   proprio_emb: proprio embedding from projector (n_embd), or nullptr if not using proprio
  //   num_action_tokens: number of action tokens (typically 56 = 8 chunks * 7 dims)
  //   output: extracted hidden states from action token positions
  bool forward_oft_v2(const std::string &prompt, float *img_emb,
                      int32_t n_img_tokens, float *proprio_emb,
                      int32_t num_action_tokens,
                      std::vector<float> &output, bool use_history = false);

  // Load token embedding table from LLM GGUF file
  // This is needed for forward_oft_v2 to convert tokens to embeddings
  bool load_token_embeddings(const std::string &gguf_path);

  // Convert tokens to embeddings using loaded token embedding table
  bool tokens_to_embeddings(const std::vector<llama_token> &tokens,
                            std::vector<float> &out) const;

  // Extract hidden states from a range of positions
  bool get_hidden_states_range(int32_t start_idx, int32_t end_idx,
                               std::vector<float> &output) const;

  inline void set_empty_token(llama_token empty_token) {
    empty_token_ = empty_token;
  }

  inline void set_stop_token(llama_token stop_token) {
    stop_token_ = stop_token;
  }

  // Get embedding dimension
  int32_t get_n_embd() const { return n_embd_; }

  llama_token empty_token_ = 29871;  // 29871 for llama2, 220 for llama3.2
  llama_token stop_token_ = 2;       // </s> token
  int pad_to_multiple_of_ = 64;

private:
  // Token embedding table (loaded from LLM GGUF)
  std::vector<float> token_embeddings_;
  int32_t vocab_size_ = 0;
  int32_t n_embd_ = 0;
  bool token_embd_loaded_ = false;
};
