#include "llm.h"
#include "llama.h"
#include "utils.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <string.h>
#include "ggml.h"
#include "gguf.h"

static std::string common_token_to_piece(const llama_vocab *vocab,
                                         llama_token token, int32_t lstrip = 0,
                                         bool special = true) {
  char buf[256];
  int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), lstrip, special);
  if (n < 0) {
    GGML_ABORT("failed to convert token to piece\n");
  }
  std::string piece(buf, n);
  printf("%s", piece.c_str());
  fflush(stdout);
  return piece;
}
Llm::Llm() {
  // only print errors
  llama_log_set(
      [](enum ggml_log_level level, const char *text, void * /* user_data */) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
          fprintf(stderr, "%s", text);
        }
      },
      nullptr);

  // load dynamic backends
  ggml_backend_load_all();
}

bool Llm::load_model(const std::string &model_path, LlmParam params) {
  // initialize the model
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = params.ngl;
  model_ = llama_model_load_from_file(model_path.c_str(), model_params);
  if (!model_) {
    fprintf(stderr, "%s: error: unable to load model\n", __func__);
    return false;
  }
  vocab_ = llama_model_get_vocab(model_);

  // initialize the context
  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = params.n_ctx;
  ctx_params.n_batch = params.n_ctx;
  // n_ubatch must be >= n_tokens for non-causal attention to work
  ctx_params.n_ubatch = params.n_ctx;
  ctx_params.embeddings = params.embeddings;
  ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
  ctx_params.type_k = GGML_TYPE_F16;
  ctx_params.type_v = GGML_TYPE_F16;
  require_embeddings_ = params.embeddings;
  ctx_ = llama_init_from_model(model_, ctx_params);
  if (!ctx_) {
    fprintf(stderr, "%s: error: failed to create the llama_context\n",
            __func__);
    return false;
  }
  n_ctx_ = llama_n_ctx(ctx_);

  // initialize the sampler
  smpl_ = llama_sampler_chain_init(llama_sampler_chain_default_params());
  llama_sampler_chain_add(smpl_, llama_sampler_init_greedy());
  // llama_sampler_chain_add(smpl_, llama_sampler_init_min_p(0.05f, 1));
  // llama_sampler_chain_add(smpl_, llama_sampler_init_temp(0.8f));
  // llama_sampler_chain_add(smpl_,
  // llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

  if (!params.tokenizer_path.empty()) {
    init_tokenizer(params.tokenizer_path);
  }
  printf("load model %s\n", model_path.c_str());
  return true;
}

bool Llm::encode_text_by_tokenizer_cpp(const std::string &prompt,
                                       std::vector<llama_token> &prompt_tokens,
                                       bool add_special) {
  // tokenize the prompt
  prompt_tokens = tokenizer_->encode(prompt, add_special);
  return true;
}

std::string Llm::format_prompt(const std::string &prompt,
                               const std::string &system_prompt) {
  std::vector<llama_chat_message> messages;
  std::vector<std::string> message_contents;  // Hold string ownership
  std::vector<char> formatted(llama_n_ctx(ctx_));
  const char *tmpl = llama_model_chat_template(model_, /* name */ nullptr);

  // add the user input to the message list and format it
  if (!system_prompt.empty()) {
    message_contents.push_back(system_prompt);
    messages.push_back({"system", message_contents.back().c_str()});
  }
  message_contents.push_back(prompt);
  messages.push_back({"user", message_contents.back().c_str()});

  int new_len =
      llama_chat_apply_template(tmpl, messages.data(), messages.size(), true,
                                formatted.data(), formatted.size());
  if (new_len > (int)formatted.size()) {
    formatted.resize(new_len);
    new_len =
        llama_chat_apply_template(tmpl, messages.data(), messages.size(), true,
                                  formatted.data(), formatted.size());
  }
  if (new_len < 0) {
    fprintf(stderr, "failed to apply the chat template\n");
    return "";
  }
  return std::string(formatted.data(), new_len);
}

bool Llm::encode_text(const std::string &prompt,
                      std::vector<llama_token> &prompt_tokens,
                      bool add_special) {
  // tokenize the prompt
  const int n_prompt_tokens = -llama_tokenize(
      vocab_, prompt.c_str(), prompt.size(), NULL, 0, add_special, true);
  prompt_tokens.resize(n_prompt_tokens);
  if (llama_tokenize(vocab_, prompt.c_str(), prompt.size(),
                     prompt_tokens.data(), prompt_tokens.size(), add_special,
                     true) < 0) {
    GGML_ABORT("failed to tokenize the prompt\n");
  }
  return true;
}

Llm::~Llm() {
  if (smpl_) {
    llama_sampler_free(smpl_);
    smpl_ = nullptr;
  }
  if (ctx_) {
    llama_free(ctx_);
    ctx_ = nullptr;
  }
  if (model_) {
    llama_model_free(model_);
    model_ = nullptr;
  }
}

bool Llm::init_tokenizer(const std::string &tokenizer_path) {
  tokenizer_ = create_tokenizer(tokenizer_path);
  return (tokenizer_ != nullptr);
}

// 获取当前上下文中第 token_index 个 token 的最后一层 hidden state
bool get_hidden_state_at(llama_context *ctx, int32_t token_index, std::vector<float>& output) {
  int32_t n_embd = llama_n_embd(llama_get_model(ctx));
  const float* emb = llama_get_embeddings_ith(ctx, token_index);
  
  if (!emb) {
      // fallback: try global last embedding (less reliable)
      emb = llama_get_embeddings(ctx);
      if (!emb || token_index != llama_get_n_outputs(ctx) - 1) {
          fprintf(stderr, "Failed to get embedding for token %d\n", token_index);
          return false;
      }
  }

  output.assign(emb, emb + n_embd);
  return true;
}
bool Llm::eval_chunk(llama_token *tokens, float *embd, int n_tokens, bool is_last) {
  // llama_pos n_past = llama_memory_seq_pos_max(llama_get_memory(ctx_), 0) + 1;
  // std::vector<llama_pos> pos(n_tokens);
  // std::iota(pos.begin(), pos.end(), n_past);
  // std::vector<llama_seq_id> tmp_seq(n_tokens, 0);
  // std::vector<llama_seq_id *> seq_id(n_tokens, tmp_seq.data());
  // std::vector<int> n_seq_id(n_tokens, 1);

  std::vector<int8_t> logits(n_tokens, 1);
  if (is_last) {
    logits[n_tokens - 1] = true;
  }
  {
    llama_batch batch = {
        .n_tokens = n_tokens,
        .token = tokens ? tokens : nullptr,
        .embd = embd ? embd : nullptr,
        .pos = nullptr,
        .n_seq_id = nullptr,
        .seq_id = nullptr,
        .logits = logits.data(),
    };

    int ret = llama_decode(ctx_, batch);
    if (ret != 0) {
      // GGML_ABORT("failed to decode, ret = %d\n", ret);
      return false;
    }
  }
  return true;
}

bool Llm::get_last_hidden_state(std::vector<float> &output) const {
  uint32_t n_outputs = llama_get_n_outputs(ctx_);
  int32_t n_embd = llama_model_n_embd(model_);
  output.resize(n_embd);
  float *all_embeddings = llama_get_embeddings_ith(ctx_, 0);
  memcpy(output.data(), all_embeddings + (n_outputs - 1) * n_embd,
         n_embd * sizeof(float));

  return true;
}

bool Llm::get_last_hidden_state(std::vector<float>& output, int32_t i) const {
  uint32_t n_outputs = llama_get_n_outputs(ctx_);
  int32_t n_embd = llama_model_n_embd(model_);
  output.resize(n_embd);
  float *all_embeddings = llama_get_embeddings_ith(ctx_, i);
  memcpy(output.data(), all_embeddings + (n_outputs - 1) * n_embd,
         n_embd * sizeof(float));

  return true;
}

bool Llm::get_last_logit(std::vector<float> &output) const {
  uint32_t n_outputs = llama_get_n_outputs(ctx_);
  size_t n_logits = llama_vocab_n_tokens(vocab_);
  output.resize(n_logits);
  float *logits = llama_get_logits_ith(ctx_, 0);
  memcpy(output.data(), logits + (n_outputs - 1) * n_logits,
         n_logits * sizeof(float));

  return true;
}

std::string Llm::generate(const std::string &prompt,
                          const std::string &system_prompt,
                          std::vector<llama_token> &generated_tokens,
                          std::vector<float> &embeddings, bool use_history) {
  if (!use_history) {
    llama_memory_clear(llama_get_memory(ctx_), true);
    llama_synchronize(ctx_);
    llama_perf_context_reset(ctx_);
    llama_set_warmup(ctx_, false);
  }
  const bool is_first =
      llama_memory_seq_pos_max(llama_get_memory(ctx_), 0) == -1;

  // apply template
  std::string formated_prompt = format_prompt(prompt, system_prompt);
  std::vector<llama_token> prompt_tokens;
  if (!encode_text(formated_prompt, prompt_tokens, is_first)) {
    fprintf(stderr, "%s: Failed to encode text\n", __func__);
    return "";
  };

  if (!eval_chunk(prompt_tokens.data(), nullptr, prompt_tokens.size(), true)) {
    fprintf(stderr, "%s: prefill failed.\n", __func__);
    return "";
  }

  {
    std::vector<float> v_logits;
    get_last_logit(v_logits);
  }

  if (require_embeddings_) {
    get_last_hidden_state(embeddings);
    return "";
  }

  std::string response = "";
  generated_tokens.clear();
  while (true) {
    // sample the next token
    llama_token new_token_id = llama_sampler_sample(smpl_, ctx_, -1);

    // is it an end of generation?
    if (llama_vocab_is_eog(vocab_, new_token_id)) {
      printf("\n");
      fflush(stdout);
      break;
    }
    generated_tokens.push_back(new_token_id);

    std::string piece = common_token_to_piece(vocab_, new_token_id, 0, true);
    response += piece;

    if (!eval_chunk(&new_token_id, nullptr, 1, true)) {
      fprintf(stderr, "%s: decode failed.\n", __func__);
      return "";
    }
  }
  return response;
}

std::string OpenvlaLlm::format_prompt(const std::string &prompt,
                                      const std::string &system_prompt) {
  std::string formated_prompt =
      "<__media__>In: What action should the robot take to " + prompt +
      "?\nOut:";
  return formated_prompt;
}

bool Llm::get_hidden_state_at(int32_t token_index, std::vector<float>& output) const {
  int32_t n_embd = llama_n_embd(model_);
  int32_t current_seq_len = llama_get_n_outputs(ctx_);

  if (token_index < 0 || token_index >= current_seq_len) {
      fprintf(stderr, "%s: invalid token index %d (seq_len=%d)\n", 
              __func__, token_index, current_seq_len);
      return false;
  }

  const float* emb = llama_get_embeddings_ith(ctx_, token_index);
  if (!emb) {
      // Fallback to global last embedding only if requesting last token
      if (token_index == current_seq_len - 1) {
          emb = llama_get_embeddings(ctx_);
      }
      if (!emb) {
          fprintf(stderr, "%s: failed to get embedding for token %d\n", __func__, token_index);
          return false;
      }
  }

  output.assign(emb, emb + n_embd);

  return true;
}

bool OpenvlaLlm::generate(const std::string &prompt, float *img_emb,
                          int32_t n_img_tokens,
                          std::vector<llama_token> &generated_tokens,
                          std::vector<float> &output, bool use_history) {
    if (!use_history) {
        llama_memory_clear(llama_get_memory(ctx_), true);
        llama_synchronize(ctx_);
        llama_perf_context_reset(ctx_);
        llama_set_warmup(ctx_, false);
    }
    // Apply template
    std::string formated_prompt = format_prompt(prompt, "");
    std::vector<llama_token> prompt_tokens;
    std::string template_img = "<__media__>";
    std::vector<std::string> texts = split_text(formated_prompt, template_img);
    if (texts.size() > 1 && texts[0] == template_img) {
        texts.insert(texts.begin(), "");
    }

    // Prefill: process prompt and image embeddings
    for (size_t i = 0; i < texts.size(); i++) {
        llama_token *p_tokens = nullptr;
        int n_tokens = 0;
        float *embd = nullptr;
        bool is_last = (i == texts.size() - 1);
        if (texts[i] == template_img) {
            embd = img_emb;
            n_tokens = n_img_tokens;
            p_tokens = nullptr;
        } else {
            if (tokenizer_) {
                encode_text_by_tokenizer_cpp(texts[i], prompt_tokens, i == 0);
            } else {
                encode_text(texts[i], prompt_tokens, i == 0);
                if (!prompt_tokens.empty() && prompt_tokens[0] == 797) {
                    prompt_tokens[0] = 512; // for debug
                }
            }

            if (is_last && !prompt_tokens.empty() &&
                prompt_tokens.back() != empty_token_) {
                prompt_tokens.push_back(empty_token_);
            }

            p_tokens = prompt_tokens.data();
            n_tokens = prompt_tokens.size();
            embd = nullptr;
        }

        if (!eval_chunk(p_tokens, embd, n_tokens, is_last)) {
            fprintf(stderr, "%s: prefill failed.\n", __func__);
            return false;
        }
    }

    // Now generate exactly 2 tokens and collect their hidden states
    const int num_new_tokens = 1;
    generated_tokens.clear();
    std::vector<std::vector<float>> all_hidden_states;
    // If embeddings are required, extract hidden state of this newly generated token
    if (require_embeddings_) {
      int32_t current_seq_len = llama_get_n_outputs(ctx_);
      int32_t target_idx = current_seq_len - 1; // index of the token we just generated

      std::vector<float> hs;
      if (!get_hidden_state_at(target_idx, hs)) {
          fprintf(stderr, "%s: failed to get hidden state.\n", __func__);
          return false;
      }
      all_hidden_states.push_back(hs);
    }

    for (int step = 0; step < num_new_tokens; ++step) {
        // Sample next token
        llama_token new_token_id = llama_sampler_sample(smpl_, ctx_, -1);
        generated_tokens.push_back(new_token_id);

        // Decode the token (must set logits=true for last token to compute embeddings)
        if (!eval_chunk(&new_token_id, nullptr, 1, true)) {
            fprintf(stderr, "%s: decode failed at step %d.\n", __func__, step);
            return false;
        }

        // If embeddings are required, extract hidden state of this newly generated token
        if (require_embeddings_) {
            int32_t current_seq_len = llama_get_n_outputs(ctx_);
            int32_t target_idx = current_seq_len - 1; // index of the token we just generated

            std::vector<float> hs;
            if (!get_hidden_state_at(target_idx, hs)) {
                fprintf(stderr, "%s: failed to get hidden state at step %d.\n", __func__, step);
                return false;
            }
            all_hidden_states.push_back(hs);
        }
    }

    // Post-process output
    if (require_embeddings_) {
        // Flatten [hs0, hs1] into a single vector
        output.clear();
        for (const auto& hs : all_hidden_states) {
            output.insert(output.end(), hs.begin(), hs.end());
        }
    } else {
        // Optional: post-process generated_tokens (e.g., reverse mapping for OpenVLA actions)
        // Note: now only 2 tokens, not 7
        if (generated_tokens.size() >= 2) {
            size_t vocab_size = llama_vocab_n_tokens(vocab_) - pad_to_multiple_of_;
            std::vector<llama_token> predicted(generated_tokens.end() - 2, generated_tokens.end());
            std::transform(predicted.begin(), predicted.end(), predicted.begin(),
                [vocab_size](llama_token t) { return std::max<int>(vocab_size - t - 1, 0); });
            generated_tokens = predicted;
        }
    }

    return true;
}

// ==========================================
// OpenVLA-OFT specific LLM implementation
// ==========================================

std::string OpenvlaOFTLlm::format_prompt(const std::string &prompt,
                                          const std::string &system_prompt) {
  // Same format as OpenvlaLlm, but convert prompt to lowercase to match Python behavior
  // Python uses: f"In: What action should the robot take to {task_label.lower()}?\nOut:"
  std::string prompt_lower = prompt;
  std::transform(prompt_lower.begin(), prompt_lower.end(), prompt_lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  std::string formated_prompt =
      "<__media__>In: What action should the robot take to " + prompt_lower +
      "?\nOut:";
  return formated_prompt;
}

bool OpenvlaOFTLlm::get_hidden_states_range(int32_t start_idx, int32_t end_idx,
                                             std::vector<float> &output) const {
  int32_t n_embd = llama_n_embd(model_);
  int32_t current_seq_len = llama_get_n_outputs(ctx_);

  if (start_idx < 0 || end_idx > current_seq_len || start_idx >= end_idx) {
    fprintf(stderr,
            "%s: invalid range [%d, %d) (seq_len=%d)\n",
            __func__, start_idx, end_idx, current_seq_len);
    return false;
  }

  int32_t num_positions = end_idx - start_idx;
  output.resize(num_positions * n_embd);

  for (int32_t i = start_idx; i < end_idx; ++i) {
    const float *emb = llama_get_embeddings_ith(ctx_, i);
    if (!emb) {
      fprintf(stderr, "%s: failed to get embedding for token %d\n", __func__, i);
      return false;
    }
    memcpy(output.data() + (i - start_idx) * n_embd, emb, n_embd * sizeof(float));
  }

  return true;
}

bool OpenvlaOFTLlm::forward_oft(const std::string &prompt, float *img_emb,
                                 int32_t n_img_tokens, int32_t num_action_tokens,
                                 std::vector<float> &output, bool use_history) {
  if (!use_history) {
    llama_memory_clear(llama_get_memory(ctx_), true);
    llama_synchronize(ctx_);
    llama_perf_context_reset(ctx_);
    llama_set_warmup(ctx_, false);
  }

  // NOTE: OpenVLA-OFT in Python uses bidirectional attention (non-causal) for the entire sequence.
  // llama.cpp doesn't support non-causal attention well with KV cache, so we keep causal for now.
  // This will cause output differences until a proper non-causal implementation is added.
  // TODO: Implement proper non-causal attention by processing entire sequence in single batch
  // without KV cache, using input embeddings directly.

  // Apply template
  std::string formated_prompt = format_prompt(prompt, "");
  std::vector<llama_token> prompt_tokens;
  std::string template_img = "<__media__>";
  std::vector<std::string> texts = split_text(formated_prompt, template_img);
  if (texts.size() > 1 && texts[0] == template_img) {
    texts.insert(texts.begin(), "");
  }

  // Collect ALL tokens and embeddings first, then do a single forward pass
  // This ensures we can extract embeddings from all positions
  std::vector<llama_token> all_tokens;
  std::vector<float> all_embeddings;
  std::vector<bool> is_embedding;  // true if position uses embeddings, false if token

  // Phase 1: Collect all tokens and embeddings
  for (size_t i = 0; i < texts.size(); i++) {
    if (texts[i] == template_img) {
      // Image embeddings - mark positions as embedding type
      for (int j = 0; j < n_img_tokens; j++) {
        all_tokens.push_back(0);  // placeholder token
        is_embedding.push_back(true);
      }
      // Store image embeddings
      int prev_emb_size = all_embeddings.size();
      int32_t n_embd = llama_n_embd(model_);
      all_embeddings.resize(prev_emb_size + n_img_tokens * n_embd);
      memcpy(all_embeddings.data() + prev_emb_size, img_emb, n_img_tokens * n_embd * sizeof(float));
    } else {
      if (tokenizer_) {
        encode_text_by_tokenizer_cpp(texts[i], prompt_tokens, i == 0);
      } else {
        encode_text(texts[i], prompt_tokens, i == 0);
        if (!prompt_tokens.empty() && prompt_tokens[0] == 797) {
          prompt_tokens[0] = 512;  // for debug
        }
      }

      // Add empty token at the end of the prompt (before action tokens)
      if (i == texts.size() - 1 && !prompt_tokens.empty() &&
          prompt_tokens.back() != empty_token_) {
        prompt_tokens.push_back(empty_token_);
      }

      for (auto tok : prompt_tokens) {
        all_tokens.push_back(tok);
        is_embedding.push_back(false);
      }
    }
  }

  // Record position where action tokens will start
  int32_t action_tokens_start_pos = all_tokens.size();
  printf("Action tokens start position: %d\n", action_tokens_start_pos);

  // Debug: Print sequence structure
  int32_t num_token_positions = 0;
  int32_t num_emb_positions = 0;
  for (size_t i = 0; i < is_embedding.size(); i++) {
    if (is_embedding[i]) num_emb_positions++;
    else num_token_positions++;
  }
  printf("Sequence structure before action tokens: %d token positions, %d embedding positions\n",
         num_token_positions, num_emb_positions);

  // Debug: Print all text token IDs (non-embedding tokens)
  printf("Text token IDs: ");
  for (size_t i = 0; i < all_tokens.size(); i++) {
    if (!is_embedding[i]) printf("%d ", all_tokens[i]);
  }
  printf("\n");

  int32_t n_embd = llama_n_embd(model_);

  // Phase 2: Add action tokens with ZERO embeddings (matching Python behavior)
  // Python zeros out action token embeddings before the LLM forward pass
  // We achieve this by using embedding mode with zero-filled vectors
  for (int i = 0; i < num_action_tokens; i++) {
    all_tokens.push_back(0);  // placeholder token (not used since is_embedding=true)
    is_embedding.push_back(true);  // treat as embedding segment
  }
  // Add zero embeddings for action tokens
  int prev_emb_size = all_embeddings.size();
  all_embeddings.resize(prev_emb_size + num_action_tokens * n_embd, 0.0f);  // zero-filled

  // Add STOP token at the end (as regular token, not zeroed)
  all_tokens.push_back(stop_token_);
  is_embedding.push_back(false);

  int32_t total_tokens = all_tokens.size();

  // Process in segments, switching between token and embedding mode
  // Like Vote, we set all logits=1 to compute embeddings for all positions
  int32_t pos = 0;
  int32_t emb_offset = 0;

  // Storage for action token hidden states - we save them immediately after decoding
  // because with multiple decode calls, we only have access to the last batch's outputs
  std::vector<float> action_hidden_states;
  bool action_segment_processed = false;

  while (pos < total_tokens) {
    // Find the end of current segment (same type)
    int32_t seg_start = pos;
    bool seg_is_emb = is_embedding[pos];
    while (pos < total_tokens && is_embedding[pos] == seg_is_emb) {
      pos++;
    }
    int32_t seg_len = pos - seg_start;

    // Check if this segment contains the action tokens (zero embedding segment at action_tokens_start_pos)
    bool is_action_segment = seg_is_emb && (seg_start == action_tokens_start_pos);

    // Set all logits=1 for this segment (like Vote does)
    // This ensures all positions have embeddings computed
    std::vector<int8_t> logits_flags(seg_len, 1);

    llama_batch batch;
    if (seg_is_emb) {
      // Embedding segment (image/proprio or action tokens with zeros)
      batch = {
          .n_tokens = seg_len,
          .token = nullptr,
          .embd = all_embeddings.data() + emb_offset,
          .pos = nullptr,
          .n_seq_id = nullptr,
          .seq_id = nullptr,
          .logits = logits_flags.data(),
      };
      emb_offset += seg_len * n_embd;
    } else {
      // Token segment (text or stop token)
      batch = {
          .n_tokens = seg_len,
          .token = all_tokens.data() + seg_start,
          .embd = nullptr,
          .pos = nullptr,
          .n_seq_id = nullptr,
          .seq_id = nullptr,
          .logits = logits_flags.data(),
      };
    }

    int ret = llama_decode(ctx_, batch);
    if (ret != 0) {
      fprintf(stderr, "%s: decode failed at segment starting %d, ret = %d\n", __func__, seg_start, ret);
      return false;
    }

    // If this is the action token segment, extract hidden states immediately
    // because they will be lost after the next decode call
    if (is_action_segment && require_embeddings_) {
      int32_t n_outputs = llama_get_n_outputs(ctx_);
      printf("Action segment: extracting %d hidden states (n_outputs=%d)\n", seg_len, n_outputs);

      action_hidden_states.resize(seg_len * n_embd);
      for (int32_t i = 0; i < seg_len; i++) {
        const float *emb = llama_get_embeddings_ith(ctx_, i);
        if (!emb) {
          fprintf(stderr, "%s: failed to get embedding for action token %d\n", __func__, i);
          return false;
        }
        memcpy(action_hidden_states.data() + i * n_embd, emb, n_embd * sizeof(float));
      }
      action_segment_processed = true;
      printf("Extracted %d hidden states for action tokens\n", seg_len);
    }
  }

  // Phase 3: Copy saved action token hidden states to output
  if (require_embeddings_) {
    if (!action_segment_processed) {
      fprintf(stderr, "%s: action segment was not processed\n", __func__);
      return false;
    }
    output = std::move(action_hidden_states);
  }

  return true;
}

// ==========================================
// Token Embedding Loading and Conversion
// ==========================================

bool OpenvlaOFTLlm::load_token_embeddings(const std::string &gguf_path) {
  // Calculate memory needed for meta context
  // First pass: get tensor count
  struct gguf_init_params params_count = {
      .no_alloc = true,
      .ctx = nullptr,
  };
  gguf_context *ctx_gguf = gguf_init_from_file(gguf_path.c_str(), params_count);
  if (!ctx_gguf) {
    fprintf(stderr, "%s: failed to open GGUF file: %s\n", __func__, gguf_path.c_str());
    return false;
  }
  int64_t n_tensors = gguf_get_n_tensors(ctx_gguf);

  // Find the token_embd.weight tensor index
  int tensor_idx = -1;
  for (int64_t i = 0; i < n_tensors; i++) {
    const char *name = gguf_get_tensor_name(ctx_gguf, i);
    if (strcmp(name, "token_embd.weight") == 0) {
      tensor_idx = i;
      break;
    }
  }

  if (tensor_idx < 0) {
    fprintf(stderr, "%s: token_embd.weight not found in GGUF file\n", __func__);
    gguf_free(ctx_gguf);
    return false;
  }
  gguf_free(ctx_gguf);

  // Second pass: create context for tensors
  size_t meta_size = n_tensors * ggml_tensor_overhead() + ggml_graph_overhead();
  std::vector<uint8_t> meta_buf(meta_size);
  struct ggml_init_params meta_params = {
      .mem_size = meta_buf.size(),
      .mem_buffer = meta_buf.data(),
      .no_alloc = true,
  };
  ggml_context *ctx = ggml_init(meta_params);
  if (!ctx) {
    fprintf(stderr, "%s: failed to create meta context\n", __func__);
    return false;
  }

  struct gguf_init_params params = {
      .no_alloc = true,
      .ctx = &ctx,
  };
  ctx_gguf = gguf_init_from_file(gguf_path.c_str(), params);
  if (!ctx_gguf) {
    fprintf(stderr, "%s: failed to re-open GGUF file\n", __func__);
    ggml_free(ctx);
    return false;
  }

  // Find the tensor in the context
  ggml_tensor *token_embd = ggml_get_tensor(ctx, "token_embd.weight");
  if (!token_embd) {
    fprintf(stderr, "%s: token_embd.weight not found in context\n", __func__);
    gguf_free(ctx_gguf);
    ggml_free(ctx);
    return false;
  }

  // Get dimensions - token_embd has shape (embd_dim, vocab_size)
  n_embd_ = token_embd->ne[0];
  vocab_size_ = token_embd->ne[1];
  printf("Token embedding table: dim=%d, vocab_size=%d, type=%s\n",
         n_embd_, vocab_size_, ggml_type_name(token_embd->type));

  // Calculate file offset for this tensor
  size_t data_offset = gguf_get_data_offset(ctx_gguf) +
                       gguf_get_tensor_offset(ctx_gguf, tensor_idx);

  // Open file and read tensor data
  std::ifstream fin(gguf_path, std::ios::binary);
  if (!fin) {
    fprintf(stderr, "%s: failed to open file for reading\n", __func__);
    gguf_free(ctx_gguf);
    ggml_free(ctx);
    return false;
  }

  fin.seekg(data_offset);

  // Allocate buffer for raw tensor data
  size_t tensor_size = ggml_nbytes(token_embd);
  std::vector<uint8_t> raw_data(tensor_size);
  fin.read(reinterpret_cast<char *>(raw_data.data()), tensor_size);
  fin.close();

  // Convert to float32 based on tensor type
  token_embeddings_.resize(n_embd_ * vocab_size_);

  if (token_embd->type == GGML_TYPE_F32) {
    memcpy(token_embeddings_.data(), raw_data.data(), tensor_size);
  } else if (token_embd->type == GGML_TYPE_F16) {
    const ggml_fp16_t *src = reinterpret_cast<const ggml_fp16_t *>(raw_data.data());
    for (size_t i = 0; i < token_embeddings_.size(); i++) {
      token_embeddings_[i] = ggml_fp16_to_fp32(src[i]);
    }
  } else if (token_embd->type == GGML_TYPE_BF16) {
    const uint16_t *src = reinterpret_cast<const uint16_t *>(raw_data.data());
    for (size_t i = 0; i < token_embeddings_.size(); i++) {
      // BF16 to FP32: shift bits left by 16 (BF16 is just truncated FP32)
      uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
      memcpy(&token_embeddings_[i], &bits, sizeof(float));
    }
  } else {
    // Handle quantized types using GGML's dequantization API
    const struct ggml_type_traits *traits = ggml_get_type_traits(token_embd->type);
    if (traits && traits->to_float) {
      printf("Dequantizing token embeddings from %s to F32...\n",
             ggml_type_name(token_embd->type));

      // Dequantize row by row (each row is one token's embedding)
      // Token embedding table has shape (n_embd, vocab_size), stored row-major
      // So we have vocab_size rows of n_embd elements each
      size_t row_size_bytes = ggml_row_size(token_embd->type, n_embd_);

      for (int32_t row = 0; row < vocab_size_; row++) {
        const void *src_row = raw_data.data() + row * row_size_bytes;
        float *dst_row = token_embeddings_.data() + row * n_embd_;
        traits->to_float(src_row, dst_row, n_embd_);
      }
    } else {
      fprintf(stderr, "%s: unsupported tensor type: %s (no dequantization available)\n",
              __func__, ggml_type_name(token_embd->type));
      gguf_free(ctx_gguf);
      ggml_free(ctx);
      return false;
    }
  }

  gguf_free(ctx_gguf);
  ggml_free(ctx);

  token_embd_loaded_ = true;
  printf("Loaded token embedding table: %d tokens x %d dimensions\n",
         vocab_size_, n_embd_);
  return true;
}

bool OpenvlaOFTLlm::tokens_to_embeddings(const std::vector<llama_token> &tokens,
                                          std::vector<float> &out) const {
  if (!token_embd_loaded_) {
    fprintf(stderr, "%s: token embeddings not loaded\n", __func__);
    return false;
  }

  out.resize(tokens.size() * n_embd_);

  for (size_t i = 0; i < tokens.size(); i++) {
    int32_t token_id = tokens[i];
    if (token_id < 0 || token_id >= vocab_size_) {
      fprintf(stderr, "%s: token_id %d out of range [0, %d)\n",
              __func__, token_id, vocab_size_);
      return false;
    }
    // Copy embedding for this token
    memcpy(out.data() + i * n_embd_,
           token_embeddings_.data() + token_id * n_embd_,
           n_embd_ * sizeof(float));
  }

  return true;
}

// ==========================================
// OFT Forward Pass V2 - Single Batch Non-Causal
// ==========================================

bool OpenvlaOFTLlm::forward_oft_v2(const std::string &prompt, float *img_emb,
                                    int32_t n_img_tokens, float *proprio_emb,
                                    int32_t num_action_tokens,
                                    std::vector<float> &output, bool use_history) {
  if (!token_embd_loaded_) {
    fprintf(stderr, "%s: token embeddings not loaded - call load_token_embeddings() first\n",
            __func__);
    return false;
  }

  // Clear the KV cache before each forward pass to avoid stale data from previous runs.
  // mctx->apply() assigns positions to KV cells BEFORE the mask is computed in
  // set_input_kq_mask(), so cleared cells will be re-populated by the time the mask
  // is built. Without clearing, previous run's KV values pollute attention.
  llama_memory_clear(llama_get_memory(ctx_), true);
  llama_synchronize(ctx_);
  llama_perf_context_reset(ctx_);
  llama_set_warmup(ctx_, false);

  // Use non-causal (bidirectional) attention to match Python's modified LLaMA
  // Python uses a custom transformers library with is_causal=False in LlamaSdpaAttention
  // This duplicates the last row of the causal mask so all tokens attend to all tokens
  bool use_causal = false;
  llama_set_causal_attn(ctx_, use_causal);

  // Get embedding dimension from llama model
  int32_t llm_n_embd = llama_n_embd(model_);
  if (llm_n_embd != n_embd_) {
    fprintf(stderr, "%s: embedding dimension mismatch: llama=%d, token_embd=%d\n",
            __func__, llm_n_embd, n_embd_);
    return false;
  }

  // Apply template and tokenize
  std::string formated_prompt = format_prompt(prompt, "");
  std::string template_img = "<__media__>";
  std::vector<std::string> texts = split_text(formated_prompt, template_img);
  if (texts.size() > 1 && texts[0] == template_img) {
    texts.insert(texts.begin(), "");
  }

  // Build the complete embedding sequence:
  // [BOS_emb, vision_embs, proprio_emb (optional), text_embs, zero_action_embs, STOP_emb]
  std::vector<float> all_embeddings;
  int32_t total_tokens = 0;
  int32_t action_tokens_start_pos = -1;

  // Start with BOS token embedding
  llama_token bos_token = 1;  // BOS token for Llama2
  std::vector<llama_token> bos_tokens = {bos_token};
  std::vector<float> bos_embedding;
  if (!tokens_to_embeddings(bos_tokens, bos_embedding)) {
    fprintf(stderr, "%s: failed to convert BOS token to embedding\n", __func__);
    llama_set_causal_attn(ctx_, true);
    return false;
  }
  all_embeddings.insert(all_embeddings.end(), bos_embedding.begin(), bos_embedding.end());
  total_tokens += 1;

  for (size_t i = 0; i < texts.size(); i++) {
    if (texts[i] == template_img) {
      // Add vision embeddings
      all_embeddings.insert(all_embeddings.end(),
                            img_emb, img_emb + n_img_tokens * n_embd_);
      total_tokens += n_img_tokens;

      // Add proprio embedding if provided (after vision, before text)
      if (proprio_emb != nullptr) {
        all_embeddings.insert(all_embeddings.end(),
                              proprio_emb, proprio_emb + n_embd_);
        total_tokens += 1;
      }
    } else if (texts[i].empty()) {
      // Skip empty text segments (BOS already added at the beginning)
      continue;
    } else {
      // Tokenize text (without BOS since it's already added)
      std::vector<llama_token> prompt_tokens;
      if (tokenizer_) {
        encode_text_by_tokenizer_cpp(texts[i], prompt_tokens, false);  // No special tokens
      } else {
        encode_text(texts[i], prompt_tokens, false);  // No special tokens
        if (!prompt_tokens.empty() && prompt_tokens[0] == 797) {
          prompt_tokens[0] = 512;  // for debug
        }
      }

      // NOTE: Do NOT skip the placeholder token (512) - Python keeps it in the sequence!
      // In Python, multimodal_embeddings = [BOS, vision+proprio, input_embeddings[1:]]
      // where input_embeddings[1:] includes the placeholder (512), text tokens, action tokens, stop token
      // The placeholder's embedding is kept, it's not removed.

      // Add empty token (29871) at the end of text if not already present
      // Python does this in predict_action: if not torch.all(input_ids[:, -1] == 29871): ...
      if (i == texts.size() - 1 && !prompt_tokens.empty() &&
          prompt_tokens.back() != empty_token_) {
        prompt_tokens.push_back(empty_token_);
      }

      // Convert tokens to embeddings
      std::vector<float> text_embeddings;
      if (!tokens_to_embeddings(prompt_tokens, text_embeddings)) {
        fprintf(stderr, "%s: failed to convert tokens to embeddings\n", __func__);
        llama_set_causal_attn(ctx_, true);  // Restore causal attention
        return false;
      }

      all_embeddings.insert(all_embeddings.end(),
                            text_embeddings.begin(), text_embeddings.end());
      total_tokens += prompt_tokens.size();
    }
  }

  // Record position where action tokens start
  action_tokens_start_pos = total_tokens;

  // Add ZERO embeddings for action tokens (matching Python behavior)
  // Python zeros out action token embeddings before the LLM forward pass
  all_embeddings.resize(all_embeddings.size() + num_action_tokens * n_embd_, 0.0f);
  total_tokens += num_action_tokens;

  // Add STOP token embedding at the end
  std::vector<llama_token> stop_tokens = {stop_token_};
  std::vector<float> stop_embedding;
  if (!tokens_to_embeddings(stop_tokens, stop_embedding)) {
    fprintf(stderr, "%s: failed to convert STOP token to embedding\n", __func__);
    llama_set_causal_attn(ctx_, true);  // Restore causal attention
    return false;
  }
  all_embeddings.insert(all_embeddings.end(),
                        stop_embedding.begin(), stop_embedding.end());
  total_tokens += 1;

  // Set logits=1 for hidden state extraction positions
  // Python extracts hidden states starting at NUM_PATCHES + NUM_PROMPT_TOKENS.
  // extraction_start = action_tokens_start_pos - 1 to match Python's indexing.
  int extraction_start = action_tokens_start_pos - 1;
  std::vector<int8_t> logits_flags(total_tokens, 0);
  for (int i = extraction_start; i < extraction_start + num_action_tokens; i++) {
    logits_flags[i] = 1;
  }

  // Create position IDs (0, 1, 2, ..., total_tokens-1)
  // Required when using embeddings input directly
  std::vector<llama_pos> positions(total_tokens);
  for (int i = 0; i < total_tokens; i++) {
    positions[i] = i;
  }

  // Create sequence IDs (all belong to sequence 0)
  std::vector<int32_t> n_seq_ids(total_tokens, 1);
  std::vector<llama_seq_id*> seq_ids(total_tokens);
  std::vector<llama_seq_id> seq_id_storage(total_tokens, 0);  // All tokens in sequence 0
  for (int i = 0; i < total_tokens; i++) {
    seq_ids[i] = &seq_id_storage[i];
  }

  // Create and run single batch with all embeddings
  llama_batch batch = {
      .n_tokens = total_tokens,
      .token = nullptr,
      .embd = all_embeddings.data(),
      .pos = positions.data(),
      .n_seq_id = n_seq_ids.data(),
      .seq_id = seq_ids.data(),
      .logits = logits_flags.data(),
  };

  int ret = llama_decode(ctx_, batch);
  if (ret != 0) {
    fprintf(stderr, "%s: llama_decode failed with error %d\n", __func__, ret);
    llama_set_causal_attn(ctx_, true);  // Restore causal attention
    return false;
  }

  // Extract hidden states from action token positions
  if (require_embeddings_) {
    output.resize(num_action_tokens * n_embd_);

    // Get hidden states for action tokens
    // Since we set logits=1 only for action tokens, the output indices
    // correspond to those positions
    for (int32_t i = 0; i < num_action_tokens; i++) {
      // Use extraction_start + i to index into the correct sequence position.
      // When cparams.embeddings=true, llama.cpp overrides all logits to true,
      // so output_ids[pos] = pos for all positions. We need the action token
      // positions (starting at extraction_start), not positions 0..N.
      const float *emb = llama_get_embeddings_ith(ctx_, extraction_start + i);
      if (!emb) {
        fprintf(stderr, "%s: failed to get embedding for action token %d (pos %d)\n",
                __func__, i, extraction_start + i);
        llama_set_causal_attn(ctx_, true);  // Restore causal attention
        return false;
      }
      memcpy(output.data() + i * n_embd_, emb, n_embd_ * sizeof(float));
    }
  }

  // Restore causal attention for potential future operations
  llama_set_causal_attn(ctx_, true);

  return true;
}
