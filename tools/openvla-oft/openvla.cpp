#include "openvla.h"
#include "timer.hpp"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <functional>
#include <future>

namespace fs = std::filesystem;

DatasetStatistics DatasetStatistics::load(const std::string &json_path, const std::string &task_suite_name) {
  std::ifstream f(json_path);
  if (!f.is_open()) {
    throw std::runtime_error("Failed to open dataset_statistics.json: " + json_path);
  }
  nlohmann::json j = nlohmann::json::parse(f);

  if (j.empty()) {
    throw std::runtime_error("Empty dataset_statistics.json: " + json_path);
  }

  // If task_suite_name provided, look for exact match or with "_no_noops" suffix
  nlohmann::json::iterator it;
  if (!task_suite_name.empty()) {
    it = j.find(task_suite_name);
    if (it == j.end()) {
      it = j.find(task_suite_name + "_no_noops");
    }
    if (it == j.end()) {
      throw std::runtime_error("Task suite '" + task_suite_name + "' not found in " + json_path +
                               ". Available keys: " + [&]() {
                                 std::string keys;
                                 for (auto &el : j.items()) {
                                   if (!keys.empty()) keys += ", ";
                                   keys += el.key();
                                 }
                                 return keys;
                               }());
    }
  } else {
    // Fallback: grab the first top-level key (single-task stats files)
    it = j.begin();
  }
  const auto &dataset = it.value();

  DatasetStatistics stats;
  stats.action_high = dataset["action"]["q99"].get<std::vector<float>>();
  stats.action_low = dataset["action"]["q01"].get<std::vector<float>>();
  stats.action_mask = dataset["action"]["mask"].get<std::vector<bool>>();

  if (dataset.contains("proprio")) {
    stats.proprio_q01 = dataset["proprio"]["q01"].get<std::vector<float>>();
    stats.proprio_q99 = dataset["proprio"]["q99"].get<std::vector<float>>();
  }

  printf("Loaded dataset statistics from %s (dataset: %s)\n",
         json_path.c_str(), it.key().c_str());
  return stats;
}

bool OpenvlaProjector::run(const std::string &img_path,
                           std::vector<float> &out) {
  std::vector<float> dinov2_out, siglip_out;

  Timer t;
  t.start();
  auto fut_dinov2 = std::async(
      std::launch::async, [&]() { return dinov2_.run(img_path, dinov2_out); });


  auto fut_siglip = std::async(
      std::launch::async, [&] { return siglip_.run(img_path, siglip_out); });

  if (!fut_dinov2.get() || !fut_siglip.get()) {
    return false;
  }

  if (!proj_.run(dinov2_out, siglip_out, out)) {
    return false;
  }

  return true;
}

bool OpenvlaProjector::run(const uint8_t *img_data, int width, int height,
                           std::vector<float> &out) {
  std::vector<float> dinov2_out, siglip_out;

  Timer t;
  t.start();
  auto fut_dinov2 = std::async(std::launch::async, [&]() {
    return dinov2_.run(img_data, width, height, dinov2_out);
  });
  auto fut_siglip = std::async(std::launch::async, [&] {
    return siglip_.run(img_data, width, height, siglip_out);
  });
  if (!fut_dinov2.get() || !fut_siglip.get()) {
    return false;
  }

  if (!proj_.run(dinov2_out, siglip_out, out)) {
    return false;
  }

  return true;
}

// Two-image processing (full image + wrist image)
// Note: Images are processed sequentially because the Vit instances
// are not thread-safe for parallel processing on the same GPU
bool OpenvlaProjector::run(const std::string &img_path1, const std::string &img_path2,
                           std::vector<float> &out) {
  std::vector<float> out1, out2;

  // Process images sequentially (not parallel due to GPU memory constraints)
  if (!run(img_path1, out1)) {
    return false;
  }
  if (!run(img_path2, out2)) {
    return false;
  }

  // Concatenate embeddings: (256 + 256) tokens = 512 tokens
  out.reserve(out1.size() + out2.size());
  out.insert(out.end(), out1.begin(), out1.end());
  out.insert(out.end(), out2.begin(), out2.end());

  return true;
}

bool OpenvlaProjector::run(const uint8_t *img_data1, int width1, int height1,
                           const uint8_t *img_data2, int width2, int height2,
                           std::vector<float> &out) {
  std::vector<float> out1, out2;

  // Process images sequentially (not parallel due to GPU memory constraints)
  if (!run(img_data1, width1, height1, out1)) {
    return false;
  }
  if (!run(img_data2, width2, height2, out2)) {
    return false;
  }

  // Concatenate embeddings: (256 + 256) tokens = 512 tokens
  out.reserve(out1.size() + out2.size());
  out.insert(out.end(), out1.begin(), out1.end());
  out.insert(out.end(), out2.begin(), out2.end());

  return true;
}

Openvla::Openvla(const std::string &dinov2_path, const std::string &siglip_path,
                 const std::string &proj_path, const std::string &llm_path,
                 ContextParams &ctx_params, LlmParam &llm_params)
    : proj_(dinov2_path, siglip_path, proj_path, ctx_params) {
  llm_.load_model(llm_path, llm_params);
  if (!llm_params.tokenizer_path.empty() &&
      fs::exists(llm_params.tokenizer_path)) {
    llm_.init_tokenizer(llm_params.tokenizer_path);
  }
  llm_.set_empty_token(29871);
}

bool Openvla::run(const std::string &img_path, const std::string &prompt,
                  std::vector<float> &out) {
  std::vector<float> vision_embedding;
  if (!proj_.run(img_path, vision_embedding)) {
    return false;
  }
  Timer timer;
  timer.start();
  std::vector<llama_token> generated_tokens;
  if (!llm_.generate(prompt, vision_embedding.data(), 256, generated_tokens,
                     out, false)) {
    return false;
  }

  printf("llm time: %.2f ms\n", timer.stop<Timer::ms>());
  processor_.process(generated_tokens, out);
  return true;
}

bool Openvla::run(const uint8_t *img_data, int width, int height,
                  const std::string &prompt, std::vector<float> &out) {
  std::vector<float> vision_embedding;
  if (!proj_.run(img_data, width, height, vision_embedding)) {
    return false;
  }
  Timer timer;
  timer.start();
  std::vector<llama_token> generated_tokens;
  if (!llm_.generate(prompt, vision_embedding.data(), 256, generated_tokens,
                     out, false)) {
    return false;
  }

  printf("llm time: %.2f ms\n", timer.stop<Timer::ms>());
  processor_.process(generated_tokens, out);
  return true;
}

bool OpenvlaActionProcessor::init_bin_centers() {
  float start = -1.f, end = 1.f;
  int n = 256;
  float step = (end - start) / (n - 1);
  bin_centers_.resize(n - 1);
  std::vector<float> bins(n, 0.f);
  for (int i = 0; i < n; ++i) {
    bins[i] = start + i * step;
  }
  for (int i = 0; i < n - 1; ++i) {
    bin_centers_[i] = (bins[i] + bins[i + 1]) / 2.f;
  }
  return true;
}

bool OpenvlaActionProcessor::process(
    std::vector<llama_token> &predicted_action_token_ids,
    std::vector<float> &output) {
  std::transform(
      predicted_action_token_ids.begin(), predicted_action_token_ids.end(),
      predicted_action_token_ids.begin(), [&](llama_token token) {
        return std::min(std::max(token, 0), int(bin_centers_.size() - 1));
      });
  std::vector<float> normalized_actions;
  normalized_actions.reserve(predicted_action_token_ids.size());
  for (size_t i = 0; i < predicted_action_token_ids.size(); i++) {
    normalized_actions.push_back(bin_centers_[predicted_action_token_ids[i]]);
  }

  std::vector<bool> mask = {true, true, true, true, true, true, false};
  output.resize(7);
  for (int i = 0; i < 7; ++i) {
    float tmp_value = 0.5 * (normalized_actions[i] + 1) *
                          (action_high_[i] - action_low_[i] + 1e-8) +
                      action_low_[i];
    if (mask[i]) {
      output[i] = tmp_value;
    } else {
      output[i] = normalized_actions[i];
    }
  }
  return true;
}

bool VoteActionProcessor::process(const std::vector<float> &hidden_states,
                                  std::vector<float> &out) {
  constexpr size_t EXPECTED_SIZE = 56 * 2;  // 16 actions * 7 dims
  constexpr size_t ACTION_DIM = 7;
  constexpr size_t NUM_ACTIONS = 16;

  std::vector<float> normalized_actions;
  reg_.run(hidden_states, normalized_actions);
  if (normalized_actions.size() != EXPECTED_SIZE) {
    throw std::runtime_error("Invalid output size");
  }

  out.resize(normalized_actions.size());

  for (size_t idx = 0; idx < normalized_actions.size(); ++idx) {
    size_t j = idx % ACTION_DIM;
    float tmp_value = 0.5f * (normalized_actions[idx] + 1.0f) *
                          (action_high_[j] - action_low_[j] + 1e-8f) +
                      action_low_[j];
    if (action_mask_[j]) {
      out[idx] = tmp_value;
    } else {
      out[idx] = normalized_actions[idx];
    }
  }

  return true;
}

OpenvlaWithRegression::OpenvlaWithRegression(const std::string &dinov2_path,
                                             const std::string &siglip_path,
                                             const std::string &proj_path,
                                             const std::string &llm_path,
                                             const std::string &reg_path,
                                             const DatasetStatistics &stats,
                                             ContextParams &ctx_params,
                                             LlmParam &llm_params)
    : proj_(dinov2_path, siglip_path, proj_path, ctx_params),
      processor_(reg_path, ctx_params, stats.action_high, stats.action_low, stats.action_mask) {
  llm_.load_model(llm_path, llm_params);
  if (!llm_params.tokenizer_path.empty() &&
      fs::exists(llm_params.tokenizer_path)) {
    llm_.init_tokenizer(llm_params.tokenizer_path);
  }
  llm_.set_empty_token(29871);
}

bool OpenvlaWithRegression::run(const std::string &img_path,
                                const std::string &prompt,
                                std::vector<float> &out) {
  // const int num_actions_chunk = 8;
  // const int num_actions_per_token = 8;
  std::vector<float> vision_embedding;
  if (!proj_.run(img_path, vision_embedding)) {
    return false;
  }

  Timer timer;
  timer.start();
  std::vector<llama_token> generated_tokens;
  std::vector<float> hidden_states;
  bool ret = llm_.generate(prompt, vision_embedding.data(), 256, generated_tokens,
                     hidden_states, false);
  if (!ret) {
    return false;
  }

  processor_.process(hidden_states, out);
  return true;
}

bool OpenvlaWithRegression::run(const uint8_t *img_data, int width, int height,
                                const std::string &prompt,
                                std::vector<float> &out) {
  std::vector<float> vision_embedding;
  if (!proj_.run(img_data, width, height, vision_embedding)) {
    return false;
  }

  Timer timer;
  timer.start();
  std::vector<llama_token> generated_tokens;
  std::vector<float> hidden_states;
  bool ret = llm_.generate(prompt, vision_embedding.data(), 256, generated_tokens,
                     hidden_states, false);
  if (!ret) {
    return false;
  }

  processor_.process(hidden_states, out);
  return true;
}

// ==========================================
// OpenVLA-OFT Implementation
// ==========================================

bool OFTActionProcessor::process(const std::vector<float> &hidden_states,
                                  std::vector<float> &out) {
  // hidden_states: flattened (num_action_tokens, hidden_dim)
  // num_action_tokens = num_actions_chunk * action_dim = 8 * 7 = 56
  // hidden_dim = 2048
  // Total size: 56 * 2048 = 114688

  const uint32_t action_dim = reg_.get_action_dim();
  const uint32_t num_actions_chunk = reg_.get_num_actions_chunk();
  const uint32_t hidden_dim = reg_.get_input_dim();
  const uint32_t num_action_tokens = num_actions_chunk * action_dim;

  // Verify input size
  size_t expected_input_size = num_action_tokens * hidden_dim;
  if (hidden_states.size() != expected_input_size) {
    fprintf(stderr, "OFTActionProcessor::process: unexpected input size %zu, expected %zu\n",
            hidden_states.size(), expected_input_size);
    return false;
  }

  // Reshape hidden states from (num_action_tokens, hidden_dim) to (num_actions_chunk, action_dim * hidden_dim)
  // Input layout: [token0_h0..h2047, token1_h0..h2047, ..., token55_h0..h2047]
  // Need to rearrange to: [chunk0_a0h0..a6h2047, chunk1_a0h0..a6h2047, ..., chunk7_a0h0..a6h2047]
  //
  // Token mapping: token[chunk * action_dim + action] -> chunk[chunk], position[action * hidden_dim : (action+1) * hidden_dim]
  //
  // For OFT, the reshape is:
  // (chunk_len * action_dim, hidden_dim) -> (chunk_len, action_dim * hidden_dim)
  // So chunk i gets tokens [i*action_dim, (i+1)*action_dim)
  // Each chunk position gets: concatenation of hidden_dim values from action_dim tokens

  std::vector<float> reshaped_hidden_states(num_actions_chunk * action_dim * hidden_dim);

  for (uint32_t chunk = 0; chunk < num_actions_chunk; ++chunk) {
    for (uint32_t action = 0; action < action_dim; ++action) {
      uint32_t token_idx = chunk * action_dim + action;
      uint32_t src_offset = token_idx * hidden_dim;
      uint32_t dst_offset = chunk * (action_dim * hidden_dim) + action * hidden_dim;

      memcpy(reshaped_hidden_states.data() + dst_offset,
             hidden_states.data() + src_offset,
             hidden_dim * sizeof(float));
    }
  }

  // Run through the OFT action head
  std::vector<float> normalized_actions;
  if (!reg_.run(reshaped_hidden_states, normalized_actions)) {
    fprintf(stderr, "OFTActionProcessor::process: action head forward failed\n");
    return false;
  }

  // Output should be (num_actions_chunk, action_dim) = (8, 7) = 56 values
  size_t expected_output_size = num_actions_chunk * action_dim;
  if (normalized_actions.size() != expected_output_size) {
    fprintf(stderr, "OFTActionProcessor::process: unexpected output size %zu, expected %zu\n",
            normalized_actions.size(), expected_output_size);
    return false;
  }


  // Denormalize actions
  out.resize(normalized_actions.size());

  for (size_t idx = 0; idx < normalized_actions.size(); ++idx) {
    size_t j = idx % action_dim;
    float tmp_value = 0.5f * (normalized_actions[idx] + 1.0f) *
                          (action_high_[j] - action_low_[j] + 1e-8f) +
                      action_low_[j];
    if (action_mask_[j]) {
      out[idx] = tmp_value;
    } else {
      out[idx] = normalized_actions[idx];
    }
  }

  return true;
}

// Constructor without proprio projector
OpenvlaOFT::OpenvlaOFT(const std::string &dinov2_path,
                       const std::string &siglip_path,
                       const std::string &proj_path,
                       const std::string &llm_path,
                       const std::string &reg_path,
                       const DatasetStatistics &stats,
                       ContextParams &ctx_params,
                       LlmParam &llm_params)
    : proj_(dinov2_path, siglip_path, proj_path, ctx_params),
      processor_(reg_path, ctx_params, stats.action_high, stats.action_low, stats.action_mask),
      llm_path_(llm_path),
      use_proprio_(false) {
  llm_.load_model(llm_path, llm_params);
  if (!llm_params.tokenizer_path.empty() &&
      fs::exists(llm_params.tokenizer_path)) {
    llm_.init_tokenizer(llm_params.tokenizer_path);
  }
  llm_.set_empty_token(29871);  // Llama2 empty token
  llm_.set_stop_token(2);       // </s> token

  // Load token embeddings for non-causal forward pass (forward_oft_v2)
  if (!llm_.load_token_embeddings(llm_path)) {
    fprintf(stderr, "Warning: Failed to load token embeddings. "
                    "forward_oft_v2 will not be available.\n");
  }
}

// Constructor with proprio projector
OpenvlaOFT::OpenvlaOFT(const std::string &dinov2_path,
                       const std::string &siglip_path,
                       const std::string &proj_path,
                       const std::string &llm_path,
                       const std::string &reg_path,
                       const std::string &proprio_proj_path,
                       const DatasetStatistics &stats,
                       ContextParams &ctx_params,
                       LlmParam &llm_params)
    : proj_(dinov2_path, siglip_path, proj_path, ctx_params),
      processor_(reg_path, ctx_params, stats.action_high, stats.action_low, stats.action_mask),
      llm_path_(llm_path),
      proprio_norm_stats_(stats.proprio_q01, stats.proprio_q99),
      use_proprio_(true) {
  llm_.load_model(llm_path, llm_params);
  if (!llm_params.tokenizer_path.empty() &&
      fs::exists(llm_params.tokenizer_path)) {
    llm_.init_tokenizer(llm_params.tokenizer_path);
  }
  llm_.set_empty_token(29871);  // Llama2 empty token
  llm_.set_stop_token(2);       // </s> token

  // Load token embeddings for non-causal forward pass (forward_oft_v2)
  if (!llm_.load_token_embeddings(llm_path)) {
    fprintf(stderr, "Warning: Failed to load token embeddings. "
                    "forward_oft_v2 will not be available.\n");
  }

  // Initialize proprio projector
  proprio_proj_ = std::make_unique<ProprioProjector>(proprio_proj_path, ctx_params);
  proprio_proj_->set_initialized(true);
}

// Internal helper function to run inference with optional proprio
// num_images: 1 for single image (256 patches), 2 for dual image (512 patches)
bool OpenvlaOFT::run_internal(const std::vector<float> &vision_embedding,
                              const std::string &prompt,
                              const std::vector<float> *proprio,
                              int32_t num_images,
                              std::vector<float> &out) {
  int32_t n_img_tokens = PATCHES_PER_IMAGE * num_images;  // 256 per image
  std::vector<float> proprio_embedding;
  float *proprio_emb_ptr = nullptr;

  if (use_proprio_ && proprio != nullptr && proprio_proj_) {
    // Normalize proprio state
    std::vector<float> normalized_proprio = normalize_proprio(*proprio, proprio_norm_stats_);

    // Project proprio to LLM embedding space
    if (!proprio_proj_->run(normalized_proprio, proprio_embedding)) {
      fprintf(stderr, "OpenvlaOFT::run_internal: proprio projection failed\n");
      return false;
    }
    proprio_emb_ptr = proprio_embedding.data();

  }

  // Run LLM forward with non-causal attention (v2)
  // This matches Python's approach: single batch, bidirectional attention
  std::vector<float> hidden_states;

  // Cast away const for vision_embedding since forward_oft_v2 takes float*
  // (the function doesn't modify the data)
  float *vision_emb_ptr = const_cast<float*>(vision_embedding.data());

  if (!llm_.forward_oft_v2(prompt, vision_emb_ptr, n_img_tokens,
                           proprio_emb_ptr, NUM_ACTION_TOKENS,
                           hidden_states, false)) {
    fprintf(stderr, "OpenvlaOFT::run_internal: LLM forward (v2) failed\n");
    return false;
  }

  // Process hidden states through action head
  if (!processor_.process(hidden_states, out)) {
    fprintf(stderr, "OpenvlaOFT::run_internal: action processing failed\n");
    return false;
  }

  return true;
}

// Run without proprio (backward compatible, single image)
bool OpenvlaOFT::run(const std::string &img_path, const std::string &prompt,
                     std::vector<float> &out) {
  // Step 1: Run vision encoding and projection
  std::vector<float> vision_embedding;
  if (!proj_.run(img_path, vision_embedding)) {
    fprintf(stderr, "OpenvlaOFT::run: vision projection failed\n");
    return false;
  }

  // Step 2-3: Run internal (without proprio, single image)
  return run_internal(vision_embedding, prompt, nullptr, 1, out);
}

bool OpenvlaOFT::run(const uint8_t *img_data, int width, int height,
                     const std::string &prompt, std::vector<float> &out) {
  // Step 1: Run vision encoding and projection
  std::vector<float> vision_embedding;
  if (!proj_.run(img_data, width, height, vision_embedding)) {
    fprintf(stderr, "OpenvlaOFT::run: vision projection failed\n");
    return false;
  }

  // Step 2-3: Run internal (without proprio, single image)
  return run_internal(vision_embedding, prompt, nullptr, 1, out);
}

// Run with proprio input (single image)
bool OpenvlaOFT::run(const std::string &img_path, const std::string &prompt,
                     const std::vector<float> &proprio, std::vector<float> &out) {
  if (!use_proprio_ || !proprio_proj_) {
    fprintf(stderr, "OpenvlaOFT::run: proprio projector not initialized. "
                    "Use constructor with proprio_proj_path.\n");
    return false;
  }

  // Step 1: Run vision encoding and projection
  std::vector<float> vision_embedding;
  if (!proj_.run(img_path, vision_embedding)) {
    fprintf(stderr, "OpenvlaOFT::run: vision projection failed\n");
    return false;
  }

  // Step 2-3: Run internal (with proprio, single image)
  return run_internal(vision_embedding, prompt, &proprio, 1, out);
}

bool OpenvlaOFT::run(const uint8_t *img_data, int width, int height,
                     const std::string &prompt, const std::vector<float> &proprio,
                     std::vector<float> &out) {
  if (!use_proprio_ || !proprio_proj_) {
    fprintf(stderr, "OpenvlaOFT::run: proprio projector not initialized. "
                    "Use constructor with proprio_proj_path.\n");
    return false;
  }

  // Step 1: Run vision encoding and projection
  std::vector<float> vision_embedding;
  if (!proj_.run(img_data, width, height, vision_embedding)) {
    fprintf(stderr, "OpenvlaOFT::run: vision projection failed\n");
    return false;
  }

  // Step 2-3: Run internal (with proprio, single image)
  return run_internal(vision_embedding, prompt, &proprio, 1, out);
}

// Two-image version with proprio (full image + wrist image)
bool OpenvlaOFT::run(const std::string &img_path1, const std::string &img_path2,
                     const std::string &prompt, const std::vector<float> &proprio,
                     std::vector<float> &out) {
  if (!use_proprio_ || !proprio_proj_) {
    fprintf(stderr, "OpenvlaOFT::run: proprio projector not initialized. "
                    "Use constructor with proprio_proj_path.\n");
    return false;
  }

  // Step 1: Run vision encoding and projection for both images
  std::vector<float> vision_embedding;
  if (!proj_.run(img_path1, img_path2, vision_embedding)) {
    fprintf(stderr, "OpenvlaOFT::run: vision projection failed\n");
    return false;
  }

  // Step 2-3: Run internal (with proprio, two images)
  return run_internal(vision_embedding, prompt, &proprio, 2, out);
}

bool OpenvlaOFT::run(const uint8_t *img_data1, int width1, int height1,
                     const uint8_t *img_data2, int width2, int height2,
                     const std::string &prompt, const std::vector<float> &proprio,
                     std::vector<float> &out) {
  if (!use_proprio_ || !proprio_proj_) {
    fprintf(stderr, "OpenvlaOFT::run: proprio projector not initialized. "
                    "Use constructor with proprio_proj_path.\n");
    return false;
  }

  // Step 1: Run vision encoding and projection for both images
  std::vector<float> vision_embedding;
  if (!proj_.run(img_data1, width1, height1, img_data2, width2, height2, vision_embedding)) {
    fprintf(stderr, "OpenvlaOFT::run: vision projection failed\n");
    return false;
  }

  // Step 2-3: Run internal (with proprio, two images)
  return run_internal(vision_embedding, prompt, &proprio, 2, out);
}
