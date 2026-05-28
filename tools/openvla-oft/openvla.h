#pragma once

#include "llm.h"
#include "proj.hpp"
#include "reg.hpp"
#include "utils.h"
#include "vit.hpp"
#include <algorithm>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>

// Dataset statistics loaded from dataset_statistics.json
// Contains normalization bounds for actions and proprioception
struct DatasetStatistics {
  std::vector<float> action_high;  // from action.q99
  std::vector<float> action_low;   // from action.q01
  std::vector<bool> action_mask;   // from action.mask
  std::vector<float> proprio_q01;  // from proprio.q01
  std::vector<float> proprio_q99;  // from proprio.q99

  static DatasetStatistics load(const std::string &json_path, const std::string &task_suite_name = "");
};

class OpenvlaProjector {
public:
  OpenvlaProjector(const std::string &dinov2_path,
                   const std::string &siglip_path, const std::string &proj_path,
                   ContextParams &ctx_params)
      : proj_(proj_path, ctx_params), dinov2_(dinov2_path, ctx_params),
        siglip_(siglip_path, ctx_params) {}

  // Single image processing
  bool run(const std::string &img_path, std::vector<float> &out);
  bool run(const uint8_t *img_data, int width, int height,
           std::vector<float> &out);

  // Two-image processing (full image + wrist image)
  // Output: concatenated embeddings (512 tokens = 256 * 2)
  bool run(const std::string &img_path1, const std::string &img_path2,
           std::vector<float> &out);
  bool run(const uint8_t *img_data1, int width1, int height1,
           const uint8_t *img_data2, int width2, int height2,
           std::vector<float> &out);

private:
  Projector proj_;
  Vit dinov2_;
  Vit siglip_;
};

class OpenvlaActionProcessor {
public:
  OpenvlaActionProcessor() { init_bin_centers(); }

  bool process(std::vector<llama_token> &predicted_action_token_ids,
               std::vector<float> &output);

private:
  bool init_bin_centers();

  std::vector<float> bin_centers_;
  // std::vector<float> action_high_ = {0.028309678435325586,
  //                                    0.040855254605412394,
  //                                    0.040161586627364146,
  //                                    0.08192047759890528,
  //                                    0.07792850524187081,
  //                                    0.20382574498653397,
  //                                    1.0};
  // std::vector<float> action_low_ = {-0.02872725307941437,
  //                                   -0.04170349963009357,
  //                                   -0.026093858778476715,
  //                                   -0.08092105075716972,
  //                                   -0.09288699507713317,
  //                                   -0.20718276381492615,
  //                                   0.0};


  std::vector<float> action_high_ = {0.8464285731315613,
    0.84375,
    0.9375,
    0.08142857253551483,
    0.14892856776714325,
    0.0867857113480568,
    1.0};
  std::vector<float> action_low_ = {-0.5383928418159485,
    -0.8758928775787354,
    -0.9375,
    -0.06964285671710968,
    -0.11678571254014969,
    -0.15964286029338837,
    0.0};
};

class VoteActionProcessor {
public:
  VoteActionProcessor(const std::string &model_path,
                      const ContextParams &params,
                      const std::vector<float> &action_high,
                      const std::vector<float> &action_low,
                      const std::vector<bool> &action_mask)
      : reg_(model_path, params), action_high_(action_high),
        action_low_(action_low), action_mask_(action_mask) {}
  bool process(const std::vector<float> &hidden_states,
               std::vector<float> &out);

private:
  L1RegressionHead reg_;
  std::vector<float> action_high_;
  std::vector<float> action_low_;
  std::vector<bool> action_mask_;
};

class Openvla {
public:
  Openvla(const std::string &dinov2_path, const std::string &siglip_path,
          const std::string &proj_path, const std::string &llm_path,
          ContextParams &ctx_params, LlmParam &llm_params);
  bool run(const std::string &img_path, const std::string &prompt,
           std::vector<float> &out);
  bool run(const uint8_t *img_data, int width, int height,
           const std::string &prompt, std::vector<float> &out);

private:
  OpenvlaProjector proj_;
  OpenvlaLlm llm_;
  OpenvlaActionProcessor processor_;
};

class OpenvlaWithRegression {
public:
  OpenvlaWithRegression(const std::string &dinov2_path,
                        const std::string &siglip_path,
                        const std::string &proj_path,
                        const std::string &llm_path,
                        const std::string &reg_path,
                        const DatasetStatistics &stats,
                        ContextParams &ctx_params,
                        LlmParam &llm_params);
  bool run(const std::string &img_path, const std::string &prompt,
           std::vector<float> &out);
  bool run(const uint8_t *img_data, int width, int height,
           const std::string &prompt, std::vector<float> &out);

private:
  OpenvlaProjector proj_;
  OpenvlaLlm llm_;
  VoteActionProcessor processor_;
};

// ==========================================
// OpenVLA-OFT specific classes
// ==========================================

// ProprioProjector - Projects proprioception state into LLM embedding space
// Architecture: fc1 -> GELU -> fc2
// Input: normalized proprio state vector (proprio_dim,) e.g., 8 for LIBERO
// Output: (llm_dim,) e.g., 4096 for Llama2-7B
class ProprioProjector {
public:
  ProprioProjector() = default;
  ProprioProjector(const std::string &model_path, const ContextParams &params)
      : model_(model_path, params) {}

  bool run(const std::vector<float> &proprio_input, std::vector<float> &out) {
    model_.set_input("proprio_input", proprio_input);
    if (!model_.run(out)) {
      return false;
    }
    return true;
  }

  bool is_initialized() const { return initialized_; }
  void set_initialized(bool val) { initialized_ = val; }

private:
  InferenceSession<ProprioProjectorModel> model_;
  bool initialized_ = false;
};

// Proprio normalization parameters (BOUNDS_Q99 normalization)
// Loaded from dataset_statistics.json at runtime
struct ProprioNormStats {
  std::vector<float> q01;
  std::vector<float> q99;
  std::vector<bool> mask;

  ProprioNormStats() = default;
  ProprioNormStats(const std::vector<float> &q01_, const std::vector<float> &q99_)
      : q01(q01_), q99(q99_), mask(q01_.size(), true) {}
};

// Normalize proprio state using BOUNDS_Q99 normalization
// Maps from raw values to [-1, 1] range using q01 and q99 percentiles
inline std::vector<float> normalize_proprio(
    const std::vector<float> &proprio,
    const ProprioNormStats &stats) {
  std::vector<float> normalized(proprio.size());
  for (size_t i = 0; i < proprio.size(); ++i) {
    if (stats.mask[i]) {
      float low = stats.q01[i];
      float high = stats.q99[i];
      float val = 2.0f * (proprio[i] - low) / (high - low + 1e-8f) - 1.0f;
      // Clip to [-1, 1]
      normalized[i] = std::max(-1.0f, std::min(1.0f, val));
    } else {
      normalized[i] = proprio[i];
    }
  }
  return normalized;
}

// OFT Action Processor
// Takes hidden states from all action token positions and processes them
// through the OFT-specific action head
class OFTActionProcessor {
public:
  OFTActionProcessor(const std::string &model_path,
                     const ContextParams &params,
                     const std::vector<float> &action_high,
                     const std::vector<float> &action_low,
                     const std::vector<bool> &action_mask)
      : reg_(model_path, params), action_high_(action_high),
        action_low_(action_low), action_mask_(action_mask) {}

  bool process(const std::vector<float> &hidden_states,
               std::vector<float> &out);

  uint32_t get_action_dim() const { return reg_.get_action_dim(); }
  uint32_t get_num_actions_chunk() const { return reg_.get_num_actions_chunk(); }
  uint32_t get_input_dim() const { return reg_.get_input_dim(); }

private:
  L1RegressionHeadOFT reg_;
  std::vector<float> action_high_;
  std::vector<float> action_low_;
  std::vector<bool> action_mask_;
};
// Main OpenVLA-OFT class
// Implements the full OFT pipeline:
// 1. Vision encoding (DINOv2 + SigLIP)
// 2. Projection to LLM space
// 3. (Optional) Proprioception projection
// 4. LLM forward with placeholder action tokens
// 5. Extract hidden states from action token positions
// 6. Reshape and process through OFT action head
class OpenvlaOFT {
public:
  // Constructor without proprio projector
  OpenvlaOFT(const std::string &dinov2_path, const std::string &siglip_path,
             const std::string &proj_path, const std::string &llm_path,
             const std::string &reg_path, const DatasetStatistics &stats,
             ContextParams &ctx_params, LlmParam &llm_params);

  // Constructor with proprio projector
  OpenvlaOFT(const std::string &dinov2_path, const std::string &siglip_path,
             const std::string &proj_path, const std::string &llm_path,
             const std::string &reg_path, const std::string &proprio_proj_path,
             const DatasetStatistics &stats,
             ContextParams &ctx_params, LlmParam &llm_params);

  // Run without proprio (backward compatible, single image)
  bool run(const std::string &img_path, const std::string &prompt,
           std::vector<float> &out);
  bool run(const uint8_t *img_data, int width, int height,
           const std::string &prompt, std::vector<float> &out);

  // Run with proprio input (raw proprio values, will be normalized internally)
  // Single image version
  bool run(const std::string &img_path, const std::string &prompt,
           const std::vector<float> &proprio, std::vector<float> &out);
  bool run(const uint8_t *img_data, int width, int height,
           const std::string &prompt, const std::vector<float> &proprio,
           std::vector<float> &out);

  // Two-image version with proprio (full image + wrist image)
  // This is the recommended method for LIBERO evaluation
  bool run(const std::string &img_path1, const std::string &img_path2,
           const std::string &prompt, const std::vector<float> &proprio,
           std::vector<float> &out);
  bool run(const uint8_t *img_data1, int width1, int height1,
           const uint8_t *img_data2, int width2, int height2,
           const std::string &prompt, const std::vector<float> &proprio,
           std::vector<float> &out);

  // Check if proprio projector is initialized
  bool has_proprio_projector() const { return use_proprio_; }

private:
  OpenvlaProjector proj_;
  OpenvlaOFTLlm llm_;
  OFTActionProcessor processor_;

  // LLM path (stored for potential reloading)
  std::string llm_path_;

  // Proprio projector (optional)
  std::unique_ptr<ProprioProjector> proprio_proj_;
  bool use_proprio_ = false;
  ProprioNormStats proprio_norm_stats_;

  // Helper function to run inference with optional proprio
  // num_images: 1 for single image (256 patches), 2 for dual image (512 patches)
  bool run_internal(const std::vector<float> &vision_embedding,
                    const std::string &prompt,
                    const std::vector<float> *proprio,
                    int32_t num_images,
                    std::vector<float> &out);

  // OFT parameters (LIBERO defaults)
  static constexpr int32_t ACTION_DIM = 7;
  static constexpr int32_t NUM_ACTIONS_CHUNK = 8;
  static constexpr int32_t NUM_ACTION_TOKENS = ACTION_DIM * NUM_ACTIONS_CHUNK;  // 56
  static constexpr int32_t PATCHES_PER_IMAGE = 256;  // 224x224 / 14x14 patch size
};
