#include "infer_session.hpp"
#include "model_defs.h"

class L1RegressionHead {
public:
  L1RegressionHead() = default;
  L1RegressionHead(const std::string &model_path, const ContextParams &params)
      : model_(model_path, params) {}

  bool run(const std::vector<float> &hidden_states, std::vector<float> &out) {
    model_.set_input("inp_raw", hidden_states);
    if (!model_.run(out)) {
      return false;
    }
    return true;
  }

private:
  InferenceSession<L1RegressionActionHeadmulmlpk> model_;
};

// OFT-specific regression head wrapper
// Input: hidden states from all action token positions, reshaped appropriately
// Shape: (num_actions_chunk, action_dim * llm_hidden_dim)
class L1RegressionHeadOFT {
public:
  L1RegressionHeadOFT() = default;
  L1RegressionHeadOFT(const std::string &model_path, const ContextParams &params)
      : model_(model_path, params) {
    // Cache hyperparameters from loaded model
    const auto &hparams = model_.get_model().hparams;
    action_dim_ = hparams.action_dim;
    num_actions_chunk_ = hparams.num_actions_chunk;
    input_dim_ = hparams.input_dim;
    hidden_dim_ = hparams.hidden_dim;
  }

  bool run(const std::vector<float> &hidden_states, std::vector<float> &out) {
    model_.set_input("inp_raw", hidden_states);
    if (!model_.run(out)) {
      return false;
    }
    return true;
  }

  // Get hyperparameters for reshaping
  uint32_t get_action_dim() const { return action_dim_; }
  uint32_t get_num_actions_chunk() const { return num_actions_chunk_; }
  uint32_t get_input_dim() const { return input_dim_; }
  uint32_t get_hidden_dim() const { return hidden_dim_; }

private:
  InferenceSession<L1RegressionActionHeadOFT> model_;

  // Cached hyperparameters
  uint32_t action_dim_ = 7;
  uint32_t num_actions_chunk_ = 8;
  uint32_t input_dim_ = 4096;   // LLM hidden dim (loaded from GGUF)
  uint32_t hidden_dim_ = 4096;  // MLP hidden dim (loaded from GGUF)
};