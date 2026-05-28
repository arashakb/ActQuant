#pragma once

#include "build_graph.h"
#include "ctx_manager.h"
#include "ggml.h"
#include "model_loader.h"
#include <unordered_set>
#include <vector>
#include <functional>

class BaseModel {
public:
  bool load_tensors(ModelLoader &model_loader, ContextManager &ctx_manager);

  virtual bool load_hparams(const ModelLoader &model_loader) { return true; }
  virtual std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) {
    return {};
  }
  virtual ~BaseModel() = default;
};

class FakeModel: public BaseModel {
  public:
  std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
  std::vector<ggml_tensor *> build_graph(ggml_context *ctx0);

  ggml_tensor *embed_tokens = nullptr;
  ggml_tensor *llm_head = nullptr;
};

struct VisionParams {
  uint32_t image_size = 224;
  uint32_t patch_size;
  uint32_t n_embd;
  uint32_t n_ff;
  uint32_t n_head;
  uint32_t n_layer;
  std::vector<float> image_mean;
  std::vector<float> image_std;
  FfnOpType ffn_op = FfnOpType::Gelu;
  float eps = 1e-6;
  uint32_t projection_dim;
  std::unordered_set<int32_t> vision_feature_layer;
};

struct ClipLayer {
  // attention
  ggml_tensor *k_w = nullptr;
  ggml_tensor *k_b = nullptr;
  ggml_tensor *q_w = nullptr;
  ggml_tensor *q_b = nullptr;
  ggml_tensor *v_w = nullptr;
  ggml_tensor *v_b = nullptr;

  ggml_tensor *o_w = nullptr;
  ggml_tensor *o_b = nullptr;

  ggml_tensor *k_norm = nullptr;
  ggml_tensor *q_norm = nullptr;

  // layernorm 1
  ggml_tensor *ln_1_w = nullptr;
  ggml_tensor *ln_1_b = nullptr;

  ggml_tensor *ff_up_w = nullptr;
  ggml_tensor *ff_up_b = nullptr;
  ggml_tensor *ff_gate_w = nullptr;
  ggml_tensor *ff_gate_b = nullptr;
  ggml_tensor *ff_down_w = nullptr;
  ggml_tensor *ff_down_b = nullptr;

  // layernorm 2
  ggml_tensor *ln_2_w = nullptr;
  ggml_tensor *ln_2_b = nullptr;

  // layer scale (no bias)
  ggml_tensor *ls_1_w = nullptr;
  ggml_tensor *ls_2_w = nullptr;
};

class ProjectorModel : public BaseModel {
public:
  std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
  std::vector<ggml_tensor *> build_graph(ggml_context *ctx0);

  ggml_tensor *fc1_weight = nullptr;
  ggml_tensor *fc1_bias = nullptr;
  ggml_tensor *fc2_weight = nullptr;
  ggml_tensor *fc2_bias = nullptr;
  ggml_tensor *fc3_weight = nullptr;
  ggml_tensor *fc3_bias = nullptr;
};

class VisionTransformerModel : public BaseModel {
public:
  std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
  bool load_hparams(const ModelLoader &model_loader) override;
  std::vector<ggml_tensor *> build_graph(ggml_context *ctx0);

  ggml_tensor *build_vit(
      ggml_context *ctx0, ggml_tensor *inp, int n_pos, NormType norm_t,
      std::function<ggml_tensor *(ggml_tensor *, const ClipLayer &)> add_pos);

  VisionParams hparams;
  // embeddings
  ggml_tensor *class_embedding = nullptr;
  ggml_tensor *reg_embedding = nullptr;
  ggml_tensor *patch_embeddings_0 = nullptr;
  ggml_tensor *patch_embeddings_1 =
      nullptr; // second Conv2D kernel when we decouple Conv3D along temproal
               // dimension (Qwen2VL)
  ggml_tensor *patch_bias = nullptr;
  ggml_tensor *position_embeddings = nullptr;

  ggml_tensor *pre_ln_w = nullptr;
  ggml_tensor *pre_ln_b = nullptr;

  std::vector<ClipLayer> layers;

  ggml_tensor *post_ln_w;
  ggml_tensor *post_ln_b;
};

struct RegressionParams {
  uint32_t action_dim = 7;
  uint32_t num_actions_chunk = 8;
  uint32_t num_actions_per_token = 8;
  uint32_t num_blocks = 4;
  uint32_t input_dim = 2048;
  uint32_t hidden_dim = 512;
  uint32_t expansion = 4;
};

class MLPResNetBlockV2 {
public:
  /*
  class MLPResNetBlockV2(nn.Module):
      def __init__(self, dim, expansion=4, dropout=0.1):
          super().__init__()
          self.ffn = nn.Sequential(
              nn.LayerNorm(dim),
              nn.Linear(dim, dim * expansion),
              nn.SiLU(),
              nn.Linear(dim * expansion, dim)
          )
          self.dropout = nn.Dropout(dropout)

      def forward(self, x):
          identity = x
          x_ffn = self.ffn(x)
          x_dropped = self.dropout(x_ffn)
          x = x_dropped + identity
          return x
  */
  ggml_tensor *ffn_ln_w = nullptr;
  ggml_tensor *ffn_ln_b = nullptr;
  ggml_tensor *ffn_fc_w = nullptr;
  ggml_tensor *ffn_fc_b = nullptr;
  ggml_tensor *ffn_fc2_w = nullptr;
  ggml_tensor *ffn_fc2_b = nullptr;

  ggml_tensor *build_graph(ggml_context *ctx0, ggml_tensor* inp);
};

class MLPResNetBlock {
public:
  /*
  class MLPResNetBlock(nn.Module):
      def __init__(self, dim, expansion=4, dropout=0.1):
          super().__init__()
          self.ffn = nn.Sequential(
              nn.LayerNorm(dim),
              nn.Linear(dim, dim * expansion),
              nn.SiLU(),
          )
          self.dropout = nn.Dropout(dropout)

      def forward(self, x):
          identity = x
          x_ffn = self.ffn(x)
          x_dropped = self.dropout(x_ffn)
          x = x_dropped + identity
          return x
  */
  ggml_tensor *ffn_ln_w = nullptr;
  ggml_tensor *ffn_ln_b = nullptr;
  ggml_tensor *ffn_fc_w = nullptr;
  ggml_tensor *ffn_fc_b = nullptr;

  ggml_tensor *build_graph(ggml_context *ctx0, ggml_tensor* inp);
};

class L1RegressionActionHeadFunnelModel : public BaseModel {
public:
  std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
  bool load_hparams(const ModelLoader &model_loader) override;
  std::vector<ggml_tensor *> build_graph(ggml_context *ctx0);

  RegressionParams hparams;

  /*
    self.input_proj = nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
    )
  */
  ggml_tensor *input_proj_ln_w = nullptr;
  ggml_tensor *input_proj_ln_b = nullptr;
  ggml_tensor *input_proj_fc_w = nullptr;
  ggml_tensor *input_proj_fc_b = nullptr;

  std::vector<MLPResNetBlockV2> resnet_body;

  ggml_tensor *output_head_ln_w = nullptr;
  ggml_tensor *output_head_ln_b = nullptr;
  ggml_tensor *output_head_fc_w = nullptr;
  ggml_tensor *output_head_fc_b = nullptr;
};

class L1RegressionActionHeadmulmlpk : public BaseModel {
public:
  std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
  bool load_hparams(const ModelLoader &model_loader) override;
  std::vector<ggml_tensor *> build_graph(ggml_context *ctx0);

  RegressionParams hparams;

  ggml_tensor *input_ln1_w = nullptr;
  ggml_tensor *input_ln1_b = nullptr;

  ggml_tensor *input_fc1_w = nullptr;
  ggml_tensor *input_fc1_b = nullptr;

  std::vector<MLPResNetBlock> resnet_body;

  ggml_tensor *output_ln2_w = nullptr;
  ggml_tensor *output_ln2_b = nullptr;

  ggml_tensor *output_fc2_w = nullptr;
  ggml_tensor *output_fc2_b = nullptr;
};

// ==========================================
// OpenVLA-OFT specific action head
// Uses ReLU activation and processes hidden states from all action token positions
// Input: (num_actions_chunk, action_dim * llm_hidden_dim)
// Output: (num_actions_chunk, action_dim)
// ==========================================

// MLPResNetBlock for OFT - uses ReLU instead of SiLU
class MLPResNetBlockOFT {
public:
  /*
  class MLPResNetBlock(nn.Module):
      def __init__(self, dim):
          super().__init__()
          self.ffn = nn.Sequential(
              nn.LayerNorm(dim),
              nn.Linear(dim, dim),
              nn.ReLU(),
          )

      def forward(self, x):
          identity = x
          x = self.ffn(x)
          x = x + identity
          return x
  */
  ggml_tensor *ffn_ln_w = nullptr;
  ggml_tensor *ffn_ln_b = nullptr;
  ggml_tensor *ffn_fc_w = nullptr;
  ggml_tensor *ffn_fc_b = nullptr;

  ggml_tensor *build_graph(ggml_context *ctx0, ggml_tensor *inp);
};

struct OFTRegressionParams {
  uint32_t action_dim = 7;
  uint32_t num_actions_chunk = 8;
  uint32_t num_blocks = 2;         // OFT uses 2 blocks by default
  uint32_t input_dim = 4096;       // LLM hidden dim (Llama2-7B uses 4096)
  uint32_t hidden_dim = 4096;      // MLP hidden dim
};

class L1RegressionActionHeadOFT : public BaseModel {
public:
  std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
  bool load_hparams(const ModelLoader &model_loader) override;
  std::vector<ggml_tensor *> build_graph(ggml_context *ctx0);

  OFTRegressionParams hparams;

  // Layer norm and FC before blocks
  // Input: (chunk_len, action_dim * input_dim)
  // self.layer_norm1 = nn.LayerNorm(input_dim)  -- but input_dim here is action_dim * llm_hidden_dim
  ggml_tensor *layer_norm1_w = nullptr;
  ggml_tensor *layer_norm1_b = nullptr;

  // self.fc1 = nn.Linear(input_dim, hidden_dim)
  ggml_tensor *fc1_w = nullptr;
  ggml_tensor *fc1_b = nullptr;

  // MLP ResNet blocks (2 by default)
  std::vector<MLPResNetBlockOFT> mlp_resnet_blocks;

  // Layer norm and FC after blocks
  // self.layer_norm2 = nn.LayerNorm(hidden_dim)
  ggml_tensor *layer_norm2_w = nullptr;
  ggml_tensor *layer_norm2_b = nullptr;

  // self.fc2 = nn.Linear(hidden_dim, output_dim)  -- output_dim = action_dim
  ggml_tensor *fc2_w = nullptr;
  ggml_tensor *fc2_b = nullptr;
};

// ==========================================
// ProprioProjector Model
// Projects proprioception state into LLM embedding space
// Input: (proprio_dim,) e.g., 8 for LIBERO
// Output: (llm_dim,) e.g., 4096 for Llama2-7B
// Architecture: fc1 -> GELU -> fc2
// ==========================================

struct ProprioProjectorParams {
  uint32_t proprio_dim = 8;       // Proprioception input dimension (8 for LIBERO)
  uint32_t llm_dim = 4096;        // LLM hidden dimension (4096 for Llama2-7B)
};

class ProprioProjectorModel : public BaseModel {
public:
  std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
  bool load_hparams(const ModelLoader &model_loader) override;
  std::vector<ggml_tensor *> build_graph(ggml_context *ctx0);

  ProprioProjectorParams hparams;

  // self.fc1 = nn.Linear(proprio_dim, llm_dim, bias=True)
  ggml_tensor *fc1_w = nullptr;
  ggml_tensor *fc1_b = nullptr;

  // self.fc2 = nn.Linear(llm_dim, llm_dim, bias=True)
  ggml_tensor *fc2_w = nullptr;
  ggml_tensor *fc2_b = nullptr;
};

// ==========================================
// TokenEmbedder Model
// Loads the token embedding table from LLM GGUF
// Used to convert token IDs to embeddings for single-batch processing
// ==========================================

class TokenEmbedderModel : public BaseModel {
public:
  std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;

  // Get embedding dimension
  int32_t get_embd_dim() const { return embd_dim_; }

  // Get embeddings for a list of tokens
  // Output: vector of size (tokens.size() * embd_dim)
  bool get_embeddings(const std::vector<int32_t> &tokens,
                      std::vector<float> &out) const;

  ggml_tensor *token_embd = nullptr;
  int32_t embd_dim_ = 4096;
  int32_t vocab_size_ = 32000;
};
