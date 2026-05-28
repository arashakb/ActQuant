#pragma once

#include "build_graph.h"
#include "ctx_manager.h"
#include "model_loader.h"
#include "pi05_config.h"
#include "ggml.h"
#include <vector>

// KV Cache for joint transformer
// Stores computed K/V values for prefix tokens to avoid recomputation during ODE steps
// Layout: [n_layer, n_kv_heads, n_prefix, head_dim] = [18, 1, n_prefix, 256]
struct KVCache {
    std::vector<float> k;   // Key cache for all layers
    std::vector<float> v;   // Value cache for all layers
    int n_layers;           // Number of transformer layers (18)
    int n_kv_heads;         // Number of KV heads (1 for GQA)
    int n_prefix;           // Number of prefix tokens
    int head_dim;           // Head dimension (256)

    // Size of cache per layer: n_kv_heads * n_prefix * head_dim
    size_t layer_size() const { return n_kv_heads * n_prefix * head_dim; }
    // Total cache size
    size_t total_size() const { return n_layers * layer_size(); }

    // Get K cache for specific layer: returns pointer to [n_kv_heads * n_prefix * head_dim]
    float* k_layer(int il) { return k.data() + il * layer_size(); }
    const float* k_layer(int il) const { return k.data() + il * layer_size(); }

    // Get V cache for specific layer: returns pointer to [n_kv_heads * n_prefix * head_dim]
    float* v_layer(int il) { return v.data() + il * layer_size(); }
    const float* v_layer(int il) const { return v.data() + il * layer_size(); }

    // Allocate cache for given dimensions
    void allocate(int layers, int kv_heads, int prefix_len, int dim) {
        n_layers = layers;
        n_kv_heads = kv_heads;
        n_prefix = prefix_len;
        head_dim = dim;
        k.resize(total_size());
        v.resize(total_size());
    }

    bool empty() const { return k.empty(); }
};

class BaseModel {
public:
    bool load_tensors(ModelLoader &model_loader, ContextManager &ctx_manager);
    virtual bool load_hparams(const ModelLoader &model_loader) { return true; }
    virtual std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) { return {}; }
    // For InferenceSession template - returns output tensors
    virtual std::vector<ggml_tensor *> build_graph(ggml_context *ctx0) { return {}; }
    virtual ~BaseModel() = default;
};

struct Pi0VisionLayer {
    ggml_tensor *ln_1_w = nullptr;
    ggml_tensor *ln_1_b = nullptr;
    ggml_tensor *q_w = nullptr;
    ggml_tensor *q_b = nullptr;
    ggml_tensor *k_w = nullptr;
    ggml_tensor *k_b = nullptr;
    ggml_tensor *v_w = nullptr;
    ggml_tensor *v_b = nullptr;
    ggml_tensor *o_w = nullptr;
    ggml_tensor *o_b = nullptr;
    ggml_tensor *ln_2_w = nullptr;
    ggml_tensor *ln_2_b = nullptr;
    ggml_tensor *ff_up_w = nullptr;
    ggml_tensor *ff_up_b = nullptr;
    ggml_tensor *ff_down_w = nullptr;
    ggml_tensor *ff_down_b = nullptr;
};

class Pi05VisionModel : public BaseModel {
public:
    std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
    bool load_hparams(const ModelLoader &model_loader) override;
    std::vector<ggml_tensor *> build_graph(ggml_context *ctx0) override;
    ggml_tensor *build_graph(ggml_context *ctx0, ggml_tensor *inp);

    pi05_vision_config hparams;

    ggml_tensor *patch_embd_w = nullptr;
    ggml_tensor *patch_embd_b = nullptr;
    ggml_tensor *position_embd = nullptr;
    ggml_tensor *post_ln_w = nullptr;
    ggml_tensor *post_ln_b = nullptr;

    std::vector<Pi0VisionLayer> layers;
};

class Pi05ProjectorModel : public BaseModel {
public:
    std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
    bool load_hparams(const ModelLoader &model_loader) override;
    std::vector<ggml_tensor *> build_graph(ggml_context *ctx0) override;
    ggml_tensor *build_graph(ggml_context *ctx0, ggml_tensor *inp);

    int input_dim = 1152;   // SigLIP hidden size
    int output_dim = 2048;  // Gemma hidden size

    ggml_tensor *weight = nullptr;
    ggml_tensor *bias = nullptr;
};

struct Pi0TextLayer {
    ggml_tensor *attn_norm_w = nullptr;
    ggml_tensor *q_w = nullptr;
    ggml_tensor *k_w = nullptr;
    ggml_tensor *v_w = nullptr;
    ggml_tensor *o_w = nullptr;
    ggml_tensor *ffn_norm_w = nullptr;
    ggml_tensor *ffn_gate_w = nullptr;
    ggml_tensor *ffn_up_w = nullptr;
    ggml_tensor *ffn_down_w = nullptr;
};

class Pi05TextModel : public BaseModel {
public:
    std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
    bool load_hparams(const ModelLoader &model_loader) override;
    ggml_tensor *build_graph(ggml_context *ctx0, ggml_tensor *inp,
                             ggml_tensor *kv_cache_k, ggml_tensor *kv_cache_v,
                             int n_past, int n_tokens);

    pi05_text_config hparams;

    ggml_tensor *token_embd = nullptr;
    ggml_tensor *output_norm_w = nullptr;

    std::vector<Pi0TextLayer> layers;
};

struct Pi0ActionLayer {
    // Adaptive LayerNorm (AdaLN) - dense projection: [3*hidden_size, hidden_size]
    // Projects timestep embedding to scale, shift, gate for modulation
    ggml_tensor *attn_adaln_w = nullptr;   // [3072, 1024] -> scale, shift, gate
    ggml_tensor *attn_adaln_b = nullptr;   // [3072]
    ggml_tensor *q_w = nullptr;
    ggml_tensor *k_w = nullptr;
    ggml_tensor *v_w = nullptr;
    ggml_tensor *o_w = nullptr;
    ggml_tensor *ffn_adaln_w = nullptr;    // [3072, 1024] -> scale, shift, gate
    ggml_tensor *ffn_adaln_b = nullptr;    // [3072]
    ggml_tensor *ffn_gate_w = nullptr;
    ggml_tensor *ffn_up_w = nullptr;
    ggml_tensor *ffn_down_w = nullptr;
};

// PaliGemma layer weights for cross-attention (prefix processing with self-attention)
struct PaliGemmaLayer {
    ggml_tensor *attn_norm_w = nullptr;  // RMSNorm weight
    ggml_tensor *q_w = nullptr;          // Q projection for prefix self-attention
    ggml_tensor *k_w = nullptr;          // K projection for cross-attention
    ggml_tensor *v_w = nullptr;          // V projection for cross-attention
    ggml_tensor *o_w = nullptr;          // O projection for prefix self-attention output
    ggml_tensor *ffn_norm_w = nullptr;   // Post-attn RMSNorm weight
    ggml_tensor *ffn_gate_w = nullptr;
    ggml_tensor *ffn_up_w = nullptr;
    ggml_tensor *ffn_down_w = nullptr;
};

// Combined joint layer structure for unified prefix-suffix processing
// Matches Python GemmaJointLayer for easier verification
struct GemmaJointLayer {
    // === Prefix (PaliGemma) weights - 2048 dim hidden ===
    // Attention
    ggml_tensor *pali_attn_norm_w = nullptr;   // RMSNorm weight [2048]
    ggml_tensor *pali_q_w = nullptr;           // [2048, 2048] -> 8 heads * 256 head_dim
    ggml_tensor *pali_k_w = nullptr;           // [256, 2048]  -> 1 KV head * 256 head_dim
    ggml_tensor *pali_v_w = nullptr;           // [256, 2048]
    ggml_tensor *pali_o_w = nullptr;           // [2048, 2048]
    // MLP
    ggml_tensor *pali_ffn_norm_w = nullptr;    // RMSNorm weight [2048]
    ggml_tensor *pali_gate_w = nullptr;        // [16384, 2048]
    ggml_tensor *pali_up_w = nullptr;          // [16384, 2048]
    ggml_tensor *pali_down_w = nullptr;        // [2048, 16384]

    // === Suffix (Expert) weights - 1024 dim hidden ===
    // Attention with AdaRMSNorm
    ggml_tensor *expert_attn_adaln_w = nullptr;  // [3072, 1024] -> scale, shift, gate
    ggml_tensor *expert_attn_adaln_b = nullptr;  // [3072]
    ggml_tensor *expert_q_w = nullptr;           // [2048, 1024] -> 8 heads * 256 head_dim
    ggml_tensor *expert_k_w = nullptr;           // [256, 1024]  -> 1 KV head * 256 head_dim
    ggml_tensor *expert_v_w = nullptr;           // [256, 1024]
    ggml_tensor *expert_o_w = nullptr;           // [1024, 2048]
    // MLP with AdaRMSNorm
    ggml_tensor *expert_ffn_adaln_w = nullptr;   // [3072, 1024] -> scale, shift, gate
    ggml_tensor *expert_ffn_adaln_b = nullptr;   // [3072]
    ggml_tensor *expert_gate_w = nullptr;        // [4096, 1024]
    ggml_tensor *expert_up_w = nullptr;          // [4096, 1024]
    ggml_tensor *expert_down_w = nullptr;        // [1024, 4096]
};

// Result of prefix-only forward through action expert (for KV cache computation)
struct PrefixForwardResult {
    std::vector<ggml_tensor *> outputs;  // Output tensors for graph
    std::vector<ggml_tensor *> k_caches; // K cache tensors per layer [n_layers]
    std::vector<ggml_tensor *> v_caches; // V cache tensors per layer [n_layers]
};

class Pi05ActionExpert : public BaseModel {
public:
    std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
    bool load_hparams(const ModelLoader &model_loader) override;
    std::vector<ggml_tensor *> build_graph(ggml_context *ctx0) override;
    // New cross-attention interface: prefix_hidden [pali_hidden, n_prefix]
    ggml_tensor *build_graph(ggml_context *ctx0, ggml_tensor *prefix_hidden,
                             ggml_tensor *noisy_actions, ggml_tensor *timestep_emb);

    // ============ KV Cache Operations ============

    // Build graph for prefix-only forward (computes and returns KV cache)
    // prefix_hidden: [pali_hidden_size, n_prefix] - prefix embeddings
    // Returns: PrefixForwardResult with output and KV cache tensors
    PrefixForwardResult build_graph_prefix_only(ggml_context *ctx0,
                                                 ggml_tensor *prefix_hidden);

    // Build graph for suffix-only forward using cached KV
    // suffix_hidden: [hidden_size, n_suffix] - action embeddings
    // prefix_k_caches: [n_layers] tensors, each [head_dim * n_kv_heads, n_prefix]
    // prefix_v_caches: [n_layers] tensors, each [head_dim * n_kv_heads, n_prefix]
    // timestep_emb: [timestep_dim] - sinusoidal timestep embedding
    ggml_tensor *build_graph_with_cache(ggml_context *ctx0,
                                         ggml_tensor *suffix_hidden,
                                         const std::vector<ggml_tensor *> &prefix_k_caches,
                                         const std::vector<ggml_tensor *> &prefix_v_caches,
                                         ggml_tensor *timestep_emb);

    pi05_action_config hparams;

    // NOTE: Pi0.5 does NOT have vlm_proj - prefix embeddings go directly to joint attention
    ggml_tensor *timestep_mlp_in_w = nullptr;
    ggml_tensor *timestep_mlp_in_b = nullptr;
    ggml_tensor *timestep_mlp_out_w = nullptr;
    ggml_tensor *timestep_mlp_out_b = nullptr;
    ggml_tensor *action_in_w = nullptr;
    ggml_tensor *action_in_b = nullptr;
    ggml_tensor *action_out_w = nullptr;
    ggml_tensor *action_out_b = nullptr;
    // NOTE: Pi05.5 does NOT have state_proj - state is embedded via action_in_proj
    // Output AdaLN
    ggml_tensor *output_adaln_w = nullptr;   // [3072, 1024]
    ggml_tensor *output_adaln_b = nullptr;   // [3072]

    std::vector<Pi0ActionLayer> layers;
    // PaliGemma layers for cross-attention
    std::vector<PaliGemmaLayer> pali_layers;
};
