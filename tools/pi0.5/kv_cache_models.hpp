#pragma once

#include "model_defs.h"
#include "build_graph.h"
#include <cstring>

// Model wrapper for prefix-only forward pass that computes and outputs KV caches
// This is used once per observation to compute the prefix KV cache
class PrefixCacheModel : public BaseModel {
public:
    PrefixCacheModel() = default;

    void set_action_expert(Pi05ActionExpert *expert) {
        expert_ = expert;
    }

    void set_n_prefix(int n_prefix) {
        n_prefix_ = n_prefix;
    }

    // Phase 1: Set GPU-persistent KV cache tensors
    void set_gpu_kv_cache(ggml_tensor** k_cache_gpu, ggml_tensor** v_cache_gpu, int n_layers) {
        k_cache_gpu_ = k_cache_gpu;
        v_cache_gpu_ = v_cache_gpu;
        n_layers_ = n_layers;
        use_gpu_cache_ = true;
    }

    bool load_hparams(const ModelLoader &model_loader) override {
        return true;  // Uses hparams from expert_
    }

    std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override {
        return {};  // No tensors to load, uses expert_'s tensors
    }

    std::vector<ggml_tensor *> build_graph(ggml_context *ctx0) override {
        if (!expert_) {
            throw std::runtime_error("PrefixCacheModel: expert_ not set");
        }

        int pali_hidden = expert_->hparams.pali_hidden_size;
        int n_prefix = n_prefix_;
        int n_layer = expert_->hparams.num_hidden_layers;
        int kv_dim = expert_->hparams.head_dim * expert_->hparams.num_key_value_heads;
        float eps = expert_->hparams.rms_norm_eps;

        // Create input tensor for prefix hidden states
        ggml_tensor *prefix_hidden = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, pali_hidden, n_prefix);
        ggml_set_name(prefix_hidden, "prefix_hidden");
        ggml_set_input(prefix_hidden);

        // Create prefix positions tensor [0, 1, 2, ..., n_prefix-1]
        ggml_tensor *prefix_positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_prefix);
        ggml_set_name(prefix_positions, "prefix_positions");
        ggml_set_input(prefix_positions);

        // Create prefix self-attention mask (all zeros for bidirectional)
        ggml_tensor *prefix_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_prefix, n_prefix);
        ggml_set_name(prefix_mask, "prefix_attn_mask");
        ggml_set_input(prefix_mask);

        // Process prefix through all layers
        ggml_tensor *pali_cur = prefix_hidden;
        std::vector<ggml_tensor *> outputs;

        for (int il = 0; il < n_layer; il++) {
            // Create joint layer with PaliGemma weights only
            GemmaJointLayer joint_layer = {};
            joint_layer.pali_attn_norm_w = expert_->pali_layers[il].attn_norm_w;
            joint_layer.pali_q_w = expert_->pali_layers[il].q_w;
            joint_layer.pali_k_w = expert_->pali_layers[il].k_w;
            joint_layer.pali_v_w = expert_->pali_layers[il].v_w;
            joint_layer.pali_o_w = expert_->pali_layers[il].o_w;
            joint_layer.pali_ffn_norm_w = expert_->pali_layers[il].ffn_norm_w;
            joint_layer.pali_gate_w = expert_->pali_layers[il].ffn_gate_w;
            joint_layer.pali_up_w = expert_->pali_layers[il].ffn_up_w;
            joint_layer.pali_down_w = expert_->pali_layers[il].ffn_down_w;

            PrefixLayerResult layer_result = build_prefix_layer_forward(
                ctx0, pali_cur, prefix_positions, prefix_mask,
                joint_layer, eps, il
            );

            pali_cur = layer_result.prefix_out;

            // Phase 1: If GPU cache is set, add ggml_cpy nodes to write KV to persistent tensors
            if (use_gpu_cache_ && k_cache_gpu_ && v_cache_gpu_) {
                // Create view of the persistent GPU tensor for the prefix portion
                ggml_tensor *k_dst = ggml_view_2d(ctx0, k_cache_gpu_[il],
                                                   kv_dim, n_prefix,
                                                   k_cache_gpu_[il]->nb[1], 0);
                ggml_tensor *v_dst = ggml_view_2d(ctx0, v_cache_gpu_[il],
                                                   kv_dim, n_prefix,
                                                   v_cache_gpu_[il]->nb[1], 0);

                // Add ggml_cpy nodes - these will write computed KV to persistent tensors
                ggml_tensor *k_cpy = ggml_cpy(ctx0, layer_result.k_cache, k_dst);
                ggml_tensor *v_cpy = ggml_cpy(ctx0, layer_result.v_cache, v_dst);

                std::string k_cpy_name = "k_cpy_" + std::to_string(il);
                std::string v_cpy_name = "v_cpy_" + std::to_string(il);
                ggml_set_name(k_cpy, k_cpy_name.c_str());
                ggml_set_name(v_cpy, v_cpy_name.c_str());

                // These copy operations must be in the graph to execute
                outputs.push_back(k_cpy);
                outputs.push_back(v_cpy);
            } else {
                // Legacy path: mark K/V cache tensors as outputs for CPU extraction
                std::string k_name = "k_cache_" + std::to_string(il);
                std::string v_name = "v_cache_" + std::to_string(il);
                ggml_set_name(layer_result.k_cache, k_name.c_str());
                ggml_set_output(layer_result.k_cache);
                ggml_set_name(layer_result.v_cache, v_name.c_str());
                ggml_set_output(layer_result.v_cache);

                outputs.push_back(layer_result.k_cache);
                outputs.push_back(layer_result.v_cache);
            }
        }

        // Mark final prefix output (for debugging/verification)
        ggml_set_name(pali_cur, "prefix_out");
        ggml_set_output(pali_cur);
        outputs.push_back(pali_cur);

        return outputs;
    }

private:
    Pi05ActionExpert *expert_ = nullptr;
    int n_prefix_ = 256;

    // Phase 1: GPU-persistent KV cache
    bool use_gpu_cache_ = false;
    int n_layers_ = 18;
    ggml_tensor** k_cache_gpu_ = nullptr;
    ggml_tensor** v_cache_gpu_ = nullptr;
};

// Model wrapper for suffix forward pass using cached KV
// This is called for each ODE step during flow matching
class SuffixCacheModel : public BaseModel {
public:
    SuffixCacheModel() = default;

    void set_action_expert(Pi05ActionExpert *expert) {
        expert_ = expert;
    }

    void set_dimensions(int n_prefix, int n_suffix) {
        n_prefix_ = n_prefix;
        n_suffix_ = n_suffix;
    }

    // Phase 1: Set GPU-persistent KV cache tensors
    void set_gpu_kv_cache(ggml_tensor** k_cache_gpu, ggml_tensor** v_cache_gpu, int n_layers) {
        k_cache_gpu_ = k_cache_gpu;
        v_cache_gpu_ = v_cache_gpu;
        n_layers_ = n_layers;
        use_gpu_cache_ = true;
    }

    bool load_hparams(const ModelLoader &model_loader) override {
        return true;
    }

    std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override {
        return {};
    }

    std::vector<ggml_tensor *> build_graph(ggml_context *ctx0) override {
        if (!expert_) {
            throw std::runtime_error("SuffixCacheModel: expert_ not set");
        }

        int hidden_size = expert_->hparams.hidden_size;
        int action_dim = expert_->hparams.action_dim;
        int action_horizon = expert_->hparams.action_horizon;
        int timestep_dim = expert_->hparams.timestep_sinusoidal_dim;
        int n_layer = expert_->hparams.num_hidden_layers;
        int head_dim = expert_->hparams.head_dim;
        int n_kv_heads = expert_->hparams.num_key_value_heads;
        float eps = expert_->hparams.rms_norm_eps;

        int n_prefix = n_prefix_;
        int n_suffix = n_suffix_;
        int n_total = n_prefix + n_suffix;
        int kv_dim = head_dim * n_kv_heads;  // 256 * 1 = 256

        // Create input tensors
        ggml_tensor *noisy_actions = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, action_dim, action_horizon);
        ggml_set_name(noisy_actions, "noisy_actions");
        ggml_set_input(noisy_actions);

        ggml_tensor *timestep_emb = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, timestep_dim);
        ggml_set_name(timestep_emb, "timestep_emb");
        ggml_set_input(timestep_emb);

        // Create suffix positions tensor (starting at n_prefix)
        ggml_tensor *suffix_positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_suffix);
        ggml_set_name(suffix_positions, "suffix_positions");
        ggml_set_input(suffix_positions);

        // Create attention mask for suffix queries attending to full sequence
        ggml_tensor *attn_mask = build_suffix_attn_mask(ctx0, n_prefix, n_suffix);
        ggml_set_input(attn_mask);

        // Get KV cache tensors - either from GPU persistent storage or create input tensors
        std::vector<ggml_tensor *> k_caches(n_layer);
        std::vector<ggml_tensor *> v_caches(n_layer);

        if (use_gpu_cache_ && k_cache_gpu_ && v_cache_gpu_) {
            // Phase 1: Use ggml_view to reference GPU-persistent KV cache
            for (int il = 0; il < n_layer; il++) {
                // Create view of the prefix portion of persistent GPU cache
                k_caches[il] = ggml_view_2d(ctx0, k_cache_gpu_[il],
                                             kv_dim, n_prefix,
                                             k_cache_gpu_[il]->nb[1], 0);
                v_caches[il] = ggml_view_2d(ctx0, v_cache_gpu_[il],
                                             kv_dim, n_prefix,
                                             v_cache_gpu_[il]->nb[1], 0);

                std::string k_name = "k_view_" + std::to_string(il);
                std::string v_name = "v_view_" + std::to_string(il);
                ggml_set_name(k_caches[il], k_name.c_str());
                ggml_set_name(v_caches[il], v_name.c_str());
            }
        } else {
            // Legacy path: create input tensors for CPU-based KV cache
            for (int il = 0; il < n_layer; il++) {
                std::string k_name = "k_cache_" + std::to_string(il);
                std::string v_name = "v_cache_" + std::to_string(il);

                k_caches[il] = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, kv_dim, n_prefix);
                ggml_set_name(k_caches[il], k_name.c_str());
                ggml_set_input(k_caches[il]);

                v_caches[il] = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, kv_dim, n_prefix);
                ggml_set_name(v_caches[il], v_name.c_str());
                ggml_set_input(v_caches[il]);
            }
        }

        // Timestep MLP: sinusoidal -> linear -> silu -> linear -> silu
        ggml_tensor *timestep_proj = build_linear(ctx0, timestep_emb,
                                                  expert_->timestep_mlp_in_w,
                                                  expert_->timestep_mlp_in_b);
        timestep_proj = ggml_silu(ctx0, timestep_proj);
        timestep_proj = build_linear(ctx0, timestep_proj,
                                     expert_->timestep_mlp_out_w,
                                     expert_->timestep_mlp_out_b);
        ggml_tensor *adarms_cond = ggml_silu(ctx0, timestep_proj);

        // Embed actions: [action_dim, action_horizon] -> [hidden_size, action_horizon]
        ggml_tensor *suffix_cur = build_linear(ctx0, noisy_actions,
                                               expert_->action_in_w,
                                               expert_->action_in_b);

        // Process through all layers using cached KV
        for (int il = 0; il < n_layer; il++) {
            GemmaJointLayer joint_layer = {};
            joint_layer.expert_attn_adaln_w = expert_->layers[il].attn_adaln_w;
            joint_layer.expert_attn_adaln_b = expert_->layers[il].attn_adaln_b;
            joint_layer.expert_q_w = expert_->layers[il].q_w;
            joint_layer.expert_k_w = expert_->layers[il].k_w;
            joint_layer.expert_v_w = expert_->layers[il].v_w;
            joint_layer.expert_o_w = expert_->layers[il].o_w;
            joint_layer.expert_ffn_adaln_w = expert_->layers[il].ffn_adaln_w;
            joint_layer.expert_ffn_adaln_b = expert_->layers[il].ffn_adaln_b;
            joint_layer.expert_gate_w = expert_->layers[il].ffn_gate_w;
            joint_layer.expert_up_w = expert_->layers[il].ffn_up_w;
            joint_layer.expert_down_w = expert_->layers[il].ffn_down_w;

            suffix_cur = build_suffix_layer_with_cache(
                ctx0, suffix_cur,
                k_caches[il], v_caches[il],
                suffix_positions, attn_mask, adarms_cond,
                joint_layer, eps, il
            );
        }

        // Output AdaLN and projection
        AdaRMSNormResult adaln_out = build_ada_rms_norm(ctx0, suffix_cur,
                                                         expert_->output_adaln_w,
                                                         expert_->output_adaln_b,
                                                         adarms_cond, eps);
        ggml_tensor *out = adaln_out.output;
        out = build_linear(ctx0, out, expert_->action_out_w, expert_->action_out_b);

        ggml_set_name(out, "velocity_output");
        ggml_set_output(out);

        return {out};
    }

private:
    Pi05ActionExpert *expert_ = nullptr;
    int n_prefix_ = 256;
    int n_suffix_ = 50;

    // Phase 1: GPU-persistent KV cache
    bool use_gpu_cache_ = false;
    int n_layers_ = 18;
    ggml_tensor** k_cache_gpu_ = nullptr;
    ggml_tensor** v_cache_gpu_ = nullptr;
};
