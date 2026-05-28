#include "pi05_context.h"
#include "build_graph.h"
#include "utils.h"
#include <cstdio>
#include <cmath>

bool Pi05Context::init(Pi05ActionExpert* expert, Pi05Memory* memory,
                       ggml_backend_sched_t sched, int max_nodes) {
    if (!expert || !memory || !sched) {
        fprintf(stderr, "Pi05Context::init: invalid parameters\n");
        return false;
    }

    expert_ = expert;
    memory_ = memory;
    sched_ = sched;
    max_nodes_ = max_nodes;
    initialized_ = true;

    printf("Pi05Context initialized\n");
    return true;
}

int Pi05Context::get_n_suffix() const {
    return expert_ ? expert_->hparams.action_horizon : 0;
}

void Pi05Context::clear() {
    if (memory_) {
        memory_->clear();
    }
    prefix_result_.reset();
    suffix_result_.reset();
    prefix_positions_.clear();
    suffix_positions_.clear();
    prefix_mask_data_.clear();
    suffix_mask_data_.clear();
}

void Pi05Context::build_prefix_graph(int n_prefix) {
    int pali_hidden = expert_->hparams.pali_hidden_size;
    int n_layer = expert_->hparams.num_hidden_layers;
    int kv_dim = expert_->hparams.head_dim * expert_->hparams.num_key_value_heads;
    float eps = expert_->hparams.rms_norm_eps;

    // Allocate compute context
    prefix_result_.buf_compute_meta.resize(max_nodes_ * ggml_tensor_overhead() + ggml_graph_overhead());
    struct ggml_init_params init_params = {
        prefix_result_.buf_compute_meta.size(),
        prefix_result_.buf_compute_meta.data(),
        true  // no_alloc
    };
    prefix_result_.ctx_compute.reset(ggml_init(init_params));
    prefix_result_.gf = ggml_new_graph_custom(prefix_result_.ctx_compute.get(), max_nodes_, false);

    ggml_context* ctx0 = prefix_result_.ctx_compute.get();

    // Create input tensors
    ggml_tensor* prefix_hidden = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, pali_hidden, n_prefix);
    ggml_set_name(prefix_hidden, "prefix_hidden");
    ggml_set_input(prefix_hidden);

    ggml_tensor* prefix_positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_prefix);
    ggml_set_name(prefix_positions, "prefix_positions");
    ggml_set_input(prefix_positions);

    ggml_tensor* prefix_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_prefix, n_prefix);
    ggml_set_name(prefix_mask, "prefix_attn_mask");
    ggml_set_input(prefix_mask);

    // Process prefix through all layers
    ggml_tensor* pali_cur = prefix_hidden;

    for (int il = 0; il < n_layer; il++) {
        // Create joint layer with PaliGemma weights
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

        // Add ggml_cpy nodes to write KV to persistent memory
        ggml_tensor* k_cpy = memory_->cpy_k(ctx0, layer_result.k_cache, 0, il);
        ggml_tensor* v_cpy = memory_->cpy_v(ctx0, layer_result.v_cache, 0, il);

        ggml_build_forward_expand(prefix_result_.gf, k_cpy);
        ggml_build_forward_expand(prefix_result_.gf, v_cpy);
    }

    // Mark final output
    ggml_set_name(pali_cur, "prefix_out");
    ggml_set_output(pali_cur);
    ggml_build_forward_expand(prefix_result_.gf, pali_cur);

    prefix_result_.cached_n_prefix = n_prefix;
    prefix_result_.cached_n_suffix = 0;
    prefix_result_.valid = true;
}

void Pi05Context::build_suffix_graph(int n_prefix, int n_suffix) {
    int hidden_size = expert_->hparams.hidden_size;
    int action_dim = expert_->hparams.action_dim;
    int action_horizon = expert_->hparams.action_horizon;
    int timestep_dim = expert_->hparams.timestep_sinusoidal_dim;
    int n_layer = expert_->hparams.num_hidden_layers;
    int kv_dim = expert_->hparams.head_dim * expert_->hparams.num_key_value_heads;
    float eps = expert_->hparams.rms_norm_eps;

    // Allocate compute context
    suffix_result_.buf_compute_meta.resize(max_nodes_ * ggml_tensor_overhead() + ggml_graph_overhead());
    struct ggml_init_params init_params = {
        suffix_result_.buf_compute_meta.size(),
        suffix_result_.buf_compute_meta.data(),
        true  // no_alloc
    };
    suffix_result_.ctx_compute.reset(ggml_init(init_params));
    suffix_result_.gf = ggml_new_graph_custom(suffix_result_.ctx_compute.get(), max_nodes_, false);

    ggml_context* ctx0 = suffix_result_.ctx_compute.get();

    // Create input tensors
    ggml_tensor* noisy_actions = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, action_dim, action_horizon);
    ggml_set_name(noisy_actions, "noisy_actions");
    ggml_set_input(noisy_actions);

    ggml_tensor* timestep_emb = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, timestep_dim);
    ggml_set_name(timestep_emb, "timestep_emb");
    ggml_set_input(timestep_emb);

    ggml_tensor* suffix_positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_suffix);
    ggml_set_name(suffix_positions, "suffix_positions");
    ggml_set_input(suffix_positions);

    // Create attention mask
    ggml_tensor* attn_mask = build_suffix_attn_mask(ctx0, n_prefix, n_suffix);
    ggml_set_input(attn_mask);

    // Timestep MLP
    ggml_tensor* timestep_proj = build_linear(ctx0, timestep_emb,
                                              expert_->timestep_mlp_in_w,
                                              expert_->timestep_mlp_in_b);
    timestep_proj = ggml_silu(ctx0, timestep_proj);
    timestep_proj = build_linear(ctx0, timestep_proj,
                                 expert_->timestep_mlp_out_w,
                                 expert_->timestep_mlp_out_b);
    ggml_tensor* adarms_cond = ggml_silu(ctx0, timestep_proj);

    // Embed actions
    ggml_tensor* suffix_cur = build_linear(ctx0, noisy_actions,
                                           expert_->action_in_w,
                                           expert_->action_in_b);

    // Phase 4: Process through all layers using build_suffix_layer_with_memory
    // This uses ggml_view + ggml_cpy instead of ggml_concat
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

        SuffixLayerWithMemoryResult layer_result = build_suffix_layer_with_memory(
            ctx0, suffix_cur,
            memory_, n_prefix,
            suffix_positions, attn_mask, adarms_cond,
            joint_layer, eps, il
        );

        // Add copy nodes to graph FIRST (before the layer output that reads from memory)
        // This ensures suffix K/V is written to memory before attention reads it
        ggml_build_forward_expand(suffix_result_.gf, layer_result.k_cpy);
        ggml_build_forward_expand(suffix_result_.gf, layer_result.v_cpy);

        suffix_cur = layer_result.suffix_out;
    }

    // Output AdaLN and projection
    AdaRMSNormResult adaln_out = build_ada_rms_norm(ctx0, suffix_cur,
                                                     expert_->output_adaln_w,
                                                     expert_->output_adaln_b,
                                                     adarms_cond, eps);
    ggml_tensor* out = adaln_out.output;
    out = build_linear(ctx0, out, expert_->action_out_w, expert_->action_out_b);

    ggml_set_name(out, "velocity_output");
    ggml_set_output(out);
    ggml_build_forward_expand(suffix_result_.gf, out);

    suffix_result_.cached_n_prefix = n_prefix;
    suffix_result_.cached_n_suffix = n_suffix;
    suffix_result_.valid = true;
}

void Pi05Context::set_prefix_inputs(const std::vector<float>& prefix_hidden, int n_prefix) {
    set_input_f32(prefix_result_.gf, "prefix_hidden", prefix_hidden);

    // Fill positions [0, 1, ..., n_prefix-1]
    prefix_positions_.resize(n_prefix);
    for (int i = 0; i < n_prefix; i++) {
        prefix_positions_[i] = i;
    }
    set_input_i32(prefix_result_.gf, "prefix_positions", prefix_positions_);

    // Fill mask (all zeros for bidirectional)
    prefix_mask_data_.resize(n_prefix * n_prefix, 0.0f);
    set_input_f32(prefix_result_.gf, "prefix_attn_mask", prefix_mask_data_);
}

void Pi05Context::set_suffix_inputs(const std::vector<float>& noisy_actions,
                                    const std::vector<float>& timestep_emb,
                                    int masked_start, int masked_end) {
    set_input_f32(suffix_result_.gf, "noisy_actions", noisy_actions);
    set_input_f32(suffix_result_.gf, "timestep_emb", timestep_emb);

    int n_prefix = memory_->get_prefix_len();
    int n_suffix = get_n_suffix();

    // Fill suffix positions [n_prefix, n_prefix+1, ..., n_prefix+n_suffix-1]
    suffix_positions_.resize(n_suffix);
    for (int i = 0; i < n_suffix; i++) {
        suffix_positions_[i] = n_prefix + i;
    }
    set_input_i32(suffix_result_.gf, "suffix_positions", suffix_positions_);

    // Fill suffix attention mask
    int n_kv = n_prefix + n_suffix;
    suffix_mask_data_.resize(n_kv * n_suffix);
    fill_suffix_attn_mask(suffix_mask_data_.data(), n_prefix, n_suffix, masked_start, masked_end);
    set_input_f32(suffix_result_.gf, "suffix_attn_mask", suffix_mask_data_);
}

bool Pi05Context::process_prefix(const std::vector<float>& prefix_hidden, int n_prefix) {
    if (!initialized_) {
        fprintf(stderr, "Pi05Context::process_prefix: not initialized\n");
        return false;
    }

    // Update memory state
    memory_->set_prefix_len(n_prefix);

    // Check if we need to rebuild the graph
    if (!prefix_result_.can_reuse(n_prefix, 0)) {
        build_prefix_graph(n_prefix);
    }

    // Allocate graph
    ggml_backend_sched_reset(sched_);
    if (!ggml_backend_sched_alloc_graph(sched_, prefix_result_.gf)) {
        fprintf(stderr, "Pi05Context::process_prefix: failed to allocate graph\n");
        return false;
    }

    // Set inputs
    set_prefix_inputs(prefix_hidden, n_prefix);

    // Execute graph
    auto status = ggml_backend_sched_graph_compute(sched_, prefix_result_.gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "Pi05Context::process_prefix: compute failed with status %d\n", status);
        return false;
    }

    printf("Pi05Context: prefix processed, n_prefix=%d\n", n_prefix);
    return true;
}

bool Pi05Context::process_suffix(const std::vector<float>& noisy_actions,
                                 const std::vector<float>& timestep_emb,
                                 std::vector<float>& velocity_out,
                                 int masked_start, int masked_end) {
    if (!initialized_) {
        fprintf(stderr, "Pi05Context::process_suffix: not initialized\n");
        return false;
    }

    int n_prefix = memory_->get_prefix_len();
    int n_suffix = get_n_suffix();

    if (n_prefix <= 0) {
        fprintf(stderr, "Pi05Context::process_suffix: prefix not processed yet\n");
        return false;
    }

    // Check if we need to rebuild the graph
    if (!suffix_result_.can_reuse(n_prefix, n_suffix)) {
        build_suffix_graph(n_prefix, n_suffix);
    }

    // Allocate graph
    ggml_backend_sched_reset(sched_);
    if (!ggml_backend_sched_alloc_graph(sched_, suffix_result_.gf)) {
        fprintf(stderr, "Pi05Context::process_suffix: failed to allocate graph\n");
        return false;
    }

    // Set inputs
    set_suffix_inputs(noisy_actions, timestep_emb, masked_start, masked_end);

    // Execute graph
    auto status = ggml_backend_sched_graph_compute(sched_, suffix_result_.gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "Pi05Context::process_suffix: compute failed with status %d\n", status);
        return false;
    }

    // Extract output
    ggml_tensor* out_tensor = ggml_graph_node(suffix_result_.gf, -1);
    velocity_out.resize(ggml_nelements(out_tensor));
    ggml_backend_tensor_get(out_tensor, velocity_out.data(), 0, ggml_nbytes(out_tensor));

    return true;
}
