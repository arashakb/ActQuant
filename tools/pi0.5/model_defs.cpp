#include "model_defs.h"
#include "build_graph.h"
#include "utils.h"
#include <cmath>
#include <fstream>
#include <map>

bool BaseModel::load_tensors(ModelLoader &model_loader, ContextManager &ctx_manager) {
    std::map<std::string, size_t> tensor_offset;
    gguf_context_ptr &ctx_gguf = model_loader.ctx_gguf_;

    ctx_manager.ctx_data_ = std::move(model_loader.ctx_meta_);
    ggml_context *ctx = ctx_manager.ctx_data_.get();

    for (int64_t i = 0; i < gguf_get_n_tensors(ctx_gguf.get()); ++i) {
        const char *name = gguf_get_tensor_name(ctx_gguf.get(), i);
        tensor_offset[name] = gguf_get_data_offset(ctx_gguf.get()) +
                              gguf_get_tensor_offset(ctx_gguf.get(), i);
    }

    std::vector<ggml_tensor *> tensors_to_load = get_tensors_to_load(ctx);

    std::vector<uint8_t> read_buf;
    auto fin = std::ifstream(model_loader.fname_, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open %s\n", __func__, model_loader.fname_.c_str());
        return false;
    }

    ggml_backend_buffer_type_t buft =
        ggml_backend_get_default_buffer_type(ctx_manager.backend_.get());
    ctx_manager.buffer_.reset(ggml_backend_alloc_ctx_tensors_from_buft(
        ctx_manager.ctx_data_.get(), buft));
    ggml_backend_buffer_set_usage(ctx_manager.buffer_.get(),
                                  GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    for (auto &cur : tensors_to_load) {
        auto it = tensor_offset.find(cur->name);
        if (it == tensor_offset.end()) {
            fprintf(stderr, "%s: tensor %s not found in file\n", __func__, cur->name);
            continue;
        }
        const size_t offset = it->second;
        fin.seekg(offset, std::ios::beg);
        if (!fin) {
            fprintf(stderr, "%s: failed to seek for tensor %s\n", __func__, cur->name);
            return false;
        }
        size_t num_bytes = ggml_nbytes(cur);
        if (ggml_backend_buft_is_host(buft)) {
            fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
        } else {
            read_buf.resize(num_bytes);
            fin.read(reinterpret_cast<char *>(read_buf.data()), num_bytes);
            ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
        }
    }
    fin.close();

    printf("%s: loaded %zu tensors from %s\n", __func__, tensors_to_load.size(),
           model_loader.fname_.c_str());
    return true;
}

// Pi05VisionModel
std::vector<ggml_tensor *> Pi05VisionModel::get_tensors_to_load(ggml_context *ctx) {
    std::vector<ggml_tensor *> tensors;

    patch_embd_w = get_tensor(ctx, "v.patch_embd.weight", tensors);
    patch_embd_b = get_tensor(ctx, "v.patch_embd.bias", tensors, false);
    position_embd = get_tensor(ctx, "v.position_embd.weight", tensors, false);
    post_ln_w = get_tensor(ctx, "v.post_ln.weight", tensors, false);
    post_ln_b = get_tensor(ctx, "v.post_ln.bias", tensors, false);

    int n_layer = hparams.num_hidden_layers;
    layers.resize(n_layer);

    for (int il = 0; il < n_layer; ++il) {
        auto &layer = layers[il];
        layer.ln_1_w = get_tensor(ctx, string_format("v.blk.%d.ln1.weight", il), tensors);
        layer.ln_1_b = get_tensor(ctx, string_format("v.blk.%d.ln1.bias", il), tensors, false);
        layer.q_w = get_tensor(ctx, string_format("v.blk.%d.attn_q.weight", il), tensors);
        layer.q_b = get_tensor(ctx, string_format("v.blk.%d.attn_q.bias", il), tensors, false);
        layer.k_w = get_tensor(ctx, string_format("v.blk.%d.attn_k.weight", il), tensors);
        layer.k_b = get_tensor(ctx, string_format("v.blk.%d.attn_k.bias", il), tensors, false);
        layer.v_w = get_tensor(ctx, string_format("v.blk.%d.attn_v.weight", il), tensors);
        layer.v_b = get_tensor(ctx, string_format("v.blk.%d.attn_v.bias", il), tensors, false);
        layer.o_w = get_tensor(ctx, string_format("v.blk.%d.attn_out.weight", il), tensors);
        layer.o_b = get_tensor(ctx, string_format("v.blk.%d.attn_out.bias", il), tensors, false);
        layer.ln_2_w = get_tensor(ctx, string_format("v.blk.%d.ln2.weight", il), tensors);
        layer.ln_2_b = get_tensor(ctx, string_format("v.blk.%d.ln2.bias", il), tensors, false);
        layer.ff_up_w = get_tensor(ctx, string_format("v.blk.%d.ffn_up.weight", il), tensors);
        layer.ff_up_b = get_tensor(ctx, string_format("v.blk.%d.ffn_up.bias", il), tensors, false);
        layer.ff_down_w = get_tensor(ctx, string_format("v.blk.%d.ffn_down.weight", il), tensors);
        layer.ff_down_b = get_tensor(ctx, string_format("v.blk.%d.ffn_down.bias", il), tensors, false);
    }

    return tensors;
}

bool Pi05VisionModel::load_hparams(const ModelLoader &model_loader) {
    model_loader.get_u32("clip.vision.embedding_length", hparams.hidden_size, false);
    model_loader.get_u32("clip.vision.attention.head_count", hparams.num_attention_heads, false);
    model_loader.get_u32("clip.vision.feed_forward_length", hparams.intermediate_size, false);
    model_loader.get_u32("clip.vision.block_count", hparams.num_hidden_layers, false);
    model_loader.get_u32("clip.vision.projection_dim", hparams.projection_dim, false);
    model_loader.get_u32("clip.vision.image_size", hparams.image_size, false);
    model_loader.get_u32("clip.vision.patch_size", hparams.patch_size, false);
    model_loader.get_f32("clip.vision.attention.layer_norm_epsilon", hparams.layer_norm_eps, false);
    return true;
}

std::vector<ggml_tensor *> Pi05VisionModel::build_graph(ggml_context *ctx0) {
    int n_patches = hparams.num_patches();
    int hidden_size = hparams.hidden_size;
    int patch_size = hparams.patch_size;
    int image_size = hparams.image_size;

    // Create raw image input tensor (H, W, C)
    ggml_tensor *inp_raw = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, image_size, image_size, 3);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);

    // Patch embedding via conv2d
    ggml_tensor *patches = ggml_conv_2d(ctx0, patch_embd_w, inp_raw,
                                        patch_size, patch_size, 0, 0, 1, 1);
    patches = ggml_reshape_2d(ctx0, patches, n_patches, hidden_size);
    patches = ggml_cont(ctx0, ggml_transpose(ctx0, patches));

    if (patch_embd_b) {
        patches = safe_add(ctx0, patches, patch_embd_b);
    }

    ggml_tensor *out = build_graph(ctx0, patches);
    ggml_set_name(out, "vision_output");
    ggml_set_output(out);
    return {out};
}

ggml_tensor *Pi05VisionModel::build_graph(ggml_context *ctx0, ggml_tensor *inp) {
    int n_layer = hparams.num_hidden_layers;
    int n_head = hparams.num_attention_heads;
    int d_head = hparams.head_dim();
    float eps = hparams.layer_norm_eps;
    float kq_scale = 1.0f / sqrtf((float)d_head);
    int n_patches = hparams.num_patches();

    // Detect Q4_K from weight shape: Q4_K pads input dim 1152 -> 1280
    bool use_q4k = (layers[0].q_w->ne[0] == 1280);

    if (position_embd) {
        inp = safe_add(ctx0, inp, position_embd);
    }

    ggml_tensor *cur = inp;
    for (int il = 0; il < n_layer; il++) {
        auto &layer = layers[il];
        ggml_tensor *residual = cur;

        cur = build_norm(ctx0, cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps);

        // Q4_K: pad cur 1152 -> 1280 before QKV projection
        ggml_tensor *cur_qkv = cur;
        if (use_q4k) {
            cur_qkv = ggml_pad(ctx0, cur, 128, 0, 0, 0);
        }

        ggml_tensor *Qcur = ggml_mul_mat(ctx0, layer.q_w, cur_qkv);
        if (layer.q_b) Qcur = safe_add(ctx0, Qcur, layer.q_b);
        ggml_tensor *Kcur = ggml_mul_mat(ctx0, layer.k_w, cur_qkv);
        if (layer.k_b) Kcur = safe_add(ctx0, Kcur, layer.k_b);
        ggml_tensor *Vcur = ggml_mul_mat(ctx0, layer.v_w, cur_qkv);
        if (layer.v_b) Vcur = safe_add(ctx0, Vcur, layer.v_b);

        Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_patches);
        Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_patches);
        Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_patches);

        cur = build_attn(ctx0, layer.o_w, layer.o_b, Qcur, Kcur, Vcur, nullptr, kq_scale);
        cur = ggml_add(ctx0, cur, residual);  // residual is always f32

        residual = cur;
        cur = build_norm(ctx0, cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps);

        // Q4_K: pad cur 1152 -> 1280 before FFN
        ggml_tensor *cur_ffn = cur;
        if (use_q4k) {
            cur_ffn = ggml_pad(ctx0, cur, 128, 0, 0, 0);
        }

        // Vision FFN: ff_up outputs 4352, ff_down expects 4352
        cur = ggml_mul_mat(ctx0, layer.ff_up_w, cur_ffn);
        if (layer.ff_up_b) {
            cur = safe_add(ctx0, cur, layer.ff_up_b);
        }
        cur = ggml_gelu(ctx0, cur);
        cur = ggml_mul_mat(ctx0, layer.ff_down_w, cur);
        if (layer.ff_down_b) {
            cur = safe_add(ctx0, cur, layer.ff_down_b);
        }

        cur = ggml_add(ctx0, cur, residual);  // residual is always f32
    }

    if (post_ln_w) {
        cur = build_norm(ctx0, cur, post_ln_w, post_ln_b, NORM_TYPE_NORMAL, eps);
    }

    return cur;
}

// Pi05ProjectorModel
std::vector<ggml_tensor *> Pi05ProjectorModel::get_tensors_to_load(ggml_context *ctx) {
    std::vector<ggml_tensor *> tensors;
    weight = get_tensor(ctx, "mm.0.weight", tensors);
    bias = get_tensor(ctx, "mm.0.bias", tensors, false);
    return tensors;
}

bool Pi05ProjectorModel::load_hparams(const ModelLoader &model_loader) {
    model_loader.get_u32("projector.input_dim", input_dim, false);
    model_loader.get_u32("projector.output_dim", output_dim, false);
    return true;
}

std::vector<ggml_tensor *> Pi05ProjectorModel::build_graph(ggml_context *ctx0) {
    // Create input tensor
    ggml_tensor *inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, input_dim, 256);  // 256 patches
    ggml_set_name(inp, "vision_features");
    ggml_set_input(inp);

    ggml_tensor *out = build_graph(ctx0, inp);
    ggml_set_name(out, "projected_features");
    ggml_set_output(out);
    return {out};
}

ggml_tensor *Pi05ProjectorModel::build_graph(ggml_context *ctx0, ggml_tensor *inp) {
    return build_linear(ctx0, inp, weight, bias);
}

// Pi05TextModel
std::vector<ggml_tensor *> Pi05TextModel::get_tensors_to_load(ggml_context *ctx) {
    std::vector<ggml_tensor *> tensors;

    token_embd = get_tensor(ctx, "token_embd.weight", tensors);
    output_norm_w = get_tensor(ctx, "output_norm.weight", tensors);

    int n_layer = hparams.num_hidden_layers;
    layers.resize(n_layer);

    for (int il = 0; il < n_layer; ++il) {
        auto &layer = layers[il];
        layer.attn_norm_w = get_tensor(ctx, string_format("blk.%d.attn_norm.weight", il), tensors);
        layer.q_w = get_tensor(ctx, string_format("blk.%d.attn_q.weight", il), tensors);
        layer.k_w = get_tensor(ctx, string_format("blk.%d.attn_k.weight", il), tensors);
        layer.v_w = get_tensor(ctx, string_format("blk.%d.attn_v.weight", il), tensors);
        layer.o_w = get_tensor(ctx, string_format("blk.%d.attn_output.weight", il), tensors);
        layer.ffn_norm_w = get_tensor(ctx, string_format("blk.%d.ffn_norm.weight", il), tensors);
        layer.ffn_gate_w = get_tensor(ctx, string_format("blk.%d.ffn_gate.weight", il), tensors);
        layer.ffn_up_w = get_tensor(ctx, string_format("blk.%d.ffn_up.weight", il), tensors);
        layer.ffn_down_w = get_tensor(ctx, string_format("blk.%d.ffn_down.weight", il), tensors);
    }

    return tensors;
}

bool Pi05TextModel::load_hparams(const ModelLoader &model_loader) {
    model_loader.get_u32("gemma.embedding_length", hparams.hidden_size, false);
    model_loader.get_u32("gemma.block_count", hparams.num_hidden_layers, false);
    model_loader.get_u32("gemma.feed_forward_length", hparams.intermediate_size, false);
    model_loader.get_u32("gemma.attention.head_count", hparams.num_attention_heads, false);
    model_loader.get_u32("gemma.attention.head_count_kv", hparams.num_key_value_heads, false);
    model_loader.get_u32("gemma.attention.key_length", hparams.head_dim, false);
    model_loader.get_u32("gemma.context_length", hparams.max_position_embeddings, false);
    model_loader.get_f32("gemma.attention.layer_norm_rms_epsilon", hparams.rms_norm_eps, false);
    model_loader.get_f32("gemma.rope.freq_base", hparams.rope_theta, false);
    return true;
}

ggml_tensor *Pi05TextModel::build_graph(ggml_context *ctx0, ggml_tensor *inp,
                                        ggml_tensor *kv_cache_k, ggml_tensor *kv_cache_v,
                                        int n_past, int n_tokens) {
    // Simplified forward pass - full implementation in pi05.cpp
    return inp;
}

// Pi05ActionExpert
std::vector<ggml_tensor *> Pi05ActionExpert::get_tensors_to_load(ggml_context *ctx) {
    std::vector<ggml_tensor *> tensors;

    // NOTE: Pi0.5 does NOT have vlm_proj - prefix embeddings go directly to joint attention
    timestep_mlp_in_w = get_tensor(ctx, "action.timestep_mlp.in.weight", tensors, false);
    timestep_mlp_in_b = get_tensor(ctx, "action.timestep_mlp.in.bias", tensors, false);
    timestep_mlp_out_w = get_tensor(ctx, "action.timestep_mlp.out.weight", tensors, false);
    timestep_mlp_out_b = get_tensor(ctx, "action.timestep_mlp.out.bias", tensors, false);
    action_in_w = get_tensor(ctx, "action.action_in.weight", tensors, false);
    action_in_b = get_tensor(ctx, "action.action_in.bias", tensors, false);
    action_out_w = get_tensor(ctx, "action.action_out.weight", tensors, false);
    action_out_b = get_tensor(ctx, "action.action_out.bias", tensors, false);
    // NOTE: Pi05.5 does NOT have state_proj - state is embedded via action_in_proj
    // Output AdaLN
    output_adaln_w = get_tensor(ctx, "action.output_norm.weight", tensors, false);
    output_adaln_b = get_tensor(ctx, "action.output_norm.bias", tensors, false);

    int n_layer = hparams.num_hidden_layers;
    layers.resize(n_layer);
    pali_layers.resize(n_layer);

    for (int il = 0; il < n_layer; ++il) {
        auto &layer = layers[il];
        // AdaLN dense projection weights: [3*hidden_size, hidden_size]
        layer.attn_adaln_w = get_tensor(ctx, string_format("action.blk.%d.attn_norm.weight", il), tensors, false);
        layer.attn_adaln_b = get_tensor(ctx, string_format("action.blk.%d.attn_norm.bias", il), tensors, false);
        layer.q_w = get_tensor(ctx, string_format("action.blk.%d.attn_q.weight", il), tensors, false);
        layer.k_w = get_tensor(ctx, string_format("action.blk.%d.attn_k.weight", il), tensors, false);
        layer.v_w = get_tensor(ctx, string_format("action.blk.%d.attn_v.weight", il), tensors, false);
        layer.o_w = get_tensor(ctx, string_format("action.blk.%d.attn_output.weight", il), tensors, false);
        layer.ffn_adaln_w = get_tensor(ctx, string_format("action.blk.%d.ffn_norm.weight", il), tensors, false);
        layer.ffn_adaln_b = get_tensor(ctx, string_format("action.blk.%d.ffn_norm.bias", il), tensors, false);
        layer.ffn_gate_w = get_tensor(ctx, string_format("action.blk.%d.ffn_gate.weight", il), tensors, false);
        layer.ffn_up_w = get_tensor(ctx, string_format("action.blk.%d.ffn_up.weight", il), tensors, false);
        layer.ffn_down_w = get_tensor(ctx, string_format("action.blk.%d.ffn_down.weight", il), tensors, false);

        // PaliGemma layer weights for cross-attention (with prefix self-attention)
        auto &pali_layer = pali_layers[il];
        pali_layer.attn_norm_w = get_tensor(ctx, string_format("pali.blk.%d.attn_norm.weight", il), tensors, false);
        pali_layer.q_w = get_tensor(ctx, string_format("pali.blk.%d.attn_q.weight", il), tensors, false);
        pali_layer.k_w = get_tensor(ctx, string_format("pali.blk.%d.attn_k.weight", il), tensors, false);
        pali_layer.v_w = get_tensor(ctx, string_format("pali.blk.%d.attn_v.weight", il), tensors, false);
        pali_layer.o_w = get_tensor(ctx, string_format("pali.blk.%d.attn_o.weight", il), tensors, false);
        pali_layer.ffn_norm_w = get_tensor(ctx, string_format("pali.blk.%d.ffn_norm.weight", il), tensors, false);
        pali_layer.ffn_gate_w = get_tensor(ctx, string_format("pali.blk.%d.ffn_gate.weight", il), tensors, false);
        pali_layer.ffn_up_w = get_tensor(ctx, string_format("pali.blk.%d.ffn_up.weight", il), tensors, false);
        pali_layer.ffn_down_w = get_tensor(ctx, string_format("pali.blk.%d.ffn_down.weight", il), tensors, false);
    }

    return tensors;
}

bool Pi05ActionExpert::load_hparams(const ModelLoader &model_loader) {
    model_loader.get_u32("pi05_action.hidden_size", hparams.hidden_size, false);
    model_loader.get_u32("pi05_action.intermediate_size", hparams.intermediate_size, false);
    model_loader.get_u32("pi05_action.block_count", hparams.num_hidden_layers, false);
    model_loader.get_u32("pi05_action.attention.head_count", hparams.num_attention_heads, false);
    model_loader.get_u32("pi05_action.attention.head_count_kv", hparams.num_key_value_heads, false);
    model_loader.get_u32("pi05_action.attention.head_dim", hparams.head_dim, false);
    model_loader.get_u32("pi05_action.action_dim", hparams.action_dim, false);
    model_loader.get_u32("pi05_action.state_dim", hparams.state_dim, false);
    model_loader.get_u32("pi05_action.action_horizon", hparams.action_horizon, false);
    model_loader.get_u32("pi05_action.timestep_sinusoidal_dim", hparams.timestep_sinusoidal_dim, false);
    model_loader.get_f32("pi05_action.attention.layer_norm_rms_epsilon", hparams.rms_norm_eps, false);
    // PaliGemma config for cross-attention
    model_loader.get_u32("pi05_action.pali_hidden_size", hparams.pali_hidden_size, false);
    model_loader.get_u32("pi05_action.pali_intermediate_size", hparams.pali_intermediate_size, false);
    return true;
}

std::vector<ggml_tensor *> Pi05ActionExpert::build_graph(ggml_context *ctx0) {
    int action_dim = hparams.action_dim;
    int action_horizon = hparams.action_horizon;
    int pali_hidden = hparams.pali_hidden_size;
    int timestep_dim = hparams.timestep_sinusoidal_dim;
    int n_prefix = hparams.n_prefix_tokens;  // Configurable prefix length (vision + text)

    // Create input tensors - prefix_hidden is now 2D [pali_hidden, n_prefix]
    // for cross-attention to VLM prefix tokens
    ggml_tensor *t_prefix = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, pali_hidden, n_prefix);
    ggml_set_name(t_prefix, "prefix_hidden");
    ggml_set_input(t_prefix);

    ggml_tensor *t_actions = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, action_dim, action_horizon);
    ggml_set_name(t_actions, "noisy_actions");
    ggml_set_input(t_actions);

    ggml_tensor *t_timestep = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, timestep_dim);
    ggml_set_name(t_timestep, "timestep_emb");
    ggml_set_input(t_timestep);

    ggml_tensor *out = build_graph(ctx0, t_prefix, t_actions, t_timestep);
    ggml_set_name(out, "velocity_output");
    ggml_set_output(out);
    return {out};
}

// Helper struct to return both normalized tensor and gate from AdaLN
struct AdaLNResult {
    ggml_tensor *normed;
    ggml_tensor *gate;
};

// Helper function to apply AdaLN (Adaptive Layer Normalization)
// adaln_w: [3*hidden, hidden], adaln_b: [3*hidden]
// timestep_proj: [hidden]
// x: [hidden, n_tokens]
// Returns: normalized tensor and gate for residual gating
static AdaLNResult apply_adaln(ggml_context *ctx0, ggml_tensor *x,
                                ggml_tensor *adaln_w, ggml_tensor *adaln_b,
                                ggml_tensor *timestep_proj, float eps) {
    int hidden_size = x->ne[0];

    // Project timestep to get scale, shift, gate: [3*hidden]
    ggml_tensor *adaln_out = ggml_mul_mat(ctx0, adaln_w, timestep_proj);
    if (adaln_b) {
        adaln_out = safe_add(ctx0, adaln_out, adaln_b);
    }

    // Split into scale, shift, gate (each [hidden])
    ggml_tensor *scale = ggml_view_1d(ctx0, adaln_out, hidden_size, 0);
    ggml_tensor *shift = ggml_view_1d(ctx0, adaln_out, hidden_size, hidden_size * sizeof(float));
    ggml_tensor *gate = ggml_view_1d(ctx0, adaln_out, hidden_size, 2 * hidden_size * sizeof(float));

    // Apply RMSNorm (no learned weights)
    ggml_tensor *x_norm = ggml_rms_norm(ctx0, x, eps);

    // Reshape scale and shift for broadcasting: [hidden, 1] -> broadcasts to [hidden, n_tokens]
    scale = ggml_reshape_2d(ctx0, scale, hidden_size, 1);
    shift = ggml_reshape_2d(ctx0, shift, hidden_size, 1);
    gate = ggml_reshape_2d(ctx0, gate, hidden_size, 1);

    // x_out = x_norm * (1 + scale) + shift
    ggml_tensor *x_scaled = ggml_mul(ctx0, x_norm, ggml_repeat(ctx0, scale, x_norm));
    ggml_tensor *shift_broad = ggml_repeat(ctx0, shift, x_norm);
    ggml_tensor *x_out = ggml_add(ctx0, x_scaled, x_norm);  // x_norm * scale + x_norm = x_norm * (1 + scale)
    x_out = ggml_add(ctx0, x_out, shift_broad);

    return {x_out, gate};
}

// Helper: Apply PaliGemma RMSNorm with learned weight (1 + weight)
static ggml_tensor *apply_pali_rms_norm(ggml_context *ctx0, ggml_tensor *x,
                                         ggml_tensor *weight, float eps) {
    ggml_tensor *x_norm = ggml_rms_norm(ctx0, x, eps);
    // PaliGemma uses (1 + weight) * x_norm
    // First apply x_norm, then multiply by weight, then add x_norm back
    // This is equivalent to x_norm * (1 + weight) = x_norm + x_norm * weight
    ggml_tensor *weight_2d = ggml_reshape_2d(ctx0, weight, weight->ne[0], 1);
    ggml_tensor *scaled = ggml_mul(ctx0, x_norm, ggml_repeat(ctx0, weight_2d, x_norm));
    return ggml_add(ctx0, x_norm, scaled);
}

// Helper: Create GemmaJointLayer from separate layer structures
static GemmaJointLayer make_joint_layer(const PaliGemmaLayer &pali, const Pi0ActionLayer &action) {
    GemmaJointLayer joint;
    // Prefix (PaliGemma) weights
    joint.pali_attn_norm_w = pali.attn_norm_w;
    joint.pali_q_w = pali.q_w;
    joint.pali_k_w = pali.k_w;
    joint.pali_v_w = pali.v_w;
    joint.pali_o_w = pali.o_w;
    joint.pali_ffn_norm_w = pali.ffn_norm_w;
    joint.pali_gate_w = pali.ffn_gate_w;
    joint.pali_up_w = pali.ffn_up_w;
    joint.pali_down_w = pali.ffn_down_w;
    // Suffix (Expert) weights
    joint.expert_attn_adaln_w = action.attn_adaln_w;
    joint.expert_attn_adaln_b = action.attn_adaln_b;
    joint.expert_q_w = action.q_w;
    joint.expert_k_w = action.k_w;
    joint.expert_v_w = action.v_w;
    joint.expert_o_w = action.o_w;
    joint.expert_ffn_adaln_w = action.ffn_adaln_w;
    joint.expert_ffn_adaln_b = action.ffn_adaln_b;
    joint.expert_gate_w = action.ffn_gate_w;
    joint.expert_up_w = action.ffn_up_w;
    joint.expert_down_w = action.ffn_down_w;
    return joint;
}

// ============ KV Cache Operations ============

PrefixForwardResult Pi05ActionExpert::build_graph_prefix_only(ggml_context *ctx0,
                                                               ggml_tensor *prefix_hidden) {
    // Process prefix through all layers and return KV cache for each layer
    // This is called once per observation to cache prefix K/V

    int n_layer = hparams.num_hidden_layers;
    float eps = hparams.rms_norm_eps;
    int n_prefix = prefix_hidden->ne[1];

    PrefixForwardResult result;
    result.k_caches.resize(n_layer);
    result.v_caches.resize(n_layer);

    // Create prefix positions tensor
    ggml_tensor *prefix_positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_prefix);
    ggml_set_name(prefix_positions, "prefix_positions");
    ggml_set_input(prefix_positions);

    // Create prefix self-attention mask (all zeros for bidirectional)
    ggml_tensor *prefix_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_prefix, n_prefix);
    ggml_set_name(prefix_mask, "prefix_attn_mask");
    ggml_set_input(prefix_mask);

    ggml_tensor *pali_cur = prefix_hidden;

    // Process through all layers
    for (int il = 0; il < n_layer; il++) {
        // Create joint layer from PaliGemma weights only
        GemmaJointLayer joint_layer = {};
        joint_layer.pali_attn_norm_w = pali_layers[il].attn_norm_w;
        joint_layer.pali_q_w = pali_layers[il].q_w;
        joint_layer.pali_k_w = pali_layers[il].k_w;
        joint_layer.pali_v_w = pali_layers[il].v_w;
        joint_layer.pali_o_w = pali_layers[il].o_w;
        joint_layer.pali_ffn_norm_w = pali_layers[il].ffn_norm_w;
        joint_layer.pali_gate_w = pali_layers[il].ffn_gate_w;
        joint_layer.pali_up_w = pali_layers[il].ffn_up_w;
        joint_layer.pali_down_w = pali_layers[il].ffn_down_w;

        PrefixLayerResult layer_result = build_prefix_layer_forward(
            ctx0, pali_cur, prefix_positions, prefix_mask,
            joint_layer, eps, il
        );

        pali_cur = layer_result.prefix_out;
        result.k_caches[il] = layer_result.k_cache;
        result.v_caches[il] = layer_result.v_cache;

        // Mark K/V cache tensors as outputs so they can be read after graph execution
        ggml_set_name(layer_result.k_cache, ("k_cache_" + std::to_string(il)).c_str());
        ggml_set_output(layer_result.k_cache);
        ggml_set_name(layer_result.v_cache, ("v_cache_" + std::to_string(il)).c_str());
        ggml_set_output(layer_result.v_cache);
    }

    // Mark final prefix output
    ggml_set_name(pali_cur, "prefix_out");
    ggml_set_output(pali_cur);
    result.outputs.push_back(pali_cur);

    return result;
}

ggml_tensor *Pi05ActionExpert::build_graph_with_cache(ggml_context *ctx0,
                                                       ggml_tensor *suffix_hidden,
                                                       const std::vector<ggml_tensor *> &prefix_k_caches,
                                                       const std::vector<ggml_tensor *> &prefix_v_caches,
                                                       ggml_tensor *timestep_emb) {
    // Process suffix using cached prefix K/V
    // This is called for each ODE step during flow matching

    int n_layer = hparams.num_hidden_layers;
    int action_horizon = hparams.action_horizon;
    float eps = hparams.rms_norm_eps;

    int n_prefix = prefix_k_caches[0]->ne[1];
    int n_suffix = suffix_hidden->ne[1];
    int n_total = n_prefix + n_suffix;

    // Timestep MLP: sinusoidal -> linear -> silu -> linear -> silu (adarms_cond)
    ggml_tensor *timestep_proj = build_linear(ctx0, timestep_emb,
                                              timestep_mlp_in_w, timestep_mlp_in_b);
    timestep_proj = ggml_silu(ctx0, timestep_proj);
    timestep_proj = build_linear(ctx0, timestep_proj,
                                 timestep_mlp_out_w, timestep_mlp_out_b);
    ggml_tensor *adarms_cond = ggml_silu(ctx0, timestep_proj);

    // Create suffix positions (starting at n_prefix)
    ggml_tensor *suffix_positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_suffix);
    ggml_set_name(suffix_positions, "suffix_positions");
    ggml_set_input(suffix_positions);

    // Create attention mask for suffix queries attending to full sequence
    ggml_tensor *attn_mask = build_suffix_attn_mask(ctx0, n_prefix, n_suffix);
    ggml_set_input(attn_mask);

    ggml_tensor *suffix_cur = suffix_hidden;

    // Process through all layers using cached K/V
    for (int il = 0; il < n_layer; il++) {
        // Create joint layer from Expert weights
        GemmaJointLayer joint_layer = {};
        joint_layer.expert_attn_adaln_w = layers[il].attn_adaln_w;
        joint_layer.expert_attn_adaln_b = layers[il].attn_adaln_b;
        joint_layer.expert_q_w = layers[il].q_w;
        joint_layer.expert_k_w = layers[il].k_w;
        joint_layer.expert_v_w = layers[il].v_w;
        joint_layer.expert_o_w = layers[il].o_w;
        joint_layer.expert_ffn_adaln_w = layers[il].ffn_adaln_w;
        joint_layer.expert_ffn_adaln_b = layers[il].ffn_adaln_b;
        joint_layer.expert_gate_w = layers[il].ffn_gate_w;
        joint_layer.expert_up_w = layers[il].ffn_up_w;
        joint_layer.expert_down_w = layers[il].ffn_down_w;

        suffix_cur = build_suffix_layer_with_cache(
            ctx0, suffix_cur,
            prefix_k_caches[il], prefix_v_caches[il],
            suffix_positions, attn_mask, adarms_cond,
            joint_layer, eps, il
        );
    }

    // Output AdaLN and projection
    AdaRMSNormResult adaln_out = build_ada_rms_norm(ctx0, suffix_cur,
                                                     output_adaln_w, output_adaln_b,
                                                     adarms_cond, eps);
    ggml_tensor *out = adaln_out.output;
    out = build_linear(ctx0, out, action_out_w, action_out_b);

    return out;
}

ggml_tensor *Pi05ActionExpert::build_graph(ggml_context *ctx0, ggml_tensor *prefix_hidden,
                                           ggml_tensor *noisy_actions, ggml_tensor *timestep_emb) {
    // prefix_hidden: [pali_hidden_size=2048, n_prefix] - VLM prefix hidden states
    // Uses TRUE joint attention: prefix and suffix K/V are concatenated BEFORE attention
    // Following OpenPI architecture from gemma.py Block class

    int action_horizon = hparams.action_horizon;
    int hidden_size = hparams.hidden_size;
    int n_layer = hparams.num_hidden_layers;
    float eps = hparams.rms_norm_eps;

    int n_prefix = prefix_hidden->ne[1];  // e.g., 256 vision tokens
    int n_suffix = action_horizon;        // e.g., 50 action tokens
    int n_total = n_prefix + n_suffix;

    // Timestep MLP: sinusoidal -> linear -> silu -> linear -> silu (adarms_cond)
    ggml_tensor *timestep_proj = build_linear(ctx0, timestep_emb,
                                              timestep_mlp_in_w, timestep_mlp_in_b);
    timestep_proj = ggml_silu(ctx0, timestep_proj);
    timestep_proj = build_linear(ctx0, timestep_proj,
                                 timestep_mlp_out_w, timestep_mlp_out_b);
    ggml_tensor *adarms_cond = ggml_silu(ctx0, timestep_proj);  // Final swish for adarms_cond
    ggml_set_name(adarms_cond, "adarms_cond");

    // Embed actions: [action_dim, action_horizon] -> [hidden_size, action_horizon]
    ggml_tensor *action_hidden = build_linear(ctx0, noisy_actions, action_in_w, action_in_b);
    ggml_set_name(action_hidden, "action_hidden");

    // Create joint attention mask and positions as INPUT tensors
    // These will be filled by the caller before graph execution
    ggml_tensor *attn_mask = build_joint_attn_mask(ctx0, n_prefix, n_suffix);
    ggml_set_input(attn_mask);

    ggml_tensor *positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_total);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // Initialize hidden states for joint processing
    ggml_tensor *pali_cur = prefix_hidden;
    ggml_tensor *suffix_cur = action_hidden;

    // Process all layers with TRUE joint attention
    for (int il = 0; il < n_layer; il++) {
        // Create joint layer from separate PaliGemma and Action Expert weights
        GemmaJointLayer joint_layer = make_joint_layer(pali_layers[il], layers[il]);

        // Use build_joint_layer_forward for correct joint attention
        JointLayerResult result = build_joint_layer_forward(
            ctx0, pali_cur, suffix_cur,
            positions, attn_mask, adarms_cond,
            joint_layer, eps, il
        );

        pali_cur = result.prefix_out;
        suffix_cur = result.suffix_out;
    }

    // Output AdaLN and projection (only for suffix/action tokens)
    AdaLNResult adaln_out = apply_adaln(ctx0, suffix_cur, output_adaln_w, output_adaln_b, adarms_cond, eps);
    ggml_tensor *out = adaln_out.normed;
    out = build_linear(ctx0, out, action_out_w, action_out_b);

    return out;
}
