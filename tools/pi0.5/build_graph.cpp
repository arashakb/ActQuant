#include "build_graph.h"
#include "model_defs.h"
#include "pi05_memory.h"
#include <string>
#include <cmath>

void cb(ggml_context *ctx0, ggml_tensor *cur0, const char *name, int il) {
    // Debug callback placeholder
}

ggml_tensor *build_norm(ggml_context *ctx0, ggml_tensor *cur, ggml_tensor *mw,
                        ggml_tensor *mb, norm_type type, float norm_eps, int il) {
    cur = type == NORM_TYPE_RMS ? ggml_rms_norm(ctx0, cur, norm_eps)
                                : ggml_norm(ctx0, cur, norm_eps);
    if (mw || mb) {
        cb(ctx0, cur, "norm", il);
    }
    if (mw) {
        cur = safe_mul(ctx0, cur, mw);
        if (mb) {
            cb(ctx0, cur, "norm_w", il);
        }
    }
    if (mb) {
        cur = safe_add(ctx0, cur, mb);
    }
    return cur;
}

ggml_tensor *build_linear(ggml_context *ctx0, ggml_tensor *cur, ggml_tensor *w,
                          ggml_tensor *b, int il) {
    cur = ggml_mul_mat(ctx0, w, cur);
    cb(ctx0, cur, "linear", il);
    if (b) {
        cur = safe_add(ctx0, cur, b);
        cb(ctx0, cur, "linear_b", il);
    }
    return cur;
}

ggml_tensor *build_ffn(ggml_context *ctx0, ggml_tensor *cur, ggml_tensor *up,
                       ggml_tensor *up_b, ggml_tensor *gate, ggml_tensor *gate_b,
                       ggml_tensor *down, ggml_tensor *down_b, ffn_op_type type_op, int il) {
    ggml_tensor *tmp = up ? ggml_mul_mat(ctx0, up, cur) : cur;
    cb(ctx0, tmp, "ffn_up", il);

    if (up_b) {
        tmp = safe_add(ctx0, tmp, up_b);
        cb(ctx0, tmp, "ffn_up_b", il);
    }

    if (gate) {
        cur = ggml_mul_mat(ctx0, gate, cur);
        cb(ctx0, cur, "ffn_gate", il);
        if (gate_b) {
            cur = safe_add(ctx0, cur, gate_b);
            cb(ctx0, cur, "ffn_gate_b", il);
        }
    } else {
        cur = tmp;
    }

    switch (type_op) {
    case FFN_SILU:
        if (gate) {
            cur = ggml_swiglu_split(ctx0, cur, tmp);
        } else {
            cur = ggml_silu(ctx0, cur);
        }
        break;
    case FFN_GELU:
        if (gate) {
            cur = ggml_geglu_split(ctx0, cur, tmp);
        } else {
            cur = ggml_gelu(ctx0, cur);
        }
        break;
    case FFN_GELU_ERF:
        if (gate) {
            cur = ggml_geglu_erf_split(ctx0, cur, tmp);
        } else {
            cur = ggml_gelu_erf(ctx0, cur);
        }
        break;
    case FFN_GELU_QUICK:
        if (gate) {
            cur = ggml_geglu_quick_split(ctx0, cur, tmp);
        } else {
            cur = ggml_gelu_quick(ctx0, cur);
        }
        break;
    }

    if (down) {
        cur = ggml_mul_mat(ctx0, down, cur);
    }
    if (down_b) {
        cur = safe_add(ctx0, cur, down_b);
    }
    return cur;
}

ggml_tensor *build_attn(ggml_context *ctx0, ggml_tensor *wo, ggml_tensor *wo_b,
                        ggml_tensor *q_cur, ggml_tensor *k_cur, ggml_tensor *v_cur,
                        ggml_tensor *kq_mask, float kq_scale, int il) {
    ggml_tensor *q = ggml_permute(ctx0, q_cur, 0, 2, 1, 3);
    ggml_tensor *k = ggml_permute(ctx0, k_cur, 0, 2, 1, 3);
    ggml_tensor *v = ggml_permute(ctx0, v_cur, 1, 2, 0, 3);
    v = ggml_cont(ctx0, v);

    ggml_tensor *cur;
    {
        const auto n_tokens = q->ne[1];
        const auto n_head = q->ne[2];

        ggml_tensor *kq = ggml_mul_mat(ctx0, k, q);
        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, 0.0f);

        ggml_tensor *kqv = ggml_mul_mat(ctx0, v, kq);
        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0] * n_head, n_tokens);
    }

    cb(ctx0, cur, "kqv_out", il);

    if (wo) {
        // Pad cur if needed for quantization compatibility (Q4_K)
        if (wo->ne[0] > cur->ne[0]) {
            int64_t pad = wo->ne[0] - cur->ne[0];
            cur = ggml_pad(ctx0, cur, pad, 0, 0, 0);
        }
        cur = ggml_mul_mat(ctx0, wo, cur);
    }
    if (wo_b) {
        cur = safe_add(ctx0, cur, wo_b);
    }
    return cur;
}

ggml_tensor *build_rope(ggml_context *ctx0, ggml_tensor *x, ggml_tensor *positions,
                        int n_dims, float freq_base) {
    // Apply RoPE (Rotary Position Embedding)
    // Input x: [head_dim, n_heads, n_tokens]
    // positions: [n_tokens] (I32)
    //
    // Uses GGML's ggml_rope_ext with standard parameters:
    // - mode = 0 (standard RoPE)
    // - n_ctx_orig = 0 (not using extended context)
    // - freq_scale = 1.0 (no scaling)
    // - ext_factor, attn_factor, beta_fast, beta_slow = 0 (not using YaRN)

    return ggml_rope_ext(
        ctx0,
        x,              // input tensor [head_dim, n_heads, n_tokens]
        positions,      // position indices [n_tokens]
        nullptr,        // rope_factors (nullptr for standard RoPE)
        n_dims,         // number of dimensions to rotate (head_dim)
        2,              // mode = 2 (GGML_ROPE_TYPE_NEOX: non-interleaved, Gemma style)
        0,              // n_ctx_orig (not used)
        freq_base,      // freq_base = 10000.0
        1.0f,           // freq_scale = 1.0
        0.0f,           // ext_factor = 0 (no extension)
        1.0f,           // attn_factor = 1.0
        0.0f,           // beta_fast = 0
        0.0f            // beta_slow = 0
    );
}

ggml_tensor *build_joint_attn_mask(ggml_context *ctx0, int n_prefix, int n_suffix) {
    // Build attention mask for joint prefix-suffix attention
    // Mask values: 0.0 = can attend, -inf = cannot attend
    //
    // Pattern (for query position i attending to key position j):
    //   mask[j, i] = 0    if query at i can attend to key at j
    //   mask[j, i] = -inf otherwise
    //
    // In GGML, attention mask layout is [n_kv, n_q] where:
    //   - n_kv is key/value sequence length (ne[0])
    //   - n_q is query sequence length (ne[1])
    //
    // For joint attention:
    //   Prefix queries (0:n_prefix) can attend to prefix keys only
    //   Suffix queries (n_prefix:) can attend to all keys

    int n_total = n_prefix + n_suffix;

    // Create mask tensor [n_kv, n_q] = [n_total, n_total]
    ggml_tensor *mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_total, n_total);
    ggml_set_name(mask, "joint_attn_mask");

    return mask;
}

void fill_joint_attn_mask(float *data, int n_prefix, int n_suffix) {
    // Fill attention mask values for joint prefix-suffix attention
    // Layout: [n_kv, n_q] in column-major order (GGML)
    //   data[j + i * n_total] = mask value for query i, key j
    //
    // Pattern:
    //   Prefix queries (i < n_prefix) attending to:
    //     - Prefix keys (j < n_prefix): 0.0 (can attend)
    //     - Suffix keys (j >= n_prefix): -inf (cannot attend)
    //   Suffix queries (i >= n_prefix) attending to:
    //     - All keys: 0.0 (can attend)

    int n_total = n_prefix + n_suffix;
    const float NEG_INF = -INFINITY;

    for (int i = 0; i < n_total; i++) {      // query position
        for (int j = 0; j < n_total; j++) {  // key position
            int idx = j + i * n_total;  // GGML column-major: ne[0] varies fastest

            if (i < n_prefix && j >= n_prefix) {
                // Prefix query cannot attend to suffix key
                data[idx] = NEG_INF;
            } else {
                // Can attend
                data[idx] = 0.0f;
            }
        }
    }
}

AdaRMSNormResult build_ada_rms_norm(ggml_context *ctx0, ggml_tensor *x,
                                     ggml_tensor *w, ggml_tensor *b,
                                     ggml_tensor *cond, float norm_eps) {
    // Adaptive RMSNorm with scale, shift, and gate from conditioning signal
    //
    // x: input [hidden_dim, n_tokens]
    // w: dense weight [3*hidden_dim, cond_dim]
    // b: dense bias [3*hidden_dim]
    // cond: conditioning [cond_dim, 1] (e.g., timestep embedding)
    //
    // Formula:
    // 1. modulation = linear(cond, w, b) -> [3*hidden_dim, 1]
    // 2. split: scale, shift, gate = modulation[:h], modulation[h:2h], modulation[2h:3h]
    // 3. x_norm = x * rsqrt(mean(x^2) + eps)
    // 4. output = x_norm * (1 + scale) + shift
    // 5. return (output, gate)

    int64_t hidden_dim = x->ne[0];

    // Compute modulation: [3*hidden_dim, 1]
    ggml_tensor *modulation = ggml_mul_mat(ctx0, w, cond);
    if (b) {
        modulation = safe_add(ctx0, modulation, b);
    }

    // Split into scale, shift, gate (each [hidden_dim, 1])
    // Using ggml_view_2d for slicing
    ggml_tensor *scale = ggml_view_2d(ctx0, modulation, hidden_dim, 1, modulation->nb[1], 0);
    ggml_tensor *shift = ggml_view_2d(ctx0, modulation, hidden_dim, 1, modulation->nb[1], hidden_dim * sizeof(float));
    ggml_tensor *gate = ggml_view_2d(ctx0, modulation, hidden_dim, 1, modulation->nb[1], 2 * hidden_dim * sizeof(float));

    // RMSNorm: x_norm = x * rsqrt(mean(x^2) + eps)
    ggml_tensor *x_norm = ggml_rms_norm(ctx0, x, norm_eps);

    // Apply scale and shift: output = x_norm * (1 + scale) + shift
    // x_norm + x_norm * scale = x_norm * (1 + scale)
    ggml_tensor *scaled = ggml_mul(ctx0, x_norm, scale);
    ggml_tensor *output = ggml_add(ctx0, x_norm, scaled);
    output = ggml_add(ctx0, output, shift);

    return {output, gate};
}

ggml_tensor *build_gqa_attention(ggml_context *ctx0,
                                  ggml_tensor *q, ggml_tensor *k, ggml_tensor *v,
                                  ggml_tensor *mask,
                                  int n_q_heads, int n_kv_heads, int head_dim) {
    // GQA attention for joint prefix-suffix attention
    //
    // q: [head_dim * n_q_heads, n_q_tokens]
    // k: [head_dim * n_kv_heads, n_kv_tokens]
    // v: [head_dim * n_kv_heads, n_kv_tokens]
    // mask: [n_kv_tokens, n_q_tokens] (0 = attend, -inf = block)
    //
    // Returns: [head_dim * n_q_heads, n_q_tokens]

    int64_t n_q_tokens = q->ne[1];
    int64_t n_kv_tokens = k->ne[1];
    float scale = 1.0f / sqrtf((float)head_dim);

    // Reshape Q: [head_dim * n_q_heads, n_tokens] -> [head_dim, n_q_heads, n_tokens]
    ggml_tensor *q_reshaped = ggml_reshape_3d(ctx0, q, head_dim, n_q_heads, n_q_tokens);

    // Reshape K, V: [head_dim * n_kv_heads, n_tokens] -> [head_dim, n_kv_heads, n_tokens]
    ggml_tensor *k_reshaped = ggml_reshape_3d(ctx0, k, head_dim, n_kv_heads, n_kv_tokens);
    ggml_tensor *v_reshaped = ggml_reshape_3d(ctx0, v, head_dim, n_kv_heads, n_kv_tokens);

    // Expand KV heads to match Q heads if needed (GQA: n_kv_heads < n_q_heads)
    int groups = n_q_heads / n_kv_heads;
    if (groups > 1) {
        // Repeat K and V heads to match Q heads
        // [head_dim, n_kv_heads, n_tokens] -> [head_dim, n_q_heads, n_tokens]
        k_reshaped = ggml_repeat_4d(ctx0, k_reshaped, head_dim, n_q_heads, n_kv_tokens, 1);
        v_reshaped = ggml_repeat_4d(ctx0, v_reshaped, head_dim, n_q_heads, n_kv_tokens, 1);
    }

    // Transpose for attention: [head_dim, n_heads, n_tokens] -> [n_tokens, n_heads, head_dim]
    // Then compute Q @ K^T
    // GGML's ggml_mul_mat computes A @ B^T, so we need proper shapes

    // Permute Q: [head_dim, n_heads, n_tokens] -> [head_dim, n_tokens, n_heads]
    ggml_tensor *q_perm = ggml_permute(ctx0, q_reshaped, 0, 2, 1, 3);
    // Permute K: [head_dim, n_heads, n_tokens] -> [head_dim, n_tokens, n_heads]
    ggml_tensor *k_perm = ggml_permute(ctx0, k_reshaped, 0, 2, 1, 3);
    // Permute V: [head_dim, n_heads, n_tokens] -> [head_dim, n_tokens, n_heads]
    //            Then transpose for matmul: [n_tokens, head_dim, n_heads]
    ggml_tensor *v_perm = ggml_permute(ctx0, v_reshaped, 1, 2, 0, 3);
    v_perm = ggml_cont(ctx0, v_perm);

    // Attention scores: Q @ K^T -> [n_kv_tokens, n_q_tokens, n_heads]
    ggml_tensor *scores = ggml_mul_mat(ctx0, k_perm, q_perm);

    // Apply softmax with mask and scale
    scores = ggml_soft_max_ext(ctx0, scores, mask, scale, 0.0f);

    // Output: scores @ V -> [head_dim, n_q_tokens, n_heads]
    ggml_tensor *output = ggml_mul_mat(ctx0, v_perm, scores);

    // Permute back: [head_dim, n_q_tokens, n_heads] -> [head_dim, n_heads, n_q_tokens]
    output = ggml_permute(ctx0, output, 0, 2, 1, 3);

    // Reshape to [head_dim * n_heads, n_q_tokens]
    output = ggml_cont_2d(ctx0, output, head_dim * n_q_heads, n_q_tokens);

    return output;
}

JointLayerResult build_joint_layer_forward(
    ggml_context *ctx0,
    ggml_tensor *prefix_hidden,
    ggml_tensor *suffix_hidden,
    ggml_tensor *positions,
    ggml_tensor *attn_mask,
    ggml_tensor *adarms_cond,
    const GemmaJointLayer &layer,
    float norm_eps,
    int il) {
    // Architecture constants
    const int NUM_HEADS = 8;
    const int NUM_KV_HEADS = 1;
    const int HEAD_DIM = 256;

    int64_t n_prefix = prefix_hidden->ne[1];
    int64_t n_suffix = suffix_hidden->ne[1];
    int64_t n_total = n_prefix + n_suffix;

    // =====================================================================
    // Attention Block
    // =====================================================================

    // 1. Normalize inputs
    // Prefix: standard RMSNorm with (1 + weight) scaling (Gemma/PaliGemma style)
    ggml_tensor *prefix_normed = ggml_rms_norm(ctx0, prefix_hidden, norm_eps);
    // x_norm * (1 + w) = x_norm + x_norm * w
    ggml_tensor *prefix_scaled = ggml_mul(ctx0, prefix_normed, layer.pali_attn_norm_w);
    prefix_normed = ggml_add(ctx0, prefix_normed, prefix_scaled);

    // Suffix: AdaRMSNorm with gate
    AdaRMSNormResult suffix_norm_result = build_ada_rms_norm(
        ctx0, suffix_hidden,
        layer.expert_attn_adaln_w, layer.expert_attn_adaln_b,
        adarms_cond, norm_eps
    );
    ggml_tensor *suffix_normed = suffix_norm_result.output;
    ggml_tensor *attn_gate = suffix_norm_result.gate;

    // 2. Compute Q/K/V for prefix (PaliGemma weights)
    // Q: [2048, n_prefix] -> [2048, n_prefix]
    ggml_tensor *prefix_q = ggml_mul_mat(ctx0, layer.pali_q_w, prefix_normed);
    // K, V: [2048, n_prefix] -> [256, n_prefix]
    ggml_tensor *prefix_k = ggml_mul_mat(ctx0, layer.pali_k_w, prefix_normed);
    ggml_tensor *prefix_v = ggml_mul_mat(ctx0, layer.pali_v_w, prefix_normed);

    // 3. Compute Q/K/V for suffix (Expert weights)
    // Q: [1024, n_suffix] -> [2048, n_suffix]
    ggml_tensor *suffix_q = ggml_mul_mat(ctx0, layer.expert_q_w, suffix_normed);
    // K, V: [1024, n_suffix] -> [256, n_suffix]
    ggml_tensor *suffix_k = ggml_mul_mat(ctx0, layer.expert_k_w, suffix_normed);
    ggml_tensor *suffix_v = ggml_mul_mat(ctx0, layer.expert_v_w, suffix_normed);

    // 4. Concatenate Q, K, V for full sequence (OpenPI single-attention architecture)
    // full_q: [2048, n_total], full_k: [256, n_total], full_v: [256, n_total]
    ggml_tensor *full_q = ggml_concat(ctx0, prefix_q, suffix_q, 1);
    ggml_tensor *full_k = ggml_concat(ctx0, prefix_k, suffix_k, 1);
    ggml_tensor *full_v = ggml_concat(ctx0, prefix_v, suffix_v, 1);

    // 5. Apply RoPE to Q and K (FULL sequence with continuous positions)
    // Reshape for RoPE: [dim, n_tokens] -> [HEAD_DIM, n_heads, n_tokens]
    // full_q: [2048, n_total] -> [256, 8, n_total]
    full_q = ggml_reshape_3d(ctx0, full_q, HEAD_DIM, NUM_HEADS, n_total);
    // full_k: [256, n_total] -> [256, 1, n_total]
    full_k = ggml_reshape_3d(ctx0, full_k, HEAD_DIM, NUM_KV_HEADS, n_total);

    // Apply RoPE with continuous positions [0, 1, ..., n_total-1]
    full_q = build_rope(ctx0, full_q, positions, HEAD_DIM, 10000.0f);
    full_k = build_rope(ctx0, full_k, positions, HEAD_DIM, 10000.0f);

    // Reshape back to 2D: [HEAD_DIM * n_heads, n_tokens]
    full_q = ggml_cont_2d(ctx0, full_q, HEAD_DIM * NUM_HEADS, n_total);
    full_k = ggml_cont_2d(ctx0, full_k, HEAD_DIM * NUM_KV_HEADS, n_total);
    full_v = ggml_cont_2d(ctx0, full_v, HEAD_DIM * NUM_KV_HEADS, n_total);

    // 6. SINGLE attention on full sequence (OpenPI architecture)
    // attn_mask already has full [n_total, n_total] shape
    ggml_tensor *full_attn_out = build_gqa_attention(
        ctx0, full_q, full_k, full_v, attn_mask,
        NUM_HEADS, NUM_KV_HEADS, HEAD_DIM
    );

    // 7. Split attention output for prefix and suffix
    // full_attn_out: [2048, n_total]
    ggml_tensor *prefix_attn_out = ggml_view_2d(ctx0, full_attn_out,
                                                 HEAD_DIM * NUM_HEADS, n_prefix,
                                                 full_attn_out->nb[1], 0);
    ggml_tensor *suffix_attn_out = ggml_view_2d(ctx0, full_attn_out,
                                                 HEAD_DIM * NUM_HEADS, n_suffix,
                                                 full_attn_out->nb[1],
                                                 n_prefix * full_attn_out->nb[1]);

    // 8. Output projection (applied AFTER split)
    // Prefix: [2048, n_prefix] -> [2048, n_prefix]
    prefix_attn_out = ggml_mul_mat(ctx0, layer.pali_o_w, prefix_attn_out);
    // Suffix: [2048, n_suffix] -> [1024, n_suffix]
    suffix_attn_out = ggml_mul_mat(ctx0, layer.expert_o_w, suffix_attn_out);

    // 9. Residual connections
    // Prefix: simple residual
    prefix_hidden = ggml_add(ctx0, prefix_hidden, prefix_attn_out);
    // Suffix: gated residual
    suffix_attn_out = ggml_mul(ctx0, suffix_attn_out, attn_gate);
    suffix_hidden = ggml_add(ctx0, suffix_hidden, suffix_attn_out);

    // =====================================================================
    // MLP Block
    // =====================================================================

    // Prefix MLP: RMSNorm with (1 + weight) -> GELU-gated MLP
    ggml_tensor *prefix_mlp_in = ggml_rms_norm(ctx0, prefix_hidden, norm_eps);
    // x_norm * (1 + w) = x_norm + x_norm * w
    ggml_tensor *prefix_ffn_scaled = ggml_mul(ctx0, prefix_mlp_in, layer.pali_ffn_norm_w);
    prefix_mlp_in = ggml_add(ctx0, prefix_mlp_in, prefix_ffn_scaled);

    ggml_tensor *prefix_gate_out = ggml_mul_mat(ctx0, layer.pali_gate_w, prefix_mlp_in);
    prefix_gate_out = ggml_gelu(ctx0, prefix_gate_out);  // GELU (tanh approx in GGML)
    ggml_tensor *prefix_up_out = ggml_mul_mat(ctx0, layer.pali_up_w, prefix_mlp_in);
    ggml_tensor *prefix_mlp_out = ggml_mul(ctx0, prefix_gate_out, prefix_up_out);
    prefix_mlp_out = ggml_mul_mat(ctx0, layer.pali_down_w, prefix_mlp_out);
    prefix_hidden = ggml_add(ctx0, prefix_hidden, prefix_mlp_out);

    // Suffix MLP: AdaRMSNorm -> SiLU-gated MLP -> gated residual
    AdaRMSNormResult suffix_ffn_norm_result = build_ada_rms_norm(
        ctx0, suffix_hidden,
        layer.expert_ffn_adaln_w, layer.expert_ffn_adaln_b,
        adarms_cond, norm_eps
    );
    ggml_tensor *suffix_mlp_in = suffix_ffn_norm_result.output;
    ggml_tensor *mlp_gate = suffix_ffn_norm_result.gate;

    ggml_tensor *suffix_gate_out = ggml_mul_mat(ctx0, layer.expert_gate_w, suffix_mlp_in);
    suffix_gate_out = ggml_gelu(ctx0, suffix_gate_out);  // SiLU for expert
    ggml_tensor *suffix_up_out = ggml_mul_mat(ctx0, layer.expert_up_w, suffix_mlp_in);
    ggml_tensor *suffix_mlp_out = ggml_mul(ctx0, suffix_gate_out, suffix_up_out);
    suffix_mlp_out = ggml_mul_mat(ctx0, layer.expert_down_w, suffix_mlp_out);
    suffix_mlp_out = ggml_mul(ctx0, suffix_mlp_out, mlp_gate);  // Gated residual
    suffix_hidden = ggml_add(ctx0, suffix_hidden, suffix_mlp_out);

    return {prefix_hidden, suffix_hidden};
}

// ============ KV Cache Operations ============

ggml_tensor *build_suffix_attn_mask(ggml_context *ctx0, int n_prefix, int n_suffix) {
    // Build attention mask for suffix-only queries attending to full sequence
    // Mask values: 0.0 = can attend, -inf = cannot attend
    //
    // Layout: [n_kv, n_q] = [n_prefix + n_suffix, n_suffix]
    // All zeros because suffix queries can attend to all keys (prefix + suffix)

    int n_kv = n_prefix + n_suffix;
    ggml_tensor *mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_kv, n_suffix);
    ggml_set_name(mask, "suffix_attn_mask");
    return mask;
}

void fill_suffix_attn_mask(float *data, int n_prefix, int n_suffix,
                           int masked_start, int masked_end) {
    // Fill attention mask for suffix queries attending to [prefix + suffix] keys
    // Layout: [n_kv, n_suffix] in column-major (GGML)
    //   data[j + i * n_kv] = mask value for query i (suffix), key j (prefix+suffix)
    //
    // Suffix queries can attend to all keys EXCEPT masked prefix tokens
    // masked_start/masked_end define the range of prefix tokens to mask (e.g., right_wrist)

    int n_kv = n_prefix + n_suffix;
    const float NEG_INF = -INFINITY;

    for (int i = 0; i < n_suffix; i++) {      // query position (suffix)
        for (int j = 0; j < n_kv; j++) {      // key position (prefix + suffix)
            int idx = j + i * n_kv;  // GGML column-major

            // If key j is in the masked range of prefix tokens, cannot attend
            if (masked_start >= 0 && j >= masked_start && j < masked_end) {
                data[idx] = NEG_INF;
            } else {
                data[idx] = 0.0f;
            }
        }
    }
}

PrefixLayerResult build_prefix_layer_forward(
    ggml_context *ctx0,
    ggml_tensor *prefix_hidden,
    ggml_tensor *positions,
    ggml_tensor *attn_mask,
    const GemmaJointLayer &layer,
    float norm_eps,
    int il) {
    // Process prefix through one transformer layer and return K/V for caching
    // This is similar to the prefix part of build_joint_layer_forward,
    // but without suffix and returns K/V for caching

    const int NUM_HEADS = 8;
    const int NUM_KV_HEADS = 1;
    const int HEAD_DIM = 256;

    int64_t n_prefix = prefix_hidden->ne[1];

    // =====================================================================
    // Attention Block
    // =====================================================================

    // 1. Normalize prefix with PaliGemma RMSNorm: x_norm * (1 + weight)
    ggml_tensor *prefix_normed = ggml_rms_norm(ctx0, prefix_hidden, norm_eps);
    ggml_tensor *prefix_scaled = ggml_mul(ctx0, prefix_normed, layer.pali_attn_norm_w);
    prefix_normed = ggml_add(ctx0, prefix_normed, prefix_scaled);

    // 2. Compute Q/K/V for prefix
    ggml_tensor *prefix_q = ggml_mul_mat(ctx0, layer.pali_q_w, prefix_normed);
    ggml_tensor *prefix_k = ggml_mul_mat(ctx0, layer.pali_k_w, prefix_normed);
    ggml_tensor *prefix_v = ggml_mul_mat(ctx0, layer.pali_v_w, prefix_normed);

    // 3. Apply RoPE to Q and K
    // Reshape: [dim, n_prefix] -> [HEAD_DIM, n_heads, n_prefix]
    prefix_q = ggml_reshape_3d(ctx0, prefix_q, HEAD_DIM, NUM_HEADS, n_prefix);
    prefix_k = ggml_reshape_3d(ctx0, prefix_k, HEAD_DIM, NUM_KV_HEADS, n_prefix);

    prefix_q = build_rope(ctx0, prefix_q, positions, HEAD_DIM, 10000.0f);
    prefix_k = build_rope(ctx0, prefix_k, positions, HEAD_DIM, 10000.0f);

    // Reshape back: [HEAD_DIM, n_heads, n_prefix] -> [HEAD_DIM * n_heads, n_prefix]
    prefix_q = ggml_cont_2d(ctx0, prefix_q, HEAD_DIM * NUM_HEADS, n_prefix);
    ggml_tensor *k_cache = ggml_cont_2d(ctx0, prefix_k, HEAD_DIM * NUM_KV_HEADS, n_prefix);
    ggml_tensor *v_cache = ggml_cont_2d(ctx0, prefix_v, HEAD_DIM * NUM_KV_HEADS, n_prefix);

    // 4. Self-attention on prefix only
    ggml_tensor *prefix_attn_out = build_gqa_attention(
        ctx0, prefix_q, k_cache, v_cache, attn_mask,
        NUM_HEADS, NUM_KV_HEADS, HEAD_DIM
    );

    // 5. Output projection
    prefix_attn_out = ggml_mul_mat(ctx0, layer.pali_o_w, prefix_attn_out);

    // 6. Residual connection
    prefix_hidden = ggml_add(ctx0, prefix_hidden, prefix_attn_out);

    // =====================================================================
    // MLP Block
    // =====================================================================

    // PaliGemma MLP: RMSNorm -> GELU-gated MLP
    ggml_tensor *prefix_mlp_in = ggml_rms_norm(ctx0, prefix_hidden, norm_eps);
    ggml_tensor *prefix_ffn_scaled = ggml_mul(ctx0, prefix_mlp_in, layer.pali_ffn_norm_w);
    prefix_mlp_in = ggml_add(ctx0, prefix_mlp_in, prefix_ffn_scaled);

    ggml_tensor *prefix_gate_out = ggml_mul_mat(ctx0, layer.pali_gate_w, prefix_mlp_in);
    prefix_gate_out = ggml_gelu(ctx0, prefix_gate_out);
    ggml_tensor *prefix_up_out = ggml_mul_mat(ctx0, layer.pali_up_w, prefix_mlp_in);
    ggml_tensor *prefix_mlp_out = ggml_mul(ctx0, prefix_gate_out, prefix_up_out);
    prefix_mlp_out = ggml_mul_mat(ctx0, layer.pali_down_w, prefix_mlp_out);
    prefix_hidden = ggml_add(ctx0, prefix_hidden, prefix_mlp_out);

    return {prefix_hidden, k_cache, v_cache};
}

ggml_tensor* build_suffix_layer_with_cache(
    ggml_context *ctx0,
    ggml_tensor *suffix_hidden,
    ggml_tensor *prefix_k_cache,
    ggml_tensor *prefix_v_cache,
    ggml_tensor *suffix_positions,
    ggml_tensor *attn_mask,
    ggml_tensor *adarms_cond,
    const GemmaJointLayer &layer,
    float norm_eps,
    int il) {
    // Process suffix through one layer using cached prefix K/V
    // Suffix queries attend to (cached prefix K/V) + (computed suffix K/V)

    const int NUM_HEADS = 8;
    const int NUM_KV_HEADS = 1;
    const int HEAD_DIM = 256;

    int64_t n_prefix = prefix_k_cache->ne[1];
    int64_t n_suffix = suffix_hidden->ne[1];
    int64_t n_total = n_prefix + n_suffix;

    // =====================================================================
    // Attention Block
    // =====================================================================

    // 1. Normalize suffix with AdaRMSNorm
    AdaRMSNormResult suffix_norm_result = build_ada_rms_norm(
        ctx0, suffix_hidden,
        layer.expert_attn_adaln_w, layer.expert_attn_adaln_b,
        adarms_cond, norm_eps
    );
    ggml_tensor *suffix_normed = suffix_norm_result.output;
    ggml_tensor *attn_gate = suffix_norm_result.gate;

    // 2. Compute Q/K/V for suffix
    ggml_tensor *suffix_q = ggml_mul_mat(ctx0, layer.expert_q_w, suffix_normed);
    ggml_tensor *suffix_k = ggml_mul_mat(ctx0, layer.expert_k_w, suffix_normed);
    ggml_tensor *suffix_v = ggml_mul_mat(ctx0, layer.expert_v_w, suffix_normed);

    // 3. Apply RoPE to suffix Q and K (positions start at n_prefix)
    suffix_q = ggml_reshape_3d(ctx0, suffix_q, HEAD_DIM, NUM_HEADS, n_suffix);
    suffix_k = ggml_reshape_3d(ctx0, suffix_k, HEAD_DIM, NUM_KV_HEADS, n_suffix);

    suffix_q = build_rope(ctx0, suffix_q, suffix_positions, HEAD_DIM, 10000.0f);
    suffix_k = build_rope(ctx0, suffix_k, suffix_positions, HEAD_DIM, 10000.0f);

    // Reshape back
    suffix_q = ggml_cont_2d(ctx0, suffix_q, HEAD_DIM * NUM_HEADS, n_suffix);
    suffix_k = ggml_cont_2d(ctx0, suffix_k, HEAD_DIM * NUM_KV_HEADS, n_suffix);
    suffix_v = ggml_cont_2d(ctx0, suffix_v, HEAD_DIM * NUM_KV_HEADS, n_suffix);

    // 4. Concatenate cached prefix K/V with computed suffix K/V
    // full_k: [HEAD_DIM * NUM_KV_HEADS, n_total]
    ggml_tensor *full_k = ggml_concat(ctx0, prefix_k_cache, suffix_k, 1);
    ggml_tensor *full_v = ggml_concat(ctx0, prefix_v_cache, suffix_v, 1);

    // 5. Attention: suffix queries attend to full K/V
    // attn_mask shape: [n_total, n_suffix] (suffix queries only)
    ggml_tensor *suffix_attn_out = build_gqa_attention(
        ctx0, suffix_q, full_k, full_v, attn_mask,
        NUM_HEADS, NUM_KV_HEADS, HEAD_DIM
    );

    // 6. Output projection (suffix output is in expert dim)
    suffix_attn_out = ggml_mul_mat(ctx0, layer.expert_o_w, suffix_attn_out);

    // 7. Gated residual connection
    suffix_attn_out = ggml_mul(ctx0, suffix_attn_out, attn_gate);
    suffix_hidden = ggml_add(ctx0, suffix_hidden, suffix_attn_out);

    // =====================================================================
    // MLP Block
    // =====================================================================

    // Suffix MLP: AdaRMSNorm -> SiLU-gated MLP -> gated residual
    AdaRMSNormResult suffix_ffn_norm_result = build_ada_rms_norm(
        ctx0, suffix_hidden,
        layer.expert_ffn_adaln_w, layer.expert_ffn_adaln_b,
        adarms_cond, norm_eps
    );
    ggml_tensor *suffix_mlp_in = suffix_ffn_norm_result.output;
    ggml_tensor *mlp_gate = suffix_ffn_norm_result.gate;

    ggml_tensor *suffix_gate_out = ggml_mul_mat(ctx0, layer.expert_gate_w, suffix_mlp_in);
    suffix_gate_out = ggml_gelu(ctx0, suffix_gate_out);
    ggml_tensor *suffix_up_out = ggml_mul_mat(ctx0, layer.expert_up_w, suffix_mlp_in);
    ggml_tensor *suffix_mlp_out = ggml_mul(ctx0, suffix_gate_out, suffix_up_out);
    suffix_mlp_out = ggml_mul_mat(ctx0, layer.expert_down_w, suffix_mlp_out);
    suffix_mlp_out = ggml_mul(ctx0, suffix_mlp_out, mlp_gate);
    suffix_hidden = ggml_add(ctx0, suffix_hidden, suffix_mlp_out);

    return suffix_hidden;
}

// ============ Phase 4: Suffix Layer with Memory Integration ============

SuffixLayerWithMemoryResult build_suffix_layer_with_memory(
    ggml_context *ctx0,
    ggml_tensor *suffix_hidden,
    Pi05Memory *memory,
    int n_prefix,
    ggml_tensor *suffix_positions,
    ggml_tensor *attn_mask,
    ggml_tensor *adarms_cond,
    const GemmaJointLayer &layer,
    float norm_eps,
    int il) {
    // Process suffix through one layer using Pi05Memory for KV cache
    // This version writes suffix K/V to persistent memory and reads full K/V from memory
    // NO ggml_concat is used - all K/V operations are through ggml_view and ggml_cpy

    const int NUM_HEADS = 8;
    const int NUM_KV_HEADS = 1;
    const int HEAD_DIM = 256;

    int64_t n_suffix = suffix_hidden->ne[1];
    int64_t n_total = n_prefix + n_suffix;

    // =====================================================================
    // Attention Block
    // =====================================================================

    // 1. Normalize suffix with AdaRMSNorm
    AdaRMSNormResult suffix_norm_result = build_ada_rms_norm(
        ctx0, suffix_hidden,
        layer.expert_attn_adaln_w, layer.expert_attn_adaln_b,
        adarms_cond, norm_eps
    );
    ggml_tensor *suffix_normed = suffix_norm_result.output;
    ggml_tensor *attn_gate = suffix_norm_result.gate;

    // 2. Compute Q/K/V for suffix
    ggml_tensor *suffix_q = ggml_mul_mat(ctx0, layer.expert_q_w, suffix_normed);
    ggml_tensor *suffix_k = ggml_mul_mat(ctx0, layer.expert_k_w, suffix_normed);
    ggml_tensor *suffix_v = ggml_mul_mat(ctx0, layer.expert_v_w, suffix_normed);

    // 3. Apply RoPE to suffix Q and K (positions start at n_prefix)
    suffix_q = ggml_reshape_3d(ctx0, suffix_q, HEAD_DIM, NUM_HEADS, n_suffix);
    suffix_k = ggml_reshape_3d(ctx0, suffix_k, HEAD_DIM, NUM_KV_HEADS, n_suffix);

    suffix_q = build_rope(ctx0, suffix_q, suffix_positions, HEAD_DIM, 10000.0f);
    suffix_k = build_rope(ctx0, suffix_k, suffix_positions, HEAD_DIM, 10000.0f);

    // Reshape back to 2D
    suffix_q = ggml_cont_2d(ctx0, suffix_q, HEAD_DIM * NUM_HEADS, n_suffix);
    suffix_k = ggml_cont_2d(ctx0, suffix_k, HEAD_DIM * NUM_KV_HEADS, n_suffix);
    suffix_v = ggml_cont_2d(ctx0, suffix_v, HEAD_DIM * NUM_KV_HEADS, n_suffix);

    // 4. Write suffix K/V to persistent memory at offset n_prefix
    // These cpy nodes MUST be added to the graph before attention reads the full K/V
    ggml_tensor *k_cpy = memory->cpy_k(ctx0, suffix_k, n_prefix, il);
    ggml_tensor *v_cpy = memory->cpy_v(ctx0, suffix_v, n_prefix, il);

    // 5. Get full K/V views from memory (prefix + suffix)
    // This creates views of the entire sequence [0, n_total)
    ggml_tensor *full_k = memory->get_k(ctx0, il, n_total);
    ggml_tensor *full_v = memory->get_v(ctx0, il, n_total);

    // 6. Attention: suffix queries attend to full K/V
    // attn_mask shape: [n_total, n_suffix] (suffix queries only)
    ggml_tensor *suffix_attn_out = build_gqa_attention(
        ctx0, suffix_q, full_k, full_v, attn_mask,
        NUM_HEADS, NUM_KV_HEADS, HEAD_DIM
    );

    // 7. Output projection (suffix output is in expert dim)
    suffix_attn_out = ggml_mul_mat(ctx0, layer.expert_o_w, suffix_attn_out);

    // 8. Gated residual connection
    suffix_attn_out = ggml_mul(ctx0, suffix_attn_out, attn_gate);
    suffix_hidden = ggml_add(ctx0, suffix_hidden, suffix_attn_out);

    // =====================================================================
    // MLP Block
    // =====================================================================

    // Suffix MLP: AdaRMSNorm -> GELU-gated MLP -> gated residual
    AdaRMSNormResult suffix_ffn_norm_result = build_ada_rms_norm(
        ctx0, suffix_hidden,
        layer.expert_ffn_adaln_w, layer.expert_ffn_adaln_b,
        adarms_cond, norm_eps
    );
    ggml_tensor *suffix_mlp_in = suffix_ffn_norm_result.output;
    ggml_tensor *mlp_gate = suffix_ffn_norm_result.gate;

    ggml_tensor *suffix_gate_out = ggml_mul_mat(ctx0, layer.expert_gate_w, suffix_mlp_in);
    suffix_gate_out = ggml_gelu(ctx0, suffix_gate_out);
    ggml_tensor *suffix_up_out = ggml_mul_mat(ctx0, layer.expert_up_w, suffix_mlp_in);
    ggml_tensor *suffix_mlp_out = ggml_mul(ctx0, suffix_gate_out, suffix_up_out);
    suffix_mlp_out = ggml_mul_mat(ctx0, layer.expert_down_w, suffix_mlp_out);
    suffix_mlp_out = ggml_mul(ctx0, suffix_mlp_out, mlp_gate);
    suffix_hidden = ggml_add(ctx0, suffix_hidden, suffix_mlp_out);

    return {suffix_hidden, k_cpy, v_cpy};
}
