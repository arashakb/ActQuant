#pragma once

#include "ggml.h"
#include <vector>

// Helper to add tensors with potentially different types (cast to match first operand)
inline ggml_tensor *safe_add(ggml_context *ctx0, ggml_tensor *a, ggml_tensor *b) {
    if (b->type != a->type) {
        b = ggml_cast(ctx0, b, a->type);
    }
    return ggml_add(ctx0, a, b);
}

// Helper to multiply tensors with potentially different types
inline ggml_tensor *safe_mul(ggml_context *ctx0, ggml_tensor *a, ggml_tensor *b) {
    if (b->type != a->type) {
        b = ggml_cast(ctx0, b, a->type);
    }
    return ggml_mul(ctx0, a, b);
}

enum norm_type {
    NORM_TYPE_NORMAL,
    NORM_TYPE_RMS,
};

enum ffn_op_type {
    FFN_GELU,
    FFN_GELU_ERF,
    FFN_SILU,
    FFN_GELU_QUICK,
};

void cb(ggml_context *ctx0, ggml_tensor *cur0, const char *name, int il = -1);

ggml_tensor *build_norm(ggml_context *ctx0, ggml_tensor *cur, ggml_tensor *mw,
                        ggml_tensor *mb, norm_type type, float norm_eps, int il = -1);

ggml_tensor *build_linear(ggml_context *ctx0, ggml_tensor *cur, ggml_tensor *w,
                          ggml_tensor *b, int il = -1);

ggml_tensor *build_ffn(ggml_context *ctx0, ggml_tensor *cur, ggml_tensor *up,
                       ggml_tensor *up_b, ggml_tensor *gate, ggml_tensor *gate_b,
                       ggml_tensor *down, ggml_tensor *down_b, ffn_op_type type_op, int il = -1);

ggml_tensor *build_attn(ggml_context *ctx0, ggml_tensor *wo, ggml_tensor *wo_b,
                        ggml_tensor *q_cur, ggml_tensor *k_cur, ggml_tensor *v_cur,
                        ggml_tensor *kq_mask, float kq_scale, int il = -1);

// RoPE (Rotary Position Embedding) for joint attention
// Input: x [head_dim, n_heads, n_tokens] or [head_dim, 1, n_tokens] for KV
// positions: [n_tokens] (I32) - position indices
// Returns: tensor with RoPE applied, same shape as input
ggml_tensor *build_rope(ggml_context *ctx0, ggml_tensor *x, ggml_tensor *positions,
                        int n_dims = 256, float freq_base = 10000.0f);

// Build joint attention mask for prefix-suffix attention
// Returns: mask [n_total, n_total] where:
//   - prefix can attend to prefix (bidirectional)
//   - prefix cannot attend to suffix
//   - suffix can attend to all (bidirectional with suffix)
ggml_tensor *build_joint_attn_mask(ggml_context *ctx0, int n_prefix, int n_suffix);

// Fill joint attention mask values (call after tensor allocation, before graph compute)
// data: pointer to mask tensor data
// Layout: [n_kv, n_q] in column-major order (GGML)
void fill_joint_attn_mask(float *data, int n_prefix, int n_suffix);

// AdaRMSNorm: Adaptive RMS normalization with scale, shift, and gate
// x: input tensor [hidden_dim, n_tokens]
// w: AdaLN weight [3*hidden_dim, cond_dim] - projects cond to scale, shift, gate
// b: AdaLN bias [3*hidden_dim]
// cond: conditioning tensor [cond_dim, 1] (e.g., timestep embedding)
// Returns: (normalized_output, gate) where gate is for gated residual
struct AdaRMSNormResult {
    ggml_tensor *output;
    ggml_tensor *gate;
};
AdaRMSNormResult build_ada_rms_norm(ggml_context *ctx0, ggml_tensor *x,
                                     ggml_tensor *w, ggml_tensor *b,
                                     ggml_tensor *cond, float norm_eps = 1e-6f);

// GQA attention for joint prefix-suffix attention
// q: query tensor [head_dim * n_q_heads, n_tokens]
// k: key tensor [head_dim * n_kv_heads, n_tokens]
// v: value tensor [head_dim * n_kv_heads, n_tokens]
// mask: attention mask [n_kv_tokens, n_q_tokens] (0 = attend, -inf = block)
// Returns: attention output [head_dim * n_q_heads, n_tokens]
ggml_tensor *build_gqa_attention(ggml_context *ctx0,
                                  ggml_tensor *q, ggml_tensor *k, ggml_tensor *v,
                                  ggml_tensor *mask,
                                  int n_q_heads, int n_kv_heads, int head_dim);

// Forward declaration for GemmaJointLayer
struct GemmaJointLayer;

// Result of joint layer forward pass
struct JointLayerResult {
    ggml_tensor *prefix_out;  // [pali_dim, n_prefix]
    ggml_tensor *suffix_out;  // [expert_dim, n_suffix]
};

// Joint layer forward pass
// Processes prefix (PaliGemma) and suffix (Expert) together in joint attention
//
// prefix_hidden: [pali_dim=2048, n_prefix]
// suffix_hidden: [expert_dim=1024, n_suffix]
// positions: [n_total] (I32) - position indices for RoPE
// attn_mask: [n_total, n_total] - attention mask (0=attend, -inf=block)
// adarms_cond: [cond_dim=1024, 1] - conditioning for AdaRMSNorm
// layer: GemmaJointLayer containing all weights
//
// Returns: (new_prefix_hidden, new_suffix_hidden)
JointLayerResult build_joint_layer_forward(
    ggml_context *ctx0,
    ggml_tensor *prefix_hidden,
    ggml_tensor *suffix_hidden,
    ggml_tensor *positions,
    ggml_tensor *attn_mask,
    ggml_tensor *adarms_cond,
    const GemmaJointLayer &layer,
    float norm_eps = 1e-6f,
    int il = -1);

// ============ KV Cache Operations ============

// Result of prefix-only forward pass (for cache computation)
struct PrefixLayerResult {
    ggml_tensor *prefix_out;   // [pali_dim, n_prefix] - updated hidden states
    ggml_tensor *k_cache;      // [head_dim * n_kv_heads, n_prefix] - K to cache (after RoPE)
    ggml_tensor *v_cache;      // [head_dim * n_kv_heads, n_prefix] - V to cache
};

// Forward pass for prefix only (first pass - computes and returns K/V for cache)
// This processes prefix through one layer and returns K/V to be cached
//
// prefix_hidden: [pali_dim=2048, n_prefix]
// positions: [n_prefix] (I32) - position indices for RoPE
// attn_mask: [n_prefix, n_prefix] - prefix self-attention mask (all zeros for bidirectional)
// layer: GemmaJointLayer containing PaliGemma weights
//
// Returns: (new_prefix_hidden, k_cache, v_cache)
PrefixLayerResult build_prefix_layer_forward(
    ggml_context *ctx0,
    ggml_tensor *prefix_hidden,
    ggml_tensor *positions,
    ggml_tensor *attn_mask,
    const GemmaJointLayer &layer,
    float norm_eps = 1e-6f,
    int il = -1);

// Joint layer forward using cached prefix K/V (subsequent ODE steps)
// Only processes suffix through attention, using cached prefix K/V
//
// suffix_hidden: [expert_dim=1024, n_suffix]
// prefix_k_cache: [head_dim * n_kv_heads, n_prefix] - cached K (after RoPE)
// prefix_v_cache: [head_dim * n_kv_heads, n_prefix] - cached V
// suffix_positions: [n_suffix] (I32) - position indices for suffix (starting at n_prefix)
// attn_mask: [n_prefix + n_suffix, n_suffix] - attention mask for suffix queries
// adarms_cond: [cond_dim=1024, 1] - conditioning for AdaRMSNorm
// layer: GemmaJointLayer containing Expert weights
//
// Returns: new_suffix_hidden
ggml_tensor* build_suffix_layer_with_cache(
    ggml_context *ctx0,
    ggml_tensor *suffix_hidden,
    ggml_tensor *prefix_k_cache,
    ggml_tensor *prefix_v_cache,
    ggml_tensor *suffix_positions,
    ggml_tensor *attn_mask,
    ggml_tensor *adarms_cond,
    const GemmaJointLayer &layer,
    float norm_eps = 1e-6f,
    int il = -1);

// Build attention mask for suffix-only queries attending to full sequence
// Returns: mask [n_prefix + n_suffix, n_suffix] where:
//   - All suffix queries can attend to all keys (prefix + suffix)
ggml_tensor *build_suffix_attn_mask(ggml_context *ctx0, int n_prefix, int n_suffix);

// Fill attention mask for suffix queries
// masked_start/masked_end: range of prefix tokens to mask (e.g., right_wrist placeholder)
// If masked_start < 0, no tokens are masked (all zeros - can attend to everything)
void fill_suffix_attn_mask(float *data, int n_prefix, int n_suffix,
                           int masked_start = -1, int masked_end = -1);

// Forward declaration for Pi05Memory
class Pi05Memory;

// ============ Phase 4: Suffix Layer with Memory Integration ============

// Result of suffix layer forward with memory (includes copy nodes for graph)
struct SuffixLayerWithMemoryResult {
    ggml_tensor* suffix_out;  // Updated suffix hidden states
    ggml_tensor* k_cpy;       // Copy node for K (must be in graph before attention reads)
    ggml_tensor* v_cpy;       // Copy node for V (must be in graph before attention reads)
};

// Suffix layer forward using Pi05Memory for KV cache management
// This version writes suffix K/V to persistent memory and reads full K/V from memory
// NO ggml_concat is used - all K/V operations are through ggml_view and ggml_cpy
//
// IMPORTANT: The returned k_cpy and v_cpy nodes MUST be added to the graph with
// ggml_build_forward_expand BEFORE the suffix_out node, to ensure suffix K/V is
// written to memory before attention reads the full K/V.
//
// suffix_hidden: [expert_dim=1024, n_suffix]
// memory: Pi05Memory instance holding persistent KV cache
// n_prefix: number of prefix tokens (already stored in memory)
// suffix_positions: [n_suffix] (I32) - position indices for suffix
// attn_mask: [n_prefix + n_suffix, n_suffix] - attention mask
// adarms_cond: [cond_dim=1024, 1] - conditioning for AdaRMSNorm
// layer: GemmaJointLayer containing Expert weights
// il: layer index
//
// Returns: SuffixLayerWithMemoryResult containing suffix_out and copy nodes
SuffixLayerWithMemoryResult build_suffix_layer_with_memory(
    ggml_context *ctx0,
    ggml_tensor *suffix_hidden,
    Pi05Memory *memory,
    int n_prefix,
    ggml_tensor *suffix_positions,
    ggml_tensor *attn_mask,
    ggml_tensor *adarms_cond,
    const GemmaJointLayer &layer,
    float norm_eps = 1e-6f,
    int il = 0);
