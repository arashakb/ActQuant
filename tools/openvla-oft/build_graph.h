#pragma once

#include "ggml.h"
#include <unordered_set>
#include <vector>

enum class NormType {
  Normal,
  RMS,
};

enum class FfnOpType {
  Gelu,
  GeluErf,
  Silu,
  GeluQuick,
};

void cb(ggml_context *ctx0, ggml_tensor *cur0, const char *name, int il = -1);

ggml_tensor *build_norm(ggml_context *ctx0, ggml_tensor *cur, ggml_tensor *mw,
                        ggml_tensor *mb, NormType type, float norm_eps,
                        int il = -1);

ggml_tensor *build_linear(ggml_context *ctx0, ggml_tensor *cur, ggml_tensor *w,
                          ggml_tensor *b, int il = -1);

ggml_tensor *build_ffn(ggml_context *ctx0, ggml_tensor *cur, ggml_tensor *up,
                       ggml_tensor *up_b, ggml_tensor *gate,
                       ggml_tensor *gate_b, ggml_tensor *down,
                       ggml_tensor *down_b, FfnOpType type_op, int il = -1);

ggml_tensor *build_attn(ggml_context *ctx0, ggml_tensor *wo, ggml_tensor *wo_b,
                        ggml_tensor *q_cur, ggml_tensor *k_cur,
                        ggml_tensor *v_cur, ggml_tensor *kq_mask,
                        float kq_scale, int il = -1);

