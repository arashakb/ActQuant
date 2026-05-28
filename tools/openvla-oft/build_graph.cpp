#include "build_graph.h"
#include <string>

void cb(ggml_context *ctx0, ggml_tensor *cur0, const char *name, int il) {
  // TODO
  //   if (ctx_manager_->debug_graph_) {
  //     ggml_context *ctx0 = ctx_compute_.get();
  //     ggml_tensor *cur = ggml_cpy(ctx0, cur0, ggml_dup_tensor(ctx0, cur0));
  //     std::string cur_name =
  //         il >= 0 ? std::string(name) + "_" + std::to_string(il) : name;
  //     ggml_set_name(cur, cur_name.c_str());
  //     ggml_set_output(cur);
  //     ggml_build_forward_expand(gf, cur);
  //     ctx_manager_->debug_print_tensors_.push_back(cur);
  //   }
}

ggml_tensor *build_norm(ggml_context *ctx0, ggml_tensor *cur, ggml_tensor *mw,
                        ggml_tensor *mb, NormType type, float norm_eps,
                        int il) {
  cur = type == NormType::RMS ? ggml_rms_norm(ctx0, cur, norm_eps)
                              : ggml_norm(ctx0, cur, norm_eps);

  if (mw || mb) {
    cb(ctx0, cur, "norm", il);
  }

  if (mw) {
    cur = ggml_mul(ctx0, cur, mw);
    if (mb) {
      cb(ctx0, cur, "norm_w", il);
    }
  }

  if (mb) {
    cur = ggml_add(ctx0, cur, mb);
  }

  return cur;
}

ggml_tensor *build_linear(ggml_context *ctx0, ggml_tensor *cur, ggml_tensor *w,
                          ggml_tensor *b, int il) {
  cur = ggml_mul_mat(ctx0, w, cur);
  cb(ctx0, cur, "linear", il);
  if (b) {
    cur = ggml_add(ctx0, cur, b);
    cb(ctx0, cur, "linear_b", il);
  }
  return cur;
}

ggml_tensor *build_ffn(ggml_context *ctx0, ggml_tensor *cur, ggml_tensor *up,
                       ggml_tensor *up_b, ggml_tensor *gate,
                       ggml_tensor *gate_b, ggml_tensor *down,
                       ggml_tensor *down_b, FfnOpType type_op, int il) {
  ggml_tensor *tmp = up ? ggml_mul_mat(ctx0, up, cur) : cur;
  cb(ctx0, tmp, "ffn_up", il);

  if (up_b) {
    tmp = ggml_add(ctx0, tmp, up_b);
    cb(ctx0, tmp, "ffn_up_b", il);
  }

  if (gate) {
    cur = ggml_mul_mat(ctx0, gate, cur);
    cb(ctx0, cur, "ffn_gate", il);

    if (gate_b) {
      cur = ggml_add(ctx0, cur, gate_b);
      cb(ctx0, cur, "ffn_gate_b", il);
    }
  } else {
    cur = tmp;
  }

  // we only support parallel ffn for now
  switch (type_op) {
  case FfnOpType::Silu:
    if (gate) {
      cur = ggml_swiglu_split(ctx0, cur, tmp);
      cb(ctx0, cur, "ffn_swiglu", il);
    } else {
      cur = ggml_silu(ctx0, cur);
      cb(ctx0, cur, "ffn_silu", il);
    }
    break;
  case FfnOpType::Gelu:
    if (gate) {
      cur = ggml_geglu_split(ctx0, cur, tmp);
      cb(ctx0, cur, "ffn_geglu", il);
    } else {
      cur = ggml_gelu(ctx0, cur);
      cb(ctx0, cur, "ffn_gelu", il);
    }
    break;
  case FfnOpType::GeluErf:
    if (gate) {
      cur = ggml_geglu_erf_split(ctx0, cur, tmp);
      cb(ctx0, cur, "ffn_geglu_erf", il);
    } else {
      cur = ggml_gelu_erf(ctx0, cur);
      cb(ctx0, cur, "ffn_gelu_erf", il);
    }
    break;
  case FfnOpType::GeluQuick:
    if (gate) {
      cur = ggml_geglu_quick_split(ctx0, cur, tmp);
      cb(ctx0, cur, "ffn_geglu_quick", il);
    } else {
      cur = ggml_gelu_quick(ctx0, cur);
      cb(ctx0, cur, "ffn_gelu_quick", il);
    }
    break;
  }

  if (down) {
    cur = ggml_mul_mat(ctx0, down, cur);
  }

  if (down_b) {
    cb(ctx0, cur, "ffn_down", il);
  }

  if (down_b) {
    cur = ggml_add(ctx0, cur, down_b);
  }

  return cur;
}

ggml_tensor *build_attn(ggml_context *ctx0, ggml_tensor *wo, ggml_tensor *wo_b,
                        ggml_tensor *q_cur, ggml_tensor *k_cur,
                        ggml_tensor *v_cur, ggml_tensor *kq_mask,
                        float kq_scale, int il) {
  // these nodes are added to the graph together so that they are not
  // reordered by doing so, the number of splits in the graph is reduced
  //   ggml_build_forward_expand(gf, q_cur);
  //   ggml_build_forward_expand(gf, k_cur);
  //   ggml_build_forward_expand(gf, v_cur);

  ggml_tensor *q = ggml_permute(ctx0, q_cur, 0, 2, 1, 3);
  // cb(q, "q", il);

  ggml_tensor *k = ggml_permute(ctx0, k_cur, 0, 2, 1, 3);
  // cb(k, "k", il);

  ggml_tensor *v = ggml_permute(ctx0, v_cur, 1, 2, 0, 3);
  v = ggml_cont(ctx0, v);
  // cb(k, "v", il);

  ggml_tensor *cur;

  // TODO @ngxson : support flash attention
  {
    const auto n_tokens = q->ne[1];
    const auto n_head = q->ne[2];
    // const auto n_kv     = k->ne[1]; // for flash attention

    ggml_tensor *kq = ggml_mul_mat(ctx0, k, q);
    // F32 may not needed for vision encoders?
    // ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

    kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, 0.0f);

    ggml_tensor *kqv = ggml_mul_mat(ctx0, v, kq);
    cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
    cur = ggml_cont_2d(ctx0, cur, cur->ne[0] * n_head, n_tokens);
  }

  cb(ctx0, cur, "kqv_out", il);

  if (wo) {
    cur = ggml_mul_mat(ctx0, wo, cur);
  }

  if (wo_b) {
    cur = ggml_add(ctx0, cur, wo_b);
  }

  return cur;
}