#include "model_defs.h"
#include "build_graph.h"
#include "ggml.h"
#include <cmath>
#include <cstring>
#include <map>
#include "utils.h"

bool BaseModel::load_tensors(ModelLoader &model_loader,
                             ContextManager &ctx_manager) {
  std::map<std::string, size_t> tensor_offset;
  std::map<std::string, int64_t> tensor_original_ne0;  // For padded tensors
  gguf_context_ptr &ctx_gguf = model_loader.ctx_gguf_;

  ctx_manager.ctx_data_ = std::move(model_loader.ctx_meta_);
  ggml_context *ctx = ctx_manager.ctx_data_.get();

  for (int64_t i = 0; i < gguf_get_n_tensors(ctx_gguf.get()); ++i) {
    const char *name = gguf_get_tensor_name(ctx_gguf.get(), i);
    tensor_offset[name] = gguf_get_data_offset(ctx_gguf.get()) +
                          gguf_get_tensor_offset(ctx_gguf.get(), i);
  }

  // Check for padded tensors metadata
  int padded_count_idx = gguf_find_key(ctx_gguf.get(), "quantize.padded_tensor_count");
  if (padded_count_idx >= 0) {
    uint32_t padded_count = gguf_get_val_u32(ctx_gguf.get(), padded_count_idx);
    printf("%s: found %u padded tensor(s) in metadata\n", __func__, padded_count);

    // Load original dimensions for padded tensors
    for (int64_t i = 0; i < gguf_get_n_tensors(ctx_gguf.get()); ++i) {
      const char *name = gguf_get_tensor_name(ctx_gguf.get(), i);
      std::string meta_key = "quantize.original_ne0." + std::string(name);
      int meta_idx = gguf_find_key(ctx_gguf.get(), meta_key.c_str());

      if (meta_idx >= 0) {
        int64_t original_ne0 = gguf_get_val_u32(ctx_gguf.get(), meta_idx);
        tensor_original_ne0[name] = original_ne0;
        printf("%s: tensor '%s' was padded, original ne[0] = %lld\n",
               __func__, name, (long long)original_ne0);
      }
    }
  }

  std::vector<ggml_tensor *> tensors_to_load = get_tensors_to_load(ctx);

  {
    std::vector<uint8_t> read_buf;

    auto fin = std::ifstream(model_loader.fname_, std::ios::binary);
    if (!fin) {
      throw std::runtime_error(string_format(
          "%s: failed to open %s\n", __func__, model_loader.fname_.c_str()));
    }

    ggml_backend_buffer_type_t buft =
        ggml_backend_get_default_buffer_type(ctx_manager.backend_.get());
    ctx_manager.buffer_.reset(ggml_backend_alloc_ctx_tensors_from_buft(
        ctx_manager.ctx_data_.get(), buft));
    ggml_backend_buffer_set_usage(ctx_manager.buffer_.get(),
                                  GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (auto &cur : tensors_to_load) {
      const size_t offset = tensor_offset[cur->name];
      fin.seekg(offset, std::ios::beg);
      if (!fin) {
        throw std::runtime_error(string_format(
            "%s: failed to seek for tensor %s\n", __func__, cur->name));
      }
      size_t num_bytes = ggml_nbytes(cur);
      if (ggml_backend_buft_is_host(buft)) {
        // for the CPU and Metal backend, we can read directly into the tensor
        fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
      } else {
        // read into a temporary buffer first, then copy to device memory
        read_buf.resize(num_bytes);
        fin.read(reinterpret_cast<char *>(read_buf.data()), num_bytes);
        // save_vector_to_file(read_buf, std::string(cur->name)+".raw");
        ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
      }

      // Handle padded tensors: update ne[0] to original dimension
      auto it = tensor_original_ne0.find(cur->name);
      if (it != tensor_original_ne0.end()) {
        int64_t original_ne0 = it->second;
        int64_t padded_ne0 = cur->ne[0];

        printf("%s: adjusting tensor '%s' from padded ne[0]=%lld to original ne[0]=%lld\n",
               __func__, cur->name, (long long)padded_ne0, (long long)original_ne0);

        // Update dimension - GGML will only use first original_ne0 rows
        cur->ne[0] = original_ne0;

        // Update byte strides (nb) to reflect new dimensions
        // nb[0] remains the same (element size)
        // nb[1] needs to account for the padded row size in memory
        // Keep nb[1] unchanged as it was set based on the padded size
        // This way the data is stored with padding but accessed with original dims
      }
    }
    fin.close();

    printf("%s: loaded %zu tensors from %s\n", __func__, tensors_to_load.size(),
           model_loader.fname_.c_str());
  }
  return true;
}
std::vector<ggml_tensor *> FakeModel::get_tensors_to_load(ggml_context *ctx) {
  std::vector<ggml_tensor *> tensors_to_load;
  {
    embed_tokens = get_tensor(ctx, "token_embd.weight", tensors_to_load, false);
    llm_head = get_tensor(ctx, "output.weight", tensors_to_load, false);
  }
  return tensors_to_load;
}

// ==========================================ProjectorModel===========================================
std::vector<ggml_tensor *>
ProjectorModel::get_tensors_to_load(ggml_context *ctx) {
  std::vector<ggml_tensor *> tensors_to_load;
  fc1_weight = get_tensor(ctx, "fc1.weight", tensors_to_load);
  fc1_bias = get_tensor(ctx, "fc1.bias", tensors_to_load);
  fc2_weight = get_tensor(ctx, "fc2.weight", tensors_to_load);
  fc2_bias = get_tensor(ctx, "fc2.bias", tensors_to_load);
  fc3_weight = get_tensor(ctx, "fc3.weight", tensors_to_load);
  fc3_bias = get_tensor(ctx, "fc3.bias", tensors_to_load);
  return tensors_to_load;
}

std::vector<ggml_tensor *> ProjectorModel::build_graph(ggml_context *ctx0) {
  ggml_tensor *inp_dinov2 = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1024, 256);
  ggml_set_name(inp_dinov2, "dinov2_feat");
  ggml_set_input(inp_dinov2);

  ggml_tensor *inp_siglip = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1152, 256);
  ggml_set_name(inp_siglip, "siglip_feat");
  ggml_set_input(inp_siglip);

  ggml_tensor *inp = ggml_concat(ctx0, inp_dinov2, inp_siglip, 0);
  inp = build_linear(ctx0, inp, fc1_weight, fc1_bias);
  inp = ggml_gelu(ctx0, inp);

  inp = build_linear(ctx0, inp, fc2_weight, fc2_bias);
  inp = ggml_gelu(ctx0, inp);

  inp = build_linear(ctx0, inp, fc3_weight, fc3_bias);
  return {inp};
}

// ==========================================VisionTransformerModel===========================================
std::vector<ggml_tensor *>
VisionTransformerModel::get_tensors_to_load(ggml_context *ctx) {
  std::vector<ggml_tensor *> tensors_to_load;
  {
    class_embedding = get_tensor(ctx, "v.class_embd", tensors_to_load, false);
    reg_embedding = get_tensor(ctx, "v.reg_embd", tensors_to_load, false);

    pre_ln_w = get_tensor(ctx, "v.pre_ln.weight", tensors_to_load, false);
    pre_ln_b = get_tensor(ctx, "v.pre_ln.bias", tensors_to_load, false);

    post_ln_w = get_tensor(ctx, "v.post_ln.weight", tensors_to_load, false);
    post_ln_b = get_tensor(ctx, "v.post_ln.bias", tensors_to_load, false);

    patch_bias = get_tensor(ctx, "v.patch_embd.bias", tensors_to_load, false);
    patch_embeddings_0 =
        get_tensor(ctx, "v.patch_embd.weight", tensors_to_load, false);
    patch_embeddings_1 =
        get_tensor(ctx, "v.patch_embd.weight.1", tensors_to_load, false);

    position_embeddings =
        get_tensor(ctx, "v.position_embd.weight", tensors_to_load, false);

    // layers
    int n_layer = hparams.n_layer;
    layers.resize(n_layer);
    const char *prefix = "v";
    for (int il = 0; il < n_layer; ++il) {
      auto &layer = layers[il];
      layer.k_w = get_tensor(
          ctx, string_format("%s.blk.%d.attn_k.%s", prefix, il, "weight"),
          tensors_to_load);
      layer.q_w = get_tensor(
          ctx, string_format("%s.blk.%d.attn_q.%s", prefix, il, "weight"),
          tensors_to_load);
      layer.v_w = get_tensor(
          ctx, string_format("%s.blk.%d.attn_v.%s", prefix, il, "weight"),
          tensors_to_load);
      layer.o_w = get_tensor(
          ctx, string_format("%s.blk.%d.attn_out.%s", prefix, il, "weight"),
          tensors_to_load);
      layer.k_norm = get_tensor(
          ctx, string_format("%s.blk.%d.attn_k_norm.%s", prefix, il, "weight"),
          tensors_to_load, false);
      layer.q_norm = get_tensor(
          ctx, string_format("%s.blk.%d.attn_q_norm.%s", prefix, il, "weight"),
          tensors_to_load, false);
      layer.ln_1_w = get_tensor(
          ctx, string_format("%s.blk.%d.ln1.%s", prefix, il, "weight"),
          tensors_to_load, false);
      layer.ln_2_w = get_tensor(
          ctx, string_format("%s.blk.%d.ln2.%s", prefix, il, "weight"),
          tensors_to_load, false);
      layer.ls_1_w = get_tensor(
          ctx, string_format("%s.blk.%d.ls1.%s", prefix, il, "weight"),
          tensors_to_load,
          false); // no bias
      layer.ls_2_w = get_tensor(
          ctx, string_format("%s.blk.%d.ls2.%s", prefix, il, "weight"),
          tensors_to_load,
          false); // no bias

      layer.k_b = get_tensor(
          ctx, string_format("%s.blk.%d.attn_k.%s", prefix, il, "bias"),
          tensors_to_load, false);
      layer.q_b = get_tensor(
          ctx, string_format("%s.blk.%d.attn_q.%s", prefix, il, "bias"),
          tensors_to_load, false);
      layer.v_b = get_tensor(
          ctx, string_format("%s.blk.%d.attn_v.%s", prefix, il, "bias"),
          tensors_to_load, false);
      layer.o_b = get_tensor(
          ctx, string_format("%s.blk.%d.attn_out.%s", prefix, il, "bias"),
          tensors_to_load, false);
      layer.ln_1_b =
          get_tensor(ctx, string_format("%s.blk.%d.ln1.%s", prefix, il, "bias"),
                     tensors_to_load, false);
      layer.ln_2_b =
          get_tensor(ctx, string_format("%s.blk.%d.ln2.%s", prefix, il, "bias"),
                     tensors_to_load, false);

      // ffn
      layer.ff_up_w = get_tensor(
          ctx, string_format("%s.blk.%d.ffn_up.%s", prefix, il, "weight"),
          tensors_to_load);
      layer.ff_up_b = get_tensor(
          ctx, string_format("%s.blk.%d.ffn_up.%s", prefix, il, "bias"),
          tensors_to_load, false);
      layer.ff_gate_w = get_tensor(
          ctx, string_format("%s.blk.%d.ffn_gate.%s", prefix, il, "weight"),
          tensors_to_load, false);
      layer.ff_gate_b = get_tensor(
          ctx, string_format("%s.blk.%d.ffn_gate.%s", prefix, il, "bias"),
          tensors_to_load, false);
      layer.ff_down_w = get_tensor(
          ctx, string_format("%s.blk.%d.ffn_down.%s", prefix, il, "weight"),
          tensors_to_load);
      layer.ff_down_b = get_tensor(
          ctx, string_format("%s.blk.%d.ffn_down.%s", prefix, il, "bias"),
          tensors_to_load, false);
    }
  }
  return tensors_to_load;
}

bool VisionTransformerModel::load_hparams(const ModelLoader &model_loader) {
  {
    const char *prefix = "vision";
    model_loader.get_u32(string_format("clip.%s.embedding_length", prefix),
                         hparams.n_embd);
    model_loader.get_u32(string_format("clip.%s.attention.head_count", prefix),
                         hparams.n_head);
    model_loader.get_u32(string_format("clip.%s.feed_forward_length", prefix),
                         hparams.n_ff);
    model_loader.get_u32(string_format("clip.%s.block_count", prefix),
                         hparams.n_layer);
    model_loader.get_u32(string_format("clip.%s.projection_dim", prefix),
                         hparams.projection_dim);
    model_loader.get_f32(
        string_format("clip.%s.attention.layer_norm_epsilon", prefix),
        hparams.eps);
    model_loader.get_u32("clip.vision.image_size", hparams.image_size);
    model_loader.get_u32("clip.vision.patch_size", hparams.patch_size);

    // for pinpoints, we need to convert it into a list of resolution
    // candidates

    // default warmup value

    {
      bool use_gelu = false;
      bool use_silu = false;
      model_loader.get_bool("clip.use_gelu", use_gelu, false);
      model_loader.get_bool("clip.use_silu", use_silu, false);
      if (use_gelu && use_silu) {
        throw std::runtime_error(string_format(
            "%s: both use_gelu and use_silu are set to true\n", __func__));
      }
      if (use_gelu) {
        hparams.ffn_op = FfnOpType::Gelu;
      } else if (use_silu) {
        hparams.ffn_op = FfnOpType::Silu;
      } else {
        hparams.ffn_op = FfnOpType::GeluQuick;
      }
    }

    int idx_mean =
        gguf_find_key(model_loader.ctx_gguf_.get(), "clip.vision.image_mean");
    int idx_std =
        gguf_find_key(model_loader.ctx_gguf_.get(), "clip.vision.image_std");
    GGML_ASSERT(idx_mean >= 0 && "image_mean not found");
    GGML_ASSERT(idx_std >= 0 && "image_std not found");
    const float *mean_data = (const float *)gguf_get_arr_data(
        model_loader.ctx_gguf_.get(), idx_mean);
    const float *std_data =
        (const float *)gguf_get_arr_data(model_loader.ctx_gguf_.get(), idx_std);
    hparams.image_mean.resize(3);
    hparams.image_std.resize(3);
    for (int i = 0; i < 3; ++i) {
      hparams.image_mean[i] = mean_data[i];
      hparams.image_std[i] = std_data[i];
    }

    // Load the vision feature layer indices if they are explicitly provided;
    // if multiple vision feature layers are present, the values will be
    // concatenated to form the final visual features. NOTE: gguf conversions
    // should standardize the values of the vision feature layer to be
    // non-negative, since we use -1 to mark values as unset here.
    std::vector<int> vision_feature_layer;
    model_loader.get_arr_int("clip.vision.feature_layer", vision_feature_layer,
                             false);
    // convert std::vector to std::unordered_set
    for (auto &layer : vision_feature_layer) {
      hparams.vision_feature_layer.insert(layer);
    }
  }
  return true;
}

ggml_tensor *VisionTransformerModel::build_vit(
    ggml_context *ctx0, ggml_tensor *inp, int n_pos, NormType norm_t,
    std::function<ggml_tensor *(ggml_tensor *, const ClipLayer &)> add_pos) {
  int n_layer = hparams.n_layer;
  float eps = hparams.eps;
  int d_head = hparams.n_embd / hparams.n_head;
  int n_head = hparams.n_head;
  float kq_scale = 1.0f / sqrtf((float)d_head);
  int n_patches = hparams.image_size * hparams.image_size / hparams.patch_size /
                  hparams.patch_size;
  ggml_tensor *learned_pos_embd = position_embeddings;
  FfnOpType ffn_t = hparams.ffn_op;
  if (learned_pos_embd) {
    inp = ggml_add(ctx0, inp, learned_pos_embd);
    cb(ctx0, inp, "pos_embed", -1);
  }
  {
    // concat class_embedding、reg_embedding and inp
    ggml_tensor *to_cat = nullptr;
    if (class_embedding) {
      to_cat = class_embedding;
    }
    if (reg_embedding) {
      to_cat = ggml_concat(ctx0, to_cat, reg_embedding, 1);
    }
    if (to_cat) {
      inp = ggml_concat(ctx0, to_cat, inp, 1);
    }
  }
  // pre-layernorm
  if (pre_ln_w) {
    inp = build_norm(ctx0, inp, pre_ln_w, pre_ln_b, norm_t, eps, -1);
    cb(ctx0, inp, "pre_ln", -1);
  }
  ggml_tensor *inpL = inp;
  for (int il = 0; il < n_layer - 1; il++) {
    const auto &layer = layers[il];
    ggml_tensor *cur = inpL; // inpL = residual, cur = hidden_states

    // layernorm1
    cur = build_norm(ctx0, cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
    cb(ctx0, cur, "layer_inp_normed", il);

    // self-attention
    {
      ggml_tensor *Qcur = ggml_mul_mat(ctx0, layer.q_w, cur);
      if (layer.q_b) {
        Qcur = ggml_add(ctx0, Qcur, layer.q_b);
      }

      ggml_tensor *Kcur = ggml_mul_mat(ctx0, layer.k_w, cur);
      if (layer.k_b) {
        Kcur = ggml_add(ctx0, Kcur, layer.k_b);
      }

      ggml_tensor *Vcur = ggml_mul_mat(ctx0, layer.v_w, cur);
      if (layer.v_b) {
        Vcur = ggml_add(ctx0, Vcur, layer.v_b);
      }

      if (layer.q_norm) {
        Qcur = build_norm(ctx0, Qcur, layer.q_norm, NULL, norm_t, eps, il);
        cb(ctx0, Qcur, "Qcur_norm", il);
      }

      if (layer.k_norm) {
        Kcur = build_norm(ctx0, Kcur, layer.k_norm, NULL, norm_t, eps, il);
        cb(ctx0, Kcur, "Kcur_norm", il);
      }

      Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos);
      Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos);
      Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos);

      cb(ctx0, Qcur, "Qcur", il);
      cb(ctx0, Kcur, "Kcur", il);
      cb(ctx0, Vcur, "Vcur", il);

      if (add_pos) {
        Qcur = add_pos(Qcur, layer);
        Kcur = add_pos(Kcur, layer);
        cb(ctx0, Qcur, "Qcur_pos", il);
        cb(ctx0, Kcur, "Kcur_pos", il);
      }

      cur = build_attn(ctx0, layer.o_w, layer.o_b, Qcur, Kcur, Vcur, nullptr,
                       kq_scale, il);
      cb(ctx0, cur, "attn_out", il);
    }

    if (layer.ls_1_w) {
      cur = ggml_mul(ctx0, cur, layer.ls_1_w);
      cb(ctx0, cur, "attn_out_scaled", il);
    }

    // re-add the layer input, e.g., residual
    cur = ggml_add(ctx0, cur, inpL);

    inpL = cur; // inpL = residual, cur = hidden_states

    cb(ctx0, cur, "ffn_inp", il);

    // layernorm2
    cur = build_norm(ctx0, cur, layer.ln_2_w, layer.ln_2_b, norm_t, eps, il);
    cb(ctx0, cur, "ffn_inp_normed", il);

    // ffn
    cur =
        build_ffn(ctx0, cur, layer.ff_up_w, layer.ff_up_b, layer.ff_gate_w,
                  layer.ff_gate_b, layer.ff_down_w, layer.ff_down_b, ffn_t, il);

    cb(ctx0, cur, "ffn_out", il);

    if (layer.ls_2_w) {
      cur = ggml_mul(ctx0, cur, layer.ls_2_w);
      cb(ctx0, cur, "ffn_out_scaled", il);
    }

    // residual 2
    cur = ggml_add(ctx0, inpL, cur);
    cb(ctx0, cur, "layer_out", il);

    inpL = cur;
  }
  // post-layernorm
  if (post_ln_w) {
    inpL = build_norm(ctx0, inpL, post_ln_w, post_ln_b, norm_t, eps, -1);
  }
  int offset = n_pos - n_patches;
  if (offset > 0) {
    // remove class/reg tokens
    offset = offset * inpL->nb[1];
    inpL =
        ggml_view_2d(ctx0, inpL, inpL->ne[0], n_patches, inpL->nb[1], offset);
  }
  return inpL;
}
std::vector<ggml_tensor *>
VisionTransformerModel::build_graph(ggml_context *ctx0) {
  std::vector<ggml_tensor *> outputs;
  {
    int n_patches = hparams.image_size * hparams.image_size /
                    hparams.patch_size / hparams.patch_size;
    // build_inp
    ggml_tensor *inp_raw = ggml_new_tensor_3d(
        ctx0, GGML_TYPE_F32, hparams.image_size, hparams.image_size, 3);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);

    // conv2d
    ggml_tensor *inp =
        ggml_conv_2d(ctx0, patch_embeddings_0, inp_raw, hparams.patch_size,
                     hparams.patch_size, 0, 0, 1, 1);
    inp = ggml_reshape_2d(ctx0, inp, n_patches, hparams.n_embd);
    inp = ggml_cont(ctx0, ggml_transpose(ctx0, inp));
    if (patch_bias) {
      inp = ggml_add(ctx0, inp, patch_bias);
      cb(ctx0, inp, "patch_bias", -1);
    }

    int n_pos = n_patches;
    if (class_embedding) {
      n_pos += 1;
    }
    if (reg_embedding) {
      n_pos += 4;
    }
    ggml_tensor *cur = build_vit(ctx0, inp, n_pos, NormType::Normal, nullptr);
    outputs.push_back(cur);
  }
  return outputs;
}

// ==========================================L1RegressionActionHeadFunnelModel===========================================
std::vector<ggml_tensor *>
L1RegressionActionHeadFunnelModel::get_tensors_to_load(ggml_context *ctx) {
  std::vector<ggml_tensor *> tensors_to_load;
  {
    input_proj_ln_w =
        get_tensor(ctx, "input_proj.0.weight", tensors_to_load, true);
    input_proj_ln_b =
        get_tensor(ctx, "input_proj.0.bias", tensors_to_load, false);

    input_proj_fc_w =
        get_tensor(ctx, "input_proj.1.weight", tensors_to_load, true);
    input_proj_fc_b =
        get_tensor(ctx, "input_proj.1.bias", tensors_to_load, false);

    // blocks
    int num_blocks = hparams.num_blocks;
    resnet_body.resize(num_blocks);
    for (int il = 0; il < num_blocks; ++il) {
      auto &blk = resnet_body[il];
      blk.ffn_ln_w =
          get_tensor(ctx, string_format("resnet_body.%d.ffn.0.weight", il),
                     tensors_to_load, true);
      blk.ffn_ln_b =
          get_tensor(ctx, string_format("resnet_body.%d.ffn.0.bias", il),
                     tensors_to_load, false);
      blk.ffn_fc_w =
          get_tensor(ctx, string_format("resnet_body.%d.ffn.1.weight", il),
                     tensors_to_load, true);
      blk.ffn_fc_b =
          get_tensor(ctx, string_format("resnet_body.%d.ffn.1.bias", il),
                     tensors_to_load, false);
      blk.ffn_fc2_w =
          get_tensor(ctx, string_format("resnet_body.%d.ffn.3.weight", il),
                     tensors_to_load, true);
      blk.ffn_fc2_b =
          get_tensor(ctx, string_format("resnet_body.%d.ffn.3.bias", il),
                     tensors_to_load, false);
    }

    output_head_ln_w =
        get_tensor(ctx, "output_head.0.weight", tensors_to_load, true);
    output_head_ln_b =
        get_tensor(ctx, "output_head.0.bias", tensors_to_load, false);
    output_head_fc_w =
        get_tensor(ctx, "output_head.1.weight", tensors_to_load, true);
    output_head_fc_b =
        get_tensor(ctx, "output_head.1.bias", tensors_to_load, false);
  }
  return tensors_to_load;
}

bool L1RegressionActionHeadFunnelModel::load_hparams(
    const ModelLoader &model_loader) {
  model_loader.get_u32("action_dim", hparams.action_dim);
  model_loader.get_u32("num_actions_chunk", hparams.num_actions_chunk);
  model_loader.get_u32("num_actions_per_token", hparams.num_actions_per_token);
  model_loader.get_u32("num_blocks", hparams.num_blocks);
  model_loader.get_u32("input_dim", hparams.input_dim);
  model_loader.get_u32("hidden_dim", hparams.hidden_dim);
  model_loader.get_u32("expansion", hparams.expansion);
  return true;
}

ggml_tensor *MLPResNetBlockV2::build_graph(ggml_context *ctx0,
                                           ggml_tensor *inp) {
  ggml_tensor *cur =
      build_norm(ctx0, inp, ffn_ln_w, ffn_ln_b, NormType::Normal, 1e-5, -1);
  cur = build_linear(ctx0, cur, ffn_fc_w, ffn_fc_b, -1);
  cur = ggml_silu(ctx0, cur);
  cur = build_linear(ctx0, cur, ffn_fc2_w, ffn_fc2_b, -1);
  cur = ggml_add(ctx0, inp, cur);
  return cur;
}

std::vector<ggml_tensor *>
L1RegressionActionHeadFunnelModel::build_graph(ggml_context *ctx0) {
  std::vector<ggml_tensor *> outputs;
  {
    ggml_tensor *inp_raw = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.input_dim, 1);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);

    // input_proj
    ggml_tensor *cur = build_norm(ctx0, inp_raw, input_proj_ln_w,
                                  input_proj_ln_b, NormType::Normal, 1e-5, -1);
    cur = build_linear(ctx0, cur, input_proj_fc_w, input_proj_fc_b, -1);
    cur = ggml_silu(ctx0, cur);
    // blocks
    for (size_t i = 0; i < resnet_body.size(); i++) {
      cur = resnet_body[i].build_graph(ctx0, cur);
    }
    // output_head
    cur = build_norm(ctx0, cur, output_head_ln_w, output_head_ln_b,
                     NormType::Normal, 1e-5, -1);
    cur = build_linear(ctx0, cur, output_head_fc_w, output_head_fc_b, -1);
    cur = ggml_reshape_2d(ctx0, cur, hparams.num_actions_chunk, hparams.action_dim);
    outputs.push_back(cur);
  }
  return outputs;
}

// ==========================================L1RegressionActionHeadmulmlpk===========================================
std::vector<ggml_tensor *>
L1RegressionActionHeadmulmlpk::get_tensors_to_load(ggml_context *ctx) {
  std::vector<ggml_tensor *> tensors_to_load;
  {
    input_ln1_w =
        get_tensor(ctx, "model.layer_norm1.weight", tensors_to_load, true);
    input_ln1_b =
        get_tensor(ctx, "model.layer_norm1.bias", tensors_to_load, false);

    input_fc1_w =
        get_tensor(ctx, "model.fc1.weight", tensors_to_load, true);
    input_fc1_b =
        get_tensor(ctx, "model.fc1.bias", tensors_to_load, false);

    // blocks
    int num_blocks = hparams.num_blocks;
    resnet_body.resize(num_blocks);
    for (int il = 0; il < num_blocks; ++il) {
      auto &blk = resnet_body[il];
      blk.ffn_ln_w =
          get_tensor(ctx, string_format("model.mlp_resnet_blocks.%d.ffn.0.weight", il),
                     tensors_to_load, true);
      blk.ffn_ln_b =
          get_tensor(ctx, string_format("model.mlp_resnet_blocks.%d.ffn.0.bias", il),
                     tensors_to_load, false);
      blk.ffn_fc_w =
          get_tensor(ctx, string_format("model.mlp_resnet_blocks.%d.ffn.1.weight", il),
                     tensors_to_load, true);
      blk.ffn_fc_b =
          get_tensor(ctx, string_format("model.mlp_resnet_blocks.%d.ffn.1.bias", il),
                     tensors_to_load, false);
    }

    output_ln2_w =
        get_tensor(ctx, "model.layer_norm2.weight", tensors_to_load, true);
    output_ln2_b =
        get_tensor(ctx, "model.layer_norm2.bias", tensors_to_load, false);
    output_fc2_w =
        get_tensor(ctx, "model.fc2.weight", tensors_to_load, true);
    output_fc2_b =
        get_tensor(ctx, "model.fc2.bias", tensors_to_load, false);
  }
  return tensors_to_load;
}

bool L1RegressionActionHeadmulmlpk::load_hparams(
    const ModelLoader &model_loader) {
  model_loader.get_u32("action_dim", hparams.action_dim);
  model_loader.get_u32("num_actions_chunk", hparams.num_actions_chunk);
  model_loader.get_u32("num_actions_per_token", hparams.num_actions_per_token);
  model_loader.get_u32("num_blocks", hparams.num_blocks);
  model_loader.get_u32("input_dim", hparams.input_dim);
  model_loader.get_u32("hidden_dim", hparams.hidden_dim);
  // model_loader.get_u32("expansion", hparams.expansion);

  return true;
}

ggml_tensor *MLPResNetBlock::build_graph(ggml_context *ctx0,
                                           ggml_tensor *inp) {
  ggml_tensor *cur =
      build_norm(ctx0, inp, ffn_ln_w, ffn_ln_b, NormType::Normal, 1e-5, -1);
  cur = build_linear(ctx0, cur, ffn_fc_w, ffn_fc_b, -1);
  cur = ggml_silu(ctx0, cur);
  // cur = build_linear(ctx0, cur, ffn_fc2_w, ffn_fc2_b, -1);
  cur = ggml_add(ctx0, inp, cur);
  return cur;
}
std::vector<ggml_tensor *>
L1RegressionActionHeadmulmlpk::build_graph(ggml_context *ctx0) {
  std::vector<ggml_tensor *> outputs;
  {
    ggml_tensor *inp_raw = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.input_dim, 2);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);

    // input_proj
    ggml_tensor *cur = build_norm(ctx0, inp_raw, input_ln1_w,
                                  input_ln1_b, NormType::Normal, 1e-5, -1);
    cur = build_linear(ctx0, cur, input_fc1_w, input_fc1_b, -1);
    cur = ggml_silu(ctx0, cur);
    // blocks
    for (size_t i = 0; i < resnet_body.size(); i++) {
      cur = resnet_body[i].build_graph(ctx0, cur);
    }
    // output_head
    cur = build_norm(ctx0, cur, output_ln2_w, output_ln2_b,
                     NormType::Normal, 1e-5, -1);
    cur = build_linear(ctx0, cur, output_fc2_w, output_fc2_b, -1);
    cur = ggml_reshape_2d(ctx0, cur, hparams.num_actions_chunk, hparams.action_dim);
    outputs.push_back(cur);
  }
  return outputs;
}

// ==========================================L1RegressionActionHeadOFT===========================================
// OpenVLA-OFT specific action head implementation
// Input: hidden states from all action token positions, reshaped to (chunk_len, action_dim * hidden_dim)
// Output: (chunk_len, action_dim)

ggml_tensor *MLPResNetBlockOFT::build_graph(ggml_context *ctx0, ggml_tensor *inp) {
  // OFT block uses ReLU instead of SiLU
  // ffn = LayerNorm -> Linear -> ReLU
  // output = ffn(x) + x (residual)
  ggml_tensor *cur =
      build_norm(ctx0, inp, ffn_ln_w, ffn_ln_b, NormType::Normal, 1e-5, -1);
  cur = build_linear(ctx0, cur, ffn_fc_w, ffn_fc_b, -1);
  cur = ggml_relu(ctx0, cur);  // OFT uses ReLU
  cur = ggml_add(ctx0, inp, cur);  // residual connection
  return cur;
}

std::vector<ggml_tensor *>
L1RegressionActionHeadOFT::get_tensors_to_load(ggml_context *ctx) {
  std::vector<ggml_tensor *> tensors_to_load;
  {
    // Input layer norm and FC
    layer_norm1_w =
        get_tensor(ctx, "model.layer_norm1.weight", tensors_to_load, true);
    layer_norm1_b =
        get_tensor(ctx, "model.layer_norm1.bias", tensors_to_load, false);

    fc1_w = get_tensor(ctx, "model.fc1.weight", tensors_to_load, true);
    fc1_b = get_tensor(ctx, "model.fc1.bias", tensors_to_load, false);

    // MLP ResNet blocks
    int num_blocks = hparams.num_blocks;
    mlp_resnet_blocks.resize(num_blocks);
    for (int il = 0; il < num_blocks; ++il) {
      auto &blk = mlp_resnet_blocks[il];
      blk.ffn_ln_w = get_tensor(
          ctx, string_format("model.mlp_resnet_blocks.%d.ffn.0.weight", il),
          tensors_to_load, true);
      blk.ffn_ln_b = get_tensor(
          ctx, string_format("model.mlp_resnet_blocks.%d.ffn.0.bias", il),
          tensors_to_load, false);
      blk.ffn_fc_w = get_tensor(
          ctx, string_format("model.mlp_resnet_blocks.%d.ffn.1.weight", il),
          tensors_to_load, true);
      blk.ffn_fc_b = get_tensor(
          ctx, string_format("model.mlp_resnet_blocks.%d.ffn.1.bias", il),
          tensors_to_load, false);
    }

    // Output layer norm and FC
    layer_norm2_w =
        get_tensor(ctx, "model.layer_norm2.weight", tensors_to_load, true);
    layer_norm2_b =
        get_tensor(ctx, "model.layer_norm2.bias", tensors_to_load, false);

    fc2_w = get_tensor(ctx, "model.fc2.weight", tensors_to_load, true);
    fc2_b = get_tensor(ctx, "model.fc2.bias", tensors_to_load, false);
  }
  return tensors_to_load;
}

bool L1RegressionActionHeadOFT::load_hparams(const ModelLoader &model_loader) {
  model_loader.get_u32("action_dim", hparams.action_dim);
  model_loader.get_u32("num_actions_chunk", hparams.num_actions_chunk);
  model_loader.get_u32("num_blocks", hparams.num_blocks);
  model_loader.get_u32("input_dim", hparams.input_dim);
  model_loader.get_u32("hidden_dim", hparams.hidden_dim);
  return true;
}

std::vector<ggml_tensor *>
L1RegressionActionHeadOFT::build_graph(ggml_context *ctx0) {
  std::vector<ggml_tensor *> outputs;
  {
    // Input shape: (num_actions_chunk, action_dim * input_dim)
    // For LIBERO with hidden_dim=2048, action_dim=7, chunk_len=8:
    // Input: (8, 7 * 2048) = (8, 14336)
    uint32_t mlp_input_dim = hparams.action_dim * hparams.input_dim;
    ggml_tensor *inp_raw = ggml_new_tensor_2d(
        ctx0, GGML_TYPE_F32, mlp_input_dim, hparams.num_actions_chunk);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);

    // Layer norm 1 on input
    ggml_tensor *cur = build_norm(ctx0, inp_raw, layer_norm1_w, layer_norm1_b,
                                  NormType::Normal, 1e-5, -1);

    // FC1: (action_dim * input_dim) -> hidden_dim
    cur = build_linear(ctx0, cur, fc1_w, fc1_b, -1);

    // ReLU activation (OFT uses ReLU, not SiLU)
    cur = ggml_relu(ctx0, cur);

    // MLP ResNet blocks
    for (size_t i = 0; i < mlp_resnet_blocks.size(); i++) {
      cur = mlp_resnet_blocks[i].build_graph(ctx0, cur);
    }

    // Layer norm 2
    cur = build_norm(ctx0, cur, layer_norm2_w, layer_norm2_b, NormType::Normal,
                     1e-5, -1);

    // FC2: hidden_dim -> action_dim
    cur = build_linear(ctx0, cur, fc2_w, fc2_b, -1);

    // Output shape: (num_actions_chunk, action_dim) = (8, 7)
    outputs.push_back(cur);
  }
  return outputs;
}

// ==========================================ProprioProjectorModel===========================================
// Projects proprioception state into LLM embedding space
// Architecture: fc1 -> GELU -> fc2
// Input: (proprio_dim,) e.g., 8 for LIBERO
// Output: (llm_dim,) e.g., 4096 for Llama2-7B

std::vector<ggml_tensor *>
ProprioProjectorModel::get_tensors_to_load(ggml_context *ctx) {
  std::vector<ggml_tensor *> tensors_to_load;
  {
    fc1_w = get_tensor(ctx, "fc1.weight", tensors_to_load, true);
    fc1_b = get_tensor(ctx, "fc1.bias", tensors_to_load, false);
    fc2_w = get_tensor(ctx, "fc2.weight", tensors_to_load, true);
    fc2_b = get_tensor(ctx, "fc2.bias", tensors_to_load, false);
  }
  return tensors_to_load;
}

bool ProprioProjectorModel::load_hparams(const ModelLoader &model_loader) {
  model_loader.get_u32("proprio_dim", hparams.proprio_dim);
  model_loader.get_u32("llm_dim", hparams.llm_dim);
  return true;
}

std::vector<ggml_tensor *>
ProprioProjectorModel::build_graph(ggml_context *ctx0) {
  std::vector<ggml_tensor *> outputs;
  {
    // Input: proprio state vector, shape (proprio_dim,)
    ggml_tensor *inp = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, hparams.proprio_dim);
    ggml_set_name(inp, "proprio_input");
    ggml_set_input(inp);

    // fc1: proprio_dim -> llm_dim
    ggml_tensor *cur = build_linear(ctx0, inp, fc1_w, fc1_b, -1);

    // GELU activation
    cur = ggml_gelu(ctx0, cur);

    // fc2: llm_dim -> llm_dim
    cur = build_linear(ctx0, cur, fc2_w, fc2_b, -1);

    // Output: (llm_dim,)
    outputs.push_back(cur);
  }
  return outputs;
}

// ==========================================
// TokenEmbedderModel Implementation
// ==========================================

std::vector<ggml_tensor *>
TokenEmbedderModel::get_tensors_to_load(ggml_context *ctx) {
  std::vector<ggml_tensor *> tensors_to_load;
  // Load token embedding table - this is the main embedding matrix
  token_embd = get_tensor(ctx, "token_embd.weight", tensors_to_load, true);
  if (token_embd) {
    embd_dim_ = token_embd->ne[0];
    vocab_size_ = token_embd->ne[1];
    printf("TokenEmbedder: loaded token_embd with dim=%d, vocab_size=%d\n",
           embd_dim_, vocab_size_);
  }
  return tensors_to_load;
}

bool TokenEmbedderModel::get_embeddings(const std::vector<int32_t> &tokens,
                                        std::vector<float> &out) const {
  if (!token_embd) {
    fprintf(stderr, "TokenEmbedder: token_embd not loaded\n");
    return false;
  }

  out.resize(tokens.size() * embd_dim_);

  // Get the raw data from the tensor
  // token_embd has shape (embd_dim, vocab_size) in row-major order
  // Each row is one token's embedding
  const float *embd_data = (const float *)token_embd->data;

  for (size_t i = 0; i < tokens.size(); i++) {
    int32_t token_id = tokens[i];
    if (token_id < 0 || token_id >= vocab_size_) {
      fprintf(stderr, "TokenEmbedder: token_id %d out of range [0, %d)\n",
              token_id, vocab_size_);
      return false;
    }
    // Copy embedding for this token
    // In ggml, the tensor is stored as (embd_dim, vocab_size)
    // So token i's embedding starts at offset i * embd_dim
    memcpy(out.data() + i * embd_dim_,
           embd_data + token_id * embd_dim_,
           embd_dim_ * sizeof(float));
  }

  return true;
}