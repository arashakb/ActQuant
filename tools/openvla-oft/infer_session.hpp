#pragma once

#include "ctx_manager.h"
#include "model_defs.h"
#include "model_loader.h"
#include "utils.h"
#include <memory>

template <typename ModelType = BaseModel> class InferenceSession {
public:
  InferenceSession() = default;
  InferenceSession(const std::string &model_path, const ContextParams &params)
      : ctx_manager_(params) {
    ModelLoader model_loader(model_path);
    model_.load_hparams(model_loader);
    model_.load_tensors(model_loader, ctx_manager_);
    alloc_compute_meta();
  }

  void alloc_graph() {
    ggml_backend_sched_reset(ctx_manager_.sched_.get());
    ggml_cgraph *gf = ctx_manager_.gf_;
    std::vector<ggml_tensor *> outputs =
        model_.build_graph(ctx_manager_.ctx_compute_.get());
    for (size_t i = 0; i < outputs.size(); ++i) {
      ggml_build_forward_expand(gf, outputs[i]);
    }

    // ggml_backend_sched_reserve(ctx_manager_.sched_.get(), gf);
    ggml_backend_sched_alloc_graph(ctx_manager_.sched_.get(), gf);

    // for (size_t i = 0; i < ctx_manager_.backend_ptrs_.size(); ++i) {
    //   ggml_backend_t backend = ctx_manager_.backend_ptrs_[i];
    //   ggml_backend_buffer_type_t buft = ctx_manager_.backend_buft_[i];
    //   size_t size =
    //       ggml_backend_sched_get_buffer_size(ctx_manager_.sched_.get(),
    //       backend);
    //   if (size > 1) {
    //     printf("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
    //            ggml_backend_buft_name(buft), size / 1024.0 / 1024.0);
    //   }
    // }
  }

  void set_input(const std::string &input_name,
                 const std::vector<float> &input_data) {
    set_input_f32(ctx_manager_.gf_, input_name.c_str(), input_data);
  }
  void set_input(const std::string &input_name,
                 const std::vector<int> &input_data) {
    set_input_i32(ctx_manager_.gf_, input_name.c_str(), input_data);
  }

  ModelType &get_model() { return model_; }

  bool run(std::vector<float> &out) {
    ctx_manager_.debug_print_tensors_.clear();
    if (!ctx_manager_.gf_) {
      alloc_graph();
    }
    ggml_cgraph *gf = ctx_manager_.gf_;
    auto status =
        ggml_backend_sched_graph_compute(ctx_manager_.sched_.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
      printf("%s: ggml_backend_sched_graph_compute failed with error %d\n",
             __func__, status);
      return false;
    }
    ggml_tensor *tmp_out = ggml_graph_node(gf, -1);
    out.resize(ggml_nelements(tmp_out));
    // copy the embeddings to the location passed by the user
    ggml_backend_tensor_get(tmp_out, out.data(), 0, ggml_nbytes(tmp_out));

    return true;
  }

private:
  ModelType model_;
  ContextManager ctx_manager_;

  void alloc_compute_meta() {
    std::vector<uint8_t> &buf_compute_meta = ctx_manager_.buf_compute_meta_;

    buf_compute_meta.resize(ctx_manager_.max_nodes_ * ggml_tensor_overhead() +
                            ggml_graph_overhead());
    struct ggml_init_params params = {
        /*.mem_size   =*/buf_compute_meta.size(),
        /*.mem_buffer =*/buf_compute_meta.data(),
        /*.no_alloc   =*/true,
    };
    ctx_manager_.ctx_compute_.reset(ggml_init(params));
    ctx_manager_.gf_ = ggml_new_graph_custom(ctx_manager_.ctx_compute_.get(),
                                             ctx_manager_.max_nodes_, false);

    alloc_graph();
  }
};