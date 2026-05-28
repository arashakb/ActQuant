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

  // Non-copyable (contains unique_ptr members)
  InferenceSession(const InferenceSession&) = delete;
  InferenceSession& operator=(const InferenceSession&) = delete;

  // Move semantics
  InferenceSession(InferenceSession&&) = default;
  InferenceSession& operator=(InferenceSession&&) = default;

  void alloc_graph() {
    ggml_backend_sched_reset(ctx_manager_.sched_.get());
    ggml_cgraph *gf = ctx_manager_.gf_;
    std::vector<ggml_tensor *> outputs =
        model_.build_graph(ctx_manager_.ctx_compute_.get());
    for (size_t i = 0; i < outputs.size(); ++i) {
      ggml_build_forward_expand(gf, outputs[i]);
    }

    ggml_backend_sched_alloc_graph(ctx_manager_.sched_.get(), gf);
  }

  void set_input(const std::string &input_name,
                 const std::vector<float> &input_data) {
    set_input_f32(ctx_manager_.gf_, input_name.c_str(), input_data);
  }
  void set_input(const std::string &input_name,
                 const std::vector<int> &input_data) {
    set_input_i32(ctx_manager_.gf_, input_name.c_str(), input_data);
  }

  void set_input_as_f16(const std::string &input_name,
                        const std::vector<float> &input_data) {
    set_input_f16(ctx_manager_.gf_, input_name.c_str(), input_data);
  }

  ModelType &get_model() { return model_; }

  // Get the scheduler (for sharing with cache sessions)
  ggml_backend_sched_t get_scheduler() { return ctx_manager_.sched_.get(); }

  // Get max nodes setting
  int get_max_nodes() const { return ctx_manager_.max_nodes_; }

  // Rebuild the computation graph (call after changing model parameters like n_prefix_tokens)
  void rebuild_graph() {
    // Reset compute context and graph
    ctx_manager_.ctx_compute_.reset();
    ctx_manager_.gf_ = nullptr;

    // Reallocate compute metadata and rebuild graph
    alloc_compute_meta();
  }

  bool run(std::vector<float> &out) {
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

    if (tmp_out->type == GGML_TYPE_F16) {
      get_output_f16_to_float(tmp_out, out);
    } else {
      ggml_backend_tensor_get(tmp_out, out.data(), 0, ggml_nbytes(tmp_out));
    }

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
