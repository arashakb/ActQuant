#pragma once

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "utils.h"
#include <string>

class ContextManager {
public:
  ContextManager() : ContextManager("CPU", 1, GGML_DEFAULT_GRAPH_SIZE) {}
  ContextManager(const std::string &dev_name, int n_threads = 1,
                 int max_nodes = GGML_DEFAULT_GRAPH_SIZE);
  ContextManager(const ContextParams &params)
      : ContextManager(params.device_name, params.n_threads, params.max_nodes) {
  }
  static ggml_backend_t try_init_backend(enum ggml_backend_dev_type type);

  // protected:
  bool create_backend(const std::string &dev_name, int n_threads);
  bool create_scheduler();
  bool alloc_tensors();

  ggml_context_ptr ctx_data_ = nullptr;
  ggml_backend_buffer_ptr buffer_ = nullptr;

  ggml_context_ptr ctx_compute_ = nullptr;

  std::vector<uint8_t> buf_compute_meta_;

  ggml_backend_ptr backend_cpu_ = nullptr;
  ggml_backend_ptr backend_ = nullptr;
  std::vector<ggml_backend_t> backend_ptrs_;
  std::vector<ggml_backend_buffer_type_t> backend_buft_;

  ggml_backend_sched_ptr sched_ = nullptr;

  ggml_cgraph *gf_ = nullptr;
  int max_nodes_ = GGML_DEFAULT_GRAPH_SIZE; // 8192

  // gguf_context_ptr ctx_gguf_ = nullptr;

  // for debugging
  bool debug_graph_ = false;
  std::vector<ggml_tensor *> debug_print_tensors_;
};
