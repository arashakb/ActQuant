#include "ctx_manager.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>

ContextManager::ContextManager(const std::string &dev_name, int n_threads,
                               int max_nodes)
    : max_nodes_(max_nodes) {
  debug_graph_ = std::getenv("MTMD_DEBUG_GRAPH") != nullptr;

  // load dynamic backends
  ggml_backend_load_all();

  // 1. create backend
  if (!create_backend(dev_name, n_threads)) {
    throw std::runtime_error("failed to initialize backend");
  }
  // 2. create scheduler
  if (!create_scheduler()) {
    throw std::runtime_error("failed to create scheduler");
  }
}

ggml_backend_t
ContextManager::try_init_backend(enum ggml_backend_dev_type type) {
  return ggml_backend_init_by_type(type, nullptr);
}
bool ContextManager::create_backend(const std::string &dev_name,
                                    int n_threads) {
  // create cpu backend
  backend_cpu_.reset(
      ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
  if (!backend_cpu_.get()) {
    throw std::runtime_error("failed to initialize CPU backend");
  }

  // set device
  if (!dev_name.empty()) {
    // get dev info
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_name.c_str());
    if (dev) {
      backend_.reset(ggml_backend_dev_init(dev, nullptr));
      if (!backend_) {
        // throw std::runtime_error("failed to initialize device backend");
        fprintf(stderr, "%s: failed to create backend for device %s\n",
                __func__, dev_name.c_str());
        return false;
      }
    } else {
      fprintf(stderr, "%s: ERROR: backend %s not found, available:\n", __func__,
              dev_name.c_str());
      for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev_i = ggml_backend_dev_get(i);
        fprintf(stderr, "  - %s (%s)\n", ggml_backend_dev_name(dev_i),
                ggml_backend_dev_description(dev_i));
      }
    }
  }

  // If the user-specified device does not exist, prioritize finding GPU devices, 
  // and if not found, look for IGPU devices
  if (!backend_) {
    fprintf(stderr, "%s: no backend specified, trying GPU backends\n",
            __func__);
    auto backend =
        ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    backend = backend ? backend
                      : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU,
                                                  nullptr);
    backend_.reset(backend);
  }
  // If no GPU is found, fall back to CPU
  if (!backend_) {
    fprintf(stderr, "%s: no GPU backend found, falling back to CPU backend\n",
            __func__);
    backend_.reset(
        ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
  }

  if (!backend_) {
    return false;
  }

  fprintf(stderr, "%s: using %s backend\n", __func__,
          ggml_backend_name(backend_.get()));

  // set n_threads
  set_backend_threads(backend_.get(), n_threads);
  // ggml_backend_dev_t dev = ggml_backend_get_device(backend_.get());
  // ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
  // if (reg) {
  //   auto ggml_backend_set_n_threads_fn =
  //       (ggml_backend_set_n_threads_t)ggml_backend_reg_get_proc_address(
  //           reg, "ggml_backend_set_n_threads");
  //   if (ggml_backend_set_n_threads_fn) {
  //     ggml_backend_set_n_threads_fn(backend_.get(), n_threads);
  //   }
  // }
  // if(ggml_backend_is_cpu(backend_.get())){
  //     ggml_backend_cpu_set_n_threads(backend_.get(), n_threads);
  // }

  backend_ptrs_.clear();
  backend_buft_.clear();
  if (backend_) {
    backend_ptrs_.push_back(backend_.get());
    backend_buft_.push_back(
        ggml_backend_get_default_buffer_type(backend_.get()));
  }
  if (!ggml_backend_is_cpu(backend_.get())) {
    backend_ptrs_.push_back(backend_cpu_.get());
    backend_buft_.push_back(
        ggml_backend_get_default_buffer_type(backend_cpu_.get()));
  }

  return true;
}

bool ContextManager::create_scheduler() {
  sched_.reset(
      ggml_backend_sched_new(backend_ptrs_.data(), backend_buft_.data(),
                             backend_ptrs_.size(), max_nodes_, false, true));
  if (!sched_) {
    return false;
  }
  fprintf(stderr, "%s: using %s as primary backend\n", __func__,
          ggml_backend_name(backend_ptrs_[0]));
  return true;
}

bool ContextManager::alloc_tensors() {
  if (!ctx_data_) {
    fprintf(stderr,
            "%s: ctx_data_ is null, please initialize ggml_context first",
            __func__);
    return false;
  }
  buffer_.reset(
      ggml_backend_alloc_ctx_tensors(ctx_data_.get(), backend_.get()));
  if (!buffer_.get()) {
    printf("failed to alloc buffer\n");
    return false;
  }
  ggml_backend_buffer_set_usage(buffer_.get(),
                                GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  return true;
}
