#pragma once

#include "infer_session.hpp"
#include "kv_cache_models.hpp"
#include "model_defs.h"
#include "build_graph.h"
#include <cmath>
#include <random>
#include <cstring>
#include <chrono>
#include "timer.hpp"

struct ActionProfStats {
  double kv_rebuild_ms     = 0;  // prefix rebuild + suffix rebuild (ctx reset + graph build)
  double kv_set_input_ms   = 0;  // sched_reset + alloc + prefix_hidden CPU write
  double kv_compute_ms     = 0;  // GPU: 18-layer prefix fwd + ggml_cpy write KV
  double ode_alloc_ms      = 0;  // sched_reset + alloc_graph (step 1 only)
  double ode_set_input_ms  = 0;  // noisy_actions + timestep_emb + positions + attn_mask (total)
  double ode_compute_ms    = 0;  // GPU: 18-layer suffix fwd (total)
  double ode_tensor_get_ms = 0;  // velocity GPU->CPU (total)
  int    ode_steps         = 0;
};

// Per-step metrics collected during each ODE integration (for skip-cache distribution analysis)
// All values are CPU-side, computed from velocity_out and actions (no extra GPU ops)
struct OdeStepRecord {
  int   step       = 0;    // 0-based step index within one ODE run
  float t          = 0.f;  // ODE time t at this step
  float vel_norm   = 0.f;  // mean(|v|) — velocity magnitude
  float vel_L1     = -1.f; // mean(|v - v_prev|)   — velocity step-to-step change; -1 if step 0
  float input_L1   = -1.f; // mean(|x - x_prev|)   — noisy_actions step-to-step change; -1 if step 0
  float diff_err   = -1.f; // mean(|v - (x+diff_prev)|)/vel_norm — EasyCache approx error; -1 if step 0
  float taylor_err = -1.f; // mean(|v - (2v_p-v_pp)|)/vel_norm  — 1st-order extrapolation error; -1 if step<2
  // Per-element change |x_t[i] - x_{t-1}[i]|; empty for step 0.
  // Flat layout: [action_horizon * action_dim], same order as noisy_actions.
  std::vector<float> per_elem_change;
};

// Custom session for PrefixCacheModel that shares weights from main model
class PrefixCacheSession {
public:
  PrefixCacheSession() = default;
  ~PrefixCacheSession() = default;

  // Non-copyable (contains unique_ptr members)
  PrefixCacheSession(const PrefixCacheSession&) = delete;
  PrefixCacheSession& operator=(const PrefixCacheSession&) = delete;

  // Move semantics
  PrefixCacheSession(PrefixCacheSession&&) = default;
  PrefixCacheSession& operator=(PrefixCacheSession&&) = default;

  // Initialize with shared backend from main model
  void init(Pi05ActionExpert *expert, ggml_backend_sched_t sched, int n_prefix, int max_nodes) {
    expert_ = expert;
    n_prefix_ = n_prefix;
    shared_sched_ = sched;
    max_nodes_ = max_nodes;

    // Initialize model wrapper
    model_.set_action_expert(expert);
    model_.set_n_prefix(n_prefix);

    // Allocate compute context (uses shared scheduler)
    alloc_compute_meta();
  }

  // Phase 1: Initialize with GPU-persistent KV cache tensors
  void init_with_gpu_cache(Pi05ActionExpert *expert, ggml_backend_sched_t sched,
                           int n_prefix, int max_nodes, int n_layers,
                           ggml_tensor** k_cache_gpu, ggml_tensor** v_cache_gpu) {
    expert_ = expert;
    n_prefix_ = n_prefix;
    shared_sched_ = sched;
    max_nodes_ = max_nodes;
    n_layers_ = n_layers;
    k_cache_gpu_ = k_cache_gpu;
    v_cache_gpu_ = v_cache_gpu;
    use_gpu_cache_ = true;

    // Initialize model wrapper with GPU cache reference
    model_.set_action_expert(expert);
    model_.set_n_prefix(n_prefix);
    model_.set_gpu_kv_cache(k_cache_gpu, v_cache_gpu, n_layers);

    // Allocate compute context (uses shared scheduler)
    alloc_compute_meta();
  }

  // Rebuild graph for new request - ensures all tensors are in clean unallocated state
  void rebuild_graph() {
    alloc_compute_meta();
  }

  // Phase 1: Run with GPU-persistent KV cache (no CPU copy)
  bool run_gpu(const std::vector<float> &prefix_hidden,
               const std::vector<int> &positions,
               int masked_start = -1, int masked_end = -1) {
    if (!gf_) {
      throw std::runtime_error("PrefixCacheSession graph not allocated");
    }
    if (!use_gpu_cache_) {
      throw std::runtime_error("PrefixCacheSession: GPU cache not initialized");
    }

    // Reset and allocate graph using shared scheduler
    Timer _t; _t.start();
    ggml_backend_sched_reset(shared_sched_);
    ggml_backend_sched_alloc_graph(shared_sched_, gf_);

    // Set inputs
    set_input_f32(gf_, "prefix_hidden", prefix_hidden);
    set_input_i32(gf_, "prefix_positions", positions);

    // Fill prefix attention mask
    std::vector<float> mask_data(n_prefix_ * n_prefix_, 0.0f);
    if (masked_start >= 0 && masked_end > masked_start) {
      const float NEG_INF = -INFINITY;
      for (int i = 0; i < n_prefix_; i++) {
        for (int j = 0; j < n_prefix_; j++) {
          int idx = j + i * n_prefix_;
          if ((i >= masked_start && i < masked_end) ||
              (j >= masked_start && j < masked_end)) {
            mask_data[idx] = NEG_INF;
          }
        }
      }
      printf("PrefixCacheSession: masked tokens [%d, %d) in attention\n", masked_start, masked_end);
    }
    set_input_f32(gf_, "prefix_attn_mask", mask_data);
    last_set_input_ms_ = _t.stop<Timer::ms>();

    // Run graph - KV cache is written directly to GPU tensors via ggml_cpy nodes
    _t.start();
    auto status = ggml_backend_sched_graph_compute(shared_sched_, gf_);
    last_compute_ms_ = _t.stop<Timer::ms>();
    if (status != GGML_STATUS_SUCCESS) {
      fprintf(stderr, "PrefixCacheSession: graph compute failed with error %d\n", status);
      return false;
    }

    // No CPU copy needed - KV cache stays on GPU
    return true;
  }

  // Legacy: Run with CPU vector output (for backward compatibility during transition)
  bool run(const std::vector<float> &prefix_hidden,
           const std::vector<int> &positions,
           ggml_tensor** k_caches_gpu,
           ggml_tensor** v_caches_gpu,
           int masked_start = -1, int masked_end = -1) {
    // This version is called when GPU cache is passed directly
    // Just delegate to run_gpu since the graph already contains ggml_cpy nodes
    return run_gpu(prefix_hidden, positions, masked_start, masked_end);
  }

  double last_set_input_ms() const { return last_set_input_ms_; }
  double last_compute_ms()   const { return last_compute_ms_; }

private:
  void alloc_compute_meta() {
    buf_compute_meta_.resize(max_nodes_ * ggml_tensor_overhead() + ggml_graph_overhead());
    struct ggml_init_params init_params = {
        buf_compute_meta_.size(),
        buf_compute_meta_.data(),
        true  // no_alloc
    };
    ctx_compute_.reset(ggml_init(init_params));
    gf_ = ggml_new_graph_custom(ctx_compute_.get(), max_nodes_, false);

    // Build graph
    std::vector<ggml_tensor *> outputs = model_.build_graph(ctx_compute_.get());
    for (auto *out : outputs) {
      ggml_build_forward_expand(gf_, out);
    }
  }

  double last_set_input_ms_ = 0;  // sched_reset + alloc + set_input calls
  double last_compute_ms_   = 0;  // ggml_backend_sched_graph_compute

  PrefixCacheModel model_;
  Pi05ActionExpert *expert_ = nullptr;
  int n_prefix_ = 256;
  int n_layers_ = 18;
  int max_nodes_ = GGML_DEFAULT_GRAPH_SIZE;

  ggml_backend_sched_t shared_sched_ = nullptr;  // Shared from main model
  ggml_context_ptr ctx_compute_;
  ggml_cgraph *gf_ = nullptr;
  std::vector<uint8_t> buf_compute_meta_;

  // Phase 1: GPU-persistent KV cache
  bool use_gpu_cache_ = false;
  ggml_tensor** k_cache_gpu_ = nullptr;  // Points to Pi05ActionExpertRunner::cached_k_
  ggml_tensor** v_cache_gpu_ = nullptr;  // Points to Pi05ActionExpertRunner::cached_v_
};

// Custom session for SuffixCacheModel that uses cached KV
class SuffixCacheSession {
public:
  SuffixCacheSession() = default;
  ~SuffixCacheSession() = default;

  // Non-copyable (contains unique_ptr members)
  SuffixCacheSession(const SuffixCacheSession&) = delete;
  SuffixCacheSession& operator=(const SuffixCacheSession&) = delete;

  // Move semantics
  SuffixCacheSession(SuffixCacheSession&&) = default;
  SuffixCacheSession& operator=(SuffixCacheSession&&) = default;

  // Initialize with shared backend from main model
  void init(Pi05ActionExpert *expert, ggml_backend_sched_t sched, int n_prefix, int n_suffix, int max_nodes) {
    expert_ = expert;
    n_prefix_ = n_prefix;
    n_suffix_ = n_suffix;
    shared_sched_ = sched;
    max_nodes_ = max_nodes;

    // Initialize model wrapper
    model_.set_action_expert(expert);
    model_.set_dimensions(n_prefix, n_suffix);

    // Allocate compute context (uses shared scheduler)
    alloc_compute_meta();
  }

  // Phase 1: Initialize with GPU-persistent KV cache tensors
  void init_with_gpu_cache(Pi05ActionExpert *expert, ggml_backend_sched_t sched,
                           int n_prefix, int n_suffix, int max_nodes, int n_layers,
                           ggml_tensor** k_cache_gpu, ggml_tensor** v_cache_gpu) {
    expert_ = expert;
    n_prefix_ = n_prefix;
    n_suffix_ = n_suffix;
    shared_sched_ = sched;
    max_nodes_ = max_nodes;
    n_layers_ = n_layers;
    k_cache_gpu_ = k_cache_gpu;
    v_cache_gpu_ = v_cache_gpu;
    use_gpu_cache_ = true;
    graph_allocated_ = false;  // Reset allocation state on reinit

    // Initialize model wrapper with GPU cache reference
    model_.set_action_expert(expert);
    model_.set_dimensions(n_prefix, n_suffix);
    model_.set_gpu_kv_cache(k_cache_gpu, v_cache_gpu, n_layers);

    // Allocate compute context (uses shared scheduler)
    alloc_compute_meta();
  }

  void rebuild_graph(int masked_start = -1, int masked_end = -1) {
    cached_masked_start_ = masked_start;
    cached_masked_end_ = masked_end;
    alloc_compute_meta();
    graph_allocated_ = false;
  }

  // Phase 1: Run with GPU-persistent KV cache (no CPU copy for KV, only for output)
  bool run_gpu(const std::vector<float> &noisy_actions,
               const std::vector<float> &timestep_emb,
               const std::vector<int> &suffix_positions,
               std::vector<float> &velocity_out,
               int masked_start = -1, int masked_end = -1) {
    if (!gf_) {
      throw std::runtime_error("SuffixCacheSession graph not allocated");
    }
    if (!use_gpu_cache_) {
      throw std::runtime_error("SuffixCacheSession: GPU cache not initialized");
    }

    // Only allocate graph on first run to avoid scheduler reset issues
    // Subsequent runs reuse the same allocation, preventing CPU->GPU copy cache invalidation
    Timer _t;
    if (!graph_allocated_) {
      _t.start();
      ggml_backend_sched_reset(shared_sched_);
      ggml_backend_sched_alloc_graph(shared_sched_, gf_);
      last_alloc_ms_ = _t.stop<Timer::ms>();
      graph_allocated_ = true;
    } else {
      last_alloc_ms_ = 0;
    }

    // Set inputs - writes to CPU tensors, scheduler handles CPU->GPU copy during compute
    _t.start();
    set_input_f32(gf_, "noisy_actions", noisy_actions);
    set_input_f32(gf_, "timestep_emb", timestep_emb);
    set_input_i32(gf_, "suffix_positions", suffix_positions);
    set_input_f32(gf_, "suffix_attn_mask", precomputed_attn_mask_);
    last_set_input_ms_ = _t.stop<Timer::ms>();

    // Run graph
    _t.start();
    auto status = ggml_backend_sched_graph_compute(shared_sched_, gf_);
    last_compute_ms_ = _t.stop<Timer::ms>();
    if (status != GGML_STATUS_SUCCESS) {
      fprintf(stderr, "SuffixCacheSession: graph compute failed with error %d\n", status);
      return false;
    }

    // Extract output (this is the only GPU->CPU copy we need)
    _t.start();
    ggml_tensor *out_tensor = ggml_graph_node(gf_, -1);
    velocity_out.resize(ggml_nelements(out_tensor));
    ggml_backend_tensor_get(out_tensor, velocity_out.data(), 0, ggml_nbytes(out_tensor));
    last_tensor_get_ms_ = _t.stop<Timer::ms>();

    return true;
  }

  // Legacy: Run with CPU vector KV cache (for backward compatibility)
  bool run(const std::vector<float> &noisy_actions,
           const std::vector<float> &timestep_emb,
           const std::vector<int> &suffix_positions,
           ggml_tensor** k_caches_gpu,
           ggml_tensor** v_caches_gpu,
           std::vector<float> &velocity_out,
           int masked_start = -1, int masked_end = -1) {
    // GPU cache version - delegate to run_gpu
    return run_gpu(noisy_actions, timestep_emb, suffix_positions, velocity_out, masked_start, masked_end);
  }

  double last_alloc_ms()      const { return last_alloc_ms_; }
  double last_set_input_ms()  const { return last_set_input_ms_; }
  double last_compute_ms()    const { return last_compute_ms_; }
  double last_tensor_get_ms() const { return last_tensor_get_ms_; }

private:
  void alloc_compute_meta() {
    buf_compute_meta_.resize(max_nodes_ * ggml_tensor_overhead() + ggml_graph_overhead());
    struct ggml_init_params init_params = {
        buf_compute_meta_.size(),
        buf_compute_meta_.data(),
        true  // no_alloc
    };
    ctx_compute_.reset(ggml_init(init_params));
    gf_ = ggml_new_graph_custom(ctx_compute_.get(), max_nodes_, false);

    // Build graph
    std::vector<ggml_tensor *> outputs = model_.build_graph(ctx_compute_.get());
    for (auto *out : outputs) {
      ggml_build_forward_expand(gf_, out);
    }

    precomputed_attn_mask_.resize((n_prefix_ + n_suffix_) * n_suffix_);
    fill_suffix_attn_mask(precomputed_attn_mask_.data(), n_prefix_, n_suffix_,
                          cached_masked_start_, cached_masked_end_);
  }

  double last_alloc_ms_      = 0;  // sched_reset + alloc_graph (step 1 only)
  double last_set_input_ms_  = 0;  // noisy_actions + timestep_emb + positions + attn_mask CPU write
  double last_compute_ms_    = 0;  // GPU: 18-layer suffix fwd
  double last_tensor_get_ms_ = 0;  // velocity GPU->CPU

  SuffixCacheModel model_;
  Pi05ActionExpert *expert_ = nullptr;
  int n_prefix_ = 256;
  int n_suffix_ = 50;
  int n_layers_ = 18;
  int max_nodes_ = GGML_DEFAULT_GRAPH_SIZE;

  ggml_backend_sched_t shared_sched_ = nullptr;  // Shared from main model
  ggml_context_ptr ctx_compute_;
  ggml_cgraph *gf_ = nullptr;
  std::vector<uint8_t> buf_compute_meta_;

  // Phase 1: GPU-persistent KV cache
  bool use_gpu_cache_ = false;
  ggml_tensor** k_cache_gpu_ = nullptr;
  ggml_tensor** v_cache_gpu_ = nullptr;

  int cached_masked_start_ = -1;
  int cached_masked_end_ = -1;
  std::vector<float> precomputed_attn_mask_;

  bool graph_allocated_ = false;
};

class Pi05ActionExpertRunner {
public:
  static constexpr int KV_CACHE_MAX_SEQ_LEN = 1024;  // Max prefix + suffix tokens
  static constexpr int KV_CACHE_N_LAYERS = 18;
  static constexpr int KV_CACHE_KV_DIM = 256;  // head_dim * num_kv_heads = 256 * 1

  Pi05ActionExpertRunner() = default;
  Pi05ActionExpertRunner(const std::string &model_path, const ContextParams &params)
      : model_(model_path, params), mask_initialized_(false),
        kv_cache_initialized_(false) {
    // Initialize persistent GPU KV cache
    init_kv_cache_gpu();
  }

  ~Pi05ActionExpertRunner() = default;

  // Non-copyable (contains unique_ptr members)
  Pi05ActionExpertRunner(const Pi05ActionExpertRunner&) = delete;
  Pi05ActionExpertRunner& operator=(const Pi05ActionExpertRunner&) = delete;

  // Move semantics
  Pi05ActionExpertRunner(Pi05ActionExpertRunner&&) = default;
  Pi05ActionExpertRunner& operator=(Pi05ActionExpertRunner&&) = default;

private:
  // Initialize persistent KV cache tensors on GPU
  void init_kv_cache_gpu() {
    // Create a separate ggml context for KV cache (independent of compute context)
    size_t kv_ctx_size = KV_CACHE_N_LAYERS * 2 * ggml_tensor_overhead() + ggml_graph_overhead();
    kv_buf_meta_.resize(kv_ctx_size);
    struct ggml_init_params kv_params = {
        kv_ctx_size,
        kv_buf_meta_.data(),
        true  // no_alloc - we'll allocate buffer separately
    };
    kv_ctx_.reset(ggml_init(kv_params));

    // Create KV cache tensors for each layer
    for (int il = 0; il < KV_CACHE_N_LAYERS; il++) {
      cached_k_[il] = ggml_new_tensor_2d(kv_ctx_.get(), GGML_TYPE_F32,
                                          KV_CACHE_KV_DIM, KV_CACHE_MAX_SEQ_LEN);
      cached_v_[il] = ggml_new_tensor_2d(kv_ctx_.get(), GGML_TYPE_F32,
                                          KV_CACHE_KV_DIM, KV_CACHE_MAX_SEQ_LEN);
      std::string k_name = "kv_cache_k_" + std::to_string(il);
      std::string v_name = "kv_cache_v_" + std::to_string(il);
      ggml_set_name(cached_k_[il], k_name.c_str());
      ggml_set_name(cached_v_[il], v_name.c_str());
    }

    // Allocate buffer on the same backend as the model
    ggml_backend_t backend = model_.get_scheduler() ?
        ggml_backend_sched_get_backend(model_.get_scheduler(), 0) : nullptr;
    if (backend) {
      kv_buffer_.reset(ggml_backend_alloc_ctx_tensors(kv_ctx_.get(), backend));
      if (kv_buffer_) {
        printf("KV cache allocated on GPU: %zu bytes (%d layers x %d x %d x 2)\n",
               ggml_backend_buffer_get_size(kv_buffer_.get()),
               KV_CACHE_N_LAYERS, KV_CACHE_KV_DIM, KV_CACHE_MAX_SEQ_LEN);
      }
    }
  }

public:

  // Single flow matching step: predict velocity field (full joint attention)
  bool run(const std::vector<float> &vlm_hidden,
           const std::vector<float> &noisy_actions,
           const std::vector<float> &timestep_emb,
           std::vector<float> &velocity_out) {

    // Initialize mask and positions on first run
    if (!mask_initialized_) {
      init_mask_and_positions(vlm_hidden.size(), noisy_actions.size());
      mask_initialized_ = true;
    }

    model_.set_input("prefix_hidden", vlm_hidden);
    model_.set_input("noisy_actions", noisy_actions);
    model_.set_input("timestep_emb", timestep_emb);
    model_.set_input("joint_attn_mask", attn_mask_data_);
    model_.set_input("positions", positions_data_);
    return model_.run(velocity_out);
  }

private:
  void init_mask_and_positions(size_t vlm_hidden_size, size_t noisy_actions_size) {
    // Calculate dimensions
    const Pi05ActionExpert &expert = model_.get_model();
    int pali_hidden_size = expert.hparams.pali_hidden_size;
    int action_dim = expert.hparams.action_dim;
    int action_horizon = expert.hparams.action_horizon;

    int n_prefix = vlm_hidden_size / pali_hidden_size;
    int n_suffix = action_horizon;
    int n_total = n_prefix + n_suffix;

    // Fill attention mask
    attn_mask_data_.resize(n_total * n_total);
    fill_joint_attn_mask(attn_mask_data_.data(), n_prefix, n_suffix);

    // Fill positions: [0, 1, 2, ..., n_total-1]
    positions_data_.resize(n_total);
    for (int i = 0; i < n_total; i++) {
      positions_data_[i] = i;
    }
  }

public:

  // Generate sinusoidal timestep embedding (OpenPI formula)
  // Uses min_period=4e-3, max_period=4.0 as per openpi/models/pi0.py
  void fill_sinusoidal_embedding(float timestep, int dim, std::vector<float> &emb) {
    emb.resize(dim);
    int half_dim = dim / 2;
    const float min_period = 4e-3f;
    const float max_period = 4.0f;
    const float pi = 3.14159265358979323846f;

    for (int i = 0; i < half_dim; i++) {
      // fraction goes from 0.0 to 1.0
      float fraction = static_cast<float>(i) / (half_dim - 1);
      // period = min_period * (max_period / min_period) ^ fraction
      float period = min_period * std::pow(max_period / min_period, fraction);
      // angle = timestep / period * 2 * pi
      float angle = timestep / period * 2.0f * pi;
      emb[i] = std::sin(angle);
      emb[half_dim + i] = std::cos(angle);
    }
  }

  // Full flow matching sampling (ODE integration) - without KV cache
  // NOTE: Following openpi convention where t=1 is noise and t=0 is target distribution
  // Integration goes from t=1 to t=0 with NEGATIVE dt
  bool sample_actions(const std::vector<float> &vlm_hidden,
                      int action_dim, int action_horizon,
                      int timestep_dim, int num_steps,
                      std::vector<float> &actions_out) {
    int total_size = action_dim * action_horizon;

    // Initialize with random noise (at t=1)
    actions_out.resize(total_size);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto &v : actions_out) {
      v = dist(rng);
    }

    // Euler integration from t=1 to t=0 (negative dt)
    float dt = -1.0f / num_steps;  // NEGATIVE dt for backward integration
    float t = 1.0f;                 // Start at t=1 (noise)

    while (t >= -dt / 2) {  // Loop until t reaches near 0
      std::vector<float> timestep_emb;
      fill_sinusoidal_embedding(t, timestep_dim, timestep_emb);

      std::vector<float> velocity;
      if (!run(vlm_hidden, actions_out, timestep_emb, velocity)) {
        return false;
      }

      // Update actions: x_{t+dt} = x_t + dt * v(x_t, t)
      // Since dt is negative, this moves from noise (t=1) toward data (t=0)
      for (int i = 0; i < total_size; i++) {
        actions_out[i] += dt * velocity[i];
      }

      t += dt;  // t decreases from 1 toward 0
    }

    return true;
  }

  Pi05ActionExpert &get_model() { return model_.get_model(); }

  // Set prefix length and rebuild graph (must be called before run() or sample_actions())
  // n_prefix: total prefix tokens = vision tokens (256) + text tokens
  void set_prefix_length(int n_prefix) {
    // Only rebuild if the prefix length changed
    if (model_.get_model().hparams.n_prefix_tokens != n_prefix) {
      model_.get_model().hparams.n_prefix_tokens = n_prefix;
      // Reset mask state since dimensions changed
      mask_initialized_ = false;
      // Rebuild the computation graph with new dimensions
      model_.rebuild_graph();
      // Reset KV cache sessions
      prefix_session_initialized_ = false;
      suffix_session_initialized_ = false;
    }
  }

  // Set mask range for prefix tokens (e.g., right_wrist placeholder tokens)
  // This mask is applied in suffix attention to prevent attending to masked tokens
  // masked_start: start index of masked tokens (inclusive)
  // masked_end: end index of masked tokens (exclusive)
  void set_mask_range(int masked_start, int masked_end) {
    cached_masked_start_ = masked_start;
    cached_masked_end_ = masked_end;
  }

  // ============ KV Cache Operations ============

  // Compute KV cache from prefix embeddings (called once per observation)
  // prefix_hidden: [pali_hidden_size * n_prefix] - flattened prefix embeddings
  // masked_start/masked_end: range of tokens to mask in attention (e.g., right_wrist)
  // Returns: true on success
  bool compute_kv_cache(const std::vector<float> &prefix_hidden,
                        int masked_start = -1, int masked_end = -1) {
    // Hash check: skip prefix forward when input is identical to last call
    size_t new_hash = 0;
    const uint32_t *p = reinterpret_cast<const uint32_t *>(prefix_hidden.data());
    for (size_t i = 0; i < prefix_hidden.size(); i++) {
      new_hash ^= p[i] + 0x9e3779b9u + (new_hash << 6) + (new_hash >> 2);
    }
    if (kv_cache_initialized_ && new_hash == cached_prefix_hash_ &&
        masked_start == cached_masked_start_ && masked_end == cached_masked_end_) {
      return true;
    }
    cached_prefix_hash_ = new_hash;

    const Pi05ActionExpert &expert = model_.get_model();
    int pali_hidden_size = expert.hparams.pali_hidden_size;
    int n_prefix = prefix_hidden.size() / pali_hidden_size;
    int n_suffix = expert.hparams.action_horizon;
    int n_layers = expert.hparams.num_hidden_layers;
    int max_nodes = model_.get_max_nodes();

    cached_masked_start_ = masked_start;
    cached_masked_end_ = masked_end;

    // Get shared scheduler from main model
    ggml_backend_sched_t sched = model_.get_scheduler();

    // Check if prefix length changed - need to reinit sessions
    bool prefix_changed = (cached_n_prefix_ > 0 && cached_n_prefix_ != n_prefix);

    // Phase 1: Initialize with GPU-persistent KV cache
    if (!prefix_session_initialized_ || prefix_changed) {
      prefix_session_.init_with_gpu_cache(&model_.get_model(), sched, n_prefix, max_nodes,
                                           n_layers, cached_k_, cached_v_);
      prefix_session_initialized_ = true;
    }

    // Create prefix positions [0, 1, ..., n_prefix-1]
    std::vector<int> prefix_positions(n_prefix);
    for (int i = 0; i < n_prefix; i++) {
      prefix_positions[i] = i;
    }

    // Rebuild prefix graph for each new request to ensure clean tensor state
    Timer _t; _t.start();
    prefix_session_.rebuild_graph();
    prof_stats_.kv_rebuild_ms = _t.stop<Timer::ms>();

    // Run prefix forward pass - KV cache is written directly to GPU tensors
    // NOTE: Do NOT mask tokens in prefix self-attention - all prefix tokens should
    // attend to each other normally. The mask only applies to suffix attention,
    // where action tokens should not attend to masked prefix tokens (e.g., right_wrist).
    if (!prefix_session_.run_gpu(prefix_hidden, prefix_positions)) {
      fprintf(stderr, "compute_kv_cache: prefix forward failed\n");
      return false;
    }
    prof_stats_.kv_set_input_ms = prefix_session_.last_set_input_ms();
    prof_stats_.kv_compute_ms   = prefix_session_.last_compute_ms();

    // Phase 1: Initialize suffix session with GPU-persistent KV cache
    if (!suffix_session_initialized_ || prefix_changed) {
      suffix_session_.init_with_gpu_cache(&model_.get_model(), sched, n_prefix, n_suffix,
                                           max_nodes, n_layers, cached_k_, cached_v_);
      suffix_session_initialized_ = true;
    }

    _t.start();
    suffix_session_.rebuild_graph(cached_masked_start_, cached_masked_end_);
    prof_stats_.kv_rebuild_ms += _t.stop<Timer::ms>();

    // Store dimensions after session init checks
    cached_n_prefix_ = n_prefix;

    // Create suffix positions [n_prefix, n_prefix+1, ..., n_prefix+n_suffix-1]
    suffix_positions_.resize(n_suffix);
    for (int i = 0; i < n_suffix; i++) {
      suffix_positions_[i] = n_prefix + i;
    }

    kv_cache_initialized_ = true;

    // Debug: KV cache is now on GPU, no CPU sum calculation
    printf("KV cache computed (GPU): n_layers=%d, n_prefix=%d, n_suffix=%d\n",
           n_layers, n_prefix, n_suffix);
    if (masked_start >= 0) {
      printf("  Masked token range: [%d, %d)\n", masked_start, masked_end);
    }
    return true;
  }

  // Run single flow matching step using cached KV
  bool run_with_cache(const std::vector<float> &noisy_actions,
                      const std::vector<float> &timestep_emb,
                      std::vector<float> &velocity_out) {
    if (!kv_cache_initialized_) {
      fprintf(stderr, "run_with_cache: KV cache not initialized. Call compute_kv_cache first.\n");
      return false;
    }

    // Phase 1: Use GPU-persistent KV cache (no CPU copy for KV, only for output)
    // Pass mask range so suffix tokens don't attend to masked prefix tokens
    return suffix_session_.run_gpu(noisy_actions, timestep_emb, suffix_positions_,
                                    velocity_out,
                                    cached_masked_start_, cached_masked_end_);
  }

  // Full flow matching sampling using KV cache (optimized version)
  // prefix_hidden: [pali_hidden_size * n_prefix] - prefix embeddings
  // masked_start/masked_end: range of prefix tokens to mask (e.g., right_wrist placeholder)
  bool sample_actions_with_cache(const std::vector<float> &prefix_hidden,
                                  int action_dim, int action_horizon,
                                  int timestep_dim, int num_steps,
                                  std::vector<float> &actions_out,
                                  int masked_start = -1, int masked_end = -1) {
    // Compute KV cache once (with masking for placeholder tokens)
    if (!compute_kv_cache(prefix_hidden, masked_start, masked_end)) {
      return false;
    }

    int total_size = action_dim * action_horizon;

    // Initialize with random noise (at t=1)
    actions_out.resize(total_size);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto &v : actions_out) {
      v = dist(rng);
    }

    // Euler integration from t=1 to t=0
    float dt = -1.0f / num_steps;
    float t = 1.0f;

    prof_stats_.ode_alloc_ms = prof_stats_.ode_set_input_ms = 0;
    prof_stats_.ode_compute_ms = prof_stats_.ode_tensor_get_ms = 0;
    prof_stats_.ode_steps = 0;

    last_ode_step_records_.clear();
    std::vector<float> _prev_x, _prev_v, _prev_diff, _prev_prev_v;
    int _step = 0;

    while (t >= -dt / 2) {
      std::vector<float> timestep_emb;
      fill_sinusoidal_embedding(t, timestep_dim, timestep_emb);

      const std::vector<float> x_t = actions_out;  // snapshot x before Euler update
      std::vector<float> velocity;
      if (!run_with_cache(actions_out, timestep_emb, velocity)) {
        return false;
      }

      prof_stats_.ode_alloc_ms      += suffix_session_.last_alloc_ms();
      prof_stats_.ode_set_input_ms  += suffix_session_.last_set_input_ms();
      prof_stats_.ode_compute_ms    += suffix_session_.last_compute_ms();
      prof_stats_.ode_tensor_get_ms += suffix_session_.last_tensor_get_ms();
      prof_stats_.ode_steps++;

      // Compute per-step distribution metrics (CPU-only, O(action_dim*horizon) ~350 floats)
      {
        OdeStepRecord rec;
        rec.step = _step;
        rec.t    = t;
        const float n = static_cast<float>(velocity.size());
        float vn = 0.f;
        for (float f : velocity) vn += std::fabs(f);
        rec.vel_norm = n > 0.f ? vn / n : 0.f;

        if (_step > 0) {
          float il = 0.f, vl = 0.f, de = 0.f;
          rec.per_elem_change.resize(velocity.size());
          for (size_t i = 0; i < velocity.size(); ++i) {
            rec.per_elem_change[i] = std::fabs(x_t[i] - _prev_x[i]);
            il += rec.per_elem_change[i];
            vl += std::fabs(velocity[i] - _prev_v[i]);
            de += std::fabs(velocity[i] - (x_t[i] + _prev_diff[i]));
          }
          rec.input_L1 = n > 0.f ? il / n : 0.f;
          rec.vel_L1   = n > 0.f ? vl / n : 0.f;
          rec.diff_err = rec.vel_norm > 1e-8f ? (de / n) / rec.vel_norm : -1.f;

          if (_step > 1) {
            float te = 0.f;
            for (size_t i = 0; i < velocity.size(); ++i) {
              float vp = 2.f * _prev_v[i] - _prev_prev_v[i];
              te += std::fabs(velocity[i] - vp);
            }
            rec.taylor_err = rec.vel_norm > 1e-8f ? (te / n) / rec.vel_norm : -1.f;
          }
        }
        last_ode_step_records_.push_back(rec);

        // Update diff cache: diff = v_t - x_t (used as predictor for next step)
        _prev_diff.resize(velocity.size());
        for (size_t i = 0; i < velocity.size(); ++i)
          _prev_diff[i] = velocity[i] - x_t[i];
        _prev_prev_v = _prev_v;
        _prev_v      = velocity;
        _prev_x      = x_t;
        _step++;
      }

      for (int i = 0; i < total_size; i++) {
        actions_out[i] += dt * velocity[i];
      }

      t += dt;
    }

    return true;
  }

  // Clear KV cache (call when observation changes)
  // Phase 1: KV cache tensors remain allocated on GPU, we just reset the state
  void clear_kv_cache() {
    kv_cache_initialized_ = false;
    cached_n_prefix_ = 0;
    cached_prefix_hash_ = 0;
    prefix_session_initialized_ = false;
    suffix_session_initialized_ = false;
  }

  bool has_kv_cache() const { return kv_cache_initialized_; }

  const ActionProfStats& get_prof_stats() const { return prof_stats_; }

  const std::vector<OdeStepRecord>& get_ode_step_records() const { return last_ode_step_records_; }

private:
  InferenceSession<Pi05ActionExpert> model_;
  bool mask_initialized_;
  std::vector<float> attn_mask_data_;
  std::vector<int> positions_data_;

  // KV Cache state
  PrefixCacheSession prefix_session_;
  SuffixCacheSession suffix_session_;
  bool prefix_session_initialized_ = false;
  bool suffix_session_initialized_ = false;

  // GPU-persistent KV cache tensors (Phase 1)
  ggml_tensor* cached_k_[KV_CACHE_N_LAYERS] = {nullptr};  // [kv_dim, max_seq_len] per layer
  ggml_tensor* cached_v_[KV_CACHE_N_LAYERS] = {nullptr};  // [kv_dim, max_seq_len] per layer
  ggml_context_ptr kv_ctx_;                               // Independent context for KV cache
  ggml_backend_buffer_ptr kv_buffer_;                     // GPU buffer for KV cache
  std::vector<uint8_t> kv_buf_meta_;                      // Metadata buffer for kv_ctx_

  std::vector<int> suffix_positions_;
  int cached_n_prefix_ = 0;
  int cached_masked_start_ = -1;
  int cached_masked_end_ = -1;
  size_t cached_prefix_hash_ = 0;
  bool kv_cache_initialized_;
  ActionProfStats prof_stats_;
  std::vector<OdeStepRecord> last_ode_step_records_;  // populated each sample_actions_with_cache call
};
