#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include <vector>
#include <cstdint>

/**
 * Pi05Memory - GPU-persistent KV cache management
 *
 * This class manages the Key/Value cache for the Pi0.5 action expert model.
 * The KV cache is allocated on GPU and persists across multiple inference calls.
 *
 * Key features:
 * - Independent GPU buffer (not managed by scheduler)
 * - Provides K/V views via get_k/get_v
 * - Provides K/V write nodes via cpy_k/cpy_v
 * - Manages n_prefix state
 */
class Pi05Memory {
public:
    static constexpr int MAX_SEQ_LEN = 1024;  // Max prefix + suffix tokens
    static constexpr int DEFAULT_N_LAYERS = 18;
    static constexpr int DEFAULT_KV_DIM = 256;  // head_dim * num_kv_heads

    Pi05Memory() = default;
    ~Pi05Memory() = default;

    // Non-copyable
    Pi05Memory(const Pi05Memory&) = delete;
    Pi05Memory& operator=(const Pi05Memory&) = delete;

    // Movable
    Pi05Memory(Pi05Memory&&) = default;
    Pi05Memory& operator=(Pi05Memory&&) = default;

    /**
     * Initialize KV cache on the specified backend
     * @param n_layers Number of transformer layers
     * @param kv_dim KV dimension (head_dim * num_kv_heads)
     * @param max_seq_len Maximum sequence length to support
     * @param backend Backend to allocate tensors on (typically GPU)
     * @return true on success
     */
    bool init(int n_layers, int kv_dim, int max_seq_len, ggml_backend_t backend);

    /**
     * Set the current prefix length
     * @param n_prefix Number of prefix tokens
     */
    void set_prefix_len(int n_prefix) { n_prefix_ = n_prefix; }

    /**
     * Get the current prefix length
     */
    int get_prefix_len() const { return n_prefix_; }

    /**
     * Get K cache view for a specific layer
     * @param ctx ggml context for creating view tensor
     * @param il Layer index
     * @param seq_len Sequence length for the view
     * @return View tensor of shape [kv_dim, seq_len]
     */
    ggml_tensor* get_k(ggml_context* ctx, int il, int seq_len) const;

    /**
     * Get V cache view for a specific layer
     * @param ctx ggml context for creating view tensor
     * @param il Layer index
     * @param seq_len Sequence length for the view
     * @return View tensor of shape [kv_dim, seq_len]
     */
    ggml_tensor* get_v(ggml_context* ctx, int il, int seq_len) const;

    /**
     * Create a copy node to write K values to cache
     * @param ctx ggml context for creating nodes
     * @param k_cur Source K tensor to copy [kv_dim, n_tokens]
     * @param offset Starting position in cache
     * @param il Layer index
     * @return ggml_cpy node (execute to perform copy)
     */
    ggml_tensor* cpy_k(ggml_context* ctx, ggml_tensor* k_cur, int offset, int il) const;

    /**
     * Create a copy node to write V values to cache
     * @param ctx ggml context for creating nodes
     * @param v_cur Source V tensor to copy [kv_dim, n_tokens]
     * @param offset Starting position in cache
     * @param il Layer index
     * @return ggml_cpy node (execute to perform copy)
     */
    ggml_tensor* cpy_v(ggml_context* ctx, ggml_tensor* v_cur, int offset, int il) const;

    /**
     * Clear the memory state (resets n_prefix to 0)
     * Note: GPU buffers remain allocated
     */
    void clear() { n_prefix_ = 0; }

    /**
     * Check if memory is initialized
     */
    bool is_initialized() const { return initialized_; }

    /**
     * Get raw K cache tensor pointer (for direct access)
     * @param il Layer index
     * @return K cache tensor for layer il
     */
    ggml_tensor* get_k_cache(int il) const {
        return (il >= 0 && il < n_layers_) ? k_cache_[il] : nullptr;
    }

    /**
     * Get raw V cache tensor pointer (for direct access)
     * @param il Layer index
     * @return V cache tensor for layer il
     */
    ggml_tensor* get_v_cache(int il) const {
        return (il >= 0 && il < n_layers_) ? v_cache_[il] : nullptr;
    }

    /**
     * Get number of layers
     */
    int get_n_layers() const { return n_layers_; }

    /**
     * Get KV dimension
     */
    int get_kv_dim() const { return kv_dim_; }

    /**
     * Get max sequence length
     */
    int get_max_seq_len() const { return max_seq_len_; }

private:
    bool initialized_ = false;
    int n_layers_ = 0;
    int kv_dim_ = 0;
    int max_seq_len_ = 0;
    int n_prefix_ = 0;

    // GPU-persistent KV cache tensors
    std::vector<ggml_tensor*> k_cache_;  // [n_layers], each [kv_dim, max_seq_len]
    std::vector<ggml_tensor*> v_cache_;  // [n_layers], each [kv_dim, max_seq_len]

    // Independent context and buffer for KV cache
    ggml_context_ptr kv_ctx_;
    ggml_backend_buffer_ptr kv_buffer_;
    std::vector<uint8_t> kv_buf_meta_;
};
