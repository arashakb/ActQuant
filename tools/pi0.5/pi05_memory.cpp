#include "pi05_memory.h"
#include <cstdio>
#include <stdexcept>

bool Pi05Memory::init(int n_layers, int kv_dim, int max_seq_len, ggml_backend_t backend) {
    if (initialized_) {
        fprintf(stderr, "Pi05Memory::init: already initialized\n");
        return false;
    }

    if (!backend) {
        fprintf(stderr, "Pi05Memory::init: backend is null\n");
        return false;
    }

    n_layers_ = n_layers;
    kv_dim_ = kv_dim;
    max_seq_len_ = max_seq_len;

    // Create independent ggml context for KV cache tensors
    size_t ctx_size = n_layers * 2 * ggml_tensor_overhead() + ggml_graph_overhead();
    kv_buf_meta_.resize(ctx_size);

    struct ggml_init_params ctx_params = {
        ctx_size,
        kv_buf_meta_.data(),
        true  // no_alloc - we'll allocate buffer separately
    };
    kv_ctx_.reset(ggml_init(ctx_params));
    if (!kv_ctx_) {
        fprintf(stderr, "Pi05Memory::init: failed to create ggml context\n");
        return false;
    }

    // Allocate K and V cache tensors for each layer
    k_cache_.resize(n_layers);
    v_cache_.resize(n_layers);

    for (int il = 0; il < n_layers; il++) {
        k_cache_[il] = ggml_new_tensor_2d(kv_ctx_.get(), GGML_TYPE_F32, kv_dim, max_seq_len);
        v_cache_[il] = ggml_new_tensor_2d(kv_ctx_.get(), GGML_TYPE_F32, kv_dim, max_seq_len);

        if (!k_cache_[il] || !v_cache_[il]) {
            fprintf(stderr, "Pi05Memory::init: failed to create tensor for layer %d\n", il);
            return false;
        }

        // Set names for debugging
        char name_k[32], name_v[32];
        snprintf(name_k, sizeof(name_k), "mem_k_%d", il);
        snprintf(name_v, sizeof(name_v), "mem_v_%d", il);
        ggml_set_name(k_cache_[il], name_k);
        ggml_set_name(v_cache_[il], name_v);
    }

    // Allocate GPU buffer for all KV cache tensors
    kv_buffer_.reset(ggml_backend_alloc_ctx_tensors(kv_ctx_.get(), backend));
    if (!kv_buffer_) {
        fprintf(stderr, "Pi05Memory::init: failed to allocate GPU buffer\n");
        return false;
    }

    initialized_ = true;

    printf("Pi05Memory initialized: n_layers=%d, kv_dim=%d, max_seq_len=%d, buffer_size=%zu bytes\n",
           n_layers, kv_dim, max_seq_len, ggml_backend_buffer_get_size(kv_buffer_.get()));

    return true;
}

ggml_tensor* Pi05Memory::get_k(ggml_context* ctx, int il, int seq_len) const {
    if (!initialized_ || il < 0 || il >= n_layers_) {
        return nullptr;
    }

    if (seq_len > max_seq_len_) {
        fprintf(stderr, "Pi05Memory::get_k: seq_len %d exceeds max_seq_len %d\n", seq_len, max_seq_len_);
        return nullptr;
    }

    // Create a view of the first seq_len positions
    return ggml_view_2d(ctx, k_cache_[il],
                        kv_dim_, seq_len,
                        k_cache_[il]->nb[1], 0);
}

ggml_tensor* Pi05Memory::get_v(ggml_context* ctx, int il, int seq_len) const {
    if (!initialized_ || il < 0 || il >= n_layers_) {
        return nullptr;
    }

    if (seq_len > max_seq_len_) {
        fprintf(stderr, "Pi05Memory::get_v: seq_len %d exceeds max_seq_len %d\n", seq_len, max_seq_len_);
        return nullptr;
    }

    // Create a view of the first seq_len positions
    return ggml_view_2d(ctx, v_cache_[il],
                        kv_dim_, seq_len,
                        v_cache_[il]->nb[1], 0);
}

ggml_tensor* Pi05Memory::cpy_k(ggml_context* ctx, ggml_tensor* k_cur, int offset, int il) const {
    if (!initialized_ || il < 0 || il >= n_layers_) {
        return nullptr;
    }

    int n_tokens = k_cur->ne[1];
    if (offset + n_tokens > max_seq_len_) {
        fprintf(stderr, "Pi05Memory::cpy_k: offset %d + n_tokens %d exceeds max_seq_len %d\n",
                offset, n_tokens, max_seq_len_);
        return nullptr;
    }

    // Create a view of the destination region
    size_t offset_bytes = offset * k_cache_[il]->nb[1];
    ggml_tensor* k_dst = ggml_view_2d(ctx, k_cache_[il],
                                       kv_dim_, n_tokens,
                                       k_cache_[il]->nb[1], offset_bytes);

    // Create copy node
    return ggml_cpy(ctx, k_cur, k_dst);
}

ggml_tensor* Pi05Memory::cpy_v(ggml_context* ctx, ggml_tensor* v_cur, int offset, int il) const {
    if (!initialized_ || il < 0 || il >= n_layers_) {
        return nullptr;
    }

    int n_tokens = v_cur->ne[1];
    if (offset + n_tokens > max_seq_len_) {
        fprintf(stderr, "Pi05Memory::cpy_v: offset %d + n_tokens %d exceeds max_seq_len %d\n",
                offset, n_tokens, max_seq_len_);
        return nullptr;
    }

    // Create a view of the destination region
    size_t offset_bytes = offset * v_cache_[il]->nb[1];
    ggml_tensor* v_dst = ggml_view_2d(ctx, v_cache_[il],
                                       kv_dim_, n_tokens,
                                       v_cache_[il]->nb[1], offset_bytes);

    // Create copy node
    return ggml_cpy(ctx, v_cur, v_dst);
}
