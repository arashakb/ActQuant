#pragma once

/**
 * Text Embedding Module for Pi0.5
 *
 * Replaces llama.cpp dependency with direct embedding lookup from unified GGUF.
 * Uses SentencePiece for tokenization (same as OpenPI).
 */

#include "sp_tokenizer.hpp"
#include "model_loader.h"
#include "ctx_manager.h"
#include "ggml.h"
#include "ggml-quants.h"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

struct TextEmbedConfig {
    int vocab_size = 257152;
    int hidden_size = 2048;
};

class Pi05TextEmbed {
public:
    Pi05TextEmbed() = default;

    /**
     * Initialize text embedding from unified GGUF file
     * @param model_path Path to unified GGUF containing embed.weight
     * @param tokenizer_path Path to tokenizer.model (SentencePiece)
     */
    bool init(const std::string &model_path, const std::string &tokenizer_path) {
        // Load tokenizer
        tokenizer_ = std::make_unique<SpTokenizer>();
        if (!tokenizer_->init(tokenizer_path)) {
            fprintf(stderr, "TextEmbed: Failed to load tokenizer from %s\n", tokenizer_path.c_str());
            return false;
        }
        printf("TextEmbed: Loaded SentencePiece tokenizer\n");

        // Load embedding table from GGUF
        if (!load_embedding(model_path)) {
            fprintf(stderr, "TextEmbed: Failed to load embedding table\n");
            return false;
        }

        initialized_ = true;
        return true;
    }

    /**
     * Tokenize text string
     * @param text Input text
     * @param add_special Add BOS/EOS tokens
     * @return Token IDs
     */
    std::vector<int32_t> tokenize(const std::string &text, bool add_special = false) {
        if (!tokenizer_ || !tokenizer_->is_initialized()) {
            throw std::runtime_error("TextEmbed: Tokenizer not initialized");
        }
        return tokenizer_->encode(text, add_special);
    }

    /**
     * Get text embeddings for token IDs
     * Matches OpenPI's Embedder.encode(): x = embedding[tokens] * sqrt(hidden_size)
     *
     * @param tokens Token IDs
     * @return Embeddings [n_tokens * hidden_size]
     */
    std::vector<float> get_text_embeddings(const std::vector<int32_t> &tokens) {
        if (!initialized_) {
            throw std::runtime_error("TextEmbed: Not initialized");
        }

        if (tokens.empty()) {
            return {};
        }

        int n_tokens = tokens.size();
        std::vector<float> embeddings(n_tokens * config_.hidden_size);
        float scale = std::sqrt(static_cast<float>(config_.hidden_size));

        // Embedding lookup with scaling
        if (embed_type_ == GGML_TYPE_F32) {
            const float *embd_data = reinterpret_cast<const float *>(embed_data_);
            for (int i = 0; i < n_tokens; i++) {
                int token_id = tokens[i];
                if (token_id < 0 || token_id >= config_.vocab_size) {
                    throw std::runtime_error("TextEmbed: Token ID out of range: " + std::to_string(token_id));
                }

                const float *src = embd_data + token_id * config_.hidden_size;
                float *dst = embeddings.data() + i * config_.hidden_size;
                for (int j = 0; j < config_.hidden_size; j++) {
                    dst[j] = src[j] * scale;
                }
            }
        } else if (embed_type_ == GGML_TYPE_F16) {
            const ggml_fp16_t *embd_data = reinterpret_cast<const ggml_fp16_t *>(embed_data_);
            for (int i = 0; i < n_tokens; i++) {
                int token_id = tokens[i];
                if (token_id < 0 || token_id >= config_.vocab_size) {
                    throw std::runtime_error("TextEmbed: Token ID out of range: " + std::to_string(token_id));
                }

                const ggml_fp16_t *src = embd_data + token_id * config_.hidden_size;
                float *dst = embeddings.data() + i * config_.hidden_size;
                for (int j = 0; j < config_.hidden_size; j++) {
                    dst[j] = ggml_fp16_to_fp32(src[j]) * scale;
                }
            }
        } else if (embed_type_ == GGML_TYPE_Q4_0) {
            // Q4_0: each block has QK4_0=32 elements
            const int n_blocks_per_row = config_.hidden_size / QK4_0;
            const size_t row_size = n_blocks_per_row * sizeof(block_q4_0);

            for (int i = 0; i < n_tokens; i++) {
                int token_id = tokens[i];
                if (token_id < 0 || token_id >= config_.vocab_size) {
                    throw std::runtime_error("TextEmbed: Token ID out of range: " + std::to_string(token_id));
                }

                const block_q4_0 *src = reinterpret_cast<const block_q4_0 *>(
                    static_cast<const uint8_t *>(embed_data_) + token_id * row_size);
                float *dst = embeddings.data() + i * config_.hidden_size;

                // Dequantize entire row
                dequantize_row_q4_0(src, dst, config_.hidden_size);

                // Apply scale
                for (int j = 0; j < config_.hidden_size; j++) {
                    dst[j] *= scale;
                }
            }
        } else if (embed_type_ == GGML_TYPE_Q8_0) {
            // Q8_0: each block has QK8_0=32 elements
            const int n_blocks_per_row = config_.hidden_size / QK8_0;
            const size_t row_size = n_blocks_per_row * sizeof(block_q8_0);

            for (int i = 0; i < n_tokens; i++) {
                int token_id = tokens[i];
                if (token_id < 0 || token_id >= config_.vocab_size) {
                    throw std::runtime_error("TextEmbed: Token ID out of range: " + std::to_string(token_id));
                }

                const block_q8_0 *src = reinterpret_cast<const block_q8_0 *>(
                    static_cast<const uint8_t *>(embed_data_) + token_id * row_size);
                float *dst = embeddings.data() + i * config_.hidden_size;

                // Dequantize entire row
                dequantize_row_q8_0(src, dst, config_.hidden_size);

                // Apply scale
                for (int j = 0; j < config_.hidden_size; j++) {
                    dst[j] *= scale;
                }
            }
        } else if (embed_type_ == GGML_TYPE_Q4_K) {
            // Q4_K: each row has hidden_size elements, QK_K=256 is block size
            const int n_blocks_per_row = config_.hidden_size / QK_K;
            const size_t row_size = n_blocks_per_row * sizeof(block_q4_K);

            for (int i = 0; i < n_tokens; i++) {
                int token_id = tokens[i];
                if (token_id < 0 || token_id >= config_.vocab_size) {
                    throw std::runtime_error("TextEmbed: Token ID out of range: " + std::to_string(token_id));
                }

                const block_q4_K *src = reinterpret_cast<const block_q4_K *>(
                    static_cast<const uint8_t *>(embed_data_) + token_id * row_size);
                float *dst = embeddings.data() + i * config_.hidden_size;

                // Dequantize entire row
                dequantize_row_q4_K(src, dst, config_.hidden_size);

                // Apply scale
                for (int j = 0; j < config_.hidden_size; j++) {
                    dst[j] *= scale;
                }
            }
        } else if (embed_type_ == GGML_TYPE_Q8_K) {
            // Q8_K: each row has hidden_size elements, QK_K=256 is block size
            const int n_blocks_per_row = config_.hidden_size / QK_K;
            const size_t row_size = n_blocks_per_row * sizeof(block_q8_K);

            for (int i = 0; i < n_tokens; i++) {
                int token_id = tokens[i];
                if (token_id < 0 || token_id >= config_.vocab_size) {
                    throw std::runtime_error("TextEmbed: Token ID out of range: " + std::to_string(token_id));
                }

                const block_q8_K *src = reinterpret_cast<const block_q8_K *>(
                    static_cast<const uint8_t *>(embed_data_) + token_id * row_size);
                float *dst = embeddings.data() + i * config_.hidden_size;

                // Dequantize entire row
                dequantize_row_q8_K(src, dst, config_.hidden_size);

                // Apply scale
                for (int j = 0; j < config_.hidden_size; j++) {
                    dst[j] *= scale;
                }
            }
        } else {
            throw std::runtime_error("TextEmbed: Unsupported embedding type: " + std::to_string(embed_type_) +
                                     " (supported: F32=0, F16=1, Q4_0=2, Q8_0=8, Q4_K=12, Q8_K=15)");
        }

        return embeddings;
    }

    int get_vocab_size() const { return config_.vocab_size; }
    int get_hidden_size() const { return config_.hidden_size; }
    bool is_initialized() const { return initialized_; }

private:
    bool load_embedding(const std::string &model_path) {
        // Load GGUF file
        ggml_context *meta = nullptr;
        gguf_init_params params;
        params.no_alloc = false;  // We need the data
        params.ctx = &meta;

        gguf_context *ctx_gguf = gguf_init_from_file(model_path.c_str(), params);
        if (!ctx_gguf) {
            fprintf(stderr, "TextEmbed: Failed to load GGUF from %s\n", model_path.c_str());
            return false;
        }

        // Read config
        int idx = gguf_find_key(ctx_gguf, "embed.vocab_size");
        if (idx >= 0) {
            config_.vocab_size = gguf_get_val_u32(ctx_gguf, idx);
        }

        idx = gguf_find_key(ctx_gguf, "embed.hidden_size");
        if (idx >= 0) {
            config_.hidden_size = gguf_get_val_u32(ctx_gguf, idx);
        }

        printf("TextEmbed: vocab_size=%d, hidden_size=%d\n", config_.vocab_size, config_.hidden_size);

        // Find embedding tensor
        ggml_tensor *embed_tensor = ggml_get_tensor(meta, "embed.weight");
        if (!embed_tensor) {
            fprintf(stderr, "TextEmbed: embed.weight tensor not found\n");
            gguf_free(ctx_gguf);
            ggml_free(meta);
            return false;
        }

        // Verify dimensions
        // Expected shape: [vocab_size, hidden_size] in row-major (numpy)
        // GGML stores as: ne[0] = hidden_size, ne[1] = vocab_size (column-major)
        if (embed_tensor->ne[0] != config_.hidden_size || embed_tensor->ne[1] != config_.vocab_size) {
            fprintf(stderr, "TextEmbed: Unexpected embedding shape [%lld, %lld], expected [%d, %d]\n",
                    (long long)embed_tensor->ne[0], (long long)embed_tensor->ne[1],
                    config_.hidden_size, config_.vocab_size);
            gguf_free(ctx_gguf);
            ggml_free(meta);
            return false;
        }

        // Store embedding data
        embed_type_ = embed_tensor->type;
        size_t data_size = ggml_nbytes(embed_tensor);
        embed_buffer_.resize(data_size);
        memcpy(embed_buffer_.data(), embed_tensor->data, data_size);
        embed_data_ = embed_buffer_.data();

        printf("TextEmbed: Loaded embedding table (%.2f MB, type=%d)\n",
               data_size / (1024.0 * 1024.0), embed_type_);

        // Store contexts for cleanup
        ctx_gguf_ = ctx_gguf;
        ctx_meta_ = meta;

        return true;
    }

    std::unique_ptr<SpTokenizer> tokenizer_;
    TextEmbedConfig config_;
    bool initialized_ = false;

    // Embedding storage
    std::vector<uint8_t> embed_buffer_;
    const void *embed_data_ = nullptr;
    ggml_type embed_type_ = GGML_TYPE_F32;

    // GGUF contexts (kept alive for data)
    gguf_context *ctx_gguf_ = nullptr;
    ggml_context *ctx_meta_ = nullptr;

public:
    ~Pi05TextEmbed() {
        if (ctx_gguf_) {
            gguf_free(ctx_gguf_);
        }
        if (ctx_meta_) {
            ggml_free(ctx_meta_);
        }
    }
};
