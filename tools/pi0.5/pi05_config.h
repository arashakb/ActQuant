#pragma once

#include <cstdint>

struct pi05_vision_config {
    int32_t hidden_size = 1152;        // SigLIP-So400m hidden size
    int32_t intermediate_size = 4304;  // SigLIP MLP intermediate size
    int32_t num_hidden_layers = 27;
    int32_t num_attention_heads = 16;  // SigLIP has 16 heads (head_dim=72)
    int32_t num_channels = 3;
    int32_t image_size = 224;
    int32_t patch_size = 14;           // SigLIP patch size
    int32_t projection_dim = 2048;
    float layer_norm_eps = 1e-6f;
    int32_t num_patches() const { return (image_size / patch_size) * (image_size / patch_size); }
    int32_t head_dim() const { return hidden_size / num_attention_heads; }  // 1152/16 = 72
};

struct pi05_text_config {
    int32_t vocab_size = 257152;
    int32_t hidden_size = 2048;
    int32_t intermediate_size = 16384;  // PaliGemma/Gemma-2B FFN intermediate size
    int32_t num_hidden_layers = 18;
    int32_t num_attention_heads = 8;
    int32_t num_key_value_heads = 1;
    int32_t head_dim = 256;
    int32_t max_position_embeddings = 8192;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
};

struct pi05_action_config {
    int32_t action_dim = 32;           // Robot action dimension (padded, actual LIBERO uses 7)
    int32_t state_dim = 32;            // Robot state dimension (padded, actual LIBERO uses 8)
    int32_t action_horizon = 50;       // Model output action steps (chunk_size from config)
    int32_t hidden_size = 1024;        // Action Expert (Gemma-300M) hidden size
    int32_t intermediate_size = 4096;  // Action Expert FFN intermediate size
    int32_t num_hidden_layers = 18;
    int32_t num_attention_heads = 8;   // Q heads
    int32_t num_key_value_heads = 1;   // KV heads (GQA)
    int32_t head_dim = 256;            // Explicit head_dim (Q: 8*256=2048, K/V: 1*256=256)
    int32_t max_timesteps = 1000;
    int32_t timestep_sinusoidal_dim = 1024;  // Same as hidden_size for pi0.5
    float rms_norm_eps = 1e-6f;
    bool use_adaln = true;             // Pi0.5 uses Adaptive LayerNorm (AdaRMSNorm)
    // PaliGemma config for joint attention
    int32_t pali_hidden_size = 2048;   // PaliGemma hidden size
    int32_t pali_intermediate_size = 16384;  // PaliGemma FFN intermediate size
    // Prefix length: 256 (vision) * n_images + n_text_tokens (configurable at runtime)
    int32_t n_prefix_tokens = 262;     // Default: 256 vision + 6 text (typical short prompt)
};

struct pi05_config {
    pi05_vision_config vision;
    pi05_text_config text;
    pi05_action_config action;
    int32_t image_token_index = 256000;
    float embeddings_scale = 1.0f;
};
