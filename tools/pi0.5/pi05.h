#pragma once

#include "action_expert.hpp"
#include "text_embed.hpp"
#include "pi05_config.h"
#include "proj.hpp"
#include "vit.hpp"
#include <string>
#include <vector>

// Pi05 Parameters
struct Pi05Params {
    std::string model_path;         // Path to pi05_unified.gguf
    std::string tokenizer_path;     // Path to tokenizer.model (SentencePiece)

    // Runtime parameters
    int n_threads = 4;
    int num_flow_steps = 10;
    std::string device = "CPU";

    // Profile output: if non-empty, write ODE step distribution stats to this path on destruction
    // Format: CSV with per-step aggregated metrics across all inference calls
    std::string ode_profile_path;
};

// Multi-image input structure for robots with multiple cameras
struct ImageInputs {
    std::string base_path;              // Path to base/main camera image
    std::string wrist_path;             // Path to wrist camera image (optional)
    const uint8_t *base_data = nullptr; // Raw base image data (alternative to path)
    const uint8_t *wrist_data = nullptr;// Raw wrist image data (alternative to path)
    int width = 0;                      // Image width if using raw data
    int height = 0;                     // Image height if using raw data

    bool has_wrist() const {
        return !wrist_path.empty() || wrist_data != nullptr;
    }
};

// Result of embed_prefix operation
// Contains concatenated [vision_tokens, text_tokens] embeddings
struct PrefixResult {
    std::vector<float> prefix_tokens;  // [n_prefix, pali_hidden_size=2048]
    std::vector<bool> token_mask;       // [n_prefix] - true if token is valid, false if masked
    int n_images;                       // Number of images (1-3)
    int n_vision_tokens;                // Total vision tokens (256 per image)
    int n_text_tokens;                  // Number of text tokens
    int pali_hidden_size;               // PaliGemma hidden size (2048)
    int masked_start = -1;              // Start index of masked tokens (-1 if none)
    int masked_end = -1;                // End index of masked tokens (-1 if none)

    int n_prefix() const { return n_vision_tokens + n_text_tokens; }
};

// Vision encoder + projector combined
class Pi05VisionEncoder {
public:
    Pi05VisionEncoder() = default;
    Pi05VisionEncoder(const std::string &vision_path,
                     const std::string &proj_path,
                     const ContextParams &ctx_params);

    bool encode(const std::string &img_path, std::vector<float> &out);
    bool encode(const uint8_t *img_data, int width, int height,
                std::vector<float> &out);
    bool encode(const std::vector<float> &pixel_values, std::vector<float> &out);

private:
    Pi05Vit vit_;
    Pi05Projector proj_;
};

// Main Pi05.5 inference class
class Pi05 {
public:
    Pi05(const Pi05Params &params);

    // Embed prefix: vision + text tokens concatenated
    // Returns: [n_vision_tokens + n_text_tokens, 2048] embeddings
    PrefixResult embed_prefix(
        const std::vector<float> &vision_embeds,  // Single image [256 * 2048]
        const std::vector<int32_t> &tokens
    );

    // Overload for multiple images (e.g., base + wrist camera)
    // masked_image_idx: index of image to mask in attention (-1 for none)
    // For Libero, image 2 (right_wrist) is always masked
    PrefixResult embed_prefix(
        const std::vector<std::vector<float>> &vision_embeds_list,
        const std::vector<int32_t> &tokens,
        int masked_image_idx = -1
    );

    // Sample actions using KV cache + flow matching
    // prefix: result from embed_prefix()
    // num_steps: number of ODE integration steps (default 10)
    // Returns: actions [action_horizon, action_dim]
    std::vector<float> sample_actions(const PrefixResult &prefix, int num_steps = 10);

    // Full inference: image path + prompt -> actions
    bool run(const std::string &img_path, const std::string &prompt,
             std::vector<float> &actions_out);

    // Full inference: raw image data + prompt -> actions
    bool run(const uint8_t *img_data, int width, int height,
             const std::string &prompt,
             std::vector<float> &actions_out);

    // Multi-image inference (e.g., base + wrist camera)
    bool run(const ImageInputs &images, const std::string &prompt,
             std::vector<float> &actions_out);

    // Utility methods
    std::vector<float> encode_image(const std::string &image_path);
    std::vector<int32_t> tokenize(const std::string &text);

    // Tokenize prompt matching OpenPI PaliGemmaTokenizer for pi0.5:
    //   cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    //   tokens  = sp.encode(cleaned, add_bos=True) + sp.encode("\n")
    std::vector<int32_t> tokenize_prompt(const std::string &prompt);

    int get_hidden_size() const { return text_embed_.get_hidden_size(); }
    const pi05_config &config() const { return config_; }

private:
    Pi05VisionEncoder vision_encoder_;
    Pi05TextEmbed text_embed_;
    Pi05ActionExpertRunner action_expert_;

    Pi05Params params_;
    pi05_config config_;

    // ODE step distribution accumulation across all inference calls
    struct StepAccum {
        double sum_vel_norm   = 0, sum_vel_L1    = 0, sum_input_L1  = 0;
        double sum_diff_err   = 0, sum_taylor_err = 0;
        int    n_total        = 0;  // samples with vel_norm (all steps)
        int    n_diff         = 0;  // samples with vel_L1/input_L1/diff_err (step >= 1)
        int    n_taylor       = 0;  // samples with taylor_err (step >= 2)
    };
    static constexpr int MAX_ODE_STEPS = 20;
    StepAccum step_accum_[MAX_ODE_STEPS] = {};

    // Full inference chain timing accumulation (mirrors print_inference_summary)
    struct TimingAccum {
        double sum_vision_ms     = 0;  // all vision encode calls summed
        double sum_embed_ms      = 0;
        double sum_kv_rebuild_ms = 0;
        double sum_kv_set_ms     = 0;
        double sum_kv_compute_ms = 0;
        double sum_ode_alloc_ms  = 0;
        double sum_ode_set_ms    = 0;
        double sum_ode_compute_ms= 0;
        double sum_ode_get_ms    = 0;
        int    n_ode_steps       = 0;  // total ODE steps across all calls (for per-step avg)
        int    n                 = 0;  // number of inference calls
    };
    TimingAccum timing_accum_;
    int    profile_calls_    = 0;

    // Per-horizon-position per-ODE-step change accumulation.
    // sum_hpos_change_[step_idx][horizon_pos] = sum over calls of
    //   mean_over_action_dim(|x_t[h*dim+j] - x_{t-1}[h*dim+j]|).
    // Lazy-initialized on first call (size depends on model action_horizon).
    std::vector<std::vector<double>> sum_hpos_change_;  // [MAX_ODE_STEPS][action_horizon]
    int hpos_horizon_ = 0;
    int hpos_dim_     = 0;

    void accumulate_ode_profile(const std::vector<OdeStepRecord> &records);
    void accumulate_timing(double vision_total_ms, double embed_ms,
                           const ActionProfStats &ap);
    void write_ode_profile() const;
};
