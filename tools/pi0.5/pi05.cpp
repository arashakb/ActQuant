#include "pi05.h"
#include "timer.hpp"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <utility>

namespace fs = std::filesystem;

static void print_inference_summary(
    const std::vector<std::pair<std::string, double>> &vision_times,
    double embed_prefix_ms,
    const ActionProfStats &ap) {
  int N = ap.ode_steps;
  double avg_set  = N > 0 ? ap.ode_set_input_ms  / N : 0;
  double avg_cmp  = N > 0 ? ap.ode_compute_ms    / N : 0;
  double avg_get  = N > 0 ? ap.ode_tensor_get_ms / N : 0;
  double avg_step = avg_set + avg_cmp + avg_get;
  double ode_total = ap.ode_alloc_ms + ap.ode_set_input_ms + ap.ode_compute_ms + ap.ode_tensor_get_ms;
  double kv_total  = ap.kv_rebuild_ms + ap.kv_set_input_ms + ap.kv_compute_ms;
  double vision_total = 0;
  for (auto &v : vision_times) vision_total += v.second;
  double total = vision_total + embed_prefix_ms + kv_total + ode_total;

  printf("\n=== Inference Timing Summary ===\n");
  printf("[Level 1: Chain]\n");
  for (auto &v : vision_times)
    printf("  %-32s: %7.2f ms\n", v.first.c_str(), v.second);
  printf("  %-32s: %7.2f ms\n", "embed_prefix", embed_prefix_ms);
  printf("  %-32s: %7.2f ms\n", "compute_kv_cache", kv_total);
  printf("  %-32s: %7.2f ms  (x%d)\n", "ODE loop total", ode_total, N);
  printf("  --------------------------------\n");
  printf("  %-32s: %7.2f ms\n", "Total", total);
  printf("[Level 2: compute_kv_cache]\n");
  printf("  %-32s: %7.2f ms\n", "rebuild_graph (prefix+suffix)", ap.kv_rebuild_ms);
  printf("  %-32s: %7.2f ms\n", "set_input (prefix_hidden)", ap.kv_set_input_ms);
  printf("  %-32s: %7.2f ms\n", "graph_compute (prefix fwd)", ap.kv_compute_ms);
  printf("[Level 2: ODE per step (avg of %d)]\n", N);
  printf("  %-32s: %7.2f ms  (step 1 only)\n", "alloc_graph", ap.ode_alloc_ms);
  printf("  %-32s: %7.2f ms  avg\n", "set_input", avg_set);
  printf("  %-32s: %7.2f ms  avg\n", "graph_compute (suffix fwd)", avg_cmp);
  printf("  %-32s: %7.2f ms  avg\n", "tensor_get (velocity)", avg_get);
  printf("  --------------------------------\n");
  printf("  %-32s: %7.2f ms  avg\n", "Total per step", avg_step);
  printf("================================\n\n");
  fflush(stdout);
}

// Pi05VisionEncoder implementation
Pi05VisionEncoder::Pi05VisionEncoder(const std::string &vision_path,
                                   const std::string &proj_path,
                                   const ContextParams &ctx_params)
    : vit_(vision_path, ctx_params),
      proj_(proj_path.empty() ? vision_path : proj_path, ctx_params) {}

bool Pi05VisionEncoder::encode(const std::string &img_path, std::vector<float> &out) {
    std::vector<float> vit_out;
    if (!vit_.run(img_path, vit_out)) {
        return false;
    }
    return proj_.run(vit_out, out);
}

bool Pi05VisionEncoder::encode(const uint8_t *img_data, int width, int height,
                               std::vector<float> &out) {
    std::vector<float> vit_out;
    if (!vit_.run(img_data, width, height, vit_out)) {
        return false;
    }
    return proj_.run(vit_out, out);
}

bool Pi05VisionEncoder::encode(const std::vector<float> &pixel_values,
                               std::vector<float> &out) {
    std::vector<float> vit_out;
    if (!vit_.run(pixel_values, vit_out)) {
        return false;
    }
    return proj_.run(vit_out, out);
}

// Pi05 implementation
Pi05::Pi05(const Pi05Params &params) : params_(params) {
    ContextParams ctx_params;
    ctx_params.device_name = params.device;
    ctx_params.n_threads = params.n_threads;
    ctx_params.max_nodes = 4096;
    ctx_params.verbosity = GGML_LOG_LEVEL_INFO;

    printf("Loading Pi0.5 model from %s...\n", params.model_path.c_str());

    // Initialize vision encoder (reads v.* and mm.* tensors from unified GGUF)
    vision_encoder_ = Pi05VisionEncoder(params.model_path,
                                        params.model_path,  // Same file for projector
                                        ctx_params);

    // Initialize text embedding (reads embed.* tensors from unified GGUF)
    printf("Loading text embedding...\n");
    if (!text_embed_.init(params.model_path, params.tokenizer_path)) {
        throw std::runtime_error("Failed to initialize text embedding");
    }

    // Initialize action expert (reads action.* and pali.* tensors from unified GGUF)
    printf("Loading action expert...\n");
    action_expert_ = Pi05ActionExpertRunner(params.model_path, ctx_params);

    // Load config from action expert
    config_.action = action_expert_.get_model().hparams;

    printf("Pi0.5 initialized successfully\n");
}

std::vector<float> Pi05::encode_image(const std::string &image_path) {
    std::vector<float> vision_embeds;
    if (!vision_encoder_.encode(image_path, vision_embeds)) {
        throw std::runtime_error("Failed to encode image: " + image_path);
    }
    return vision_embeds;
}

std::vector<int32_t> Pi05::tokenize(const std::string &text) {
    return text_embed_.tokenize(text);
}

std::vector<int32_t> Pi05::tokenize_prompt(const std::string &prompt) {
    // Match OpenPI PaliGemmaTokenizer for pi0.5 (discrete_state_input=False):
    //   cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    //   tokens  = sp.encode(cleaned, add_bos=True) + sp.encode("\n")

    // 1. Clean text: strip whitespace, replace _ and \n with space
    std::string cleaned = prompt;
    auto l = cleaned.find_first_not_of(" \t\r\n");
    auto r = cleaned.find_last_not_of(" \t\r\n");
    if (l == std::string::npos) {
        cleaned = "";
    } else {
        cleaned = cleaned.substr(l, r - l + 1);
    }
    for (char &c : cleaned) {
        if (c == '_' || c == '\n') c = ' ';
    }

    // 2. Tokenize with BOS
    std::vector<int32_t> tokens = text_embed_.tokenize(cleaned, /*add_special=*/true);

    // 3. Append "\n" token (start-of-answer token, ID=108 in PaliGemma tokenizer)
    //    Matches: sp.encode("\n", add_bos=False) in the reference
    std::vector<int32_t> nl_tokens = text_embed_.tokenize("\n", /*add_special=*/false);
    tokens.insert(tokens.end(), nl_tokens.begin(), nl_tokens.end());

    return tokens;
}

PrefixResult Pi05::embed_prefix(
    const std::vector<float> &vision_embeds,
    const std::vector<int32_t> &tokens
) {
    // Wrap single image in vector and call multi-image version
    // No masked image for single-image case
    std::vector<std::vector<float>> vision_embeds_list = {vision_embeds};
    return embed_prefix(vision_embeds_list, tokens, -1);
}

PrefixResult Pi05::embed_prefix(
    const std::vector<std::vector<float>> &vision_embeds_list,
    const std::vector<int32_t> &tokens,
    int masked_image_idx
) {
    // OpenPI embed_prefix implementation:
    // 1. Vision embeddings are already projected to PaliGemma hidden size (2048)
    // 2. Text tokens are embedded via Gemma embedding table and scaled by sqrt(hidden)
    // 3. Concatenate: [image_0_tokens, image_1_tokens, ..., text_tokens]
    // 4. If masked_image_idx >= 0, that image's tokens are masked in attention

    PrefixResult result;
    result.pali_hidden_size = text_embed_.get_hidden_size();  // 2048 for Gemma-2B

    // Validate vision embeddings
    const int tokens_per_image = 256;  // 16x16 patches from SigLIP
    result.n_images = vision_embeds_list.size();
    result.n_vision_tokens = 0;

    for (const auto &vision_embeds : vision_embeds_list) {
        int expected_size = tokens_per_image * result.pali_hidden_size;
        if ((int)vision_embeds.size() != expected_size) {
            throw std::runtime_error(
                "Vision embedding size mismatch: got " +
                std::to_string(vision_embeds.size()) +
                ", expected " + std::to_string(expected_size)
            );
        }
        result.n_vision_tokens += tokens_per_image;
    }

    // Set mask range for the specified image (e.g., right_wrist placeholder)
    // In OpenPI, right_wrist_0_rgb has image_mask=False, meaning it's masked in attention
    if (masked_image_idx >= 0 && masked_image_idx < result.n_images) {
        result.masked_start = masked_image_idx * tokens_per_image;
        result.masked_end = result.masked_start + tokens_per_image;
        printf("embed_prefix: masking image %d tokens [%d, %d)\n",
               masked_image_idx, result.masked_start, result.masked_end);
    } else {
        result.masked_start = -1;
        result.masked_end = -1;
    }

    // Get text embeddings (scaled by sqrt(hidden_size) as per Gemma)
    std::vector<float> text_embeds;
    if (!tokens.empty()) {
        text_embeds = text_embed_.get_text_embeddings(tokens);
        result.n_text_tokens = tokens.size();
    } else {
        result.n_text_tokens = 0;
    }

    // Allocate prefix buffer
    int n_prefix = result.n_prefix();
    result.prefix_tokens.resize(n_prefix * result.pali_hidden_size);

    // Copy vision embeddings (already in [n_tokens, hidden] layout)
    float *dst = result.prefix_tokens.data();
    for (const auto &vision_embeds : vision_embeds_list) {
        std::memcpy(dst, vision_embeds.data(), vision_embeds.size() * sizeof(float));
        dst += vision_embeds.size();
    }

    // Copy text embeddings
    if (!text_embeds.empty()) {
        std::memcpy(dst, text_embeds.data(), text_embeds.size() * sizeof(float));
    }

    printf("embed_prefix: n_images=%d, n_vision_tokens=%d, n_text_tokens=%d, n_prefix=%d\n",
           result.n_images, result.n_vision_tokens, result.n_text_tokens, n_prefix);
    fflush(stdout);

    return result;
}

std::vector<float> Pi05::sample_actions(const PrefixResult &prefix, int num_steps) {
    int action_dim = config_.action.action_dim;
    int action_horizon = config_.action.action_horizon;
    int timestep_dim = config_.action.timestep_sinusoidal_dim;

    printf("sample_actions: action_dim=%d, horizon=%d, steps=%d, mask=[%d,%d)\n",
           action_dim, action_horizon, num_steps, prefix.masked_start, prefix.masked_end);
    fflush(stdout);

    std::vector<float> actions_out;

    // Use sample_actions_with_cache which:
    // 1. Computes KV cache from prefix (stores internally)
    // 2. Runs flow matching ODE loop using cached KV
    // Pass mask range for right_wrist tokens (they should be masked in attention)
    if (!action_expert_.sample_actions_with_cache(
            prefix.prefix_tokens, action_dim, action_horizon,
            timestep_dim, num_steps, actions_out,
            prefix.masked_start, prefix.masked_end)) {
        throw std::runtime_error("Failed to sample actions with KV cache");
    }

    if (!params_.ode_profile_path.empty()) {
        accumulate_ode_profile(action_expert_.get_ode_step_records());
    }

    return actions_out;
}

bool Pi05::run(const std::string &img_path, const std::string &prompt,
               std::vector<float> &actions_out) {
    Timer timer;

    // 1. Encode image
    timer.start();
    std::vector<float> vision_embeds = encode_image(img_path);
    std::vector<std::pair<std::string, double>> vision_times = {{"Vision", timer.stop<Timer::ms>()}};

    // 2. Tokenize prompt
    printf("Tokenizing prompt: '%s'...\n", prompt.c_str());
    fflush(stdout);
    std::vector<int32_t> tokens = tokenize_prompt(prompt);
    printf("Tokenized prompt: %zu tokens\n", tokens.size());
    fflush(stdout);

    // 3. Embed prefix (vision + text)
    timer.start();
    PrefixResult prefix = embed_prefix(vision_embeds, tokens);
    double t_embed = timer.stop<Timer::ms>();

    // 4. Sample actions using KV cache architecture
    try {
        actions_out = sample_actions(prefix, params_.num_flow_steps);
    } catch (const std::exception &e) {
        fprintf(stderr, "run failed: %s\n", e.what());
        return false;
    }

    if (!params_.ode_profile_path.empty()) {
        double vision_total = 0;
        for (const auto &v : vision_times) vision_total += v.second;
        accumulate_timing(vision_total, t_embed, action_expert_.get_prof_stats());
    }
    print_inference_summary(vision_times, t_embed, action_expert_.get_prof_stats());
    return true;
}

bool Pi05::run(const uint8_t *img_data, int width, int height,
               const std::string &prompt,
               std::vector<float> &actions_out) {
    Timer timer;

    // 1. Encode image from raw data
    timer.start();
    std::vector<float> vision_embeds;
    if (!vision_encoder_.encode(img_data, width, height, vision_embeds)) {
        return false;
    }
    std::vector<std::pair<std::string, double>> vision_times = {{"Vision", timer.stop<Timer::ms>()}};

    // 2. Tokenize prompt
    std::vector<int32_t> tokens = tokenize_prompt(prompt);

    // 3. Embed prefix
    timer.start();
    PrefixResult prefix = embed_prefix(vision_embeds, tokens);
    double t_embed = timer.stop<Timer::ms>();

    // 4. Sample actions
    try {
        actions_out = sample_actions(prefix, params_.num_flow_steps);
    } catch (const std::exception &e) {
        fprintf(stderr, "run failed: %s\n", e.what());
        return false;
    }

    if (!params_.ode_profile_path.empty()) {
        double vision_total = 0;
        for (const auto &v : vision_times) vision_total += v.second;
        accumulate_timing(vision_total, t_embed, action_expert_.get_prof_stats());
    }
    print_inference_summary(vision_times, t_embed, action_expert_.get_prof_stats());
    return true;
}

bool Pi05::run(const ImageInputs &images, const std::string &prompt,
               std::vector<float> &actions_out) {
    Timer timer;
    std::vector<std::pair<std::string, double>> vision_times;

    // OpenPI/Pi0.5 always expects 3 images: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
    // The third image (right_wrist) is typically all-zeros and masked out,
    // but it MUST be present for correct position encoding and attention patterns.
    const int TOKENS_PER_IMAGE = 256;
    const int PALI_HIDDEN_SIZE = 2048;

    std::vector<std::vector<float>> vision_embeds_list;

    // 1. Encode base image (base_0_rgb)
    timer.start();
    std::vector<float> base_embeds;
    if (!images.base_path.empty()) {
        if (!vision_encoder_.encode(images.base_path, base_embeds)) {
            fprintf(stderr, "Failed to encode base image: %s\n", images.base_path.c_str());
            return false;
        }
    } else if (images.base_data != nullptr) {
        if (!vision_encoder_.encode(images.base_data, images.width, images.height, base_embeds)) {
            fprintf(stderr, "Failed to encode base image from raw data\n");
            return false;
        }
    } else {
        fprintf(stderr, "No base image provided\n");
        return false;
    }
    vision_embeds_list.push_back(std::move(base_embeds));
    vision_times.push_back({"Vision (base)", timer.stop<Timer::ms>()});

    // 2. Encode left wrist image (left_wrist_0_rgb)
    if (images.has_wrist()) {
        timer.start();
        std::vector<float> wrist_embeds;
        if (!images.wrist_path.empty()) {
            if (!vision_encoder_.encode(images.wrist_path, wrist_embeds)) {
                fprintf(stderr, "Failed to encode wrist image: %s\n", images.wrist_path.c_str());
                return false;
            }
        } else if (images.wrist_data != nullptr) {
            if (!vision_encoder_.encode(images.wrist_data, images.width, images.height, wrist_embeds)) {
                fprintf(stderr, "Failed to encode wrist image from raw data\n");
                return false;
            }
        }
        vision_embeds_list.push_back(std::move(wrist_embeds));
        vision_times.push_back({"Vision (left_wrist)", timer.stop<Timer::ms>()});
    } else {
        // If no wrist image provided, add zeros (will be masked anyway)
        std::vector<float> dummy_embeds(TOKENS_PER_IMAGE * PALI_HIDDEN_SIZE, 0.0f);
        vision_embeds_list.push_back(std::move(dummy_embeds));
        printf("Left wrist: using zero embeddings (no wrist image provided)\n");
        fflush(stdout);
    }

    // 3. Add right_wrist_0_rgb placeholder (always zeros, masked out in OpenPI)
    // This is CRITICAL for matching the training input format!
    std::vector<float> right_wrist_embeds(TOKENS_PER_IMAGE * PALI_HIDDEN_SIZE, 0.0f);
    vision_embeds_list.push_back(std::move(right_wrist_embeds));
    printf("Right wrist: using zero embeddings (placeholder)\n");
    fflush(stdout);

    printf("Total images: %zu (matching OpenPI format: base + left_wrist + right_wrist)\n",
           vision_embeds_list.size());
    fflush(stdout);

    // 2. Tokenize prompt
    std::vector<int32_t> tokens = tokenize_prompt(prompt);

    // 3. Embed prefix (multi-image version)
    // IMPORTANT: mask image index 2 (right_wrist) as per OpenPI's image_mask=False
    // This is critical for matching the training attention pattern!
    const int RIGHT_WRIST_IMAGE_IDX = 2;
    timer.start();
    PrefixResult prefix = embed_prefix(vision_embeds_list, tokens, RIGHT_WRIST_IMAGE_IDX);
    double t_embed = timer.stop<Timer::ms>();

    // 4. Sample actions
    try {
        actions_out = sample_actions(prefix, params_.num_flow_steps);
    } catch (const std::exception &e) {
        fprintf(stderr, "run failed: %s\n", e.what());
        return false;
    }

    if (!params_.ode_profile_path.empty()) {
        double vision_total = 0;
        for (const auto &v : vision_times) vision_total += v.second;
        accumulate_timing(vision_total, t_embed, action_expert_.get_prof_stats());
    }
    print_inference_summary(vision_times, t_embed, action_expert_.get_prof_stats());
    return true;
}

// ---------------------------------------------------------------------------
// ODE step distribution profiling
// ---------------------------------------------------------------------------

void Pi05::accumulate_ode_profile(const std::vector<OdeStepRecord> &records) {
    // Lazy-init per-horizon-position accumulator
    if (hpos_horizon_ == 0 && config_.action.action_horizon > 0) {
        hpos_horizon_ = config_.action.action_horizon;
        hpos_dim_     = config_.action.action_dim;
        sum_hpos_change_.assign(MAX_ODE_STEPS,
                                std::vector<double>(hpos_horizon_, 0.0));
    }

    for (const auto &r : records) {
        if (r.step < 0 || r.step >= MAX_ODE_STEPS) continue;
        StepAccum &a = step_accum_[r.step];
        a.sum_vel_norm += r.vel_norm;
        a.n_total++;
        if (r.vel_L1 >= 0.f) {
            a.sum_vel_L1   += r.vel_L1;
            a.sum_input_L1 += r.input_L1;
            a.sum_diff_err += r.diff_err >= 0.f ? r.diff_err : 0.f;
            a.n_diff++;
        }
        if (r.taylor_err >= 0.f) {
            a.sum_taylor_err += r.taylor_err;
            a.n_taylor++;
        }
        // Per-horizon-position change accumulation (step >= 1 has per_elem_change)
        if (hpos_horizon_ > 0 && !r.per_elem_change.empty()) {
            auto &hrow = sum_hpos_change_[r.step];
            for (int h = 0; h < hpos_horizon_; ++h) {
                double m = 0.0;
                for (int j = 0; j < hpos_dim_; ++j) {
                    int idx = h * hpos_dim_ + j;
                    if (idx < (int)r.per_elem_change.size())
                        m += r.per_elem_change[idx];
                }
                hrow[h] += m / hpos_dim_;
            }
        }
    }
    profile_calls_++;
    // Write after every call so data is preserved even if the process is killed (SIGKILL).
    write_ode_profile();
}

void Pi05::accumulate_timing(double vision_total_ms, double embed_ms,
                             const ActionProfStats &ap) {
    timing_accum_.sum_vision_ms      += vision_total_ms;
    timing_accum_.sum_embed_ms       += embed_ms;
    timing_accum_.sum_kv_rebuild_ms  += ap.kv_rebuild_ms;
    timing_accum_.sum_kv_set_ms      += ap.kv_set_input_ms;
    timing_accum_.sum_kv_compute_ms  += ap.kv_compute_ms;
    timing_accum_.sum_ode_alloc_ms   += ap.ode_alloc_ms;
    timing_accum_.sum_ode_set_ms     += ap.ode_set_input_ms;
    timing_accum_.sum_ode_compute_ms += ap.ode_compute_ms;
    timing_accum_.sum_ode_get_ms     += ap.ode_tensor_get_ms;
    timing_accum_.n_ode_steps        += ap.ode_steps;
    timing_accum_.n++;
}

void Pi05::write_ode_profile() const {
    std::ofstream f(params_.ode_profile_path);
    if (!f.is_open()) {
        fprintf(stderr, "write_ode_profile: cannot open %s\n",
                params_.ode_profile_path.c_str());
        return;
    }

    auto fmt_f = [](double v, int nd) -> std::string {
        char buf[32];
        snprintf(buf, sizeof(buf), "%.*f", nd, v);
        return buf;
    };
    auto mean_or_na = [&](double sum, int n) -> std::string {
        return n > 0 ? fmt_f(sum / n, 6) : "N/A";
    };

    const int    N    = timing_accum_.n > 0 ? timing_accum_.n : 1;
    const int    Ns   = timing_accum_.n_ode_steps > 0 ? timing_accum_.n_ode_steps : 1;
    const double kv_t = (timing_accum_.sum_kv_rebuild_ms + timing_accum_.sum_kv_set_ms
                       + timing_accum_.sum_kv_compute_ms) / N;
    const double od_t = (timing_accum_.sum_ode_alloc_ms + timing_accum_.sum_ode_set_ms
                       + timing_accum_.sum_ode_compute_ms + timing_accum_.sum_ode_get_ms) / N;
    const double tot  = timing_accum_.sum_vision_ms / N + timing_accum_.sum_embed_ms / N
                      + kv_t + od_t;
    // Per-step ODE averages (total_ode_field / total_ode_steps)
    const double s_set = timing_accum_.sum_ode_set_ms    / Ns;
    const double s_cmp = timing_accum_.sum_ode_compute_ms / Ns;
    const double s_get = timing_accum_.sum_ode_get_ms    / Ns;
    const double s_tot = s_set + s_cmp + s_get;  // alloc excluded (step 1 only)

    f << "# Pi0.5 ODE Step Distribution Profile\n";
    f << "# model: " << params_.model_path << "\n";
    f << "# total_inference_calls: " << profile_calls_ << "\n";
    f << "#\n";
    f << "# ---- Inference Chain Timing (mean ms, N=" << N << " calls) ----\n";
    f << "# Step                          |  Mean (ms)\n";
    f << "# --------------------------------+-----------\n";
    f << "# Vision                         | " << fmt_f(timing_accum_.sum_vision_ms / N, 2) << "\n";
    f << "# embed_prefix                   | " << fmt_f(timing_accum_.sum_embed_ms   / N, 2) << "\n";
    f << "# compute_kv_cache               | " << fmt_f(kv_t, 2) << "\n";
    f << "# ODE loop total (x" << (timing_accum_.n_ode_steps / std::max(N,1))
      << ")            | " << fmt_f(od_t, 2) << "\n";
    f << "# ................................|...........\n";
    f << "# Total                          | " << fmt_f(tot, 2) << "\n";
    f << "#\n";
    f << "# [compute_kv_cache breakdown]\n";
    f << "# rebuild_graph (prefix+suffix)  | " << fmt_f(timing_accum_.sum_kv_rebuild_ms / N, 2) << "\n";
    f << "# set_input (prefix_hidden)      | " << fmt_f(timing_accum_.sum_kv_set_ms     / N, 2) << "\n";
    f << "# graph_compute (prefix fwd)     | " << fmt_f(timing_accum_.sum_kv_compute_ms / N, 2) << "\n";
    f << "#\n";
    f << "# [ODE per step avg over " << Ns << " steps]\n";
    f << "# alloc_graph (step 1 only)      | " << fmt_f(timing_accum_.sum_ode_alloc_ms / N, 2) << "\n";
    f << "# set_input                      | " << fmt_f(s_set, 2) << "\n";
    f << "# graph_compute (suffix fwd)     | " << fmt_f(s_cmp, 2) << "\n";
    f << "# tensor_get (velocity)          | " << fmt_f(s_get, 2) << "\n";
    f << "# ................................|...........\n";
    f << "# Total per step                 | " << fmt_f(s_tot, 2) << "\n";
    f << "#\n";
    f << "# Columns:\n";
    f << "#   step        - ODE step index (0-based)\n";
    f << "#   t           - ODE time at this step\n";
    f << "#   vel_norm    - mean(|v|), velocity magnitude\n";
    f << "#   vel_L1      - mean(|v - v_prev|), velocity step-to-step change\n";
    f << "#   input_L1    - mean(|x - x_prev|), noisy_actions step-to-step change\n";
    f << "#   rate        - vel_L1 / input_L1, model amplification factor\n";
    f << "#   diff_err%   - EasyCache approx error: mean(|v-(x+diff)|)/vel_norm * 100\n";
    f << "#   taylor_err% - 1st-order extrapolation error: mean(|v-(2vp-vpp)|)/vel_norm * 100\n";
    f << "#   n           - number of samples averaged\n";
    f << "#\n";
    f << "step,t,vel_norm,vel_L1,input_L1,rate,diff_err_pct,taylor_err_pct,n\n";

    for (int s = 0; s < MAX_ODE_STEPS; ++s) {
        const StepAccum &a = step_accum_[s];
        if (a.n_total == 0) break;

        float t_val = 1.0f - s * (1.0f / params_.num_flow_steps);

        std::string vel_norm_s = mean_or_na(a.sum_vel_norm, a.n_total);
        std::string vel_L1_s   = mean_or_na(a.sum_vel_L1,   a.n_diff);
        std::string input_L1_s = mean_or_na(a.sum_input_L1, a.n_diff);
        std::string rate_s     = "N/A";
        std::string diff_err_s = "N/A";
        std::string taylor_s   = "N/A";

        if (a.n_diff > 0 && a.sum_input_L1 > 1e-12) {
            rate_s = fmt_f(a.sum_vel_L1 / a.sum_input_L1, 4);
        }
        if (a.n_diff > 0) {
            diff_err_s = fmt_f((a.sum_diff_err / a.n_diff) * 100.0, 2);
        }
        if (a.n_taylor > 0) {
            taylor_s = fmt_f((a.sum_taylor_err / a.n_taylor) * 100.0, 2);
        }

        f << s << "," << fmt_f(t_val, 4) << ","
          << vel_norm_s << "," << vel_L1_s << "," << input_L1_s << ","
          << rate_s << "," << diff_err_s << "," << taylor_s << ","
          << a.n_total << "\n";
    }

    // Per-horizon-position per-step change table.
    // Rows tagged TOKEN_CHANGE for shell-script parsing.
    // Columns: step, t, h0, h1, ..., h{horizon-1}
    // Value: mean over all calls of mean_action_dim(|x_t[h] - x_{t-1}[h]|).
    if (hpos_horizon_ > 0 && profile_calls_ > 0) {
        f << "# ---- Per-Action-Token Step Change ----\n";
        f << "# TOKEN_CHANGE,step,t,h0,h1,...,h" << (hpos_horizon_ - 1) << "\n";
        f << "# Values: mean |dx| per horizon position (averaged over action_dim and all calls)\n";
        for (int s = 1; s < MAX_ODE_STEPS; ++s) {  // step 0 has no prev
            if (step_accum_[s].n_diff == 0) break;
            const int nd = step_accum_[s].n_diff;  // same count as per-hpos
            float t_val = 1.0f - s * (1.0f / params_.num_flow_steps);
            f << "TOKEN_CHANGE," << s << "," << fmt_f(t_val, 4);
            for (int h = 0; h < hpos_horizon_; ++h) {
                f << "," << fmt_f(sum_hpos_change_[s][h] / nd, 6);
            }
            f << "\n";
        }
    }

    f.close();
}
