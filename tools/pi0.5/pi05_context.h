#pragma once

#include "pi05_memory.h"
#include "model_defs.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include <vector>
#include <cstdint>

class Pi05ActionExpert;

/**
 * Pi05Context - Unified context for Pi0.5 action expert inference
 *
 * This class manages:
 * - Pi05Memory for KV cache
 * - Scheduler for graph execution
 * - Graph result caching for reuse
 * - Unified process_prefix / process_suffix interface
 */
class Pi05Context {
public:
    /**
     * Graph result structure for caching and reuse
     */
    struct GraphResult {
        ggml_cgraph* gf = nullptr;
        ggml_context_ptr ctx_compute;
        std::vector<uint8_t> buf_compute_meta;
        int cached_n_prefix = -1;
        int cached_n_suffix = -1;
        bool valid = false;

        /**
         * Check if the graph can be reused with given parameters
         */
        bool can_reuse(int n_prefix, int n_suffix) const {
            return valid && (cached_n_prefix == n_prefix) && (cached_n_suffix == n_suffix);
        }

        /**
         * Reset the graph result
         */
        void reset() {
            gf = nullptr;
            ctx_compute.reset();
            cached_n_prefix = -1;
            cached_n_suffix = -1;
            valid = false;
        }
    };

    Pi05Context() = default;
    ~Pi05Context() = default;

    // Non-copyable
    Pi05Context(const Pi05Context&) = delete;
    Pi05Context& operator=(const Pi05Context&) = delete;

    /**
     * Initialize the context
     * @param expert Pointer to action expert model (weights)
     * @param memory Pointer to KV cache memory
     * @param sched Scheduler for graph execution
     * @param max_nodes Maximum nodes for graph allocation
     * @return true on success
     */
    bool init(Pi05ActionExpert* expert, Pi05Memory* memory,
              ggml_backend_sched_t sched, int max_nodes);

    /**
     * Process prefix tokens through the model
     * Computes KV cache and stores in memory
     *
     * @param prefix_hidden Prefix embeddings [pali_hidden_size * n_prefix]
     * @param n_prefix Number of prefix tokens
     * @return true on success
     */
    bool process_prefix(const std::vector<float>& prefix_hidden, int n_prefix);

    /**
     * Process suffix tokens using cached KV
     * Used for each flow matching step
     *
     * @param noisy_actions Noisy action embeddings [action_dim * action_horizon]
     * @param timestep_emb Timestep embedding [timestep_dim]
     * @param velocity_out Output velocity field [action_dim * action_horizon]
     * @param masked_start Start of masked token range (-1 for none)
     * @param masked_end End of masked token range (-1 for none)
     * @return true on success
     */
    bool process_suffix(const std::vector<float>& noisy_actions,
                        const std::vector<float>& timestep_emb,
                        std::vector<float>& velocity_out,
                        int masked_start = -1, int masked_end = -1);

    /**
     * Clear context state (resets memory and graph results)
     */
    void clear();

    /**
     * Check if context is initialized
     */
    bool is_initialized() const { return initialized_; }

    /**
     * Get current prefix length
     */
    int get_n_prefix() const { return memory_ ? memory_->get_prefix_len() : 0; }

    /**
     * Get suffix length (action horizon)
     */
    int get_n_suffix() const;

private:
    // Build prefix graph
    void build_prefix_graph(int n_prefix);

    // Build suffix graph
    void build_suffix_graph(int n_prefix, int n_suffix);

    // Set inputs for prefix graph
    void set_prefix_inputs(const std::vector<float>& prefix_hidden, int n_prefix);

    // Set inputs for suffix graph
    void set_suffix_inputs(const std::vector<float>& noisy_actions,
                           const std::vector<float>& timestep_emb,
                           int masked_start, int masked_end);

    bool initialized_ = false;
    Pi05ActionExpert* expert_ = nullptr;
    Pi05Memory* memory_ = nullptr;
    ggml_backend_sched_t sched_ = nullptr;
    int max_nodes_ = 0;

    // Graph results for prefix and suffix
    GraphResult prefix_result_;
    GraphResult suffix_result_;

    // Cached data
    std::vector<int> prefix_positions_;
    std::vector<int> suffix_positions_;
    std::vector<float> prefix_mask_data_;
    std::vector<float> suffix_mask_data_;
};
