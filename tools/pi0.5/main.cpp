#include "CLI11.hpp"
#include "pi05.h"
#include "timer.hpp"
#include "utils.h"
#include <cstdio>
#include <vector>

static void print_actions(const std::vector<float> &actions, int action_dim, int action_horizon) {
    printf("\nOutput actions (%d dim x %d horizon = %zu total):\n",
           action_dim, action_horizon, actions.size());

    printf("First timestep actions:\n  [");
    for (int i = 0; i < std::min(action_dim, 10); i++) {
        printf("%.4f", actions[i]);
        if (i < action_dim - 1) printf(", ");
    }
    if (action_dim > 10) printf(", ...");
    printf("]\n");

    printf("\nAction statistics:\n");
    float min_val = actions[0], max_val = actions[0], sum = 0;
    for (auto v : actions) {
        min_val = std::min(min_val, v);
        max_val = std::max(max_val, v);
        sum += v;
    }
    printf("  min: %.4f, max: %.4f, mean: %.4f\n",
           min_val, max_val, sum / actions.size());
}

int main(int argc, char **argv) {
    CLI::App app{"Pi0.5 Inference Tool"};
    argv = app.ensure_utf8(argv);

    std::string model_dir = "";
    std::string model_file = "pi05.gguf";
    std::string tokenizer_file = "tokenizer.model";
    std::string img_path = "";
    std::string prompt = "pick up the object";
    std::string device_name = "CPU";
    int n_threads = 4;
    int num_steps = 10;

    app.add_option("-m,--model_dir", model_dir,
                   "Directory containing model files")->required();
    app.add_option("--model", model_file,
                   "Model filename (default: pi05.gguf)");
    app.add_option("-t,--tokenizer", tokenizer_file,
                   "Tokenizer filename (default: tokenizer.model)");
    app.add_option("-i,--img", img_path,
                   "Path to input image")->required();
    app.add_option("-p,--prompt", prompt,
                   "Text prompt for the model");
    app.add_option("-d,--device", device_name,
                   "Device name (default: CPU)");
    app.add_option("-n,--n_threads", n_threads,
                   "Number of threads (default: 4)");
    app.add_option("-s,--steps", num_steps,
                   "Flow matching steps (default: 10)");

    CLI11_PARSE(app, argc, argv);

    // Build full paths
    std::string model_path = model_dir;
    if (!model_path.empty() && model_path.back() != '/') {
        model_path += '/';
    }
    std::string tokenizer_path = model_path + tokenizer_file;
    model_path += model_file;

    printf("Pi0.5 Inference\n");
    printf("===============\n");
    printf("Model:         %s\n", model_path.c_str());
    printf("Tokenizer:     %s\n", tokenizer_path.c_str());
    printf("Image:         %s\n", img_path.c_str());
    printf("Prompt:        %s\n", prompt.c_str());
    printf("Threads:       %d\n", n_threads);
    printf("Flow steps:    %d\n", num_steps);
    printf("Device:        %s\n", device_name.c_str());
    printf("\n");

    try {
        Pi05Params params;
        params.model_path = model_path;
        params.tokenizer_path = tokenizer_path;
        params.n_threads = n_threads;
        params.num_flow_steps = num_steps;
        params.device = device_name;

        Pi05 pi0(params);

        std::vector<float> actions;

        Timer timer(true);
        if (!pi0.run(img_path, prompt, actions)) {
            fprintf(stderr, "Inference failed\n");
            return 1;
        }

        printf("Total inference time: %.2f ms\n", timer.elapsed<Timer::ms>());

        const auto &config = pi0.config();
        print_actions(actions, config.action.action_dim, config.action.action_horizon);

        printf("\nInference completed successfully.\n");

    } catch (const std::exception &e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    return 0;
}
