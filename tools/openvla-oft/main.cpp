#include "CLI11.hpp"
#include "ctx_manager.h"
#include "hftokenizer.hpp"
#include "infer_session.hpp"
#include "llama.h"
#include "llm.h"
#include "model_defs.h"
#include "model_loader.h"
#include "openvla.h"
#include "proj.hpp"
#include "timer.hpp"
#include "utils.h"
#include "vit.hpp"
#include <memory>

int main(int argc, char **argv) {
  CLI::App app{"OpenVLA Test Suite"};
  argv = app.ensure_utf8(argv);

  std::string model_dir = "/home/ubuntu/data_1/chenxing/spatial/";
  std::string dinov2_path = "dinov2.gguf";
  std::string siglip_path = "siglip.gguf";
  std::string proj_path = "proj.gguf";
  std::string llm_path = "llama2_q8_0.gguf";
  std::string action_head_path = "action_head.gguf";
  std::string img_path = "/home/ubuntu/data_1/chenxing/model_gguf/example.png";
  std::string prompt = "pick up the black bowl between the plate and the ramekin and place it on the plate";
  std::string tokenizer_path = "tokenizer.json";
  std::string dataset_statistics_path = "dataset_statistics.json";
  std::string device_name = "CUDA0";
  int n_threads = 4;
  int n_ctx = 300;

  app.add_option(
      "-m,--model_dir", model_dir,
      "Base directory for models (default: /mount/weights/vote_model/)");
  app.add_option(
      "--dinov2_model", dinov2_path,
      "DINOv2 model filename in the model directory (default: dinov2.gguf)");
  app.add_option(
      "--siglip_model", siglip_path,
      "Siglip model filename in the model directory (default: siglip.gguf)");
  app.add_option(
      "--proj_model", proj_path,
      "Projection model filename in the model directory (default: proj.gguf)");
  app.add_option("--action_head_model", action_head_path,
                 "Action head model filename in the model directory (default: "
                 "action_head.gguf)");
  app.add_option(
      "--llm_model", llm_path,
      "LLM model filename in the model directory (default: llm_q8_0.gguf)");
  app.add_option(
      "-t,--tokenizer", tokenizer_path,
      "Path to the tokenizer (default: empty, use built-in tokenizer)");
  app.add_option("-i,--img", img_path, "Path to the input image");
  app.add_option("-p,--prompt", prompt, "Text prompt for the model");
  app.add_option("-d,--device", device_name,
                 "Device name for computation (default: CUDA0)");
  app.add_option("-n,--n_threads", n_threads,
                 "Number of threads for computation (default: 4)");
  app.add_option("-c,--n_ctx", n_ctx, "Context size for LLM (default: 300)");
  app.add_option("-s,--stats", dataset_statistics_path,
                 "Path to dataset_statistics.json (default: dataset_statistics.json)");

  CLI11_PARSE(app, argc, argv);
  if (model_dir[model_dir.size() - 1] != '/') {
    model_dir += '/';
  }
  if (!model_dir.empty()) {
    if (model_dir[model_dir.size() - 1] != '/') {
      model_dir += '/';
    }
    dinov2_path = model_dir + dinov2_path;
    siglip_path = model_dir + siglip_path;
    proj_path = model_dir + proj_path;
    llm_path = model_dir + llm_path;
    if (!action_head_path.empty()) {
      action_head_path = model_dir + action_head_path;
    }
    if (!tokenizer_path.empty()) {
      tokenizer_path = model_dir + tokenizer_path;
    }
    dataset_statistics_path = model_dir + dataset_statistics_path;
  }
  ContextParams ctx_params = {.device_name = device_name,
                              .n_threads = n_threads,
                              .max_nodes = 2048,
                              .verbosity = GGML_LOG_LEVEL_DEBUG};
  LlmParam llm_params = {.ngl = 99,
                         .n_ctx = n_ctx,
                         .tokenizer_path =
                             tokenizer_path.empty() ? "" : tokenizer_path,
                         .embeddings = !action_head_path.empty()};
  if (action_head_path.empty()) {
    printf("test openvla...\n");
    Openvla openvla(dinov2_path, siglip_path, proj_path, llm_path, ctx_params,
                    llm_params);
    std::vector<float> output;
    Timer t(true);
    for (size_t i = 0; i < 10; i++) {
      t.start();
      openvla.run(img_path, prompt, output);
      printf("Total time: %f\n", t.stop());
      print_vector(output);
    }
  } else {
    printf("test openvla with regression...\n");
    Timer t1(true);
    auto stats = DatasetStatistics::load(dataset_statistics_path);
    OpenvlaWithRegression openvla(dinov2_path, siglip_path, proj_path, llm_path,
                                  action_head_path, stats, ctx_params, llm_params);
    printf("Init time: %f\n", t1.stop());
    std::vector<float> output;
    Timer t(true);
    for (size_t i = 0; i < 1; i++) {
      t.start();
      openvla.run(img_path, prompt, output);
      printf("Total time: %f\n", t.stop());
    }

    printf("Final actions:\n");
    for (size_t i = 0; i < 16; ++i) {
      print_vector(
          std::vector<float>(output.begin() + i * 7,
                            output.begin() + (i + 1) * 7),
                            7);
    }
  }


  return 0;
}