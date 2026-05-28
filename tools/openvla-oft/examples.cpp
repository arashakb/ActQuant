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

// std::string base_dir = "/mount/weights/openvla/";
// std::string base_dir = "/mount/weights/vote_model/";
// std::string base_dir = "/Users/wjr/Downloads/vote_model/";
std::string base_dir = "/home/ubuntu/data_1/chenxing/spatial/";

static void test_tokenizer() {
  std::unique_ptr<HfTokenizer> tokenizer = create_tokenizer(
      "/home/ubuntu/data_1/chenxing/spatial/tokenizer.json", 1024);
  std::string prompt = "In: What action should the robot take to pick up the black bowl between the plate and the ramekin and place it on the plate";
  std::vector<int> tokens = tokenizer->encode(prompt, true);
  std::cout << "Tokens: ";
  for (auto token : tokens) {
    std::cout << token << " ";
  }
  std::cout << std::endl;
  std::cout << tokenizer->decode(tokens) << std::endl;
  tokens = tokenizer->encode(prompt, false);
  std::cout << "Tokens: ";
  for (auto token : tokens) {
    std::cout << token << " ";
  }
  std::cout << std::endl;
  std::cout << tokenizer->decode(tokens) << std::endl;
}

static void test_ctx_manager() {
  { ContextManager manager; }
  { ContextManager manager("Metal", 1, 2048); }
  {
    ContextParams params = {.device_name = "Metal",
                            .n_threads = 1,
                            .max_nodes = 2048,
                            .verbosity = GGML_LOG_LEVEL_DEBUG};
    ContextManager manager(params);
  }
}

static void test_model_loader() {
  ModelLoader loader(base_dir + "Qwen2.5-VL-3B-Instruct-Q8_0.gguf");
}

static void test_infer_session() {
  ContextParams params = {.device_name = "Metal",
                          .n_threads = 1,
                          .max_nodes = 2048,
                          .verbosity = GGML_LOG_LEVEL_DEBUG};
  InferenceSession<ProjectorModel> session(base_dir + "proj.gguf", params);
  std::vector<float> inp_dinov2(256 * 1024, 1.f);
  std::vector<float> inp_siglip(256 * 1152, 1.f);
  session.set_input("dinov2_feat", inp_dinov2);
  session.set_input("siglip_feat", inp_siglip);
  std::vector<float> output;
  session.run(output);
}

static void test_proj_inference() {
  {
    ContextParams params = {.device_name = "CUDA0",
                            .n_threads = 1,
                            .max_nodes = 2048,
                            .verbosity = GGML_LOG_LEVEL_DEBUG};
    Projector proj(base_dir + "proj.gguf", params);
    std::vector<float> inp_dinov2(256 * 1024, 1.f);
    std::vector<float> inp_siglip(256 * 1152, 1.f);
    std::vector<float> output;
    proj.run(inp_dinov2, inp_siglip, output);
    proj.run(inp_dinov2, inp_siglip, output);
  }
}

static void test_vit_inference() {
  {
    ContextParams params = {.device_name = "Metal",
                            .n_threads = 1,
                            .max_nodes = 2048,
                            .verbosity = GGML_LOG_LEVEL_DEBUG};
    Vit vit(base_dir + "dinov2.gguf", params);
    std::vector<float> inp(224 * 224 * 3, 1.f);
    std::vector<float> output;
    vit.run(inp, output);
    vit.run(inp, output);
    vit.run(base_dir + "pokemon.jpeg", output);
    vit.run(base_dir + "pokemon.jpeg", output);
  }
  {
    ContextParams params = {.device_name = "Metal",
                            .n_threads = 1,
                            .max_nodes = 2048,
                            .verbosity = GGML_LOG_LEVEL_DEBUG};
    Vit vit(base_dir + "siglip.gguf", params);
    std::vector<float> inp(224 * 224 * 3, 1.f);
    std::vector<float> output;
    vit.run(inp, output);
    vit.run(inp, output);
    vit.run(base_dir + "pokemon.jpeg", output);
    vit.run(base_dir + "pokemon.jpeg", output);
  }
}

static void test_openvla_projector() {
  ContextParams params = {.device_name = "Metal",
                          .n_threads = 1,
                          .max_nodes = 2048,
                          .verbosity = GGML_LOG_LEVEL_DEBUG};
  OpenvlaProjector openvla_proj(base_dir + "dinov2.gguf",
                                base_dir + "siglip.gguf",
                                base_dir + "proj.gguf", params);
  std::vector<float> out;
  openvla_proj.run(base_dir + "2.png", out);
  openvla_proj.run(base_dir + "2.png", out);
}

static void test_llm() {
  {
    std::string model_path = base_dir + "llama2_fp16.gguf";
    // std::string model_path = base_dir +
    // "Qwen2.5-VL-3B-Instruct-Q8_0.gguf";
    // std::string model_path = "/Users/wjr/weights/Qwen3-0.6B-Q8_0.gguf";
    LlmParam llm_params = {.ngl = 99,
                           .n_ctx = 2048,
                           .tokenizer_path = base_dir + "tokenizer.json",
                           .embeddings = true};
    Llm llm;
    llm.load_model(model_path, llm_params);
    std::vector<llama_token> generated_tokens;
    std::vector<float> embd;

    llm.generate("pick up the cup", "", generated_tokens, embd);
  }
  {
    std::string model_path = base_dir + "llm_fp16.gguf";
    // std::string model_path = base_dir +
    // "Qwen2.5-VL-3B-Instruct-Q8_0.gguf";
    // std::string model_path = "/Users/wjr/weights/Qwen3-0.6B-Q8_0.gguf";
    LlmParam llm_params = {.ngl = 99,
                           .n_ctx = 2048,
                           .tokenizer_path = base_dir + "tokenizer.json",
                           .embeddings = true};
    OpenvlaLlm llm;
    llm.load_model(model_path, llm_params);
    std::vector<llama_token> generated_tokens;
    std::vector<float> embd;

    llm.generate("pick up the cup", nullptr, 0, generated_tokens, embd, false);
  }
}

static void test_l1_regression() {
  std::string model_path = base_dir + "llm_q8_0.gguf";
  ContextParams ctx_params = {.device_name = "CUDA0",
                              .n_threads = 4,
                              .max_nodes = 2048,
                              .verbosity = GGML_LOG_LEVEL_DEBUG};
  InferenceSession<L1RegressionActionHeadmulmlpk> model(model_path,
                                                            ctx_params);
  std::vector<float> inp(2048, 1.f);
  model.set_input("inp_raw", inp);
  std::vector<float> out;
  model.run(out);
  model.run(out);
}

static void test_openvla() {
  ContextParams ctx_params = {.device_name = "CUDA0",
                              .n_threads = 4,
                              .max_nodes = 2048,
                              .verbosity = GGML_LOG_LEVEL_DEBUG};
  LlmParam llm_params = {
      .ngl = 99,
      .n_ctx = 300,
      .tokenizer_path = "/Users/wjr/weights/openvla/openvla-7b/tokenizer.json",
      .embeddings = false};
  Openvla openvla(base_dir + "dinov2.gguf", base_dir + "siglip.gguf",
                  base_dir + "proj.gguf", "ttt/llm_q4_new.gguf", ctx_params,
                  llm_params);

  std::vector<float> output;
  Timer t(true);
  for (size_t i = 0; i < 10; i++) {
    t.start();
    // In: What action should the robot take to pick up the black bowl between
    // the plate and the ramekin and place it on the plate?\nOut:
    // openvla.run(base_dir + "2.png",
    //             "pick up the black bowl between the plate and the ramekin and
    //             " "place it on the plate?", output);
    openvla.run(base_dir + "pokemon.jpeg", "move pokman to the left", output);
    printf("Total time: %f\n", t.stop());
    print_vector(output);
  }
}

static void test_vote() {
  ContextParams ctx_params = {.device_name = "CUDA0",
                              .n_threads = 4,
                              .max_nodes = 2048,
                              .verbosity = GGML_LOG_LEVEL_DEBUG};
  LlmParam llm_params = {
      .ngl = 99,
      .n_ctx = 300,
      // .tokenizer_path =
      // "/Users/wjr/weights/openvla/openvla-7b/tokenizer.json"
      .embeddings = true};
  OpenvlaWithRegression openvla(
      base_dir + "dinov2.gguf", base_dir + "siglip.gguf",
      base_dir + "proj.gguf", base_dir + "llm_fp16.gguf",
      base_dir + "action_head.gguf", ctx_params, llm_params);

  std::vector<float> output;
  Timer t(true);
  for (size_t i = 0; i < 10; i++) {
    t.start();
    // In: What action should the robot take to pick up the black bowl between
    // the plate and the ramekin and place it on the plate?\nOut:
    std::string prompt =
        "pick up the black bowl between the plate and the ramekin and "
        "place it on the plate";
    printf("prompt: %s\n", prompt.c_str());
    openvla.run(base_dir + "2.png",
                "pick up the black bowl between the plate and the ramekin and "
                "place it on the plate",
                output);
    printf("Total time: %f\n", t.stop());
    print_vector(output);
  }
}

int main() {
  // test_tokenizer();

  // test_ctx_manager();

  // // test_model_loader();

  // test_infer_session();
  // test_proj_inference();
  // test_vit_inference();
  // test_openvla_projector();

  test_llm();

  // test_openvla();
  // test_vote();

  return 0;
}