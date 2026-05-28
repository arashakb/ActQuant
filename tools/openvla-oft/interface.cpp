#include "openvla.h"
#include "utils.h"
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

// Legacy pipeline class for backward compatibility with Vote model
class OpenvlaPipeline {
public:
  OpenvlaPipeline(const std::string &dinov2_model_path,
                  const std::string &siglip_model_path,
                  const std::string &proj_model_path,
                  const std::string &llm_model_path,
                  const std::string &regression_head_path = "",
                  const std::string &dataset_statistics_path = "",
                  const std::string &tokenizer_path = "",
                  const std::string &device_name = "CUDA0", int n_threads = 4,
                  int max_nodes = 2048, int ngl = 99, int n_ctx = 300)
      : use_regression_(!regression_head_path.empty()) {
    ContextParams ctx_params = {.device_name = device_name,
                                .n_threads = n_threads,
                                .max_nodes = max_nodes};
    LlmParam llm_params = {.ngl = ngl,
                           .n_ctx = n_ctx,
                           .tokenizer_path = tokenizer_path,
                           .embeddings = use_regression_};

    if (use_regression_) {
      auto stats = DatasetStatistics::load(dataset_statistics_path);
      openvla_with_reg_ = std::make_unique<OpenvlaWithRegression>(
          dinov2_model_path, siglip_model_path, proj_model_path,
          llm_model_path, regression_head_path, stats, ctx_params, llm_params);
    } else {
      openvla_ = std::make_unique<Openvla>(
          dinov2_model_path, siglip_model_path, proj_model_path,
          llm_model_path, ctx_params, llm_params);
    }
  }

  py::array_t<float> run(const std::string &image_path,
                         const std::string &instruction) {
    std::vector<float> output;

    if (use_regression_) {
      openvla_with_reg_->run(image_path, instruction, output);
    } else {
      openvla_->run(image_path, instruction, output);
    }

    ssize_t n = output.size();
    return py::array_t<float>(n, output.data());
  }

  py::array_t<float> run(py::array_t<uint8_t> image, const std::string &instruction) {
    auto buf = image.request();
    if (buf.ndim != 3) {
      throw std::runtime_error("Image must be a 3D array (height, width, channels)");
    }
    if (buf.shape[2] != 3) {
      throw std::runtime_error("Image must have 3 channels (RGB)");
    }

    int height = buf.shape[0];
    int width = buf.shape[1];
    uint8_t* img_data = static_cast<uint8_t*>(buf.ptr);

    std::vector<float> output;

    if (use_regression_) {
      openvla_with_reg_->run(img_data, width, height, instruction, output);
    } else {
      openvla_->run(img_data, width, height, instruction, output);
    }

    ssize_t n = output.size();
    return py::array_t<float>(n, output.data());
  }

private:
  bool use_regression_;
  std::unique_ptr<Openvla> openvla_;
  std::unique_ptr<OpenvlaWithRegression> openvla_with_reg_;
};

// OpenVLA-OFT Pipeline class
// Uses OFT-specific architecture with bidirectional attention for action tokens
class OpenvlaOFTPipeline {
public:
  OpenvlaOFTPipeline(const std::string &dinov2_model_path,
                     const std::string &siglip_model_path,
                     const std::string &proj_model_path,
                     const std::string &llm_model_path,
                     const std::string &action_head_path,
                     const std::string &dataset_statistics_path,
                     const std::string &tokenizer_path = "",
                     const std::string &device_name = "CUDA0", int n_threads = 4,
                     int max_nodes = 2048, int ngl = 99, int n_ctx = 512) {
    ContextParams ctx_params = {.device_name = device_name,
                                .n_threads = n_threads,
                                .max_nodes = max_nodes};
    LlmParam llm_params = {.ngl = ngl,
                           .n_ctx = n_ctx,
                           .tokenizer_path = tokenizer_path,
                           .embeddings = true};

    auto stats = DatasetStatistics::load(dataset_statistics_path);
    openvla_oft_ = std::make_unique<OpenvlaOFT>(
        dinov2_model_path, siglip_model_path, proj_model_path,
        llm_model_path, action_head_path, stats, ctx_params, llm_params);
  }

  py::array_t<float> run(const std::string &image_path,
                         const std::string &instruction) {
    std::vector<float> output;

    if (!openvla_oft_->run(image_path, instruction, output)) {
      throw std::runtime_error("OpenvlaOFT inference failed");
    }

    ssize_t n = output.size();
    return py::array_t<float>(n, output.data());
  }

  py::array_t<float> run(py::array_t<uint8_t> image, const std::string &instruction) {
    auto buf = image.request();
    if (buf.ndim != 3) {
      throw std::runtime_error("Image must be a 3D array (height, width, channels)");
    }
    if (buf.shape[2] != 3) {
      throw std::runtime_error("Image must have 3 channels (RGB)");
    }

    int height = buf.shape[0];
    int width = buf.shape[1];
    uint8_t* img_data = static_cast<uint8_t*>(buf.ptr);

    std::vector<float> output;

    if (!openvla_oft_->run(img_data, width, height, instruction, output)) {
      throw std::runtime_error("OpenvlaOFT inference failed");
    }

    ssize_t n = output.size();
    return py::array_t<float>(n, output.data());
  }

private:
  std::unique_ptr<OpenvlaOFT> openvla_oft_;
};

// OpenVLA-OFT Pipeline with Proprio support
// Uses OFT-specific architecture with proprioception input
class OpenvlaOFTPipelineWithProprio {
public:
  OpenvlaOFTPipelineWithProprio(const std::string &dinov2_model_path,
                                 const std::string &siglip_model_path,
                                 const std::string &proj_model_path,
                                 const std::string &llm_model_path,
                                 const std::string &action_head_path,
                                 const std::string &proprio_proj_path,
                                 const std::string &dataset_statistics_path,
                                 const std::string &tokenizer_path = "",
                                 const std::string &device_name = "CUDA0",
                                 int n_threads = 4, int max_nodes = 2048,
                                 int ngl = 99, int n_ctx = 512,
                                 const std::string &task_suite_name = "") {
    ContextParams ctx_params = {.device_name = device_name,
                                .n_threads = n_threads,
                                .max_nodes = max_nodes};
    LlmParam llm_params = {.ngl = ngl,
                           .n_ctx = n_ctx,
                           .tokenizer_path = tokenizer_path,
                           .embeddings = true};

    auto stats = DatasetStatistics::load(dataset_statistics_path, task_suite_name);
    openvla_oft_ = std::make_unique<OpenvlaOFT>(
        dinov2_model_path, siglip_model_path, proj_model_path,
        llm_model_path, action_head_path, proprio_proj_path,
        stats, ctx_params, llm_params);
  }

  // Run with proprio input (8-dim vector for LIBERO)
  py::array_t<float> run(const std::string &image_path,
                         const std::string &instruction,
                         const std::vector<float> &proprio) {
    std::vector<float> output;

    if (!openvla_oft_->run(image_path, instruction, proprio, output)) {
      throw std::runtime_error("OpenvlaOFT inference with proprio failed");
    }

    ssize_t n = output.size();
    return py::array_t<float>(n, output.data());
  }

  // Run with proprio input (numpy array version, single image)
  py::array_t<float> run(py::array_t<uint8_t> image,
                         const std::string &instruction,
                         const std::vector<float> &proprio) {
    auto buf = image.request();
    if (buf.ndim != 3) {
      throw std::runtime_error("Image must be a 3D array (height, width, channels)");
    }
    if (buf.shape[2] != 3) {
      throw std::runtime_error("Image must have 3 channels (RGB)");
    }

    int height = buf.shape[0];
    int width = buf.shape[1];
    uint8_t* img_data = static_cast<uint8_t*>(buf.ptr);

    std::vector<float> output;

    if (!openvla_oft_->run(img_data, width, height, instruction, proprio, output)) {
      throw std::runtime_error("OpenvlaOFT inference with proprio failed");
    }

    ssize_t n = output.size();
    return py::array_t<float>(n, output.data());
  }

  // Two-image version with proprio (full image + wrist image) - path version
  py::array_t<float> run2(const std::string &image_path1,
                          const std::string &image_path2,
                          const std::string &instruction,
                          const std::vector<float> &proprio) {
    std::vector<float> output;

    if (!openvla_oft_->run(image_path1, image_path2, instruction, proprio, output)) {
      throw std::runtime_error("OpenvlaOFT inference with 2 images failed");
    }

    ssize_t n = output.size();
    return py::array_t<float>(n, output.data());
  }

  // Two-image version with proprio (full image + wrist image) - numpy array version
  py::array_t<float> run2(py::array_t<uint8_t> image1,
                          py::array_t<uint8_t> image2,
                          const std::string &instruction,
                          const std::vector<float> &proprio) {
    auto buf1 = image1.request();
    auto buf2 = image2.request();

    if (buf1.ndim != 3 || buf2.ndim != 3) {
      throw std::runtime_error("Images must be 3D arrays (height, width, channels)");
    }
    if (buf1.shape[2] != 3 || buf2.shape[2] != 3) {
      throw std::runtime_error("Images must have 3 channels (RGB)");
    }

    int height1 = buf1.shape[0];
    int width1 = buf1.shape[1];
    uint8_t* img_data1 = static_cast<uint8_t*>(buf1.ptr);

    int height2 = buf2.shape[0];
    int width2 = buf2.shape[1];
    uint8_t* img_data2 = static_cast<uint8_t*>(buf2.ptr);

    std::vector<float> output;

    if (!openvla_oft_->run(img_data1, width1, height1,
                           img_data2, width2, height2,
                           instruction, proprio, output)) {
      throw std::runtime_error("OpenvlaOFT inference with 2 images failed");
    }

    ssize_t n = output.size();
    return py::array_t<float>(n, output.data());
  }

  bool has_proprio_projector() const {
    return openvla_oft_->has_proprio_projector();
  }

private:
  std::unique_ptr<OpenvlaOFT> openvla_oft_;
};

PYBIND11_MODULE(openvla_oft, m) {
  m.doc() = "Python binding for OpenVLA-OFT C++ pipeline";

  // Legacy pipeline for backward compatibility
  py::class_<OpenvlaPipeline>(m, "OpenvlaPipeline")
      .def(py::init<const std::string &, const std::string &,
                    const std::string &, const std::string &,
                    const std::string &, const std::string &,
                    const std::string &, const std::string &,
                    int, int, int, int>(),
           py::arg("dinov2_model_path"), py::arg("siglip_model_path"),
           py::arg("proj_model_path"), py::arg("llm_model_path"),
           py::arg("regression_head_path") = "",
           py::arg("dataset_statistics_path") = "",
           py::arg("tokenizer_path") = "", py::arg("device_name") = "CUDA0",
           py::arg("n_threads") = 4, py::arg("max_nodes") = 2048,
           py::arg("ngl") = 99, py::arg("n_ctx") = 300)
      .def("run",
           py::overload_cast<const std::string &, const std::string &>(
               &OpenvlaPipeline::run),
           py::arg("image_path"), py::arg("instruction"),
           "Run Vote model on a given image path and instruction.")
      .def("run",
           py::overload_cast<py::array_t<uint8_t>, const std::string &>(
               &OpenvlaPipeline::run),
           py::arg("image"), py::arg("instruction"),
           "Run Vote model on a given image array (HxWx3 uint8) and instruction.");

  // OpenVLA-OFT Pipeline
  py::class_<OpenvlaOFTPipeline>(m, "OpenvlaOFTPipeline")
      .def(py::init<const std::string &, const std::string &,
                    const std::string &, const std::string &,
                    const std::string &, const std::string &,
                    const std::string &, const std::string &,
                    int, int, int, int>(),
           py::arg("dinov2_model_path"), py::arg("siglip_model_path"),
           py::arg("proj_model_path"), py::arg("llm_model_path"),
           py::arg("action_head_path"), py::arg("dataset_statistics_path"),
           py::arg("tokenizer_path") = "", py::arg("device_name") = "CUDA0",
           py::arg("n_threads") = 4, py::arg("max_nodes") = 2048,
           py::arg("ngl") = 99, py::arg("n_ctx") = 512)
      .def("run",
           py::overload_cast<const std::string &, const std::string &>(
               &OpenvlaOFTPipeline::run),
           py::arg("image_path"), py::arg("instruction"),
           "Run OpenVLA-OFT model on a given image path and instruction. "
           "Returns 56 actions (8 chunks x 7 dims) using OFT architecture.")
      .def("run",
           py::overload_cast<py::array_t<uint8_t>, const std::string &>(
               &OpenvlaOFTPipeline::run),
           py::arg("image"), py::arg("instruction"),
           "Run OpenVLA-OFT model on a given image array (HxWx3 uint8) and instruction. "
           "Returns 56 actions (8 chunks x 7 dims) using OFT architecture.");

  // OpenVLA-OFT Pipeline with Proprio support
  py::class_<OpenvlaOFTPipelineWithProprio>(m, "OpenvlaOFTPipelineWithProprio")
      .def(py::init<const std::string &, const std::string &,
                    const std::string &, const std::string &,
                    const std::string &, const std::string &,
                    const std::string &, const std::string &,
                    const std::string &,
                    int, int, int, int, const std::string &>(),
           py::arg("dinov2_model_path"), py::arg("siglip_model_path"),
           py::arg("proj_model_path"), py::arg("llm_model_path"),
           py::arg("action_head_path"), py::arg("proprio_proj_path"),
           py::arg("dataset_statistics_path"),
           py::arg("tokenizer_path") = "", py::arg("device_name") = "CUDA0",
           py::arg("n_threads") = 4, py::arg("max_nodes") = 2048,
           py::arg("ngl") = 99, py::arg("n_ctx") = 512,
           py::arg("task_suite_name") = "")
      .def("run",
           py::overload_cast<const std::string &, const std::string &,
                             const std::vector<float> &>(
               &OpenvlaOFTPipelineWithProprio::run),
           py::arg("image_path"), py::arg("instruction"), py::arg("proprio"),
           "Run OpenVLA-OFT model with proprio on a given image path, instruction, "
           "and proprio state (8-dim vector for LIBERO). "
           "Returns 56 actions (8 chunks x 7 dims).")
      .def("run",
           py::overload_cast<py::array_t<uint8_t>, const std::string &,
                             const std::vector<float> &>(
               &OpenvlaOFTPipelineWithProprio::run),
           py::arg("image"), py::arg("instruction"), py::arg("proprio"),
           "Run OpenVLA-OFT model with proprio on a given image array (HxWx3 uint8), "
           "instruction, and proprio state (8-dim vector for LIBERO). "
           "Returns 56 actions (8 chunks x 7 dims).")
      .def("run2",
           py::overload_cast<const std::string &, const std::string &,
                             const std::string &, const std::vector<float> &>(
               &OpenvlaOFTPipelineWithProprio::run2),
           py::arg("image_path1"), py::arg("image_path2"),
           py::arg("instruction"), py::arg("proprio"),
           "Run OpenVLA-OFT model with 2 images (full + wrist) on given image paths, "
           "instruction, and proprio state (8-dim vector for LIBERO). "
           "Returns 56 actions (8 chunks x 7 dims). This is the recommended method for LIBERO evaluation.")
      .def("run2",
           py::overload_cast<py::array_t<uint8_t>, py::array_t<uint8_t>,
                             const std::string &, const std::vector<float> &>(
               &OpenvlaOFTPipelineWithProprio::run2),
           py::arg("image1"), py::arg("image2"),
           py::arg("instruction"), py::arg("proprio"),
           "Run OpenVLA-OFT model with 2 images (full + wrist) on given image arrays (HxWx3 uint8), "
           "instruction, and proprio state (8-dim vector for LIBERO). "
           "Returns 56 actions (8 chunks x 7 dims). This is the recommended method for LIBERO evaluation.")
      .def("has_proprio_projector", &OpenvlaOFTPipelineWithProprio::has_proprio_projector,
           "Check if proprio projector is initialized");
}
