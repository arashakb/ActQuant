// Pi0.5 Python binding using pybind11
// Exposes Pi05 inference engine to Python for easy integration with OpenPI's WebSocket server

#include "pi05.h"
#include "utils.h"
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

class Pi05Pipeline {
   public:
    Pi05Pipeline(const std::string& model_path,
                 const std::string& tokenizer_path,
                 const std::string& device_name = "CPU",
                 int n_threads = 4,
                 int num_flow_steps = 10,
                 const std::string& ode_profile_path = "") {
        Pi05Params params;
        params.model_path = model_path;
        params.tokenizer_path = tokenizer_path;
        params.n_threads = n_threads;
        params.num_flow_steps = num_flow_steps;
        params.device = device_name;
        params.ode_profile_path = ode_profile_path;

        pi05_ = std::make_unique<Pi05>(params);
    }

    // Run inference with image path
    py::array_t<float> run(const std::string& image_path,
                           const std::string& prompt) {
        std::vector<float> output;
        bool success = pi05_->run(image_path, prompt, output);
        if (!success) {
            throw std::runtime_error("Inference failed");
        }
        ssize_t n = output.size();
        return py::array_t<float>(n, output.data());
    }

    // Run inference with numpy image array (HxWx3 uint8)
    py::array_t<float> run(py::array_t<uint8_t> image,
                           const std::string& prompt) {
        auto buf = image.request();
        if (buf.ndim != 3) {
            throw std::runtime_error(
                "Image must be a 3D array (height, width, channels)");
        }
        if (buf.shape[2] != 3) {
            throw std::runtime_error("Image must have 3 channels (RGB)");
        }

        int height = buf.shape[0];
        int width = buf.shape[1];
        uint8_t* img_data = static_cast<uint8_t*>(buf.ptr);

        std::vector<float> output;
        bool success = pi05_->run(img_data, width, height, prompt, output);
        if (!success) {
            throw std::runtime_error("Inference failed");
        }
        ssize_t n = output.size();
        return py::array_t<float>(n, output.data());
    }

    // Run inference with two images (base + wrist camera)
    py::array_t<float> run_multi(py::array_t<uint8_t> base_image,
                                  py::array_t<uint8_t> wrist_image,
                                  const std::string& prompt) {
        auto base_buf = base_image.request();
        auto wrist_buf = wrist_image.request();

        if (base_buf.ndim != 3 || wrist_buf.ndim != 3) {
            throw std::runtime_error("Images must be 3D arrays");
        }

        ImageInputs images;
        images.base_data = static_cast<uint8_t*>(base_buf.ptr);
        images.width = base_buf.shape[1];
        images.height = base_buf.shape[0];
        images.wrist_data = static_cast<uint8_t*>(wrist_buf.ptr);

        std::vector<float> output;
        bool success = pi05_->run(images, prompt, output);
        if (!success) {
            throw std::runtime_error("Inference failed");
        }
        ssize_t n = output.size();
        return py::array_t<float>(n, output.data());
    }

    // Get action dimensions from model config
    int get_action_horizon() const { return pi05_->config().action.action_horizon; }
    int get_action_dim() const { return pi05_->config().action.action_dim; }

   private:
    std::unique_ptr<Pi05> pi05_;
};

PYBIND11_MODULE(pi05, m) {
    m.doc() = "Python binding for Pi0.5 C++ inference pipeline";

    py::class_<Pi05Pipeline>(m, "Pi05Pipeline")
        .def(py::init<const std::string&, const std::string&,
                      const std::string&, int, int, const std::string&>(),
             py::arg("model_path"),
             py::arg("tokenizer_path"),
             py::arg("device_name") = "CPU",
             py::arg("n_threads") = 4,
             py::arg("num_flow_steps") = 10,
             py::arg("ode_profile_path") = "",
             "Initialize Pi0.5 with unified model (single GGUF file)")
        .def("run",
             py::overload_cast<const std::string&, const std::string&>(
                 &Pi05Pipeline::run),
             py::arg("image_path"),
             py::arg("prompt"),
             "Run inference on image file path")
        .def("run",
             py::overload_cast<py::array_t<uint8_t>, const std::string&>(
                 &Pi05Pipeline::run),
             py::arg("image"),
             py::arg("prompt"),
             "Run inference on numpy image array (HxWx3 uint8)")
        .def("run_multi",
             &Pi05Pipeline::run_multi,
             py::arg("base_image"),
             py::arg("wrist_image"),
             py::arg("prompt"),
             "Run inference with two images (base + wrist camera)")
        .def_property_readonly("action_horizon",
                               &Pi05Pipeline::get_action_horizon)
        .def_property_readonly("action_dim", &Pi05Pipeline::get_action_dim);
}
