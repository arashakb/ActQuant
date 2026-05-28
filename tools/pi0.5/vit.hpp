#pragma once

#include "infer_session.hpp"
#include "model_defs.h"
#include "utils.h"

class Pi05Vit {
public:
  Pi05Vit() = default;
  Pi05Vit(const std::string &model_path, const ContextParams &params)
      : model_(model_path, params) {}

  bool run(const std::vector<float> &pixel_values, std::vector<float> &out) {
    model_.set_input("inp_raw", pixel_values);
    return model_.run(out);
  }

  bool run(const std::string &img_path, std::vector<float> &out) {
    const Pi05VisionModel &model = model_.get_model();
    const std::vector<float> mean = {0.5f, 0.5f, 0.5f};  // SigLIP normalization
    const std::vector<float> std = {0.5f, 0.5f, 0.5f};
    const int target_size = model.hparams.image_size;

    std::vector<float> pixel_values;
    // Use to_chw=true: GGML conv2d expects CHW format (channels planar)
    if (!resize_normalize(img_path, target_size, target_size, mean, std,
                          pixel_values, true)) {
      return false;
    }
    return run(pixel_values, out);
  }

  bool run(const uint8_t *img_data, int width, int height,
           std::vector<float> &out) {
    const Pi05VisionModel &model = model_.get_model();
    const std::vector<float> mean = {0.5f, 0.5f, 0.5f};
    const std::vector<float> std = {0.5f, 0.5f, 0.5f};
    const int target_size = model.hparams.image_size;

    std::vector<float> pixel_values;
    // Use to_chw=true: GGML conv2d expects CHW format (channels planar)
    if (!resize_normalize(img_data, width, height, target_size, target_size,
                          mean, std, pixel_values, true)) {
      return false;
    }
    return run(pixel_values, out);
  }

  Pi05VisionModel &get_model() { return model_.get_model(); }

private:
  InferenceSession<Pi05VisionModel> model_;
};
