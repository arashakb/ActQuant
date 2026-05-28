#pragma once

#include "infer_session.hpp"
#include "model_defs.h"
#include "utils.h"

class Vit {
public:
  Vit() = default;
  Vit(const std::string &model_path, const ContextParams &params)
      : model_(model_path, params) {}

  bool run(const std::vector<float> &pixel_values, std::vector<float> &out) {
    model_.set_input("inp_raw", pixel_values);
    return model_.run(out);
  }

  bool run(const std::string &img_path, std::vector<float> &out) {
    const VisionTransformerModel &model = model_.get_model();
    const std::vector<float> &mean = model.hparams.image_mean;
    const std::vector<float> &std = model.hparams.image_std;
    const int target_size = model.hparams.image_size;

    std::vector<float> pixel_values;
    resize_normalize(img_path, target_size, target_size, mean, std,
                     pixel_values, true);

    return run(pixel_values, out);
  }

  bool run(const uint8_t *img_data, int width, int height,
           std::vector<float> &out) {
    const VisionTransformerModel &model = model_.get_model();
    const std::vector<float> &mean = model.hparams.image_mean;
    const std::vector<float> &std = model.hparams.image_std;
    const int target_size = model.hparams.image_size;

    std::vector<float> pixel_values;
    resize_normalize(img_data, width, height, target_size, target_size, mean,
                     std, pixel_values, true);
    return run(pixel_values, out);
  }

private:

  InferenceSession<VisionTransformerModel> model_;
};