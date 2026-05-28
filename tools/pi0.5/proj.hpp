#pragma once

#include "infer_session.hpp"
#include "model_defs.h"

class Pi05Projector {
public:
  Pi05Projector() = default;
  Pi05Projector(const std::string &model_path, const ContextParams &params)
      : model_(model_path, params) {}

  bool run(const std::vector<float> &vision_features, std::vector<float> &out) {
    model_.set_input("vision_features", vision_features);
    return model_.run(out);
  }

private:
  InferenceSession<Pi05ProjectorModel> model_;
};
