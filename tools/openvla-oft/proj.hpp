#pragma once

#include "infer_session.hpp"
#include "model_defs.h"

class Projector {
public:
  Projector() = default;
  Projector(const std::string &model_path, const ContextParams &params)
      : model_(model_path, params) {}

  bool run(const std::vector<float> &inp_dinov2,
           const std::vector<float> &inp_siglip, std::vector<float> &out) {
    model_.set_input("dinov2_feat", inp_dinov2);
    model_.set_input("siglip_feat", inp_siglip);
    if (!model_.run(out)) {
      return false;
    }
    return true;
  }

private:
  InferenceSession<ProjectorModel> model_;
};