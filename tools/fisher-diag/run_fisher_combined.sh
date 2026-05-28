#!/bin/bash
python tools/fisher-diag/compute_fisher.py \
    --checkpoint /path/to/openvla-oft-checkpoints/oft_combined \
    --calib-data \
        /path/to/openvla-oft/calib_data/goal_20.bin \
        /path/to/openvla-oft/calib_data/spatial_20.bin \
        /path/to/openvla-oft/calib_data/object_20.bin \
        /path/to/openvla-oft/calib_data/long_50.bin \
    --calib-targets \
        /path/to/openvla-oft/calib_data/goal_20_targets.npy \
        /path/to/openvla-oft/calib_data/spatial_20_targets.npy \
        /path/to/openvla-oft/calib_data/object_20_targets.npy \
        /path/to/openvla-oft/calib_data/long_50_targets.npy \
    --output tools/fisher-diag/fisher_diag_combined_all.gguf \
    --num-gpus 8 \
    --batch-size 8



python experiments/robot/libero/run_libero_eval_combined_cpp.py \
      --multi-gpu \
      --gpus 0,1,2,3,4,5,6,7 \
      --task_suite_name libero_object \
      --llm_gguf_name llm_iq2_xs_fisher_combined.gguf \
      --dinov2_gguf_name dinov2_q4_k_m.gguf \
      --siglip_gguf_name siglip_q4_k_m_padded.gguf 




python run_libero_eval_combined_cpp.py --multi-gpu \
      --task_suite_name libero_10 \
      --llm_gguf_name llm_iq2_xs_hsic_sens_s01_combined_fisher.gguf\
      --dinov2_gguf_name dinov2_q4_k_m.gguf \
      --siglip_gguf_name siglip_q4_k_m_padded.gguf \
      --gpus 0,1,2,3,4,5,6,7