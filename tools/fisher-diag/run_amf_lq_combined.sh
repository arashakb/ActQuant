#!/usr/bin/env bash
# Hybrid NLL + Action Fisher for oft_combined
# alpha=0.5: equal blend of NLL (action token cross-entropy) and L1 action loss
# Uses spatial_20, object_20, goal_20, long_50 calibration data
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALIB_DIR="/path/to/openvla-oft/calib_data"
CHECKPOINT="/path/to/openvla-oft-checkpoints/oft_combined"
ACTION_HEAD="${CHECKPOINT}/action_head--300000_checkpoint.pt"
OUTPUT="${SCRIPT_DIR}/af_hybrid_combined.gguf"

 python "${SCRIPT_DIR}/compute_amf_lq.py" \
    --checkpoint        "${CHECKPOINT}" \
    --action-head-checkpoint "${ACTION_HEAD}" \
    --calib-data \
        "${CALIB_DIR}/spatial_20.bin" \
        "${CALIB_DIR}/object_20.bin" \
        "${CALIB_DIR}/goal_20.bin" \
        "${CALIB_DIR}/long_50.bin" \
    --action-loss l1 \
    --q 1 \
    --alpha 0.5 \
    --batch-size 8 \
    --num-gpus 8 \
    --output "${OUTPUT}"
