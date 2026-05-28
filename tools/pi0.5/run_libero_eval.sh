#!/bin/bash
# Multi-GPU LIBERO evaluation for Pi0.5
# Automatically detects GPUs, starts one policy server per GPU, runs eval in parallel.
#
# Usage:
#   ./run_libero_eval.sh [task_suite] [num_trials] [replan_steps] [num_gpus] [base_port] [model_dir]
#
# Examples:
#   ./run_libero_eval.sh libero_spatial 50 5
#   ./run_libero_eval.sh libero_spatial 50 5 8 8000 /path/to/openpi/pi05_gguf_fp16

set -e

# ============================================================
# Config
# ============================================================
TASK_SUITE="${1:-libero_spatial}"
NUM_TRIALS="${2:-50}"
REPLAN_STEPS="${3:-5}"
NUM_GPUS_OVERRIDE="${4:-0}"  # 0 = auto-detect
BASE_PORT="${5:-8000}"
MODEL_DIR="${6:-/path/to/openpi/pi05_gguf}"
FLOW_STEPS=10
ROLLOUT_DIR="/path/to/rollouts/${TASK_SUITE}"
PI05_SERVE="/path/to/ActQuant/tools/pi0.5/serve_policy.py"
LIBERO_EVAL="/path/to/openpi/examples/libero/main.py"
PYTHON="/path/to/openpi/.venv/bin/python3"           # Python 3.11 — C++ server (pi05.so)
PYTHON_CLIENT="${HOME}/miniconda3/envs/openpi-libero/bin/python3.8"  # Python 3.8 — eval client (bddl, robosuite)
PYTHONPATH_EXTRA="/path/to/ActQuant/build_openpi/bin:/path/to/openpi/third_party/libero"
PYTHONPATH_CLIENT="/path/to/openpi/third_party/libero:/path/to/openpi/packages/openpi-client/src"

# ============================================================
# Detect GPUs
# ============================================================
TOTAL_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$TOTAL_GPUS" -eq 0 ]; then
    echo "ERROR: No GPUs found."
    exit 1
fi

if [ "$NUM_GPUS_OVERRIDE" -gt 0 ]; then
    if [ "$NUM_GPUS_OVERRIDE" -gt "$TOTAL_GPUS" ]; then
        echo "ERROR: Requested ${NUM_GPUS_OVERRIDE} GPUs but only ${TOTAL_GPUS} available."
        exit 1
    fi
    NUM_GPUS="$NUM_GPUS_OVERRIDE"
else
    NUM_GPUS="$TOTAL_GPUS"
fi
echo "============================================================"
echo "  Pi0.5 Multi-GPU LIBERO Evaluation"
echo "============================================================"
echo "  Task suite:    ${TASK_SUITE}"
echo "  Trials/task:   ${NUM_TRIALS}"
echo "  Replan steps:  ${REPLAN_STEPS}"
echo "  Flow steps:    ${FLOW_STEPS}"
echo "  GPUs:          ${NUM_GPUS}"
echo "  Base port:     ${BASE_PORT}"
echo "  Model:         ${MODEL_DIR}"
echo "  Output:        ${ROLLOUT_DIR}"
echo "============================================================"

mkdir -p "${ROLLOUT_DIR}"

# ============================================================
# Start one policy server per GPU
# ============================================================
SERVER_PIDS=()
PORTS=()

cleanup() {
    echo ""
    echo "Stopping policy servers..."
    for pid in "${SERVER_PIDS[@]}"; do
        kill "$pid" 2>/dev/null && echo "  Killed server PID $pid"
    done
}
trap cleanup EXIT INT TERM

echo ""
echo "Starting policy servers..."
for i in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    LOG_FILE="${ROLLOUT_DIR}/server_gpu${i}.log"

    CUDA_VISIBLE_DEVICES="${i}" \
    PYTHONPATH="${PYTHONPATH_EXTRA}" \
    "${PYTHON}" "${PI05_SERVE}" \
        --model-dir "${MODEL_DIR}" \
        --port "${PORT}" \
        --device "CUDA0" \
        --steps "${FLOW_STEPS}" \
        > "${LOG_FILE}" 2>&1 &

    SERVER_PIDS+=($!)
    PORTS+=("${PORT}")
    echo "  GPU ${i} -> port ${PORT} (PID ${!}, log: ${LOG_FILE})"
done

# ============================================================
# Wait for all servers to be ready
# ============================================================
echo ""
echo "Waiting for servers to be ready..."
for i in $(seq 0 $((NUM_GPUS - 1))); do
    PORT="${PORTS[$i]}"
    for attempt in $(seq 1 60); do
        if curl -sf "http://0.0.0.0:${PORT}/healthz" > /dev/null 2>&1; then
            echo "  GPU ${i} port ${PORT} ready"
            break
        fi
        if [ "$attempt" -eq 60 ]; then
            echo "  ERROR: Server on port ${PORT} did not start in time."
            echo "  Check log: ${ROLLOUT_DIR}/server_gpu${i}.log"
            exit 1
        fi
        sleep 2
    done
done

# ============================================================
# Run evaluation (tyro List[int] takes space-separated values)
# ============================================================
echo ""
echo "Starting evaluation with ports ${PORTS[*]}..."
echo ""

PYTHONPATH="${PYTHONPATH_CLIENT}" \
"${PYTHON_CLIENT}" "${LIBERO_EVAL}" \
    --args.task-suite-name "${TASK_SUITE}" \
    --args.num-trials-per-task "${NUM_TRIALS}" \
    --args.replan-steps "${REPLAN_STEPS}" \
    --args.ports "${PORTS[@]}" \
    --args.video-out-path "${ROLLOUT_DIR}" \
    2>&1 | tee "${ROLLOUT_DIR}/eval.log"

echo ""
echo "Done. Results saved to ${ROLLOUT_DIR}/eval.log"
