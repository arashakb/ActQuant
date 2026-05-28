#!/bin/bash
# run_imatrix_pi05.sh
#
# Full imatrix quantization pipeline for Pi0.5 LLM (PaliGemma backbone).
# Mirrors tools/imatrix/run_imatrix_combined_10.sh for openvla-oft.
#
# Pipeline:
#   1. Export standalone pali_llm_bf16.gguf (skipped if already exists)
#   2. Run llama-imatrix on each .bin calibration file in parallel (one per GPU)
#   3. Merge all imatrix files into one combined imatrix
#   4. llama-quantize --imatrix → pali_llm_<QUANT>.gguf
#   5. merge_pi05_llm.py → pi05_<QUANT>.gguf (patch into full pi05.gguf)
#
# Calibration files must be in OPENVLA_CALIB binary format with hidden_dim=2048.
# Use tools/imatrix/merge_calib_data.py to prepare / slice / merge .bin files.
#
# Requirements:
#   - pi05_fp16.gguf must exist in OUTPUT_DIR (export with export_pi05.py first)
#   - model.safetensors must exist in MODEL_PYTORCH_DIR
#
# Usage:
#   ./run_imatrix_pi05.sh <QUANT_TYPE> <CALIB_DIR> <MODEL_PYTORCH_DIR> <OUTPUT_DIR> [NUM_GPUS]
#
# Examples:
#   ./run_imatrix_pi05.sh IQ2_XS /data/calib /path/to/openpi/pi05_libero_base_pytorch \
#       /path/to/openpi/pi05_libero_base_gguf
#
#   ./run_imatrix_pi05.sh IQ3_S  /data/calib /path/to/openpi/pi05_libero_base_pytorch \
#       /path/to/openpi/pi05_libero_base_gguf 4

set -e

QUANT_TYPE="${1:?Usage: $0 <QUANT_TYPE> <CALIB_DIR> <MODEL_PYTORCH_DIR> <OUTPUT_DIR> [NUM_GPUS]}"
CALIB_DIR="${2:?Missing CALIB_DIR}"
MODEL_PYTORCH_DIR="${3:?Missing MODEL_PYTORCH_DIR}"
OUTPUT_DIR="${4:?Missing OUTPUT_DIR}"
NUM_GPUS_OVERRIDE="${5:-0}"   # 0 = auto-detect

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BINARY_DIR="${REPO_ROOT}/build_openpi/bin"
PYTHON="${PYTHON:-/path/to/openpi/.venv/bin/python3}"
PYTHONPATH_EXTRA="${REPO_ROOT}/build_openpi/bin:${REPO_ROOT}/gguf-py"

LLM_BF16="${OUTPUT_DIR}/pali_llm_bf16.gguf"
PI05_FP16="${OUTPUT_DIR}/pi05.gguf"          # FP16 base (must already exist)
QUANT_LOWER=$(echo "$QUANT_TYPE" | tr '[:upper:]' '[:lower:]' | tr -d '_')
LLM_QUANT="${OUTPUT_DIR}/pali_llm_${QUANT_LOWER}.gguf"
PI05_QUANT="${OUTPUT_DIR}/pi05_${QUANT_LOWER}.gguf"
TMP_DIR="/tmp/imatrix_pi05_$$"
IMATRIX_OUT="${OUTPUT_DIR}/imatrix_combined.gguf"

# ── GPU detection ─────────────────────────────────────────────────────────────
TOTAL_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || echo 0)
if [ "$TOTAL_GPUS" -eq 0 ]; then
    echo "ERROR: No GPUs detected."
    exit 1
fi
if [ "$NUM_GPUS_OVERRIDE" -gt 0 ]; then
    NUM_GPUS="$NUM_GPUS_OVERRIDE"
else
    NUM_GPUS="$TOTAL_GPUS"
fi

# ── Calibration files ─────────────────────────────────────────────────────────
CALIB_FILES=("${CALIB_DIR}"/*.bin)
if [ ${#CALIB_FILES[@]} -eq 0 ] || [ ! -f "${CALIB_FILES[0]}" ]; then
    echo "ERROR: No .bin calibration files found in ${CALIB_DIR}"
    echo "  Expected OPENVLA_CALIB format with hidden_dim=2048"
    exit 1
fi

echo "============================================================"
echo "  Pi0.5 Imatrix Quantization Pipeline"
echo "============================================================"
echo "  Quant type:  ${QUANT_TYPE}"
echo "  Calib dir:   ${CALIB_DIR} (${#CALIB_FILES[@]} files)"
echo "  Model dir:   ${MODEL_PYTORCH_DIR}"
echo "  Output dir:  ${OUTPUT_DIR}"
echo "  GPUs:        ${NUM_GPUS} / ${TOTAL_GPUS}"
echo "  LLM BF16:    ${LLM_BF16}"
echo "  PI05 FP16:   ${PI05_FP16}"
echo "  PI05 quant:  ${PI05_QUANT}"
echo "============================================================"

mkdir -p "${OUTPUT_DIR}" "${TMP_DIR}"

# Cleanup on exit
cleanup() {
    echo "Cleaning up tmp dir..."
    rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

# ── Step 1: Export standalone LLM GGUF ───────────────────────────────────────
echo ""
echo "======================================================================"
echo "Step 1: Export standalone pali_llm_bf16.gguf"
echo "======================================================================"

if [ -f "${LLM_BF16}" ]; then
    echo "  Already exists: ${LLM_BF16}"
    echo "  Skipping export."
else
    echo "  Exporting from ${MODEL_PYTORCH_DIR} ..."
    PYTHONPATH="${PYTHONPATH_EXTRA}" \
    "${PYTHON}" "${SCRIPT_DIR}/export_pi05_llm.py" \
        -d "${MODEL_PYTORCH_DIR}" \
        -o "${OUTPUT_DIR}"
    echo "  Done: ${LLM_BF16}"
fi

# ── Step 2: Run imatrix in parallel ──────────────────────────────────────────
echo ""
echo "======================================================================"
echo "Step 2: Generating imatrix (${#CALIB_FILES[@]} files, ${NUM_GPUS} GPUs)"
echo "======================================================================"

if [ -f "${IMATRIX_OUT}" ]; then
    echo "  Already exists: ${IMATRIX_OUT}"
    echo "  Skipping imatrix generation (delete it to regenerate)."
else

IMATRIX_FILES=()
PIDS=()

# Assign calibration files round-robin across GPUs
for idx in "${!CALIB_FILES[@]}"; do
    BIN="${CALIB_FILES[$idx]}"
    GPU_ID=$((idx % NUM_GPUS))
    BASENAME=$(basename "${BIN}" .bin)
    IMATRIX_FILE="${TMP_DIR}/imatrix_${BASENAME}.gguf"
    IMATRIX_FILES+=("${IMATRIX_FILE}")

    echo "  GPU ${GPU_ID}: ${BASENAME}.bin → ${IMATRIX_FILE}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${BINARY_DIR}/llama-imatrix" \
        -m "${LLM_BF16}" \
        -f "${BIN}" \
        -o "${IMATRIX_FILE}" \
        --ctx-size 2048 \
        -ngl 99 \
        --no-ppl \
        > "${TMP_DIR}/imatrix_${BASENAME}.log" 2>&1 &
    PIDS+=($!)
done

echo "  Waiting for all imatrix jobs..."
for pid in "${PIDS[@]}"; do
    wait "$pid"
done
echo "  All imatrix jobs complete."

# ── Step 3: Merge imatrix files ───────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "Step 3: Merging ${#IMATRIX_FILES[@]} imatrix files → ${IMATRIX_OUT}"
echo "======================================================================"

IN_FILE_ARGS=()
for f in "${IMATRIX_FILES[@]}"; do
    IN_FILE_ARGS+=(--in-file "$f")
done

CUDA_VISIBLE_DEVICES=0 "${BINARY_DIR}/llama-imatrix" \
    "${IN_FILE_ARGS[@]}" \
    -o "${IMATRIX_OUT}"

echo "  Merged imatrix: ${IMATRIX_OUT}"

fi  # end imatrix skip check

# ── Step 4: Quantize LLM ─────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "Step 4: Quantizing LLM to ${QUANT_TYPE} with imatrix"
echo "======================================================================"

"${BINARY_DIR}/llama-quantize" \
    --imatrix "${IMATRIX_OUT}" \
    "${LLM_BF16}" \
    "${LLM_QUANT}" \
    "${QUANT_TYPE}"

echo "  Quantized LLM: ${LLM_QUANT}"

# ── Step 5: Merge back into pi05.gguf ────────────────────────────────────────
echo ""
echo "======================================================================"
echo "Step 5: Merging quantized LLM into full pi05.gguf"
echo "======================================================================"

if [ ! -f "${PI05_FP16}" ]; then
    echo "ERROR: Base pi05.gguf not found at ${PI05_FP16}"
    echo "  Please export it first with:"
    echo "    python export_pi05.py -d ${MODEL_PYTORCH_DIR} -o ${OUTPUT_DIR}"
    exit 1
fi

PYTHONPATH="${PYTHONPATH_EXTRA}" \
"${PYTHON}" "${SCRIPT_DIR}/merge_pi05_llm.py" \
    --base "${PI05_FP16}" \
    --llm  "${LLM_QUANT}" \
    --output "${PI05_QUANT}" \
    --quant-type "${QUANT_TYPE}"

# Copy supporting files if not already there
for f in tokenizer.model norm_stats.json; do
    SRC="${OUTPUT_DIR}/${f}"
    if [ -f "${SRC}" ] && [ "${PI05_QUANT}" != "${SRC}" ]; then
        DST="$(dirname "${PI05_QUANT}")/${f}"
        if [ "${SRC}" != "${DST}" ]; then
            cp "${SRC}" "${DST}" && echo "  Copied ${f} to $(dirname "${PI05_QUANT}")"
        fi
    fi
done

echo ""
echo "======================================================================"
echo "Done!"
echo "======================================================================"
echo "  Output model: ${PI05_QUANT}"
echo ""
echo "  To evaluate (10 trials for quick validation):"
echo "    cd ${SCRIPT_DIR}"
echo "    ./run_libero_eval.sh libero_spatial 10 5 8 8000 $(dirname "${PI05_QUANT}")"
echo "======================================================================"
