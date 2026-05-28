#!/bin/bash
# HSIC + Fisher Quantization Pipeline for Pi 0.5
#
# Computes HSIC sensitivity scores, runs greedy allocation, then quantizes
# the PaliGemma LLM portion using the Fisher imatrix and HSIC-derived
# per-tensor type overrides. Finally merges back into pi05.gguf.
#
# Steps:
#   1. compute_hsic_pi05.py    → tools/hsic/scores/hsic_pi05_${TAG}.json
#   2. allocate_pi05.py        → /tmp/alloc_pi05_${TAG}.json
#   3. extract --tensor-type overrides
#   4. llama-quantize --imatrix fisher_flow_perweight.gguf + overrides
#                              → pali_llm_${BASE}_hsic.gguf
#   5. merge_pi05_llm.py       → pi05_${BASE}_hsic.gguf
#   6. set up symlink eval dir → pi05_libero_base_gguf/${BASE}_hsic_eval/
#
# Usage:
#   bash tools/hsic/run_hsic_quant_pi05.sh [--base-type IQ2_XS] [--max-type Q4_K] \
#                                          [--target-bpw 2.41] [--score-key F_out] \
#                                          [--num-gpus 8] [--no-recompute]

set -e

# ── Defaults ─────────────────────────────────────────────────────────────────
BASE_TYPE=IQ2_XS
MAX_TYPE=Q4_K
SCORE_KEY=F_out
TARGET_BPW=""
SIGMA=""
LX=1.0
LY=1.0
NUM_GPUS=8
BATCH_SIZE=8
NO_RECOMPUTE=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --base-type)   BASE_TYPE=$2;   shift 2 ;;
        --max-type)    MAX_TYPE=$2;    shift 2 ;;
        --target-bpw)  TARGET_BPW=$2;  shift 2 ;;
        --score-key)   SCORE_KEY=$2;   shift 2 ;;
        --sigma)       SIGMA=$2;       shift 2 ;;
        --lx)          LX=$2;          shift 2 ;;
        --ly)          LY=$2;          shift 2 ;;
        --num-gpus)    NUM_GPUS=$2;    shift 2 ;;
        --batch-size)  BATCH_SIZE=$2;  shift 2 ;;
        --no-recompute) NO_RECOMPUTE=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Block BPW per preset's hardcoded llama.cpp heuristic, adapted to Pi 0.5
# (18 layers, GQA n_kv=1 → attn_k/v are 1/8th of attn_q size).
# Computed: average bpw across 126 tensors with the IQ2_XS / IQ2_S / Q2_K rules.
declare -A PRESET_BLOCK_BPW
PRESET_BLOCK_BPW["IQ2_XS"]=2.3471   # heuristic upgrades attn_v + ffn_down (L0-1)
PRESET_BLOCK_BPW["IQ2_S"]=2.5304
PRESET_BLOCK_BPW["Q2_K"]=2.9402
PRESET_BLOCK_BPW["IQ3_S"]=3.4375
PRESET_BLOCK_BPW["Q3_K_M"]=3.3500

if [[ -z "$TARGET_BPW" ]]; then
    TARGET_BPW=${PRESET_BLOCK_BPW[$BASE_TYPE]}
    if [[ -z "$TARGET_BPW" ]]; then
        echo "No default target-bpw for base type '$BASE_TYPE'. Set --target-bpw explicitly."
        exit 1
    fi
    echo "Auto-selected target_bpw=$TARGET_BPW to match $BASE_TYPE heuristic block BPW."
fi

# ── Fixed paths ──────────────────────────────────────────────────────────────
REPO=/path/to/ActQuant
PYTHON=${HOME}/miniconda3/envs/openpi-server/bin/python
QUANTIZE=$REPO/build_openpi/bin/llama-quantize
GGUF_DIR=/path/to/openpi/pi05_libero_base_gguf
PYTORCH_DIR=/path/to/openpi/pi05_libero_base_pytorch
CALIB_DIR=/path/to/openpi/calib_data_raw
LLM_BF16=$GGUF_DIR/pali_llm_bf16.gguf
PI05_FP16=$GGUF_DIR/pi05.gguf
FISHER=$GGUF_DIR/fisher_flow_perweight.gguf

# Activations cache: same model+data → reusable across (sigma,lx,ly) sweeps
CACHE_ACTS=/tmp/hsic_pi05_acts.npz

# ── Derived names ────────────────────────────────────────────────────────────
SIGMA_TAG=${SIGMA:-median}
SIGMA_TAG=${SIGMA_TAG//./_}
LX_TAG=${LX//./_}
LY_TAG=${LY//./_}
BPW_TAG=${TARGET_BPW//./_}
TAG="s${SIGMA_TAG}_lx${LX_TAG}_ly${LY_TAG}_bpw${BPW_TAG}_${MAX_TYPE}"
HSIC_SCORES=$REPO/tools/hsic/scores/hsic_pi05_${TAG}.json
ALLOC_JSON=/tmp/alloc_pi05_${TAG}.json
BASE_LOWER=$(echo $BASE_TYPE | tr '[:upper:]' '[:lower:]')
LLM_OUT=$GGUF_DIR/pali_llm_${BASE_LOWER}_hsic_${SCORE_KEY//\//_}_${TAG}_fisher.gguf
PI05_OUT=$GGUF_DIR/pi05_${BASE_LOWER}_hsic_${SCORE_KEY//\//_}_${TAG}_fisher.gguf
EVAL_DIR=$GGUF_DIR/${BASE_LOWER}_hsic_${SCORE_KEY//\//_}_eval

mkdir -p $REPO/tools/hsic/scores

echo "============================================================"
echo " Pi 0.5 HSIC + Fisher Quantization Pipeline"
echo "============================================================"
echo "  base_type   : $BASE_TYPE"
echo "  max_type    : $MAX_TYPE"
echo "  target_bpw  : $TARGET_BPW"
echo "  score_key   : $SCORE_KEY"
echo "  sigma/lx/ly : ${SIGMA:-median}/$LX/$LY"
echo "  num_gpus    : $NUM_GPUS    batch_size: $BATCH_SIZE"
echo "  fisher      : $FISHER"
echo "  hsic scores : $HSIC_SCORES"
echo "  alloc JSON  : $ALLOC_JSON"
echo "  LLM output  : $LLM_OUT"
echo "  Pi 0.5 out  : $PI05_OUT"
echo "============================================================"

# ── Step 1: Compute HSIC scores ──────────────────────────────────────────────
if [[ -f "$HSIC_SCORES" && "$NO_RECOMPUTE" == "1" ]]; then
    echo ">>> Step 1: SKIP — HSIC scores already exist: $HSIC_SCORES"
elif [[ -f "$HSIC_SCORES" ]]; then
    echo ">>> Step 1: SKIP — HSIC scores exist (use --no-recompute to force keep): $HSIC_SCORES"
else
    echo ">>> Step 1: Computing HSIC sensitivity scores..."
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYTHON" "$REPO/tools/hsic/compute_hsic_pi05.py" \
        --checkpoint "$PYTORCH_DIR" \
        --calib-dir  "$CALIB_DIR" \
        --output     "$HSIC_SCORES" \
        --num-gpus $NUM_GPUS \
        --batch-size $BATCH_SIZE \
        --kernel-hidden rbf \
        --kernel-y linear \
        ${SIGMA:+--sigma $SIGMA} \
        --lx $LX --ly $LY \
        --cache-acts $CACHE_ACTS \
        --use-gpu-hsic
    echo "    Saved $HSIC_SCORES"
fi
echo ""

# ── Step 2: Allocate ──────────────────────────────────────────────────────────
echo ">>> Step 2: Greedy L2 allocation (target_bpw=$TARGET_BPW)..."
"$PYTHON" "$REPO/tools/hsic/allocate_pi05.py" \
    --scores      "$HSIC_SCORES" \
    --score-key   "$SCORE_KEY" \
    --target-bpw  "$TARGET_BPW" \
    --base-type   "$BASE_TYPE" \
    --max-type    "$MAX_TYPE" \
    --output      "$ALLOC_JSON"
echo "    Saved $ALLOC_JSON"
echo ""

# ── Step 3: Extract --tensor-type overrides ──────────────────────────────────
echo ">>> Step 3: Extracting tensor-type overrides..."
mapfile -t TENSOR_ARGS < <("$PYTHON" -c "
import json
with open('$ALLOC_JSON') as f:
    d = json.load(f)
base = d['base_type']
for name, qtype in sorted(d['assignments'].items()):
    if qtype != base:
        print('--tensor-type')
        print(f'{name}={qtype}')
")
N_OVERRIDES=$(( ${#TENSOR_ARGS[@]} / 2 ))
echo "    ${N_OVERRIDES} tensors upgraded from $BASE_TYPE"
echo ""

# ── Step 4: Quantize LLM ─────────────────────────────────────────────────────
if [[ ! -f "$LLM_BF16" ]]; then
    echo "ERROR: standalone LLM not found: $LLM_BF16"
    echo "  Run tools/pi0.5/export_pi05_llm.py first."
    exit 1
fi
if [[ ! -f "$FISHER" ]]; then
    echo "ERROR: Fisher imatrix not found: $FISHER"
    exit 1
fi

echo ">>> Step 4: Quantizing LLM..."
echo "    imatrix : $FISHER"
echo "    input   : $LLM_BF16"
echo "    output  : $LLM_OUT"
"$QUANTIZE" \
    --imatrix "$FISHER" \
    "${TENSOR_ARGS[@]}" \
    "$LLM_BF16" \
    "$LLM_OUT" \
    "$BASE_TYPE"
echo ""

# ── Step 5: Merge into pi05.gguf ─────────────────────────────────────────────
echo ">>> Step 5: Merging LLM into pi05.gguf..."
"$PYTHON" "$REPO/tools/pi0.5/merge_pi05_llm.py" \
    --base   "$PI05_FP16" \
    --llm    "$LLM_OUT" \
    --output "$PI05_OUT" \
    --quant-type "$BASE_TYPE"
echo ""

# ── Step 6: Eval dir setup ───────────────────────────────────────────────────
echo ">>> Step 6: Creating eval directory $EVAL_DIR ..."
mkdir -p "$EVAL_DIR"
ln -sf "$PI05_OUT"               "$EVAL_DIR/pi05.gguf"
ln -sf "$GGUF_DIR/tokenizer.model" "$EVAL_DIR/tokenizer.model"
ln -sf "$GGUF_DIR/norm_stats.json" "$EVAL_DIR/norm_stats.json"
ls -la "$EVAL_DIR"
echo ""

echo "============================================================"
echo " Done!"
echo " Quantized model: $PI05_OUT"
echo " Eval dir:        $EVAL_DIR"
echo ""
echo " Run libero eval with:"
echo "   cd $REPO/tools/pi0.5"
echo "   ./run_libero_eval.sh libero_spatial 50 5 8 8000 $EVAL_DIR"
echo "============================================================"
