#!/bin/bash
# HSIC + Fisher Quantization Pipeline
#
# Computes HSIC sensitivity scores, runs greedy allocation, then quantizes
# the combined LLM using Fisher diagonal as per-weight imatrix.
#
# Usage:
#   bash tools/hsic/run_hsic_fisher_quant.sh [OPTIONS]
#
# Options:
#   --target-bpw FLOAT   Target bits-per-weight for block tensors (default: 2.41)
#   --sigma FLOAT        RBF kernel bandwidth for HSIC (omit for median heuristic)
#   --lx FLOAT           Lambda_x regularization (default: 1.0)
#   --ly FLOAT           Lambda_y regularization (default: 1.0)
#   --max-type STR       Max quant type for allocation (default: Q2_K)
#   --score-key STR      Score key from HSIC output (default: F_out)
#   --base-type STR      Base quantization type (default: IQ2_XS)

set -e

# ── Defaults ─────────────────────────────────────────────────────────────────
TARGET_BPW=""   # empty = auto-derive from base type
SIGMA=""   # empty = use median heuristic (passes no --sigma flag)
LX=1.0
LY=1.0
MAX_TYPE=Q2_K
SCORE_KEY=F_out
BASE_TYPE=IQ2_XS
NUM_GPUS=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --target-bpw) TARGET_BPW=$2; shift 2 ;;
        --sigma)      SIGMA=$2;      shift 2 ;;
        --lx)         LX=$2;         shift 2 ;;
        --ly)         LY=$2;         shift 2 ;;
        --max-type)   MAX_TYPE=$2;   shift 2 ;;
        --score-key)  SCORE_KEY=$2;  shift 2 ;;
        --base-type)  BASE_TYPE=$2;  shift 2 ;;
        --num-gpus)   NUM_GPUS=$2;   shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Block BPW of each preset's hardcoded heuristic (LLaMA 7B, 32 layers).
# This is what the allocator must match so we use the same bits as the baseline.
#   IQ2_XS : attn_v→Q2_K, ffn_down(L0-3)→Q2_K, rest→IQ2_XS
#   IQ2_S  : attn_v→IQ3_S, attn_output→IQ3_S, ffn_down(L0-3)→IQ3_S, rest→IQ2_S
#   Q2_K   : attn_v→Q3_K, ffn_down(all)→Q3_K, attn_output→Q3_K, rest→Q2_K
#   Q3_K_M : attn_v→Q5_K/Q4_K, ffn_down→Q5_K/Q4_K, attn_output→Q4_K, rest→Q3_K
declare -A PRESET_BLOCK_BPW
PRESET_BLOCK_BPW["IQ2_XXS"]=2.1305
PRESET_BLOCK_BPW["IQ2_XS"]=2.3471
PRESET_BLOCK_BPW["IQ2_S"]=2.5304
PRESET_BLOCK_BPW["Q2_K"]=2.9402
PRESET_BLOCK_BPW["IQ3_XXS"]=3.1250
PRESET_BLOCK_BPW["IQ3_S"]=3.4375
PRESET_BLOCK_BPW["Q3_K_M"]=3.3500

if [[ -z "$TARGET_BPW" ]]; then
    TARGET_BPW=${PRESET_BLOCK_BPW[$BASE_TYPE]}
    if [[ -z "$TARGET_BPW" ]]; then
        echo "No default target-bpw for base type '$BASE_TYPE'. Please set --target-bpw explicitly."
        exit 1
    fi
    echo "Auto-selected target-bpw=$TARGET_BPW to match $BASE_TYPE hardcoded heuristic."
fi

# ── Fixed paths ───────────────────────────────────────────────────────────────
CHECKPOINT=/path/to/openvla-oft-checkpoints/oft_combined
CALIB_DIR=/path/to/openvla-oft/calib_data
FISHER_GGUF=tools/fisher-diag/fisher_diag_combined_all.gguf
LLM_BF16=/path/to/openvla-oft-checkpoints/oft_combined_gguf/llm_bf16.gguf
OUT_DIR=/path/to/openvla-oft-checkpoints/oft_combined_gguf
QUANTIZE=./build_oft/bin/llama-quantize

# Activations are model+data dependent, not sigma dependent — reuse across runs
CACHE_ACTS=/tmp/hsic_acts_combined_all.npz

# ── Derived names ─────────────────────────────────────────────────────────────
# Replace dots with underscores for clean filenames
if [[ -z "$SIGMA" ]]; then
    SIGMA_TAG="median"
else
    SIGMA_TAG=$(echo $SIGMA | tr '.' '_')
fi
LX_TAG=$(echo $LX | tr '.' '_')
LY_TAG=$(echo $LY | tr '.' '_')
BPW_TAG=$(echo $TARGET_BPW | tr '.' '_')

TAG="s${SIGMA_TAG}_lx${LX_TAG}_ly${LY_TAG}_bpw${BPW_TAG}_${MAX_TYPE}"
HSIC_SCORES=tools/hsic/scores/hsic_combined_${TAG}.json
ALLOC_JSON=/tmp/alloc_hsic_${TAG}.json
BASE_LOWER=$(echo $BASE_TYPE | tr '[:upper:]' '[:lower:]')
OUT_GGUF=${OUT_DIR}/llm_${BASE_LOWER}_hsic_${SCORE_KEY}_${TAG}_fisher.gguf

echo "============================================================"
echo " HSIC + Fisher Quantization Pipeline"
echo "============================================================"
echo "  checkpoint  : $CHECKPOINT"
echo "  sigma       : ${SIGMA:-median}   lx: $LX   ly: $LY"
echo "  n_frames    : all"
echo "  target_bpw  : $TARGET_BPW   max_type: $MAX_TYPE   base_type: $BASE_TYPE"
echo "  score_key   : $SCORE_KEY"
echo "  hsic_scores : $HSIC_SCORES"
echo "  fisher_gguf : $FISHER_GGUF"
echo "  output      : $OUT_GGUF"
echo "============================================================"
echo ""

# ── Step 1: Compute HSIC scores ──────────────────────────────────────────────
if [[ -f "$HSIC_SCORES" ]]; then
    echo ">>> Step 1: Skipping — scores already exist: $HSIC_SCORES"
else
echo ">>> Step 1: Computing HSIC sensitivity scores..."
python tools/hsic/compute_hsic_tensor.py \
    --checkpoint $CHECKPOINT \
    --calib-data \
        $CALIB_DIR/spatial_20.bin \
        $CALIB_DIR/long_20.bin \
        $CALIB_DIR/goal_20.bin \
        $CALIB_DIR/object_20.bin \
    --calib-targets \
        $CALIB_DIR/spatial_20_targets.npy \
        $CALIB_DIR/long_20_targets.npy \
        $CALIB_DIR/goal_20_targets.npy \
        $CALIB_DIR/object_20_targets.npy \
    --y-mode actions \
    --kernel-hidden rbf \
    --kernel-y linear \
    --pool-all \
    ${SIGMA:+--sigma $SIGMA} \
    --lx $LX --ly $LY \
    --n-frames 999999 \
    --num-gpus $NUM_GPUS \
    --cache-acts $CACHE_ACTS \
    --output $HSIC_SCORES
echo "    Scores saved to $HSIC_SCORES"
fi
echo ""

# ── Step 2: Allocate quant types ──────────────────────────────────────────────
echo ">>> Step 2: Running greedy allocation (target_bpw=$TARGET_BPW)..."
python tools/hsic/allocate.py \
    --scores $HSIC_SCORES \
    --score-key $SCORE_KEY \
    --target-bpw $TARGET_BPW \
    --base-type $BASE_TYPE \
    --max-type $MAX_TYPE \
    --n-layers 32 \
    --output $ALLOC_JSON
echo "    Allocation saved to $ALLOC_JSON"
echo ""

# ── Step 3: Extract --tensor-type overrides from allocation JSON ──────────────
echo ">>> Step 3: Extracting tensor-type overrides..."
mapfile -t TENSOR_ARGS < <(python3 -c "
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

# ── Step 4: Quantize ──────────────────────────────────────────────────────────
echo ">>> Step 4: Quantizing LLM..."
echo "    imatrix : $FISHER_GGUF"
echo "    input   : $LLM_BF16"
echo "    output  : $OUT_GGUF"
echo ""
$QUANTIZE \
    --imatrix $FISHER_GGUF \
    "${TENSOR_ARGS[@]}" \
    $LLM_BF16 \
    $OUT_GGUF \
    $BASE_TYPE

echo ""
echo "============================================================"
echo " Done!"
echo " Quantized model: $OUT_GGUF"
echo "============================================================"
