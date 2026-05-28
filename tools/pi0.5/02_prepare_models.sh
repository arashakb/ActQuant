#!/bin/bash
# =============================================================================
# 02_prepare_models.sh - Model Preparation for Pi0.5 Quantization Evaluation
# =============================================================================
# This script downloads model weights, compiles VLA.cpp, and exports GGUF
# models in different quantization formats (fp16, q4_0, q8_0, q4_k).
#
# Prerequisites:
#   - 01_setup_env.sh has been run successfully
#   - VLA.cpp has been cloned to the workspace
#
# Usage:
#   ./02_prepare_models.sh [WORKSPACE_DIR]
#
# Example:
#   ./02_prepare_models.sh /home/arash/pi05_libero_quant_eval
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default workspace
WORKSPACE="${1:-${HOME}/pi05_libero_quant_eval}"

# Load configuration from setup script
if [ -f "$WORKSPACE/config.sh" ]; then
    source "$WORKSPACE/config.sh"
else
    echo "Error: config.sh not found. Run 01_setup_env.sh first."
    exit 1
fi

# HuggingFace model IDs
HF_MODEL_ID="lerobot/pi05_libero_finetuned_v044"
HF_TOKENIZER_ID="google/paligemma-3b-pt-224"

# CUDA configuration
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda-12.6}"

# Build configuration
BUILD_TYPE="Release"
BUILD_THREADS=$(nproc)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# =============================================================================
# Helper Functions
# =============================================================================
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# Step 1: Verify VLA.cpp Installation
# =============================================================================
verify_vla_cpp() {
    log_info "Verifying VLA.cpp installation..."

    # Check if VLA.cpp exists (either in workspace or use the one from script dir)
    if [ -d "$WORKSPACE/VLA.cpp" ]; then
        VLA_CPP_DIR="$WORKSPACE/VLA.cpp"
        log_info "Using VLA.cpp from workspace: $VLA_CPP_DIR"
    elif [ -d "$SCRIPT_DIR/../.." ]; then
        VLA_CPP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
        log_info "Using VLA.cpp from script location: $VLA_CPP_DIR"
    else
        log_error "VLA.cpp not found. Please clone it to $WORKSPACE/VLA.cpp"
        exit 1
    fi

    # Verify key files exist
    if [ ! -f "$VLA_CPP_DIR/tools/pi0.5/export_pi05.py" ]; then
        log_error "export_pi05.py not found in VLA.cpp/tools/pi0.5/"
        exit 1
    fi

    if [ ! -f "$VLA_CPP_DIR/CMakeLists.txt" ]; then
        log_error "CMakeLists.txt not found in VLA.cpp/"
        exit 1
    fi

    # Update config with actual VLA.cpp path
    export VLA_CPP_DIR
    log_success "VLA.cpp verified at $VLA_CPP_DIR"
}

# =============================================================================
# Step 2: Download Model Weights from HuggingFace
# =============================================================================
download_weights() {
    log_info "Downloading model weights from HuggingFace..."

    WEIGHTS_DIR="$WORKSPACE/models/weights"

    # Download in subshell with conda environment
    (
        eval "$(conda shell.bash hook)"
        conda activate "$CONDA_ENV_NAME"

        # Download main model weights using Python API
        log_info "Downloading $HF_MODEL_ID..."
        if [ -f "$WEIGHTS_DIR/model.safetensors" ]; then
            log_info "Model weights already exist, skipping download"
        else
            python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$HF_MODEL_ID',
    local_dir='$WEIGHTS_DIR',
    local_dir_use_symlinks=False
)
print('Model download complete')
"
        fi

        # Download tokenizer using Python API
        log_info "Downloading tokenizer from $HF_TOKENIZER_ID..."
        if [ -f "$WEIGHTS_DIR/tokenizer.model" ]; then
            log_info "Tokenizer already exists, skipping download"
        else
            python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='$HF_TOKENIZER_ID',
    filename='tokenizer.model',
    local_dir='$WEIGHTS_DIR',
    local_dir_use_symlinks=False
)
print('Tokenizer download complete')
"
        fi
    )

    # Verify downloads
    if [ ! -f "$WEIGHTS_DIR/model.safetensors" ]; then
        log_error "Model weights download failed"
        exit 1
    fi

    if [ ! -f "$WEIGHTS_DIR/tokenizer.model" ]; then
        log_error "Tokenizer download failed"
        exit 1
    fi

    log_success "Model weights downloaded to $WEIGHTS_DIR"
}

# =============================================================================
# Step 3: Build VLA.cpp
# =============================================================================
build_vla_cpp() {
    log_info "Building VLA.cpp..."

    BUILD_DIR="$VLA_CPP_DIR/build"

    # Create build directory
    mkdir -p "$BUILD_DIR"

    # Build in conda environment to link against correct Python version
    (
        eval "$(conda shell.bash hook)"
        conda activate "$CONDA_ENV_NAME"

        cd "$BUILD_DIR"

        # Get Python paths from conda environment
        PYTHON_EXE=$(which python)
        PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        PYTHON_INCLUDE=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")
        PYTHON_LIB=$(python -c "import sysconfig; import os; libdir=sysconfig.get_config_var('LIBDIR'); ldlib=sysconfig.get_config_var('LDLIBRARY'); print(os.path.join(libdir, ldlib))")

        # Configure with CMake
        log_info "Configuring CMake..."
        log_info "  Python executable: $PYTHON_EXE"
        log_info "  Python version: $PYTHON_VERSION"
        log_info "  Python include: $PYTHON_INCLUDE"
        log_info "  Python library: $PYTHON_LIB"

        cmake .. \
            -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
            -DGGML_NATIVE=ON \
            -DGGML_AVX2=ON \
            -DGGML_FMA=ON \
            -DGGML_F16C=ON \
            -DGGML_CUDA=ON \
            -DBUILD_PI05_PYTHON=ON \
            -DCMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc" \
            -DPython_EXECUTABLE="$PYTHON_EXE" \
            -DPython_INCLUDE_DIR="$PYTHON_INCLUDE" \
            -DPython_LIBRARY="$PYTHON_LIB"

        # Build
        log_info "Building targets (using $BUILD_THREADS threads)..."
        cmake --build . --target pi05 pi05_py -j"$BUILD_THREADS"
    )

    # Verify build outputs
    if [ ! -f "$BUILD_DIR/bin/pi05" ]; then
        log_error "pi05 binary not found after build"
        exit 1
    fi

    # Find the Python module (could be pi05.so or pi05.cpython-*.so)
    PI05_PY_SO=$(find "$BUILD_DIR" -name "pi05*.so" | head -1)
    if [ -z "$PI05_PY_SO" ]; then
        log_error "pi05_py module not found after build"
        exit 1
    fi

    log_success "VLA.cpp built successfully"
    log_info "  Binary: $BUILD_DIR/bin/pi05"
    log_info "  Python module: $PI05_PY_SO"
}

# =============================================================================
# Step 4: Export GGUF Models
# =============================================================================
export_gguf_models() {
    log_info "Exporting GGUF models in different quantization formats..."

    EXPORT_SCRIPT="$VLA_CPP_DIR/tools/pi0.5/export_pi05.py"
    WEIGHTS_DIR="$WORKSPACE/models/weights"

    # Export all models in subshell with conda environment
    (
        eval "$(conda shell.bash hook)"
        conda activate "$CONDA_ENV_NAME"

        # Export FP16 (no quantization)
        log_info "Exporting FP16 model..."
        python "$EXPORT_SCRIPT" \
            -d "$WEIGHTS_DIR" \
            -o "$WORKSPACE/models/fp16" \
            2>&1 | tee "$WORKSPACE/logs/export_fp16.log"

        # Export Q8_0
        log_info "Exporting Q8_0 model..."
        python "$EXPORT_SCRIPT" \
            -d "$WEIGHTS_DIR" \
            -o "$WORKSPACE/models/q8_0" \
            --quant_llm q8 \
            --quant_vision q8 \
            --quant_embedding q8 \
            2>&1 | tee "$WORKSPACE/logs/export_q8_0.log"

        # Export Q4_0
        log_info "Exporting Q4_0 model..."
        python "$EXPORT_SCRIPT" \
            -d "$WEIGHTS_DIR" \
            -o "$WORKSPACE/models/q4_0" \
            --quant_llm q4 \
            --quant_vision q4 \
            --quant_embedding q4 \
            2>&1 | tee "$WORKSPACE/logs/export_q4_0.log"

        # Export Q4_K
        log_info "Exporting Q4_K model..."
        python "$EXPORT_SCRIPT" \
            -d "$WEIGHTS_DIR" \
            -o "$WORKSPACE/models/q4_k" \
            --quant_llm q4k \
            --quant_vision q4k \
            --quant_embedding q4k \
            2>&1 | tee "$WORKSPACE/logs/export_q4_k.log"
    )

    # Verify exports
    for quant in fp16 q8_0 q4_0 q4_k; do
        if [ -f "$WORKSPACE/models/$quant/pi05.gguf" ]; then
            log_success "$quant model exported"
            ls -lh "$WORKSPACE/models/$quant/pi05.gguf"
        else
            log_error "$quant export failed"
        fi
    done

    # Summary
    echo ""
    log_info "Model sizes summary:"
    echo "----------------------------------------"
    for quant in fp16 q8_0 q4_0 q4_k; do
        if [ -f "$WORKSPACE/models/$quant/pi05.gguf" ]; then
            size=$(ls -lh "$WORKSPACE/models/$quant/pi05.gguf" | awk '{print $5}')
            echo "  $quant: $size"
        else
            echo "  $quant: FAILED"
        fi
    done
    echo "----------------------------------------"
}

# =============================================================================
# Step 5: Verify Exports
# =============================================================================
verify_exports() {
    log_info "Verifying exported models..."

    local errors=0

    for quant in fp16 q8_0 q4_0 q4_k; do
        if [ -f "$WORKSPACE/models/$quant/pi05.gguf" ]; then
            log_success "$quant model exists"
        else
            log_error "$quant model not found"
            ((errors++))
        fi

        # Check tokenizer
        if [ -f "$WORKSPACE/models/$quant/tokenizer.model" ]; then
            log_info "  $quant tokenizer exists"
        else
            log_warn "  $quant tokenizer not found, copying from weights..."
            cp "$WORKSPACE/models/weights/tokenizer.model" "$WORKSPACE/models/$quant/" 2>/dev/null || true
        fi
    done

    if [ $errors -eq 0 ]; then
        log_success "All models verified successfully"
    else
        log_error "Verification failed with $errors errors"
        exit 1
    fi
}

# =============================================================================
# Step 6: Update Configuration
# =============================================================================
update_config() {
    log_info "Updating configuration..."

    # Check if VLA_CPP_BUILD_DIR is already in config (avoid duplicates)
    if ! grep -q "VLA_CPP_BUILD_DIR" "$WORKSPACE/config.sh" 2>/dev/null; then
        # Append VLA.cpp build paths to config
        cat >> "$WORKSPACE/config.sh" << EOF

# VLA.cpp build paths (added by 02_prepare_models.sh)
export VLA_CPP_BUILD_DIR="$VLA_CPP_DIR/build"
export PI05_BIN="$VLA_CPP_DIR/build/bin/pi05"
EOF
    fi

    log_success "Configuration updated"
}

# =============================================================================
# Main
# =============================================================================
main() {
    echo "============================================================"
    echo "Pi0.5 Libero Quantization Evaluation - Model Preparation"
    echo "============================================================"
    echo ""
    echo "Workspace: $WORKSPACE"
    echo ""

    verify_vla_cpp
    download_weights
    build_vla_cpp
    export_gguf_models
    verify_exports
    update_config

    echo ""
    echo "============================================================"
    log_success "Model preparation complete!"
    echo "============================================================"
    echo ""
    echo "Exported models:"
    for quant in fp16 q8_0 q4_0 q4_k; do
        if [ -f "$WORKSPACE/models/$quant/pi05.gguf" ]; then
            size=$(ls -lh "$WORKSPACE/models/$quant/pi05.gguf" | awk '{print $5}')
            echo "  - $WORKSPACE/models/$quant/pi05.gguf ($size)"
        fi
    done
    echo ""
    echo "Next steps:"
    echo "  Run: ./03_run_benchmark.sh $WORKSPACE"
    echo ""
}

main "$@"
