#!/bin/bash
# =============================================================================
# 01_setup_env.sh - Environment Setup for Pi0.5 Libero Quantization Evaluation
# =============================================================================
# This script sets up the environment for running Pi0.5 quantization benchmarks
# on the LIBERO simulation benchmark.
#
# Prerequisites:
#   - Ubuntu 22.04
#   - CUDA 12.6
#   - Conda (miniconda/anaconda)
#
# Usage:
#   ./01_setup_env.sh [WORKSPACE_DIR]
#
# Example:
#   ./01_setup_env.sh /home/arash/pi05_libero_quant_eval
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLA_CPP_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default workspace (can be overridden by argument)
WORKSPACE="${1:-${HOME}/pi05_libero_quant_eval}"

# Environment names
CONDA_ENV_NAME="pi05"
LIBERO_VENV_NAME=".venv"

# Python versions
PI05_PYTHON_VERSION="3.12"
LIBERO_PYTHON_VERSION="3.8"

# CUDA configuration
CUDA_PATH="/usr/local/cuda-12.6"

# OpenPI repository
OPENPI_REPO="https://github.com/Physical-Intelligence/openpi.git"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 is not installed"
        return 1
    fi
    return 0
}

# =============================================================================
# Step 1: Check System Dependencies
# =============================================================================
check_dependencies() {
    log_info "Checking system dependencies..."

    local missing=()

    # Check required commands
    check_command "git" || missing+=("git")
    check_command "cmake" || missing+=("cmake")
    check_command "gcc" || missing+=("gcc")
    check_command "conda" || missing+=("conda")

    # Check CUDA
    if [ ! -d "$CUDA_PATH" ]; then
        log_error "CUDA not found at $CUDA_PATH"
        missing+=("cuda")
    else
        log_info "CUDA found at $CUDA_PATH"
    fi

    # Check nvidia-smi
    if ! command -v nvidia-smi &> /dev/null; then
        log_warn "nvidia-smi not found, GPU may not be available"
    else
        log_info "GPU info:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -4
    fi

    if [ ${#missing[@]} -ne 0 ]; then
        log_error "Missing dependencies: ${missing[*]}"
        exit 1
    fi

    log_success "All system dependencies are available"
}

# =============================================================================
# Step 2: Install uv (Python package manager)
# =============================================================================
install_uv() {
    log_info "Checking uv installation..."

    if command -v uv &> /dev/null; then
        log_info "uv is already installed: $(uv --version)"
        return 0
    fi

    log_info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"

    if command -v uv &> /dev/null; then
        log_success "uv installed successfully: $(uv --version)"
    else
        log_error "Failed to install uv"
        exit 1
    fi
}

# =============================================================================
# Step 3: Create Workspace Directory Structure
# =============================================================================
create_workspace() {
    log_info "Creating workspace at $WORKSPACE..."

    mkdir -p "$WORKSPACE"
    mkdir -p "$WORKSPACE/models/weights"
    mkdir -p "$WORKSPACE/models/fp16"
    mkdir -p "$WORKSPACE/models/q4_0"
    mkdir -p "$WORKSPACE/models/q8_0"
    mkdir -p "$WORKSPACE/models/q4_k"
    mkdir -p "$WORKSPACE/results/fp16"
    mkdir -p "$WORKSPACE/results/q4_0"
    mkdir -p "$WORKSPACE/results/q8_0"
    mkdir -p "$WORKSPACE/results/q4_k"
    mkdir -p "$WORKSPACE/logs"

    log_success "Workspace directory structure created"
}

# =============================================================================
# Step 4: Clone OpenPI Repository
# =============================================================================
clone_openpi() {
    log_info "Setting up OpenPI repository..."

    OPENPI_DIR="$WORKSPACE/openpi"

    if [ -d "$OPENPI_DIR" ]; then
        log_info "OpenPI directory exists, checking for updates..."
        cd "$OPENPI_DIR"
        git fetch origin
        log_info "OpenPI repository already cloned"
    else
        log_info "Cloning OpenPI repository..."
        cd "$WORKSPACE"
        git clone "$OPENPI_REPO"
    fi

    cd "$OPENPI_DIR"

    # Initialize submodules (includes LIBERO)
    log_info "Initializing git submodules (LIBERO)..."
    git submodule update --init --recursive

    log_success "OpenPI and LIBERO submodules ready"
}

# =============================================================================
# Step 5: Setup LIBERO Environment (uv venv)
# =============================================================================
setup_libero_env() {
    log_info "Setting up LIBERO environment..."

    OPENPI_DIR="$WORKSPACE/openpi"
    LIBERO_EXAMPLE_DIR="$OPENPI_DIR/examples/libero"
    LIBERO_VENV_DIR="$LIBERO_EXAMPLE_DIR/.venv"

    cd "$OPENPI_DIR"

    # Ensure Python 3.8 is available (uv can auto-download if needed)
    log_info "Checking Python $LIBERO_PYTHON_VERSION availability..."
    if ! uv python list 2>/dev/null | grep -q "$LIBERO_PYTHON_VERSION"; then
        log_info "Installing Python $LIBERO_PYTHON_VERSION via uv..."
        uv python install "$LIBERO_PYTHON_VERSION"
    fi

    # Create virtual environment with Python 3.8
    if [ -d "$LIBERO_VENV_DIR" ]; then
        log_info "LIBERO venv already exists at $LIBERO_VENV_DIR"
    else
        log_info "Creating LIBERO venv with Python $LIBERO_PYTHON_VERSION..."
        uv venv --python "$LIBERO_PYTHON_VERSION" "$LIBERO_EXAMPLE_DIR/.venv"
    fi

    # Activate and install dependencies
    log_info "Installing LIBERO dependencies..."
    source "$LIBERO_VENV_DIR/bin/activate"

    # Install requirements with PyTorch CUDA 11.3 (as specified in openpi README)
    uv pip sync \
        examples/libero/requirements.txt \
        third_party/libero/requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu113 \
        --index-strategy=unsafe-best-match

    # Install openpi-client package
    uv pip install -e packages/openpi-client

    # Install LIBERO in editable mode
    uv pip install -e third_party/libero

    deactivate

    log_success "LIBERO environment setup complete"
}

# =============================================================================
# Step 6: Setup Pi0.5 Conda Environment
# =============================================================================
setup_pi05_env() {
    log_info "Setting up Pi0.5 conda environment..."

    # Check if environment exists
    if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        log_info "Conda environment '$CONDA_ENV_NAME' already exists"
    else
        log_info "Creating conda environment '$CONDA_ENV_NAME' with Python $PI05_PYTHON_VERSION..."
        conda create -n "$CONDA_ENV_NAME" python="$PI05_PYTHON_VERSION" -y
    fi

    # Install packages
    log_info "Installing Pi0.5 dependencies..."

    # Activate conda environment and install packages
    (
        eval "$(conda shell.bash hook)"
        conda activate "$CONDA_ENV_NAME"

        pip install --upgrade pip

        # Core dependencies for Pi0.5 inference
        pip install \
            torch \
            safetensors \
            gguf \
            numpy \
            opencv-python \
            websockets \
            msgpack \
            pybind11 \
            huggingface_hub
    )

    log_success "Pi0.5 conda environment setup complete"
}

# =============================================================================
# Step 7: Create Environment Activation Scripts
# =============================================================================
create_activation_scripts() {
    log_info "Creating activation scripts..."

    # Create activate_libero.sh
    cat > "$WORKSPACE/activate_libero.sh" << 'EOF'
#!/bin/bash
# Activate LIBERO environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/openpi/examples/libero/.venv/bin/activate"
export PYTHONPATH="$SCRIPT_DIR/openpi/third_party/libero:$PYTHONPATH"
echo "LIBERO environment activated"
EOF
    chmod +x "$WORKSPACE/activate_libero.sh"

    # Create activate_pi05.sh
    cat > "$WORKSPACE/activate_pi05.sh" << EOF
#!/bin/bash
# Activate Pi0.5 environment
eval "\$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME
export PYTHONPATH="$VLA_CPP_DIR/build/bin:\$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0}
echo "Pi0.5 environment activated (CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES)"
EOF
    chmod +x "$WORKSPACE/activate_pi05.sh"

    log_success "Activation scripts created"
}

# =============================================================================
# Step 8: Save Configuration
# =============================================================================
save_config() {
    log_info "Saving configuration..."

    cat > "$WORKSPACE/config.sh" << EOF
#!/bin/bash
# Pi0.5 Libero Quantization Evaluation Configuration
# Generated on $(date)

# Paths
export WORKSPACE="$WORKSPACE"
export VLA_CPP_DIR="$VLA_CPP_DIR"
export OPENPI_DIR="$WORKSPACE/openpi"
export LIBERO_VENV_DIR="$WORKSPACE/openpi/examples/libero/.venv"

# Model paths
export WEIGHTS_DIR="$WORKSPACE/models/weights"
export MODEL_FP16_DIR="$WORKSPACE/models/fp16"
export MODEL_Q4_0_DIR="$WORKSPACE/models/q4_0"
export MODEL_Q8_0_DIR="$WORKSPACE/models/q8_0"
export MODEL_Q4_K_DIR="$WORKSPACE/models/q4_k"

# Results paths
export RESULTS_DIR="$WORKSPACE/results"

# CUDA configuration
export CUDA_PATH="$CUDA_PATH"
export PATH="\$CUDA_PATH/bin:\$PATH"
export LD_LIBRARY_PATH="\$CUDA_PATH/lib64:\$LD_LIBRARY_PATH"

# Environment names
export CONDA_ENV_NAME="$CONDA_ENV_NAME"

# Script directory (VLA.cpp/tools/pi0.5)
export PI05_SCRIPT_DIR="$SCRIPT_DIR"
EOF

    chmod +x "$WORKSPACE/config.sh"

    log_success "Configuration saved to $WORKSPACE/config.sh"
}

# =============================================================================
# Step 9: Verify Installation
# =============================================================================
verify_installation() {
    log_info "Verifying installation..."

    local errors=0

    # Check workspace
    if [ ! -d "$WORKSPACE" ]; then
        log_error "Workspace not found"
        ((errors++))
    fi

    # Check OpenPI
    if [ ! -d "$WORKSPACE/openpi" ]; then
        log_error "OpenPI not found"
        ((errors++))
    fi

    # Check LIBERO submodule
    if [ ! -d "$WORKSPACE/openpi/third_party/libero" ]; then
        log_error "LIBERO submodule not found"
        ((errors++))
    fi

    # Check LIBERO venv
    if [ ! -d "$WORKSPACE/openpi/examples/libero/.venv" ]; then
        log_error "LIBERO venv not found"
        ((errors++))
    fi

    # Check conda environment
    if ! conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        log_error "Conda environment '$CONDA_ENV_NAME' not found"
        ((errors++))
    fi

    if [ $errors -eq 0 ]; then
        log_success "All components verified successfully"
    else
        log_error "Verification failed with $errors errors"
        exit 1
    fi
}

# =============================================================================
# Main
# =============================================================================
main() {
    echo "============================================================"
    echo "Pi0.5 Libero Quantization Evaluation - Environment Setup"
    echo "============================================================"
    echo ""
    echo "Workspace: $WORKSPACE"
    echo "VLA.cpp:   $VLA_CPP_DIR"
    echo ""

    check_dependencies
    install_uv
    create_workspace
    clone_openpi
    setup_libero_env
    setup_pi05_env
    create_activation_scripts
    save_config
    verify_installation

    echo ""
    echo "============================================================"
    log_success "Environment setup complete!"
    echo "============================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Run: $VLA_CPP_DIR/tools/pi0.5/02_prepare_models.sh $WORKSPACE"
    echo ""
    echo "Note: Using VLA.cpp from: $VLA_CPP_DIR"
    echo ""
    echo "To activate environments:"
    echo "  - LIBERO:  source $WORKSPACE/activate_libero.sh"
    echo "  - Pi0.5:   source $WORKSPACE/activate_pi05.sh"
    echo ""
}

main "$@"
