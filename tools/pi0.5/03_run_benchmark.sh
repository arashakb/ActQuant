#!/bin/bash
# =============================================================================
# 03_run_benchmark.sh - Run Libero Benchmark for Pi0.5 Quantization Evaluation
# =============================================================================
# This script runs the LIBERO benchmark to evaluate different quantization
# formats (fp16, q4_0, q8_0, q4_k) of the Pi0.5 model.
#
# Prerequisites:
#   - 01_setup_env.sh and 02_prepare_models.sh have been run successfully
#
# Usage:
#   ./03_run_benchmark.sh [WORKSPACE_DIR] [OPTIONS]
#
# Options:
#   --parallel        Run all quantization types in parallel (default: serial)
#   --quant TYPE      Run only specific quantization type (fp16|q4_0|q8_0|q4_k)
#   --trials N        Number of trials per task (default: 50)
#   --task SUITE      Task suite name (default: libero_spatial)
#   --dry-run         Print commands without executing
#
# Examples:
#   ./03_run_benchmark.sh /home/arash/pi05_libero_quant_eval
#   ./03_run_benchmark.sh /home/arash/pi05_libero_quant_eval --parallel
#   ./03_run_benchmark.sh /home/arash/pi05_libero_quant_eval --quant fp16 --trials 10
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse workspace from first non-option argument
WORKSPACE=""
for arg in "$@"; do
    if [[ "$arg" != --* ]]; then
        WORKSPACE="$arg"
        break
    fi
done
WORKSPACE="${WORKSPACE:-${HOME}/pi05_libero_quant_eval}"

# Load configuration
if [ -f "$WORKSPACE/config.sh" ]; then
    source "$WORKSPACE/config.sh"
else
    echo "Error: config.sh not found. Run 01_setup_env.sh first."
    exit 1
fi

# Default parameters
PARALLEL_MODE=false
SINGLE_QUANT=""
NUM_TRIALS=50
TASK_SUITE="libero_spatial"
DRY_RUN=false
MUJOCO_GL="${MUJOCO_GL:-egl}"

# Quantization types and their configurations
QUANT_TYPES=("fp16" "q8_0" "q4_0" "q4_k")
declare -A QUANT_PORTS=( ["fp16"]=8000 ["q8_0"]=8001 ["q4_0"]=8002 ["q4_k"]=8003 )
declare -A QUANT_GPUS=( ["fp16"]="CUDA0" ["q8_0"]="CUDA1" ["q4_0"]="CUDA2" ["q4_k"]="CUDA3" )

# Server configuration
SERVER_THREADS=8
FLOW_STEPS=10

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# =============================================================================
# Parse Arguments
# =============================================================================
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --parallel)
                PARALLEL_MODE=true
                shift
                ;;
            --quant)
                SINGLE_QUANT="$2"
                shift 2
                ;;
            --trials)
                NUM_TRIALS="$2"
                shift 2
                ;;
            --task)
                TASK_SUITE="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                # Skip non-option arguments (workspace path)
                shift
                ;;
        esac
    done

    # If single quant specified, only run that one
    if [ -n "$SINGLE_QUANT" ]; then
        QUANT_TYPES=("$SINGLE_QUANT")
        PARALLEL_MODE=false
    fi
}

show_help() {
    echo "Usage: $0 [WORKSPACE_DIR] [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --parallel        Run all quantization types in parallel"
    echo "  --quant TYPE      Run only specific type (fp16|q4_0|q8_0|q4_k)"
    echo "  --trials N        Number of trials per task (default: 50)"
    echo "  --task SUITE      Task suite (default: libero_spatial)"
    echo "  --dry-run         Print commands without executing"
    echo "  --help, -h        Show this help message"
}

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

log_quant() {
    local quant="$1"
    local msg="$2"
    echo -e "${CYAN}[$quant]${NC} $msg"
}

# Get PID file path for a quant type
get_pid_file() {
    echo "$WORKSPACE/logs/server_${1}.pid"
}

# Check if server is running
is_server_running() {
    local quant="$1"
    local port="${QUANT_PORTS[$quant]}"
    pgrep -f "serve_policy.py.*--port $port" > /dev/null 2>&1
}

# Wait for server to be ready
wait_for_server() {
    local port="$1"
    local max_wait=120
    local waited=0

    log_info "Waiting for server on port $port..."

    while [ $waited -lt $max_wait ]; do
        if curl -s "http://localhost:$port/healthz" > /dev/null 2>&1; then
            log_success "Server on port $port is ready"
            return 0
        fi
        sleep 2
        ((waited+=2))
    done

    log_error "Server on port $port did not start within $max_wait seconds"
    return 1
}

# =============================================================================
# Start Policy Server
# =============================================================================
start_server() {
    local quant="$1"
    local port="${QUANT_PORTS[$quant]}"
    local gpu="${QUANT_GPUS[$quant]}"
    local model_dir="$WORKSPACE/models/$quant"
    local log_file="$WORKSPACE/logs/server_${quant}.log"
    local pid_file=$(get_pid_file "$quant")

    log_quant "$quant" "Starting policy server on port $port (GPU: $gpu)..."

    # Check if model exists
    if [ ! -f "$model_dir/pi05.gguf" ]; then
        log_error "Model not found: $model_dir/pi05.gguf"
        return 1
    fi

    # Kill existing server if running
    if is_server_running "$quant"; then
        log_warn "Server for $quant already running, stopping it..."
        stop_server "$quant"
        sleep 2
    fi

    # Build the command
    local serve_script="$SCRIPT_DIR/serve_policy.py"
    local pythonpath="$VLA_CPP_DIR/build/bin"

    local cmd="conda activate $CONDA_ENV_NAME && PYTHONPATH=$pythonpath python $serve_script \
        --model-dir $model_dir \
        --port $port \
        --device $gpu \
        --threads $SERVER_THREADS \
        --steps $FLOW_STEPS"

    if [ "$DRY_RUN" = true ]; then
        log_quant "$quant" "DRY-RUN: $cmd"
        return 0
    fi

    # ODE step distribution profile output path (written on server shutdown)
    local ode_profile_path="$WORKSPACE/results/$quant/ode_step_profile.csv"

    # Start server in background (using subshell with conda activate)
    (
        eval "$(conda shell.bash hook)"
        conda activate "$CONDA_ENV_NAME"
        export PYTHONPATH="$VLA_CPP_DIR/build/bin:$PYTHONPATH"
        python "$serve_script" \
            --model-dir "$model_dir" \
            --port "$port" \
            --device "$gpu" \
            --threads "$SERVER_THREADS" \
            --steps "$FLOW_STEPS" \
            --ode-profile "$ode_profile_path"
    ) > "$log_file" 2>&1 &

    local server_pid=$!
    echo "$server_pid" > "$pid_file"

    log_quant "$quant" "Server started with PID $server_pid (log: $log_file)"

    # Wait for server to be ready
    if ! wait_for_server "$port"; then
        log_error "Server for $quant failed to start"
        cat "$log_file" | tail -20
        return 1
    fi

    return 0
}

# =============================================================================
# Stop Policy Server
# =============================================================================
stop_server() {
    local quant="$1"
    local port="${QUANT_PORTS[$quant]}"
    local pid_file=$(get_pid_file "$quant")

    log_quant "$quant" "Stopping server on port $port..."
    pkill -f "serve_policy.py.*--port $port" 2>/dev/null || true
    sleep 1
    pkill -9 -f "serve_policy.py.*--port $port" 2>/dev/null || true

    rm -f "$pid_file"
}

# Stop all servers
stop_all_servers() {
    log_info "Stopping all servers..."
    for quant in "${QUANT_TYPES[@]}"; do
        stop_server "$quant"
    done
}

# =============================================================================
# Run LIBERO Evaluation
# =============================================================================
run_evaluation() {
    local quant="$1"
    local port="${QUANT_PORTS[$quant]}"
    local result_dir="$WORKSPACE/results/$quant"
    local video_dir="$result_dir/videos"
    local log_file="$WORKSPACE/logs/eval_${quant}.log"

    log_quant "$quant" "Running LIBERO evaluation ($TASK_SUITE, $NUM_TRIALS trials)..."

    mkdir -p "$video_dir"

    # Build the command
    local openpi_dir="$WORKSPACE/openpi"
    local main_script="$openpi_dir/examples/libero/main.py"
    local venv_activate="$openpi_dir/examples/libero/.venv/bin/activate"

    local cmd="MUJOCO_GL=$MUJOCO_GL PYTHONPATH=$openpi_dir/third_party/libero:\$PYTHONPATH \
        python $main_script \
        --args.port $port \
        --args.task-suite-name $TASK_SUITE \
        --args.num-trials-per-task $NUM_TRIALS \
        --args.video-out-path $video_dir"

    if [ "$DRY_RUN" = true ]; then
        log_quant "$quant" "DRY-RUN: source $venv_activate && $cmd"
        return 0
    fi

    # Run evaluation in LIBERO venv
    (
        source "$venv_activate"
        export PYTHONPATH="$openpi_dir/third_party/libero:$PYTHONPATH"
        export MUJOCO_GL="$MUJOCO_GL"

        python "$main_script" \
            --args.port "$port" \
            --args.task-suite-name "$TASK_SUITE" \
            --args.num-trials-per-task "$NUM_TRIALS" \
            --args.video-out-path "$video_dir" \
            2>&1 | tee "$log_file"
    )

    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        log_quant "$quant" "Evaluation completed successfully"
    else
        log_error "Evaluation for $quant failed with exit code $exit_code"
    fi

    return $exit_code
}

# =============================================================================
# Extract Results
# =============================================================================
extract_results() {
    local quant="$1"
    local eval_log="$WORKSPACE/logs/eval_${quant}.log"
    local server_log="$WORKSPACE/logs/server_${quant}.log"
    local result_dir="$WORKSPACE/results/${quant}"
    local result_file="$result_dir/summary.txt"
    local task_results_file="$result_dir/task_results.txt"

    # Ensure result directory exists
    mkdir -p "$result_dir"

    if [ ! -f "$eval_log" ]; then
        log_warn "Eval log file not found for $quant"
        return
    fi

    log_quant "$quant" "Extracting results..."

    # Extract total success rate from eval log
    local success_rate=$(grep -oP "Total success rate: \K[0-9.]+" "$eval_log" | tail -1)
    local total_episodes=$(grep -oP "Total episodes: \K[0-9]+" "$eval_log" | tail -1)

    # Extract average inference time from server log
    local avg_infer_time="N/A"
    if [ -f "$server_log" ]; then
        # Extract all inference times and calculate average
        local infer_times=$(grep -oP "Inference time: \K[0-9.]+" "$server_log")
        if [ -n "$infer_times" ]; then
            avg_infer_time=$(echo "$infer_times" | awk '{sum+=$1; count++} END {if(count>0) printf "%.1f", sum/count; else print "N/A"}')
        fi
    fi

    # Extract per-task success rates
    # Format in log: "Task: <description>" followed by "Current task success rate: <rate>"
    > "$task_results_file"  # Clear file
    echo "# Per-Task Success Rates for $quant" >> "$task_results_file"
    echo "# Task Suite: $TASK_SUITE" >> "$task_results_file"
    echo "" >> "$task_results_file"

    # Use awk to pair task names with their success rates
    awk '
    /Task:/ {
        # Extract task description
        gsub(/.*Task: /, "")
        task = $0
    }
    /Current task success rate:/ {
        # Extract success rate
        gsub(/.*Current task success rate: /, "")
        rate = $0
        if (task != "") {
            printf "%-60s | %.4f\n", task, rate
            task = ""
        }
    }
    ' "$eval_log" >> "$task_results_file"

    # Write summary file
    cat > "$result_file" << EOF
# Pi0.5 $quant Evaluation Results
# Task Suite: $TASK_SUITE
# Trials per task: $NUM_TRIALS
# Date: $(date)

Success Rate: ${success_rate:-N/A}
Total Episodes: ${total_episodes:-N/A}
Avg Inference Time: ${avg_infer_time}ms
EOF

    log_quant "$quant" "Results saved to $result_file"
}

# =============================================================================
# Run Single Evaluation (Serial Mode)
# =============================================================================
run_single_eval() {
    local quant="$1"

    log_info "=========================================="
    log_info "Evaluating: $quant"
    log_info "=========================================="

    # Start server
    if ! start_server "$quant"; then
        log_error "Failed to start server for $quant"
        return 1
    fi

    # Run evaluation
    local eval_exit=0
    run_evaluation "$quant" || eval_exit=$?

    # Stop server
    stop_server "$quant"

    # Extract results
    extract_results "$quant"

    return $eval_exit
}

# =============================================================================
# Run Parallel Evaluation
# =============================================================================
run_parallel_eval() {
    log_info "Running parallel evaluation for: ${QUANT_TYPES[*]}"

    # Start all servers
    for quant in "${QUANT_TYPES[@]}"; do
        start_server "$quant" &
    done
    wait

    # Verify all servers are running
    for quant in "${QUANT_TYPES[@]}"; do
        local port="${QUANT_PORTS[$quant]}"
        if ! curl -s "http://localhost:$port/healthz" > /dev/null 2>&1; then
            log_error "Server for $quant is not responding"
            stop_all_servers
            return 1
        fi
    done

    log_success "All servers started successfully"

    # Run all evaluations in parallel
    local pids=()
    for quant in "${QUANT_TYPES[@]}"; do
        (
            run_evaluation "$quant"
            extract_results "$quant"
        ) &
        pids+=($!)
        log_quant "$quant" "Evaluation started in background (PID: ${pids[-1]})"
    done

    # Wait for all evaluations to complete
    log_info "Waiting for all evaluations to complete..."
    local failed=0
    for i in "${!pids[@]}"; do
        local quant="${QUANT_TYPES[$i]}"
        local pid="${pids[$i]}"
        if wait "$pid"; then
            log_quant "$quant" "Evaluation completed successfully"
        else
            log_error "Evaluation for $quant failed"
            ((failed++))
        fi
    done

    # Stop all servers
    stop_all_servers

    if [ $failed -gt 0 ]; then
        log_error "$failed evaluation(s) failed"
        return 1
    fi

    return 0
}

# =============================================================================
# Generate Summary Report
# =============================================================================
generate_report() {
    local report_file="$WORKSPACE/results/benchmark_report.txt"

    log_info "Generating benchmark report..."

    cat > "$report_file" << EOF
================================================================================
Pi0.5 Quantization Benchmark Report
================================================================================
Date: $(date)
Task Suite: $TASK_SUITE
Trials per task: $NUM_TRIALS
Mode: $([ "$PARALLEL_MODE" = true ] && echo "Parallel" || echo "Serial")

================================================================================
Overall Results Summary
================================================================================
EOF

    # Print header
    printf "%-10s | %-10s | %-12s | %-15s\n" "Quant" "Model Size" "Success Rate" "Avg Infer (ms)" >> "$report_file"
    printf "%s\n" "--------------------------------------------------------------" >> "$report_file"

    for quant in fp16 q8_0 q4_0 q4_k; do
        local summary="$WORKSPACE/results/$quant/summary.txt"
        local model_size="N/A"
        local success_rate="N/A"
        local avg_infer="N/A"

        if [ -f "$WORKSPACE/models/$quant/pi05.gguf" ]; then
            model_size=$(ls -lh "$WORKSPACE/models/$quant/pi05.gguf" | awk '{print $5}')
        fi

        if [ -f "$summary" ]; then
            success_rate=$(grep "Success Rate:" "$summary" | cut -d: -f2 | tr -d ' ')
            avg_infer=$(grep "Avg Inference Time:" "$summary" | grep -oP "[0-9.]+")
        fi

        printf "%-10s | %-10s | %-12s | %-15s\n" "$quant" "$model_size" "$success_rate" "${avg_infer:-N/A}" >> "$report_file"
    done

    cat >> "$report_file" << EOF

================================================================================
Per-Task Success Rate Comparison
================================================================================
EOF

    # Get task list from first available result
    local first_task_file=""
    for quant in fp16 q8_0 q4_0 q4_k; do
        if [ -f "$WORKSPACE/results/$quant/task_results.txt" ]; then
            first_task_file="$WORKSPACE/results/$quant/task_results.txt"
            break
        fi
    done

    if [ -n "$first_task_file" ]; then
        # Print header
        printf "%-50s |" "Task" >> "$report_file"
        for quant in fp16 q8_0 q4_0 q4_k; do
            printf " %-8s |" "$quant" >> "$report_file"
        done
        echo "" >> "$report_file"
        printf "%s\n" "$(printf '%0.s-' {1..100})" >> "$report_file"

        # Extract unique task names (skip comments and empty lines)
        grep -v "^#" "$first_task_file" | grep -v "^$" | while read -r line; do
            # Extract task name (everything before the last |)
            local task_name=$(echo "$line" | sed 's/ *|.*//' | head -c 50)

            printf "%-50s |" "$task_name" >> "$report_file"

            for quant in fp16 q8_0 q4_0 q4_k; do
                local task_file="$WORKSPACE/results/$quant/task_results.txt"
                local rate="N/A"
                if [ -f "$task_file" ]; then
                    # Find matching task and extract rate
                    rate=$(grep -F "$task_name" "$task_file" 2>/dev/null | awk -F'|' '{gsub(/^ *| *$/, "", $2); print $2}' | head -1)
                fi
                printf " %-8s |" "${rate:-N/A}" >> "$report_file"
            done
            echo "" >> "$report_file"
        done
    else
        echo "  No task results available yet." >> "$report_file"
    fi

    cat >> "$report_file" << EOF

================================================================================
Log Files
================================================================================
EOF

    for quant in fp16 q8_0 q4_0 q4_k; do
        echo "  Server: $WORKSPACE/logs/server_${quant}.log" >> "$report_file"
        echo "  Eval:   $WORKSPACE/logs/eval_${quant}.log" >> "$report_file"
    done

    cat >> "$report_file" << EOF

================================================================================
Video Outputs
================================================================================
EOF

    for quant in fp16 q8_0 q4_0 q4_k; do
        local video_dir="$WORKSPACE/results/$quant/videos"
        local video_count=0
        if [ -d "$video_dir" ]; then
            video_count=$(find "$video_dir" -name "*.mp4" 2>/dev/null | wc -l)
        fi
        echo "  $quant: $video_count videos in $video_dir" >> "$report_file"
    done

    # Inference Timing Summary
    cat >> "$report_file" << EOF

================================================================================
Inference Timing Summary (mean ms per call)
================================================================================
EOF

    for quant in fp16 q8_0 q4_0 q4_k; do
        local profile_csv="$WORKSPACE/results/$quant/ode_step_profile.csv"
        echo "" >> "$report_file"
        echo "[$quant]" >> "$report_file"
        if [ -f "$profile_csv" ]; then
            # Extract timing table lines (from "Inference Chain Timing" block through blank #)
            grep "^#" "$profile_csv" \
                | grep -v "^# Columns:" \
                | sed 's/^# /  /' | sed 's/^#//' >> "$report_file"
        else
            echo "  (no profile — server may not have run with --ode-profile)" >> "$report_file"
        fi
    done

    # Per-Action-Token Step Change
    cat >> "$report_file" << EOF

================================================================================
Per-Action-Token Step Change (mean |dx| per horizon position, mean over all calls)
Grouped into 5 buckets of 10 horizon positions each (h0-9, h10-19, ..., h40-49)
================================================================================
EOF

    for quant in fp16 q8_0 q4_0 q4_k; do
        local profile_csv="$WORKSPACE/results/$quant/ode_step_profile.csv"
        echo "" >> "$report_file"
        echo "[$quant]" >> "$report_file"
        if [ -f "$profile_csv" ] && grep -q "^TOKEN_CHANGE" "$profile_csv"; then
            # Parse TOKEN_CHANGE rows and show bucketed summary (5 groups of 10 horizon positions)
            awk -F',' 'BEGIN{r=0}
            /^TOKEN_CHANGE/ {
                step=$2; t=$3;
                # Compute mean of each bucket of 10 columns (cols 4..53 = h0..h49)
                b0=0; b1=0; b2=0; b3=0; b4=0; n=0;
                for(i=4; i<=13 && i<=NF; i++) b0+=$i;
                for(i=14; i<=23 && i<=NF; i++) b1+=$i;
                for(i=24; i<=33 && i<=NF; i++) b2+=$i;
                for(i=34; i<=43 && i<=NF; i++) b3+=$i;
                for(i=44; i<=53 && i<=NF; i++) b4+=$i;
                if (r==0) {
                    printf "  %-4s %-6s %-10s %-10s %-10s %-10s %-10s\n",
                           "step","t","h0-9","h10-19","h20-29","h30-39","h40-49"
                    printf "  %s\n","--------------------------------------------------------------"
                    r=1
                }
                printf "  %-4s %-6s %-10.6f %-10.6f %-10.6f %-10.6f %-10.6f\n",
                       step,t,b0/10,b1/10,b2/10,b3/10,b4/10
            }' "$profile_csv" >> "$report_file"
        else
            echo "  (no data)" >> "$report_file"
        fi
    done

    echo "" >> "$report_file"
    echo "Report generated at: $report_file" >> "$report_file"

    log_success "Report saved to $report_file"
    echo ""
    cat "$report_file"
}

# =============================================================================
# Cleanup Handler
# =============================================================================
cleanup() {
    log_warn "Received interrupt signal, cleaning up..."
    stop_all_servers
    exit 1
}

# =============================================================================
# Main
# =============================================================================
main() {
    # Parse arguments
    parse_args "$@"

    echo "============================================================"
    echo "Pi0.5 Libero Quantization Evaluation - Benchmark"
    echo "============================================================"
    echo ""
    echo "Workspace:    $WORKSPACE"
    echo "Task Suite:   $TASK_SUITE"
    echo "Trials:       $NUM_TRIALS per task"
    echo "Mode:         $([ "$PARALLEL_MODE" = true ] && echo "Parallel" || echo "Serial")"
    echo "Quant Types:  ${QUANT_TYPES[*]}"
    echo "MUJOCO_GL:    $MUJOCO_GL"
    echo ""

    # Set up cleanup handler
    trap cleanup SIGINT SIGTERM

    # Ensure required directories exist
    mkdir -p "$WORKSPACE/logs"
    for quant in "${QUANT_TYPES[@]}"; do
        mkdir -p "$WORKSPACE/results/$quant/videos"
    done

    # Verify models exist
    for quant in "${QUANT_TYPES[@]}"; do
        if [ ! -f "$WORKSPACE/models/$quant/pi05.gguf" ]; then
            log_error "Model not found: $WORKSPACE/models/$quant/pi05.gguf"
            log_error "Run 02_prepare_models.sh first"
            exit 1
        fi
    done

    # Run evaluations
    local start_time=$(date +%s)

    if [ "$PARALLEL_MODE" = true ]; then
        run_parallel_eval
    else
        for quant in "${QUANT_TYPES[@]}"; do
            run_single_eval "$quant"
        done
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # Generate report
    generate_report

    echo ""
    echo "============================================================"
    log_success "Benchmark complete! (Duration: ${duration}s)"
    echo "============================================================"
    echo ""
    echo "Results: $WORKSPACE/results/"
    echo "Report:  $WORKSPACE/results/benchmark_report.txt"
    echo ""
}

main "$@"
