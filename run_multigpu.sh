#!/bin/bash
# Launch script for multi-GPU FP8 training

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================================
# Configuration
# ============================================================================

# Get number of GPUs
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}
MASTER_PORT=${MASTER_PORT:-29500}

# ============================================================================
# Helper Functions
# ============================================================================

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# Launch Functions
# ============================================================================

# Standard DDP training (small models)
launch_ddp_training() {
    local model_name=$1
    local use_te=$2
    local batch_size=$3
    
    print_info "Launching DDP training on $NUM_GPUS GPUs"
    print_info "Model: $model_name"
    print_info "Batch size per GPU: $batch_size"
    
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        train_multi_gpu.py \
        --model_name "$model_name" \
        --dataset_name nvidia/OpenMathInstruct-2 \
        --split_name train_1M \
        --batch_size "$batch_size" \
        --gradient_accumulation_steps 2 \
        --mixed_precision fp8 \
        --max_seq_length 1024 \
        --num_epochs 3 \
        --learning_rate 1e-5 \
        --num_warmup_steps 100 \
        --output_dir "./checkpoints/${model_name##*/}-fp8" \
        --log_interval 10 \
        --use_wandb \
        --wandb_project llm-fp8-multi \
        --wandb_run_name "${model_name##*/}-fp8-${NUM_GPUS}gpu" \
        $use_te
}

# FSDP training (large models)
launch_fsdp_training() {
    local model_name=$1
    local use_te=$2
    local batch_size=$3
    
    print_info "Launching FSDP training on $NUM_GPUS GPUs"
    print_info "Model: $model_name"
    print_info "Batch size per GPU: $batch_size"
    
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        train_multi_gpu.py \
        --model_name "$model_name" \
        --dataset_name nvidia/OpenMathInstruct-2 \
        --split_name train_1M \
        --batch_size "$batch_size" \
        --gradient_accumulation_steps 4 \
        --mixed_precision fp8 \
        --max_seq_length 1024 \
        --num_epochs 3 \
        --learning_rate 5e-6 \
        --num_warmup_steps 200 \
        --use_fsdp \
        --gradient_checkpointing \
        --output_dir "./checkpoints/${model_name##*/}-fp8-fsdp" \
        --log_interval 10 \
        --use_wandb \
        --wandb_project llm-fp8-multi \
        --wandb_run_name "${model_name##*/}-fp8-fsdp-${NUM_GPUS}gpu" \
        $use_te
}

# Debug mode (single GPU, small batch)
launch_debug() {
    local model_name=$1
    local use_te=$2
    
    print_info "Launching DEBUG mode on 1 GPU"
    print_info "Model: $model_name"
    
    CUDA_VISIBLE_DEVICES=0 python train_multi_gpu.py \
        --model_name "$model_name" \
        --dataset_name nvidia/OpenMathInstruct-2 \
        --split_name train_1M \
        --num_of_samples 100 \
        --batch_size 1 \
        --gradient_accumulation_steps 1 \
        --mixed_precision fp8 \
        --max_seq_length 512 \
        --num_epochs 1 \
        --learning_rate 1e-5 \
        --output_dir "./debug_output" \
        --log_interval 1 \
        --debug_mode \
        $use_te
}

# ============================================================================
# Preset Configurations
# ============================================================================

case "$1" in
    # Llama 3.2 1B
    "llama-1b")
        launch_ddp_training "meta-llama/Llama-3.2-1B" "--use_te" 12
        ;;
    
    # Llama 3.2 3B
    "llama-3b")
        launch_ddp_training "meta-llama/Llama-3.2-3B" "--use_te" 8
        ;;
    
    # Llama 3.1 8B
    "llama-8b")
        if [ "$NUM_GPUS" -ge 4 ]; then
            launch_ddp_training "meta-llama/Meta-Llama-3.1-8B" "--use_te" 4
        else
            print_warning "8B model needs at least 4 GPUs, using FSDP"
            launch_fsdp_training "meta-llama/Meta-Llama-3.1-8B" "--use_te" 2
        fi
        ;;
    
    # Qwen 14B (FSDP required)
    "qwen-14b")
        launch_fsdp_training "Qwen/Qwen2.5-14B" "--use_te" 2
        ;;
    
    # Debug modes
    "debug-llama")
        launch_debug "meta-llama/Llama-3.2-1B" "--use_te"
        ;;
    
    "debug-qwen")
        launch_debug "Qwen/Qwen2.5-1.5B" "--use_te"
        ;;
    
    # Custom configuration
    "custom")
        if [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]; then
            print_error "Usage: $0 custom <model_name> <batch_size> <use_te:0|1> [use_fsdp:0|1]"
            exit 1
        fi
        
        model_name=$2
        batch_size=$3
        use_te_flag=""
        if [ "$4" = "1" ]; then
            use_te_flag="--use_te"
        fi
        
        if [ "$5" = "1" ]; then
            launch_fsdp_training "$model_name" "$use_te_flag" "$batch_size"
        else
            launch_ddp_training "$model_name" "$use_te_flag" "$batch_size"
        fi
        ;;
    
    *)
        echo "Multi-GPU FP8 Training Launcher"
        echo "================================"
        echo ""
        echo "Usage: $0 <preset|custom> [options]"
        echo ""
        echo "Presets:"
        echo "  llama-1b    - Llama 3.2 1B with TE"
        echo "  llama-3b    - Llama 3.2 3B with TE"
        echo "  llama-8b    - Llama 3.1 8B with TE"
        echo "  qwen-14b    - Qwen 2.5 14B with TE (FSDP)"
        echo "  debug-llama - Debug mode with Llama 1B"
        echo "  debug-qwen  - Debug mode with Qwen 1.5B"
        echo ""
        echo "Custom:"
        echo "  custom <model_name> <batch_size> <use_te:0|1> [use_fsdp:0|1]"
        echo ""
        echo "Environment Variables:"
        echo "  NUM_GPUS     - Number of GPUs to use (default: all)"
        echo "  MASTER_PORT  - Master port for distributed training (default: 29500)"
        echo ""
        echo "Examples:"
        echo "  $0 llama-3b                                         # Train Llama 3B"
        echo "  NUM_GPUS=4 $0 llama-8b                             # Train Llama 8B on 4 GPUs"
        echo "  $0 custom meta-llama/Llama-3.2-3B 8 1 0            # Custom DDP"
        echo "  $0 custom Qwen/Qwen2.5-14B 2 1 1                   # Custom FSDP"
        exit 1
        ;;
esac

print_info "Training launched successfully!"