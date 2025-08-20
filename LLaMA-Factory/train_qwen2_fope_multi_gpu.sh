#!/bin/bash

# Multi-GPU training script for Qwen2.5-0.5B with FoPE on 4x RTX 4090 using LLaMA-Factory
# Usage: ./train_qwen2_fope_multi_gpu.sh [config_name]

set -e

# Default configuration
CONFIG_NAME=${1:-"train_qwen2_fope_optimized"}
CONFIG_FILE="./examples/${CONFIG_NAME}.yaml"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Configuration file not found: $CONFIG_FILE"
    echo "Available configs:"
    ls -la ./examples/train_qwen2_fope*.yaml
    exit 1
fi

echo "🚀 Starting Qwen2.5-0.5B FoPE pretraining on 4x RTX 4090 using LLaMA-Factory..."
echo "📁 Config file: $CONFIG_FILE"
echo "🔧 Model: Qwen2.5-0.5B with FoPE"
echo "📊 Stage: Pretraining (PT)"
echo "🖥️  GPUs: 4x RTX 4090 (24GB each)"
echo "🏭 Framework: LLaMA-Factory (built-in multi-GPU)"

# Check CUDA availability and GPU count
if command -v nvidia-smi &> /dev/null; then
    echo "✅ CUDA detected"
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "🔍 Found $GPU_COUNT GPU(s)"
    
    if [ $GPU_COUNT -lt 4 ]; then
        echo "⚠️  Warning: Found only $GPU_COUNT GPU(s), but config is optimized for 4x RTX 4090"
        echo "   Consider adjusting batch sizes in the config file"
    fi
    
    echo "📊 GPU Memory Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv,noheader,nounits
else
    echo "❌ CUDA not detected, cannot run multi-GPU training"
    exit 1
fi

# Set environment variables for optimal multi-GPU performance
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_BLOCKING_WAIT=1

echo "🔧 Setting up multi-GPU environment..."
echo "   CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "   NCCL_DEBUG: $NCCL_DEBUG"

# Calculate effective batch size
TRAIN_BATCH_SIZE=$(grep "per_device_train_batch_size:" "$CONFIG_FILE" | awk '{print $2}')
ACCUMULATION_STEPS=$(grep "gradient_accumulation_steps:" "$CONFIG_FILE" | awk '{print $2}')
EFFECTIVE_BATCH_SIZE=$((TRAIN_BATCH_SIZE * 4 * ACCUMULATION_STEPS))

echo "📊 Training Configuration:"
echo "   Per-device batch size: $TRAIN_BATCH_SIZE"
echo "   Gradient accumulation: $ACCUMULATION_STEPS"
echo "   Effective batch size: $EFFECTIVE_BATCH_SIZE"
echo "   Sequence length: $(grep "max_seq_length:" "$CONFIG_FILE" | awk '{print $2}')"

# Start training with LLaMA-Factory's built-in multi-GPU support
echo "🎯 Starting multi-GPU training with LLaMA-Factory..."
python src/train.py \
    --config_file "$CONFIG_FILE" \
    --output_dir "./saves/qwen2-0.5b-fope-pretrain-multi-gpu" \
    --overwrite_output_dir \
    --logging_steps 25 \
    --save_steps 500 \
    --eval_steps 500 \
    --dataloader_num_workers 8 \
    --ddp_find_unused_parameters false \
    --ddp_backend nccl \
    --ddp_bucket_cap_mb 25 \
    --ddp_broadcast_buffers false \
    --ddp_static_graph true

echo "✅ Multi-GPU training completed!"
echo "📁 Checkpoints saved to: ./saves/qwen2-0.5b-fope-pretrain-multi-gpu"
echo "🎉 FoPE model successfully trained on 4x RTX 4090 using LLaMA-Factory!"
