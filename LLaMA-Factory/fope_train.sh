#!/bin/bash
set -x -e

# Basic environment setup
export WANDB_DISABLED=true
export CPUS_PER_TASK=8

# FoPE specific configuration
export JOB_NAME=fope
export output_dir=cache/Qwen2-${JOB_NAME}-pretrain
export model_name_or_path=./Qwen2-0.5B-Instruct

# Training parameters
export num_gpus=1
export full_batch_size=16
export batch_size=8
export gradient_accumulation_steps=$[$full_batch_size/($batch_size*$num_gpus)]

echo "Starting FoPE training with:"
echo "  Model: $model_name_or_path"
echo "  Output: $output_dir"
echo "  Rope mode: $JOB_NAME"
echo "  Batch size: $batch_size"
echo "  Gradient accumulation: $gradient_accumulation_steps"

# Run training following VideoRope pattern
python LLaMA-Factory/src/train.py \
    --model_name_or_path $model_name_or_path \
    --stage pt \
    --do_train true \
    --finetuning_type full \
    --dataset identity \
    --template qwen \
    --output_dir $output_dir \
    --num_train_epochs 5.0 \
    --logging_steps 10 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --overwrite_output_dir true \
    --per_device_train_batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate 1.0e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 true \
    --ddp_timeout 180000000 \
    --val_size 0.1 \
    --per_device_eval_batch_size 4 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --max_seq_length 2048 \
    --which_rope ${JOB_NAME} \
    --fourier_learnable true \
    --fourier_init eye_xavier_norm \
    --fourier_dim 0 \
    --fourier_init_norm_gain 0.3 \
    --fourier_separate_basis true \
    --fourier_separate_head true \
    --fourier_norm false \
    --fourier_ignore_zero true \
    --report_to none