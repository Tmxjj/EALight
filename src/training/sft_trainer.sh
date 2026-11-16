#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec accelerate launch --config_file configs/accelerate_config.yaml \
    --num_processes "${NUM_GPUS:-4}" -m src.training.trainer \
    --method sft \
    --model_path "${MODEL_PATH:-models/Qwen3-VL-8B-Instruct}" \
    --dataset_path "${DATA_PATH:-data/sft_dataset/sft_train.jsonl}" \
    --output_dir "${OUTPUT_DIR:-models/checkpoints_sft}" \
    --per_device_train_batch_size "${BATCH_SIZE:-4}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
    --learning_rate "${LEARNING_RATE:-1e-5}" \
    --num_train_epochs "${EPOCHS:-3}" \
    --report_to "${REPORT_TO:-tensorboard}" "$@"
