#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec accelerate launch --config_file configs/accelerate_config.yaml \
    --num_processes "${NUM_GPUS:-4}" -m src.training.rlvr.rlvr_grpo_trainer \
    --model_path "${MODEL_PATH:-models/checkpoints_sft}" \
    --dataset_path "${DATA_PATH:-data/rlvr_dataset/rlvr_train.jsonl}" \
    --output_dir "${OUTPUT_DIR:-models/checkpoints_rlvr}" \
    --num_train_epochs "${EPOCHS:-1}" \
    --per_device_train_batch_size "${BATCH_SIZE:-1}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-16}" \
    --learning_rate "${LEARNING_RATE:-1e-6}" \
    --num_generations "${NUM_GENERATIONS:-8}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-512}" \
    --temperature 0.9 --top_p 0.95 --kl_coef 0.02 \
    --freeze_vit true --freeze_merger true \
    --use_lora false --ds3_gather_for_generation true \
    --use_vllm "${USE_VLLM:-false}" \
    --report_to "${REPORT_TO:-tensorboard}" \
    --bf16 "$@"
