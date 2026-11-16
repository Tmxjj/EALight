#!/bin/bash
# SFT 训练启动脚本
# 使用前请根据实际环境修改以下路径变量：
#   model_path  — 基础模型目录（内含 config.json / tokenizer 等）
#   data_file   — SFT 训练数据（由 src/dataset/ 流水线生成的 .jsonl 文件）
#   output_dir  — checkpoint 输出目录

# DEBUG_MODE="-m debugpy --listen 127.0.0.1:5679 --wait-for-client"

# ============================================================
# 训练方法标识
# ============================================================
method=sft

# ============================================================
# 模型与数据路径（按实际环境修改）
# ============================================================
model_name=Qwen3-VL-8B-Instruct
model_path=models/${model_name}           # 基础模型目录
data_file=data/sft_dataset/sft_train.jsonl  # SFT 训练数据
output_base_dir=models/checkpoints_sft    # checkpoint 输出根目录

# ============================================================
# 训练超参数
# ============================================================
beta=0.1
num_train_epochs=3
per_device_train_batch_size=4
gradient_accumulation_steps=4
learning_rate=1e-5
num_gpus=4
batch_size=$((per_device_train_batch_size * gradient_accumulation_steps * num_gpus))

# ============================================================
# 输出目录
# ============================================================
run_name="sft_${model_name}-epoch${num_train_epochs}-batch${batch_size}-lr${learning_rate}"
output_dir="${output_base_dir}/${run_name}"
mkdir -p "${output_dir}"

echo "SFT run: ${run_name}"
echo "Output : ${output_dir}"

# ============================================================
# 启动训练
# ============================================================
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

accelerate launch \
    --config_file "${PROJECT_ROOT}/configs/accelerate_config.yaml" \
    "${PROJECT_ROOT}/src/training/trainer.py" \
    --method                       ${method} \
    --model_path                   ${model_path} \
    --dataset_path                 ${data_file} \
    --output_dir                   ${output_dir} \
    --per_device_train_batch_size  ${per_device_train_batch_size} \
    --gradient_accumulation_steps  ${gradient_accumulation_steps} \
    --beta                         ${beta} \
    --learning_rate                ${learning_rate} \
    --num_train_epochs             ${num_train_epochs}
