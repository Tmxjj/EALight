
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================
# 训练方法标识
# ============================================================
method=rlvr

# ============================================================
# 模型与数据路径
# ============================================================
# 建议从 SFT checkpoint 初始化，充分利用已有的指令跟随能力
# 请将下面的路径替换为实际的 SFT checkpoint 目录
model_path=models/checkpoints_sft/<sft_run_name>
# RLVR 训练数据集（由 src/training/rlvr/rlvr_data_collector.py 生成）
data_file=data/rlvr_dataset/rlvr_train.jsonl
output_base_dir=models/checkpoints_${method}

# ============================================================
# GRPO 核心超参数
# ============================================================
num_generations=8        # 每个 prompt 采样 G=8 个响应（GRPO 组内相对优势）
max_new_tokens=512        # CoT 推理链最大生成长度
temperature=0.9           # 采样温度（高温鼓励探索多样化推理）
top_p=0.95
kl_coef=0.02              # KL 惩罚系数（防止策略严重偏离参考模型）
ds3_gather_for_generation="false"  # 4x80GB 推荐 true；显存不足可设 false 换取更低峰值、更慢生成
mask_truncated_completions="true" # max token 受限时避免截断 completion 贡献噪声 loss
num_iterations=1          # GRPO μ；每个 batch 采样后执行几轮 policy update，默认 1 最稳

# ============================================================
# 基础训练超参数
# ============================================================
num_train_epochs=1
max_steps=-1            # smoke run 可设为 20-50；-1 表示按 epoch 训练
# GRPO 每步需要前向推理 G=8 次，显存压力远高于 DPO
# 建议 per_device_train_batch_size=1，靠梯度累积扩大有效 batch
per_device_train_batch_size=1
gradient_accumulation_steps=16
# GRPO 在线策略更新，学习率比 DPO 更小，防止快速遗忘
learning_rate=1e-6
num_gpus=4
# max_prompt_length：TRL 0.29.1 GRPOConfig 不包含此字段，传入后会被 _filter_kwargs_for_callable 静默过滤。
# 保留此变量是为了兼容旧调用方式；prompt 截断需在数据预处理阶段（rlvr_data_collector.py）手动控制。
max_prompt_length=3584

lr_scheduler_type=cosine
warmup_ratio=0.03
warmup_steps=0

# ============================================================
# 日志与 WandB / TensorBoard
# ============================================================
# 设置 report_to=wandb 并配置 WANDB_PROJECT/WANDB_ENTITY 即可上报到 W&B
# 设置 report_to=tensorboard 仅写本地 TensorBoard（默认）
# 可同时使用：report_to="wandb,tensorboard"
report_to=wandb
wandb_project=VLMTraffic
wandb_entity=your-wandb-entity   # 填你的 W&B username 或 team name
logging_steps=1

# ============================================================
# vLLM 生成加速
# ============================================================
# 当前全参数 ZeRO-3 训练默认使用 transformers generate。
# 若 use_vllm=true 且 vllm_mode=server，必须用独立 GPU 启动 trl vllm-serve。
use_vllm="true"
vllm_mode="colocate"       # colocate 或 server
vllm_server_host="127.0.0.1"
vllm_server_port=8000
vllm_gpu_memory_utilization=0.35
# ValueError: The decoder prompt (length 5822) is longer than the maximum model length of 4096. Make sure that `max_model_len` is no smaller than the number of text tokens plus multimodal tokens. For image inputs, the number of image tokens depends on the number of images, and possibly their aspect ratios as well.
vllm_max_model_length=6656

# ============================================================
# LoRA 超参数
# ============================================================
use_lora="false"
lora_r=16
lora_alpha=32
lora_dropout=0.05

# ============================================================
# 参数冻结策略（与 DPO 保持一致）
# ============================================================
freeze_vit="true"
freeze_merger="true"

# ============================================================
# 计算全局 Batch Size（仅用于目录命名）
# ============================================================
batch_size=$((per_device_train_batch_size * gradient_accumulation_steps * num_gpus))

# 是否使用 LoRA，用于 run_name 区分实验
if [ "${use_lora}" = "true" ]; then
    lora_tag="lora-r${lora_r}-a${lora_alpha}"
else
    lora_tag="full"
fi

# 训练启动时间，用于 run_name 区分不同运行
run_time=$(date +"%Y%m%d-%H%M%S")

run_name="${method}_qwen3-G${num_generations}-kl${kl_coef}-epoch${num_train_epochs}-batch${batch_size}-lr${learning_rate}-${lora_tag}-${run_time}"
run_dir="${output_base_dir}/${run_name}"
logging_dir="${run_dir}/tensorboard"

# 确保输出目录存在
mkdir -p ${run_dir}

echo "RLVR run_dir: ${run_dir}"
echo "TensorBoard log_dir: ${logging_dir}"
echo "TensorBoard command: tensorboard --logdir ${logging_dir} --host 0.0.0.0 --port 6006"
echo "W&B project: ${wandb_project} / entity: ${wandb_entity} / run: ${run_name}"

# 将 W&B 配置注入环境变量（W&B SDK 从环境变量中读取这些值）
export WANDB_PROJECT=${wandb_project}
export WANDB_ENTITY=${wandb_entity}
export WANDB_RUN_NAME=${run_name}

# ============================================================
# 启动 GRPO 训练
# ============================================================
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

accelerate launch \
    --config_file "${PROJECT_ROOT}/configs/accelerate_config.yaml" \
    --num_processes ${num_gpus} \
    src/training/rlvr/rlvr_grpo_trainer.py \
    --model_path      ${model_path} \
    --dataset_path    ${data_file} \
    --output_dir      ${run_dir} \
    --num_generations       ${num_generations} \
    --max_new_tokens        ${max_new_tokens} \
    --temperature           ${temperature} \
    --top_p                 ${top_p} \
    --kl_coef               ${kl_coef} \
    --num_train_epochs      ${num_train_epochs} \
    --max_steps             ${max_steps} \
    --per_device_train_batch_size  ${per_device_train_batch_size} \
    --gradient_accumulation_steps  ${gradient_accumulation_steps} \
    --learning_rate         ${learning_rate} \
    --max_prompt_length     ${max_prompt_length} \
    --logging_steps         ${logging_steps} \
    --logging_dir           ${logging_dir} \
    --report_to             ${report_to} \
    --run_name              ${run_name} \
    --freeze_vit            ${freeze_vit} \
    --freeze_merger         ${freeze_merger} \
    --use_lora              ${use_lora} \
    --lora_r                ${lora_r} \
    --lora_alpha            ${lora_alpha} \
    --lora_dropout          ${lora_dropout} \
    --ds3_gather_for_generation ${ds3_gather_for_generation} \
    $( [ "${mask_truncated_completions}" = "true" ] && printf %s "--mask_truncated_completions" ) \
    --use_vllm              ${use_vllm} \
    --vllm_mode             ${vllm_mode} \
    --vllm_server_host      ${vllm_server_host} \
    --vllm_server_port      ${vllm_server_port} \
    --vllm_gpu_memory_utilization ${vllm_gpu_memory_utilization} \
    --vllm_max_model_length ${vllm_max_model_length} \
    --num_iterations        ${num_iterations} \
    --lr_scheduler_type     ${lr_scheduler_type} \
    --warmup_ratio          ${warmup_ratio} \
    --warmup_steps          ${warmup_steps} \
    --bf16
