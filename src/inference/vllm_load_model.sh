#!/bin/bash
# vLLM 推理服务启动脚本（OpenAI 兼容接口）
#
# 使用前请修改 MODEL_PATH 为本地模型目录。
# 使用 scripts/download_model.py 或 HuggingFace 下载后，默认路径为：
#   models/Qwen3-VL-8B-Instruct        （基础模型，用于流程验证）
#   models/checkpoints_sft/<run_name>  （SFT checkpoint，用于完整复现）
#
# 用法示例：
#   bash src/inference/vllm_load_model.sh
#   bash src/inference/vllm_load_model.sh --gpu 0 --port 8000
#
# 默认启动后，评测脚本通过 http://localhost:8000/v1/chat/completions 调用。

# ─── 可配置参数 ────────────────────────────────────────────────────────────────
GPU_ID=0
PORT=8000
MODEL_PATH=models/Qwen3-VL-8B-Instruct   # 修改为实际模型路径
SERVED_MODEL_NAME=qwen3-vl-8b             # 需与 configs/model_config.py 中 model_name 一致
GPU_MEM_UTIL=0.85
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=8

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu)   GPU_ID="$2"; shift ;;
        --port)  PORT="$2";   shift ;;
        --model) MODEL_PATH="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "启动 vLLM 推理服务..."
echo "  GPU            : ${GPU_ID}"
echo "  Port           : ${PORT}"
echo "  Model          : ${MODEL_PATH}"
echo "  Served name    : ${SERVED_MODEL_NAME}"

CUDA_VISIBLE_DEVICES=${GPU_ID} python -m vllm.entrypoints.openai.api_server \
    --model                   "${MODEL_PATH}" \
    --trust-remote-code \
    --dtype                   bfloat16 \
    --gpu-memory-utilization  ${GPU_MEM_UTIL} \
    --max-model-len           ${MAX_MODEL_LEN} \
    --max-num-seqs            ${MAX_NUM_SEQS} \
    --served-model-name       "${SERVED_MODEL_NAME}" \
    --port                    ${PORT}
