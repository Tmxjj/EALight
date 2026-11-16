# ——————LoRA adapter 未合并：自动 merge 后再用 vLLM 启动————————————

has_model_weights() {
    local model_dir="$1"
    find "$model_dir" -maxdepth 1 \( \
        -name "*.safetensors" -o \
        -name "model.safetensors.index.json" -o \
        -name "pytorch_model*.bin" -o \
        -name "pytorch_model.bin.index.json" \
    \) -print -quit | grep -q .
}

merge_lora_if_needed() {
    local base_model_path="$1"
    local adapter_path="$2"
    local merged_model_path="$3"

    if [[ -f "${merged_model_path}/config.json" ]] && has_model_weights "${merged_model_path}"; then
        echo "[LoRA Merge] 已检测到合并后的模型，跳过 merge: ${merged_model_path}"
        return 0
    fi

    echo "[LoRA Merge] 开始合并 LoRA adapter"
    echo "[LoRA Merge] base model : ${base_model_path}"
    echo "[LoRA Merge] adapter    : ${adapter_path}"
    echo "[LoRA Merge] output     : ${merged_model_path}"

    BASE_MODEL_PATH="${base_model_path}" \
    ADAPTER_PATH="${adapter_path}" \
    MERGED_MODEL_PATH="${merged_model_path}" \
    python - <<'PY'
import os
import shutil
from pathlib import Path

import torch
import transformers
from peft import PeftModel
from transformers import AutoConfig, AutoTokenizer, AutoProcessor

base_model_path = Path(os.environ["BASE_MODEL_PATH"]).expanduser()
adapter_path = Path(os.environ["ADAPTER_PATH"]).expanduser()
merged_model_path = Path(os.environ["MERGED_MODEL_PATH"]).expanduser()
tmp_path = Path(str(merged_model_path) + "_tmp")

if not base_model_path.exists():
    raise FileNotFoundError(f"Base model path not found: {base_model_path}")
if not adapter_path.exists():
    raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

if tmp_path.exists():
    shutil.rmtree(tmp_path)
tmp_path.mkdir(parents=True, exist_ok=True)

# 尽量兼容纯文本 Qwen / Qwen-VL：
# 1) 新版 transformers 若支持 AutoModelForImageTextToText，优先尝试
# 2) 否则回退到 AutoModelForVision2Seq / AutoModelForCausalLM
model_classes = []
for cls_name in [
    "AutoModelForImageTextToText",
    "AutoModelForVision2Seq",
    "AutoModelForCausalLM",
]:
    cls = getattr(transformers, cls_name, None)
    if cls is not None:
        model_classes.append((cls_name, cls))

last_err = None
base_model = None

for cls_name, cls in model_classes:
    try:
        print(f"[LoRA Merge] 尝试使用 {cls_name} 加载 base model")
        base_model = cls.from_pretrained(
            str(base_model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"[LoRA Merge] base model 加载成功: {cls_name}")
        break
    except Exception as e:
        last_err = e
        print(f"[LoRA Merge] {cls_name} 加载失败，尝试下一个。原因: {repr(e)}")

if base_model is None:
    raise RuntimeError(f"无法加载 base model，最后一次错误: {repr(last_err)}")

print("[LoRA Merge] 加载 adapter")
model = PeftModel.from_pretrained(
    base_model,
    str(adapter_path),
    is_trainable=False,
)

print("[LoRA Merge] merge_and_unload")
model = model.merge_and_unload(safe_merge=True)

print("[LoRA Merge] 保存合并后的完整模型")
model.save_pretrained(
    str(tmp_path),
    safe_serialization=True,
    max_shard_size="4GB",
)

# 保存 tokenizer / processor，vLLM 需要 tokenizer_config、chat_template 等文件
for loader_name, loader in [
    ("AutoTokenizer", AutoTokenizer),
    ("AutoProcessor", AutoProcessor),
]:
    try:
        obj = loader.from_pretrained(str(base_model_path), trust_remote_code=True)
        obj.save_pretrained(str(tmp_path))
        print(f"[LoRA Merge] 已保存 {loader_name}")
    except Exception as e:
        print(f"[LoRA Merge] 跳过 {loader_name}: {repr(e)}")

# 复制常见配置文件；已存在则不覆盖
for filename in [
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.json",
]:
    src = base_model_path / filename
    dst = tmp_path / filename
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)

if merged_model_path.exists():
    shutil.rmtree(merged_model_path)

tmp_path.rename(merged_model_path)
print(f"[LoRA Merge] 完成: {merged_model_path}")
PY
}

# 启动第 4 个实例 (GPU 3, 端口 8003)
RLVR_BASE_MODEL_PATH="<SERVER_PATH>/results/checkpoints_rlvr/rlvr_qwen3-G8-kl0.02-epoch1-batch64-lr1e-6-lora/checkpoint-50"
RLVR_ADAPTER_PATH="<SERVER_PATH>/results/checkpoints_rlvr/rlvr_qwen3-G8-kl0.02-epoch1-batch64-lr1e-6-lora/checkpoint-50/ref"
RLVR_MERGED_MODEL_PATH="<SERVER_PATH>/results/checkpoints_rlvr/rlvr_qwen3-G8-kl0.02-epoch1-batch64-lr1e-6-lora/checkpoint-50/merged-ref"

CUDA_VISIBLE_DEVICES=3 merge_lora_if_needed \
    "${RLVR_BASE_MODEL_PATH}" \
    "${RLVR_ADAPTER_PATH}" \
    "${RLVR_MERGED_MODEL_PATH}"

CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
    --model "${RLVR_MERGED_MODEL_PATH}" \
    --trust-remote-code --dtype bfloat16 --gpu-memory-utilization 0.8 \
    --max-model-len 8192 --max-num-seqs 16 \
    --served-model-name rlvr_qwen3-8b-G8-kl0.02-epoch1-batch64-lr1e-6-lora-checkpoint-50-merged-ref \
    --port 8003 &