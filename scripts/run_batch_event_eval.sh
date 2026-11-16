#!/bin/bash
# 事件场景批量评测脚本
# 覆盖所有包内数据集（JiNan / SouthKorea_Songdo / France_Massy / Hongkong_YMT）
# 的两类事件路由：emergy_bus（紧急车辆+公交）和 accident_debris（事故+占道）
#
# 用法示例：
#   bash scripts/run_batch_event_eval.sh --baseline-only
#   bash scripts/run_batch_event_eval.sh --port 8000 --model_name ealight-8b
#   bash scripts/run_batch_event_eval.sh --port 8000 --model_name ealight-8b --with-baseline
#   bash scripts/run_batch_event_eval.sh --event emergy_bus --baseline-only
#   bash scripts/run_batch_event_eval.sh --port 8000 --model_name ealight-8b --gpu_id 1

set -uo pipefail
FAILED_TASKS=0

# ─── 默认配置 ──────────────────────────────────────────────────────────────────
BUFFER_S=180
MIN_SUMO_SECONDS=300
MAX_SUMO_SECONDS_CAP=3600
LOG_DIR="./log/eval_results"
SEED_IDS=(42)

API_PORT=""
MODEL_NAME=""
RESULT_NAME=""
TEMPERATURE=""
MAX_NEW_TOKENS=""
PROMPT_BUILDER="configs.prompt_builder"
EVENT_BULLETIN="auto"
WARM_UP_TIME=""
ROBUST=""
ROBUST_LEVEL=""
GPU_ID=""
BASELINE_ONLY=false
WITH_BASELINE=false
EVENT_FILTER=""   # 可选：emergy_bus / accident_debris（不指定则两类都跑）

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --baseline-only|--with-baseline) ;;
        *) if [[ "$#" -lt 2 || "$2" == --* ]]; then
               echo "Missing value for $1" >&2; exit 2
           fi ;;
    esac
    case $1 in
        --port)           API_PORT="$2";       shift ;;
        --model_name)     MODEL_NAME="$2";     shift ;;
        --result_name)    RESULT_NAME="$2";    shift ;;
        --temperature)    TEMPERATURE="$2";    shift ;;
        --max_new_tokens) MAX_NEW_TOKENS="$2"; shift ;;
        --prompt_builder) PROMPT_BUILDER="$2"; shift ;;
        --event_bulletin) EVENT_BULLETIN="$2"; shift ;;
        --warm_up_time)   WARM_UP_TIME="$2";   shift ;;
        --robust)         ROBUST="$2";         shift ;;
        --robust_level)   ROBUST_LEVEL="$2";   shift ;;
        --gpu_id)         GPU_ID="$2";         shift ;;
        --baseline-only)  BASELINE_ONLY=true   ;;
        --with-baseline)  WITH_BASELINE=true   ;;
        --event)          EVENT_FILTER="$2";   shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR"
if [[ -n "$EVENT_FILTER" && "$EVENT_FILTER" != emergy_bus && "$EVENT_FILTER" != accident_debris ]]; then
    echo "Unsupported event: $EVENT_FILTER" >&2
    exit 2
fi

# ─── 数据集条目 ────────────────────────────────────────────────────────────────
# 格式：DATASET|ENV_DIR|BASE_STEM
# 事件路由文件 = ENV_DIR/${BASE_STEM}_{emergy_bus|accident_debris}.rou.xml
declare -a EVENT_ENTRIES

EVENT_ENTRIES+=("JiNan|data/raw/JiNan/env|anon_3_4_jinan_real")
EVENT_ENTRIES+=("SouthKorea_Songdo|data/raw/SouthKorea_Songdo/env|songdo")
EVENT_ENTRIES+=("France_Massy|data/raw/France_Massy/env|massy")
EVENT_ENTRIES+=("Hongkong_YMT|data/raw/Hongkong_YMT/env|YMT")

# 未包含在数据包中（如需使用，补充数据后取消注释）：
# EVENT_ENTRIES+=("Hangzhou|data/raw/Hangzhou/env|anon_4_4_hangzhou_real")
# EVENT_ENTRIES+=("NewYork|data/raw/NewYork/env|anon_28_7_newyork_real_double")

EVENT_TYPES=("emergy_bus" "accident_debris")

# ─── 工具函数 ──────────────────────────────────────────────────────────────────

get_max_sumo_seconds() {
    python3 - "$1" "$BUFFER_S" "$MIN_SUMO_SECONDS" "$MAX_SUMO_SECONDS_CAP" <<'PY_HORIZON'
import math
import sys
import xml.etree.ElementTree as ET
route, buffer, minimum, cap = sys.argv[1:]
buffer, minimum, cap = map(int, (buffer, minimum, cap))
try:
    root = ET.parse(route).getroot()
    times = []
    for tag in ("vehicle", "trip", "flow"):
        for vehicle in root.findall(tag):
            value = vehicle.get("end") if tag == "flow" else vehicle.get("depart")
            value = value or vehicle.get("begin")
            try:
                number = float(value)
                if math.isfinite(number):
                    times.append(number)
            except (ValueError, TypeError):
                pass
    print(min(cap, max(minimum, int(max(times) + buffer))) if times else minimum)
except (OSError, ET.ParseError):
    print(cap)
PY_HORIZON
}

cleanup_sumo() { sleep 5; }

run_baseline() {
    local DATASET="$1"
    local ROUTE_FILE="$2"
    local EVENT_TYPE="$3"
    local METHOD_FLAG="$4"
    local METHOD_NAME="$5"
    local MAX_SUMO_S="$6"
    local SEED_ID="$7"
    local WARM_UP_ARGS=()
    [ -n "$WARM_UP_TIME" ] && WARM_UP_ARGS=(--warm_up_time "$WARM_UP_TIME")

    cleanup_sumo
    echo "  [${METHOD_NAME}] ${DATASET}/${ROUTE_FILE}  sumo_s=${MAX_SUMO_S}  seed=${SEED_ID}"
    python src/evaluation/run_eval.py \
        --scenario         "$DATASET" \
        --log_dir          "$LOG_DIR" \
        --route_file       "$ROUTE_FILE" \
        --scene_type       "$EVENT_TYPE" \
        --max_sumo_seconds "$MAX_SUMO_S" \
        --sumo_seed        "$SEED_ID" \
        "${WARM_UP_ARGS[@]}" \
        "$METHOD_FLAG"
    if [[ $? -ne 0 ]]; then
        FAILED_TASKS=$((FAILED_TASKS + 1))
        echo "  [WARNING] ${METHOD_NAME} failed: ${DATASET}/${ROUTE_FILE}"
    fi
}

run_vlm() {
    local DATASET="$1"
    local ROUTE_FILE="$2"
    local EVENT_TYPE="$3"
    local MAX_SUMO_S="$4"
    local SEED_ID="$5"
    local WARM_UP_ARGS=()
    [ -n "$WARM_UP_TIME" ] && WARM_UP_ARGS=(--warm_up_time "$WARM_UP_TIME")

    cleanup_sumo
    echo "  [VLM] ${DATASET}/${ROUTE_FILE}  sumo_s=${MAX_SUMO_S}  seed=${SEED_ID}"
    "${PROJECT_ROOT}/vgl_python.sh" src/evaluation/run_eval.py \
        --scenario         "$DATASET" \
        --log_dir          "$LOG_DIR" \
        --route_file       "$ROUTE_FILE" \
        --scene_type       "$EVENT_TYPE" \
        --max_sumo_seconds "$MAX_SUMO_S" \
        --sumo_seed        "$SEED_ID" \
        "${WARM_UP_ARGS[@]}" \
        "${EXTRA_VLM_ARGS[@]}"
    if [[ $? -ne 0 ]]; then
        FAILED_TASKS=$((FAILED_TASKS + 1))
        echo "  [WARNING] VLM failed: ${DATASET}/${ROUTE_FILE}"
    fi
}

# ─── 构建 VLM 额外参数 ─────────────────────────────────────────────────────────
EXTRA_VLM_ARGS=()
[ -n "$API_PORT" ] && EXTRA_VLM_ARGS+=(--api_url "http://localhost:${API_PORT}/v1/chat/completions")
[ -n "$MODEL_NAME" ] && EXTRA_VLM_ARGS+=(--model_name "$MODEL_NAME")
[ -n "$TEMPERATURE" ] && EXTRA_VLM_ARGS+=(--temperature "$TEMPERATURE")
[ -n "$MAX_NEW_TOKENS" ] && EXTRA_VLM_ARGS+=(--max_new_tokens "$MAX_NEW_TOKENS")
EXTRA_VLM_ARGS+=(--prompt_builder "$PROMPT_BUILDER" --event_bulletin "$EVENT_BULLETIN")
[ -n "$RESULT_NAME" ] && EXTRA_VLM_ARGS+=(--result_name "$RESULT_NAME")
[ -n "$GPU_ID" ] && EXTRA_VLM_ARGS+=(--gpu_id "$GPU_ID")
[ -n "$ROBUST" ] && EXTRA_VLM_ARGS+=(--robust_perturb "$ROBUST")
[ -n "$ROBUST_LEVEL" ] && EXTRA_VLM_ARGS+=(--robust_level "$ROBUST_LEVEL")

# ─── 打印头部信息 ──────────────────────────────────────────────────────────────
echo "================================================================"
echo "  EALight 事件场景批量评测"
echo "  LOG_DIR: $LOG_DIR"
echo "  Seeds  : ${SEED_IDS[*]}"
if [ "$BASELINE_ONLY" = true ]; then
    echo "  模式   : 仅基线 (FixedTime + MaxPressure + Random)"
elif [ -n "$MODEL_NAME" ]; then
    echo "  模式   : VLM [${MODEL_NAME}]$([ "$WITH_BASELINE" = true ] && echo " + 基线")"
fi
[ -n "$GPU_ID"         ] && echo "  GPU_ID : ${GPU_ID}"
[ -n "$RESULT_NAME"    ] && echo "  result_name   : ${RESULT_NAME}"
[ -n "$PROMPT_BUILDER"  ] && echo "  prompt_builder: ${PROMPT_BUILDER}"
[ -n "$EVENT_BULLETIN"  ] && echo "  event_bulletin: ${EVENT_BULLETIN}"
[ -n "$EVENT_FILTER"   ] && echo "  场景过滤: ${EVENT_FILTER}" || echo "  场景类型: emergy_bus + accident_debris"
echo "================================================================"

# ─── 核心评测循环 ──────────────────────────────────────────────────────────────

run_all_baselines() {
    echo ""
    echo "=== [基线] FixedTime + MaxPressure + Random ==="
    for SEED_ID in "${SEED_IDS[@]}"; do
        echo ""
        echo "── Seed: ${SEED_ID} ───────────────────────────────────────────"
        for EVENT in "${EVENT_TYPES[@]}"; do
            [ -n "$EVENT_FILTER" ] && [ "$EVENT" != "$EVENT_FILTER" ] && continue
            echo ""
            echo "── Scene: ${EVENT} ──────────────────────────────────────────────"
            for entry in "${EVENT_ENTRIES[@]}"; do
                IFS='|' read -r DATASET ENV_DIR BASE_STEM <<< "$entry"
                ROUTE_FILE="${BASE_STEM}_${EVENT}.rou.xml"
                FULL_PATH="${ENV_DIR}/${ROUTE_FILE}"
                if [ ! -f "$FULL_PATH" ]; then
                    echo "  [SKIP] 路由文件不存在: ${FULL_PATH}"
                    continue
                fi
                MAX_S=$(get_max_sumo_seconds "$FULL_PATH")
                run_baseline "$DATASET" "$ROUTE_FILE" "$EVENT" "--fixed_time"   "FixedTime"   "$MAX_S" "$SEED_ID"
                run_baseline "$DATASET" "$ROUTE_FILE" "$EVENT" "--max_pressure" "MaxPressure" "$MAX_S" "$SEED_ID"
                run_baseline "$DATASET" "$ROUTE_FILE" "$EVENT" "--random"       "Random"      "$MAX_S" "$SEED_ID"
            done
        done
    done
}

run_all_vlm() {
    echo ""
    echo "=== [VLM] ${MODEL_NAME} ==="
    for SEED_ID in "${SEED_IDS[@]}"; do
        echo ""
        echo "── Seed: ${SEED_ID} ───────────────────────────────────────────"
        for EVENT in "${EVENT_TYPES[@]}"; do
            [ -n "$EVENT_FILTER" ] && [ "$EVENT" != "$EVENT_FILTER" ] && continue
            echo ""
            echo "── Scene: ${EVENT} ──────────────────────────────────────────────"
            for entry in "${EVENT_ENTRIES[@]}"; do
                IFS='|' read -r DATASET ENV_DIR BASE_STEM <<< "$entry"
                ROUTE_FILE="${BASE_STEM}_${EVENT}.rou.xml"
                FULL_PATH="${ENV_DIR}/${ROUTE_FILE}"
                if [ ! -f "$FULL_PATH" ]; then
                    echo "  [SKIP] 路由文件不存在: ${FULL_PATH}"
                    continue
                fi
                MAX_S=$(get_max_sumo_seconds "$FULL_PATH")
                run_vlm "$DATASET" "$ROUTE_FILE" "$EVENT" "$MAX_S" "$SEED_ID"
            done
        done
    done
}

# ─── 执行 ──────────────────────────────────────────────────────────────────────
if [ "$BASELINE_ONLY" = true ]; then
    run_all_baselines
elif [ -n "$MODEL_NAME" ]; then
    [ "$WITH_BASELINE" = true ] && run_all_baselines
    run_all_vlm
else
    echo "请指定运行模式："
    echo "  --baseline-only                                   # 基线（FixedTime + MaxPressure + Random）"
    echo "  --port 8000 --model_name <name>                   # VLM 评测"
    echo "  --port 8000 --model_name <name> --with-baseline   # 先基线后VLM"
    echo "  --event emergy_bus    [--baseline-only|--model_name ...]   # 仅跑紧急车辆+公交"
    echo "  --event accident_debris [...]                              # 仅跑事故+占道"
    exit 1
fi

echo ""
echo "================================================================"
echo "  事件评测完成。"
echo "  结果目录: data/eval/{dataset}/{route_stem}/{method}_seed_{seed}/"
echo "  收集指标: 编辑 src/evaluation/metrics.py 底部的 secnario_dict 和 methods，然后运行："
echo "    python src/evaluation/metrics.py"
echo "================================================================"

echo "Failed tasks: ${FAILED_TASKS}"
[[ "$FAILED_TASKS" -eq 0 ]]
