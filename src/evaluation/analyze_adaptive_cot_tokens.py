"""
自适应 CoT 长短路径 token 分布统计脚本。

用途：
  本脚本用于论文中 adaptive CoT 的输出开销分析。它会递归读取某个测试场景
  结果目录下的所有 response.txt，提取其中的 [Model Response] 段，根据
  configs/prompt_builder.py 中的自适应推理规则将样本划分为：

    1. 短 CoT：Condition: Normal，对应 prompt 中的 Path 1
    2. 长 CoT：Condition: Special，对应 prompt 中的 Path 2

  随后脚本会统计每条 Model Response 的 token 数，保存逐样本 CSV、汇总 JSON，
  并绘制长/短 CoT token 数分布箱线图。

默认输入目录：
  data/eval/JiNan/anon_3_4_jinan_real_emergy_bus/qwen3-vl-8b

常用命令：
  1. 使用默认目录，采用正则近似 token 计数：
     python src/evaluation/analyze_adaptive_cot_tokens.py

  2. 指定某个评估结果目录：
     python src/evaluation/analyze_adaptive_cot_tokens.py \
       data/eval/JiNan/anon_3_4_jinan_real_emergy_bus/qwen3-vl-8b

  3. 指定模型目录，用模型 tokenizer 统计 token：
     python src/evaluation/analyze_adaptive_cot_tokens.py \
       data/eval/JiNan/anon_3_4_jinan_real_emergy_bus/qwen3-vl-8b \
       --model_path models/base_models/Qwen3-VL-8B-Instruct

  4. 只统计 C. Adaptive Reasoning 段，而不是完整 Model Response：
     python src/evaluation/analyze_adaptive_cot_tokens.py \
       data/eval/JiNan/anon_3_4_jinan_real_emergy_bus/qwen3-vl-8b \
       --text_scope adaptive_reasoning

输出文件：
  默认保存到 results/adaptive_cot_tokens/<模型目录名>/ 下：
    - adaptive_cot_token_records.csv：每条 response.txt 的路径类型和 token 数
    - adaptive_cot_token_summary.json：长/短 CoT 的 count、min、q1、median、mean、q3、max、std
    - adaptive_cot_token_boxplot.png：长/短 CoT token 数箱线图

注意：
  - 如果安装了 transformers 且 --model_path 指向有效本地模型/tokenizer 目录，
    脚本会使用模型 tokenizer 精确计数。
  - 如果未安装 transformers，或指定目录无法加载 tokenizer，脚本会自动退回到
    正则近似计数，并在终端打印警告。
  - tokenizer 加载使用 local_files_only=True，不会尝试从网络下载模型文件。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional


DEFAULT_EVAL_DIR = (
    "data/eval/JiNan/anon_3_4_jinan_real_emergy_bus/qwen3-vl-8b"
)


def extract_model_response(response_txt: str) -> str:
    """从 response.txt 中提取真正由模型生成的 [Model Response] 段。

    run_eval.py 保存的 response.txt 通常同时包含 [User Prompt] 和
    [Model Response]。论文统计的是模型输出开销，因此这里必须排除 prompt，
    避免把输入模板中的 CoT 指令也算入 token 数。
    """
    match = re.search(
        r"\[Model Response\]\s*(.*?)(?=\n\s*\[Thinking Process\]|\Z)",
        response_txt,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return response_txt.strip()


def extract_adaptive_reasoning(model_response: str) -> str:
    """按需只提取 C. Adaptive Reasoning 段。

    默认统计完整 Model Response，更符合端到端输出成本；如果只关心自适应推理
    段本身，可通过 --text_scope adaptive_reasoning 调用本函数。
    """
    match = re.search(
        r"C\.\s*Adaptive Reasoning\s*:\s*(.*?)(?=\n\s*\]\s*\n\s*Action\s*:|\n\s*Action\s*:|\Z)",
        model_response,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return model_response.strip()


def classify_adaptive_path(model_response: str) -> str:
    """将模型输出分类为 long / short / unknown。

    configs/prompt_builder.py 对 adaptive CoT 的定义为：
      - Condition == Normal  -> Path 1 -> 短 CoT
      - Condition == Special -> Path 2 -> 长 CoT

    优先解析模型输出中的 Condition 字段。考虑到 VLM 输出格式可能偶尔不严格，
    如果 Condition 缺失，则用长路径特有字段 Impact Analysis / Broadcast Notice
    作为兜底；如果只看到 Phase Reasoning / Dur Reasoning，则归为短路径。
    """
    condition_match = re.search(
        r"(?:^|\n)\s*-?\s*Condition\s*:\s*`?(Normal|Special)`?",
        model_response,
        flags=re.IGNORECASE,
    )
    if condition_match:
        condition = condition_match.group(1).lower()
        return "long" if condition == "special" else "short"

    if re.search(
        r"(?:^|\n)\s*(?:Impact Analysis|Broadcast Notice)\s*:",
        model_response,
        flags=re.IGNORECASE,
    ):
        return "long"

    if re.search(
        r"(?:^|\n)\s*(?:Phase Reasoning|Dur Reasoning)\s*:",
        model_response,
        flags=re.IGNORECASE,
    ):
        return "short"

    return "unknown"


def regex_token_count(text: str) -> int:
    """无法加载模型 tokenizer 时使用的近似 token 计数器。

    这个规则会把英文单词、数字、单个中文字符和标点分别视作 token。它不是
    Qwen tokenizer 的精确结果，但在没有 transformers 或模型文件时可以提供
    稳定、可复现的相对分布。
    """
    return len(
        re.findall(
            r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[\u4e00-\u9fff]|[^\s]",
            text,
        )
    )


def build_token_counter(model_path: Optional[str]) -> tuple[Callable[[str], int], str]:
    """根据模型目录构建 token 计数函数。

    参数 model_path 可以指向完整模型目录，也可以指向仅包含 tokenizer 文件的目录。
    加载顺序：
      1. AutoTokenizer.from_pretrained(model_path)
      2. AutoProcessor.from_pretrained(model_path).tokenizer
      3. regex_token_count 兜底

    这样既支持普通 LLM tokenizer，也兼容部分 VLM 只通过 processor 暴露
    tokenizer 的情况。
    """
    if not model_path:
        return regex_token_count, "regex"

    try:
        from transformers import AutoProcessor, AutoTokenizer
    except Exception as exc:
        print(
            f"[WARN] 无法导入 transformers：{exc}。"
            "将退回到 regex 近似 token 计数。"
        )
        return regex_token_count, "regex"

    tokenizer = None
    tokenizer_source = ""

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        tokenizer_source = "AutoTokenizer"
    except Exception as tokenizer_exc:
        try:
            processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
            tokenizer = getattr(processor, "tokenizer", None)
            tokenizer_source = "AutoProcessor.tokenizer"
        except Exception as processor_exc:
            print(
                f"[WARN] 无法从 {model_path} 加载 tokenizer。\n"
                f"       AutoTokenizer 错误：{tokenizer_exc}\n"
                f"       AutoProcessor 错误：{processor_exc}\n"
                "       将退回到 regex 近似 token 计数。"
            )
            return regex_token_count, "regex"

    if tokenizer is None:
        print(
            f"[WARN] {model_path} 已加载 processor，但其中没有 tokenizer。"
            "将退回到 regex 近似 token 计数。"
        )
        return regex_token_count, "regex"

    def count_with_tokenizer(text: str) -> int:
        # 不加入 BOS/EOS 等特殊符号，只统计模型实际生成文本本身的 token 数。
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count_with_tokenizer, f"transformers:{tokenizer_source}:{model_path}"


def iter_response_files(eval_dir: Path) -> Iterable[Path]:
    """递归查找目标目录下所有 response.txt。"""
    return sorted(eval_dir.rglob("response.txt"))


def infer_metadata(response_path: Path) -> Dict[str, str]:
    """从 data/eval 风格路径中推断场景、route、模型、路口和 step 信息。

    典型路径：
      data/eval/<scenario>/<route>/<model>/<intersection>/<step>/response.txt
    """
    parts = response_path.parts
    metadata = {
        "scenario": "",
        "route": "",
        "model": "",
        "intersection": response_path.parent.parent.name,
        "step": response_path.parent.name,
    }

    try:
        idx = parts.index("eval")
        metadata["scenario"] = parts[idx + 1]
        metadata["route"] = parts[idx + 2]
        metadata["model"] = parts[idx + 3]
    except (ValueError, IndexError):
        pass

    return metadata


def percentile(values: List[int], q: float) -> Optional[float]:
    """计算分位数，q 取值范围为 [0, 1]。"""
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return float(ordered[lower])
    weight = pos - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def summarize(values: List[int]) -> Dict[str, Optional[float]]:
    """汇总一组 token 数，返回箱线图和论文表格常用统计量。"""
    if not values:
        return {
            "count": 0,
            "min": None,
            "q1": None,
            "median": None,
            "mean": None,
            "q3": None,
            "max": None,
            "std": None,
        }

    return {
        "count": len(values),
        "min": min(values),
        "q1": percentile(values, 0.25),
        "median": percentile(values, 0.50),
        "mean": float(sum(values) / len(values)),
        "q3": percentile(values, 0.75),
        "max": max(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def write_records_csv(path: Path, records: List[Dict[str, object]]) -> None:
    """保存逐 response.txt 的明细统计，便于定位离群样本。"""
    fieldnames = [
        "path_type",
        "token_count",
        "text_scope",
        "response_path",
        "scenario",
        "route",
        "model",
        "intersection",
        "step",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_boxplot(path: Path, grouped_tokens: Dict[str, List[int]], title: str) -> None:
    """绘制长/短 CoT token 数箱线图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = []
    data = []
    for path_type in ("short", "long"):
        values = grouped_tokens.get(path_type, [])
        if values:
            labels.append(f"{path_type} CoT\nn={len(values)}")
            data.append(values)

    if not data:
        raise ValueError("No short/long token counts available for boxplot.")

    fig_width = max(5.5, 1.8 * len(data) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, 4.5), dpi=180)
    ax.boxplot(
        data,
        labels=labels,
        showmeans=True,
        patch_artist=True,
        boxprops={"facecolor": "#d9e8f5", "edgecolor": "#2d526b"},
        medianprops={"color": "#b3362d", "linewidth": 1.6},
        meanprops={
            "marker": "o",
            "markerfacecolor": "#1f7a4d",
            "markeredgecolor": "#1f7a4d",
            "markersize": 4,
        },
    )
    ax.set_title(title)
    ax.set_ylabel("Output tokens")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "统计 response.txt 中 adaptive CoT 长/短路径的 token 数分布，并绘制箱线图。"
        )
    )
    parser.add_argument(
        "eval_dir",
        nargs="?",
        default=DEFAULT_EVAL_DIR,
        help=f"递归扫描的评估结果目录。默认：{DEFAULT_EVAL_DIR}",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="CSV/JSON/PNG 输出目录。默认：results/adaptive_cot_tokens/<eval_dir_name>",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help=(
            "本地模型或 tokenizer 目录。指定后脚本会优先使用该目录中的 tokenizer "
            "统计 token 数；加载失败则退回 regex。"
        ),
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "兼容旧参数：本地 HuggingFace tokenizer 路径。建议改用 --model_path。"
        ),
    )
    parser.add_argument(
        "--text_scope",
        choices=("model_response", "adaptive_reasoning"),
        default="model_response",
        help=(
            "token 统计范围。model_response 表示完整 [Model Response]；"
            "adaptive_reasoning 表示只统计 C. Adaptive Reasoning 段。默认：model_response。"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation directory does not exist: {eval_dir}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("results/adaptive_cot_tokens") / eval_dir.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # --model_path 是新的推荐参数；--tokenizer 保留给已有命令兼容。
    # 两者同时指定时，以 --model_path 为准，避免同一脚本运行时 tokenizer 来源不明确。
    tokenizer_path = args.model_path or args.tokenizer
    count_tokens, tokenizer_name = build_token_counter(tokenizer_path)
    records: List[Dict[str, object]] = []
    grouped_tokens: Dict[str, List[int]] = {"short": [], "long": [], "unknown": []}

    for response_path in iter_response_files(eval_dir):
        # response.txt 可能包含 prompt 和模型输出，必须先提取 [Model Response]。
        raw_text = response_path.read_text(encoding="utf-8", errors="replace")
        model_response = extract_model_response(raw_text)

        # 默认统计完整模型输出；如只分析自适应推理段，可用 --text_scope 切换。
        text_for_count = (
            extract_adaptive_reasoning(model_response)
            if args.text_scope == "adaptive_reasoning"
            else model_response
        )

        # 根据 prompt_builder.py 中 Condition 的 Normal/Special 规则划分短/长 CoT。
        path_type = classify_adaptive_path(model_response)
        token_count = count_tokens(text_for_count)
        grouped_tokens.setdefault(path_type, []).append(token_count)

        metadata = infer_metadata(response_path)
        records.append(
            {
                "path_type": path_type,
                "token_count": token_count,
                "text_scope": args.text_scope,
                "response_path": str(response_path),
                **metadata,
            }
        )

    if not records:
        raise RuntimeError(f"No response.txt files found under {eval_dir}")

    csv_path = output_dir / "adaptive_cot_token_records.csv"
    summary_path = output_dir / "adaptive_cot_token_summary.json"
    boxplot_path = output_dir / "adaptive_cot_token_boxplot.png"

    write_records_csv(csv_path, records)

    summary = {
        "eval_dir": str(eval_dir),
        "text_scope": args.text_scope,
        "tokenizer": tokenizer_name,
        "total_files": len(records),
        "stats": {
            "short": summarize(grouped_tokens.get("short", [])),
            "long": summarize(grouped_tokens.get("long", [])),
            "unknown": summarize(grouped_tokens.get("unknown", [])),
        },
        "outputs": {
            "records_csv": str(csv_path),
            "summary_json": str(summary_path),
            "boxplot_png": str(boxplot_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    write_boxplot(
        boxplot_path,
        grouped_tokens,
        title=f"Adaptive CoT Token Distribution ({args.text_scope})",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
