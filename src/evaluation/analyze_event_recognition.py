
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.training.rlvr.parser.event_recognition_parser import parse_event_recognition
from src.training.rlvr.reward.event_reward import compute_r_event_recog

# 路由文件名后缀 → 场景类型
_EVENT_ROUTE_PATTERNS = ("_emergy_bus", "_accident_debris", "_emergency", "_accident", "_debris")
_NORMAL_ROUTE_PATTERNS = ("_real", "_real_2000", "_real_2500", "_synthetic", "jinan", "hangzhou",
                          "songdo", "massy", "ymt", "newyork")


def _is_event_route(route_stem: str) -> bool:
    lowered = route_stem.lower()
    return any(p in lowered for p in _EVENT_ROUTE_PATTERNS)


def _extract_model_response(response_txt: str) -> str:
    """从 response.txt 中提取模型回答部分。

    新格式包含 [User Prompt] / [Model Response] 分块；旧格式直接是 CoT 内容。
    """
    marker = "[Model Response]"
    idx = response_txt.find(marker)
    if idx != -1:
        text = response_txt[idx + len(marker):].strip()
        # 去掉可能存在的 [Thinking Process] 段，只保留 CoT 正文
        thinking_idx = text.find("[Thinking Process]")
        if thinking_idx != -1:
            text = text[:thinking_idx].strip()
        return text
    return response_txt.strip()


def _has_long_path(response_text: str) -> bool:
    """判断模型是否触发了长路径推理（Event Recognition 段存在且非 None）。"""
    section_start = re.search(r"Event Recognition\s*:\s*", response_text, re.IGNORECASE)
    if section_start is None:
        return False
    tail = response_text[section_start.end():].strip()
    # 截断到下一个段
    end_m = re.search(
        r"\n\s*-?\s*(?:Neighbor(?:ing)?\s*Msgs?|Condition|Selection\s*Logic|Action|"
        r"Broadcast\s*Notice|Impact\s*Analysis|Phase\s*Reasoning)\b",
        tail,
        re.IGNORECASE,
    )
    section_content = tail[: end_m.start()].strip() if end_m else tail
    # None 表示短路径
    return not re.fullmatch(r"`?none`?\.?", section_content, re.IGNORECASE)


def _collect_step_dirs(eval_root: str, scenario_filter: Optional[str],
                       route_filter: Optional[str], model_filter: Optional[str]) -> List[Tuple[str, str, str, str]]:
    """遍历 data/eval 目录，返回 (scenario, route_stem, model, step_dir) 四元组列表。

    只处理包含 response.txt 的叶子目录。
    """
    results = []
    for scenario in sorted(os.listdir(eval_root)):
        if scenario_filter and scenario != scenario_filter:
            continue
        scenario_dir = os.path.join(eval_root, scenario)
        if not os.path.isdir(scenario_dir):
            continue
        for route_stem in sorted(os.listdir(scenario_dir)):
            if route_filter and route_stem != route_filter:
                continue
            route_dir = os.path.join(scenario_dir, route_stem)
            if not os.path.isdir(route_dir):
                continue
            for model in sorted(os.listdir(route_dir)):
                if model_filter and model != model_filter:
                    continue
                model_dir = os.path.join(route_dir, model)
                if not os.path.isdir(model_dir):
                    continue
                # 跳过非 VLM 方法（fixed_time / max_pressure / random 不产生 response.txt）
                if any(model.startswith(p) for p in ("fixed_time", "max_pressure", "random")):
                    continue
                for jid in sorted(os.listdir(model_dir)):
                    jid_dir = os.path.join(model_dir, jid)
                    if not os.path.isdir(jid_dir):
                        continue
                    for step in sorted(os.listdir(jid_dir)):
                        step_dir = os.path.join(jid_dir, step)
                        if not os.path.isdir(step_dir):
                            continue
                        if os.path.exists(os.path.join(step_dir, "response.txt")):
                            results.append((scenario, route_stem, model, step_dir))
    return results


class PathTriggerStats:
    """路径触发统计器。"""

    def __init__(self):
        # 事件场景：应触发长路径
        self.event_total = 0
        self.event_long_path = 0
        # 常规场景：应触发短路径
        self.normal_total = 0
        self.normal_short_path = 0

    def update(self, is_event_route: bool, triggered_long: bool):
        if is_event_route:
            self.event_total += 1
            if triggered_long:
                self.event_long_path += 1
        else:
            self.normal_total += 1
            if not triggered_long:
                self.normal_short_path += 1

    @property
    def long_path_trigger_rate(self) -> Optional[float]:
        return self.event_long_path / self.event_total if self.event_total else None

    @property
    def short_path_trigger_rate(self) -> Optional[float]:
        return self.normal_short_path / self.normal_total if self.normal_total else None


class EventRecogStats:
    """事件识别精确率/召回率统计器。"""

    def __init__(self):
        self.r_scores: List[float] = []
        # per-type 统计：{event_type: {tp, fp, fn}}
        self.per_type: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        self.tp_total = 0
        self.fp_total = 0
        self.fn_total = 0
        self.n_gt_events = 0
        self.n_pred_events = 0

    def update(self, pred_events: Optional[List[Dict]], gt_events: List[Dict]):
        r, _ = compute_r_event_recog(pred_events=pred_events, gt_events=gt_events)
        self.r_scores.append(r)

        pred_list = pred_events or []
        self.n_pred_events += len(pred_list)
        self.n_gt_events += len(gt_events)

        # 简单的 per-type precision/recall（按 event_type 匹配）
        gt_types = [e.get("event_type") for e in gt_events]
        pred_types = [e.get("event_type") for e in pred_list]

        matched_gt = set()
        matched_pred = set()
        for pi, pt in enumerate(pred_types):
            for gi, gt in enumerate(gt_types):
                if gi in matched_gt:
                    continue
                if pt == gt:
                    self.per_type[pt]["tp"] += 1
                    self.tp_total += 1
                    matched_gt.add(gi)
                    matched_pred.add(pi)
                    break

        for gi, gt in enumerate(gt_types):
            if gi not in matched_gt:
                self.per_type[gt]["fn"] += 1
                self.fn_total += 1

        for pi, pt in enumerate(pred_types):
            if pi not in matched_pred:
                self.per_type[pt]["fp"] += 1
                self.fp_total += 1

    @property
    def mean_r(self) -> Optional[float]:
        return sum(self.r_scores) / len(self.r_scores) if self.r_scores else None

    def precision(self, event_type: str) -> Optional[float]:
        d = self.per_type[event_type]
        denom = d["tp"] + d["fp"]
        return d["tp"] / denom if denom else None

    def recall(self, event_type: str) -> Optional[float]:
        d = self.per_type[event_type]
        denom = d["tp"] + d["fn"]
        return d["tp"] / denom if denom else None

    @property
    def overall_precision(self) -> Optional[float]:
        denom = self.tp_total + self.fp_total
        return self.tp_total / denom if denom else None

    @property
    def overall_recall(self) -> Optional[float]:
        denom = self.tp_total + self.fn_total
        return self.tp_total / denom if denom else None


def analyze(eval_root: str, scenario_filter: Optional[str] = None,
            route_filter: Optional[str] = None, model_filter: Optional[str] = None,
            verbose: bool = False) -> None:
    """主分析函数，遍历 eval 目录，计算并打印指标。"""
    step_dirs = _collect_step_dirs(eval_root, scenario_filter, route_filter, model_filter)
    if not step_dirs:
        print(f"[analyze] 未找到任何 response.txt，请确认 eval_root={eval_root} 及过滤条件。")
        return

    # 按 (scenario, route_stem, model) 分组统计
    path_stats: Dict[Tuple[str, str, str], PathTriggerStats] = defaultdict(PathTriggerStats)
    recog_stats: Dict[Tuple[str, str, str], EventRecogStats] = defaultdict(EventRecogStats)
    gt_missing_count = 0
    total_steps = len(step_dirs)

    for scenario, route_stem, model, step_dir in step_dirs:
        key = (scenario, route_stem, model)
        resp_path = os.path.join(step_dir, "response.txt")
        gt_path = os.path.join(step_dir, "gt.json")

        try:
            with open(resp_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
        except Exception:
            continue

        model_resp = _extract_model_response(raw_text)
        triggered_long = _has_long_path(model_resp)
        is_event = _is_event_route(route_stem)
        path_stats[key].update(is_event, triggered_long)

        # 事件识别精确率/召回率（需要 gt.json）
        if not os.path.exists(gt_path):
            gt_missing_count += 1
            continue

        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_payload = json.load(f)
        except Exception:
            continue

        gt_events = gt_payload.get("event_gt", [])
        pred_events = parse_event_recognition(model_resp)
        recog_stats[key].update(pred_events, gt_events)

    # ── 打印结果 ──────────────────────────────────────────────────────────────
    sep = "=" * 70

    print(f"\n{sep}")
    print("§6.4.3  事件识别精确率/召回率与自适应路径触发分析")
    print(f"总决策步数: {total_steps}  |  缺少 gt.json: {gt_missing_count}")
    print(sep)

    all_keys = sorted(set(path_stats) | set(recog_stats))
    for key in all_keys:
        scenario, route_stem, model = key
        ps = path_stats.get(key, PathTriggerStats())
        rs = recog_stats.get(key)
        print(f"\n[{scenario} / {route_stem} / {model}]")

        # 路径触发分析
        print("  ── 路径触发分析 ──")
        ltr = ps.long_path_trigger_rate
        str_ = ps.short_path_trigger_rate
        if ltr is not None:
            print(f"  事件场景  长路径触发率: {ltr*100:.1f}%  ({ps.event_long_path}/{ps.event_total})")
        if str_ is not None:
            print(f"  常规场景  短路径触发率: {str_*100:.1f}%  ({ps.normal_short_path}/{ps.normal_total})")
        if ltr is None and str_ is None:
            print("  （本组无决策步数据）")

        # 事件识别精确率/召回率
        if rs and rs.r_scores:
            print("  ── 事件识别精确率/召回率 ──")
            print(f"  均值 R_event_recog: {rs.mean_r:.4f}  (n={len(rs.r_scores)} 步)")
            print(f"  GT 事件总数: {rs.n_gt_events}  |  Pred 事件总数: {rs.n_pred_events}")
            overall_prec = rs.overall_precision
            overall_rec = rs.overall_recall
            p_str = f"{overall_prec*100:.1f}%" if overall_prec is not None else "N/A"
            r_str = f"{overall_rec*100:.1f}%" if overall_rec is not None else "N/A"
            print(f"  整体精确率 / 召回率: Precision={p_str}  Recall={r_str}")
            if rs.per_type:
                print("  Per-type Precision / Recall:")
                for etype in sorted(rs.per_type):
                    prec = rs.precision(etype)
                    rec = rs.recall(etype)
                    d = rs.per_type[etype]
                    p_str = f"{prec*100:.1f}%" if prec is not None else "N/A"
                    r_str = f"{rec*100:.1f}%" if rec is not None else "N/A"
                    print(f"    {etype:<22}  P={p_str}  R={r_str}  "
                          f"(TP={d['tp']}, FP={d['fp']}, FN={d['fn']})")
        elif gt_missing_count == total_steps:
            print("  ── 事件识别精确率/召回率 ──")
            print("  （所有步骤均缺少 gt.json，请重新运行 run_eval.py 后再分析）")

    print(f"\n{sep}")

    # 汇总（跨所有 key）
    all_ps_list = list(path_stats.values())
    all_rs_list = [rs for rs in recog_stats.values() if rs.r_scores]

    total_event_long = sum(p.event_long_path for p in all_ps_list)
    total_event = sum(p.event_total for p in all_ps_list)
    total_normal_short = sum(p.normal_short_path for p in all_ps_list)
    total_normal = sum(p.normal_total for p in all_ps_list)
    all_r = [r for rs in all_rs_list for r in rs.r_scores]

    print("\n[汇总]")
    if total_event:
        print(f"  全局 事件场景长路径触发率: {total_event_long/total_event*100:.1f}%  "
              f"({total_event_long}/{total_event})")
    if total_normal:
        print(f"  全局 常规场景短路径触发率: {total_normal_short/total_normal*100:.1f}%  "
              f"({total_normal_short}/{total_normal})")
    if all_r:
        print(f"  全局 均值 R_event_recog: {sum(all_r)/len(all_r):.4f}  (n={len(all_r)})")
    total_tp = sum(rs.tp_total for rs in all_rs_list)
    total_fp = sum(rs.fp_total for rs in all_rs_list)
    total_fn = sum(rs.fn_total for rs in all_rs_list)
    if total_tp + total_fp:
        print(f"  全局 整体精确率: {total_tp/(total_tp+total_fp)*100:.1f}%  "
              f"(TP={total_tp}, FP={total_fp})")
    if total_tp + total_fn:
        print(f"  全局 整体召回率: {total_tp/(total_tp+total_fn)*100:.1f}%  "
              f"(TP={total_tp}, FN={total_fn})")
    print(sep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="事件识别精确率/召回率与自适应路径触发分析（§6.4.3）"
    )
    parser.add_argument("--eval_root", "-e", type=str, default="data/eval",
                        help="评测结果根目录（默认 data/eval）")
    parser.add_argument("--scenario",  "-s", type=str, default=None,
                        help="只分析指定场景，如 JiNan")
    parser.add_argument("--route",     "-r", type=str, default=None,
                        help="只分析指定路由 stem，如 anon_3_4_jinan_real_emergy_bus")
    parser.add_argument("--model",     "-m", type=str, default=None,
                        help="只分析指定模型目录名，如 qwen3-vl-8b")
    parser.add_argument("--verbose",   "-v", action="store_true",
                        help="打印每步详细信息")
    args = parser.parse_args()

    analyze(
        eval_root=args.eval_root,
        scenario_filter=args.scenario,
        route_filter=args.route,
        model_filter=args.model,
        verbose=args.verbose,
    )
