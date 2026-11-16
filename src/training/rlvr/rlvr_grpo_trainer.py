"""Qwen3-VL 的 RLVR / GRPO 训练入口。

这个模块把三个层次串起来：
1. 数据：读取 `rlvr_data_collector.py` 产出的 RLVR JSONL；
2. 奖励：通过 `verifier.compute_rlvr_reward` 和 `RolloutEvaluator` 计算单样本 reward；
3. 训练：用 TRL 的 GRPOTrainer 完成在线策略更新。

实现上保留较强的容错性：
- 若当前环境中的 TRL 版本不包含 GRPOTrainer，会给出明确报错；(使用TRL 0.29.1 )
- reward wrapper 同时兼容多种可能的回调入参形态；
- 数据集中的图片路径会在 map 阶段被加载成 PIL 图像，和仓库内现有的
  视觉偏好训练脚本保持一致。
"""

from __future__ import annotations

import argparse
import inspect
import math
import os
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional

import PIL.Image
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from src.training.rlvr.rollout_evaluator import RolloutEvaluator
from src.training.rlvr.verifier import compute_rlvr_reward
from src.training.trainer import freeze_multimodal_components
from src.training.rlvr.grpo_debug_metrics import (
    compute_reward_debug_metrics,
    compute_loss_debug_metrics,
    _maybe_dump_grpo_debug_samples,
)

try:
    from trl import GRPOConfig, GRPOTrainer
except Exception:  # pragma: no cover - 依赖版本未准备好时给出清晰错误
    GRPOConfig = None
    GRPOTrainer = None


def _load_image(image_path: str) -> PIL.Image.Image:
    """把磁盘图片加载为 RGB PIL 对象。"""
    return PIL.Image.open(image_path).convert("RGB")


def _normalize_images_in_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """将图片路径加载为 PIL 图像，并直接嵌入 prompt content，删除独立的 images 字段。

    TRL 0.29.1 从 inputs["images"] 取图像后，通过 prepare_multimodal_messages 注入 prompt content，
    但 DataLoader collation 会破坏 PIL Image 类型（变成 dict）。
    解决方案：直接把 PIL Image 嵌入 prompt content（structured multimodal format），
    删除 images 字段，TRL 检测到 images=None 后跳过 prepare_multimodal_messages，
    apply_chat_template 直接从 content 里提取 PIL Image 完成 tokenization。
    """
    image_paths = sample.get("images") or sample.get("image_paths") or []
    images: List[PIL.Image.Image] = []
    for image_path in image_paths:
        if isinstance(image_path, str) and os.path.exists(image_path):
            images.append(_load_image(image_path))

    # 提取 prompt 文本（兼容字符串和旧有 messages list 两种格式）
    prompt = sample.get("prompt")
    if isinstance(prompt, list):
        # 已是 messages list，提取 user content 文本
        text = ""
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                text = content if isinstance(content, str) else text
                break
    else:
        text = str(prompt) if prompt else ""

    # 构建 structured multimodal content，将图像直接嵌入 prompt
    # 格式：[{"type": "image", "image": PIL}, ..., {"type": "text", "text": "..."}]
    content: List[Dict[str, Any]] = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": text})

    sample = dict(sample)
    # 用 structured content 替换 prompt，删除独立 images 字段以绕过 TRL 的 collation 路径
    sample["prompt"] = [{"role": "user", "content": content}]
    sample.pop("images", None)
    sample.pop("image_paths", None)

    return sample


def _safe_extract_response_text(completion: Any) -> str:
    """从 GRPO 生成结果里尽可能抽出 response text。"""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        for key in ("content", "text", "response", "completion"):
            value = completion.get(key)
            if isinstance(value, str):
                return value
    if isinstance(completion, list):
        texts: List[str] = []
        for item in completion:
            if isinstance(item, dict) and isinstance(item.get("content"), str):
                texts.append(item["content"])
            elif isinstance(item, str):
                texts.append(item)
        if texts:
            return "\n".join(texts)
    return str(completion)


def _extract_samples_from_kwargs(kwargs: Dict[str, Any], batch_size: int) -> List[Dict[str, Any]]:
    """兼容 TRL 不同版本对 reward callback 传参的差异。"""
    for key in ("samples", "sample", "inputs", "prompts"):
        value = kwargs.get(key)
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            return value
        if isinstance(value, dict) and value:
            return [value] * batch_size

    column_values = {
        key: value
        for key, value in kwargs.items()
        if isinstance(value, list) and len(value) == batch_size
    }
    if column_values:
        return [
            {key: values[index] for key, values in column_values.items()}
            for index in range(batch_size)
        ]

    return [{} for _ in range(batch_size)]


def _safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _event_bucket_names(sample: Dict[str, Any]) -> List[str]:
    event_gt = sample.get("event_gt") or []
    if not event_gt:
        return ["None"]
    names = sorted(
        {
            str(event.get("event_type") or "Unknown").replace(" ", "_")
            for event in event_gt
            if isinstance(event, dict)
        }
    )
    return names or ["Unknown"]


def _entropy_from_counter(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        if count <= 0:
            continue
        prob = count / total
        entropy -= prob * math.log(prob)
    return float(entropy)


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return float(math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)))


class RLVRMetricBuffer:
    """Accumulates reward-callback diagnostics until Trainer.log flushes them."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.reward_values: List[float] = []
        self.fallback_count = 0
        self.margin_values: List[float] = []
        self.low_margin_count = 0
        self.phase_counts: Counter = Counter()
        self.duration_counts: Counter = Counter()
        self.component_values: Dict[str, List[float]] = defaultdict(list)
        self.sampling_meta_values: Dict[str, List[float]] = defaultdict(list)
        self.scenario_counts: Counter = Counter()
        self.scenario_reward_values: Dict[str, List[float]] = defaultdict(list)
        self.scenario_traffic_values: Dict[str, List[float]] = defaultdict(list)
        self.event_counts: Counter = Counter()
        self.event_reward_values: Dict[str, List[float]] = defaultdict(list)
        self.event_traffic_values: Dict[str, List[float]] = defaultdict(list)
        self.event_neg_traffic_counts: Counter = Counter()

    def add(self, sample: Dict[str, Any], reward_dict: Dict[str, Any]) -> None:
        reward = _safe_float(reward_dict.get("reward"))
        if reward is None:
            return

        self.count += 1
        self.reward_values.append(reward)
        # fallback 判断：动作解析失败（parsed_action 为 None），不依赖 r_traffic 是否存在
        # （常规场景的 r_traffic 可能不写入 dict，不应视为 fallback）
        if reward_dict.get("parsed_action") is None:
            self.fallback_count += 1

        rts = sample.get("reward_table_stats") or {}
        margin = _safe_float(rts.get("reward_margin"))
        if margin is not None:
            self.margin_values.append(margin)
            if margin < 0.02:
                self.low_margin_count += 1

        parsed_action = reward_dict.get("parsed_action") or {}
        phase_id = parsed_action.get("phase_id") if isinstance(parsed_action, dict) else None
        if phase_id is not None:
            try:
                self.phase_counts[int(phase_id)] += 1
            except (TypeError, ValueError):
                pass
        duration = parsed_action.get("duration") if isinstance(parsed_action, dict) else None
        if duration is not None:
            try:
                self.duration_counts[int(duration)] += 1
            except (TypeError, ValueError):
                pass

        for key in (
            "r_base",
            "r_overlay",
            "r_traffic",
            "r_lane",
            "r_event_recog",
            "r_vision",
            "r_mix",
            "emergency_wait_norm",
            "transit_wait_norm",
        ):
            value = _safe_float(reward_dict.get(key))
            if value is not None:
                self.component_values[key].append(value)

        sampling_meta = sample.get("sampling_meta") or {}
        if isinstance(sampling_meta, dict):
            for key, raw_value in sampling_meta.items():
                value = _safe_float(raw_value)
                if value is not None:
                    self.sampling_meta_values[str(key)].append(value)

        r_traffic = _safe_float(reward_dict.get("r_traffic"))
        scenario = str(sample.get("scenario") or "Unknown").replace(" ", "_")
        self.scenario_counts[scenario] += 1
        self.scenario_reward_values[scenario].append(reward)
        if r_traffic is not None:
            self.scenario_traffic_values[scenario].append(r_traffic)

        for event_type in _event_bucket_names(sample):
            self.event_counts[event_type] += 1
            self.event_reward_values[event_type].append(reward)
            if r_traffic is not None:
                self.event_traffic_values[event_type].append(r_traffic)
                if r_traffic < 0:
                    self.event_neg_traffic_counts[event_type] += 1

    def flush(self) -> Dict[str, float]:
        if self.count <= 0:
            return {}

        metrics: Dict[str, float] = {
            "rlvr/reward_mean": _mean(self.reward_values),
            "rlvr/reward_std": _std(self.reward_values),
            "rlvr/reward_min": min(self.reward_values),
            "rlvr/reward_max": max(self.reward_values),
            "rlvr/fallback_rate": self.fallback_count / self.count,
            "rlvr/reward_margin_count": float(len(self.margin_values)),
            "rlvr/duration_entropy": _entropy_from_counter(self.duration_counts),
            "rlvr/phase_entropy": _entropy_from_counter(self.phase_counts),
        }

        if self.margin_values:
            metrics["rlvr/reward_margin_mean"] = _mean(self.margin_values)
            metrics["rlvr/reward_margin_std"] = _std(self.margin_values)
            metrics["rlvr/reward_margin_min"] = min(self.margin_values)
            metrics["rlvr/reward_margin_max"] = max(self.margin_values)
            metrics["rlvr/low_reward_margin_rate"] = self.low_margin_count / len(self.margin_values)

        phase_total = sum(self.phase_counts.values())
        if phase_total:
            for phase_id in sorted(self.phase_counts):
                metrics[f"rlvr/phase/{phase_id}_rate"] = self.phase_counts[phase_id] / phase_total

        duration_total = sum(self.duration_counts.values())
        if duration_total:
            for duration in (15, 20, 25, 30, 35, 40):
                metrics[f"rlvr/duration/{duration}_rate"] = self.duration_counts[duration] / duration_total

        for key, values in self.component_values.items():
            if values:
                metrics[f"rlvr/{key}_mean"] = _mean(values)
                metrics[f"rlvr/{key}_std"] = _std(values)
                metrics[f"rlvr/{key}_min"] = min(values)
                metrics[f"rlvr/{key}_max"] = max(values)

        for key, values in self.sampling_meta_values.items():
            if values:
                metrics[f"rlvr/sampling_meta/{key}_mean"] = _mean(values)

        for scenario, count in self.scenario_counts.items():
            if count <= 0:
                continue
            prefix = f"rlvr/scenario/{scenario}"
            metrics[f"{prefix}/count"] = float(count)
            metrics[f"{prefix}/reward_mean"] = _mean(self.scenario_reward_values[scenario])
            traffic_values = self.scenario_traffic_values.get(scenario) or []
            if traffic_values:
                metrics[f"{prefix}/r_traffic_mean"] = _mean(traffic_values)
                metrics[f"{prefix}/neg_r_traffic_rate"] = sum(1 for value in traffic_values if value < 0) / len(traffic_values)

        for event_type, count in self.event_counts.items():
            if count <= 0:
                continue
            prefix = f"rlvr/event_type/{event_type}"
            metrics[f"{prefix}/count"] = float(count)
            metrics[f"{prefix}/reward_mean"] = _mean(self.event_reward_values[event_type])
            traffic_values = self.event_traffic_values.get(event_type) or []
            if traffic_values:
                metrics[f"{prefix}/r_traffic_mean"] = _mean(traffic_values)
                metrics[f"{prefix}/neg_r_traffic_rate"] = self.event_neg_traffic_counts[event_type] / len(traffic_values)

        self.reset()
        return metrics


def wrap_reward_for_grpo(
    verifier_fn: Callable[..., Dict[str, Any]],
    metric_buffer: Optional[RLVRMetricBuffer] = None,
) -> Callable[..., List[float]]:
    """把 verifier 适配成 GRPOTrainer 可用的 reward 函数。"""

    def _reward_fn(*args, **kwargs) -> List[float]:
        completions = None
        if args:
            completions = args[0]
        completions = kwargs.get("completions", completions)
        if completions is None:
            return []

        if not isinstance(completions, list):
            completions = [completions]

        samples = _extract_samples_from_kwargs(kwargs, len(completions))
        rewards: List[float] = []

        for index, completion in enumerate(completions):
            sample = samples[index] if index < len(samples) else {}
            response_text = _safe_extract_response_text(completion)
            evaluator = RolloutEvaluator(sample)
            reward_dict = verifier_fn(response_text, sample, evaluator)
            if metric_buffer is not None:
                metric_buffer.add(sample, reward_dict)
            rewards.append(float(reward_dict.get("reward", 0.0)))

        return rewards

    _reward_fn.__name__ = "rlvr_reward"
    return _reward_fn


def _filter_kwargs_for_callable(callable_obj: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """只保留目标 callable 接受的参数，降低不同 TRL 版本的兼容风险。"""
    try:
        signature = inspect.signature(callable_obj)
    except Exception:
        return kwargs

    accepted = set(signature.parameters.keys())
    return {key: value for key, value in kwargs.items() if key in accepted}


def _build_grpo_config(**kwargs):
    """根据当前 TRL 版本的 GRPOConfig 签名，筛掉未知参数。"""
    if GRPOConfig is None:
        raise RuntimeError("当前环境未找到 trl.GRPOConfig / trl.GRPOTrainer，请先安装 trl==0.29.1")
    filtered_kwargs = _filter_kwargs_for_callable(GRPOConfig, kwargs)
    return GRPOConfig(**filtered_kwargs)


class Qwen3VLGRPOTrainer(GRPOTrainer if GRPOTrainer is not None else object):
    """Qwen3-VL 的 GRPOTrainer 包装器。

    新增 debug metrics 配置（通过 trainer_kwargs 或直接属性设置）：
      log_grpo_debug_metrics:      bool  = True   — 是否开启 debug 指标
      debug_log_samples:           bool  = False  — 是否 dump 样本到 JSONL
      debug_log_samples_steps:     int   = 20     — 每隔多少步 dump 一次
      debug_num_samples:           int   = 8      — 每次 dump 多少个 prompt
      debug_reward_low_std_threshold: float = 1e-6
      debug_output_dir:            str   = "./debug_grpo"
    """

    def __init__(
        self,
        *args,
        reward_fn: Callable[..., Dict[str, Any]],
        log_grpo_debug_metrics: bool = True,
        debug_log_samples: bool = False,
        debug_log_samples_steps: int = 20,
        debug_num_samples: int = 8,
        debug_reward_low_std_threshold: float = 1e-6,
        debug_output_dir: str = "./debug_grpo",
        **kwargs,
    ):
        if GRPOTrainer is None:
            raise RuntimeError("当前环境未找到 trl.GRPOTrainer，请先安装 trl==0.29.1")

        # debug 配置
        self._log_grpo_debug_metrics = log_grpo_debug_metrics
        self._debug_log_samples = debug_log_samples
        self._debug_log_samples_steps = debug_log_samples_steps
        self._debug_num_samples = debug_num_samples
        self._debug_reward_low_std_threshold = debug_reward_low_std_threshold
        self._debug_output_dir = debug_output_dir

        # 用于在 log 时合并 reward debug 指标的 buffer
        self._pending_debug_metrics: Dict[str, float] = {}
        # 保存本次 generate 阶段的中间数据供 dump 使用
        self._last_gen_snapshot: Dict[str, Any] = {}

        # _calculate_rewards override 存储最近一次 gather 后的全进程 rewards_per_func tensor。
        # TRL 的 _prepare_inputs 只在 step % steps_per_generation == 0 时真正调用
        # _generate_and_score_completions（其余 step 走 _buffered_inputs 缓存）。
        # 用 _last_rewards_per_func 保存真正计算时的结果，供 _generate_and_score_completions
        # override 读取，避免依赖 deque/cache 的截断问题。
        self._last_rewards_per_func: Optional[torch.Tensor] = None  # (B*G_global, num_funcs)

        self._rlvr_metric_buffer = RLVRMetricBuffer()
        self._reward_fn = wrap_reward_for_grpo(
            reward_fn,
            metric_buffer=self._rlvr_metric_buffer,
        )
        reward_funcs = list(kwargs.pop("reward_funcs", []))
        reward_funcs.insert(0, self._reward_fn)
        kwargs["reward_funcs"] = reward_funcs
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # 钩子一：_generate_and_score_completions 之后
    #   TRL 在这个方法里计算 rewards / advantages 并存入 inputs dict，
    #   我们通过 override 在其返回后注入 reward-level debug metrics。
    # ------------------------------------------------------------------
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        """调用父类实现，把 gather 后的全进程 rewards_per_func 存到 self._last_rewards_per_func。

        TRL 内部 _prepare_inputs 只在 step % steps_per_generation == 0 时调用
        _generate_and_score_completions（进而调用本方法）；其余 step 走 _buffered_inputs。
        因此只需在这里保存一次，供 _generate_and_score_completions override 读取即可。
        """
        rewards_per_func = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        if self._log_grpo_debug_metrics:
            try:
                self._last_rewards_per_func = rewards_per_func.detach().cpu()
            except Exception:
                pass
        return rewards_per_func

    def _generate_and_score_completions(self, inputs):
        """调用父类实现，然后附加 reward-level debug 指标。

        rewards_per_func 来自 _calculate_rewards override 存储的 _last_rewards_per_func，
        这是 gather 后的全进程数据（shape: B*G_global × num_funcs），可直接做 group 统计。
        """
        result = super()._generate_and_score_completions(inputs)

        if not self._log_grpo_debug_metrics:
            return result

        try:
            completion_mask = result.get("completion_mask")  # (B_local, T)
            num_gen = self.num_generations
            rewards_per_func = self._last_rewards_per_func  # (B*G_global, num_funcs) or None

            if rewards_per_func is not None and rewards_per_func.numel() > 0:
                # 聚合成总 reward（nansum over funcs，复现 TRL 的做法）
                rewards_now = rewards_per_func.float().nansum(dim=1)  # (B*G_global,)
                reward_func_names = getattr(self, "reward_func_names", None)

                debug_m = compute_reward_debug_metrics(
                    rewards=rewards_now,
                    rewards_per_func=rewards_per_func.float(),
                    reward_func_names=reward_func_names,
                    num_generations=num_gen,
                    low_std_threshold=self._debug_reward_low_std_threshold,
                )
                self._pending_debug_metrics.update(debug_m)

            # sample dump
            if self._debug_log_samples and self.accelerator.is_main_process and rewards_per_func is not None:
                rewards_now_dump = rewards_per_func.float().nansum(dim=1)
                n_dump = rewards_now_dump.numel()
                # TRL 0.29.1 将文本日志存储在 _textual_logs（defaultdict(list)），而非 _logs
                _textual = getattr(self, "_textual_logs", None) or {}
                prompts_text = list(_textual.get("prompt", []))[-n_dump:]
                completions_text = list(_textual.get("completion", []))[-n_dump:]
                adv_all_list = list(_textual.get("advantages", []))[-n_dump:]
                adv_all_tensor = (
                    torch.tensor(adv_all_list, dtype=torch.float32)
                    if adv_all_list else torch.tensor([])
                )

                try:
                    import wandb
                    wandb_run = wandb.run
                except Exception:
                    wandb_run = None

                _maybe_dump_grpo_debug_samples(
                    global_step=self.state.global_step,
                    prompts_text=prompts_text,
                    completions_text=completions_text,
                    rewards=rewards_now_dump,
                    advantages_all=adv_all_tensor,
                    completion_mask=completion_mask,
                    num_generations=num_gen,
                    reward_func_names=getattr(self, "reward_func_names", None),
                    rewards_per_func=rewards_per_func.float(),
                    dump_steps=self._debug_log_samples_steps,
                    num_samples=self._debug_num_samples,
                    output_dir=self._debug_output_dir,
                    is_main_process=True,
                    wandb_run=wandb_run,
                )
        except Exception:
            pass  # debug 逻辑绝不影响训练主流程

        return result

    # ------------------------------------------------------------------
    # 钩子二：_compute_loss 之后
    #   通过 override compute_loss 获取 logps / mask / advantage。
    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """调用父类 loss，然后附加 advantage / ratio / length debug 指标。"""
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs,
                                    num_items_in_batch=num_items_in_batch)

        if not self._log_grpo_debug_metrics:
            return loss

        try:
            with torch.no_grad():
                advantages = inputs.get("advantages")
                completion_mask = inputs.get("completion_mask")
                old_logps = inputs.get("old_per_token_logps")

                if advantages is not None and completion_mask is not None:
                    adv_metrics = compute_loss_debug_metrics(
                        advantages=advantages,
                        per_token_logps=old_logps if old_logps is not None else torch.zeros_like(completion_mask.float()),
                        old_per_token_logps=old_logps if old_logps is not None else torch.zeros_like(completion_mask.float()),
                        completion_mask=completion_mask,
                        epsilon_low=getattr(self, "epsilon_low", 0.2),
                        epsilon_high=getattr(self, "epsilon_high", 0.2),
                    )
                    # num_iterations=1 且不使用 vLLM IS 时，old_logps == per_token_logps，
                    # ratio ≈ 1，debug/policy_ratio_* 数值接近常数，仅供参考；
                    # 使用 vLLM importance sampling 时才有完整诊断价值。
                    self._pending_debug_metrics.update(adv_metrics)
        except Exception:
            pass

        return loss

    # ------------------------------------------------------------------
    # 钩子三：log() — 把所有 pending debug metrics 一起推给 Trainer.log
    # ------------------------------------------------------------------
    def log(self, logs: Dict[str, float], *args, **kwargs):  # type: ignore[override]
        """Append RLVR reward diagnostics and debug metrics to Trainer logs."""
        if isinstance(logs, dict):
            logs = dict(logs)
            logs.update(self._rlvr_metric_buffer.flush())
            if self._log_grpo_debug_metrics and self._pending_debug_metrics:
                logs.update(self._pending_debug_metrics)
                self._pending_debug_metrics = {}
        return super().log(logs, *args, **kwargs)


    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """LoRA 模式只保存 adapter；全参数模式走父类标准保存路径。"""
        output_dir = output_dir or self.args.output_dir
        unwrapped = self.accelerator.unwrap_model(self.model)

        if isinstance(unwrapped, PeftModel):
            # LoRA：gather 所有进程，只在 main process 写盘
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                unwrapped.save_pretrained(output_dir, safe_serialization=True)
                if self.processing_class is not None:
                    self.processing_class.save_pretrained(output_dir)
            self.accelerator.wait_for_everyone()
        else:
            # 全参数：直接委托给父类，父类内部已包含必要的 barrier 与 ZeRO gather
            super().save_model(output_dir, _internal_call)

    # def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
    #     """LoRA 模式下：将 adapter 合并到 base model 后保存完整模型权重。"""
    #     output_dir = output_dir or self.args.output_dir
    #     if isinstance(self.model, PeftModel):
    #         # unwrap Accelerate / DeepSpeed 包裹，获取原始 PeftModel
    #         unwrapped = self.accelerator.unwrap_model(self.model)
    #         merged = unwrapped.merge_and_unload()
    #         if self.accelerator.is_main_process:
    #             merged.save_pretrained(output_dir)
    #             if self.processing_class is not None:
    #                 self.processing_class.save_pretrained(output_dir)
    #     else:
    #         super().save_model(output_dir, _internal_call)

    # _prepare_inputs 不做重写：TRL 0.29.1 的 _prepare_inputs 会自行调用 _generate_and_score_completions，
    # 后者内部通过 prepare_multimodal_messages 将 images + conversational prompt 合并后再 tokenize。
    # 额外调用 processor(images=...) 会与 TRL 内部图像处理流程冲突，故移除该重写。


def load_rlvr_dataset(dataset_path: str):
    """加载 RLVR JSONL，并注册运行时图像加载变换。

    使用 set_transform 而非 map：
    - map 会将 PIL Image 序列化为 {"bytes": ..., "path": ...} dict（HF datasets 内部存储格式），
      导致 TRL 在 _generate_and_score_completions 里收到 dict 而非 PIL Image。
    - set_transform 在数据被 DataLoader 读取时才执行，完全绕过 datasets 的序列化存储，
      GRPOTrainer 的 identity collator 保证 PIL Image 以原始 Python 对象形式传递给 Trainer。
    """
    dataset = load_dataset("json", data_files={"train": dataset_path}, split="train")

    def _batch_transform(batch: Dict[str, Any]) -> Dict[str, Any]:
        keys = list(batch.keys())
        n = len(batch[keys[0]]) if keys else 0
        transformed_samples = [
            _normalize_images_in_sample({k: batch[k][i] for k in keys})
            for i in range(n)
        ]
        out_keys = set(transformed_samples[0].keys()) if transformed_samples else set()
        return {k: [s[k] for s in transformed_samples] for k in out_keys}

    dataset.set_transform(_batch_transform)
    return dataset


def build_trainer_from_args(args) -> Qwen3VLGRPOTrainer:
    """根据命令行参数组装模型、处理器、数据集和 Trainer。"""
    config = AutoConfig.from_pretrained(args.model_path)
    if hasattr(config.text_config, "rope_parameters") and getattr(config.text_config, "rope_scaling", None) is None:
        config.text_config.rope_scaling = config.text_config.rope_parameters

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        config=config,
        dtype=torch.bfloat16,
        device_map=None,
        attn_implementation="flash_attention_2",
    )
    # LoRA：只对语言解码器的 attention/MLP projection 插入 adapter，ViT/Merger 冻结不受影响
    if args.use_lora.lower() == "true":
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            # 只注入 LLM 部分；qkv + gate/up/down proj 覆盖 attention 与 MLP
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        # gradient_checkpointing 下输入不保留梯度，必须显式启用才能回传到 adapter
        model.enable_input_require_grads()
        model.print_trainable_parameters()
    # freeze_multimodal_components 必须在 get_peft_model() 之后调用，
    # 否则 PEFT wrapper 会将参数名改为 base_model.model.visual.*，导致冻结失效。
    model = freeze_multimodal_components(
        model,
        freeze_vit=args.freeze_vit.lower() == "true",
        freeze_merger=args.freeze_merger.lower() == "true",
    )


    processor = AutoProcessor.from_pretrained(args.model_path)
    train_dataset = load_rlvr_dataset(args.dataset_path)

    config_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        save_strategy="steps",
        save_steps=2,
        # save_strategy="epoch",
        save_only_model=True,
        save_total_limit=8,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        logging_dir=args.logging_dir,
        report_to=args.report_to,
        run_name=args.run_name,
        bf16=args.bf16,
        gradient_checkpointing=True,
        # max_prompt_length 在 TRL 0.29.1 GRPOConfig 中不存在，故不传入；
        # prompt 长度由调用方在数据预处理阶段控制（见 rvlr_run_guide.md §7.3.3）。
        max_completion_length=args.max_new_tokens,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.kl_coef,
        ds3_gather_for_generation=args.ds3_gather_for_generation.lower() == "true",
        mask_truncated_completions=args.mask_truncated_completions,
        use_vllm=args.use_vllm.lower() == "true",
        vllm_mode=args.vllm_mode,
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_length=args.vllm_max_model_length,
        num_iterations=args.num_iterations,
    )
    training_args = _build_grpo_config(**config_kwargs)

    trainer = Qwen3VLGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=processor,
        reward_fn=compute_rlvr_reward,
        # debug metrics 配置
        log_grpo_debug_metrics=getattr(args, "log_grpo_debug_metrics", True),
        debug_log_samples=getattr(args, "debug_log_samples", False),
        debug_log_samples_steps=getattr(args, "debug_log_samples_steps", 20),
        debug_num_samples=getattr(args, "debug_num_samples", 8),
        debug_reward_low_std_threshold=getattr(args, "debug_reward_low_std_threshold", 1e-6),
        debug_output_dir=getattr(args, "debug_output_dir", "./debug_grpo"),
    )
    return trainer


def parse_args():
    """命令行参数。"""
    parser = argparse.ArgumentParser(description="Qwen3-VL RLVR / GRPO 训练入口")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, required=True)
    parser.add_argument("--max_steps", type=int, default=-1, help="可选 smoke run 步数；-1 表示按 epoch 训练。")
    parser.add_argument("--per_device_train_batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        help="学习率调度器类型，如 linear / cosine / constant_with_warmup。")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="LR warmup 占总训练步数的比例；warmup_steps > 0 时该项会被覆盖。")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="固定 warmup 步数；设为 0 时使用 warmup_ratio。")
    parser.add_argument("--num_generations", type=int, required=True)
    parser.add_argument("--max_new_tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top_p", type=float, required=True)
    parser.add_argument("--kl_coef", type=float, required=True)
    # TRL 0.29.1 GRPOConfig 不支持 max_prompt_length，此参数仅保留以兼容旧调用脚本，不会传入 GRPOConfig。
    parser.add_argument("--max_prompt_length", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--logging_dir", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--freeze_vit", type=str, default="true")
    parser.add_argument("--freeze_merger", type=str, default="true")
    parser.add_argument("--use_lora", type=str, default="false", help="是否启用 LoRA 微调。")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--ds3_gather_for_generation",
        type=str,
        default="true",
        help="TRL GRPO ZeRO-3 generation gather 开关；显存不足时可设为 false，但会更慢。",
    )
    parser.add_argument(
        "--mask_truncated_completions",
        action="store_true",
        help="忽略被 max_completion_length 截断的 completion loss，降低截断样本噪声。",
    )
    parser.add_argument("--use_vllm", type=str, default="false")
    parser.add_argument("--vllm_mode", type=str, default="colocate", choices=("colocate", "server"))
    parser.add_argument("--vllm_server_host", type=str, default="0.0.0.0")
    parser.add_argument("--vllm_server_port", type=int, default=8000)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3)
    parser.add_argument("--vllm_max_model_length", type=int, default=None)
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--bf16", action="store_true")
    # debug metrics 配置
    parser.add_argument("--log_grpo_debug_metrics", action="store_true", default=True,
                        help="开启 GRPO debug metrics（group reward std / advantage / completion length 等）。")
    parser.add_argument("--no_log_grpo_debug_metrics", dest="log_grpo_debug_metrics", action="store_false",
                        help="关闭 GRPO debug metrics。")
    parser.add_argument("--debug_log_samples", action="store_true", default=False,
                        help="每隔 debug_log_samples_steps 步将样本 dump 到 JSONL。")
    parser.add_argument("--debug_log_samples_steps", type=int, default=20,
                        help="样本 dump 间隔步数。")
    parser.add_argument("--debug_num_samples", type=int, default=8,
                        help="每次 dump 的 prompt 数量（每个 prompt 含 G 个 completion）。")
    parser.add_argument("--debug_reward_low_std_threshold", type=float, default=1e-6,
                        help="group reward std 低于此阈值视为 low-std（学习信号弱）。")
    parser.add_argument("--debug_output_dir", type=str, default="./debug_grpo",
                        help="样本 JSONL dump 目录。")
    return parser.parse_args()


def main() -> None:
    """脚本入口。"""
    args = parse_args()
    # 仅在未指定 wandb 时才禁用，允许 report_to=wandb 正常上报
    if args.report_to != "wandb":
        os.environ.setdefault("WANDB_DISABLED", "true")
    trainer = build_trainer_from_args(args)
    trainer.train()


if __name__ == "__main__":
    main()
