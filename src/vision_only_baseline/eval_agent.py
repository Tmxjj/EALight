"""Local vision-only inference with the evaluator's batched decision interface."""

import time
from typing import Optional

import torch
from loguru import logger

from src.vision_only_baseline.dataset import (
    build_transform,
    canonical_image_paths,
    load_image_tensor,
)
from src.vision_only_baseline.model import build_model, DualActionHead, MAX_PHASES


class VisionOnlyAgent:
    def __init__(
        self, checkpoint_path, num_valid_phases=None, device=None, image_size=None
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = checkpoint.get("config", {})
        self.image_size = (
            image_size if image_size is not None else config.get("image_size", 224)
        )
        self.transform = build_transform(self.image_size)
        self.num_valid_phases = (
            MAX_PHASES if num_valid_phases is None else num_valid_phases
        )
        if not 1 <= self.num_valid_phases <= MAX_PHASES:
            raise ValueError(f"Expected 1..{MAX_PHASES} valid phases")
        self.model = build_model(pretrained=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        logger.info(
            f"[VisionOnlyAgent] checkpoint={checkpoint_path}, device={self.device}, phases={self.num_valid_phases}"
        )

    def _load_images(self, image_paths):
        paths = canonical_image_paths(image_paths, allow_missing=True)
        return load_image_tensor(paths, self.transform, self.image_size)

    def get_decision(self, image_paths):
        return self.get_batch_decision([image_paths])[0][2]

    @torch.inference_mode()
    def get_batch_decision(self, batch_image_paths, batch_prompts=None):
        if not batch_image_paths:
            return []
        start = time.perf_counter()
        images = torch.stack(
            [self._load_images(paths) for paths in batch_image_paths]
        ).to(self.device)
        phase_logits, duration_logits = self.model(images)
        phases = self.model.action_head.masked_phase_predict(
            phase_logits, self.num_valid_phases
        ).tolist()
        duration_indices = duration_logits.argmax(dim=-1).tolist()
        # Host copies above synchronize CUDA before wall-clock timing is recorded.
        latency = (time.perf_counter() - start) / len(batch_image_paths)
        results = []
        for phase, index in zip(phases, duration_indices):
            duration = DualActionHead.DURATION_CANDIDATES[index]
            response = f'Action: {{"phase": {phase}, "duration": {duration}}}'
            results.append((response, latency, (phase, duration), None))
        return results


def build_agent_from_scenario(
    checkpoint_path, scenario_key, device: Optional[str] = None
):
    from configs.scenairo_config import SCENARIO_CONFIGS

    if scenario_key not in SCENARIO_CONFIGS:
        raise ValueError(f"Unknown scenario: {scenario_key}")
    return VisionOnlyAgent(
        checkpoint_path, SCENARIO_CONFIGS[scenario_key]["PHASE_NUMBER"], device
    )
