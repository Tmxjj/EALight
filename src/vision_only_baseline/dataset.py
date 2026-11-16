"""ShareGPT image/action pairs used by the vision-only supervised baseline."""

import ast
import json
import re
from pathlib import Path
from typing import Optional, Tuple

import torch
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

DURATION_CANDIDATES = [15, 20, 25, 30, 35, 40]
DURATION_TO_INDEX = {
    duration: index for index, duration in enumerate(DURATION_CANDIDATES)
}
VIEW_DIRECTIONS = ("N", "E", "S", "W")


def parse_action_from_response(response: str) -> Optional[Tuple[int, int]]:
    """Read the final Action object, without taking a proposed action from Thought."""
    if not isinstance(response, str):
        return None
    actions = list(
        re.finditer(
            r"^\s*Action\s*:\s*(\{[^{}]*\})", response, re.IGNORECASE | re.MULTILINE
        )
    )
    if actions:
        text = actions[-1].group(1)
    else:
        text = response.strip()
        if not (text.startswith("{") and text.endswith("}")):
            return None
    try:
        try:
            action = json.loads(text)
        except json.JSONDecodeError:
            action = ast.literal_eval(text)
        phase, duration = action["phase"], action["duration"]
        if type(phase) is not int or type(duration) is not int:
            return None
        return phase, duration
    except (ValueError, SyntaxError, TypeError, KeyError):
        return None


def build_transform(image_size: int = 224, augment: bool = False):
    if image_size != 224:
        raise ValueError(
            "EfficientFormerV2-S1 uses fixed 224x224 inputs; image_size must be 224"
        )
    operations = [transforms.Resize((image_size, image_size))]
    if augment:
        # Geometric flips change lane/turn semantics without changing the target action.
        operations.append(
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(operations)


def canonical_image_paths(image_paths, allow_missing=False):
    """Place named views in N/E/S/W slots; unnamed four-view lists retain their order."""
    if not isinstance(image_paths, (list, tuple)) or not 1 <= len(image_paths) <= 4:
        raise ValueError("Expected one to four image paths")
    slots = [None] * 4
    named = 0
    for path in image_paths:
        if not isinstance(path, (str, Path)):
            raise ValueError("Image paths must be strings or Path objects")
        match = re.search(
            r"(?:^|_)(N|E|S|W)(?:_(?:no_watermark|perturbed))?$", Path(path).stem
        )
        if match:
            named += 1
            index = VIEW_DIRECTIONS.index(match.group(1))
            if slots[index] is not None:
                raise ValueError(f"Duplicate {match.group(1)} view")
            slots[index] = str(path)
    if named == len(image_paths):
        if not allow_missing and any(path is None for path in slots):
            raise ValueError("Training requires all four N/E/S/W views")
        return slots
    if named or len(image_paths) != 4:
        raise ValueError(
            "Partial view lists must have explicit N/E/S/W filename suffixes"
        )
    return [str(path) for path in image_paths]


def load_image_tensor(image_paths, transform, image_size=224):
    """Fail on unreadable supplied images; zeros represent absent approaches only."""
    images = []
    for path in image_paths:
        if path is None:
            images.append(torch.zeros(3, image_size, image_size))
        else:
            with Image.open(path) as image:
                images.append(transform(image.convert("RGB")))
    return torch.stack(images)


class TSCImageDataset(Dataset):
    def __init__(
        self, data_path, num_phases=4, image_size=224, augment=True, image_root=None
    ):
        self.num_phases = num_phases
        self.image_size = image_size
        self.transform = build_transform(image_size, augment)
        data_path = Path(data_path).resolve()
        root = Path(image_root).resolve() if image_root else data_path.parent
        content = data_path.read_text(encoding="utf-8").strip()
        try:
            raw = json.loads(content)
            raw = raw if isinstance(raw, list) else [raw]
        except json.JSONDecodeError:
            raw = [json.loads(line) for line in content.splitlines() if line.strip()]
        self.samples = []
        skipped = 0
        for item in raw:
            if not isinstance(item, dict):
                skipped += 1
                continue
            conversations = item.get("conversations", [])
            if not isinstance(conversations, list):
                skipped += 1
                continue
            response = next(
                (
                    entry.get("value", "")
                    for entry in reversed(conversations)
                    if isinstance(entry, dict) and entry.get("from") == "gpt"
                ),
                "",
            )
            action = parse_action_from_response(response)
            if (
                action is None
                or not 0 <= action[0] < num_phases
                or action[1] not in DURATION_TO_INDEX
            ):
                skipped += 1
                continue
            try:
                paths = canonical_image_paths(item.get("image", []))
            except ValueError:
                skipped += 1
                continue
            paths = [
                str(Path(path) if Path(path).is_absolute() else root / path)
                for path in paths
            ]
            for path in paths:
                if not Path(path).is_file():
                    raise FileNotFoundError(
                        f"Sample {item.get('id', '?')}: missing image {path}"
                    )
            self.samples.append(
                {
                    "id": item.get("id", "unknown"),
                    "image_paths": paths,
                    "phase_id": action[0],
                    "duration_index": DURATION_TO_INDEX[action[1]],
                }
            )
        if not self.samples:
            raise ValueError(
                f"No valid image/action samples in {data_path}; skipped {skipped}"
            )
        logger.info(
            f"[Dataset] {len(self.samples)} samples, {skipped} invalid labels/views skipped: {data_path}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return (
            load_image_tensor(sample["image_paths"], self.transform, self.image_size),
            sample["phase_id"],
            sample["duration_index"],
        )


def build_dataloader(
    data_path,
    num_phases=4,
    batch_size=32,
    num_workers=4,
    augment=True,
    shuffle=True,
    image_size=224,
    image_root=None,
):
    dataset = TSCImageDataset(data_path, num_phases, image_size, augment, image_root)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
