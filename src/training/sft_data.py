"""Normalize public ShareGPT or prompt/chosen records for multimodal SFT."""

from copy import deepcopy
from pathlib import Path
import re


def normalize_sft_record(record, image_root):
    """Return conversational prompt/completion and paths, without inventing rejected data."""
    image_paths = record.get("images", record.get("image", []))
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    image_paths = list(image_paths)
    if "conversations" in record:
        roles = {"human": "user", "gpt": "assistant", "system": "system"}
        messages = [
            {"role": roles[turn["from"]], "content": turn["value"]}
            for turn in record["conversations"]
        ]
        if len(messages) < 2 or messages[-1]["role"] != "assistant":
            raise ValueError("SFT conversations must end with an assistant answer")
        prompt, completion = messages[:-1], [messages[-1]]
    else:
        prompt = deepcopy(record["prompt"])
        completion = deepcopy(record.get("completion", record.get("chosen")))
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        if isinstance(completion, str):
            completion = [{"role": "assistant", "content": completion}]
        if not isinstance(completion, list) or not completion:
            raise ValueError("SFT requires conversations, completion, or chosen")
    placeholders = 0
    embedded_paths = []
    for message in prompt:
        content = message["content"]
        if isinstance(content, str):
            pieces = re.split(r"(<image>)", content)
            parts = [
                {"type": "image"}
                if piece == "<image>"
                else {"type": "text", "text": piece}
                for piece in pieces
                if piece
            ]
        else:
            parts = deepcopy(content)
        for part in parts:
            if part.get("type") == "image":
                placeholders += 1
                image = part.pop("image", None)
                if image is not None:
                    if not isinstance(image, str):
                        raise ValueError(
                            "Embedded image entries must contain file paths"
                        )
                    embedded_paths.append(image)
        message["content"] = parts
    if not image_paths:
        image_paths = embedded_paths
    if not image_paths:
        raise ValueError("SFT sample has no images")
    if placeholders == 0:
        user = next((message for message in prompt if message["role"] == "user"), None)
        if user is None:
            raise ValueError("SFT prompt has no user message")
        user["content"] = [{"type": "image"} for _ in image_paths] + user["content"]
    elif placeholders != len(image_paths):
        raise ValueError(
            f"Image placeholder count {placeholders} differs from image count {len(image_paths)}"
        )
    resolved = []
    for image in image_paths:
        path = Path(image)
        path = path if path.is_absolute() else Path(image_root) / path
        if not path.is_file():
            raise FileNotFoundError(f"Missing SFT image: {path}")
        resolved.append(str(path.resolve()))
    # TRL's vision-language collator accepts conversational prompt/completion records.
    for message in completion:
        if message["role"] != "assistant":
            raise ValueError("SFT completion must contain assistant messages")
        if isinstance(message["content"], list):
            message["content"] = "".join(
                part["text"]
                for part in message["content"]
                if part.get("type") == "text"
            )
    return {"prompt": prompt, "completion": completion, "images": resolved}


def load_sft_dataset(dataset_path, image_root=None):
    from datasets import load_dataset
    from PIL import Image

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    root = image_root or str(Path(dataset_path).resolve().parent)
    dataset = dataset.map(
        lambda record: normalize_sft_record(record, root),
        remove_columns=dataset.column_names,
        desc="Normalize SFT data",
    )

    def load_images(batch):
        images = []
        for paths in batch["images"]:
            sample_images = []
            for path in paths:
                with Image.open(path) as image:
                    sample_images.append(image.convert("RGB"))
            images.append(sample_images)
        return {**batch, "images": images}

    dataset.set_transform(load_images)
    return dataset
