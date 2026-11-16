"""Download base model assets independently; no EALight weights are distributed."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--output_dir", default="models/Qwen3-VL-8B-Instruct")
    args = parser.parse_args()
    from modelscope.hub.snapshot_download import snapshot_download

    destination = Path(args.output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(args.model_id, local_dir=str(destination))


if __name__ == "__main__":
    main()
