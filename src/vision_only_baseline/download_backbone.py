"""
EfficientFormerV2-S1 ImageNet 权重下载脚本（本地有网环境运行）
=============================================================

用途：在「能联网的本地机器」上把 EfficientFormerV2-S1 的 ImageNet-1k 预训练权重
下载成单个 .pth 文件，再上传到内网/离线服务器，供 train.py 通过
--backbone_ckpt 离线加载，避免服务器联网失败。

下载的权重格式与 timm 内部结构一致（state_dict 键名不带 'model.' 前缀），
可直接被 model.py 中 `self.backbone.load_state_dict(state_dict, strict=True)` 使用。

依赖：
    pip install timm torch

用法：
    python download_backbone.py \
        --out /path/to/save/efficientformerv2_s1_in1k.pth

默认输出：./efficientformerv2_s1_in1k.pth
"""

import argparse
from pathlib import Path

import torch
import timm


def download_and_save(out_path: str):
    """
    下载 EfficientFormerV2-S1 (num_classes=0) 权重并保存。

    注意：必须 num_classes=0，与 model.py 中 backbone 定义一致，
    否则 load_state_dict(strict=True) 会在分类头维度上报错。
    """
    print(f"[Download] 正在用 timm 拉取 efficientformerv2_s1 (num_classes=0) ...")
    model = timm.create_model(
        "efficientformerv2_s1",
        pretrained=True,
        num_classes=0,  # 与 model.py 一致：移除分类头，输出特征向量
    )
    model.eval()

    # 取出 state_dict（键名形如 'stem.xxx', 'stages.xxx'，无 'model.' 前缀）
    state_dict = model.state_dict()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, out_path)
    print(f"[Download] 权重已保存: {out_path}")
    print(f"[Download] 键数量: {len(state_dict)} | "
          f"示例键: {list(state_dict.keys())[:3]}")
    print("[Download] 将该 .pth 文件上传到服务器，训练时用 "
          "--backbone_ckpt 指向它即可。")


def main():
    parser = argparse.ArgumentParser(
        description="下载 EfficientFormerV2-S1 ImageNet 权重为本地 .pth"
    )
    parser.add_argument(
        "--out", type=str, default="./efficientformerv2_s1_in1k.pth",
        help="输出 .pth 文件路径（默认 ./efficientformerv2_s1_in1k.pth）",
    )
    args = parser.parse_args()
    download_and_save(args.out)


if __name__ == "__main__":
    main()
