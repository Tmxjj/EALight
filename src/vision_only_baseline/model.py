"""EfficientFormerV2-S1 + attention pooling + phase/duration classification.

The original baseline architecture is retained. Pooling has no explicit direction
encoding. Masking guarantees legal phase IDs, not semantic alignment across
networks with different signal movements. See README.md for transfer limitations.
"""

import os

import torch
import torch.nn as nn
import timm
from loguru import logger

# 训练时使用的最大相位数（与 Syn-Train 场景一致）
MAX_PHASES = 4


class AttentionPooling(nn.Module):
    """
    跨视角注意力池化模块。

    使用可学习的 query token 对多视角特征进行加权聚合，
    借鉴 AttendLight (NeurIPS 2020) 的 attention-pooling 思想，
    使模型能够自适应地关注不同方向的交通状况。

    Args:
        embed_dim: 输入特征维度
        num_heads: 多头注意力头数
    """

    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))  # 可学习 query
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, num_views, D] 多视角特征
        Returns:
            [B, D] 融合后的全局特征
        """
        B = x.size(0)
        query = self.query.expand(B, -1, -1)  # [B, 1, D]
        # query 关注所有视角的特征
        out, _ = self.attn(query, x, x, need_weights=False)  # [B, 1, D]
        out = self.norm(out.squeeze(1))        # [B, D]
        return out


class DualActionHead(nn.Module):
    """
    双分类头：分别预测 phase 和 duration。

    - Phase Head: 固定输出 MAX_PHASES=4 维 logits（训练与 Syn-Train 对齐）。
      推理时通过 masked_phase_predict() 将非法相位屏蔽，实现 zero-shot 跨场景迁移。
    - Duration Head: 输出 num_durations 个类别的 logits，
      对应候选集 {15, 20, 25, 30, 35, 40}s。

    Args:
        embed_dim: 输入特征维度
        num_phases: 训练时相位数（固定为 MAX_PHASES=4）
        num_durations: 时长候选数量（默认 6）
        dropout: Dropout 比率
    """

    # 时长候选集（与 EALight 完全一致）
    DURATION_CANDIDATES = [15, 20, 25, 30, 35, 40]

    def __init__(
        self,
        embed_dim: int,
        num_phases: int = MAX_PHASES,
        num_durations: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_phases = num_phases
        self.num_durations = num_durations

        # Phase 分类头（固定 MAX_PHASES=4 维输出）
        self.phase_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_phases),
        )

        # Duration 分类头
        self.duration_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_durations),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, D] 融合后的全局特征
        Returns:
            phase_logits:    [B, MAX_PHASES=4]  始终输出 4 维（训练和推理统一）
            duration_logits: [B, num_durations]
        """
        phase_logits = self.phase_head(x)
        duration_logits = self.duration_head(x)
        return phase_logits, duration_logits

    def masked_phase_predict(
        self, phase_logits: torch.Tensor, num_valid_phases: int
    ) -> torch.Tensor:
        """
        Zero-Shot 跨场景相位预测：Masked Argmax。

        将超出当前场景合法相位范围的 logits 置为 -inf，
        确保 argmax 只在 [0, num_valid_phases) 范围内选择。

        场景对应关系：
          - France_Massy:   num_valid_phases=2，屏蔽 logits[2], logits[3]
          - Hongkong_YMT:   num_valid_phases=3，屏蔽 logits[3]
          - JiNan/Hangzhou/Songdo: num_valid_phases=4，不屏蔽

        Args:
            phase_logits:    [B, MAX_PHASES] 原始 logits
            num_valid_phases: 当前场景实际相位数（2/3/4）
        Returns:
            [B] 预测的相位 ID（保证在 [0, num_valid_phases) 内）
        """
        if type(num_valid_phases) is not int or not 1 <= num_valid_phases <= self.num_phases:
            raise ValueError(f"num_valid_phases must be in [1, {self.num_phases}]")
        if num_valid_phases == self.num_phases:
            # 相位数与训练一致，直接 argmax
            return phase_logits.argmax(dim=-1)

        # 将非法相位的 logits 置为 -inf（屏蔽）
        masked_logits = phase_logits.clone()
        masked_logits[:, num_valid_phases:] = float('-inf')
        return masked_logits.argmax(dim=-1)


class VisionOnlyTSC(nn.Module):
    """
    Vision-Only Traffic Signal Controller

    纯视觉端到端交通信号控制器，不含任何语言推理通道。

    架构流程：
      1. 4 张多视角图像分别通过共享权重的 EfficientFormerV2-S1 提取特征
      2. 4 个视角特征经 Attention Pooling 融合为全局表征
      3. Dual Action Head 分别输出 phase 和 duration 的分类结果

    Zero-Shot 跨场景支持：
      - 训练：固定在 Syn-Train（4 相位）上训练
      - 推理：通过 predict(num_valid_phases=N) 支持 2/3/4 相位场景

    Args:
        num_phases: 训练时相位数（固定为 MAX_PHASES=4）
        num_durations: 时长候选数量（默认 6）
        pretrained: 是否使用 ImageNet 预训练权重
        dropout: 分类头 Dropout 比率
    """

    def __init__(
        self,
        num_phases: int = MAX_PHASES,
        num_durations: int = 6,
        pretrained: bool = True,
        dropout: float = 0.1,
        backbone_ckpt: str = None,
    ):
        """
        Args:
            num_phases:    训练时相位数（固定为 MAX_PHASES=4）
            num_durations: 时长候选数量（默认 6）
            pretrained:    是否使用 ImageNet 预训练权重（联网下载）
            dropout:       分类头 Dropout 比率
            backbone_ckpt: 本地 backbone 权重文件路径（.pth）。
                           - 指定后：离线加载该文件中的 ImageNet 预训练权重，
                             不再联网；pretrained 参数被忽略。
                           - None：沿用 timm 默认行为（pretrained=True 联网下载）。
        """
        super().__init__()
        self.num_phases = num_phases
        self.num_durations = num_durations

        # ── 视觉编码器：EfficientFormerV2-S1 ──
        # 使用 timm 库加载，移除原始分类头（num_classes=0 返回特征向量）
        if backbone_ckpt is not None:
            # 离线模式：先用 pretrained=False 搭结构，再从本地文件加载权重
            if not os.path.exists(backbone_ckpt):
                raise FileNotFoundError(f"本地 backbone 权重不存在: {backbone_ckpt}")
            logger.info(f"[Model] 离线加载 backbone 权重: {backbone_ckpt}")
            self.backbone = timm.create_model(
                "efficientformerv2_s1",
                pretrained=False,
                num_classes=0,
            )
            state_dict = torch.load(backbone_ckpt, map_location="cpu", weights_only=True)
            state_dict = state_dict.get("state_dict", state_dict.get("model", state_dict))
            # 容错：若传入的是完整 timm 权重（含 'model.' 前缀）则去掉前缀
            state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
            self.backbone.load_state_dict(state_dict, strict=True)
        else:
            # 在线模式：timm 自动联网下载 ImageNet 预训练权重
            self.backbone = timm.create_model(
                "efficientformerv2_s1",
                pretrained=pretrained,
                num_classes=0,  # 移除分类头，输出 [B, embed_dim] 特征
            )
        # 获取 backbone 输出特征维度
        self.embed_dim = self.backbone.num_features

        # ── 跨视角注意力池化 ──
        self.attention_pooling = AttentionPooling(
            embed_dim=self.embed_dim,
            num_heads=4,
        )

        # ── 双分类头（Phase Head 固定 MAX_PHASES=4 维）──
        self.action_head = DualActionHead(
            embed_dim=self.embed_dim,
            num_phases=num_phases,
            num_durations=num_durations,
            dropout=dropout,
        )

    def encode_views(self, images: torch.Tensor) -> torch.Tensor:
        """
        对多视角图像分别编码（共享 backbone 权重）。

        Args:
            images: [B, num_views, C, H, W] 多视角图像张量
        Returns:
            [B, num_views, D] 各视角的特征向量
        """
        B, V, C, H, W = images.shape
        # 展平为 [B*V, C, H, W] 以批量通过 backbone
        x = images.reshape(B * V, C, H, W)
        features = self.backbone(x)          # [B*V, D]
        features = features.view(B, V, -1)   # [B, V, D]
        return features

    def forward(self, images: torch.Tensor):
        """
        前向传播：从多视角图像到 (phase, duration) 动作 logits。

        Args:
            images: [B, num_views, C, H, W] 多视角 stop-line 图像
        Returns:
            phase_logits:    [B, MAX_PHASES=4]
            duration_logits: [B, num_durations]
        """
        view_features = self.encode_views(images)           # [B, V, D]
        fused = self.attention_pooling(view_features)       # [B, D]
        phase_logits, duration_logits = self.action_head(fused)
        return phase_logits, duration_logits

    def predict(self, images: torch.Tensor, num_valid_phases: int = None):
        """
        推理接口：支持 zero-shot 跨场景（2/3/4 相位）。

        通过 num_valid_phases 参数将非法相位屏蔽，
        确保输出的 phase_id 始终在当前场景的合法范围内。

        Args:
            images:           [B, num_views, C, H, W]
            num_valid_phases: 当前场景实际相位数（2/3/4）。
                              None 表示使用训练时的 MAX_PHASES=4。
        Returns:
            phase_ids: [B] 预测的相位 ID（保证合法）
            durations: [B] 预测的绿灯时长（秒）
        """
        self.eval()
        if num_valid_phases is None:
            num_valid_phases = self.action_head.num_phases

        with torch.no_grad():
            phase_logits, duration_logits = self.forward(images)
            # Masked Argmax：保证相位合法性
            phase_ids = self.action_head.masked_phase_predict(
                phase_logits, num_valid_phases
            )
            duration_indices = duration_logits.argmax(dim=-1)
            durations = torch.tensor(
                DualActionHead.DURATION_CANDIDATES, device=images.device
            )[duration_indices]
        return phase_ids, durations


def build_model(pretrained: bool = True, backbone_ckpt: str = None) -> VisionOnlyTSC:
    """
    构建 Vision-Only TSC 模型的工厂函数。

    训练时始终使用 MAX_PHASES=4（与 Syn-Train 对齐）。
    推理时通过 predict(num_valid_phases=N) 支持 2/3/4 相位场景的 zero-shot 迁移。

    Args:
        pretrained:    是否加载 ImageNet 预训练权重（联网下载）
        backbone_ckpt: 本地 backbone 权重路径（.pth）。指定后离线加载，
                       无需联网，覆盖 pretrained 行为。
    Returns:
        VisionOnlyTSC 模型实例
    """
    model = VisionOnlyTSC(
        num_phases=MAX_PHASES,  # 固定 4，与 Syn-Train 一致
        num_durations=6,
        pretrained=pretrained,
        dropout=0.1,
        backbone_ckpt=backbone_ckpt,
    )
    return model
