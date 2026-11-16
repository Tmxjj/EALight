
import os
import argparse
import zlib
from typing import Optional, Union

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────
# 扰动类型与强度预设
# ─────────────────────────────────────────────────────────────────
# 每类扰动设三档强度，数值含义见各子函数说明。
# 论文若需调参，集中修改此处即可，无需改动算法主体。
ROBUST_LEVELS = {
    # occlusion（部分遮挡）：用中性灰矩形块遮住画面的一部分。
    #   area_ratio : 被遮挡总面积的目标比例（各块随机摆放可能轻微重叠，
    #                故实测覆盖比例可能略低于该目标，属预期行为）；
    #   n_patches  : 遮挡块数量（目标总面积在各块间均分）。
    "occlusion": {
        "light":  {"area_ratio": 0.05, "n_patches": 1},
        "medium": {"area_ratio": 0.12, "n_patches": 2},
        "heavy":  {"area_ratio": 0.35, "n_patches": 3},
    },
    # noise（高斯噪声）：在 0~255 像素值上叠加均值为 0 的高斯噪声后裁剪。
    #   sigma : 高斯噪声标准差（0~255 灰度尺度）。
    "noise": {
        "light":  {"sigma": 10.0},
        "medium": {"sigma": 25.0},
        "heavy":  {"sigma": 50.0},
    },
    # lowres（低分辨率）：先按 scale 下采样，再放大回原尺寸，制造细节丢失/块状感。
    #   scale : 下采样比例（越小越模糊）。
    "lowres": {
        "light":  {"scale": 0.5},
        "medium": {"scale": 0.25},
        "heavy":  {"scale": 0.125},
    },
}

# 单一扰动类型（各自拥有独立强度参数，见 ROBUST_LEVELS）
SINGLE_PERTURB_TYPES = tuple(ROBUST_LEVELS.keys())
# 组合扰动："all" 表示在同一档强度下依次叠加全部三类扰动。
COMBINED_PERTURB_TYPE = "all"
# 组合扰动的施加顺序：先物理遮挡 → 再降分辨率 → 最后叠加传感器噪声。
# 噪声放在最后，避免被后续下采样平滑掉，确保三种退化在最终图像上都清晰可见。
COMBINED_ORDER = ("occlusion", "lowres", "noise")
# 合法的扰动类型与强度档位（供 CLI / 上游参数校验复用），含组合扰动 "all"
VALID_PERTURB_TYPES = SINGLE_PERTURB_TYPES + (COMBINED_PERTURB_TYPE,)
VALID_LEVELS = ("light", "medium", "heavy")

# 遮挡块填充颜色（中性灰，避免引入额外的明暗偏置）
_OCCLUSION_FILL = (128, 128, 128)
# 遮挡块可贴合的四条画面边缘（真实遮挡多由边缘的杆件/树木/建筑/邻车造成）
_OCCLUSION_EDGES = ("top", "bottom", "left", "right")


def _resolve_rng(seed: Optional[Union[int, str]], perturb_type: str) -> np.random.RandomState:
    """根据 seed 与扰动类型构造一个确定性随机数发生器。

    说明：
      - 直接用 Python 内置 hash() 不可靠（受 PYTHONHASHSEED 影响，跨进程不稳定），
        因此用 zlib.crc32 对 "seed_类型" 字符串求一个稳定的 32 位整数作为种子；
      - 把 perturb_type 拼进种子，可使同一张图在不同扰动类型下得到不同的随机模式，
        避免遮挡与（潜在的）其它随机扰动产生相同布局；
      - seed 为 None 时返回非确定性 RNG（仅用于临时预览，不建议在正式评测中使用）。
    """
    if seed is None:
        return np.random.RandomState()
    base = zlib.crc32(f"{seed}_{perturb_type}".encode("utf-8")) & 0xFFFFFFFF
    return np.random.RandomState(base)


def _apply_occlusion(image: np.ndarray, area_ratio: float, n_patches: int,
                     rng: np.random.RandomState) -> np.ndarray:
    """部分遮挡：在画面边缘贴合放置若干中性灰矩形块。

    真实监控中的遮挡绝大多数由画面边缘的前景物体造成（杆件、树木、建筑物、
    紧贴镜头的邻车等），而非凭空出现在画面中央。因此遮挡块一律"从某条边缘
    向画面内侵入"，且长边沿边缘方向延展，呈条带状，更贴近真实遮挡形态。

    参数：
      image      : HxWx3 的 numpy 图像（RGB 或 BGR 均可，遮挡为灰色，与通道顺序无关）。
      area_ratio : 被遮挡总面积的目标比例（0~1；相邻边缘的条带在画面角部可能
                   轻微重叠，实测覆盖可能略低于该目标，属预期行为）。
      n_patches  : 遮挡块数量，目标总面积在各块间近似均分。
      rng        : 确定性随机数发生器（控制所贴边缘、条带长宽比与沿边缘的位置）。

    实现：
      - 每个遮挡块的目标面积 = 整图面积 * area_ratio / n_patches；
      - 尽量让各块落在不同边缘（先随机打乱四条边缘再依次取用），使遮挡分布在
        画面四周、减少彼此重叠；
      - 长边沿边缘方向延展（长宽比 1.5~3.0），短边为向画面内侵入的深度；
      - 沿边缘方向随机平移摆放，并把另一维紧贴对应边缘（top/left 贴 0 侧，
        bottom/right 贴末端）。
    """
    out = image.copy()
    h, w = out.shape[:2]
    total_area = float(h * w)
    # 单块目标面积（按块数均分总遮挡面积）
    per_patch_area = total_area * float(area_ratio) / max(1, n_patches)

    # 先随机打乱四条边缘，依次分配给各遮挡块：n_patches<=4 时各块落在不同边缘。
    edges = list(rng.permutation(list(_OCCLUSION_EDGES)))

    for i in range(n_patches):
        edge = str(edges[i % len(edges)])
        # 条带形：长边沿边缘方向（1.5~3.0 倍），短边为侵入深度
        aspect = rng.uniform(1.5, 3.0)
        short_side = max(1, int(round((per_patch_area / aspect) ** 0.5)))
        long_side = max(1, int(round(short_side * aspect)))

        if edge in ("top", "bottom"):
            # 水平边缘：宽=沿边缘延展(long)，高=向内侵入深度(short)
            pw = min(long_side, w)
            ph = min(short_side, h)
            x0 = int(rng.randint(0, max(1, w - pw + 1)))   # 沿水平边缘随机平移
            y0 = 0 if edge == "top" else h - ph            # 紧贴上/下边缘
        else:
            # 垂直边缘：高=沿边缘延展(long)，宽=向内侵入深度(short)
            pw = min(short_side, w)
            ph = min(long_side, h)
            y0 = int(rng.randint(0, max(1, h - ph + 1)))   # 沿垂直边缘随机平移
            x0 = 0 if edge == "left" else w - pw            # 紧贴左/右边缘

        out[y0:y0 + ph, x0:x0 + pw] = _OCCLUSION_FILL

    return out


def _apply_noise(image: np.ndarray, sigma: float,
                 rng: np.random.RandomState) -> np.ndarray:
    """高斯噪声：在像素值上叠加均值 0、标准差 sigma 的高斯噪声后裁剪到 [0,255]。"""
    noise = rng.normal(0.0, float(sigma), size=image.shape)
    out = image.astype(np.float32) + noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def _apply_lowres(image: np.ndarray, scale: float) -> np.ndarray:
    """低分辨率：下采样到 scale 倍后再放大回原尺寸，制造细节丢失与块状感。

    - 下采样用 INTER_AREA（抗锯齿、贴近真实降采样）；
    - 上采样用 INTER_NEAREST（保留块状感，凸显低分辨率退化）。
    """
    h, w = image.shape[:2]
    small_w = max(1, int(round(w * float(scale))))
    small_h = max(1, int(round(h * float(scale))))
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return out


def _apply_single_perturbation(image: np.ndarray, perturb_type: str, level: str,
                               seed: Optional[Union[int, str]]) -> np.ndarray:
    """施加单一类型扰动（occlusion / noise / lowres），返回扰动后图像。

    调用方需保证 perturb_type ∈ SINGLE_PERTURB_TYPES、level ∈ VALID_LEVELS。
    每类扰动按自身类型独立取随机种子（见 _resolve_rng），保证组合扰动时各子扰动
    的随机模式互不相同且可复现。
    """
    params = ROBUST_LEVELS[perturb_type][level]
    rng = _resolve_rng(seed, perturb_type)

    if perturb_type == "occlusion":
        return _apply_occlusion(image, params["area_ratio"], params["n_patches"], rng)
    if perturb_type == "noise":
        return _apply_noise(image, params["sigma"], rng)
    if perturb_type == "lowres":
        return _apply_lowres(image, params["scale"])

    # 理论上不可达（调用方已校验），保留防御性分支。
    raise ValueError(f"未实现的扰动类型: {perturb_type}")


def apply_visual_perturbation(image: np.ndarray,
                              perturb_type: Optional[str],
                              level: str = "medium",
                              seed: Optional[Union[int, str]] = None) -> np.ndarray:
    """对单张图像施加指定类型与强度的视觉扰动，返回扰动后的图像副本。

    参数：
      image        : HxWx3 的 numpy 图像（RGB 或 BGR 均可，三类扰动均与通道顺序无关）。
      perturb_type : 扰动类型：
                       - occlusion / noise / lowres：单一扰动；
                       - all：组合扰动，在同一档强度下按 COMBINED_ORDER
                         （occlusion → lowres → noise）依次叠加全部三类扰动；
                       - None / "none"：不做任何处理（返回原图副本）。
      level        : 强度档位，取值 light / medium / heavy（默认 medium）；
                     组合扰动时三类子扰动统一使用该档位。
      seed         : 确定性随机种子（可为 int 或字符串）。给定相同 seed 与类型/强度，
                     遮挡位置与噪声分布完全一致，保证实验可复现。

    返回：
      扰动后的 numpy 图像（dtype=uint8，与输入同形状）。

    异常：
      perturb_type / level 非法时抛出 ValueError。
    """
    # 关闭扰动：直接返回副本，保证调用方拿到的是独立对象。
    if perturb_type is None or str(perturb_type).strip().lower() in ("", "none"):
        return image.copy()

    perturb_type = str(perturb_type).strip().lower()
    level = str(level).strip().lower()

    if level not in VALID_LEVELS:
        raise ValueError(
            f"非法强度档位: {level}，合法取值为 {VALID_LEVELS}"
        )

    # 组合扰动：按固定顺序依次叠加全部三类扰动，统一使用同一强度档位。
    if perturb_type == COMBINED_PERTURB_TYPE:
        out = image
        for sub_type in COMBINED_ORDER:
            out = _apply_single_perturbation(out, sub_type, level, seed)
        return out

    # 单一扰动
    if perturb_type not in SINGLE_PERTURB_TYPES:
        raise ValueError(
            f"非法扰动类型: {perturb_type}，合法取值为 {VALID_PERTURB_TYPES}"
        )
    return _apply_single_perturbation(image, perturb_type, level, seed)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="视觉鲁棒性扰动预览：对一张样例图生成各类型/各强度的扰动结果，"
                    "用于论文配图与本地核验。"
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="输入图像路径（任意常见格式）")
    parser.add_argument("--output_dir", "-o", type=str, default=None,
                        help="输出目录，默认与输入图像同目录")
    parser.add_argument("--perturb", "-p", type=str, default=None,
                        choices=list(VALID_PERTURB_TYPES),
                        help="只生成指定扰动类型；不指定则三类全生成")
    parser.add_argument("--level", "-L", type=str, default=None,
                        choices=list(VALID_LEVELS),
                        help="只生成指定强度档位；不指定则三档全生成")
    parser.add_argument("--seed", "-s", type=str, default="preview",
                        help="确定性随机种子，默认 preview")
    return parser


def _preview_main():
    """命令行预览入口：

    示例：
      # 生成三类扰动 × 三档强度共 9 张结果
      python scripts/visual_robustness.py -i data/test/Hongkong_YMT/normal/23/J1_1_front.png 

      # 只看中等强度遮挡
      python scripts/visual_robustness.py -i sample.png -p occlusion -L medium
    """
    args = _build_arg_parser().parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"找不到输入图像: {args.input}")

    # cv2.imread 读入的是 BGR；三类扰动均与通道顺序无关，故直接处理后再写回。
    image = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图像（格式不支持或文件损坏）: {args.input}")

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(out_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(args.input))[0]
    perturb_types = [args.perturb] if args.perturb else list(VALID_PERTURB_TYPES)
    levels = [args.level] if args.level else list(VALID_LEVELS)

    for ptype in perturb_types:
        for lvl in levels:
            perturbed = apply_visual_perturbation(
                image, perturb_type=ptype, level=lvl, seed=args.seed
            )
            out_path = os.path.join(out_dir, f"{stem}_{ptype}_{lvl}.png")
            cv2.imwrite(out_path, perturbed)
            print(f"✅ 已生成: {out_path}")


if __name__ == "__main__":
    _preview_main()
