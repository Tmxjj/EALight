# EALight: Event-Aware Traffic Signal Control via Vision-Language World Knowledge

> **[中文说明见下方 / Chinese version below ↓](#中文说明)**

---

## Project Structure

```
EALight/
├── README.md
├── requirements.txt
├── install_deps.sh             # One-command environment setup
├── vgl_python.sh               # VirtualGL launcher for headless 3D rendering
│
├── configs/
│   ├── env_config.py           # SUMO / TransSimHub environment settings
│   ├── model_config.py         # VLM API config (local vLLM / Gemini / OpenAI)
│   ├── prompt_builder.py       # Adaptive CoT prompt (main)
│   ├── prompt_builder_fixed_cot.py     # Ablation: forced slow path
│   ├── prompt_builder_fixed_durtion.py # Ablation: fixed green duration
│   ├── scenairo_config.py      # Scenario topology & phase-space definitions
│   ├── accelerate_config.yaml  # Distributed training config
│   ├── ds_config_lora.json     # DeepSpeed LoRA config
│   └── train_config.yaml
│
├── src/
│   ├── bev_generation/         # Multi-view stop-line image rendering
│   ├── evaluation/
│   │   ├── run_eval.py         # Main evaluation script
│   │   ├── metrics.py          # ATT / AWT / sp-ATT / sp-AWT / MaxQL / TCR
│   │   ├── collect_metrics.py  # Batch result aggregation
│   │   └── analyze_event_recognition.py
│   ├── inference/
│   │   ├── vlm_agent.py        # VLM agent: prompt build + action parse
│   │   ├── vllm_load_model.sh  # Start vLLM inference server
│   │   └── vllm_load_model_lora.sh
│   ├── training/
│   │   ├── trainer.py          # SFT trainer
│   │   ├── sft_trainer.sh
│   │   ├── rlvr_trainer.sh     # RLVR / GRPO launch script
│   │   └── rlvr/               # GRPO trainer, reward functions, parsers
│   └── utils/
│       ├── event_bulletin.py   # Inter-intersection event broadcast (Prompt layer)
│       ├── make_tsc_env.py
│       └── tsc_env/            # TSC wrapper: action decode, state management
│
├── scripts/
│   ├── run_batch_event_eval.sh # Batch evaluation (baselines + VLM + all event types)
│   ├── add_lane_watermarks.py  # Add directional labels to intersection images
│   ├── visual_robustness.py    # Visual perturbation pipeline (occlusion / noise / lowres)
│   └── download_model.py       # Download base LVLM from ModelScope
│
├── TransSimHub/                # 3D traffic simulation environment
│   │                           # Full source + 3D assets hosted on anonymous GitHub:
│   │                           #   <ANONYMOUS_TRANSSIMHUB_GITHUB_URL>
│   └── tshub/                  # Core Python package (pip install -e .)
│
└── data/
    └── raw/                    # Included in this software package
        ├── JiNan/              # 12-intersection, real traffic (China)
        ├── SouthKorea_Songdo/  # Large asymmetric 6-lane junction
        ├── France_Massy/       # T-junction (2-phase)
        ├── Hongkong_YMT/       # Left-hand traffic (3-phase)
        └── Syn-Train/          # 4×4 synthetic grid (for RLVR training rollouts)
```

> **TransSimHub** (with 3D vehicle assets, ~650 MB) is hosted separately on an anonymous GitHub repository due to file size. Clone it as described in Step 3 below.

---

## Requirements

| Component | Version |
|-----------|---------|
| Python | 3.10 |
| SUMO | ≥ 1.20 |
| CUDA | ≥ 11.8 |
| VirtualGL | 3.1 (headless GPU 3D rendering) |
| GPU VRAM | ≥ 40 GB (eval); 4 × 80 GB A100 recommended for RLVR training |

---

## Installation

### Step 1 — Create Conda environment

```bash
conda create -n EALight python=3.10 -y
conda activate EALight
```

### Step 2 — Install system dependencies and Python packages

```bash
bash install_deps.sh
```

Or manually:

```bash
# System packages
sudo apt update && sudo apt install -y \
    sumo sumo-tools sumo-doc \
    xvfb x11-xserver-utils libgl1-mesa-glx wget

export SUMO_HOME=/usr/share/sumo
echo 'export SUMO_HOME=/usr/share/sumo' >> ~/.bashrc

# VirtualGL (headless GPU rendering on servers)
wget -O virtualgl.deb \
    https://sourceforge.net/projects/virtualgl/files/3.1/virtualgl_3.1_amd64.deb/download
sudo dpkg -i virtualgl.deb || sudo apt install -f -y

# Python packages
pip install -r requirements.txt
```

### Step 3 — Install TransSimHub (3D simulation environment)

TransSimHub (full source + 3D vehicle assets) is hosted on an anonymous GitHub repository.

```bash
# Clone the anonymous repository
git clone https://anonymous.4open.science/r/EMNLP-5798/ TransSimHub

# Install as a Python package
cd TransSimHub && pip install -e . && cd ..
```


### Step 4 — Prepare the model

The fine-tuned **EALight-8B** model will be released publicly upon paper acceptance. For evaluation purposes, the base **Qwen3-VL-8B-Instruct** model can be used as a drop-in replacement to verify the evaluation pipeline. Complete training hyperparameters and scripts are provided in `src/training/` for full reproducibility.

```bash
# Download base model from ModelScope (recommended for users in China)
python scripts/download_model.py

# Or from HuggingFace
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-VL-8B-Instruct', local_dir='models/qwen3-vl-8b')
"
```

After downloading, update `configs/model_config.py` (`requests.model_name`) and `src/inference/vllm_load_model.sh` (`--model` path) accordingly.

---

## Evaluation

The `data/` directory is already included in this software package — no separate download required.

### Run baseline methods (no GPU required)

```bash
bash scripts/run_batch_event_eval.sh --baseline-only
```

This runs **FixedTime** and **MaxPressure** across all included scenarios and event types.

### Run EALight evaluation

**Step 1** — Start the vLLM inference server (edit model path first):

```bash
# Edit --model path in src/inference/vllm_load_model.sh
bash src/inference/vllm_load_model.sh
```

**Step 2** — Run evaluation on event scenarios:

```bash
# Emergency vehicles + school/transit buses
bash scripts/run_batch_event_eval.sh --event emergy_bus      --port 8000 --model_name ealight-8b

# Traffic accidents + road debris / construction
bash scripts/run_batch_event_eval.sh --event accident_debris --port 8000 --model_name ealight-8b
```

**Step 3** — Collect and display results:

```bash
# Edit the scenario_dict and methods list at the bottom of metrics.py,
# then run to print per-scenario metrics to stdout:
python src/evaluation/metrics.py
```

The script prints `ATT`, `AWT`, `sp-ATT`, `sp-AWT`, `MaxQL`, and `TCR` for each
`(scenario, route_file, method)` combination.

### Dataset configuration

The batch script (`scripts/run_batch_event_eval.sh`) includes four datasets by default.
To add datasets not in this package (Hangzhou, New York), place the data under `data/raw/`
and uncomment the corresponding lines in the `NORMAL_ENTRIES` / `EVENT_ENTRIES` section.

---

## Training

> The steps below are for full training reproducibility.

### Phase 1 — Supervised Fine-Tuning (SFT)

```bash
# Edit src/training/sft_trainer.sh: set model_path, data_file, output_dir
bash src/training/sft_trainer.sh
```

### Phase 2 — RLVR Training (Online GRPO with Dual Verifiable Rewards)

```bash
# Step 1: Generate RLVR rollout data via SUMO simulation
#         (uses Syn-Train synthetic network included in data/raw/Syn-Train/)
python src/training/rlvr/rlvr_data_collector.py \
    data/rlvr_dataset/training_pool.jsonl \
    data/rlvr_dataset/rlvr_train.jsonl

# Step 2: Edit src/training/rlvr_trainer.sh: set model_path (SFT checkpoint), data_file
# Step 3: Launch training (4 × A100 80 GB recommended)
bash src/training/rlvr_trainer.sh
```

---

## Datasets

| Dataset | Intersections | Phase Space | Included |
|---------|:---:|:---:|:---:|
| JiNan (China) | 12 | 4-phase | ✓ |
| SouthKorea\_Songdo | 1 | 4-phase | ✓ |
| France\_Massy | 1 | 2-phase | ✓ |
| Hongkong\_YMT | 1 | 3-phase | ✓ |
| Syn-Train (4×4 grid) | 16 | 4-phase | ✓ (training only) |
| Hangzhou (China) | 16 | 4-phase | — |
| New York | 196 | 4-phase | — |

> **Note on Hangzhou and New York**: These two datasets involve substantially larger simulation environments. Due to file size constraints in the submission package, they are not included here. Both datasets will be made publicly available upon paper acceptance.

**Event route files** (automatically paired with each scenario):

| Suffix | Events | Key metrics |
|--------|--------|-------------|
| `*_emergy_bus.rou.xml` | Emergency vehicles + school/transit buses | sp-ATT, sp-AWT |
| `*_accident_debris.rou.xml` | Traffic accidents + road debris / construction | MaxQL, TCR |

---

## Key Metrics

| Metric | Description |
|--------|-------------|
| ATT / AWT | Average travel time / waiting time (all vehicles) |
| sp-ATT / sp-AWT | Special vehicle average travel / waiting time (emergency vehicles + buses) |
| MaxQL | Peak queue length across all lanes and timesteps |
| TCR | Trip Completion Rate (completed / total routed trips) |

---

## Key Configuration Files

| File | What to configure |
|------|-------------------|
| `configs/model_config.py` | `requests.url` and `requests.model_name` for local vLLM; or API keys for Gemini / OpenAI |
| `src/inference/vllm_load_model.sh` | `--model` path to the model checkpoint |
| `src/training/sft_trainer.sh` | `model_path`, `data_file`, `output_dir` |
| `src/training/rlvr_trainer.sh` | `model_path` (SFT checkpoint), `data_file`, `wandb_entity` |

---

## License

The EALight code is released under the **MIT License**.  
TransSimHub is released under its original license (see `TransSimHub/LICENSE`).

---
---

# 中文说明

> **[English version above ↑](#ealight-event-aware-traffic-signal-control-via-vision-language-world-knowledge)**

---

## 项目结构

```
EALight/
├── README.md
├── requirements.txt
├── install_deps.sh             # 一键环境配置
├── vgl_python.sh               # 无头服务器 VirtualGL 启动器
│
├── configs/                    # 配置文件
│   ├── env_config.py           # SUMO / TransSimHub 环境参数
│   ├── model_config.py         # VLM API 配置（本地 vLLM / Gemini / OpenAI）
│   ├── prompt_builder.py       # 自适应 CoT 提示构建（主版本）
│   ├── prompt_builder_fixed_cot.py     # 消融：强制慢路径
│   ├── prompt_builder_fixed_durtion.py # 消融：固定绿灯时长
│   ├── scenairo_config.py      # 场景拓扑与相位空间定义
│   ├── accelerate_config.yaml  # 分布式训练配置
│   ├── ds_config_lora.json     # DeepSpeed LoRA 配置
│   └── train_config.yaml
│
├── src/
│   ├── bev_generation/         # 多视角进口道图像渲染
│   ├── evaluation/
│   │   ├── run_eval.py         # 主评测脚本
│   │   ├── metrics.py          # ATT / AWT / sp-ATT / sp-AWT / MaxQL / TCR
│   │   ├── collect_metrics.py  # 批量结果汇总
│   │   └── analyze_event_recognition.py  # 事件识别准确率分析
│   ├── inference/
│   │   ├── vlm_agent.py        # VLM agent：Prompt 构建 + 动作解析
│   │   ├── vllm_load_model.sh  # 启动 vLLM 推理服务
│   │   └── vllm_load_model_lora.sh
│   ├── training/
│   │   ├── trainer.py          # SFT 训练主程序
│   │   ├── sft_trainer.sh      # SFT 启动脚本
│   │   ├── rlvr_trainer.sh     # RLVR / GRPO 启动脚本
│   │   └── rlvr/               # GRPO 训练器、奖励函数、解析器
│   └── utils/
│       ├── event_bulletin.py   # 路口间事件广播（Prompt 层）
│       ├── make_tsc_env.py
│       └── tsc_env/            # TSC 封装：动作解码、状态管理
│
├── scripts/
│   ├── run_batch_event_eval.sh # 批量评测（基线 + VLM + 所有事件类型）
│   ├── add_lane_watermarks.py  # 为路口图像添加方向水印
│   ├── visual_robustness.py    # 视觉鲁棒性扰动流水线
│   └── download_model.py       # 从 ModelScope 下载基础模型
│
├── TransSimHub/                # 3D 交通仿真环境
│   │                           # 完整源码 + 3D 资产托管于匿名 GitHub：
│   │                           #   <ANONYMOUS_TRANSSIMHUB_GITHUB_URL>
│   └── tshub/                  # 核心 Python 包（pip install -e .）
│
└── data/
    └── raw/                    # 与代码一同打包，无需单独下载
        ├── JiNan/              # 12 路口，真实交通流（中国济南）
        ├── SouthKorea_Songdo/  # 大型非对称 6 车道路口
        ├── France_Massy/       # T 字路口（2 相位）
        ├── Hongkong_YMT/       # 左行交通特例（3 相位）
        └── Syn-Train/          # 4×4 合成路网（RLVR 训练用）
```

> **TransSimHub**（含 3D 车辆资产，约 650 MB）因文件体积较大，单独托管于匿名 GitHub 仓库。
> 按下方步骤 3 克隆即可。

---

## 环境要求

| 组件 | 版本 |
|------|------|
| Python | 3.10 |
| SUMO | ≥ 1.20 |
| CUDA | ≥ 11.8 |
| VirtualGL | 3.1（无头服务器 GPU 渲染） |
| GPU 显存 | 评测 ≥ 40 GB；RLVR 训练推荐 4 × A100 80 GB |

---

## 安装步骤

### 步骤 1 — 创建 Conda 环境

```bash
conda create -n EALight python=3.10 -y
conda activate EALight
```

### 步骤 2 — 安装系统依赖与 Python 包

```bash
bash install_deps.sh
```

或手动安装：

```bash
# 系统依赖
sudo apt update && sudo apt install -y \
    sumo sumo-tools sumo-doc \
    xvfb x11-xserver-utils libgl1-mesa-glx wget

export SUMO_HOME=/usr/share/sumo
echo 'export SUMO_HOME=/usr/share/sumo' >> ~/.bashrc

# VirtualGL（无头服务器 GPU 渲染）
wget -O virtualgl.deb \
    https://sourceforge.net/projects/virtualgl/files/3.1/virtualgl_3.1_amd64.deb/download
sudo dpkg -i virtualgl.deb || sudo apt install -f -y

# Python 依赖
pip install -r requirements.txt
```

### 步骤 3 — 安装 TransSimHub（3D 仿真环境）

TransSimHub 完整版（含 3D 车辆资产）托管于匿名 GitHub 仓库。

```bash
# 克隆匿名仓库
git clone https://anonymous.4open.science/r/EMNLP-5798/ TransSimHub

# 安装为 Python 包
cd TransSimHub && pip install -e . && cd ..
```


### 步骤 4 — 准备模型

微调后的 **EALight-8B** 模型将在论文接受后公开发布。在此之前，评测人员可使用基础版 **Qwen3-VL-8B-Instruct** 模型作为替代，以验证评测流程的完整性。所有训练超参数与训练脚本均已在 `src/training/` 中提供，支持完整的训练复现。

```bash
# 从 ModelScope 下载基础模型（国内推荐）
python scripts/download_model.py

# 或从 HuggingFace 下载
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-VL-8B-Instruct', local_dir='models/qwen3-vl-8b')
"
```

下载完成后，请更新 `configs/model_config.py`（`requests.model_name`）和 `src/inference/vllm_load_model.sh`（`--model` 路径）。

---

## 评测

数据集已包含在本软件包 `data/` 目录中，**无需单独下载**。

### 运行基线方法（无需 GPU）

```bash
bash scripts/run_batch_event_eval.sh --baseline-only
```

自动在所有包含场景上运行 **FixedTime** 和 **MaxPressure** 基线（含事件路由）。

### 运行 EALight 评测

**第 1 步** — 启动 vLLM 推理服务（先编辑模型路径）：

```bash
# 编辑 src/inference/vllm_load_model.sh 中的 --model 路径
bash src/inference/vllm_load_model.sh
```

**第 2 步** — 针对事件场景运行评测：

```bash
# 紧急车辆 + 校车与公交
bash scripts/run_batch_event_eval.sh --event emergy_bus      --port 8000 --model_name ealight-8b

# 交通事故 + 路面碎片/施工占道
bash scripts/run_batch_event_eval.sh --event accident_debris --port 8000 --model_name ealight-8b
```

**第 3 步** — 汇总结果：

```bash
# 编辑 metrics.py 底部的 secnario_dict 和 methods 列表，然后运行：
python src/evaluation/metrics.py
```

脚本将每个 `（场景、路由文件、方法）` 组合的 `ATT`、`AWT`、`sp-ATT`、`sp-AWT`、`MaxQL`、`TCR` 指标直接打印至标准输出。

### 数据集配置说明

评测脚本 `scripts/run_batch_event_eval.sh` 默认激活 4 个包内数据集。
如需加入 Hangzhou、New York 等未包含的数据集，请将数据放至 `data/raw/` 对应目录，
然后取消脚本中 `NORMAL_ENTRIES` / `EVENT_ENTRIES` 相关行的注释即可。

---

## 训练

> 如需完整复现训练流程，参考以下步骤。

### 阶段 1 — 监督微调（SFT）

```bash
# 编辑 src/training/sft_trainer.sh：配置 model_path、data_file、output_dir
bash src/training/sft_trainer.sh
```

### 阶段 2 — RLVR 训练（在线 GRPO + 双路可验证奖励）

```bash
# 第 1 步：通过 SUMO 仿真生成 RLVR rollout 数据
#          使用 data/raw/Syn-Train/ 中的合成路网
python src/training/rlvr/rlvr_data_collector.py \
    data/rlvr_dataset/training_pool.jsonl \
    data/rlvr_dataset/rlvr_train.jsonl

# 第 2 步：编辑 src/training/rlvr_trainer.sh：配置 model_path（SFT checkpoint）、data_file
# 第 3 步：启动训练（推荐 4 × A100 80 GB）
bash src/training/rlvr_trainer.sh
```

---

## 数据集说明

| 数据集 | 路口数 | 相位数 | 是否包含 |
|--------|:---:|:---:|:---:|
| 济南 JiNan（中国） | 12 | 4 | ✓ |
| 松岛 SouthKorea\_Songdo | 1 | 4 | ✓ |
| 马西 France\_Massy | 1 | 2 | ✓ |
| 油麻地 Hongkong\_YMT | 1 | 3 | ✓ |
| Syn-Train 合成路网 | 16 | 4 | ✓（仅训练） |
| 杭州 Hangzhou（中国） | 16 | 4 | — |
| 纽约 New York | 196 | 4 | — |

> **关于 Hangzhou 与 New York**：这两个数据集对应规模较大的仿真场景，文件体积超出本次提交包的限制，暂未随代码一同打包。论文接受后，两个数据集将随完整代码一并公开发布。

**事件路由文件**（自动与各场景配对）：

| 文件后缀 | 注入事件类型 | 核心指标 |
|----------|------------|---------|
| `*_emergy_bus.rou.xml` | 紧急车辆（消防/救护/警车）+ 校车与公交 | sp-ATT、sp-AWT |
| `*_accident_debris.rou.xml` | 交通事故（碰撞/异常停驻）+ 路面碎片/施工占道 | MaxQL、TCR |

---

## 核心指标说明

| 指标 | 说明 |
|------|------|
| ATT / AWT | 全体车辆平均行程时间 / 等待时间 |
| sp-ATT / sp-AWT | 特殊车辆平均行程时间 / 等待时间（紧急车辆 + 公交/校车合并） |
| MaxQL | 峰值排队长度（所有车道所有时刻的最大值） |
| TCR | 行程完成率（完成行程数 / 路由总行程数） |

---

## 关键配置项

| 文件 | 需配置内容 |
|------|-----------|
| `configs/model_config.py` | 本地 vLLM 的 `requests.url` 和 `requests.model_name`；或 Gemini / OpenAI API Key |
| `src/inference/vllm_load_model.sh` | `--model` 指向模型 checkpoint 路径 |
| `src/training/sft_trainer.sh` | `model_path`、`data_file`、`output_dir` |
| `src/training/rlvr_trainer.sh` | `model_path`（SFT checkpoint）、`data_file`、`wandb_entity` |

---

## 开源协议

EALight 代码以 **MIT License** 开源发布。  
TransSimHub 依据其原始协议发布（详见 `TransSimHub/LICENSE`）。
