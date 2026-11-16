# EALight: A Vision-Grounded LVLM Controller for Event-Aware Traffic Signal Control

> **[中文说明见下方 / Chinese version below ↓](#中文说明)**

---

## 1. Project Structure

```
EALight/
├── README.md
├── requirements.txt
├── install_deps.sh             # System and Python dependency installation
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
│   ├── vision_only_baseline/  # Vision-only model, training and evaluation scripts
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
│   ├── event_scene_generation/ # Event route generation / 事件路由生成
│   ├── run_batch_event_eval.sh # Batch evaluation (baselines + VLM + all event types)
│   ├── add_lane_watermarks.py  # Add directional labels to intersection images
│   ├── visual_robustness.py    # Visual perturbation pipeline (occlusion / noise / lowres)
│   └── download_model.py       # Download base LVLM from ModelScope
│
├── TransSimHub/                # 3D traffic simulation environment
│   └── tshub/                  # Core Python package (pip install -e .)
│
└── data/
    └── raw/                    # Download map/scenario assets here (Step 4)
        ├── JiNan/              # 12-intersection, real traffic (China)
        ├── SouthKorea_Songdo/  # Large asymmetric 6-lane junction
        ├── France_Massy/       # T-junction (2-phase)
        ├── Hongkong_YMT/       # Left-hand traffic (3-phase)
        └── Syn-Train/          # 4×4 synthetic grid (for RLVR training rollouts)
```

> Install the project-specific TransSimHub fork in Step 3, then prepare map/scenario and 3D vehicle assets in Step 4.

---

## 2. Installation

**Requirements**

| Component | Version |
|-----------|---------|
| Python | 3.10 |
| SUMO | ≥ 1.20 |
| CUDA | ≥ 11.8 |
| VirtualGL | 3.1 (headless GPU 3D rendering) |
| GPU VRAM | ≥ 30 GB (evaluation) |

### 2.1 Step 1 — Create Conda environment

```bash
conda create -n EALight python=3.10 -y
conda activate EALight
```

### 2.2 Step 2 — Install system dependencies and Python packages

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

### 2.3 Step 3 — Install TransSimHub (3D simulation environment)

Install the project's [TransSimHub fork](https://github.com/Tmxjj/TransSimHub) in the same Python environment. This repository is **forked from [Traffic-Alpha/TransSimHub](https://github.com/Traffic-Alpha/TransSimHub)** and includes project-specific extensions:

```bash
git clone https://github.com/Tmxjj/TransSimHub.git /path/to/TransSimHub
python -m pip install -e '/path/to/TransSimHub[all]'
export SUMO_HOME=/usr/share/sumo  # Adjust to your SUMO installation.
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

Run these commands from the EALight repository root, replacing `/path/to/TransSimHub` with your local destination. If you use the existing `TransSimHub/` directory, ensure it contains the required fork before installing it.

[`install_deps.sh`](./install_deps.sh) provides system and Python dependency installation commands. [`vgl_python.sh`](./vgl_python.sh) uses the active environment's Python by default; override `PYTHON` or `VGLRUN` if needed. The single-scenario example below invokes Xvfb/VirtualGL directly with the active Python environment.

### 2.4 Step 4 — Prepare simulation and 3D assets

Download and extract the [map/scenario assets](https://drive.google.com/drive/folders/1oTIScFrKsmmwSaG6bGCBmkSHD3fWdq86?usp=sharing) into `data/raw/`. Extract the [vehicle models](https://github.com/Traffic-Alpha/TransSimHub/releases/download/v5/vehicles_models.zip) into `tshub/tshub_env3d/_assets_3d/` inside the installed TransSimHub checkout (for example, `/path/to/TransSimHub/tshub/tshub_env3d/_assets_3d/`).

Check that each scenario has the files required by the evaluator, for example:

```text
data/raw/JiNan/
├── jinan.sumocfg
├── env/                        # Network and route files referenced by SUMO
├── add/
│   ├── e2.add.xml
│   └── tls_programs.add.xml
└── 3d_assets/
```

[`scripts/event_scene_generation/`](./scripts/event_scene_generation) provides utilities for emergency vehicles, buses, accidents, road debris, pedestrians, and mixed events. The corresponding event routes are already included in `data/` and can be used directly for evaluation. To change the random seed or event proportions, edit parameters such as `SEED`, `EMERGENCY_RATIO`, `BUS_RATIO`, `ACCIDENT_RATE`, and `DEBRIS_RATE` in the batch generator, then regenerate the mixed event routes:

```bash
bash scripts/event_scene_generation/batch_generate_mixed_events.sh
```


**Datasets**

| Dataset | Intersections | Phase Space | Asset source |
|---------|:---:|:---:|:---:|
| JiNan (China) | 12 | 4-phase | Download |
| SouthKorea\_Songdo | 1 | 4-phase | Download |
| France\_Massy | 1 | 2-phase | Download |
| Hongkong\_YMT | 1 | 3-phase | Download |
| Syn-Train (4×4 grid) | 16 | 4-phase | Download (training only) |
| Hangzhou (China) | 16 | 4-phase | Download |
| New York | 196 | 4-phase | Download |

**Event route files** (automatically paired with each scenario):

| Suffix | Events | Key metrics |
|--------|--------|-------------|
| `*_emergy_bus.rou.xml` | Emergency vehicles + school/transit buses | sp-ATT, sp-AWT |
| `*_accident_debris.rou.xml` | Traffic accidents + road debris / construction | MaxQL, TCR |

### 2.5 Step 5 — Prepare the model

This repository provides EALight and vision-only training/evaluation scripts. Obtain the base **Qwen3-VL-8B-Instruct** model independently and train locally. The base model can validate the evaluation pipeline, but its results do not reproduce a trained EALight checkpoint.

```bash
# Download base model from ModelScope (recommended for users in China)
python scripts/download_model.py

# Or from HuggingFace
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-VL-8B-Instruct', local_dir='models/Qwen3-VL-8B-Instruct')
"
```

After downloading, update `configs/model_config.py` (`requests.model_name`) and `src/inference/vllm_load_model.sh` (`--model` path) accordingly.

---

## 3. Evaluation

Complete Step 4 first to prepare the simulation and 3D assets. The event routes included in `data/` can be used directly for the examples.

### 3.1 Run baseline methods (no GPU required)

```bash
bash scripts/run_batch_event_eval.sh --baseline-only
```

This runs **FixedTime** and **MaxPressure** across all enabled scenarios and event types.

### 3.2 Run EALight evaluation

**Step 1** — Start the vLLM inference server (edit model path first):

```bash
# Edit --model path in src/inference/vllm_load_model.sh
bash src/inference/vllm_load_model.sh
```

**Step 2** — Run evaluation on event scenarios:

```bash
# Emergency vehicles + school/transit buses
bash scripts/run_batch_event_eval.sh --event emergy_bus      --port 8000 --model_name qwen3-vl-8b --result_name ealight

# Traffic accidents + road debris / construction
bash scripts/run_batch_event_eval.sh --event accident_debris --port 8000 --model_name qwen3-vl-8b --result_name ealight
```

To evaluate one prepared scenario directly (adjust the VirtualGL path if needed):

```bash
xvfb-run -a /opt/VirtualGL/bin/vglrun -d egl python src/evaluation/run_eval.py \
  --scenario JiNan \
  --route_file anon_3_4_jinan_real_emergy_bus.rou.xml \
  --scene_type emergy_bus \
  --api_url http://localhost:8000/v1/chat/completions \
  --model_name qwen3-vl-8b \
  --result_name ealight \
  --max_sumo_seconds 3600 \
  --sumo_seed 42
```

**Step 3** — Collect and display results:

```bash
# Edit the scenario_dict and methods list at the bottom of metrics.py,
# then run to print per-scenario metrics to stdout:
python src/evaluation/metrics.py
```

The script prints `ATT`, `AWT`, `sp-ATT`, `sp-AWT`, `MaxQL`, and `TCR` for each
`(scenario, route_file, method)` combination.

### 3.3 Dataset configuration

The batch script (`scripts/run_batch_event_eval.sh`) enables four scenarios by default.
To evaluate Hangzhou or New York, download the corresponding assets into `data/raw/`, check that the required event routes are present,
and uncomment the corresponding lines in the `EVENT_ENTRIES` section.

---

## 4. Training

### 4.1 Phase 1 — Supervised Fine-Tuning (SFT)

```bash
# Override MODEL_PATH, DATA_PATH and OUTPUT_DIR as needed
bash src/training/sft_trainer.sh
```

### 4.2 Phase 2 — RLVR Training (GRPO with Cached Rollout Rewards)

```bash
# Step 1: Convert raw samples that already contain cached SUMO rollout rewards
python src/training/rlvr/rlvr_data_collector.py \
    data/rlvr_dataset/training_pool.jsonl \
    data/rlvr_dataset/rlvr_train.jsonl

# Step 2: Set MODEL_PATH to your SFT checkpoint and DATA_PATH to the converted data
# Step 3: Launch training (4 × A100 80 GB recommended)
bash src/training/rlvr_trainer.sh
```

---

## 5. Key Metrics

| Metric | Description |
|--------|-------------|
| ATT / AWT | Average travel time / waiting time (all vehicles) |
| sp-ATT / sp-AWT | Special vehicle average travel / waiting time (emergency vehicles + buses) |
| MaxQL | Peak queue length across all lanes and timesteps |
| TCR | Trip Completion Rate (completed / total routed trips) |

---

## 6. Key Configuration Files

| File | What to configure |
|------|-------------------|
| `configs/model_config.py` | `requests.url` and `requests.model_name` for local vLLM; or API keys for Gemini / OpenAI |
| `src/inference/vllm_load_model.sh` | `--model` path to the model checkpoint |
| `src/training/sft_trainer.sh` | `MODEL_PATH`, `DATA_PATH`, `OUTPUT_DIR` |
| `src/training/rlvr_trainer.sh` | `MODEL_PATH` (SFT checkpoint), `DATA_PATH`, `REPORT_TO` |

---

## 7. License

The EALight code is released under the **MIT License**.  
TransSimHub is released under its original license (see `TransSimHub/LICENSE`).

---
---

# 中文说明

> **[English version above ↑](#ealight-a-vision-grounded-lvlm-controller-for-event-aware-traffic-signal-control)**

---

## 1. 项目结构

```
EALight/
├── README.md
├── requirements.txt
├── install_deps.sh             # 系统与 Python 依赖安装
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
│   ├── vision_only_baseline/  # Vision-only model, training and evaluation scripts
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
│   ├── event_scene_generation/ # Event route generation / 事件路由生成
│   ├── run_batch_event_eval.sh # 批量评测（基线 + VLM + 所有事件类型）
│   ├── add_lane_watermarks.py  # 为路口图像添加方向水印
│   ├── visual_robustness.py    # 视觉鲁棒性扰动流水线
│   └── download_model.py       # 从 ModelScope 下载基础模型
│
├── TransSimHub/                # 3D 交通仿真环境
│   └── tshub/                  # 核心 Python 包（pip install -e .）
│
└── data/
    └── raw/                    # 下载地图/场景资源至此（步骤 4）
        ├── JiNan/              # 12 路口，真实交通流（中国济南）
        ├── SouthKorea_Songdo/  # 大型非对称 6 车道路口
        ├── France_Massy/       # T 字路口（2 相位）
        ├── Hongkong_YMT/       # 左行交通特例（3 相位）
        └── Syn-Train/          # 4×4 合成路网（RLVR 训练用）
```

> 按步骤 3 安装本项目使用的 TransSimHub 分支，再按步骤 4 准备地图/场景及 3D 车辆资源。

---

## 2. 安装步骤

**环境要求**

| 组件 | 版本 |
|------|------|
| Python | 3.10 |
| SUMO | ≥ 1.20 |
| CUDA | ≥ 11.8 |
| VirtualGL | 3.1（无头服务器 GPU 渲染） |
| GPU 显存 | 评测 ≥ 30 GB |

### 2.1 步骤 1 — 创建 Conda 环境

```bash
conda create -n EALight python=3.10 -y
conda activate EALight
```

### 2.2 步骤 2 — 安装系统依赖与 Python 包

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

### 2.3 步骤 3 — 安装 TransSimHub（3D 仿真环境）

在同一 Python 环境中安装本项目使用的 [TransSimHub fork](https://github.com/Tmxjj/TransSimHub)。该仓库 **fork 自 [Traffic-Alpha/TransSimHub](https://github.com/Traffic-Alpha/TransSimHub)**，并在此基础上扩展：

```bash
git clone https://github.com/Tmxjj/TransSimHub.git /path/to/TransSimHub
python -m pip install -e '/path/to/TransSimHub[all]'
export SUMO_HOME=/usr/share/sumo  # 按实际 SUMO 安装位置调整。
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

请在 EALight 项目根目录执行上述命令，并将 `/path/to/TransSimHub` 替换为本机目标路径。如果使用已有的 `TransSimHub/` 目录，请先确认其中包含所需分支，再安装。

[`install_deps.sh`](./install_deps.sh) 提供系统依赖与 Python 包安装命令。[`vgl_python.sh`](./vgl_python.sh) 默认使用当前环境的 Python，可按需通过 `PYTHON` 和 `VGLRUN` 覆盖路径。下方单场景示例直接调用 Xvfb/VirtualGL，并使用当前环境中的 Python。

### 2.4 步骤 4 — 准备仿真与 3D 资源

将[地图/场景资源](https://drive.google.com/drive/folders/1oTIScFrKsmmwSaG6bGCBmkSHD3fWdq86?usp=sharing)下载并解压至 `data/raw/`，将[车辆模型](https://github.com/Traffic-Alpha/TransSimHub/releases/download/v5/vehicles_models.zip)解压至实际安装的 TransSimHub 目录下的 `tshub/tshub_env3d/_assets_3d/`（例如 `/path/to/TransSimHub/tshub/tshub_env3d/_assets_3d/`）。

核对各场景是否具备评估入口所需的文件，例如：

```text
data/raw/JiNan/
├── jinan.sumocfg
├── env/                        # SUMO 配置引用的路网与路由文件
├── add/
│   ├── e2.add.xml
│   └── tls_programs.add.xml
└── 3d_assets/
```

[`scripts/event_scene_generation/`](./scripts/event_scene_generation) 提供紧急车辆、公交、事故、道路障碍、行人及混合事件生成工具。`data/` 中已包含对应的事件路由，可直接用于评测。如需修改随机种子、事件比例或发生频率，请调整批量生成脚本中的 `SEED`、`EMERGENCY_RATIO`、`BUS_RATIO`、`ACCIDENT_RATE`、`DEBRIS_RATE` 等参数，再运行以下命令重新生成混合事件路由：

```bash
bash scripts/event_scene_generation/batch_generate_mixed_events.sh
```


**数据集说明**

| 数据集 | 路口数 | 相位数 | 资源来源 |
|--------|:---:|:---:|:---:|
| 济南 JiNan（中国） | 12 | 4 | 下载 |
| 松岛 SouthKorea\_Songdo | 1 | 4 | 下载 |
| 马西 France\_Massy | 1 | 2 | 下载 |
| 油麻地 Hongkong\_YMT | 1 | 3 | 下载 |
| Syn-Train 合成路网 | 16 | 4 | 下载（仅训练） |
| 杭州 Hangzhou（中国） | 16 | 4 | 下载 |
| 纽约 New York | 196 | 4 | 下载 |

**事件路由文件**（自动与各场景配对）：

| 文件后缀 | 注入事件类型 | 核心指标 |
|----------|------------|---------|
| `*_emergy_bus.rou.xml` | 紧急车辆（消防/救护/警车）+ 校车与公交 | sp-ATT、sp-AWT |
| `*_accident_debris.rou.xml` | 交通事故（碰撞/异常停驻）+ 路面碎片/施工占道 | MaxQL、TCR |

### 2.5 步骤 5 — 准备模型

仓库提供 EALight 和 vision-only 的训练、评测代码及启动脚本，请自行获取 **Qwen3-VL-8B-Instruct** 基础模型并训练。基础模型可验证评测流程，但其结果不能代表已训练 EALight 的性能。

```bash
# 从 ModelScope 下载基础模型（国内推荐）
python scripts/download_model.py

# 或从 HuggingFace 下载
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-VL-8B-Instruct', local_dir='models/Qwen3-VL-8B-Instruct')
"
```

下载完成后，请更新 `configs/model_config.py`（`requests.model_name`）和 `src/inference/vllm_load_model.sh`（`--model` 路径）。

---

## 3. 评测

请先按步骤 4 准备仿真与 3D 资源。示例可直接使用 `data/` 中已有的事件路由。

### 3.1 运行基线方法（无需 GPU）

```bash
bash scripts/run_batch_event_eval.sh --baseline-only
```

自动在所有启用场景上运行 **FixedTime** 和 **MaxPressure** 基线（含事件路由）。

### 3.2 运行 EALight 评测

**第 1 步** — 启动 vLLM 推理服务（先编辑模型路径）：

```bash
# 编辑 src/inference/vllm_load_model.sh 中的 --model 路径
bash src/inference/vllm_load_model.sh
```

**第 2 步** — 针对事件场景运行评测：

```bash
# 紧急车辆 + 校车与公交
bash scripts/run_batch_event_eval.sh --event emergy_bus      --port 8000 --model_name qwen3-vl-8b --result_name ealight

# 交通事故 + 路面碎片/施工占道
bash scripts/run_batch_event_eval.sh --event accident_debris --port 8000 --model_name qwen3-vl-8b --result_name ealight
```

也可直接评测单个已准备好的场景（按实际安装位置调整 VirtualGL 路径）：

```bash
xvfb-run -a /opt/VirtualGL/bin/vglrun -d egl python src/evaluation/run_eval.py \
  --scenario JiNan \
  --route_file anon_3_4_jinan_real_emergy_bus.rou.xml \
  --scene_type emergy_bus \
  --api_url http://localhost:8000/v1/chat/completions \
  --model_name qwen3-vl-8b \
  --result_name ealight \
  --max_sumo_seconds 3600 \
  --sumo_seed 42
```

**第 3 步** — 汇总结果：

```bash
# 编辑 metrics.py 底部的 secnario_dict 和 methods 列表，然后运行：
python src/evaluation/metrics.py
```

脚本将每个 `（场景、路由文件、方法）` 组合的 `ATT`、`AWT`、`sp-ATT`、`sp-AWT`、`MaxQL`、`TCR` 指标直接打印至标准输出。

### 3.3 数据集配置说明

评测脚本 `scripts/run_batch_event_eval.sh` 默认激活 4 个场景。
如需评测 Hangzhou、New York，请下载数据至 `data/raw/` 对应目录并确认所需事件路由已就绪，
然后取消脚本中 `EVENT_ENTRIES` 相关行的注释。

---

## 4. 训练

### 4.1 阶段 1 — 监督微调（SFT）

```bash
# 按需通过环境变量配置 MODEL_PATH、DATA_PATH、OUTPUT_DIR
bash src/training/sft_trainer.sh
```

### 4.2 阶段 2 — RLVR 训练（GRPO + 缓存 rollout 奖励）

```bash
# 第 1 步：转换已包含 SUMO rollout 缓存奖励的原始样本
python src/training/rlvr/rlvr_data_collector.py \
    data/rlvr_dataset/training_pool.jsonl \
    data/rlvr_dataset/rlvr_train.jsonl

# 第 2 步：通过 MODEL_PATH 指定自己的 SFT checkpoint，通过 DATA_PATH 指定数据
# 第 3 步：启动训练（推荐 4 × A100 80 GB）
bash src/training/rlvr_trainer.sh
```

---

## 5. 核心指标说明

| 指标 | 说明 |
|------|------|
| ATT / AWT | 全体车辆平均行程时间 / 等待时间 |
| sp-ATT / sp-AWT | 特殊车辆平均行程时间 / 等待时间（紧急车辆 + 公交/校车合并） |
| MaxQL | 峰值排队长度（所有车道所有时刻的最大值） |
| TCR | 行程完成率（完成行程数 / 路由总行程数） |

---

## 6. 关键配置项

| 文件 | 需配置内容 |
|------|-----------|
| `configs/model_config.py` | 本地 vLLM 的 `requests.url` 和 `requests.model_name`；或 Gemini / OpenAI API Key |
| `src/inference/vllm_load_model.sh` | `--model` 指向模型 checkpoint 路径 |
| `src/training/sft_trainer.sh` | `MODEL_PATH`、`DATA_PATH`、`OUTPUT_DIR` |
| `src/training/rlvr_trainer.sh` | `MODEL_PATH`（SFT checkpoint）、`DATA_PATH`、`REPORT_TO` |

---

## 7. 开源协议

EALight 代码以 **MIT License** 开源发布。  
TransSimHub 依据其原始协议发布（详见 `TransSimHub/LICENSE`）。
