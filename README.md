﻿# 项目说明：K12 教育大模型 SFT + DPO/ORPO 完整微调对齐流程

本项目提供从数据准备、指令微调（SFT）、到偏好对齐（DPO/ORPO）的完整大模型训练流程，专为打造面向 K12 中文教学场景的问答与讲解模型而设计。

所有训练脚本默认以 dry-run 方式运行（不下载/不写入），在远程 GPU 环境请传入 `--dry-run false` 才会执行真实处理与训练。

## 🎯 数据集与核心输出
- **SFT 数据集**：`Mxode/Chinese-Instruct`
- **DPO 数据集**：`llamafactory/DPO-En-Zh-20k`
- **主要输出**：
  - `data_proc/sft_{train,val,test}.jsonl`（包含 `messages` 字段的对话数据）
  - `data_proc/pref_{train,val,test}.jsonl`（包含 `prompt`、`chosen`、`rejected` 字段的偏好对齐数据）

## 📁 目录结构
- `scripts/`：数据准备、训练、评估、模型导出脚本。
  - `compare_chat.py`：支持 Base vs SFT vs DPO 三个阶段模型的实时交互对比。
- `configs/`：YAML 配置文件（兼容 4090/5090 等单卡环境），包含按阶段划分的评估配置（`eval_base.yaml`, `eval_sft.yaml`, `eval_dpo.yaml`）。
- `docs/`：详细的文档与技术报告，如 `DATA_PROCESSING.md` 和 `TECH_REPORT.md`。
- `data_raw/`、`data_proc/`、`outputs/`：原始处理数据与模型输出目录（会自动生成带有时间戳的子目录以防止权重覆盖）。

## 🛠️ 环境与数据准备
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 环境初始化 (自动配置路径并设置环境变量)
source scripts/setup_env.sh --dry-run false

# 3. 执行数据处理 (过滤质量、去重、按长度与中文比例进行采样)
python -m scripts.prepare_data --config configs/data.yaml --dry-run false

```

*注：如果处于离线环境，可以将下载好的数据集 JSON 文件直接放入 `data_raw/mxode_chinese_instruct/` 或 `data_raw/dpo_en_zh_20k/` 中，脚本将优先使用本地文件。*

## 🚀 模型训练流程

训练脚本现已支持动态时间戳路径（例如 `outputs/sft/times/YYYYMMDD_HHMMSS`），多次实验不会互相覆盖权重。

### 1. 指令微调 (SFT)

```bash
accelerate launch -m scripts.train_sft --config configs/sft.yaml --dry-run false

```

* **基座模型**：默认为 `Qwen/Qwen2.5-7B-Instruct`。
* **关键配置**：支持 LoRA/QLoRA 训练，长序列 `max_seq_length: 4096`，并启用 packing 提升训练效率。

### 2. 偏好对齐 (DPO / ORPO)

完成 SFT 后，使用偏好数据集进一步对齐人类意图。脚本会自动寻址并加载 SFT 阶段最新的 Adapter。

* **DPO 训练**：
```bash
accelerate launch -m scripts.train_dpo --config configs/dpo.yaml --dry-run false

```


* **ORPO 训练**（可选）：
```bash
accelerate launch -m scripts.train_orpo --config configs/orpo.yaml --dry-run false

```



## ⚖️ 评估与对比

### 1. 自动化客观评测 (lm-eval-harness)

可以使用预设好的配置，分别对不同阶段的模型进行评估，支持动态加载并合并 LoRA 权重：

```bash
# 评估基座模型
python -m scripts.eval_harness --config configs/eval_base.yaml --dry-run false

# 评估 SFT 模型
python -m scripts.eval_harness --config configs/eval_sft.yaml --dry-run false

# 评估 DPO 对齐后模型
python -m scripts.eval_harness --config configs/eval_dpo.yaml --dry-run false

```

### 2. 交互式模型竞技场 (Model Arena)

运行以下脚本，可以在终端中输入问题，实时观测并对比 基座 (Base)、微调 (SFT) 和 对齐 (DPO) 模型的生成效果：

```bash
python -m scripts.compare_chat

```

## 📦 模型导出

将训练好的 LoRA 权重与基座模型合并，导出为完整的独立模型：

```bash
python -m scripts.export_lora \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --adapter_path outputs/align/times/<latest_timestamp> \
  --output_path outputs/merged_model

```

## ❓ 常见问题（FAQ）

* **中文文本不足**：若 `allow_english_fallback=false`，样本将被按中文池截断，可放宽 `general.min_cn_ratio`。
* **OOM 内存溢出**：尝试在 `configs/sft.yaml` 中启用 `qlora.enable: true` 开启 4-bit 量化，或降低 batch size 与序列长度。
* **无法连接 Hugging Face**：执行 `setup_env.sh` 时可传入 `--use-hf-mirror` 标志以使用国内镜像源。

