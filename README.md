# mT5 多语言文本去毒化

基于 mT5-base + LoRA + SimCSE 对比学习的多语言 toxic→neutral 文本去毒系统，参加 [Text Detoxification @ CodaBench](https://www.codabench.org/competitions/15765/)。

覆盖 15 种语言：am, ar, de, en, es, fr, he, hi, hin, it, ja, ru, tt, uk, zh。

## 环境

```bash
conda create -n detox python=3.10 -y && conda activate detox
pip install -r requirements.txt
```

**CUDA 版本匹配**（PyTorch 编译的 CUDA 版本 ≤ 驱动版本）：

```bash
# 查看驱动版本
nvidia-smi | head -5

# 根据驱动选 PyTorch:
#   驱动 12.x → pip install torch --index-url https://download.pytorch.org/whl/cu121
#   驱动 11.x → pip install torch --index-url https://download.pytorch.org/whl/cu118

# 验证
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
```

如果遇到 `module 'torch' has no attribute 'float8_e8m0fnu'` 或 `dtype / torch_dtype` 等参数错误，是 `transformers` 和 `torch` 版本不匹配。推荐组合：

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.48.0
```

## 快速开始

```bash
# 1. 环境
pip install -r requirements.txt

# 2. 下载模型 (~5GB, 一次性)
HF_ENDPOINT=https://hf-mirror.com python download_models.py --model all

# 3. 导出数据
python export_dataset.py

# 4. 预翻译跨语言数据 (~1.5小时 GPU)
python -m src.cross_lingual_data --device cuda

# 5. 训练
python -m src.train_mt5 \
  --contrastive_weight 0.1 \
  --cross_lingual_data ./data/cross_lingual_pairs.json \
  --cross_lingual_weight 0.1

# 6. 生成提交
python -m src.generate_submission --input test.tsv --output submission.zip
```

## 项目结构

```
├── download_models.py          # 下载 mT5 + NLLB 模型
├── export_dataset.py           # 导出训练数据 → CSV
├── src/
│   ├── train_mt5.py            # 训练脚本 (含对比学习)
│   ├── cross_lingual_data.py   # NLLB 跨语言预翻译
│   ├── back_translate.py       # 回译管道 (NLLB 翻译)
│   ├── generate_submission.py  # 推理 + 提交文件生成
│   ├── toxic_lexicon.py        # 有害词检测与消除
│   └── data_augment.py         # LLM 数据增强
├── data/                       # 训练数据 CSV + 跨语言 JSON
├── models/                     # 下载的预训练模型
└── mt5_detox_lora/             # 训练输出 (LoRA 权重)
```

## 训练策略

### 核心模型

**mT5-base (580M)** + **LoRA** (r=16, α=32, 目标 q/v, 仅 0.3% 参数可训)

### 损失函数

```
total_loss = CE_loss + λ₁ · Contrastive₁ + λ₂ · Contrastive₂

CE_loss:          标准 Seq2Seq 交叉熵 (toxic → neutral)
Contrastive₁:     无监督对比 —— 同一输入 × 2次 dropout → 表征一致
Contrastive₂:     跨语言对比 —— 同一 toxic 意图 × 不同语言 → 表征一致
```

> 两个对比损失的实现相同（NT-Xent），区别仅在于正样本对的来源：前者来自同一句话两次编码的 dropout 差异，后者来自 NLLB 翻译产生的跨语言版本。两者共享同一个 ProjectionHead 和 Encoder，都属于对比学习。```

### 推理管道

```
训练语言 (9种): toxic → mT5 直接去毒 → 有害词消除 → 输出
零资源语言 (6种): toxic → NLLB→英语 → mT5 去毒 → NLLB→源语言 → 有害词消除 → 输出
```

## 关键参数

| 参数                       | 默认值 | 说明                        |
| -------------------------- | ------ | --------------------------- |
| `--contrastive_weight`   | 0.1    | SimCSE 对比损失权重，0=禁用 |
| `--contrastive_temp`     | 0.05   | NT-Xent 温度系数            |
| `--cross_lingual_weight` | 0.1    | 跨语言对比损失权重          |
| `--batch_size`           | 8      | 每卡 batch                  |
| `--lr`                   | 3e-4   | 学习率                      |
| `--epochs`               | 15     | 训练轮数                    |

## 数据增强建议

项目已有三种增强方式，按推荐优先级排列：

### 1. 跨语言翻译增强 (已实现 ✅)

**原理**：同一条 toxic 句翻译到多种语言，让模型学会"骂人意图"的语言无关表征。

```bash
# 每条随机翻译到 1 种语言 (快速, ~9分钟)
python -m src.cross_lingual_data --device cuda

# 每条随机翻译到 3 种语言 (覆盖更全, ~30分钟)
python -m src.cross_lingual_data --device cuda --pairs_per 3
```

**结论**：强烈推荐，尤其对零资源语言效果明显。`--pairs_per 3` 性价比最高。

### 2. LLM 数据增强 (可扩展 🔧)

**原理**：用 LLM 生成更多 toxic→neutral 对，扩充小数据集 (每语言仅 400 条)。

```bash
python -m src.data_augment --api_key YOUR_KEY --strategy all --all_languages
```

四个策略：

| 策略              | 说明                         | 适用场景               |
| ----------------- | ---------------------------- | ---------------------- |
| `toxic_para`    | 替换有害词的同义变体         | 扩充 toxic 侧多样性    |
| `neutral_para`  | neutral 同义改写             | 让模型学会更多去毒表达 |
| `new_pairs`     | 生成全新的 toxic→neutral 对 | 补充新场景             |
| `cross_lingual` | 为零资源语言凭空生成数据     | 覆盖 fr, ja, he 等     |

```bash
# 训练时合并增强数据
python -m src.train_mt5 --augmented_data ./data/augmented.json ...
```

**结论**：效果不错但需要 API 费用。建议至少跑 `neutral_para` 策略。

### 3. 回译增强 (可扩展 🔧)

**原理**：toxic → 翻译到英语 → detox → 翻译回源语言 → 作为额外 (toxic, neutral) 对。

当前 `back_translate.py` 已实现推理管道，可轻易改为数据生成模式。每条训练样本可生成 14 种回译增强对。

### 4. 文本级增强 (可扩展 🔧)

- **EDA** (Easy Data Augmentation): 同义词替换、随机插入/删除/交换，用 NLLB 或本地词典实现
- **Span-level augment**: 对 toxic 句中的有害词片段做随机替换，用有害词表 `toxic_lexicon` 的同语言/跨语言词
- **Sentence-level augment**: 用 NLLB 做 pivot 翻译 (zh→de→zh)，引入翻译噪声

## 硬件要求

| 模式         | GPU 显存 | 训练时间         |
| ------------ | -------- | ---------------- |
| mT5 + LoRA   | ≥8GB    | ~15 分钟 (A40)   |
| 仅推理 (CPU) | 无需 GPU | ~2 秒/条         |
| NLLB 翻译    | ≥6GB    | ~9 分钟 / 3600条 |
