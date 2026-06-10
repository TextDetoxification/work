"""
mT5 + LoRA 多语言文本去毒化训练脚本
Component 1: Seq2Seq Fine-tuning

模型: google/mt5-base (580M)
数据: 9 语 × 400 对 textdetox/multilingual_paradetox
策略: LoRA (r=16, alpha=32)
"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中（兼容 python src/train_mt5.py 和 python -m src.train_mt5）
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
import torch
import numpy as np
from typing import Dict, List
from argparse import ArgumentParser

from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from src.contrastive_trainer import (
    ContrastiveSeq2SeqTrainer,
    ProjectionHead,
)


# ── 配置 ──────────────────────────────────────────────────────
# 本地模型路径（优先）或 HuggingFace Hub 名称
DEFAULT_MODEL_PATH = "./models/mt5-base"       # download_models.py 下载的目标
FALLBACK_MODEL_NAME = "google/mt5-base"        # 本地不存在时自动从 HF Hub 加载
OUTPUT_DIR = "./mt5_detox_lora"
LOGGING_DIR = "./logs"

# LoRA 配置
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q", "v"]  # mT5 的 attention 线性层

# 对比学习配置
CONTRASTIVE_WEIGHT = 0.1         # λ: 对比损失权重 (0 = 禁用)
CONTRASTIVE_TEMPERATURE = 0.05   # NT-Xent 温度系数
PROJECTION_DIM = 256             # 投影头输出维度
AUGMENTED_DATA_PATH = "./data/augmented.json"  # 增强数据默认路径

# 训练配置
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4  # 有效 batch = 8 * 4 = 32
LEARNING_RATE = 3e-4
NUM_EPOCHS = 10
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 256
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
FP16 = False  # 混合精度（小数据集 + 多语言容易数值溢出，建议关闭）
SAVE_STEPS = 100
EVAL_STEPS = 100
LOGGING_STEPS = 20


def load_parallel_data(languages: List[str] = None) -> DatasetDict:
    """
    加载多语言平行去毒数据。

    Args:
        languages: 要加载的语言列表，None=全部9语

    Returns:
        DatasetDict with train/val splits
    """
    print("Loading parallel detox data...")
    ds = load_dataset("textdetox/multilingual_paradetox")

    if languages is not None:
        ds = DatasetDict({k: v for k, v in ds.items() if k in languages})

    print(f"Loaded {len(ds)} languages: {list(ds.keys())}")

    # 转换为统一格式
    all_data = []
    for lang in ds:
        for row in ds[lang]:
            all_data.append({
                "lang": lang,
                "toxic": row["toxic_sentence"],
                "neutral": row["neutral_sentence"],
            })

    full_ds = Dataset.from_list(all_data)
    # 将 lang 转为 ClassLabel 以支持分层抽样
    langs = sorted(set(full_ds["lang"]))
    from datasets import ClassLabel
    full_ds = full_ds.cast_column("lang", ClassLabel(names=langs))
    print(f"Total samples: {len(full_ds)}")

    # 按语言分层切分 80/20
    split_ds = full_ds.train_test_split(test_size=0.2, seed=42, stratify_by_column="lang")
    print(f"Train: {len(split_ds['train'])}, Val: {len(split_ds['test'])}")

    return split_ds


def load_augmented_data(aug_path: str, tokenizer):
    """
    加载增强数据 JSON 文件并 tokenize。

    Args:
        aug_path: 增强数据 JSON 文件路径
        tokenizer: tokenizer 实例

    Returns:
        Dataset 或 None（文件不存在时）
    """
    if not os.path.exists(aug_path):
        print(f"Warning: augmented data not found at {aug_path}, skipping.")
        return None

    print(f"Loading augmented data from {aug_path}")
    with open(aug_path, "r", encoding="utf-8") as f:
        aug_data = json.load(f)

    print(f"  Loaded {len(aug_data)} augmented samples")

    # 过滤有效数据（必须包含 toxic 和 neutral 字段）
    valid_data = []
    for item in aug_data:
        if "toxic" in item and "neutral" in item and item["toxic"].strip() and item["neutral"].strip():
            valid_data.append({
                "lang": item.get("lang", "en"),
                "toxic": item["toxic"],
                "neutral": item["neutral"],
            })

    print(f"  Valid augmented samples: {len(valid_data)}")
    if not valid_data:
        return None

    aug_dataset = Dataset.from_list(valid_data)
    aug_dataset = aug_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=aug_dataset.column_names,
    )
    return aug_dataset


def preprocess_function(examples, tokenizer):
    """将数据格式化为 Seq2Seq 输入输出"""
    inputs = [f"detoxify: {t}" for t in examples["toxic"]]
    targets = [n for n in examples["neutral"]]

    # Tokenize 输入
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False,
    )

    # Tokenize 标签
    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(tokenizer):
    """计算验证指标（BLEU 可选，失败时仅用 eval_loss）"""
    try:
        import evaluate
        bleu = evaluate.load("sacrebleu")
    except Exception:
        return lambda _: {}

    def _compute(eval_preds):
        preds, labels = eval_preds

        # 清理负数（padding 标记），替换为 pad_token_id
        preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # Decode
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # sacreBLEU 需要 list of references
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [[l.strip()] for l in decoded_labels]

        result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": round(result["score"], 2)}

    return _compute


def train(args=None):
    """主训练函数"""

    # ── 参数解析 ──
    parser = ArgumentParser(description="Train mT5 LoRA for text detoxification")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="基座模型路径（本地目录或 HF Hub 名称）")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--languages", nargs="+", default=None, help="Languages to train on (default: all 9)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--lora_r", type=int, default=LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--fp16", action="store_true", default=FP16)
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    # 对比学习
    parser.add_argument("--contrastive_weight", type=float, default=CONTRASTIVE_WEIGHT,
                        help="对比损失权重 (0 = 禁用对比学习)")
    parser.add_argument("--contrastive_temp", type=float, default=CONTRASTIVE_TEMPERATURE,
                        help="NT-Xent 温度系数")
    parser.add_argument("--proj_dim", type=int, default=PROJECTION_DIM,
                        help="投影头输出维度")
    # 数据增强
    parser.add_argument("--augmented_data", default=None,
                        help="增强数据 JSON 路径 (与训练数据合并)")
    args = parser.parse_args(args if args else [])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"FP16: {args.fp16 and device == 'cuda'}")

    # ── 加载数据 ──
    split_ds = load_parallel_data(languages=args.languages)

    # ── 加载模型和分词器 ──
    # 优先用本地路径，不存在则自动从 HuggingFace Hub 下载
    model_path = args.model if Path(args.model).exists() else FALLBACK_MODEL_NAME
    print(f"\nLoading model: {args.model}")
    if model_path != args.model:
        print(f"  本地不存在，从 HF Hub 加载: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # mT5 没有默认 pad_token，使用 eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        dtype=torch.float16 if args.fp16 and device == "cuda" else torch.float32,
        use_safetensors=True,
    )

    # ── 配置 LoRA ──
    print(f"\nConfiguring LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 数据预处理 ──
    print("\nPreprocessing data...")
    train_dataset = split_ds["train"].map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=split_ds["train"].column_names,
    )
    val_dataset = split_ds["test"].map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=split_ds["test"].column_names,
    )

    # ── 训练参数 ──
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=args.epochs,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=100,
        fp16=args.fp16 and device == "cuda",
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        generation_num_beams=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        report_to="none",  # 不用 wandb/tensorboard
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Windows 下设为 0 避免多进程问题
        max_grad_norm=1.0,  # 梯度裁剪，防止 FP32 下也出现数值不稳定
    )

    # ── 对比学习 ──
    use_contrastive = args.contrastive_weight > 0
    proj_head = None

    if use_contrastive:
        print(f"\n对比学习已启用:")
        print(f"  Weight:        {args.contrastive_weight}")
        print(f"  Temperature:   {args.contrastive_temp}")
        print(f"  Projection dim: {args.proj_dim}")

        proj_head = ProjectionHead(
            hidden_dim=model.config.d_model,
            proj_dim=args.proj_dim,
            dropout=LORA_DROPOUT,
        ).to(device)
        if args.fp16 and device == "cuda":
            proj_head = proj_head.half()

        # 注册到 model 上，优化器自动包含投影头参数
        model.add_module("proj_head", proj_head)

        # 合并增强数据（如果提供）
        if args.augmented_data:
            aug_dataset = load_augmented_data(args.augmented_data, tokenizer)
            if aug_dataset is not None:
                train_dataset = concatenate_datasets([train_dataset, aug_dataset])
                print(f"  合并后训练集: {len(train_dataset)} 条")

    # ── Trainer ──
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    trainer = ContrastiveSeq2SeqTrainer(
        proj_head=proj_head,
        contrastive_weight=args.contrastive_weight,
        contrastive_temperature=args.contrastive_temp,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # ── 训练 ──
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch size:    {args.batch_size} x {GRAD_ACCUM_STEPS} = {args.batch_size * GRAD_ACCUM_STEPS}")
    print(f"  Learning rate: {args.lr}")
    print(f"  LoRA r/alpha:  {args.lora_r}/{args.lora_alpha}")
    if use_contrastive:
        print(f"  Contrastive:   weight={args.contrastive_weight}, temp={args.contrastive_temp}")
    print(f"{'='*60}\n")

    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    # ── 保存最终模型 ──
    final_dir = os.path.join(args.output, "final")
    print(f"\nSaving final model to {final_dir}")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # 保存投影头（单独保存，推理时不需要）
    if proj_head is not None:
        proj_head_path = os.path.join(final_dir, "proj_head.pt")
        torch.save(proj_head.state_dict(), proj_head_path)
        print(f"Projection head saved to {proj_head_path}")

    # 保存训练配置
    config = {
        "base_model": args.model,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "languages": args.languages or list(load_dataset("textdetox/multilingual_paradetox").keys()),
        "max_input_length": MAX_INPUT_LENGTH,
        "max_target_length": MAX_TARGET_LENGTH,
        "contrastive_weight": args.contrastive_weight,
        "contrastive_temperature": args.contrastive_temp if use_contrastive else None,
        "projection_dim": args.proj_dim if use_contrastive else None,
        "augmented_data": args.augmented_data,
    }
    with open(os.path.join(final_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("Training completed!")

    # ── 快速测试 ──
    print(f"\n{'='*60}")
    print("Quick test on validation set:")
    print("=" * 60)
    model.eval()
    model.to(device)

    # 抽几条验证集样本测试
    test_samples = split_ds["test"].select(range(min(3, len(split_ds["test"]))))
    for i, sample in enumerate(test_samples):
        input_text = f"detoxify: {sample['toxic']}"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LENGTH).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_TARGET_LENGTH,
                num_beams=5,
                early_stopping=True,
            )

        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n[{sample['lang']}]")
        # 用 ascii 兜底，避免 Windows GBK console 对特殊字符报错
        try:
            print(f"  Toxic:   {sample['toxic'][:100]}")
            print(f"  Neutral: {sample['neutral'][:100]}")
            print(f"  Predict: {prediction[:100]}")
        except UnicodeEncodeError:
            print(f"  Toxic:   {sample['toxic'][:100].encode('ascii', errors='replace').decode()}")
            print(f"  Neutral: {sample['neutral'][:100].encode('ascii', errors='replace').decode()}")
            print(f"  Predict: {prediction[:100].encode('ascii', errors='replace').decode()}")


if __name__ == "__main__":
    train()
