import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from argparse import ArgumentParser
from datasets import concatenate_datasets, Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments,
    Seq2SeqTrainer, DataCollatorForSeq2Seq, EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType


DEFAULT_MODEL_PATH = "./models/mt5-base"
FALLBACK_MODEL_NAME = "google/mt5-base"
OUTPUT_DIR = "./mt5_detox_lora"

LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.1
LORA_TARGET_MODULES = ["q", "v"]

CONTRASTIVE_WEIGHT, TEMPERATURE, PROJECTION_DIM = 0.1, 0.05, 256

BATCH_SIZE, GRAD_ACCUM_STEPS = 8, 4
LEARNING_RATE, NUM_EPOCHS, WEIGHT_DECAY = 3e-4, 15, 0.01
MAX_LENGTH = 256
SAVE_STEPS, EVAL_STEPS, LOGGING_STEPS = 500, 500, 100


class ProjectionHead(nn.Module):
    def __init__(self, hidden_dim=768, proj_dim=256, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, proj_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


def mean_pool(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def nt_xent_loss(z1, z2, temperature=0.05):
    B = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    sim = torch.mm(z, z.t()) / temperature
    labels = torch.cat([torch.arange(B) + B, torch.arange(B)]).to(z1.device)
    sim.masked_fill_(torch.eye(2*B, device=z1.device, dtype=torch.bool), float("-inf"))
    return F.cross_entropy(sim, labels)


class ContrastiveSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, proj_head=None, contrastive_weight=0.0, temperature=0.05, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proj_head = proj_head
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature

    @property
    def _use_contrastive(self):
        return self.model.training and self.proj_head is not None and self.contrastive_weight > 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                        labels=inputs.get("labels"), return_dict=True)
        ce_loss = outputs.loss
        if not self._use_contrastive:
            return (ce_loss, outputs) if return_outputs else ce_loss

        h1 = outputs.encoder_last_hidden_state
        h2 = model.get_encoder()(input_ids=inputs["input_ids"],
                                  attention_mask=inputs["attention_mask"],
                                  return_dict=True).last_hidden_state
        z1 = self.proj_head(mean_pool(h1, inputs["attention_mask"]))
        z2 = self.proj_head(mean_pool(h2, inputs["attention_mask"]))
        total = ce_loss + self.contrastive_weight * nt_xent_loss(z1, z2, self.temperature)
        return (total, outputs) if return_outputs else total


import pandas as pd
DATA_DIR = "./data"


def load_parallel_data(languages=None):
    data_dir = Path(DATA_DIR)
    all_data = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        lang = csv_path.stem
        if lang == "all_languages":
            continue
        if languages is not None and lang not in languages:
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            all_data.append({"lang": lang, "toxic": row["toxic_sentence"],
                             "neutral": row["neutral_sentence"]})
    full_ds = Dataset.from_list(all_data)
    from datasets import ClassLabel
    full_ds = full_ds.cast_column("lang", ClassLabel(names=sorted(set(full_ds["lang"]))))
    print(f"Loaded {len(full_ds)} samples from {len(set(d['lang'] for d in all_data))} languages")
    return full_ds.train_test_split(test_size=0.2, seed=42, stratify_by_column="lang")


def preprocess_function(examples, tokenizer):
    model_inputs = tokenizer([f"detoxify: {t}" for t in examples["toxic"]],
                             max_length=MAX_LENGTH, truncation=True, padding=False)
    labels = tokenizer(text_target=[n for n in examples["neutral"]],
                       max_length=MAX_LENGTH, truncation=True, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def load_augmented_data(aug_path, tokenizer):
    if not os.path.exists(aug_path):
        return None
    with open(aug_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    valid = [{"lang": d.get("lang", "en"), "toxic": d["toxic"], "neutral": d["neutral"]}
             for d in data if d.get("toxic") and d.get("neutral")]
    if not valid:
        return None
    print(f"Loaded {len(valid)} augmented samples")
    ds = Dataset.from_list(valid)
    return ds.map(lambda x: preprocess_function(x, tokenizer), batched=True,
                  remove_columns=ds.column_names)


def compute_metrics(tokenizer):
    try:
        import evaluate
        bleu = evaluate.load("sacrebleu")
    except Exception:
        return lambda _: {}
    def _compute(eval_preds):
        preds, labels = eval_preds
        preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        d_preds = [p.strip() for p in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        d_labels = [[l.strip()] for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        return {"bleu": round(bleu.compute(predictions=d_preds, references=d_labels)["score"], 2)}
    return _compute


def train(args=None):
    parser = ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--lora_r", type=int, default=LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--contrastive_weight", type=float, default=CONTRASTIVE_WEIGHT)
    parser.add_argument("--contrastive_temp", type=float, default=TEMPERATURE)
    parser.add_argument("--proj_dim", type=int, default=PROJECTION_DIM)
    parser.add_argument("--augmented_data", default=None)
    args = parser.parse_args(args) if args is not None else parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    split_ds = load_parallel_data(args.languages)

    model_path = args.model if Path(args.model).exists() else FALLBACK_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if args.fp16 and device == "cuda" else torch.float32,
        use_safetensors=True)

    lora_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
                             lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGET_MODULES)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = split_ds["train"].map(lambda x: preprocess_function(x, tokenizer),
                                          batched=True, remove_columns=split_ds["train"].column_names)
    val_dataset = split_ds["test"].map(lambda x: preprocess_function(x, tokenizer),
                                       batched=True, remove_columns=split_ds["test"].column_names)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output, eval_strategy="steps", save_strategy="steps",
        eval_steps=EVAL_STEPS, save_steps=SAVE_STEPS, logging_steps=LOGGING_STEPS,
        learning_rate=args.lr, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size, gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=args.epochs, weight_decay=WEIGHT_DECAY, warmup_steps=100,
        fp16=args.fp16 and device == "cuda", predict_with_generate=True,
        generation_max_length=MAX_LENGTH, generation_num_beams=4,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        save_total_limit=3, report_to="none", remove_unused_columns=False,
        dataloader_num_workers=0, max_grad_norm=1.0)

    use_contrastive = args.contrastive_weight > 0
    proj_head = None
    if use_contrastive:
        print(f"SimCSE: weight={args.contrastive_weight}, temp={args.contrastive_temp}, dim={args.proj_dim}")
        proj_head = ProjectionHead(hidden_dim=model.config.d_model, proj_dim=args.proj_dim,
                                   dropout=LORA_DROPOUT).to(device)
        if args.fp16 and device == "cuda":
            proj_head = proj_head.half()
        model.add_module("proj_head", proj_head)
        if args.augmented_data:
            aug = load_augmented_data(args.augmented_data, tokenizer)
            if aug is not None:
                train_dataset = concatenate_datasets([train_dataset, aug])

    trainer = ContrastiveSeq2SeqTrainer(
        proj_head=proj_head, contrastive_weight=args.contrastive_weight,
        temperature=args.contrastive_temp,
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True),
        compute_metrics=compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)])

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | "
          f"B={args.batch_size}×{GRAD_ACCUM_STEPS} | LR={args.lr} | Epochs={args.epochs} | LoRA r/a={args.lora_r}/{args.lora_alpha}")
    trainer.train(resume_from_checkpoint=args.resume) if args.resume else trainer.train()

    final_dir = os.path.join(args.output, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    if proj_head is not None:
        torch.save(proj_head.state_dict(), os.path.join(final_dir, "proj_head.pt"))
    langs_used = args.languages or sorted({p.stem for p in Path(DATA_DIR).glob("*.csv") if p.stem != "all_languages"})
    config = {"base_model": args.model, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
              "languages": langs_used, "contrastive": use_contrastive}
    with open(os.path.join(final_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("Quick test:")
    model.eval().to(device)
    for sample in split_ds["test"].select(range(min(3, len(split_ds["test"])))):
        inputs = tokenizer(f"detoxify: {sample['toxic']}", return_tensors="pt",
                           truncation=True, max_length=MAX_LENGTH).to(device)
        with torch.no_grad():
            pred = tokenizer.decode(model.generate(**inputs, max_new_tokens=MAX_LENGTH,
                                     num_beams=5, early_stopping=True)[0], skip_special_tokens=True)
        print(f"[{sample['lang']}] {sample['toxic'][:60]} → {pred[:60]}")


if __name__ == "__main__":
    train()
