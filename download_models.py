"""
下载所需的基础模型到本地缓存，与训练/推理解耦。

模型清单:
  - google/mt5-base        (~2.5 GB)  — 去毒 Seq2Seq 基座
  - facebook/nllb-200-distilled-600M (~2.4 GB) — 回译翻译引擎

使用方法:
  python download_models.py              # 下载全部
  python download_models.py --model mt5  # 只下载 mT5
  python download_models.py --model nllb # 只下载 NLLB
"""

import os
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODELS = {
    "mt5": {
        "name": "google/mt5-base",
        "save_dir": "./models/mt5-base",
        "desc": "mT5-base (多语言 Seq2Seq 去毒基座)",
    },
    "nllb": {
        "name": "facebook/nllb-200-distilled-600M",
        "save_dir": "./models/nllb-200-distilled-600M",
        "desc": "NLLB-200-600M (回译翻译引擎)",
    },
}


def download_model(model_key: str, force: bool = False):
    """下载单个模型并保存到本地"""
    info = MODELS[model_key]
    save_dir = Path(info["save_dir"])

    if save_dir.exists() and list(save_dir.glob("*.json")) and not force:
        print(f"[{model_key}] 已存在: {save_dir}  →  跳过（用 --force 强制重下）")
        return save_dir

    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"[{model_key}] 下载 {info['desc']}")
    print(f"  源:  {info['name']}")
    print(f"  目标: {save_dir}")
    print(f"{'='*60}")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"  dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(info["name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(
        info["name"], dtype=dtype, use_safetensors=True
    )

    tokenizer.save_pretrained(str(save_dir))
    model.save_pretrained(str(save_dir))

    size_mb = sum(
        f.stat().st_size for f in save_dir.rglob("*") if f.is_file()
    ) / (1024 * 1024)
    print(f"[{model_key}] 完成! 大小: {size_mb:.0f} MB")
    return save_dir


def main():
    parser = argparse.ArgumentParser(description="下载基础模型到本地")
    parser.add_argument(
        "--model", choices=["mt5", "nllb", "all"], default="all",
        help="要下载的模型 (默认: all)"
    )
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    args = parser.parse_args()

    if args.model == "all":
        keys = list(MODELS.keys())
    else:
        keys = [args.model]

    for key in keys:
        try:
            download_model(key, force=args.force)
        except Exception as e:
            print(f"[{key}] 下载失败: {e}")

    print(f"\n所有模型已就绪。")
    print(f"  模型目录: {Path('./models').absolute()}")


if __name__ == "__main__":
    main()
