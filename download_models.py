import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODELS = {
    "mt0":  {"name": "bigscience/mt0-large", "dir": "./models/mt0-large"},
    "nllb": {"name": "facebook/nllb-200-distilled-600M", "dir": "./models/nllb-200-distilled-600M"},
}

def download(key, force=False):
    info = MODELS[key]
    save_dir = Path(info["dir"])
    if save_dir.exists() and list(save_dir.glob("*.json")) and not force:
        print(f"[{key}] exists, skip")
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(info["name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(info["name"], torch_dtype=dtype, use_safetensors=True)
    tokenizer.save_pretrained(str(save_dir))
    model.save_pretrained(str(save_dir))
    print(f"[{key}] done")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["mt0", "nllb", "all"], default="all")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    keys = list(MODELS) if args.model == "all" else [args.model]
    for k in keys:
        download(k, args.force)

if __name__ == "__main__":
    main()
