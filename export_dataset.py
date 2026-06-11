"""
下载并导出全部数据（训练数据 + 有害词表）。

使用方法:
  python export_dataset.py              # CSV + JSON + 合并 + 有害词表
  python export_dataset.py --format csv  # 仅 CSV
  HF_ENDPOINT=https://hf-mirror.com python export_dataset.py  # 国内镜像
"""

import argparse, pandas as pd
from pathlib import Path
from datasets import load_dataset

DATA_DIR = "./data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["csv", "json", "all"], default="all")
    parser.add_argument("--merge", action="store_true", help="额外导出合并文件 all_languages")
    args = parser.parse_args()

    # ── 训练数据 ──
    print("Loading parallel detox data...")
    ds = load_dataset("textdetox/multilingual_paradetox")
    print(f"  {len(ds)} languages, {sum(len(ds[l]) for l in ds)} samples")

    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    export_json = args.format in ("json", "all")
    export_csv = args.format in ("csv", "all")
    all_rows = []

    for lang in ds:
        rows = [{"language": lang, "toxic_sentence": r["toxic_sentence"],
                 "neutral_sentence": r["neutral_sentence"]} for r in ds[lang]]
        df = pd.DataFrame(rows)
        if export_csv:
            df.to_csv(f"{DATA_DIR}/{lang}.csv", index=False)
        if export_json:
            df.to_json(f"{DATA_DIR}/{lang}.json", orient="records", force_ascii=False)
        all_rows.extend(rows)

    if args.merge:
        df_all = pd.DataFrame(all_rows)
        if export_csv:
            df_all.to_csv(f"{DATA_DIR}/all_languages.csv", index=False)
        if export_json:
            df_all.to_json(f"{DATA_DIR}/all_languages.json", orient="records", force_ascii=False)
        print(f"  Merged: {len(df_all)} samples")

    # ── 有害词表 ──
    print("Loading toxic lexicon...")
    lex = load_dataset("textdetox/multilingual_toxic_lexicon")
    lex_path = Path(f"{DATA_DIR}/toxic_lexicon")
    lex.save_to_disk(str(lex_path))
    total_words = sum(len(lex[lang]) for lang in lex)
    print(f"  {total_words} words in {len(lex)} languages → {lex_path}")

    print("Done.")


if __name__ == "__main__":
    main()
