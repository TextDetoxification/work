"""
下载并导出多语言去毒数据集。

功能:
  1. 从 HuggingFace 下载 textdetox/multilingual_paradetox
  2. 导出为 CSV / JSON（按语言拆分 + 全部合并）

使用方法:
  python export_dataset.py                  # CSV + JSON + 合并
  python export_dataset.py --format csv     # 仅 CSV
"""

import argparse
import pandas as pd
from pathlib import Path
from datasets import load_dataset

DATA_DIR = "./data"


def main():
    parser = argparse.ArgumentParser(description="下载并导出多语言去毒数据集")
    parser.add_argument("--format", choices=["csv", "json", "all"], default="all",
                        help="导出格式 (默认: all)")
    parser.add_argument("--no-merge", action="store_true",
                        help="跳过合并导出")
    args = parser.parse_args()

    # ── 下载 ──
    print("Loading dataset from HuggingFace...")
    ds = load_dataset("textdetox/multilingual_paradetox")
    print(f"  语言数: {len(ds)}  总条数: {sum(len(ds[lang]) for lang in ds)}")

    # ── 导出 ──
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    export_json = args.format in ("json", "all")
    export_csv = args.format in ("csv", "all")

    all_rows = []
    for lang in ds:
        rows = []
        for row in ds[lang]:
            rows.append({
                "language": lang,
                "toxic_sentence": row["toxic_sentence"],
                "neutral_sentence": row["neutral_sentence"],
            })
        lang_df = pd.DataFrame(rows)

        if export_csv:
            lang_df.to_csv(f"{DATA_DIR}/{lang}.csv", index=False)
            print(f"  CSV:  {DATA_DIR}/{lang}.csv  ({len(lang_df)} 条)")

        if export_json:
            lang_df.to_json(f"{DATA_DIR}/{lang}.json", orient="records", force_ascii=False)
            print(f"  JSON: {DATA_DIR}/{lang}.json  ({len(lang_df)} 条)")

        all_rows.extend(rows)

    # ── 合并 ──
    if not args.no_merge:
        df_all = pd.DataFrame(all_rows)
        if export_csv:
            df_all.to_csv(f"{DATA_DIR}/all_languages.csv", index=False)
        if export_json:
            df_all.to_json(f"{DATA_DIR}/all_languages.json", orient="records", force_ascii=False)
        print(f"\n  合并导出: {len(df_all)} 条, {len(ds)} 种语言")

    print("Done.")


if __name__ == "__main__":
    main()
