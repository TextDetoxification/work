import argparse, pandas as pd
from pathlib import Path
from datasets import load_dataset

DATA_DIR = "./data"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["csv", "json", "all"], default="all")
    parser.add_argument("--no-merge", action="store_true")
    args = parser.parse_args()

    ds = load_dataset("textdetox/multilingual_paradetox")
    print(f"Loaded {len(ds)} languages, {sum(len(ds[l]) for l in ds)} samples")

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

    if not args.no_merge:
        df_all = pd.DataFrame(all_rows)
        if export_csv:
            df_all.to_csv(f"{DATA_DIR}/all_languages.csv", index=False)
        if export_json:
            df_all.to_json(f"{DATA_DIR}/all_languages.json", orient="records", force_ascii=False)
        print(f"Merged: {len(df_all)} samples")
    print("Done.")

if __name__ == "__main__":
    main()
