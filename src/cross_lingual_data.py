import sys, json, random, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
from tqdm import tqdm
from src.back_translate import BackTranslator

ALL_LANGS = ["am", "ar", "de", "en", "es", "fr", "he", "hi", "hin",
             "it", "ja", "ru", "tt", "uk", "zh"]
DATA_DIR = "./data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nllb_model", default=None)
    parser.add_argument("--output", default="./data/cross_lingual_pairs.json")
    parser.add_argument("--max_per_lang", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    sentences = []
    for csv_path in sorted(Path(DATA_DIR).glob("*.csv")):
        lang = csv_path.stem
        if lang == "all_languages":
            continue
        df = pd.read_csv(csv_path)
        items = [{"toxic": r["toxic_sentence"], "neutral": r["neutral_sentence"], "lang": lang}
                 for _, r in df.iterrows()]
        if args.max_per_lang and len(items) > args.max_per_lang:
            items = random.sample(items, args.max_per_lang)
        sentences.extend(items)

    langs = sorted(set(s["lang"] for s in sentences))
    print(f"Sentences: {len(sentences)} across {langs}")

    translator = BackTranslator(nllb_model_name=args.nllb_model, device=args.device)
    total = len(sentences) * 14
    print(f"Translating {len(sentences)} × 14 = {total} times")
    pbar = tqdm(total=total)
    errors = 0

    groups = []
    for item in sentences:
        src = item["lang"]
        versions = [{"lang": src, "toxic": item["toxic"]}]
        for tgt in ALL_LANGS:
            if tgt == src:
                continue
            try:
                versions.append({"lang": tgt, "toxic": translator.translate(item["toxic"], src, tgt)})
            except Exception as e:
                errors += 1
                if errors <= 3:
                    tqdm.write(f"[{src}→{tgt}] {e}")
            pbar.update(1)
        groups.append({"toxic_versions": versions, "neutral": item["neutral"]})
    pbar.close()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(groups, f, ensure_ascii=False, indent=2)

    total_v = sum(len(g["toxic_versions"]) for g in groups)
    ok_rate = (total - errors) / total * 100 if total else 0
    print(f"Saved {len(groups)} groups, {total_v} versions ({total_v/len(groups):.1f}/group) | "
          f"Translation success: {ok_rate:.0f}% ({total - errors}/{total})")


if __name__ == "__main__":
    main()
