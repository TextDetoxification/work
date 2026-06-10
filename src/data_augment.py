import json, time, re, argparse
from pathlib import Path
from datasets import load_dataset
from openai import OpenAI

LANG_NAMES = {"en":"English","zh":"Chinese","ru":"Russian","uk":"Ukrainian","de":"German",
              "es":"Spanish","am":"Amharic","ar":"Arabic","hi":"Hindi","fr":"French",
              "he":"Hebrew","it":"Italian","ja":"Japanese","tt":"Tatar","hin":"Hinglish"}
ZERO_SHOT_LANGS = ["ja","fr","it","he","tt","hin"]


class LLMAugmenter:
    def __init__(self, client, model="deepseek-chat", temperature=0.8):
        self.client, self.model, self.temperature = client, model, temperature

    def _call(self, system, user, retries=3):
        for i in range(retries):
            try:
                return self.client.chat.completions.create(
                    model=self.model, temperature=self.temperature, max_tokens=4096,
                    messages=[{"role":"system","content":system},{"role":"user","content":user}]
                ).choices[0].message.content
            except Exception as e:
                if i == retries - 1:
                    raise
                time.sleep(2 ** i)

    def _parse_json(self, text):
        try: return json.loads(text)
        except: pass
        m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text or "")
        return json.loads(m.group(1)) if m else None

    def augment_toxic_paraphrase(self, toxic, neutral, lang="en", n=5):
        name = LANG_NAMES.get(lang, lang)
        system = f"Generate {n} toxic paraphrases in {name}. Keep same toxicity level."
        user = f'Toxic: "{toxic}"\nNeutral: "{neutral}"\nOutput JSON: [{{"toxic":"...","neutral":"{neutral}"}},...]'
        return self._parse_json(self._call(system, user)) or []

    def augment_neutral_paraphrase(self, toxic, neutral, lang="en", n=5):
        name = LANG_NAMES.get(lang, lang)
        system = f"Generate {n} alternative neutral versions in {name}."
        user = f'Toxic: "{toxic}"\nNeutral: "{neutral}"\nOutput JSON: [{{"toxic":"{toxic}","neutral":"..."}},...]'
        return self._parse_json(self._call(system, user)) or []

    def augment_new_pairs(self, examples, lang="en", n=10):
        name = LANG_NAMES.get(lang, lang)
        few = "\n".join(f'toxic: "{e["toxic"]}"\nneutral: "{e["neutral"]}"' for e in examples[:3])
        system = f"Create {n} new diverse toxic→neutral pairs in {name}."
        return self._parse_json(self._call(system, f"Examples:\n{few}\n\nJSON:")) or []

    def augment_cross_lingual(self, examples, src_lang="en", tgt_lang="ja", n=20):
        src_n, tgt_n = LANG_NAMES.get(src_lang,src_lang), LANG_NAMES.get(tgt_lang,tgt_lang)
        few = "\n".join(f'toxic: "{e["toxic"]}"\nneutral: "{e["neutral"]}"' for e in examples[:3])
        system = f"Generate native {tgt_n} toxic→neutral pairs (NOT translations from {src_n})."
        result = self._parse_json(self._call(system, f"Reference:\n{few}\n\n{n} pairs in {tgt_n}:")) or []
        for r in result:
            r["lang"] = tgt_lang
        return result


def load_pool(languages, n=20):
    ds = load_dataset("textdetox/multilingual_paradetox")
    pool = {}
    for lang in languages:
        if lang in ds:
            d = ds[lang].shuffle(42).select(range(min(n, len(ds[lang]))))
            pool[lang] = [{"toxic":r["toxic_sentence"],"neutral":r["neutral_sentence"]} for r in d]
    return pool


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api_key", required=True)
    p.add_argument("--base_url", default="https://api.deepseek.com")
    p.add_argument("--model", default="deepseek-chat")
    p.add_argument("--strategy", default="all",
                   choices=["all","toxic_para","neutral_para","new_pairs","cross_lingual"])
    p.add_argument("--langs", nargs="+", default=None)
    p.add_argument("--all_languages", action="store_true")
    p.add_argument("--n_per_pair", type=int, default=5)
    p.add_argument("--n_new", type=int, default=20)
    p.add_argument("--n_cross", type=int, default=30)
    p.add_argument("--output", default="./data/augmented.json")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    if args.all_languages:
        trained = ["en","zh","ru","uk","de","es","am","ar","hi"]
    elif args.langs:
        trained = args.langs
    else:
        trained = ["en","zh"]
    print(f"Trained: {trained}  |  Zero-shot: {ZERO_SHOT_LANGS}")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    aug = LLMAugmenter(client, model=args.model)
    pool = load_pool(trained)
    all_data = []

    for lang in trained:
        if lang not in pool:
            continue
        samples = pool[lang]
        for i, pair in enumerate(samples[:5]):
            if args.strategy in ("all","toxic_para") and not args.dry_run:
                d = aug.augment_toxic_paraphrase(pair["toxic"],pair["neutral"],lang,args.n_per_pair)
                for x in d:
                    x["lang"] = lang
                all_data.extend(d)
            if args.strategy in ("all","neutral_para") and not args.dry_run:
                d = aug.augment_neutral_paraphrase(pair["toxic"],pair["neutral"],lang,args.n_per_pair)
                for x in d:
                    x["lang"] = lang
                all_data.extend(d)
        if args.strategy in ("all","new_pairs") and not args.dry_run:
            d = aug.augment_new_pairs(samples[:5], lang, args.n_new)
            for x in d:
                x["lang"] = lang
            all_data.extend(d)

    if args.strategy in ("all","cross_lingual") and not args.dry_run:
        en_samples = pool.get("en", list(pool.values())[0] if pool else [])
        for tgt in ZERO_SHOT_LANGS:
            all_data.extend(aug.augment_cross_lingual(en_samples[:5], "en", tgt, args.n_cross))

    if all_data:
        seen, unique = set(), []
        for item in all_data:
            k = (item.get("toxic",""), item.get("neutral",""))
            if k not in seen:
                seen.add(k)
                unique.append(item)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(unique, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(unique)} samples to {args.output}")


if __name__ == "__main__":
    main()
