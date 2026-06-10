import re
from pathlib import Path
from typing import List, Tuple
from datasets import load_dataset, load_from_disk


NLLB_LANG_MAP = {
    "amh_Ethi": "am", "arb_Arab": "ar", "deu_Latn": "de", "eng_Latn": "en",
    "spa_Latn": "es", "fra_Latn": "fr", "heb_Hebr": "he", "hin_Deva": "hi",
    "ita_Latn": "it", "jpn_Jpan": "ja", "rus_Cyrl": "ru", "tat_Cyrl": "tt",
    "ukr_Cyrl": "uk", "zho_Hans": "zh",
}


class ToxicLexicon:
    def __init__(self, cache_dir="./toxic_lexicon_cache/toxic_lexicon"):
        self.cache_dir = Path(cache_dir)
        self.toxic_sets = {}
        self.toxic_neutral_map = {}
        self._load()

    def _load(self):
        if self.cache_dir.exists():
            ds = load_from_disk(str(self.cache_dir))
        else:
            ds = load_dataset("textdetox/multilingual_toxic_lexicon")
            self.cache_dir.parent.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(self.cache_dir))
        for lang in ds:
            self.toxic_sets[lang] = {r["text"].strip().lower() for r in ds[lang] if r["text"].strip()}
        print(f"Loaded {sum(len(s) for s in self.toxic_sets.values())} toxic words in {len(self.toxic_sets)} langs")

    def _normalize(self, lang):
        return lang if lang in self.toxic_sets else NLLB_LANG_MAP.get(lang, None)

    def detect_toxic_words(self, text, lang) -> List[Tuple[int, int, str]]:
        lang = self._normalize(lang)
        if lang is None or lang not in self.toxic_sets:
            return []
        found = []
        for word in self.toxic_sets[lang]:
            if lang in ("zh", "ja"):
                pos = text.find(word)
                if pos != -1:
                    found.append((pos, pos + len(word), word))
            else:
                for m in re.finditer(r'\b' + re.escape(word) + r'\b', text, re.IGNORECASE):
                    found.append((m.start(), m.end(), word))
        found.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        filtered, last_end = [], -1
        for s, e, w in found:
            if s >= last_end:
                filtered.append((s, e, w))
                last_end = e
        return filtered

    def has_toxic_words(self, text, lang):
        return len(self.detect_toxic_words(text, lang)) > 0

    def mark_toxic_words(self, text, lang):
        result = text
        for s, e, _ in reversed(self.detect_toxic_words(text, lang)):
            result = result[:s] + f"<toxic>{result[s:e]}</toxic>" + result[e:]
        return result

    def build_toxic_neutral_map(self, parallel_data_dir="./data"):
        import pandas as pd
        data_dir = Path(parallel_data_dir)
        if not data_dir.exists():
            return
        for lang in self.toxic_sets:
            csv_path = data_dir / f"{lang}.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            mapping = {}
            for _, row in df.iterrows():
                toxic_text = str(row["toxic_sentence"]).lower()
                neutral_text = str(row["neutral_sentence"]).lower()
                for word in self.toxic_sets[lang]:
                    if lang in ("zh", "ja"):
                        if word in toxic_text and word not in neutral_text:
                            mapping[word] = ""
                    else:
                        if re.search(r'\b' + re.escape(word) + r'\b', toxic_text, re.IGNORECASE) and \
                           not re.search(r'\b' + re.escape(word) + r'\b', neutral_text, re.IGNORECASE):
                            mapping[word] = ""
            self.toxic_neutral_map[lang] = mapping

    def eliminate_toxic(self, text, lang, strategy="replace", model=None, tokenizer=None):
        lang = self._normalize(lang)
        if not lang:
            return text, False
        found = self.detect_toxic_words(text, lang)
        if not found:
            return text, False
        if strategy == "delete":
            result = text
            for s, e, _ in reversed(found):
                result = result[:s].rstrip() + " " + result[e:].lstrip()
            return result.strip(), True
        if strategy == "replace" and lang in self.toxic_neutral_map:
            result, m = text, self.toxic_neutral_map[lang]
            for s, e, w in reversed(found):
                rep = m.get(w.lower(), None)
                result = result[:s].rstrip() + " " + result[e:].lstrip() if rep == "" else \
                         result[:s] + rep + result[e:] if rep else result
            return result.strip(), True
        if strategy == "mask_fill" and model is not None:
            masked = text
            for i, (s, e, _) in enumerate(reversed(found[:5])):
                masked = masked[:s] + f"<extra_id_{i}>" + masked[e:]
            inputs = tokenizer(masked, return_tensors="pt").to(model.device)
            out = model.generate(**inputs, max_new_tokens=100, num_beams=3, early_stopping=True)
            return tokenizer.decode(out[0], skip_special_tokens=True), True
        return text, False

    def auto_eliminate(self, text, lang, model=None, tokenizer=None):
        for strategy in ["replace", "delete", "mask_fill"]:
            text, _ = self.eliminate_toxic(text, lang, strategy, model=model, tokenizer=tokenizer)
            if not self.has_toxic_words(text, lang):
                return text
        return text
