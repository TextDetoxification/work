import argparse, csv, json, random, re, sys, time
from pathlib import Path
from collections import defaultdict, Counter

_LOCAL_PYARROW = Path(__file__).resolve().parent.parent / "pyarrow_local"
if _LOCAL_PYARROW.exists():
    sys.path.insert(0, str(_LOCAL_PYARROW))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "new_data"

CJK_LANGS = {"zh", "ja"}
TRAINED_LANGS = ["am", "ar", "de", "en", "es", "hi", "ru", "uk", "zh"]

LANG_NAMES = {"en":"English","zh":"Chinese","ru":"Russian","uk":"Ukrainian","de":"German",
              "es":"Spanish","am":"Amharic","ar":"Arabic","hi":"Hindi","fr":"French",
              "he":"Hebrew","it":"Italian","ja":"Japanese","tt":"Tatar","hin":"Hinglish"}

_CJK_RANGES = [
    (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF),
    (0xF900, 0xFAFF), (0x3040, 0x309F), (0x30A0, 0x30FF),
]

def _is_cjk_char(ch):
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)

def _is_cjk_boundary(ch):
    if not ch: return True
    return (ch.isspace() or
            ch in "，。！？；、：,\.!\?;:\"\"\(\)\[\]{}@#\$%^&\*\+=\|\\/<>～〜" or
            ("\u0000" <= ch <= "\u007f" and not ch.isalnum()))


class HarmfulWordAugmenter:

    def __init__(self, data_dir=None, lexicon_dir=None, augmented_path=None, seed=42,
                 api_key=None, base_url="https://api.deepseek.com", model="deepseek-chat"):
        self.data_dir = Path(data_dir or DATA_DIR)
        self.lexicon_dir = Path(lexicon_dir or (self.data_dir / "toxic_lexicon"))
        self.augmented_path = Path(augmented_path or (self.data_dir / "augmented.json"))
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = None
        if api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key, base_url=base_url)
                print(f"LLM client ready: {model} @ {base_url}")
            except ImportError:
                print("  openai not installed, LLM strategies disabled")
            except Exception as e:
                print(f"  LLM init failed: {e}")
        random.seed(seed)
        self.toxic_groups = defaultdict(lambda: defaultdict(list))
        self.toxic_words_list = {}
        self.toxic_words_set = {}
        self.toxic_cjk_index = {}
        self.all_neutrals = {}
        self.toxic_samples = {}
        t0 = time.time()
        self._load_csv_data()
        self._load_augmented_data()
        self._build_neutral_pool()
        self._load_toxic_words()
        self._build_indices()
        print(f"Init done in {time.time()-t0:.1f}s")
        if not self.toxic_words_list:
            print("ERROR: No toxic words loaded.")
            sys.exit(1)

    def _load_csv_data(self):
        for csv_path in sorted(self.data_dir.glob("*.csv")):
            lang = csv_path.stem
            if lang == "all_languages": continue
            with open(csv_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    t = row["toxic_sentence"]; n = row["neutral_sentence"]
                    if n not in self.toxic_groups[lang][t]:
                        self.toxic_groups[lang][t].append(n)
                    pair = {"toxic": t, "neutral": n}
                    if lang not in self.toxic_samples:
                        self.toxic_samples[lang] = []
                    if pair not in self.toxic_samples[lang]:
                        self.toxic_samples[lang].append(pair)
        n = sum(len(g) for g in self.toxic_groups.values())
        print(f"CSV: {sum(len(g) for g in self.toxic_groups.values())} toxics, "
              f"{sum(len(g) for lg in self.toxic_groups.values() for g in lg.values())} pairs, "
              f"{len(self.toxic_groups)} langs")

    def _load_augmented_data(self):
        if not self.augmented_path.exists():
            print("  (no augmented.json found)"); return
        with open(self.augmented_path, "r", encoding="utf-8") as f:
            aug_data = json.load(f)
        added = 0
        for item in aug_data:
            lang = item.get("lang",""); toxic = item.get("toxic",""); neutral = item.get("neutral","")
            if not lang or not toxic or not neutral: continue
            if neutral not in self.toxic_groups[lang][toxic]:
                self.toxic_groups[lang][toxic].append(neutral); added += 1
        total = sum(len(g) for lg in self.toxic_groups.values() for g in lg.values())
        n_multi = sum(1 for lg in self.toxic_groups.values() for g in lg.values() if len(g) > 1)
        per = {}
        for lang, groups in self.toxic_groups.items():
            multi = sum(1 for ns in groups.values() if len(ns) > 1)
            per[lang] = f"{multi}/{len(groups)}"
        print(f"Augmented: +{added} neutral paraphrases -> {total} total pairs, "
              f"{n_multi} toxics have >1 neutral")
        print(f"  Multi-neutral by lang: {per}")

    def _build_neutral_pool(self):
        for lang, groups in self.toxic_groups.items():
            seen = set(); uniq = []
            for neutrals in groups.values():
                for n in neutrals:
                    if n not in seen: seen.add(n); uniq.append(n)
            self.all_neutrals[lang] = uniq

    def _load_toxic_words(self):
        official = self._load_official_lexicon()
        extracted = self._extract_toxic_words_from_data()
        for lang in set(list(official.keys()) + list(extracted.keys())):
            off = official.get(lang, []); ext = extracted.get(lang, [])
            offset = set(w.lower() for w in off)
            merged = list(off)
            for w in ext:
                if w.lower() not in offset: merged.append(w); offset.add(w.lower())
            if merged: self.toxic_words_list[lang] = merged
        total = sum(len(v) for v in self.toxic_words_list.values())
        off_total = sum(len(v) for v in official.values())
        print(f"Toxic words: {total} ({off_total} official + "
              f"{total - off_total} extracted), {len(self.toxic_words_list)} langs")

    def _build_indices(self):
        for lang, words in self.toxic_words_list.items():
            self.toxic_words_set[lang] = {w.lower() for w in words}
            if lang in CJK_LANGS:
                idx = defaultdict(list)
                for w in words:
                    if w: idx[w[0].lower()].append(w)
                self.toxic_cjk_index[lang] = dict(idx)
        print("Indices built")

    def _load_official_lexicon(self):
        words = {}
        try:
            import pyarrow as pa
        except ImportError:
            return words
        if not self.lexicon_dir.exists(): return words
        for lang_dir in sorted(self.lexicon_dir.iterdir()):
            if not lang_dir.is_dir(): continue
            lang = lang_dir.name
            f = lang_dir / "data-00000-of-00001.arrow"
            if not f.exists(): continue
            try:
                with pa.memory_map(str(f)) as src:
                    tbl = pa.ipc.open_stream(src).read_all()
                if "text" in tbl.column_names:
                    lw = [str(r.as_py()).strip() for r in tbl.column("text")
                          if r.as_py() and str(r.as_py()).strip()]
                    if lw: words[lang] = lw
            except Exception: pass
        if words:
            print(f"  Official: {sum(len(v) for v in words.values())} words, {len(words)} langs")
        return words

    def _extract_toxic_words_from_data(self):
        STOP = {"the","a","an","is","are","was","were","be","been","being","have","has",
                "had","do","does","did","will","would","could","should","may","might",
                "can","shall","i","me","my","we","our","you","your","he","she","it",
                "they","them","his","her","its","their","this","that","these","those",
                "and","but","or","not","no","so","if","then","than","too","very","just",
                "now","here","there","also","of","in","to","for","on","with","at","by",
                "from","as","into","about","all","up","out","some","more","one","two",
                "get","got","go","see","know","make","like","come","take","think","say",
                "said","way","even","still","well","back","any","much","only","other",
                "new","good","first","last","most","really","because","what","when",
                "where","who","how","which","don","doesn","didn","won","isn","aren",
                "s","t","ll","ve","re","m","d"}
        words = {}
        for lang, groups in self.toxic_groups.items():
            wc = Counter()
            for toxic, neutrals in groups.items():
                tt = self._tokenize_for_diff(toxic, lang)
                ns_all = set()
                for n in neutrals: ns_all.update(self._tokenize_for_diff(n, lang))
                uq = [t for t in tt if t not in ns_all and t.lower() not in STOP]
                for t in set(uq): wc[t] += 1
            mc = 2 if lang in CJK_LANGS else 3
            ml = 1 if lang in CJK_LANGS else 3
            lw = [w for w, c in wc.items() if c >= mc and ml <= len(w) <= 40]
            lw.sort(key=lambda w: -wc[w])
            if lw: words[lang] = lw
        if words:
            print(f"  Extracted: {sum(len(v) for v in words.values())} words, {len(words)} langs")
        return words

    @staticmethod
    def _tokenize_for_diff(text, lang):
        text = text.lower().strip()
        if lang in CJK_LANGS:
            tokens = []
            for seg in re.split(r"[，。！？；、：\s,\.!\?;:\"\"\(\)\[\]{}@#\$%^&\*\+=\|\\/<>]+", text):
                seg = seg.strip()
                if not seg: continue
                tokens.append(seg)
                if len(seg) >= 4:
                    for i in range(len(seg) - 1): tokens.append(seg[i:i+2])
            return tokens
        return re.findall(r"\b\w+\b", text)

    def _find_toxic(self, text, lang):
        if lang not in self.toxic_words_list: return []
        if lang in CJK_LANGS: return self._find_cjk(text, lang)
        return self._find_non_cjk(text, lang)

    def _find_non_cjk(self, text, lang):
        tset = self.toxic_words_set.get(lang, set())
        found = []
        for m in re.finditer(r"\b\w+\b", text):
            if m.group().lower() in tset:
                found.append((m.start(), m.end(), m.group()))
        return found

    def _find_cjk(self, text, lang):
        idx = self.toxic_cjk_index.get(lang, {})
        if not idx: return []
        tl = text.lower(); tlen = len(text); found = []
        for pos in range(tlen):
            ch = tl[pos]
            candidates = idx.get(ch, [])
            if not candidates: continue
            for w in candidates:
                wlen = len(w); end = pos + wlen
                if end > tlen: continue
                if tl[pos:end] != w.lower(): continue
                if wlen == 1:
                    ca = text[end] if end < tlen else ""
                    if not _is_cjk_boundary(ca) and _is_cjk_char(ca): continue
                found.append((pos, end, w))
        found.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        filtered, le = [], -1
        for s, e, w in found:
            if s >= le: filtered.append((s, e, w)); le = e
        return filtered

    def _pick_alt(self, word, lang):
        pool = [w for w in self.toxic_words_list.get(lang, []) if w.lower() != word.lower()]
        if not pool: return None
        ol = len(word)
        sim = [w for w in pool if abs(len(w) - ol) <= max(3, ol * 0.4)]
        return random.choice(sim) if sim and len(sim) >= 3 else random.choice(pool)

    @staticmethod
    def _replace(text, s, e, repl):
        orig = text[s:e]
        if orig.isupper(): repl = repl.upper()
        elif orig and orig[0].isupper(): repl = repl[0].upper() + repl[1:]
        return text[:s] + repl + text[e:]

    def _llm_call(self, system, user, retries=3):
        if not self.client: return None
        for i in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, temperature=0.8, max_tokens=4096,
                    messages=[{"role":"system","content":system},
                              {"role":"user","content":user}])
                return resp.choices[0].message.content
            except Exception as e:
                msg = str(e)
                is_limit = "429" in msg or "quota" in msg.lower() or "rate" in msg.lower()
                if is_limit and i < retries - 1:
                    wait = 10 * (i + 1)
                    print(f"    Rate limited, waiting {wait}s...")
                    time.sleep(wait); continue
                if i < retries - 1: time.sleep(2)
                else:
                    print(f"    LLM call failed: {msg[:120]}")
                    if is_limit:
                        print("    -> Quota may be exhausted, progress saved")
                        raise SystemExit(0)
                    return None

    def _parse_json(self, text):
        try: return json.loads(text)
        except: pass
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text or "")
        return json.loads(m.group(1)) if m else None

    def augment_synonym(self, lang, n_variants=3):
        if lang not in self.toxic_groups or lang not in self.toxic_words_list: return []
        results, seen = [], set()
        for toxic, neutrals in self.toxic_groups[lang].items():
            found = self._find_toxic(toxic, lang)
            if not found: continue
            variants = []
            for _ in range(n_variants):
                nt = toxic; ok = False
                for s, e, w in reversed(found):
                    r = self._pick_alt(w, lang)
                    if r is None: continue
                    nt = self._replace(nt, s, e, r); ok = True
                if ok and nt != toxic:
                    if nt.strip() not in variants: variants.append(nt.strip())
            for v in variants:
                for neutral in neutrals:
                    k = (v, neutral.strip())
                    if k not in seen:
                        seen.add(k)
                        results.append({"toxic": v, "neutral": neutral, "lang": lang, "parent_toxic": toxic})
        return results

    def _ins_points(self, text, lang):
        pts = []
        if lang in CJK_LANGS:
            for m in re.finditer(r"[，。！？；、,\.!\?;]", text): pts.append(m.end())
        else:
            for m in re.finditer(r"[,\.!\?;:]", text): pts.append(m.end())
            for m in re.finditer(r"\s", text): pts.append(m.start())
        if not pts and text: pts = [len(text)//2, 0, len(text)]
        return sorted(set(p for p in pts if 0 <= p <= len(text)))

    def _ins_word(self, text, pos, word, lang):
        if lang in CJK_LANGS: return text[:pos] + word + text[pos:]
        pre, suf = text[:pos], text[pos:]
        nb = pre and not pre[-1].isspace()
        na = suf and not suf[0].isspace()
        return (pre + (" " if nb else "") + word + (" " if na else "") + suf)

    def augment_new_rule(self, lang, n=50):
        if lang not in self.toxic_groups or lang not in self.toxic_words_list: return []
        pool = self.all_neutrals.get(lang, []); tw = self.toxic_words_list[lang]
        if len(pool) < 3 or len(tw) < 5: return []
        results, seen = [], set()
        att, ma = 0, n * 20
        while len(results) < n and att < ma:
            att += 1
            neutral = random.choice(pool)
            if len(neutral) < 5: continue
            nw = random.randint(1, min(3, len(tw)))
            chosen = random.sample(tw, nw)
            pts = self._ins_points(neutral, lang)
            if not pts: continue
            toxic = neutral
            spots = sorted(random.sample(pts, min(len(chosen), len(pts))))
            for pos, word in sorted(zip(spots, chosen[:len(spots)]), key=lambda x: -x[0]):
                toxic = self._ins_word(toxic, pos, word, lang)
            tc = " ".join(toxic.split()); nc = " ".join(neutral.split())
            if tc.lower() == nc.lower(): continue
            k = (tc.lower(), nc.lower())
            if k in seen: continue
            seen.add(k)
            results.append({"toxic": tc, "neutral": nc, "lang": lang})
        return results

    def augment_new_llm(self, lang, n=20):
        if not self.client:
            print("    No LLM client, skipping"); return []
        if lang not in self.toxic_samples: return []
        name = LANG_NAMES.get(lang, lang)
        examples = self.toxic_samples[lang]
        few_shot = random.sample(examples, min(5, len(examples)))
        few_text = "\n".join("toxic: \"" + e["toxic"] + "\"\nneutral: \"" + e["neutral"] + "\"" for e in few_shot)
        system = (
            f"You are a native {name} speaker. Generate {n} BRAND NEW, diverse "
            f"toxic->neutral sentence pairs in {name}. "
            f"These must be ORIGINAL -- NOT translations of the examples, NOT variations. "
            f"Create genuinely new scenarios, topics, and expressions. "
            f"The toxic sentence should contain harmful/offensive language. "
            f"The neutral sentence should convey the same meaning without toxicity. "
            f"Output ONLY a JSON array: [{{\"toxic\":\"...\",\"neutral\":\"...\"}},...]"
        )
        user = (
            f"Here are some examples of the style (create DIFFERENT ones):\n"
            f"{few_text}\n\nNow generate {n} entirely new pairs in {name}:"
        )
        result = self._llm_call(system, user)
        if result is None: return []
        parsed = self._parse_json(result)
        if not isinstance(parsed, list): return []
        pairs = []
        for item in parsed:
            if isinstance(item, dict) and "toxic" in item and "neutral" in item:
                pairs.append({"toxic": item["toxic"], "neutral": item["neutral"], "lang": lang})
        return pairs

    def run(self, langs=None, strategies=None, n_synonym=3, n_new_rule=50, n_new_llm=20):
        if langs is None:
            langs = [l for l in TRAINED_LANGS if l in self.toxic_groups]
        if strategies is None: strategies = ["all"]
        do_synonym = "all" in strategies or "synonym" in strategies
        do_new_rule = "all" in strategies or "new_rule" in strategies
        do_new_llm = "all" in strategies or "new_llm" in strategies
        if "new_pair" in strategies:
            if self.client: do_new_llm = True
            else: do_new_rule = True

        allr, stats = [], {}
        t0 = time.time()
        for lang in langs:
            lr = []
            groups = self.toxic_groups.get(lang, {})
            n_toxics = len(groups)
            n_multi = sum(1 for ns in groups.values() if len(ns) > 1)
            n_words = len(self.toxic_words_list.get(lang, []))
            print(f"\n[{'<' + lang + '>':-^45}]")
            print(f"  {n_toxics} toxics, {n_multi} with >1 neutral, {n_words} toxic words")
            t1 = time.time()
            if do_synonym:
                syn = self.augment_synonym(lang, n_synonym); lr.extend(syn)
                print(f"  Synonym variants:  {len(syn)}")
            if do_new_rule:
                nr = self.augment_new_rule(lang, n_new_rule); lr.extend(nr)
                print(f"  New pairs (rule):  {len(nr)}")
            if do_new_llm:
                nl = self.augment_new_llm(lang, n_new_llm); lr.extend(nl)
                print(f"  New pairs (LLM):   {len(nl)}")
            stats[lang] = len(lr)
            allr.extend(lr)
            print(f"  -> {len(lr)} total, {time.time()-t1:.1f}s")

        seen = set(); uniq = []
        orig_added = 0

        # Build index: parent_toxic -> synonym variants
        syn_by_parent = defaultdict(list)
        new_pairs = []
        for item in allr:
            if "parent_toxic" in item:
                syn_by_parent[item["parent_toxic"]].append(item)
            else:
                new_pairs.append(item)

        # Interleave: for each original toxic, originals first then synonym variants
        for lang in langs:
            for toxic, neutrals in self.toxic_groups.get(lang, {}).items():
                for neutral in neutrals:
                    k = (toxic.strip(), neutral.strip(), lang)
                    if k not in seen:
                        seen.add(k); uniq.append({"toxic": toxic, "neutral": neutral, "lang": lang})
                        orig_added += 1
                for item in syn_by_parent.get(toxic, []):
                    k = (item["toxic"].strip(), item["neutral"].strip(), item["lang"])
                    if k not in seen:
                        seen.add(k); uniq.append({k2: v2 for k2, v2 in item.items() if k2 != "parent_toxic"})

        # New pairs (rule/LLM) at the end
        for item in new_pairs:
            k = (item["toxic"].strip(), item["neutral"].strip(), item["lang"])
            if k not in seen: seen.add(k); uniq.append(item)

        elapsed = time.time() - t0
        print(f"\n{'='*50}")
        print(f"Augmented: {len(allr)} -> {len(allr)} unique")
        print(f"Original:  +{orig_added}")
        print(f"Total:     {len(uniq)} in {elapsed:.1f}s")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self._save(uniq)
        self._summary(stats, len(uniq), orig_added)
        return uniq

    def _save(self, results):
        # Preserve order from run(): originals followed by synonym variants
        by_lang = defaultdict(list)
        for item in results:
            by_lang[item["lang"]].append(item)

        for lang in sorted(by_lang):
            p = OUTPUT_DIR / f"{lang}.csv"
            for attempt in range(5):
                try:
                    with open(p, "w", encoding="utf-8", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["language", "toxic_sentence", "neutral_sentence"])
                        row_count = 0
                        seen_pair = set()
                        for item in by_lang[lang]:
                            k = (item["toxic"].strip(), item["neutral"].strip())
                            if k not in seen_pair:
                                seen_pair.add(k)
                                w.writerow([lang, item["toxic"], item["neutral"]])
                                row_count += 1
                    print(f"  {p.name:<20} {row_count:>6} rows")
                    break
                except PermissionError:
                    print(f"    PermissionError on {p.name}, retry {attempt+1}/5...")
                    time.sleep(3)
        pj = OUTPUT_DIR / "augmented_new.json"
        with open(pj, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  {'augmented_new.json':<20} {len(results):>6} items")

    def _summary(self, stats, total, orig):
        print(f"\n{'-'*45}")
        print(f"{'Language':<12} {'New':>8}  {'Multi-N':>8}")
        print(f"{'-'*45}")
        for lang in sorted(stats):
            groups = self.toxic_groups.get(lang, {})
            multi = sum(1 for ns in groups.values() if len(ns) > 1)
            print(f"{lang:<12} {stats[lang]:>8}  {multi:>8}")
        print(f"{'-'*45}")
        print(f"{'Total new':<12} {sum(stats.values()):>8}")
        print(f"{'Total orig':<12} {orig:>8}")
        print(f"{'TOTAL':<12} {total:>8}")

def main():
    p = argparse.ArgumentParser(description="Data Augmentation V2")
    p.add_argument("--langs", nargs="+", default=None)
    p.add_argument("--strategy", default="all",
                   choices=["all","synonym","new_pair","new_rule","new_llm"])
    p.add_argument("--n_synonym", type=int, default=3)
    p.add_argument("--n_new_rule", type=int, default=50)
    p.add_argument("--n_new_llm", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    p.add_argument("--api_key", default=None)
    p.add_argument("--base_url", default="https://api.deepseek.com")
    p.add_argument("--model", default="deepseek-chat")
    args = p.parse_args()
    if args.output:
        global OUTPUT_DIR; OUTPUT_DIR = Path(args.output)
    print("="*50)
    print(f"Data Augmentation V2 | strategy={args.strategy} | "
          f"n_synonym={args.n_synonym} | n_new_rule={args.n_new_rule}")
    if args.api_key:
        print(f"LLM: {args.model} @ {args.base_url} | n_new_llm={args.n_new_llm}")
    print(f"Output: {OUTPUT_DIR.resolve()}")
    print("="*50)
    HarmfulWordAugmenter(
        seed=args.seed, api_key=args.api_key,
        base_url=args.base_url, model=args.model,
    ).run(langs=args.langs, strategies=[args.strategy],
          n_synonym=args.n_synonym, n_new_rule=args.n_new_rule, n_new_llm=args.n_new_llm)

if __name__ == "__main__":
    main()
