"""
LLM 数据增强：扩充多语言去毒训练数据

支持策略:
  1. toxic 有害词替换扩增
  2. neutral 同义改写扩增
  3. 全新 toxic→neutral 对生成
  4. 跨语言数据生成（为 0-shot 语言造数据）

默认使用 DeepSeek API（OpenAI 兼容），换其他平台仅需改 base_url。

使用方法:
  python -m src.data_augment --api_key YOUR_KEY --strategy all
  python -m src.data_augment --api_key YOUR_KEY --strategy paraphrase --lang en --n 5
"""

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datasets import load_dataset

from openai import OpenAI


# ── 语言名称映射 ──────────────────────────────────────────────
LANG_NAMES = {
    "en": "English", "zh": "Chinese", "ru": "Russian", "uk": "Ukrainian",
    "de": "German", "es": "Spanish", "am": "Amharic", "ar": "Arabic", "hi": "Hindi",
    "fr": "French", "he": "Hebrew", "it": "Italian", "ja": "Japanese",
    "tt": "Tatar", "hin": "Hinglish",
}

# ── 零资源语言（目标生成语言）────────────────────────────────
ZERO_SHOT_LANGS = ["ja", "fr", "it", "he", "tt", "hin"]


class LLMAugmenter:
    """用 LLM 对去毒数据做多策略增强"""

    def __init__(self, client: OpenAI, model: str = "deepseek-chat", temperature: float = 0.8):
        self.client = client
        self.model = model
        self.temperature = temperature

    # ── 底层调用 ──────────────────────────────────────────────

    def _call_llm(self, system: str, user: str, n_retries: int = 3) -> Optional[str]:
        """调用 LLM，带重试"""
        for attempt in range(n_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self.temperature,
                    max_tokens=4096,
                )
                return resp.choices[0].message.content
            except Exception as e:
                print(f"  API 调用失败 (尝试 {attempt+1}/{n_retries}): {e}")
                if attempt < n_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
        return None

    def _extract_json(self, text: str) -> Optional[List[Dict]]:
        """从 LLM 回复中提取 JSON"""
        if not text:
            return None
        # 尝试直接解析
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        # 尝试提取 ```json ... ``` 代码块
        import re
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass
        print(f"  ⚠ 无法解析 JSON: {text[:200]}...")
        return None

    # ── 策略 1：有害词替换扩增 ────────────────────────────────

    def augment_toxic_paraphrase(
        self,
        toxic: str,
        neutral: str,
        lang: str = "en",
        n_variants: int = 5,
    ) -> List[Dict[str, str]]:
        """
        保持 neutral 不变，为 toxic 生成有害词替换变体。
        扩增 toxic 侧的多样性。
        """
        lang_name = LANG_NAMES.get(lang, lang)
        system = f"""You are a data augmentation expert for multilingual text detoxification.
Your task: given a toxic sentence and its neutral version, generate {n_variants} new
toxic sentences by replacing the harmful/insulting words with DIFFERENT harmful words
of the SAME intensity. The meaning/topic should vary, but the detoxification difficulty
should stay similar. Output in {lang_name}."""

        user = f"""Original toxic: "{toxic}"
Original neutral: "{neutral}"

Generate {n_variants} new toxic sentences that:
1. Replace the harmful words with different harmful synonyms
2. Keep the same level of toxicity (don't make it milder or harsher)
3. Vary the topic slightly where natural
4. Are grammatically correct {lang_name}

Output as JSON array:
[{{"toxic": "new toxic sentence 1", "neutral": "{neutral}"}},
 {{"toxic": "new toxic sentence 2", "neutral": "{neutral}"}},
 ...]"""

        text = self._call_llm(system, user)
        results = self._extract_json(text)
        return results if results else []

    # ── 策略 2：Neutral 同义改写扩增 ──────────────────────────

    def augment_neutral_paraphrase(
        self,
        toxic: str,
        neutral: str,
        lang: str = "en",
        n_variants: int = 5,
    ) -> List[Dict[str, str]]:
        """
        保持 toxic 不变，为 neutral 生成多种同义改写。
        扩增 neutral 侧的多样性，让模型学会更灵活的去毒表达。
        """
        lang_name = LANG_NAMES.get(lang, lang)
        system = f"""You are a data augmentation expert for text detoxification.
Given a toxic sentence and its neutral version, generate {n_variants} DIFFERENT
ways to express the same neutral meaning. Each variant should:
- Convey the SAME non-toxic message
- Use DIFFERENT words and sentence structures
- Sound natural in {lang_name}"""

        user = f"""Toxic: "{toxic}"
Neutral: "{neutral}"

Generate {n_variants} alternative neutral paraphrases in {lang_name}.
Output as JSON:
[{{"toxic": "{toxic}", "neutral": "paraphrase 1"}},
 {{"toxic": "{toxic}", "neutral": "paraphrase 2"}},
 ...]"""

        text = self._call_llm(system, user)
        results = self._extract_json(text)
        return results if results else []

    # ── 策略 3：生成全新 toxic→neutral 对 ─────────────────────

    def augment_new_pairs(
        self,
        examples: List[Dict[str, str]],
        lang: str = "en",
        n_pairs: int = 10,
    ) -> List[Dict[str, str]]:
        """
        用少量示例作为 few-shot 提示，生成全新的 toxic→neutral 对。
        """
        lang_name = LANG_NAMES.get(lang, lang)
        system = f"""You are a data generation expert for multilingual text detoxification.
Create NEW, DIVERSE, REALISTIC toxic→neutral sentence pairs in {lang_name}.
- Toxic sentences should contain insults, hate speech, or offensive language
- Neutral sentences should convey the same core meaning WITHOUT any toxicity
- Cover diverse topics: social media, workplace, family, politics, etc.
- Toxicity levels should vary from mild to strong"""

        few_shot = "\n".join(
            f"toxic: \"{e['toxic']}\"\nneutral: \"{e['neutral']}\""
            for e in examples[:3]
        )

        user = f"""Example pairs in {lang_name}:

{few_shot}

Now generate {n_pairs} completely NEW pairs in {lang_name}.
Make them diverse in topic, toxicity type (insults, threats, slurs, etc.), and style.
Output as JSON:
[{{"toxic": "...", "neutral": "..."}}, ...]"""

        text = self._call_llm(system, user)
        results = self._extract_json(text)
        return results if results else []

    # ── 策略 4：跨语言数据生成 ────────────────────────────────

    def augment_cross_lingual(
        self,
        examples: List[Dict[str, str]],
        src_lang: str = "en",
        tgt_lang: str = "ja",
        n_pairs: int = 20,
    ) -> List[Dict[str, str]]:
        """
        基于源语言示例，生成目标语言的 toxic→neutral 对。
        用于为零资源语言创建训练数据。
        """
        src_name = LANG_NAMES.get(src_lang, src_lang)
        tgt_name = LANG_NAMES.get(tgt_lang, tgt_lang)

        system = f"""You are a multilingual text detoxification expert.
Generate toxic→neutral sentence pairs in {tgt_name}.
The sentences should feel NATIVE in {tgt_name}, NOT translations from {src_name}.
- Toxic sentences: use culturally appropriate insults/offensive language in {tgt_name}
- Neutral sentences: natural, non-toxic rewrites in the same language"""

        few_shot = "\n".join(
            f"toxic: \"{e['toxic']}\"\nneutral: \"{e['neutral']}\""
            for e in examples[:3]
        )

        user = f"""Reference pairs in {src_name} (for style/tone reference only, DO NOT translate):

{few_shot}

Generate {n_pairs} NEW {tgt_name} sentence pairs that:
1. Are culturally NATIVE to {tgt_name} speakers
2. Cover diverse real-life scenarios
3. Match the toxicity intensity of the reference pairs

Output as JSON:
[{{"toxic": "...", "neutral": "..."}}, ...]"""

        text = self._call_llm(system, user)
        results = self._extract_json(text)
        # 添加语言标签
        for r in (results or []):
            r["lang"] = tgt_lang
        return results if results else []


# ── 主流程 ────────────────────────────────────────────────────

def load_sample_pool(languages: List[str], n_per_lang: int = 20) -> Dict[str, List[Dict]]:
    """每语种随机采样 n_per_lang 条作为 few-shot 示例"""
    ds = load_dataset("textdetox/multilingual_paradetox")
    pool = {}
    for lang in languages:
        if lang in ds:
            d = ds[lang].shuffle(seed=42).select(range(min(n_per_lang, len(ds[lang]))))
            pool[lang] = [{"toxic": r["toxic_sentence"], "neutral": r["neutral_sentence"]} for r in d]
    return pool


def save_augmented(data: List[Dict], output_path: str):
    """保存增强数据"""
    # 去重
    seen = set()
    unique = []
    for item in data:
        key = (item.get("toxic", ""), item.get("neutral", ""))
        if key not in seen:
            seen.add(key)
            unique.append(item)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)
    print(f"\n已保存 {len(unique)} 条增强数据到 {output_path}")


def main():
    parser = argparse.ArgumentParser(description="LLM 数据增强")
    parser.add_argument("--api_key", required=True, help="API Key")
    parser.add_argument("--base_url", default="https://api.deepseek.com", help="API 地址")
    parser.add_argument("--model", default="deepseek-chat", help="模型名称")
    parser.add_argument("--strategy", default="all",
                        choices=["all", "toxic_para", "neutral_para", "new_pairs", "cross_lingual"],
                        help="增强策略")
    parser.add_argument("--langs", nargs="+", default=None,
                        help="目标语言（与 --all_languages 二选一）")
    parser.add_argument("--all_languages", action="store_true",
                        help="覆盖全部 15 种语言")
    parser.add_argument("--n_per_pair", type=int, default=5, help="每对扩增倍数")
    parser.add_argument("--n_new_pairs", type=int, default=20, help="策略3 生成数量/语言")
    parser.add_argument("--n_cross_lingual", type=int, default=30, help="策略4 生成数量/语言")
    parser.add_argument("--output", default="./data/augmented.json", help="输出路径")
    parser.add_argument("--dry_run", action="store_true", help="干跑，只打印不调用 API")
    args = parser.parse_args()

    # ── 语言列表 ──
    if args.all_languages:
        TRAINED_LANGS = ["en", "zh", "ru", "uk", "de", "es", "am", "ar", "hi"]
    elif args.langs:
        TRAINED_LANGS = args.langs
    else:
        TRAINED_LANGS = ["en", "zh"]
    print(f"训练语言 ({len(TRAINED_LANGS)}): {TRAINED_LANGS}")
    print(f"零资源语言 ({len(ZERO_SHOT_LANGS)}): {ZERO_SHOT_LANGS}")

    # ── 初始化 ──
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    augmenter = LLMAugmenter(client, model=args.model)
    pool = load_sample_pool(TRAINED_LANGS)
    all_augmented = []

    for lang in TRAINED_LANGS:
        if lang not in pool or not pool[lang]:
            print(f"⚠ 语言 {lang} 无训练数据，跳过（跨语言生成请用 cross_lingual 策略）")
            continue

        samples = pool[lang]
        print(f"\n{'='*60}")
        print(f"[{lang}] {len(samples)} 条示例可用于增强")

        # ── 策略 1+2 ──
        if args.strategy in ("all", "toxic_para"):
            print(f"\n[策略1] 有害词替换扩增 ({lang})...")
            for i, pair in enumerate(samples[:5]):  # 只取前 5 条做扩增，控制 API 调用量
                print(f"  处理 {i+1}/5: {pair['toxic'][:50]}...")
                if args.dry_run:
                    continue
                try:
                    aug = augmenter.augment_toxic_paraphrase(
                        pair["toxic"], pair["neutral"], lang=lang, n_variants=args.n_per_pair
                    )
                    if aug:
                        for a in aug:
                            a["lang"] = lang
                        all_augmented.extend(aug)
                        print(f"    → 生成 {len(aug)} 条")
                except Exception as e:
                    print(f"    ✗ 失败: {e}")

        if args.strategy in ("all", "neutral_para"):
            print(f"\n[策略2] Neutral 同义改写 ({lang})...")
            for i, pair in enumerate(samples[:5]):
                print(f"  处理 {i+1}/5: {pair['neutral'][:50]}...")
                if args.dry_run:
                    continue
                try:
                    aug = augmenter.augment_neutral_paraphrase(
                        pair["toxic"], pair["neutral"], lang=lang, n_variants=args.n_per_pair
                    )
                    if aug:
                        for a in aug:
                            a["lang"] = lang
                        all_augmented.extend(aug)
                        print(f"    → 生成 {len(aug)} 条")
                except Exception as e:
                    print(f"    ✗ 失败: {e}")

        # ── 策略 3 ──
        if args.strategy in ("all", "new_pairs"):
            print(f"\n[策略3] 生成全新 toxic→neutral 对 ({lang})...")
            if args.dry_run:
                continue
            try:
                aug = augmenter.augment_new_pairs(
                    samples[:5], lang=lang, n_pairs=args.n_new_pairs
                )
                if aug:
                    for a in aug:
                        a["lang"] = lang
                    all_augmented.extend(aug)
                    print(f"  → 生成 {len(aug)} 条")
            except Exception as e:
                print(f"  ✗ 失败: {e}")

    # ── 策略 4：跨语言 ──
    if args.strategy in ("all", "cross_lingual"):
        print(f"\n[策略4] 跨语言数据生成（0-shot 语言: {ZERO_SHOT_LANGS}）...")
        en_samples = pool.get("en", pool.get(list(pool.keys())[0], []))
        for tgt_lang in ZERO_SHOT_LANGS:
            print(f"  生成 {tgt_lang} 数据...")
            if args.dry_run:
                continue
            try:
                aug = augmenter.augment_cross_lingual(
                    en_samples[:5], src_lang="en", tgt_lang=tgt_lang, n_pairs=args.n_cross_lingual
                )
                if aug:
                    all_augmented.extend(aug)
                    print(f"    → 生成 {len(aug)} 条")
            except Exception as e:
                print(f"    ✗ 失败: {e}")

    # ── 保存 ──
    if all_augmented:
        save_augmented(all_augmented, args.output)

    # ── 统计 ──
    print(f"\n{'='*60}")
    lang_counts = {}
    for item in all_augmented:
        lang_counts[item.get("lang", "unknown")] = lang_counts.get(item.get("lang", "unknown"), 0) + 1
    for l, c in sorted(lang_counts.items()):
        print(f"  {l}: {c} 条")
    print(f"  总计: {len(all_augmented)} 条")


if __name__ == "__main__":
    main()
