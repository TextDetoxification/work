"""
多语言有害词汇查询表 & 残留消除
Component 3: Toxic Lexicon Lookup + Elimination

数据源: textdetox/multilingual_toxic_lexicon
覆盖语言: am, ar, de, en, es, fr, he, hi, hin, it, ja, ru, tt, uk, zh (15语)
"""

import re
import os
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from datasets import load_dataset, load_from_disk

# ── 语言代码映射 ──────────────────────────────────────────────
# NLLB 语言代码 → 数据集语言代码
NLLB_LANG_MAP = {
    "amh_Ethi": "am",   # 阿姆哈拉语
    "arb_Arab": "ar",   # 阿拉伯语
    "deu_Latn": "de",   # 德语
    "eng_Latn": "en",   # 英语
    "spa_Latn": "es",   # 西班牙语
    "fra_Latn": "fr",   # 法语
    "heb_Hebr": "he",   # 希伯来语
    "hin_Deva": "hi",   # 印地语
    "hin_Latn": "hi",   # Hinglish（拉丁化印地语，复用 hi 词表）
    "ita_Latn": "it",   # 意大利语
    "jpn_Jpan": "ja",   # 日语
    "rus_Cyrl": "ru",   # 俄语
    "tat_Cyrl": "tt",   # 鞑靼语
    "ukr_Cyrl": "uk",   # 乌克兰语
    "zho_Hans": "zh",   # 中文（简体）
}

# 支持的短语言代码
SHORT_LANG_MAP = {
    "am": "am", "ar": "ar", "de": "de", "en": "en", "es": "es",
    "fr": "fr", "he": "he", "hi": "hi", "hin": "hi", "it": "it",
    "ja": "ja", "ru": "ru", "tt": "tt", "uk": "uk", "zh": "zh",
}


class ToxicLexicon:
    """多语言有害词汇表，支持检测、标记和消除"""

    def __init__(self, cache_dir: str = "./toxic_lexicon_cache/toxic_lexicon"):
        self.cache_dir = Path(cache_dir)
        self.toxic_sets: Dict[str, Set[str]] = {}  # lang_code -> {word1, word2, ...}
        self.toxic_neutral_map: Dict[str, Dict[str, str]] = {}  # lang -> {toxic_word: neutral_word}
        self._load_or_download()

    # ── 加载 ──────────────────────────────────────────────────

    def _load_or_download(self):
        """从本地缓存加载，若不存在则从 HuggingFace 下载"""
        if self.cache_dir.exists():
            ds = load_from_disk(str(self.cache_dir))
        else:
            print("Downloading toxic lexicon from HuggingFace...")
            ds = load_dataset("textdetox/multilingual_toxic_lexicon")
            self.cache_dir.parent.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(self.cache_dir))

        for lang in ds:
            words = set()
            for row in ds[lang]:
                word = row["text"].strip().lower()
                if word:  # 过滤空字符串
                    words.add(word)
            self.toxic_sets[lang] = words
            print(f"  [{lang}] {len(words):>6} 个有害词")

        total = sum(len(s) for s in self.toxic_sets.values())
        print(f"  总计: {total} 个有害词, {len(self.toxic_sets)} 种语言")

    # ── 语言代码归一化 ────────────────────────────────────────

    def _normalize_lang(self, lang: str) -> Optional[str]:
        """将 NLLB 代码或短代码归一化为数据集语言代码"""
        if lang in self.toxic_sets:
            return lang
        if lang in NLLB_LANG_MAP:
            return NLLB_LANG_MAP[lang]
        if lang in SHORT_LANG_MAP:
            return SHORT_LANG_MAP[lang]
        return None

    # ── 检测 ──────────────────────────────────────────────────

    def detect_toxic_words(self, text: str, lang: str) -> List[Tuple[int, int, str]]:
        """
        检测文本中的有害词。

        Args:
            text: 输入文本
            lang: 语言代码 (如 'en', 'zh', 'eng_Latn')

        Returns:
            List of (start_pos, end_pos, toxic_word) 按出现位置排序
        """
        lang = self._normalize_lang(lang)
        if lang is None or lang not in self.toxic_sets:
            return []

        toxic_set = self.toxic_sets[lang]
        text_lower = text.lower()
        found = []

        for word in toxic_set:
            # 对非字母语言（中文、日语等）使用子串匹配
            if lang in ("zh", "ja"):
                pos = text.find(word)
                if pos != -1:
                    found.append((pos, pos + len(word), word))
            else:
                # 对字母语言使用词边界匹配，避免误匹配
                pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                for m in pattern.finditer(text):
                    found.append((m.start(), m.end(), word))

        # 按位置排序，移除嵌套匹配（保留最长的）
        found.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        filtered = []
        last_end = -1
        for start, end, word in found:
            if start >= last_end:
                filtered.append((start, end, word))
                last_end = end
        return filtered

    def has_toxic_words(self, text: str, lang: str) -> bool:
        """检测文本是否含有害词"""
        return len(self.detect_toxic_words(text, lang)) > 0

    # ── 标记 ──────────────────────────────────────────────────

    def mark_toxic_words(self, text: str, lang: str) -> str:
        """
        用 <toxic>...</toxic> 标签包裹检测到的有害词。
        用于增强 mT5 的输入。
        """
        found = self.detect_toxic_words(text, lang)
        if not found:
            return text

        # 从后往前替换，避免位置偏移
        result = text
        for start, end, word in reversed(found):
            result = result[:start] + f"<toxic>{result[start:end]}</toxic>" + result[end:]
        return result

    # ── 有害词→中性词映射 ────────────────────────────────────

    def build_toxic_neutral_map(self, parallel_data_dir: str = "./data"):
        """
        从训练数据中提取 (有害词, 中性词) 映射。
        通过对比 toxic 和 neutral 句子的差异来发现替换模式。

        这是个简化版：统计哪些词在 toxic 句中出现但在 neutral 句中消失，
        以及 neutral 句中新出现的词。
        """
        import pandas as pd
        from pathlib import Path

        data_dir = Path(parallel_data_dir)
        if not data_dir.exists():
            print(f"  数据目录 {data_dir} 不存在，跳过映射构建")
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

                # 找到同时出现在 toxic 文本和有害词表中的词
                toxic_words_in_text = set()
                for word in self.toxic_sets[lang]:
                    if lang in ("zh", "ja"):
                        if word in toxic_text:
                            toxic_words_in_text.add(word)
                    else:
                        if re.search(r'\b' + re.escape(word) + r'\b', toxic_text, re.IGNORECASE):
                            toxic_words_in_text.add(word)

                for tw in toxic_words_in_text:
                    # 查找在 neutral 中对应的替换：找到位置相近的词
                    # 简化处理：如果 neutral 句中不包含该有害词，记录为可删除
                    if lang in ("zh", "ja"):
                        if tw not in neutral_text:
                            mapping[tw] = ""
                    else:
                        if not re.search(r'\b' + re.escape(tw) + r'\b', neutral_text, re.IGNORECASE):
                            mapping[tw] = ""

            self.toxic_neutral_map[lang] = mapping
            print(f"  [{lang}] {len(mapping)} 个有害词→空 映射")

    # ── 消除策略 ──────────────────────────────────────────────

    def eliminate_toxic(
        self,
        text: str,
        lang: str,
        strategy: str = "replace",
        model=None,
        tokenizer=None,
    ) -> Tuple[str, bool]:
        """
        消除文本中残留的有害词。

        Args:
            text: 待处理的文本
            lang: 语言代码
            strategy: 消除策略
                - 'replace': 用训练数据中的映射直接替换（策略1）
                - 'delete': 直接删除有害词（策略3）
                - 'mask_fill': 用 <extra_id_0> 遮蔽后用 mT5 填空（策略2，需传 model/tokenizer）
            model: mT5 模型（仅在 strategy='mask_fill' 时需要）
            tokenizer: mT5 tokenizer（仅在 strategy='mask_fill' 时需要）

        Returns:
            (cleaned_text, was_modified)
        """
        lang = self._normalize_lang(lang)
        if lang is None:
            return text, False

        found = self.detect_toxic_words(text, lang)
        if not found:
            return text, False

        if strategy == "delete":
            # 策略3：直接删除有害词
            result = text
            for start, end, _ in reversed(found):
                # 删除时注意空格处理
                prefix = result[:start].rstrip()
                suffix = result[end:].lstrip()
                if prefix and suffix and not prefix[-1].isspace() and not suffix[0].isspace():
                    result = result[:start] + " " + result[end:]
                else:
                    result = result[:start] + result[end:]
            return result, True

        elif strategy == "replace":
            # 策略1：用映射表替换
            if lang not in self.toxic_neutral_map:
                return text, False

            result = text
            modified = False
            for start, end, word in reversed(found):
                word_lower = word.lower()
                if word_lower in self.toxic_neutral_map[lang]:
                    replacement = self.toxic_neutral_map[lang][word_lower]
                    if replacement:
                        result = result[:start] + replacement + result[end:]
                    else:
                        # 映射为空 = 删除
                        result = result[:start].rstrip() + " " + result[end:].lstrip()
                    modified = True
            return result.strip(), modified

        elif strategy == "mask_fill":
            # 策略2：遮蔽有害词后用 mT5 填空
            if model is None or tokenizer is None:
                raise ValueError("mask_fill strategy requires model and tokenizer")

            # 用 <extra_id_0> 遮蔽每个有害词
            masked_text = text
            for i, (start, end, _) in enumerate(reversed(found)):
                masked_text = masked_text[:start] + f"<extra_id_{i}>" + masked_text[end:]

            # 限制最多遮蔽 5 个词
            if len(found) > 5:
                return text, False

            inputs = tokenizer(masked_text, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=3,
                early_stopping=True,
            )
            filled = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return filled, True

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    # ── 自适应消除 ────────────────────────────────────────────

    def auto_eliminate(
        self,
        text: str,
        lang: str,
        model=None,
        tokenizer=None,
    ) -> str:
        """
        按优先级依次尝试消除策略：
        replace → delete → mask_fill（如有模型）
        """
        # 1) 先尝试映射替换
        cleaned, modified = self.eliminate_toxic(text, lang, strategy="replace")
        if not self.has_toxic_words(cleaned, lang):
            return cleaned

        # 2) 仍有残留 → 删除
        cleaned, modified = self.eliminate_toxic(cleaned, lang, strategy="delete")
        if not self.has_toxic_words(cleaned, lang):
            return cleaned

        # 3) 还有残留且可用模型 → mask_fill
        if model is not None:
            cleaned, modified = self.eliminate_toxic(
                cleaned, lang, strategy="mask_fill", model=model, tokenizer=tokenizer
            )

        return cleaned


# ── 快速测试 ──────────────────────────────────────────────────
if __name__ == "__main__":
    lex = ToxicLexicon()
    print(f"\n{'='*50}")
    print("测试有毒词检测与消除")
    print("=" * 50)

    tests = [
        ("en", "you are such an idiot and a stupid fool"),
        ("zh", "你真是个白痴，太蠢了"),
        ("de", "du bist so ein idiot"),
        ("ru", "ты такой дурак"),
    ]

    for lang, text in tests:
        print(f"\n[{lang}] 原文: {text}")
        found = lex.detect_toxic_words(text, lang)
        print(f"  检测到有害词: {[(text[s:e], w) for s, e, w in found]}")
        marked = lex.mark_toxic_words(text, lang)
        print(f"  标记后: {marked}")
        cleaned, mod = lex.eliminate_toxic(text, lang, strategy="delete")
        print(f"  删除后: {cleaned}  (修改={mod})")
