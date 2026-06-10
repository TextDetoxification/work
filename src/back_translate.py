"""
回译管道（Back-translation Pipeline）
Component 2: X → EN → Detox → EN → X

使用 NLLB-200 作为翻译引擎，覆盖全部 15 种语言。
"""

from typing import Optional, Dict, List
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)


# ── 语言代码映射 ──────────────────────────────────────────────
# 短代码 / 语言名 → NLLB-200 FLORES-200 代码
# https://github.com/facebookresearch/flores/blob/main/flores200/README.md
LANG_TO_NLLB = {
    # 9 种训练语言
    "am": "amh_Ethi", "amh": "amh_Ethi",
    "ar": "arb_Arab", "ara": "arb_Arab",
    "de": "deu_Latn", "deu": "deu_Latn", "ger": "deu_Latn",
    "en": "eng_Latn", "eng": "eng_Latn",
    "es": "spa_Latn", "spa": "spa_Latn",
    "hi": "hin_Deva", "hin": "hin_Deva",
    "ru": "rus_Cyrl", "rus": "rus_Cyrl",
    "uk": "ukr_Cyrl", "ukr": "ukr_Cyrl",
    "zh": "zho_Hans", "zho": "zho_Hans", "chi": "zho_Hans",
    # 6 种零资源语言
    "fr": "fra_Latn", "fra": "fra_Latn", "fre": "fra_Latn",
    "he": "heb_Hebr", "heb": "heb_Hebr",
    "it": "ita_Latn", "ita": "ita_Latn",
    "ja": "jpn_Jpan", "jpn": "jpn_Jpan",
    "tt": "tat_Cyrl", "tat": "tat_Cyrl",
    "hin": "hin_Deva",  # Hinglish → 印地语的拉丁化，用印地语翻译
}

# 从 NLLB 代码反查短代码
NLLB_TO_SHORT = {v: k for k, v in LANG_TO_NLLB.items()}


class BackTranslator:
    """
    回译去毒管道：
    1. 源语言 X → 英语 (NLLB)
    2. 英语 toxic → 英语 neutral (mT5)
    3. 英语 neutral → 源语言 X (NLLB)
    """

    # NLLB 默认路径（本地优先）
    DEFAULT_NLLB_LOCAL = "./models/nllb-200-distilled-600M"
    DEFAULT_NLLB_HUB = "facebook/nllb-200-distilled-600M"

    def __init__(
        self,
        nllb_model_name: str = None,
        device: str = None,
    ):
        """
        Args:
            nllb_model_name: NLLB-200 模型路径（本地目录）或 HF Hub 名称。
                             留空时自动先查本地再查 Hub。
            device: 'cuda', 'cpu', 或 None（自动选择）
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[BackTranslator] Device: {self.device}")

        # 模型路径解析：本地优先
        if nllb_model_name is None:
            if Path(self.DEFAULT_NLLB_LOCAL).exists():
                nllb_model_name = self.DEFAULT_NLLB_LOCAL
            else:
                nllb_model_name = self.DEFAULT_NLLB_HUB
        print(f"[BackTranslator] Loading NLLB model: {nllb_model_name}")

        self.trans_tokenizer = AutoTokenizer.from_pretrained(
            nllb_model_name, src_lang="eng_Latn", tgt_lang="eng_Latn"
        )
        self.trans_model = AutoModelForSeq2SeqLM.from_pretrained(
            nllb_model_name, dtype=torch.float16 if "cuda" in self.device else torch.float32,
            use_safetensors=True,
        ).to(self.device)
        self.trans_model.eval()
        print("[BackTranslator] NLLB model loaded.")

        # mT5 去毒模型（后续通过 load_detox_model 注入）
        self.detox_model = None
        self.detox_tokenizer = None

    # ── 语言代码工具 ──────────────────────────────────────────

    @staticmethod
    def to_nllb_code(lang: str) -> str:
        """将任意语言代码转为 NLLB FLORES-200 代码"""
        lang = lang.lower().strip()
        if lang in LANG_TO_NLLB:
            return LANG_TO_NLLB[lang]
        if lang in LANG_TO_NLLB.values():
            return lang
        raise ValueError(f"Unknown language: {lang}. Supported: {list(LANG_TO_NLLB.keys())}")

    @staticmethod
    def to_short_code(nllb_code: str) -> str:
        """NLLB 代码 → 短代码"""
        return NLLB_TO_SHORT.get(nllb_code, nllb_code)

    def supported_languages(self) -> List[str]:
        """返回所有支持的语言代码"""
        return sorted(set(LANG_TO_NLLB.keys()))

    # ── 翻译 ──────────────────────────────────────────────────

    def translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        max_length: int = 512,
        num_beams: int = 4,
    ) -> str:
        """
        用 NLLB-200 翻译单条文本。

        Args:
            text: 源文本
            src_lang: 源语言代码（短代码或 NLLB 代码）
            tgt_lang: 目标语言代码
            max_length: 最大生成长度
            num_beams: beam search 数量

        Returns:
            翻译后的文本
        """
        src_code = self.to_nllb_code(src_lang)
        tgt_code = self.to_nllb_code(tgt_lang)

        # NLLB tokenizer 需要设置源语言
        self.trans_tokenizer.src_lang = src_code
        inputs = self.trans_tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)

        # 获取目标语言的 forced_bos_token_id
        forced_bos_token_id = self.trans_tokenizer.lang_code_to_id.get(tgt_code)
        if forced_bos_token_id is None:
            raise ValueError(f"No token ID for target language: {tgt_code}")

        with torch.no_grad():
            outputs = self.trans_model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
            )

        translated = self.trans_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated

    def translate_batch(
        self,
        texts: List[str],
        src_lang: str,
        tgt_lang: str,
        batch_size: int = 8,
        max_length: int = 512,
    ) -> List[str]:
        """批量翻译"""
        src_code = self.to_nllb_code(src_lang)
        tgt_code = self.to_nllb_code(tgt_lang)

        self.trans_tokenizer.src_lang = src_code
        forced_bos_token_id = self.trans_tokenizer.lang_code_to_id.get(tgt_code)

        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.trans_tokenizer(
                batch, return_tensors="pt", truncation=True, max_length=max_length, padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.trans_model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True,
                )

            results.extend(
                self.trans_tokenizer.decode(o, skip_special_tokens=True) for o in outputs
            )
        return results

    # ── mT5 去毒模型 ──────────────────────────────────────────

    # mT5 默认路径（本地优先）
    DEFAULT_MT5_LOCAL = "./models/mt5-base"
    DEFAULT_MT5_HUB = "google/mt5-base"

    def load_detox_model(self, model_path: str, base_model_path: str = None):
        """
        加载微调后的 mT5 LoRA 去毒模型。

        Args:
            model_path: LoRA adapter 权重路径
            base_model_path: 基座模型路径（本地目录）或 HF Hub 名称。
                             留空时自动先查本地再查 Hub。
        """
        from peft import PeftModel, PeftConfig
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        print(f"[BackTranslator] Loading detox model from {model_path}")

        # 基座模型路径解析：本地优先
        if base_model_path is None:
            if Path(self.DEFAULT_MT5_LOCAL).exists():
                base_model_path = self.DEFAULT_MT5_LOCAL
            else:
                base_model_path = self.DEFAULT_MT5_HUB
        print(f"[BackTranslator] Base model: {base_model_path}")

        self.detox_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_path,
            dtype=torch.float16 if "cuda" in self.device else torch.float32,
            use_safetensors=True,
        )
        self.detox_model = PeftModel.from_pretrained(base_model, model_path)
        self.detox_model = self.detox_model.merge_and_unload()  # 合并 LoRA 权重以加速推理
        self.detox_model.to(self.device)
        self.detox_model.eval()
        print("[BackTranslator] Detox model loaded.")

    def detoxify(self, toxic_text: str, lang: str = "en") -> str:
        """
        使用 mT5 对有毒文本进行去毒。

        Args:
            toxic_text: 有毒文本
            lang: 语言代码（用于决定是否需要翻译）

        Returns:
            去毒后的文本
        """
        if self.detox_model is None:
            raise RuntimeError("Detox model not loaded. Call load_detox_model() first.")

        input_text = f"detoxify: {toxic_text}"
        inputs = self.detox_tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=256
        ).to(self.device)

        with torch.no_grad():
            outputs = self.detox_model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        return self.detox_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ── 回译去毒管道 ──────────────────────────────────────────

    def pipeline(
        self,
        toxic_text: str,
        src_lang: str,
        pivot_lang: str = "en",
        skip_translation: bool = False,
    ) -> Dict[str, str]:
        """
        完整的回译去毒管道。

        Args:
            toxic_text: 原始有毒文本
            src_lang: 源语言代码
            pivot_lang: 枢纽语言（默认英语）
            skip_translation: 如果源语言=枢纽语言，跳过翻译步骤

        Returns:
            {
                "src_lang": 源语言,
                "toxic_original": 原始有毒文本,
                "toxic_pivot": 翻译到枢纽语言后的 toxic 文本,
                "neutral_pivot": 枢纽语言下的去毒结果,
                "neutral_final": 翻译回源语言后的最终结果,
            }
        """
        src_nllb = self.to_nllb_code(src_lang)
        pivot_nllb = self.to_nllb_code(pivot_lang)

        result = {
            "src_lang": src_lang,
            "toxic_original": toxic_text,
            "toxic_pivot": None,
            "neutral_pivot": None,
            "neutral_final": None,
        }

        # Step 1: 翻译到枢纽语言
        if skip_translation or src_lang == pivot_lang:
            result["toxic_pivot"] = toxic_text
        else:
            result["toxic_pivot"] = self.translate(
                toxic_text, src_lang=src_lang, tgt_lang=pivot_lang
            )

        # Step 2: 在枢纽语言上去毒
        result["neutral_pivot"] = self.detoxify(result["toxic_pivot"], lang=pivot_lang)

        # Step 3: 翻译回源语言
        if skip_translation or src_lang == pivot_lang:
            result["neutral_final"] = result["neutral_pivot"]
        else:
            result["neutral_final"] = self.translate(
                result["neutral_pivot"], src_lang=pivot_lang, tgt_lang=src_lang
            )

        return result

    def pipeline_with_toxic_filter(
        self,
        toxic_text: str,
        src_lang: str,
        toxic_lexicon=None,  # ToxicLexicon 实例
        pivot_lang: str = "en",
    ) -> Dict[str, str]:
        """
        回译去毒 + 有害词后处理。

        在回译完成后，检查最终结果是否仍含有害词，
        如有则调用消除策略。
        """
        result = self.pipeline(toxic_text, src_lang=src_lang, pivot_lang=pivot_lang)

        # Step 4: 有害词检测与消除
        if toxic_lexicon is not None:
            final = result["neutral_final"]
            if toxic_lexicon.has_toxic_words(final, src_lang):
                final = toxic_lexicon.auto_eliminate(
                    final, src_lang,
                    model=self.detox_model,
                    tokenizer=self.detox_tokenizer,
                )
                result["neutral_final_cleaned"] = final
            else:
                result["neutral_final_cleaned"] = final

        return result


# ── 快速测试 ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("BackTranslator module loaded.")
    print("Supported languages:", BackTranslator(None).supported_languages() if False else list(LANG_TO_NLLB.keys()))
