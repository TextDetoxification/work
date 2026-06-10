from typing import Optional, Dict, List
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

LANG_TO_NLLB = {
    "am": "amh_Ethi", "ar": "arb_Arab", "de": "deu_Latn", "en": "eng_Latn",
    "es": "spa_Latn", "hi": "hin_Deva", "ru": "rus_Cyrl", "uk": "ukr_Cyrl",
    "zh": "zho_Hans", "fr": "fra_Latn", "he": "heb_Hebr", "it": "ita_Latn",
    "ja": "jpn_Jpan", "tt": "tat_Cyrl", "hin": "hin_Deva",
}
NLLB_TO_SHORT = {v: k for k, v in LANG_TO_NLLB.items()}


class BackTranslator:
    DEFAULT_NLLB_LOCAL = "./models/nllb-200-distilled-600M"
    DEFAULT_NLLB_HUB = "facebook/nllb-200-distilled-600M"
    DEFAULT_MT5_LOCAL = "./models/mt5-base"
    DEFAULT_MT5_HUB = "google/mt5-base"

    def __init__(self, nllb_model_name=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if nllb_model_name is None:
            nllb_model_name = self.DEFAULT_NLLB_LOCAL if Path(self.DEFAULT_NLLB_LOCAL).exists() else self.DEFAULT_NLLB_HUB
        self.trans_tokenizer = AutoTokenizer.from_pretrained(nllb_model_name, src_lang="eng_Latn", tgt_lang="eng_Latn")
        self.trans_model = AutoModelForSeq2SeqLM.from_pretrained(
            nllb_model_name,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            use_safetensors=True).to(self.device).eval()
        self.detox_model = self.detox_tokenizer = None

    @staticmethod
    def to_nllb_code(lang):
        lang = lang.lower().strip()
        return LANG_TO_NLLB[lang] if lang in LANG_TO_NLLB else (lang if lang in LANG_TO_NLLB.values() else None)

    def translate(self, text, src_lang, tgt_lang, max_length=512, num_beams=4):
        src_code = self.to_nllb_code(src_lang)
        tgt_code = self.to_nllb_code(tgt_lang)
        self.trans_tokenizer.src_lang = src_code
        inputs = self.trans_tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)
        # NLLB forced_bos_token_id: 兼容不同 transformers 版本
        try:
            forced_bos = self.trans_tokenizer.lang_code_to_id[tgt_code]
        except AttributeError:
            forced_bos = self.trans_tokenizer.convert_tokens_to_ids(tgt_code)
        with torch.no_grad():
            out = self.trans_model.generate(**inputs, forced_bos_token_id=forced_bos,
                                            max_length=max_length, num_beams=num_beams, early_stopping=True)
        return self.trans_tokenizer.decode(out[0], skip_special_tokens=True)

    def load_detox_model(self, model_path, base_model_path=None):
        from peft import PeftModel
        if base_model_path is None:
            base_model_path = self.DEFAULT_MT5_LOCAL if Path(self.DEFAULT_MT5_LOCAL).exists() else self.DEFAULT_MT5_HUB
        self.detox_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            use_safetensors=True)
        self.detox_model = PeftModel.from_pretrained(base_model, model_path).merge_and_unload()
        self.detox_model.to(self.device).eval()

    def detoxify(self, toxic_text, lang="en"):
        inputs = self.detox_tokenizer(f"detoxify: {toxic_text}", return_tensors="pt",
                                       truncation=True, max_length=256).to(self.device)
        with torch.no_grad():
            out = self.detox_model.generate(**inputs, max_new_tokens=256, num_beams=5,
                                            early_stopping=True, no_repeat_ngram_size=3)
        return self.detox_tokenizer.decode(out[0], skip_special_tokens=True)

    def pipeline(self, toxic_text, src_lang, pivot_lang="en"):
        result = {"toxic_original": toxic_text}
        result["toxic_pivot"] = toxic_text if src_lang == pivot_lang else self.translate(toxic_text, src_lang, pivot_lang)
        result["neutral_pivot"] = self.detoxify(result["toxic_pivot"])
        result["neutral_final"] = result["neutral_pivot"] if src_lang == pivot_lang else self.translate(result["neutral_pivot"], pivot_lang, src_lang)
        return result

    def pipeline_with_toxic_filter(self, toxic_text, src_lang, toxic_lexicon=None, pivot_lang="en"):
        result = self.pipeline(toxic_text, src_lang, pivot_lang)
        if toxic_lexicon and toxic_lexicon.has_toxic_words(result["neutral_final"], src_lang):
            result["neutral_final_cleaned"] = toxic_lexicon.auto_eliminate(
                result["neutral_final"], src_lang, model=self.detox_model, tokenizer=self.detox_tokenizer)
        else:
            result["neutral_final_cleaned"] = result["neutral_final"]
        return result
