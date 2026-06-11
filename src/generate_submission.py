"""
生成 Codabench 提交文件。

使用方法:
    python -m src.generate_submission --input test.tsv --output submission.zip
    python -m src.generate_submission --input test.tsv --output submission.tsv
"""

import sys
import argparse
import zipfile
import os
from pathlib import Path
from typing import List, Dict

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.toxic_lexicon import ToxicLexicon
from src.back_translate import BackTranslator


# ── 推理管道 ──────────────────────────────────────────────────────

TRAINED_LANGUAGES = {"am", "ar", "de", "en", "es", "hi", "ru", "uk", "zh"}
ZERO_SHOT_LANGUAGES = {"fr", "he", "it", "ja", "tt", "hin"}


class DetoxPipeline:
    """
    完整去毒管道：
      - 训练语言 → mT5 直接去毒 + 有害词消除
      - 零资源语言 → NLLB 回译去毒 + 有害词消除
    """

    def __init__(self, detox_model_path="./mt5_detox_lora/final",
                 lexicon_cache="./data/toxic_lexicon",
                 nllb_model=None, device=None):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Pipeline] Device: {self.device}")

        print("[Pipeline] Loading toxic lexicon...")
        self.lexicon = ToxicLexicon(cache_dir=lexicon_cache)

        self.translator = BackTranslator(nllb_model_name=nllb_model, device=self.device)

        if Path(detox_model_path).exists():
            self.translator.load_detox_model(detox_model_path)
        else:
            print(f"[Pipeline] Warning: model not found at {detox_model_path}")

    def _classify_lang(self, lang):
        lang = lang.lower().strip()
        if lang in TRAINED_LANGUAGES:
            return "trained"
        if lang in ZERO_SHOT_LANGUAGES:
            return "zero_shot"
        short = lang[:2] if len(lang) > 2 else lang
        if short in TRAINED_LANGUAGES:
            return "trained"
        if short in ZERO_SHOT_LANGUAGES:
            return "zero_shot"
        return "unknown"

    def detoxify(self, toxic_text, lang, verbose=True):
        lang_type = self._classify_lang(lang)
        result = {"toxic_original": toxic_text, "lang": lang, "lang_type": lang_type,
                  "neutral_result": None, "toxic_words_remaining": []}

        result["toxic_words_found"] = [
            w for _, _, w in self.lexicon.detect_toxic_words(toxic_text, lang)]

        if verbose:
            print(f"\n{'='*60}")
            print(f"[去毒] lang={lang} ({lang_type})")
            print(f"  原文: {toxic_text[:150]}")

        if self.translator.detox_model is None:
            result["neutral_result"] = "[ERROR: model not loaded]"
            return result

        if lang_type == "trained":
            text = self.lexicon.mark_toxic_words(toxic_text, lang)
            for _ in range(3):
                text = self.translator.detoxify(text, lang=lang)
                if not self.lexicon.has_toxic_words(text, lang):
                    break
            result["neutral_result"] = text
        else:
            pipe_result = self.translator.pipeline(toxic_text, src_lang=lang)
            text = pipe_result["neutral_final"]
            for _ in range(3):
                if not self.lexicon.has_toxic_words(text, lang):
                    break
                text = self.translator.detoxify(self.lexicon.mark_toxic_words(text, lang), lang=lang)
            result["neutral_result"] = text
            if verbose:
                print(f"  英译: {pipe_result.get('toxic_pivot', '')[:150]}")
                print(f"  去毒: {pipe_result.get('neutral_pivot', '')[:150]}")
                print(f"  回译: {pipe_result.get('neutral_final', '')[:150]}")

        remaining = self.lexicon.detect_toxic_words(result["neutral_result"], lang)
        result["toxic_words_remaining"] = [w for _, _, w in remaining]

        if verbose:
            print(f"  最终: {result['neutral_result'][:150]}")
            if remaining:
                print(f"  ⚠ 残留: {result['toxic_words_remaining']}")
            else:
                print(f"  ✅ 清除完毕")

        return result

    def detoxify_batch(self, texts, langs, verbose=False):
        results = []
        for text, lang in zip(texts, langs):
            results.append(self.detoxify(text, lang, verbose=verbose))
        total = len(results)
        clean = sum(1 for r in results if len(r["toxic_words_remaining"]) == 0)
        print(f"\n批次: {clean}/{total} 条清除 ({100*clean/max(total,1):.1f}%)")
        return results


# ── 提交文件生成 ──────────────────────────────────────────────────

def load_test_data(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t", encoding="utf-8")
    print(f"加载测试数据: {len(df)} 条")
    print(f"列名: {list(df.columns)}")
    print(f"语言分布:\n{df['lang'].value_counts().to_string()}")
    return df


def run_inference(df, pipeline):
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="推理中"):
        toxic_text = str(row["toxic_sentence"])
        lang = str(row["lang"]).strip().lower()
        try:
            result = pipeline.detoxify(toxic_text, lang, verbose=False)
            neutral = result.get("neutral_result", toxic_text)
        except Exception as e:
            print(f"  [错误] 第 {idx} 行 ({lang}): {e}")
            neutral = toxic_text
        results.append({"toxic_text": toxic_text, "neutral_text": neutral, "lang": lang})
    return pd.DataFrame(results)


def validate_submission(df, original_df):
    errors = []
    if len(df) != len(original_df):
        errors.append(f"行数不匹配: {len(original_df)} → {len(df)}")
    required = {"toxic_text", "neutral_text", "lang"}
    if set(df.columns) != required:
        errors.append(f"列名不匹配: 需要 {required}, 实际 {set(df.columns)}")
    if len(df) == len(original_df):
        if (df["toxic_text"] != original_df["toxic_sentence"]).any():
            errors.append("toxic 列与原始不一致!")
        if (df["lang"] != original_df["lang"]).any():
            errors.append("lang 列与原始不一致!")
    nan_count = df["neutral_text"].isna().sum()
    if nan_count > 0:
        errors.append(f"neutral_text 包含 {nan_count} 个 NaN!")
    if errors:
        print("\n⚠️ 验证失败:")
        for e in errors:
            print(f"  - {e}")
        return False
    print("\n✅ 验证通过")
    return True


def main():
    parser = argparse.ArgumentParser(description="生成 Codabench 提交文件")
    parser.add_argument("--input", required=True, help="测试 TSV 路径")
    parser.add_argument("--output", required=True, help="输出路径 (.tsv 或 .zip)")
    parser.add_argument("--model_path", default="./mt5_detox_lora/final")
    parser.add_argument("--lexicon_cache", default="./data/toxic_lexicon")
    parser.add_argument("--nllb_model", default=None)
    parser.add_argument("--skip_validation", action="store_true")
    args = parser.parse_args()

    original_df = load_test_data(args.input)

    print("\n初始化推理管道...")
    pipeline = DetoxPipeline(detox_model_path=args.model_path,
                             lexicon_cache=args.lexicon_cache,
                             nllb_model=args.nllb_model)

    print(f"\n批量推理 ({len(original_df)} 条)...")
    result_df = run_inference(original_df, pipeline)

    if not args.skip_validation:
        validate_submission(result_df, original_df)

    output_path = Path(args.output)
    tsv_path = output_path.with_suffix(".tsv") if output_path.suffix == ".zip" else output_path
    result_df.to_csv(tsv_path, sep="\t", index=False, encoding="utf-8")
    print(f"\nTSV 已保存: {tsv_path} ({len(result_df)} 条)")

    if output_path.suffix == ".zip":
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(tsv_path, tsv_path.name)
        print(f"ZIP 已生成: {output_path}")

    print(f"\n{'='*60}")
    print("统计:")
    for lang, count in result_df["lang"].value_counts().items():
        print(f"  {lang}: {count} 条")
    print(f"  总计: {len(result_df)} 条")


if __name__ == "__main__":
    main()
