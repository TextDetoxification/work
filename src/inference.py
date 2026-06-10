"""
多语言文本去毒化推理管道
Inference Pipeline: 回译 → mT5去毒 → 翻译回 → 有害词消除

使用方法:
    python -m src.inference --text "有毒文本" --lang zh
    python -m src.inference --input_file toxic.txt --output_file clean.txt
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.toxic_lexicon import ToxicLexicon
from src.back_translate import BackTranslator


# ── 9 种训练语言（可直接去毒，无需回译） ──
TRAINED_LANGUAGES = {"am", "ar", "de", "en", "es", "hi", "ru", "uk", "zh"}

# ── 6 种零资源语言（需要回译管道） ──
ZERO_SHOT_LANGUAGES = {"fr", "he", "it", "ja", "tt", "hin"}


class DetoxPipeline:
    """
    完整去毒管道：
    1. 语言检测 / 分类
    2. 对训练语言 → 直接 mT5 去毒 + 有害词消除
    3. 对零资源语言 → 回译去毒 + 有害词消除
    """

    def __init__(
        self,
        detox_model_path: str = "./mt5_detox_lora/final",
        lexicon_cache: str = "./toxic_lexicon_cache/toxic_lexicon",
        nllb_model: str = None,  # None=自动先查本地再查 Hub
        device: str = None,
    ):
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Pipeline] Device: {self.device}")

        # 加载有害词表
        print("[Pipeline] Loading toxic lexicon...")
        self.lexicon = ToxicLexicon(cache_dir=lexicon_cache)
        self.lexicon.build_toxic_neutral_map(parallel_data_dir="./data")

        # 加载回译器（NLLB + mT5）
        self.translator = BackTranslator(
            nllb_model_name=nllb_model,
            device=self.device,
        )

        # 加载去毒模型
        if Path(detox_model_path).exists():
            self.translator.load_detox_model(detox_model_path)
        else:
            print(f"[Pipeline] Warning: Detox model not found at {detox_model_path}")
            print("  Please train the model first: python -m src.train_mt5")

    def _classify_lang(self, lang: str) -> str:
        """判断语言是训练语言还是零资源语言"""
        lang = lang.lower().strip()
        if lang in TRAINED_LANGUAGES:
            return "trained"
        elif lang in ZERO_SHOT_LANGUAGES:
            return "zero_shot"
        else:
            # 尝试映射
            from src.back_translate import LANG_TO_NLLB
            short = lang[:2] if len(lang) > 2 else lang
            if short in TRAINED_LANGUAGES:
                return "trained"
            elif short in ZERO_SHOT_LANGUAGES:
                return "zero_shot"
            return "unknown"

    def detoxify(
        self,
        toxic_text: str,
        lang: str,
        verbose: bool = True,
    ) -> Dict:
        """
        单条文本去毒。

        Args:
            toxic_text: 有毒文本
            lang: 语言代码
            verbose: 是否打印中间结果

        Returns:
            {
                "toxic_original": ...,
                "lang": ...,
                "lang_type": "trained" | "zero_shot",
                "neutral_result": ...,
                "pipeline_steps": [...],
                "toxic_words_found": [...],
                "toxic_words_remaining": [...],
            }
        """
        lang_type = self._classify_lang(lang)
        result = {
            "toxic_original": toxic_text,
            "lang": lang,
            "lang_type": lang_type,
            "neutral_result": None,
            "pipeline_steps": [],
            "toxic_words_found": [],
            "toxic_words_remaining": [],
        }

        # 检测有害词
        result["toxic_words_found"] = [
            w for _, _, w in self.lexicon.detect_toxic_words(toxic_text, lang)
        ]

        if verbose:
            print(f"\n{'='*60}")
            print(f"[去毒] lang={lang} ({lang_type})")
            print(f"  原文: {toxic_text[:150]}")
            print(f"  原始有害词: {result['toxic_words_found']}")

        if self.translator.detox_model is None:
            result["neutral_result"] = "[ERROR: detox model not loaded]"
            return result

        if lang_type == "trained":
            # ── 直接去毒（训练语言） ──
            # 标记有害词增强输入
            marked_input = self.lexicon.mark_toxic_words(toxic_text, lang)
            detoxed = self.translator.detoxify(marked_input, lang=lang)
            result["pipeline_steps"].append("direct_detox")

            if verbose:
                print(f"  直接去毒: {detoxed[:150]}")

            # 有害词消除
            cleaned = self.lexicon.auto_eliminate(
                detoxed, lang,
                model=self.translator.detox_model,
                tokenizer=self.translator.detox_tokenizer,
            )
            result["neutral_result"] = cleaned
            result["pipeline_steps"].append("toxic_elimination")

        else:
            # ── 回译去毒（零资源语言） ──
            pipeline_result = self.translator.pipeline_with_toxic_filter(
                toxic_text, src_lang=lang,
                toxic_lexicon=self.lexicon,
                pivot_lang="en",
            )
            result["pipeline_steps"].extend([
                "translate_to_en",
                "detox_en",
                "translate_back",
                "toxic_elimination",
            ])
            result["neutral_result"] = pipeline_result.get("neutral_final_cleaned", pipeline_result["neutral_final"])
            result["_pipeline_detail"] = pipeline_result

            if verbose:
                print(f"  英译:   {pipeline_result.get('toxic_pivot', '')[:150]}")
                print(f"  去毒:   {pipeline_result.get('neutral_pivot', '')[:150]}")
                print(f"  回译:   {pipeline_result.get('neutral_final', '')[:150]}")

        # 检查残留
        remaining = self.lexicon.detect_toxic_words(result["neutral_result"], lang)
        result["toxic_words_remaining"] = [w for _, _, w in remaining]

        if verbose:
            print(f"  最终结果: {result['neutral_result'][:150]}")
            if remaining:
                print(f"  ⚠ 残留有害词: {result['toxic_words_remaining']}")
            else:
                print(f"  ✅ 全部清除")

        return result

    def detoxify_batch(
        self,
        texts: List[str],
        langs: List[str],
        verbose: bool = False,
    ) -> List[Dict]:
        """批量去毒"""
        results = []
        for text, lang in zip(texts, langs):
            result = self.detoxify(text, lang, verbose=verbose)
            results.append(result)

        # 统计
        total = len(results)
        clean = sum(1 for r in results if len(r["toxic_words_remaining"]) == 0)
        print(f"\n{'='*60}")
        print(f"批次处理完成: {clean}/{total} 条完全清除有害词 ({100*clean/max(total,1):.1f}%)")
        return results


# ── 命令行接口 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="多语言文本去毒化推理管道"
    )
    parser.add_argument("--text", type=str, help="要去毒的文本")
    parser.add_argument("--lang", type=str, default="en", help="语言代码")
    parser.add_argument("--input_file", type=str, help="输入文件（每行一条）")
    parser.add_argument("--output_file", type=str, help="输出文件")
    parser.add_argument("--model_path", type=str, default="./mt5_detox_lora/final")
    parser.add_argument("--quiet", action="store_true", help="安静模式")

    args = parser.parse_args()

    # 初始化管道
    pipeline = DetoxPipeline(detox_model_path=args.model_path)

    if args.text:
        # 单条推理
        result = pipeline.detoxify(args.text, args.lang, verbose=not args.quiet)
        print(f"\n结果: {result['neutral_result']}")

        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

    elif args.input_file:
        # 批量推理
        with open(args.input_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        # 简单格式：每行 "lang<TAB>text" 或只有 text（默认英文）
        langs = []
        texts = []
        for line in lines:
            if "\t" in line:
                lang, text = line.split("\t", 1)
            else:
                lang, text = "en", line
            langs.append(lang)
            texts.append(text)

        results = pipeline.detoxify_batch(texts, langs, verbose=not args.quiet)

        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"结果已保存到 {args.output_file}")


if __name__ == "__main__":
    main()
