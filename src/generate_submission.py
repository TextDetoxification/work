"""
生成 Codabench 提交文件
=======================
读取测试 TSV → 批量推理去毒 → 生成提交 ZIP

使用方法:
    python -m src.generate_submission --input test.tsv --output submission.zip
    python -m src.generate_submission --input test.tsv --output submission.tsv  # 不压缩
"""

import sys
import argparse
import zipfile
import os
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import DetoxPipeline


def load_test_data(tsv_path: str) -> pd.DataFrame:
    """加载测试 TSV 文件"""
    df = pd.read_csv(tsv_path, sep="\t", encoding="utf-8")
    print(f"加载测试数据: {len(df)} 条")
    print(f"列名: {list(df.columns)}")
    print(f"语言分布:\n{df['lang'].value_counts().to_string()}")
    return df


def run_inference(df: pd.DataFrame, pipeline: DetoxPipeline) -> pd.DataFrame:
    """对每条测试数据运行去毒推理"""
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="推理中"):
        toxic_text = str(row["toxic_text"])
        lang = str(row["lang"]).strip().lower()

        try:
            result = pipeline.detoxify(toxic_text, lang, verbose=False)
            neutral = result.get("neutral_result", toxic_text)
        except Exception as e:
            print(f"  [错误] 第 {idx} 行 ({lang}): {e}")
            neutral = toxic_text  # 出错时回退到原文

        results.append({
            "toxic_text": toxic_text,
            "neutral_text": neutral,
            "lang": lang,
        })

    return pd.DataFrame(results)


def validate_submission(df: pd.DataFrame, original_df: pd.DataFrame) -> bool:
    """验证提交文件格式是否正确"""
    errors = []

    # 检查行数
    if len(df) != len(original_df):
        errors.append(f"行数不匹配: 原始 {len(original_df)} 条, 提交 {len(df)} 条")

    # 检查列
    required_cols = {"toxic_text", "neutral_text", "lang"}
    if set(df.columns) != required_cols:
        errors.append(f"列名不匹配: 需要 {required_cols}, 实际 {set(df.columns)}")

    # 检查 toxic_text 和 lang 是否与原始一致
    if len(df) == len(original_df):
        toxic_match = (df["toxic_text"] == original_df["toxic_text"]).all()
        lang_match = (df["lang"] == original_df["lang"]).all()
        if not toxic_match:
            errors.append("toxic_text 列与原始数据不一致!")
        if not lang_match:
            errors.append("lang 列与原始数据不一致!")

    # 检查 NaN
    nan_count = df["neutral_text"].isna().sum()
    if nan_count > 0:
        errors.append(f"neutral_text 包含 {nan_count} 个 NaN 值!")

    if errors:
        print("\n⚠️ 验证失败:")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("\n✅ 验证通过")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="生成 Codabench 文本去毒化提交文件"
    )
    parser.add_argument("--input", required=True, help="测试 TSV 文件路径")
    parser.add_argument("--output", required=True, help="输出文件路径 (.tsv 或 .zip)")
    parser.add_argument("--model_path", default="./mt5_detox_lora/final", help="LoRA 模型路径")
    parser.add_argument("--lexicon_cache", default="./toxic_lexicon_cache/toxic_lexicon")
    parser.add_argument("--nllb_model", default=None, help="NLLB 模型路径（可选）")
    parser.add_argument("--batch_size", type=int, default=1, help="批量大小（暂未使用）")
    parser.add_argument("--skip_validation", action="store_true", help="跳过验证")

    args = parser.parse_args()

    # 1. 加载测试数据
    original_df = load_test_data(args.input)

    # 2. 初始化推理管道
    print("\n初始化推理管道...")
    pipeline = DetoxPipeline(
        detox_model_path=args.model_path,
        lexicon_cache=args.lexicon_cache,
        nllb_model=args.nllb_model,
    )

    # 3. 批量推理
    print(f"\n开始批量推理 ({len(original_df)} 条)...")
    result_df = run_inference(original_df, pipeline)

    # 4. 验证
    if not args.skip_validation:
        validate_submission(result_df, original_df)

    # 5. 保存
    output_path = Path(args.output)
    tsv_path = output_path.with_suffix(".tsv") if output_path.suffix == ".zip" else output_path

    result_df.to_csv(tsv_path, sep="\t", index=False, encoding="utf-8")
    print(f"\nTSV 已保存: {tsv_path} ({len(result_df)} 条)")

    # 如果需要打包为 ZIP
    if output_path.suffix == ".zip":
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(tsv_path, tsv_path.name)
        print(f"ZIP 已生成: {output_path}")
        # 可选：删除临时 TSV
        # tsv_path.unlink()

    # 6. 统计
    print(f"\n{'='*60}")
    print("提交统计:")
    for lang, count in result_df["lang"].value_counts().items():
        print(f"  {lang}: {count} 条")
    print(f"  总计: {len(result_df)} 条")


if __name__ == "__main__":
    main()
