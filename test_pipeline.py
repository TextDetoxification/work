# -*- coding: utf-8 -*-
"""测试完整去毒链路（训练语言 + 零资源语言）"""
import sys
sys.path.insert(0, '.')

from src.generate_submission import DetoxPipeline

pipeline = DetoxPipeline(
    detox_model_path="./mt0_detox_lora/final",
    lexicon_cache="./data/toxic_lexicon",
    device="cuda"
)

tests = [
    # ── 中文测试（训练语言，9条）──
    ("zh", "他妈的这破游戏简直就是垃圾，开发者都是傻逼吧？"),
    ("zh", "你这种人就该滚出论坛，除了喷粪什么都不会。"),
    ("zh", "你们这些脑残粉能不能消停会儿，到处刷存在感。"),
    ("zh", "我操，这剧情简直弱智到了极点，编剧吃屎了吧。"),
    ("zh", "废物一个，连这么简单的事都做不好。"),
    ("zh", "别理那个贱人，整天在背后说人坏话。"),
    ("zh", "这电影烂得令人发指，白白浪费我两小时。"),
    ("zh", "你算什么东西，也配在这里指手画脚？"),
    ("zh", "一群傻吊，天天在网上瞎逼逼。"),
]

print("=" * 65)
print("训练语言（直接去毒）".center(65))
print("=" * 65)

for lang, text in tests:
    result = pipeline.detoxify(text, lang, verbose=False)
    print(f"[{lang}] 有毒: {text}")
    print(f"[{lang}] 结果: {result['neutral_result']}")
    if result['toxic_words_remaining']:
        print(f"[{lang}] 残留: {result['toxic_words_remaining']}")
    else:
        print(f"[{lang}] 状态: 清除完毕")
    print()

print("=" * 65)
print("完成。满意后运行生成提交:")
print("python -m src.generate_submission --input test_inputs_upd.tsv --output submission.zip")
