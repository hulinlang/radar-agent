"""Agent 提示词 —— **唯一来源**（P6）。

⚠️ 零样本基线探针（`scripts/p6_probe_zeroshot.py`）与 ReAct 主循环（`react.py`）
必须共用这里的同一份 SYSTEM / FEWSHOT。
两处各写一份 = 基线数字与端到端数字**不可比**，而"相对基线提升多少"正是 P5b 的验收口径。

内容改动会让历史基线作废 —— 改之前先想清楚是否需要重跑基线。
"""

from __future__ import annotations

import json
from typing import Any

SYSTEM = (
    "你是雷达领域的问答助手，可以调用工具来回答问题。\n"
    "规则：\n"
    "1. 涉及具体数值计算时，必须调用 calc，把要算的式子写成 expr（如 3e8/(2*10e6)），不要心算。\n"
    "2. 需要引用原文依据时，必须调用 corpus_search，并在最终答案里附上 chunk_id。\n"
    "3. 可以直接回答的常识性问题，不必调用工具。\n"
    "4. 一次回复里可以调用多个工具。"
)


# ⚠️ P7 端到端评测的**对照组**提示词（2026-09-21 新增）。必须与 SYSTEM 同源管理，
#    否则「有检索 vs 无检索」的差距会混进提示词差异，结论不可信。
#
#    A 组（无检索基线）：**不提任何工具**，只让模型凭自己答。
#       —— 不能用 SYSTEM（它写着"可以调用工具"），否则模型会输出调用块却没有工具可执行，
#          测出来的是"报错率"而不是"无检索能力"。
SYSTEM_PLAIN = (
    "你是雷达领域的问答助手。请直接回答用户的问题。\n"
    "规则：\n"
    "1. 涉及具体数值计算时，写出所用公式并算出结果。\n"
    "2. 回答要简洁准确；语料里没有把握的内容，应说明无法确定。\n"
    "3. 你可以直接回答的常识性问题，不必额外说明。"
)

# C/D 组（黄金上下文 / 随机上下文对照）：把资料**直接塞进提示词**，不走检索。
#    ⚠️ 这两组是 §5.10 ②「检索-生成解耦」的关键：
#      若 C 明显优于 B → 瓶颈在**检索**（没翻到）；
#      若 C ≈ B       → 瓶颈在**生成**（翻到了也答不好）或检索已足够；
#      若 D 也涨      → 说明"多给点文字"本身就有效，C 的提升不能算检索的功劳。
SYSTEM_WITH_CTX = (
    "你是雷达领域的问答助手。请**只依据下面给出的参考资料**回答用户的问题。\n"
    "规则：\n"
    "1. 参考资料里有的，据此作答并在末尾注明出处编号（形如 book_txt_p0036_b011）。\n"
    "2. 参考资料里没有的，直接说明“依据不足，无法回答”，不要编造。\n"
    "3. 涉及数值计算时，写出所用公式并算出结果。\n"
    "\n"
    "参考资料：\n"
    "{context}"
)


def mk_call(name: str, args: dict) -> dict:
    """OpenAI 风格的 tool_call 片段。"""
    return {"type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


# few-shot 示例。**必须用真渲染**（走 apply_chat_template），不要手写字符串 ——
# 手写的格式一旦与模型预期不符，示例反而把模型带偏。
#
# ⚠️ 实测（2026-09-20）：给这 2 组示例后，格式合规率 33.3% → **95.8%**。
#    但按题型拆开看，它让模型对**所有题都调工具**（纯概念题也去检索），
#    即"万物皆工具"退化。保留它是因为本项目要求答案有依据，检索几乎总是对的；
#    "该不该调"的判断留给微调或后续路由优化。
FEWSHOT: list[dict[str, Any]] = [
    {"role": "user", "content": "带宽 10MHz 的距离分辨率是多少？"},
    {"role": "assistant", "content": "", "tool_calls": [
        mk_call("calc", {"expr": "3e8/(2*10e6)", "unit": "m"})]},
    # ⚠️ role 必须是 **"tool"**！实测 role="function" 时观察内容会被**静默丢弃**
    #    （不报错，但模型看不到工具返回了什么 → 整个循环失效）。
    {"role": "tool", "name": "calc", "content": "15 m"},
    {"role": "user", "content": "什么是空时自适应处理？"},
    {"role": "assistant", "content": "", "tool_calls": [
        mk_call("corpus_search", {"query": "空时自适应处理 STAP", "top_k": 5})]},
    {"role": "tool", "name": "corpus_search",
     "content": "已检索到 5 段，最相关：book_txt_p0123_b002（第2章 空时自适应处理原理）"},
]


# 末步强制收尾时追加的指令（配合解码层屏蔽 <tool_call>）
FORCE_FINAL = "已经没有更多步骤了。请直接给出最终答案，不要再调用任何工具。"


def build_messages(user_q: str, fewshot: bool = True) -> list[dict[str, Any]]:
    """拼出初始 messages（system + 可选 few-shot + 用户问题）。"""
    msgs: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM}]
    if fewshot:
        msgs += FEWSHOT
    msgs.append({"role": "user", "content": user_q})
    return msgs
