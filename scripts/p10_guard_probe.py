# -*- coding: utf-8 -*-
"""P10-A4 护栏等效性探针：验证 llama.cpp 的 logit_bias 真生效（P6-10 纪律）。

背景：HF 侧末步屏蔽 `<tool_call>` 用 `bad_words_ids`；llama.cpp 没有该参数，
部署形态改用 `logit_bias: [[151657, -100]]`（`<tool_call>` 是单个特殊 token）。
**不假设它生效，必须实测**（对应 p6_guard_probe.py 对 HF 侧做过的验证）。

三组探测：
  A. 开 bias，末步强制作答渲染 → 输出必须不含 `<tool_call>`（critical 断言）
  B. 关 bias，同输入 → 记录模型尝试调工具的比例（信息项：证明 A 的断言不是
     "模型本来就不调"的水样）
  C. 端到端：react() 走 gen_backend 跑 3 条典型问题，答案非空且步骤正常

用法（需 llama-server 已运行；--spawn 可按 deploy_gguf.yaml 自动拉起）：
  python scripts/p10_guard_probe.py --spawn
  python scripts/p10_guard_probe.py --url http://127.0.0.1:8080
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Windows 控制台是 GBK（§4.4）：✓/✗ 等符号会直接 UnicodeEncodeError，
# 显式把 stdout 切成 UTF-8（失败退化为 replace，不影响判分逻辑）
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 容易触发工具调用的问题（计算题 + 语料细节题）
PROBE_QUESTIONS = [
    "某雷达信号带宽 B = 100 MHz，该信号的距离分辨率是多少米？",
    "空时自适应处理(STAP)的基本原理是什么？",
    "某雷达 PRF = 10 kHz，其最大无模糊距离是多少？",
    "教材里关于机载雷达在海杂波背景下目标检测困难的原因，是怎么说的？",
    "波长 0.03 m 的雷达，目标径向速度 150 m/s，多普勒频率是多少？",
]

SMOKE_QUESTIONS = [
    ("某雷达信号带宽 B = 100 MHz，该信号的距离分辨率是多少米？", "应调 calc"),
    ("空时自适应处理(STAP)的基本原理是什么？", "应调 corpus_search"),
    ("用一句话说明什么是雷达。", "常识题，不应调工具"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="P10 logit_bias 护栏等效性探针")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--spawn", action="store_true",
                    help="按 configs/deploy_gguf.yaml 自动拉起本地 llama-server")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from src.config import PROJECT_ROOT, load_config
    from src.agent.prompts import FORCE_FINAL
    from src.engines.react_llamacpp import LlamaCppGenBackend

    engine = None
    if args.spawn:
        import yaml
        from src.engines import substitute_paths
        from src.engines.llamacpp_engine import LlamaCppEngine
        base_cfg = load_config()
        deploy = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "deploy_gguf.yaml").read_text(encoding="utf-8"))
        substitute_paths(deploy, base_cfg)
        engine = LlamaCppEngine(deploy, base_cfg)
        print("[probe] 拉起 llama-server …", flush=True)
        engine.start()

    try:
        base_cfg = load_config()
        tok = AutoTokenizer.from_pretrained(
            base_cfg["paths"]["model_base_dir"], trust_remote_code=False)
        backend = LlamaCppGenBackend(tok, base_url=args.url)
        print(f"[probe] 服务就绪: {backend.server_info}\n", flush=True)

        # ---- A：开 bias → 必须无 <tool_call>（critical）----
        print("=== A. 开 logit_bias（critical：输出不得含 <tool_call>）===")
        a_violations = []
        for q in PROBE_QUESTIONS:
            out = backend.probe_forced_final(q, FORCE_FINAL, use_bias=True)
            hit = "<tool_call>" in out
            a_violations.append(hit)
            print(f"  [{'✗' if hit else '✓'}] {q[:30]}… → {len(out)} chars")
        n_a = sum(a_violations)

        # ---- B：关 bias → 信息项（模型自然行为下多大概率还想调工具）----
        print("\n=== B. 关 logit_bias（信息项）===")
        b_calls = 0
        for q in PROBE_QUESTIONS:
            out = backend.probe_forced_final(q, FORCE_FINAL, use_bias=False)
            hit = "<tool_call>" in out
            b_calls += int(hit)
            print(f"  [{'调工具' if hit else '未调'}] {q[:30]}…")

        # ---- C：端到端 react() 走 gen_backend ----
        print("\n=== C. 端到端 react(gen_backend=llama-server) ===")
        from src.agent.react import react
        from src.tools import registry
        from src.tools.registry import to_openai_tools
        tools_spec = to_openai_tools(enabled=["corpus_search", "calc", "web_search"])
        c_fail = []
        for q, expect in SMOKE_QUESTIONS:
            rr = react(q, model=None, tok=None, tools_spec=tools_spec,
                       execute=registry.run, gen_backend=backend)
            n_calls = sum(len(s.calls) for s in rr.steps)
            ok = bool(rr.answer.strip())
            c_fail.append(not ok)
            print(f"  [{'✓' if ok else '✗'}] {q[:26]}… ({expect})")
            print(f"      步数={len(rr.steps)} 调用={n_calls} "
                  f"计数器={ {k: v for k, v in rr.counters.items() if v} }")
            print(f"      答案: {rr.answer[:80].replace(chr(10), ' ')}…")

        print("\n=== 结论 ===")
        print(f"A 违规: {n_a}/{len(PROBE_QUESTIONS)}（critical，必须为 0）")
        print(f"B 关 bias 仍想调工具: {b_calls}/{len(PROBE_QUESTIONS)}（信息项）")
        print(f"C 端到端失败: {sum(c_fail)}/{len(SMOKE_QUESTIONS)}")
        if n_a > 0:
            print("✗ critical 断言失败：logit_bias 未生效，部署护栏不成立，"
                  "需改用「检测到 <tool_call> 即截断」兜底方案")
            return 1
        if any(c_fail):
            print("✗ 端到端冒烟有空答案")
            return 1
        print("✓ 护栏等效性实测通过 + 端到端冒烟通过")
        return 0
    finally:
        if engine is not None:
            engine.stop()


if __name__ == "__main__":
    sys.exit(main())
