#!/usr/bin/env python
"""P8 · 启动演示界面（thin wrapper：只做参数解析与启动）。

    python scripts/p8_serve.py                       # 默认挂 p5b_tool5，端口 7860
    python scripts/p8_serve.py --port 8080
    python scripts/p8_serve.py --adapter ""          # 用基座，对比微调前后
    python scripts/p8_serve.py --embed-cpu           # 显存紧张时把向量编码挪到 CPU

启动后浏览器打开 http://127.0.0.1:7860

⚠️ 首次启动要加载模型（约 1 min）+ 索引与 bge-m3（约 30 s），
   页面顶栏显示「就绪」后才能提问。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    # 2026-09-21：默认切到 p5b_tool6（calc 改「传算术表达式」接口后的版本）
    ap.add_argument("--adapter", default="outputs/p5_lora/p5b_tool6",
                    help="LoRA 适配器；传空字符串则用基座")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--embed-cpu", action="store_true",
                    help="检索的向量编码走 CPU（显存不足时用，单条 +0.3s）")
    args = ap.parse_args()

    import uvicorn

    from src.serve.app import build_app

    adapter = args.adapter or None
    print(f"[p8] 适配器 = {adapter or '（基座，未挂载 LoRA）'}")
    print("[p8] 正在加载模型与索引，请稍候…", flush=True)
    app = build_app(adapter=adapter, embed_cpu=args.embed_cpu)
    print(f"[p8] 就绪 → http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
