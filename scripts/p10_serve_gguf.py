# -*- coding: utf-8 -*-
"""P10 · GGUF 部署形态启动器：llama-server（SFT Q4_K_M）+ FastAPI 演示页。

    python scripts/p10_serve_gguf.py                     # 拉起 llama-server + 演示页
    python scripts/p10_serve_gguf.py --port 7860
    python scripts/p10_serve_gguf.py --external-url http://127.0.0.1:8080
        # 不拉起本地 server，直连一个已在运行的服务（WSL2 CUDA 部署时用这项）

与 p8_serve.py 的关系：同一套前端与 Agent 代码（src/serve/app.py），
只是 backend=llamacpp —— 演示页两模式（rag/plain）的生成都走 llama-server，
不再进程内加载 HF 权重。这正是 docs/17 §4 S7 要求的「前端对接」。

⚠️ llama-server 进程由本脚本托管：退出时（含 Ctrl+C）负责终止，不留孤儿进程。
"""
from __future__ import annotations

import argparse
import atexit
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description="P10 GGUF 部署形态启动器")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860, help="演示页端口")
    ap.add_argument("--llama-port", type=int, default=8080, help="llama-server 端口")
    ap.add_argument("--external-url", default=None,
                    help="直连已运行的 llama-server（如 WSL2 里的 CUDA 服务），"
                         "传了就不在本地拉起进程")
    ap.add_argument("--embed-cpu", action="store_true",
                    help="检索的向量编码走 CPU（此时 GPU 只有 llama-server，通常不需要）")
    ap.add_argument("--config", default="configs/deploy_gguf.yaml")
    args = ap.parse_args()

    import uvicorn
    import yaml

    from src.config import PROJECT_ROOT, load_config
    from src.engines import substitute_paths
    from src.serve.app import build_app

    base_cfg = load_config()

    engine = None
    if args.external_url:
        llama_url = args.external_url
        print(f"[p10] 直连外部 llama-server: {llama_url}")
    else:
        from src.engines.llamacpp_engine import LlamaCppEngine
        deploy_path = PROJECT_ROOT / args.config
        deploy = yaml.safe_load(deploy_path.read_text(encoding="utf-8"))
        substitute_paths(deploy, base_cfg)
        deploy["server"]["port"] = args.llama_port
        engine = LlamaCppEngine(deploy, base_cfg)
        print(f"[p10] 正在拉起 llama-server（SFT Q4_K_M）…", flush=True)
        engine.start()
        info = engine.info()
        print(f"[p10] llama-server 就绪: {info['llm_gguf']}"
              f"（{info['llm_size_bytes'] / 2**20:.0f} MiB）"
              f" 设备={info['resolved_device']} 启动耗时={info['startup_seconds']:.1f}s",
              flush=True)
        llama_url = f"http://127.0.0.1:{args.llama_port}"
        atexit.register(engine.stop)

    print("[p10] 正在加载索引与工具（不加载模型权重）…", flush=True)
    app = build_app(adapter=None, embed_cpu=args.embed_cpu,
                    backend="llamacpp", llama_url=llama_url)
    print(f"[p10] 就绪 → http://{args.host}:{args.port}", flush=True)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        if engine is not None:
            engine.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
