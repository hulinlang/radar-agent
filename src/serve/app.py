"""P8 演示界面 —— 把 P0~P7 的成果"装进一个能打开的网页"。

设计取舍（为什么不用 gradio）：
  本项目最值得展示的不是"问答"，而是 **Agent 的中间过程** ——
  它调了什么工具、看到了什么、怎么决定再查一次、最后引用了哪一段。
  gradio 的 Chatbot 组件只能塞文本，展示不了这种分层结构；
  自己写单页 HTML 能精确控制"每一步"的呈现，也省掉 gradio 的版本兼容风险。

⚠️ 与前面各阶段的**同一套代码**：直接调 `src.agent.react` 与 `src.tools`，
  不在服务端另写一套推理逻辑 —— 否则演示的和评测的就不是同一个东西了。

接口：
  GET  /              单页界面
  POST /api/ask       {question, mode} → 答案 + 逐步轨迹 + 引用原文
  GET  /api/health    就绪状态（模型 / 适配器 / 索引规模）
  GET  /api/samples   预设示例问题
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pydantic import BaseModel  # noqa: E402

from src.config import PROJECT_ROOT, load_config  # noqa: E402

# 预设示例：覆盖四种典型行为，正好对应 P6/P7 实测过的场景
SAMPLES = [
    {"q": "某雷达信号带宽 B = 100 MHz，该信号的距离分辨率是多少米？",
     "why": "计算题 → 应调计算器"},
    {"q": "空时自适应处理(STAP)的基本原理是什么？",
     "why": "需要原文依据 → 应检索并附出处"},
    {"q": "教材里关于机载雷达在海杂波背景下目标检测困难的原因，是怎么说的？",
     "why": "教材细节 → 检索 + 引用"},
    {"q": "请说明 2026 年最新一代机载有源相控阵雷达的装备价格是多少。",
     "why": "语料里没有 → 应当老实说不知道"},
    {"q": "用一句话说明什么是雷达。",
     "why": "常识题 → 不该调工具（tool5 已修好这个判断）"},
]


class Engine:
    """持有模型 / 分词器 / 工具 / 语料。进程内单例。"""

    def __init__(self, adapter: str | None = None, embed_cpu: bool = False):
        import yaml
        from transformers import AutoTokenizer

        from src.agent.react import DEFAULTS as REACT_DEFAULTS
        from src.agent.react import tool_call_bad_ids
        from src.modeling import load_model
        from src.retrieval.index import load_corpus
        from src.tools import corpus_search
        from src.tools.registry import to_openai_tools

        import torch  # noqa: F401

        self.cfg = load_config()
        acfg = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "agent.yaml").read_text(encoding="utf-8"))
        corpus_search.DEFAULTS.update(acfg.get("tool_params", {}).get("corpus_search") or {})
        if embed_cpu:
            corpus_search.DEFAULTS["use_dense_device"] = "cpu"
        REACT_DEFAULTS.update({k: v for k, v in (acfg.get("react") or {}).items()
                               if k in REACT_DEFAULTS})
        self.react_defaults = REACT_DEFAULTS

        enabled = [n for n, on in (acfg.get("tools") or {}).items() if on]
        self.tools_spec = to_openai_tools(enabled=enabled)
        self.tool_names = enabled

        self.tok = AutoTokenizer.from_pretrained(
            self.cfg["paths"]["model_base_dir"], trust_remote_code=False)
        self.model, self.impl, _ = load_model(
            self.cfg["paths"]["model_base_dir"], dtype="bfloat16", device="cuda")

        if adapter:
            from peft import PeftModel
            ap = Path(adapter)
            if not ap.is_absolute():
                ap = PROJECT_ROOT / ap
            self.model = PeftModel.from_pretrained(self.model, str(ap))
            self.adapter = ap.name
        else:
            self.adapter = None

        self.bad = tool_call_bad_ids(self.tok)

        # 语料（用于把 chunk_id 还原成原文 —— 引用可溯源要有"可点开看"的落点）
        chunks, _ = load_corpus([("book", self.cfg["paths"]["chunks_file"]),
                                 ("paper", self.cfg["paths"]["papers_chunks_file"])])
        self.by_id = {c["chunk_id"]: c for c in chunks}
        self.n_chunks = len(chunks)

        self.lock = asyncio.Lock()
        self.ready = False

    def warm(self) -> None:
        from src.tools import registry
        t = time.perf_counter()
        r = registry.run("corpus_search", {"query": "雷达", "top_k": 1})
        self.warm_s = time.perf_counter() - t
        self.warm_ok = bool(r.ok)
        self.ready = True

    # ---------------- 两种模式 ----------------
    def ask_rag(self, q: str) -> dict[str, Any]:
        """完整系统：ReAct 自己决定要不要用工具。"""
        from src.agent.react import react
        from src.tools import registry

        tool_ms = [0.0]
        orig = registry.run

        def timed(name, args):
            t = time.perf_counter()
            r = orig(name, args)
            tool_ms[0] += (time.perf_counter() - t) * 1000.0
            return r

        rr = react(q, model=self.model, tok=self.tok, tools_spec=self.tools_spec,
                   bad_ids=self.bad, execute=timed)
        steps = [{
            "step": s.step, "status": s.status,
            "calls": s.calls, "observations": s.observations,
            "forced": s.forced_final,
            "t_gen_ms": round(s.t_gen_ms or 0.0, 1),
            "t_exec_ms": round(s.t_exec_ms or 0.0, 1),
        } for s in rr.steps]
        return {
            "mode": "rag", "answer": rr.answer, "steps": steps,
            "counters": dict(rr.counters), "wall_ms": round(rr.wall_ms, 1),
            "tool_ms": round(tool_ms[0], 1), "tokens": rr.total_tokens,
        }

    def ask_plain(self, q: str) -> dict[str, Any]:
        """无工具对照：模型凭自己答（与 P7 的 A 组同条件）。

        ⚠️ 必须屏蔽 `<tool_call>`：没有工具可执行时让它输出调用块 = 无效答案。
        """
        import torch
        from src.agent.prompts import SYSTEM_PLAIN

        msgs = [{"role": "system", "content": SYSTEM_PLAIN}, {"role": "user", "content": q}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt").to(self.model.device)
        t = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=512, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id,
                                      eos_token_id=self.tok.eos_token_id,
                                      bad_words_ids=self.bad)
        wall = (time.perf_counter() - t) * 1000.0
        seg = self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
        cut = seg.find("<|im_end|>")
        return {
            "mode": "plain", "answer": (seg[:cut] if cut >= 0 else seg).strip(),
            "steps": [], "counters": {}, "wall_ms": round(wall, 1),
            "tool_ms": 0.0, "tokens": int(out.shape[1] - ids["input_ids"].shape[1]),
        }

    # ---------------- 引用 ----------------
    CITE_RE = None

    def citations(self, answer: str) -> list[dict]:
        import re
        if Engine.CITE_RE is None:
            # ⚠️ 字符类必须含 `-`：真实 chunk_id 形如 book_txt_p0352_b009-b005
            Engine.CITE_RE = re.compile(r"\b(?:book|mmwave|paper)_[A-Za-z0-9_\-]+")
        out, seen = [], set()
        for cid in Engine.CITE_RE.findall(answer or ""):
            if cid in seen:
                continue
            seen.add(cid)
            c = self.by_id.get(cid)
            out.append({
                "chunk_id": cid, "found": c is not None,
                "title_path": (c or {}).get("title_path") or "",
                "source": (c or {}).get("source_type") or "",
                "text": ((c or {}).get("text") or "")[:900],
            })
        return out


class AskIn(BaseModel):
    """⚠️ 必须定义在**模块级**！

    本文件顶部有 `from __future__ import annotations`（注解全部变成字符串），
    FastAPI 解析 `body: AskIn` 时需要在模块命名空间里找到这个类。
    把它定义在 `build_app()` 内部 → FastAPI 找不到 → 请求一律 **422**，
    且日志**只报 422 不说原因**（pydantic v2 + 新版 FastAPI 的典型静默失败）。
    """
    question: str
    mode: str = "rag"


def build_app(adapter: str | None, embed_cpu: bool = False):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="radar-agent 演示", version="1.0")
    eng = Engine(adapter=adapter, embed_cpu=embed_cpu)

    @app.on_event("startup")
    async def _warm():
        eng.warm()

    @app.get("/api/health")
    def health():
        return {"ready": eng.ready, "adapter": eng.adapter, "impl": eng.impl,
                "n_chunks": eng.n_chunks, "tools": eng.tool_names,
                "warm_ok": getattr(eng, "warm_ok", None),
                "warm_s": round(getattr(eng, "warm_s", 0.0), 1)}

    @app.get("/api/samples")
    def samples():
        return SAMPLES

    @app.post("/api/ask")
    async def ask(body: AskIn):
        q = (body.question or "").strip()
        if not q:
            raise HTTPException(400, "question 不能为空")
        # GPU 独占：串行处理，避免两个请求同时占显存
        async with eng.lock:
            if body.mode == "plain":
                r = await asyncio.to_thread(eng.ask_plain, q)
            else:
                r = await asyncio.to_thread(eng.ask_rag, q)
        r["question"] = q
        r["citations"] = eng.citations(r["answer"])
        return r

    @app.get("/", response_class=HTMLResponse)
    def index():
        p = Path(__file__).resolve().parent / "static" / "index.html"
        return HTMLResponse(p.read_text(encoding="utf-8"))

    return app
