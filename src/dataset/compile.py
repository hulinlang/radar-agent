"""作者格式 → 标准 JSONL 的编译流水线。

数据流：
    人写的 YAML（人类语言题面 + 公式参数）
        └─> validate_authoring   （结构 / 注入 / 单位 / 容差）
        └─> formulas.evaluate    （**真值由程序算出**，人手写不了）
        └─> 答案模板填充          （docs/05 §5.6 的固定模板）
        └─> build_messages       （拼成模型要求的 messages 结构）
        └─> validate_compiled    （messages 形状 / answer_check / token 预算）
        └─> 写 JSONL + sha256 清单

铁律：
    - 任何 critical 问题 → **该条不产出**，并计入报告（禁止静默跳过）；
    - 训练集产出统一标记 meta.status = "candidate"（docs/05 §9.0：
      **L2 语料到位前不得产出评测结论**）；
    - 评测集样本必须有权威出处，否则 critical。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import formulas, schema
from .schema import Issue, has_critical

ROOT = schema.ROOT


@dataclass
class CompileResult:
    records: list[dict[str, Any]] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    source: str = ""

    @property
    def n_in(self) -> int:
        return len(self.records) + len(self.dropped)

    @property
    def ok(self) -> bool:
        return not has_critical(self.issues)


# ---------------------------------------------------------------------------
def load_authoring(path: str | Path) -> list[dict[str, Any]]:
    """读取作者格式 YAML。支持顶层为 list，或 {records: [...]}。"""
    import yaml

    p = Path(path)
    with open(p, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if isinstance(doc, dict) and "records" in doc:
        recs = doc["records"]
    else:
        recs = doc
    if not isinstance(recs, list):
        raise ValueError(f"{p}: 顶层必须是记录列表，或 {{records: [...]}}；得到 {type(recs).__name__}")
    return recs


def _fill(template: str, values: dict[str, Any]) -> str:
    out = template
    for k, v in values.items():
        out = out.replace(f"<<{k}>>", str(v))
    return out


def _spec_fingerprint(spec: dict[str, Any]) -> dict[str, str]:
    p = Path(spec["_path"])
    raw = p.read_bytes()
    return {
        "spec_path": str(p),
        "spec_sha256": hashlib.sha256(raw).hexdigest(),
        "spec_version": str(spec.get("spec_version", "")),
    }


# ---------------------------------------------------------------------------
def _compute_calc(rec: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    out = formulas.evaluate(rec["formula"], rec.get("inputs") or {})
    tol = float(rec.get("tol_rel", spec["tolerance"]["by_task"].get("calc", 0.01)))
    return {
        "value": out["value"],
        "unit": rec.get("unit") or out["unit"],
        "tol": tol,
        "formula": rec["formula"],
        "formula_latex": out["latex"],
        "substitution_latex": out["substitution_latex"],
        "defaults_used": out["defaults_used"],
    }


def _build_answer(rec: dict[str, Any], spec: dict[str, Any], computed: dict[str, Any] | None) -> str:
    task = rec["task"]
    tpl = spec["answer_templates"][task]
    vals: dict[str, Any] = dict(rec)
    if computed:
        vals.update(
            {
                "value": formulas.fmt_latex_num(computed["value"]),
                "unit": computed["unit"],
                "formula_latex": computed["formula_latex"],
                "substitution_latex": computed["substitution_latex"],
            }
        )
    return _fill(tpl, vals)


def _build_answer_check(rec: dict[str, Any], computed: dict[str, Any] | None) -> dict[str, Any]:
    task = rec["task"]

    def _with_kp(d: dict[str, Any]) -> dict[str, Any]:
        """可选要点集：**纯增量**。只有作者显式写了 keypoints 才带上。
        为什么要它：calc 合并了原 formula 题型后，"公式形式对不对"不再有 symbolic 判分，
        只能靠要点覆盖补回来；choice 加了「解析」后，也需要第二层判据兜住"解析与字母不一致"。
        """
        if rec.get("keypoints"):
            d["keypoints"] = list(rec["keypoints"])
        return d

    if task == "calc":
        return _with_kp(
            {
                "type": "numeric",
                "value": computed["value"],
                "unit": computed["unit"],
                "tol": computed["tol"],
                "formula": computed["formula"],
                "formula_latex": computed["formula_latex"],
            }
        )
    if task in ("choice", "regime_trap"):
        # ⚠️ regime_trap 是**选择题的变体**，判分必须走 choice（字母精确匹配）。
        #    若漏在这里落到下面的 keyword 兜底，就会用"在输出里搜到字母"来判对错 ——
        #    模型只要提到 B 就被判对，静默虚高（2026-09-15 评审渲染时发现并修正）。
        return _with_kp({"type": "choice", "value": rec["answer"]})
    if task == "formula":
        return _with_kp({"type": "symbolic", "value": rec["answer"]})
    if task == "unanswerable":
        return {"type": "refusal", "value": rec["answer"]}
    # 半硬题（concept / contrast / term / clarify / 视觉题）带 keypoints：判分器按"要点覆盖"打分。
    base = {"type": "keyword", "value": rec["answer"]}
    if rec.get("keypoints"):
        base["keypoints"] = list(rec["keypoints"])
    return base


def _render_question(rec: dict[str, Any]) -> str:
    """题面 = 题干 + 选项列表。

    ⚠️ 历史 bug（2026-09-18 发现并修）：此前只取 `rec["question"]`，**从不拼 `rec["options"]`**，
    于是 choice / regime_trap 的题面里没有任何选项，答案却是一个字母（"最终答案：A"）。
    后果链（均已实测）：
      1. 模型看不到选项 → 只能学"题面 → 字母"的**噪声映射**（字母顺序本身是任意的）；
      2. 由此学到"短输出 = 字母"这个**表面模式**，并泛化到开放题 ——
         微调后 20 条非选择题型输出纯字母（如 concept 开放题答「最终答案：B」），基座 0 条；
      3. 评测集 21 条选择题本质不可答：正确 6/21 = 29%，随机基线 25%，
         二项检验 p = 0.433（不显著），无法排除蒙对。
    影响面：训练 204 条（choice 131 + regime_trap 73，占 21%）、评测 21 条（占 17.5%）。
    """
    q = rec["question"]
    opts = rec.get("options")
    if opts:
        q = q + "\n\n选项：\n" + "\n".join(str(o).strip() for o in opts)
    return q


def _build_messages(rec: dict[str, Any], spec: dict[str, Any], answer: str) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []

    system = rec.get("system") or spec.get("defaults", {}).get("system")
    if system:
        msgs.append({"role": "system", "content": system})

    q = _render_question(rec)
    modality = rec.get("modality", "text")
    if modality == "vision":
        img = rec["image"]
        content: list[dict[str, Any]] = [{"type": "image"} for _ in _as_image_list(img)]
        content.append({"type": "text", "text": q})
        msgs.append({"role": "user", "content": content})
    else:
        msgs.append({"role": "user", "content": q})

    msgs.append({"role": "assistant", "content": answer})
    return msgs


def _as_image_list(img: Any) -> list[Any]:
    if isinstance(img, list):
        return img
    return [img]


# ---------------------------------------------------------------------------
def count_tokens(msgs: list[dict[str, Any]], rec: dict[str, Any], spec: dict[str, Any], tok: Any) -> dict[str, int]:
    """估算 token 数。

    文本：直接 apply_chat_template(tokenize=True)。
    图表：先用 tokenize=False 渲染（此时每张图只占 1 个 <|image_pad|>），
          再把每个 image_pad 替换成实测展开数 (W/32)*(H/32)。
    """
    rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    ids = tok(rendered, add_special_tokens=False)["input_ids"]
    n_img_placeholders = sum(1 for i in ids if i == 151655)

    extra = 0
    if rec.get("modality", "text") == "vision":
        for img in _as_image_list(rec["image"]):
            w, h = img["size"]
            extra += schema.image_token_count(w, h, spec) - 1
    return {
        "text_and_placeholders": len(ids),
        "image_extra": extra,
        "total": len(ids) + extra,
        "image_count": n_img_placeholders,
    }


# ---------------------------------------------------------------------------
def compile_record(
    rec: dict[str, Any],
    spec: dict[str, Any],
    *,
    tok: Any = None,
    source: str = "",
    index: int = -1,
) -> tuple[dict[str, Any] | None, list[Issue]]:
    """编译单条。返回 (编译结果 或 None, issues)。有 critical 则返回 None。"""
    issues = schema.validate_authoring(rec, spec)
    if has_critical(issues):
        return None, issues

    computed = None
    if spec["tasks"][rec["task"]]["answer_mode"] == "computed":
        computed = _compute_calc(rec, spec)

    answer = _build_answer(rec, spec, computed)
    msgs = _build_messages(rec, spec, answer)
    ac = _build_answer_check(rec, computed)

    fp = _spec_fingerprint(spec)
    out: dict[str, Any] = {
        "id": rec["id"],
        "split": rec["split"],
        "modality": rec.get("modality", "text"),
        "task": rec["task"],
        "subdomain": rec["subdomain"],
        "regime": rec.get("regime"),
        "regime_b": rec.get("regime_b"),
        "ask_style": rec.get("ask_style"),
        "difficulty": rec["difficulty"],
        "messages": msgs,
        "answer_check": ac,
        "evidence": rec.get("evidence") or {"source": "", "quote": ""},
        "meta": {
            "created_by": rec.get("created_by", "authored"),
            "status": "candidate" if rec["split"] == "train" else "frozen-pending-L2",
            "source_file": source,
            "source_index": index,
            # ⚠️ 这里**刻意不放时间戳**：`compiled_at` 会让"同样输入 → 不同字节"，
            #    产物的 sha256 就再也无法复现 —— 那"冻结评测集（S4）"时拿 sha256 当身份就失去意义。
            #    需要时间看 manifest 的 `written_at` 就够了（已实测：全项目无任何代码读 compiled_at）。
            **fp,
        },
    }
    if computed:
        out["solution_steps"] = [computed["formula_latex"], computed["substitution_latex"]]
        if computed["defaults_used"]:
            out["meta"]["defaults_used"] = computed["defaults_used"]
    if rec.get("notes"):
        out["meta"]["notes"] = rec["notes"]

    if rec.get("modality", "text") == "vision":
        img = _as_image_list(rec["image"])[0]
        out["image"] = _normalize_image(img)

    issues.extend(schema.validate_compiled(out, spec))

    if tok is not None and not has_critical(issues):
        tb = count_tokens(msgs, rec, spec, tok)
        out["meta"]["token_estimate"] = tb
        budget = spec["model"]["context"]["max_seq_len"] - spec["model"]["context"]["reserve_for_output"]
        if tb["total"] > budget:
            issues.append(
                Issue(
                    "critical",
                    "E_TOKEN_BUDGET",
                    f"{rec['id']}.meta.token_estimate",
                    f"输入 {tb['total']} token 超过预算 {budget}"
                    f"（max_seq_len={spec['model']['context']['max_seq_len']} - "
                    f"reserve_for_output={spec['model']['context']['reserve_for_output']}）",
                )
            )

    if has_critical(issues):
        return None, issues
    return out, issues


def _normalize_image(img: dict[str, Any]) -> dict[str, Any]:
    out = {
        "path": img.get("path"),
        "size": list(img.get("size") or []),
        "figure_truth": img.get("figure_truth"),
    }
    p = img.get("path")
    if p:
        f = (ROOT / p) if not Path(p).is_absolute() else Path(p)
        if f.exists():
            out["sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
        elif img.get("sha256"):
            out["sha256"] = img["sha256"]
    return out


# ---------------------------------------------------------------------------
def compile_file(
    src: str | Path,
    *,
    spec_path: str | Path | None = None,
    tok: Any = None,
) -> CompileResult:
    spec = schema.load_spec(spec_path)
    res = CompileResult(source=str(src))

    records = load_authoring(src)

    # 批次级校验（重复 id / 配额）
    res.issues.extend(schema.validate_batch(records, spec))

    for i, rec in enumerate(records):
        out, issues = compile_record(rec, spec, tok=tok, source=Path(src).name, index=i)
        res.issues.extend(issues)
        if out is None:
            res.dropped.append({"id": rec.get("id", f"<index {i}>"), "reason": [str(x) for x in issues if x.severity == "critical"]})
        else:
            res.records.append(out)
    return res


def write_jsonl(records: list[dict[str, Any]], out_path: str | Path, *, manifest_extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """落盘 JSONL 并返回清单（含 sha256）。"""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + ("\n" if records else "")
    p.write_text(payload, encoding="utf-8")

    manifest = {
        "output": str(p),
        "n_records": len(records),
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "bytes": p.stat().st_size,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if manifest_extra:
        manifest.update(manifest_extra)

    mpath = p.with_suffix(".manifest.json")
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_tokenizer(spec: dict[str, Any] | None = None) -> Any:
    from transformers import AutoTokenizer

    spec = spec or schema.load_spec()
    return AutoTokenizer.from_pretrained(spec["model"]["tokenizer_dir"])
