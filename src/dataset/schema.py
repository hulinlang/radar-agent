"""数据集 schema 与校验器（S1）。

职责边界（**重要**）：
    本模块只做**机器可判定**的约束——结构、类型、单位、容差、id 前缀、注入清洗、
    图像尺寸/预算。它**不判断"内容对不对"**（公式是否用对、概念解释是否准确）：
    数值正确性由 `src/dataset/formulas.py` 程序复核，
    语义正确性只能由人 / 后续的裁判 LLM 把关（docs/05 §八）。

断言分级（规范 §5.8）：
    critical — 结果不可信；调用方必须以非零退出码结束，且**不得产出数据集**
    warning  — 可疑，需人工确认；允许继续，但必须写进报告
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from . import formulas, term_registry

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = ROOT / "configs" / "dataset.yaml"

Severity = Literal["critical", "warning"]

# 实测来源：logs/probe/p3_chat_template_probe.txt（双信源交叉校验的兜底值）
_FALLBACK_SPECIAL_IDS = {151643, 151644, 151645, 151652, 151653, 151654, 151655, 151656}

_ID_RE = re.compile(r"^(txt|vis|evt|evv)_\d{6}$")
_FAKE_SPECIAL_RE = re.compile(r"<\|[^|<>]{1,32}\|>")


@dataclass(frozen=True)
class Issue:
    severity: Severity
    code: str
    path: str
    message: str

    def __str__(self) -> str:
        mark = "CRITICAL" if self.severity == "critical" else "warning "
        return f"[{mark}] {self.code} @ {self.path}: {self.message}"


def has_critical(issues: list[Issue]) -> bool:
    return any(i.severity == "critical" for i in issues)


def count_critical(issues: list[Issue]) -> int:
    return sum(1 for i in issues if i.severity == "critical")


def count_warning(issues: list[Issue]) -> int:
    return sum(1 for i in issues if i.severity == "warning")


# ---------------------------------------------------------------------------
# spec 加载
# ---------------------------------------------------------------------------
_spec_cache: dict[str, Any] = {}


class SpecError(ValueError):
    """spec 文件本身有问题（如重复键）—— 必须响亮失败，不能静默取后者。"""


def _build_strict_loader():
    """构造一个**拒绝重复键**的 YAML Loader。

    为什么要这个（2026-09-15 的教训）：YAML 允许同一个 mapping 出现重复键，
    但行为是「后者静默覆盖前者」—— 配置里两条 `clarify:` 定义，第一条的字段
    会被无声丢弃。这正是本项目最在意的静默错误类型，所以直接 fail fast。
    """
    import yaml

    class StrictLoader(yaml.SafeLoader):
        pass

    def _no_dup(loader, node, deep=False):  # noqa: ANN001, ANN202
        seen: set[Any] = set()
        for k, _ in node.value:
            key = loader.construct_object(k, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"重复键 {key!r}：YAML 允许重复但后者会**静默覆盖前者**，"
                    f"此前已因此丢失过配置字段。请合并为一条。",
                    node.start_mark,
                )
            seen.add(key)
        return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)

    StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_dup)
    return StrictLoader


def load_spec(path: str | Path | None = None) -> dict[str, Any]:
    import yaml

    p = Path(path) if path else DEFAULT_SPEC
    key = str(p)
    if key not in _spec_cache:
        with open(p, encoding="utf-8") as fh:
            try:
                data = yaml.load(fh, Loader=_build_strict_loader())
            except yaml.constructor.ConstructorError as exc:
                raise SpecError(f"{p} 存在重复键：{exc.problem}") from exc
        data["_path"] = key
        _spec_cache[key] = data
    return _spec_cache[key]


# ---------------------------------------------------------------------------
# 特殊 token / 注入
# ---------------------------------------------------------------------------
def special_token_ids(*, with_tokenizer: bool = False) -> tuple[set[int], list[Issue]]:
    """返回特殊 token id 集合。

    with_tokenizer=True 时从真实 tokenizer 读取，并与 spec 里记录的期望值**交叉校验**
    （规范 §5.8「交叉信源校验」）——不一致即 critical，防止"模型换了常量没换"。
    """
    spec = load_spec()
    expected = set(spec["model"]["special_token_ids_expected"])
    issues: list[Issue] = []

    if not with_tokenizer:
        return expected, issues

    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(spec["model"]["tokenizer_dir"])
        actual = {i for i, t in tok.added_tokens_decoder.items() if t.special}
        if actual != expected:
            issues.append(
                Issue(
                    "critical",
                    "E_SPECIAL_TOKEN_DRIFT",
                    "model.special_token_ids_expected",
                    f"tokenizer 实测特殊 token 集合与 spec 记录不一致："
                    f"多出 {sorted(actual - expected)}，缺失 {sorted(expected - actual)}",
                )
            )
        return actual, issues
    except Exception as exc:  # noqa: BLE001
        issues.append(
            Issue(
                "warning",
                "W_TOKENIZER_UNAVAILABLE",
                "model.tokenizer_dir",
                f"无法加载 tokenizer（{type(exc).__name__}: {exc}）；改用 spec 记录值，"
                f"本次**未做**双信源交叉校验",
            )
        )
        return expected, issues


def scan_text(text: str, spec: dict[str, Any] | None = None) -> list[Issue]:
    """扫描单个文本字段：banned 标记（critical）+ 疑似自造 `<|...|>` 标记（warning）。"""
    spec = spec or load_spec()
    issues: list[Issue] = []
    if not isinstance(text, str):
        return issues

    for bad in spec["model"]["banned_token_strings"]:
        if bad in text:
            issues.append(
                Issue(
                    "critical",
                    "E_INJECTION",
                    "text",
                    f"文本含被禁标记 {bad!r}（R13 注入清洗）：入库后会变成真 token，"
                    f"在使用者未预期处**切断训练样本**。整条样本必须丢弃",
                )
            )

    for m in _FAKE_SPECIAL_RE.finditer(text):
        token = m.group(0)
        if token not in spec["model"]["banned_token_strings"]:
            issues.append(
                Issue(
                    "warning",
                    "W_FAKE_SPECIAL",
                    "text",
                    f"出现疑似自造特殊标记 {token!r}（R16）：会与真特殊 token 视觉混淆，"
                    f"建议改用普通文本定界（如 LaTeX $...$）",
                )
            )
    return issues


def _iter_text_fields(obj: Any, path: str = "$"):
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _iter_text_fields(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _iter_text_fields(v, f"{path}[{i}]")


# ---------------------------------------------------------------------------
# 图像
# ---------------------------------------------------------------------------
def image_token_count(w: int, h: int, spec: dict[str, Any] | None = None) -> int:
    """按实测规则算视觉 token 数： (W/align) * (H/align)，align = patch*merge = 32。"""
    spec = spec or load_spec()
    a = spec["model"]["image"]["align_multiple"]
    return (int(w) // a) * (int(h) // a)


def check_image(meta: dict[str, Any], path: str, spec: dict[str, Any] | None = None) -> list[Issue]:
    spec = spec or load_spec()
    cfg = spec["model"]["image"]
    issues: list[Issue] = []

    size = meta.get("size")
    if not (isinstance(size, (list, tuple)) and len(size) == 2):
        issues.append(Issue("critical", "E_IMG_SIZE", path, f"image.size 必须是 [w, h]，得到 {size!r}"))
        return issues

    w, h = int(size[0]), int(size[1])
    a = cfg["align_multiple"]
    if w % a or h % a:
        issues.append(
            Issue(
                "critical",
                "E_IMG_ALIGN",
                path,
                f"图像尺寸 {w}x{h} 未对齐 {a} 的倍数（R14）——"
                f"否则实际视觉 token 数与预算不符",
            )
        )

    px = w * h
    if px < cfg["min_pixels"]:
        issues.append(
            Issue(
                "warning",
                "W_IMG_UPSCALED",
                path,
                f"{w}x{h} = {px} 像素 < min_pixels={cfg['min_pixels']}：会被**上采样**，"
                f"'用小图省 token' 无效（实测 224x224 仍占 64 token）",
            )
        )
    if px > cfg["max_pixels"]:
        issues.append(
            Issue("critical", "E_IMG_TOO_LARGE", path, f"{w}x{h} 超过 max_pixels={cfg['max_pixels']}")
        )

    img_path = meta.get("path") or path
    if img_path:
        f = (ROOT / img_path) if not Path(img_path).is_absolute() else Path(img_path)
        if not f.exists():
            issues.append(Issue("warning", "W_IMG_MISSING", path, f"图像文件不存在：{f}"))
        else:
            if not meta.get("sha256"):
                issues.append(
                    Issue(
                        "critical",
                        "E_IMG_NO_SHA256",
                        path,
                        "图像存在但缺 sha256：将无法证明'评测用的图'='训练过的图'",
                    )
                )
            else:
                real = hashlib.sha256(f.read_bytes()).hexdigest()
                if real != meta["sha256"]:
                    issues.append(
                        Issue(
                            "critical",
                            "E_IMG_SHA256_MISMATCH",
                            path,
                            f"sha256 不符：记录 {meta['sha256'][:12]}… 实际 {real[:12]}…",
                        )
                    )

    if not meta.get("figure_truth"):
        issues.append(
            Issue("critical", "E_IMG_NO_TRUTH", path, "图表样本必须有 figure_truth（一图一真值，R4）")
        )
    return issues


# ---------------------------------------------------------------------------
# 作者格式校验
# ---------------------------------------------------------------------------
def validate_authoring(rec: dict[str, Any], spec: dict[str, Any] | None = None) -> list[Issue]:
    spec = spec or load_spec()
    issues: list[Issue] = []
    rid = rec.get("id", "<no-id>")

    def crit(code: str, path: str, msg: str) -> None:
        issues.append(Issue("critical", code, f"{rid}{path}", msg))

    def warn(code: str, path: str, msg: str) -> None:
        issues.append(Issue("warning", code, f"{rid}{path}", msg))

    # --- 注入扫描（对所有文本字段）---
    for p, txt in _iter_text_fields(rec):
        for iss in scan_text(txt, spec):
            issues.append(Issue(iss.severity, iss.code, f"{rid}{p[1:]}", iss.message))

    # --- id / 分类 ---
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        crit("E_ID_FORMAT", ".id", f"id 必须形如 txt_000001 / vis_000001 / evt_… / evv_…，得到 {rid!r}")

    split = rec.get("split")
    if split not in ("train", "eval"):
        crit("E_SPLIT", ".split", f"split 必须是 train|eval，得到 {split!r}")

    modality = rec.get("modality", "text")
    if modality not in ("text", "vision"):
        crit("E_MODALITY", ".modality", f"modality 必须是 text|vision，得到 {modality!r}")
    else:
        want = spec["id_prefix"].get(f"{split}.{modality}")
        if isinstance(rid, str) and want and not rid.startswith(want + "_"):
            crit("E_ID_PREFIX", ".id", f"{split}/{modality} 的 id 前缀应为 {want}，得到 {rid!r}")

    task = rec.get("task")
    task_cfg = spec["tasks"].get(task) if isinstance(task, str) else None
    if task_cfg is None:
        crit("E_TASK", ".task", f"task 必须属于 {sorted(spec['tasks'])}，得到 {task!r}")

    if rec.get("subdomain") not in spec["subdomains"]:
        crit("E_SUBDOMAIN", ".subdomain", f"subdomain 必须属于 {spec['subdomains']}，得到 {rec.get('subdomain')!r}")

    if rec.get("difficulty") not in spec["difficulties"]:
        crit("E_DIFFICULTY", ".difficulty", f"difficulty 必须属于 {spec['difficulties']}，得到 {rec.get('difficulty')!r}")

    # --- 题面 ---
    q = rec.get("question")
    if not (isinstance(q, str) and q.strip()):
        crit("E_QUESTION", ".question", "question 必须是非空字符串")
    else:
        if len(q) < 8:
            warn("W_Q_SHORT", ".question", f"题面仅 {len(q)} 字，可能信息不足")
        if len(q) > 300:
            warn("W_Q_LONG", ".question", f"题面 {len(q)} 字，偏长（会挤占上下文预算）")

    # --- 体制（regime）：与 subdomain 正交的第二维 ---
    # 为什么必须显式：同一名词在不同体制下口径不同（"距离分辨率"在脉冲雷达里 B 是信号带宽，
    # 在 FMCW 里 B 是扫频带宽，在 SAR 方位向则与带宽无关）。不声明体制 → 题目没有唯一答案。
    regimes = spec.get("regimes", {})
    rg = rec.get("regime")
    if rg not in regimes:
        crit("E_REGIME", ".regime", f"regime 必填且必须属于 {sorted(regimes)}，得到 {rg!r}")
    else:
        q_text = q if isinstance(q, str) else ""
        terms = regimes[rg].get("surface_terms") or []
        # universal（普适）与 unspecified（未指定）都**不要求**出现体制词
        if rg not in ("universal", "unspecified") and not any(t in q_text for t in terms):
            crit(
                "E_REGIME_SURFACE",
                ".question",
                f"regime={rg}，但题面没有出现该体制的任何表面词 {terms} —— "
                f"读者无从判断在问哪个体制，答案就没有唯一口径",
            )
        if rg == "universal":
            hits = [
                (k, t)
                for k, v in regimes.items()
                if k != "universal"
                for t in (v.get("surface_terms") or [])
                if t in q_text
            ]
            if hits:
                warn(
                    "W_REGIME_MISMATCH",
                    ".question",
                    f"regime=universal 但题面出现具体体制词 {hits[:3]} —— 请确认是否应改为对应体制",
                )

    # --- 问法（ask_style，可选）：用于统计"同一知识点是否被多种问法覆盖" ---
    styles = spec.get("ask_styles", {})
    ask = rec.get("ask_style")
    if ask is not None and ask not in styles:
        crit("E_ASK_STYLE", ".ask_style", f"ask_style 必须属于 {sorted(styles)}，得到 {ask!r}")

    # --- regime=unspecified 与 task=clarify 必须**成对**出现 ---
    # 理由：给"未给体制"的题配一个确定答案，等于教模型猜（正是要打断的捷径）。
    if rg == "unspecified" and task != "clarify":
        crit(
            "E_UNSPECIFIED_TASK",
            ".task",
            f"regime=unspecified 表示题面未给体制，此时 task 必须是 clarify（先澄清），"
            f"得到 {task!r}",
        )
    if task == "clarify" and rg != "unspecified":
        crit("E_CLARIFY_REGIME", ".regime", f"task=clarify 的样本必须标 regime=unspecified，得到 {rg!r}")

    if task_cfg is None:
        return issues

    # --- 任务必填项 ---
    for field_name in task_cfg.get("required", []):
        if field_name not in rec or rec[field_name] in (None, "", [], {}):
            crit("E_REQUIRED", f".{field_name}", f"task={task} 缺少必填字段 {field_name}")

    # --- 按任务的专项校验 ---
    if task == "calc":
        fname = rec.get("formula")
        if isinstance(fname, str):
            if fname not in formulas.REGISTRY:
                crit("E_FORMULA_UNKNOWN", ".formula", f"未知公式 {fname!r}；可用：{formulas.formula_names()}")
            else:
                try:
                    out = formulas.evaluate(fname, rec.get("inputs") or {})
                    expect_unit = rec.get("unit")
                    if expect_unit and expect_unit != out["unit"]:
                        crit(
                            "E_UNIT_MISMATCH",
                            ".unit",
                            f"unit 写成 {expect_unit!r}，但公式 {fname} 的输出单位是 {out['unit']!r}",
                        )
                except formulas.FormulaError as exc:
                    crit("E_FORMULA_PARAMS", ".inputs", str(exc))
        tol = rec.get("tol_rel", spec["tolerance"]["by_task"].get("calc"))
        if not isinstance(tol, (int, float)) or not (0 < float(tol) < 1):
            crit("E_TOL", ".tol_rel", f"tol_rel 必须是 (0,1) 的实数，得到 {tol!r}")

    elif task in ("choice", "regime_trap"):
        opts = rec.get("options")
        if not (isinstance(opts, list) and len(opts) >= 2):
            crit("E_OPTIONS", ".options", f"选择题至少 2 个选项，得到 {opts!r}")
        else:
            keys = []
            for o in opts:
                if isinstance(o, dict):
                    keys.append(o.get("key"))
                elif isinstance(o, str) and len(o) >= 2 and o[1] in ":：.)、 ":
                    keys.append(o[0])
            ans = rec.get("answer")
            if keys and ans not in keys:
                crit("E_ANSWER_KEY", ".answer", f"answer={ans!r} 不在选项 key {keys} 中")
        if task == "regime_trap" and rec.get("regime") == "universal":
            crit(
                "E_TRAP_NO_REGIME",
                ".regime",
                "regime_trap 必须绑定一个**具体体制**（不能是 universal）—— "
                "这类题的意义就是测「把别的体制的结论套过来」",
            )
        # 2026-09-18 新增：答案模板已改为「选 X，因为 <<explanation>>」，
        # 缺 explanation 会让占位符被替换成空串（静默产出半截答案），必须拦在校验层。
        expl = rec.get("explanation")
        if not (isinstance(expl, str) and expl.strip()):
            crit(
                "E_CHOICE_NO_EXPLANATION",
                ".explanation",
                "choice / regime_trap 的答案模板需要 <<explanation>>，"
                f"但本条缺失或为空（得到 {expl!r}）",
            )

    elif task == "concept":
        ans = rec.get("answer")
        limit = task_cfg.get("max_chars", 120)
        if isinstance(ans, str) and len(ans) > limit:
            crit("E_CONCEPT_LONG", ".answer", f"concept 答案 {len(ans)} 字 > max_chars={limit}")

    elif task == "unanswerable":
        must = task_cfg.get("refusal_must_equal")
        if rec.get("answer") != must:
            crit(
                "E_REFUSAL_TEMPLATE",
                ".answer",
                f"不可答题必须使用统一拒答句式，应为 {must!r}，得到 {rec.get('answer')!r}",
            )

    elif task == "contrast":
        regs = spec.get("regimes", {})
        rb = rec.get("regime_b")
        if rb not in regs:
            crit("E_CONTRAST_REGIME", ".regime_b", f"regime_b 必填且必须属于 {sorted(regs)}，得到 {rb!r}")
        elif rb == rec.get("regime"):
            crit(
                "E_CONTRAST_REGIME",
                ".regime_b",
                f"regime_b 必须与 regime 不同（两者都是 {rb!r}）—— 对比题需要**两个**体制",
            )
        elif rb == "unspecified":
            crit("E_CONTRAST_REGIME", ".regime_b", "regime_b 不能是 unspecified（未指定的体制无从对比）")

    elif task == "clarify":
        must = task_cfg.get("clarify_must_contain") or []
        ans = rec.get("answer")
        if isinstance(ans, str) and must and not all(w in ans for w in must):
            crit(
                "E_CLARIFY_TEMPLATE",
                ".answer",
                f"clarify 的答案必须包含 {must}（明确要求对方补充体制信息），得到 {ans!r}",
            )

    # --- keypoints（半硬题的要点集）：让"简答题"变成可自动判分 ---
    # 设计要点：要点必须是 answer 的**子串**。
    #   理由：keypoints 是判分器要用关键词去匹配参考答案的依据；若某个要点在
    #   参考答案里都不出现，判分器就会对任何输出都判不过 —— 属于**静默失效**
    #   （指标恒为 0，但流程不报错）。所以把它升级成 critical 断言。
    kp_cfg = task_cfg.get("keypoints")
    if kp_cfg:
        kps = rec.get("keypoints")
        # ⚠️ 参照文本按题型选：选择题的答案只有**一个字母**，要点当然不可能出现在里面；
        #    它要校验的是「**解析**里必须说到这些依据」。若不区分，
        #    给 choice 配 keypoints 会被 E_KEYPOINT_NOT_IN_ANSWER 无脑判死。
        _ref_field = "explanation" if task in ("choice", "regime_trap") else "answer"
        ref_for_kp = rec.get(_ref_field)
        if not isinstance(kps, list) or not kps or not all(
            isinstance(k, str) and k.strip() for k in kps
        ):
            crit("E_KEYPOINTS", ".keypoints", f"task={task} 的 keypoints 必须是非空字符串列表，得到 {kps!r}")
        else:
            lo, hi = int(kp_cfg.get("min", 1)), int(kp_cfg.get("max", 99))
            if not (lo <= len(kps) <= hi):
                crit("E_KEYPOINTS_COUNT", ".keypoints", f"keypoints 数量 {len(kps)} 不在 [{lo}, {hi}] 内")
            if len(set(kps)) != len(kps):
                crit("E_KEYPOINTS_DUP", ".keypoints", f"keypoints 有重复项：{kps!r}")
            if isinstance(ref_for_kp, str):
                missing = [k for k in kps if k not in ref_for_kp]
                if missing:
                    crit(
                        "E_KEYPOINT_NOT_IN_ANSWER",
                        ".keypoints",
                        f"要点 {missing} 不是 {_ref_field} 的子串 —— 判分器会永远判不过（静默失效）。"
                        f"请让 {_ref_field} 覆盖这些要点，或改用其中真实出现的措辞",
                    )

    # --- 术语口径（可选增强）：把题目挂到术语表上，便于跨体制核对与统计 ---
    # ⚠️ 两项检查必须**独立**执行，不能写成 elif 链：否则"该体制下没有口径"这条
    #    warning 会顺带把后面那条 critical 的评测集准入检查**吞掉**（静默放行）。
    tk = rec.get("term_key")
    if tk and tk not in term_registry.TERMS:
        crit("E_TERM_UNKNOWN", ".term_key", f"term_key {tk!r} 不在术语表；可用：{term_registry.term_keys()}")
    elif tk:
        t = term_registry.TERMS[tk]
        rg_name = rg if isinstance(rg, str) else None
        if rg_name not in (None, "universal", "unspecified") and rg_name not in t.by_regime:
            warn(
                "W_TERM_UNCOVERED",
                ".term_key",
                f"术语 {tk}（{t.zh}）在体制 {rg_name} 下**尚无口径记录** —— "
                f"建议补进 `src/dataset/term_registry.py`",
            )
        if split == "eval" and not term_registry.is_verified_for(tk, rg_name):
            crit(
                "E_TERM_UNVERIFIED",
                ".term_key",
                f"术语 {tk}（{t.zh}）在 regime={rg_name!r} 这一层的口径**尚未与语料交叉核对** —— "
                f"可用于训练集（candidate），**不得进入评测集**（§9.0 纪律）。"
                f"核对后把 `verified` 改为 true 并填 `source` 即可解锁",
            )

    # --- 出处：**评测集必填**（§5.2 可回溯），训练集可空（docs/05 §5.3）---
    # 训练集不报 warning：日志噪音会掩盖真正的警告（P0 教训）。
    if split == "eval" and not (rec.get("evidence") or {}).get("source"):
        crit("E_EVAL_NO_SOURCE", ".evidence.source", "评测集样本必须有权威出处（§5.2 可回溯）")
    if modality == "vision" and not rec.get("image"):
        crit("E_VISION_NO_IMAGE", ".image", "modality=vision 必须有 image 字段")

    if modality == "vision" and isinstance(rec.get("image"), dict):
        issues.extend(check_image(rec["image"], f"{rid}.image", spec))

    return issues


# ---------------------------------------------------------------------------
# 编译后格式校验
# ---------------------------------------------------------------------------
def validate_compiled(rec: dict[str, Any], spec: dict[str, Any] | None = None) -> list[Issue]:
    spec = spec or load_spec()
    issues: list[Issue] = []
    rid = rec.get("id", "<no-id>")

    for p, txt in _iter_text_fields(rec):
        for iss in scan_text(txt, spec):
            issues.append(Issue(iss.severity, iss.code, f"{rid}{p[1:]}", iss.message))

    msgs = rec.get("messages")
    if not (isinstance(msgs, list) and msgs):
        issues.append(Issue("critical", "E_MESSAGES", f"{rid}.messages", "messages 必须是非空列表"))
        return issues

    roles = [m.get("role") for m in msgs if isinstance(m, dict)]
    if len(roles) != len(msgs):
        issues.append(Issue("critical", "E_MSG_SHAPE", f"{rid}.messages", "每个元素必须是 {role, content}"))

    body = roles[1:] if roles and roles[0] == "system" else roles
    if not body or body[-1] != "assistant":
        issues.append(Issue("critical", "E_MSG_TAIL", f"{rid}.messages", f"最后一条必须是 assistant，得到 {body[-1:]!r}"))
    expect = ["user", "assistant"] * ((len(body) + 1) // 2)
    if body != expect[: len(body)]:
        issues.append(Issue("critical", "E_MSG_ORDER", f"{rid}.messages", f"user/assistant 必须交替，得到 {body!r}"))

    for idx, m in enumerate(msgs):
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if m.get("role") == "assistant":
            if isinstance(c, str) and not c.strip():
                issues.append(Issue("critical", "E_EMPTY_ANSWER", f"{rid}.messages[{idx}].content", "assistant 内容为空"))
        if isinstance(c, list):
            n_img = sum(1 for x in c if isinstance(x, dict) and x.get("type") == "image")
            if n_img > spec["model"]["context"]["max_images_per_sample"]:
                issues.append(
                    Issue(
                        "critical",
                        "E_TOO_MANY_IMAGES",
                        f"{rid}.messages[{idx}].content",
                        f"{n_img} 张图 > max_images_per_sample={spec['model']['context']['max_images_per_sample']}",
                    )
                )

    ac = rec.get("answer_check")
    if not isinstance(ac, dict):
        issues.append(Issue("critical", "E_ANSWER_CHECK", f"{rid}.answer_check", "缺少 answer_check（无法机器判分）"))
    else:
        t = ac.get("type")
        if t not in ("numeric", "symbolic", "choice", "refusal", "keyword"):
            issues.append(Issue("critical", "E_CHECK_TYPE", f"{rid}.answer_check.type", f"非法判分类型 {t!r}"))
        if t == "numeric":
            if ac.get("value") is None:
                issues.append(Issue("critical", "E_CHECK_VALUE", f"{rid}.answer_check.value", "numeric 判分必须有 value"))
            if not ac.get("unit"):
                issues.append(Issue("critical", "E_CHECK_UNIT", f"{rid}.answer_check.unit", "numeric 判分必须有 unit（R2）"))
            tol = ac.get("tol")
            if not isinstance(tol, (int, float)) or not (0 < float(tol) < 1):
                issues.append(Issue("critical", "E_CHECK_TOL", f"{rid}.answer_check.tol", f"tol 必须 ∈ (0,1)，得到 {tol!r}"))

        # 声明了 keypoints 的题型，编译产物里必须把要点集带出来，
        # 否则判分器拿不到要点 —— 又是一类"看起来生成了、其实判不了"的静默失效。
        if spec["tasks"].get(rec.get("task"), {}).get("keypoints"):
            kps = ac.get("keypoints")
            if not (isinstance(kps, list) and kps):
                issues.append(
                    Issue(
                        "critical",
                        "E_CHECK_KEYPOINTS",
                        f"{rid}.answer_check.keypoints",
                        "该题型要求 keypoints，但 answer_check 里没有要点集（无法自动判分）",
                    )
                )
            elif spec["tasks"][rec["task"]].get("answer_mode") == "computed":
                # ⚠️ 计算题（calc）的 answer 是**程序生成的**，作者格式里没有 `answer` 字段，
                #    所以 validate_authoring 里那条"要点必须是 answer 子串"的检查**对它根本不执行**。
                #    → 在这里用**编译后的真实 assistant 内容**补上这条断言。
                #    不加这一条，"给 calc 配 keypoints"就等于"配了也没人验"（静默失效）。
                asst = ""
                for m in msgs:
                    if isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("content"), str):
                        asst = m["content"]
                missing = [k for k in kps if isinstance(k, str) and k not in asst]
                if missing:
                    issues.append(
                        Issue(
                            "critical",
                            "E_KEYPOINT_NOT_IN_ANSWER",
                            f"{rid}.answer_check.keypoints",
                            f"要点 {missing} 不在**编译后的答案**里 —— 判分器会永远判不过（静默失效）。"
                            f"注意：计算题的答案是程序生成的（公式/代入/最终答案三行），"
                            f"**不要把 LaTeX 当要点**（表述多样、极易漏配），要点应取语义要点",
                        )
                    )

    return issues


# ---------------------------------------------------------------------------
# 批次级校验
# ---------------------------------------------------------------------------
def validate_batch(records: list[dict[str, Any]], spec: dict[str, Any] | None = None) -> list[Issue]:
    spec = spec or load_spec()
    issues: list[Issue] = []

    seen: dict[str, int] = {}
    for i, r in enumerate(records):
        rid = r.get("id")
        if rid in seen:
            issues.append(Issue("critical", "E_DUP_ID", f"record[{i}].id", f"id {rid!r} 与 record[{seen[rid]}] 重复"))
        else:
            seen[rid] = i

    n = len(records)
    if n:
        from collections import Counter

        c = Counter(r.get("subdomain") for r in records)
        for sd, k in c.most_common(1):
            if k / n > 0.4:
                issues.append(
                    Issue(
                        "warning",
                        "W_QUOTA",
                        "batch",
                        f"子方向 {sd} 占 {k}/{n} = {k / n:.0%} > 40%，分布失衡（R7 子方向配额）",
                    )
                )

        # 体制维度：universal 应占多数，否则数据集难以归因
        # （§5.9「只改一个变量」：混着多个体制时，微调提升说不清来自哪个体制）
        if "regimes" in spec:
            n_univ = sum(1 for r in records if r.get("regime") == "universal")
            if n_univ / n < 0.3:
                issues.append(
                    Issue(
                        "warning",
                        "W_REGIME_SPREAD",
                        "batch",
                        f"universal 仅占 {n_univ}/{n} = {n_univ / n:.0%} < 30%："
                        f"体制相关题占比过高，会让「微调带来了什么」无法归因",
                    )
                )
            # 只统计**具体体制**（universal 不是需要配额的"体制"）；
            # ⚠️ 必须容忍 regime 为 None/非法值 —— 批次校验跑在逐条校验之外，
            #    不能因为一条脏数据就崩掉整个批次（否则会掩盖真正的 critical）。
            specific = sorted(
                {
                    r.get("regime")
                    for r in records
                    if r.get("regime") not in (None, "universal", "unspecified")
                },
                key=str,
            )
            if len(specific) > 3:
                issues.append(
                    Issue(
                        "warning",
                        "W_REGIME_TOO_MANY",
                        "batch",
                        f"本批涉及 {len(specific)} 个具体体制 {specific} —— "
                        f"V0 建议只做 2–3 个真实差距大的体制，否则样本需求会组合爆炸",
                    )
                )

            # clarify 配额：过多会让模型"过度澄清"（对明确的问题也反问）
            n_clarify = sum(1 for r in records if r.get("task") == "clarify")
            if n_clarify / n > 0.15:
                issues.append(
                    Issue(
                        "warning",
                        "W_CLARIFY_SHARE",
                        "batch",
                        f"clarify 占 {n_clarify}/{n} = {n_clarify / n:.0%} > 15% —— "
                        f"过多会让模型对本来明确的问题也反问（SFT 的典型副作用）",
                    )
                )

            # 打破"关键词捷径"：有多个体制时，必须存在 隐含/纠错/对比 类问法
            styles_used = {r.get("ask_style") for r in records}
            if specific and not (styles_used & {"implicit", "correct", "contrast"}):
                issues.append(
                    Issue(
                        "warning",
                        "W_NO_SHORTCUT_BREAKER",
                        "batch",
                        "本批有多个具体体制，但没有任何 implicit / correct / contrast 问法 —— "
                        "模型可能学成「看到体制词就答固定公式」的关键词捷径",
                    )
                )
    return issues


def format_issues(issues: list[Issue], *, limit: int = 60) -> str:
    if not issues:
        return "  （无问题）"
    lines = [f"  {i}" for i in issues[:limit]]
    if len(issues) > limit:
        lines.append(f"  … 另有 {len(issues) - limit} 条未显示")
    return "\n".join(lines)
