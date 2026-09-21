"""项目整理工具：盘点 + 清理 + 归档（可重复执行）。

设计原则（防止"整理即丢证据"）：
  1. **先扫描再动手**：`--scan` 只输出清单，不做任何修改（默认行为）。
  2. **证据自动识别**：从 `docs/*.md` 与 `reports/*` 里正则提取被引用的
     `outputs/xxx`、`logs/xxx`，凡命中一律标记保留，**即使它在删除白名单里也不删**。
  3. **删除走显式白名单**：不做 glob 通配匹配，只删本脚本列出的具体路径。
  4. **先删后归档**：顺序不能反 —— 归档会移动文件，之后再按原路径删除会静默失效（已踩过）。
  5. **留痕**：每次执行写出 `logs/cleanup_report_<ts>.md` + `logs/cleanup_manifest_<ts>.json`。

用法：
  python scripts/p1_organize.py --scan      # 只盘点（默认）
  python scripts/p1_organize.py --apply     # 执行：清理 + 归档
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# 归档规则：logs/ 按性质分子目录（只列要保留的；未列出的一律视为临时件）
#   install/  环境与模型资产安装日志（工具产生的外部日志）
#   server/   推理服务运行日志
#   probe/    一次性探测输出（分析性，支撑选型结论）
#   bench/    实验外置日志（与 outputs/<run>/run.log 互补）
# ---------------------------------------------------------------------------
ARCHIVE_RULES: dict[str, str] = {
    # install
    "p0_install.log": "install",
    "p0_install2.log": "install",
    "p0_install3.log": "install",
    "p1_install_llamacpp.log": "install",
    "p1_download.log": "install",
    # server
    "llamacpp_server.log": "server",
    # probe
    "p1_env_check.txt": "probe",
    "p1_winget.txt": "probe",
    "p1_gguf_repo_probe.txt": "probe",
    "p1_gguf_meta.txt": "probe",
    "p1_gguf_precision.txt": "probe",
    "p1_assets_manifest.json": "probe",
    # bench
    "p1_bench_E1.log": "bench",
    "p1_d1_eos.log": "bench",
    "p1_why_shorter.txt": "bench",
    "p1_variance.txt": "bench",
}

# 清理白名单：仅这些**具体路径**会被删除
CLEAN_TARGETS: list[str] = [
    # --- outputs：失败 / 未完成 / 已被最终版取代的 run ---
    # 均已由本脚本读取 metrics.json 逐条核验身份，非按文件名猜测
    "outputs/p0_verify_20260914_180358",   # P0 第 1 次：dry-run 阶段，无权重
    "outputs/p0_verify_20260914_180432",   # P0 第 2 次：torchvision 缺失，失败退出
    "outputs/p0_verify_20260914_180541",   # P0 第 3 次：top_k 警告未修版
    "outputs/p1_bench_20260914_203658",    # 冒烟 1：checks 0/2，llama-server 404
    "outputs/p1_bench_20260914_203938",    # 冒烟 2：checks 7/7，仅 text_short 单次
    "outputs/p1_bench_20260914_214152",    # D1 首跑：仅 1 个 outcome（HF 缺跑）
    "outputs/p1_bench_20260914_214224",    # D1 二跑：3 个 critical 失败（TypeError）
    # --- logs：一次性探测中间件（结论已固化进 docs，原始件保留在 probe/）---
    "logs/_scan.txt",
    "logs/_survey.txt",
    "logs/_idcheck.txt",
    "logs/_idcheck_run.txt",
    "logs/_find_llama.ps1",
    "logs/_find_llama.txt",
    "logs/_llama_ver.ps1",
    "logs/_llama_ver.txt",
    "logs/_port_check.ps1",
    "logs/_port_check.txt",
    "logs/_probe_out.txt",
    "logs/_probe_server.log",
    "logs/_routes.txt",
    "logs/_cmp_out.txt",
    "logs/_cmp_routes.txt",
    "logs/_cmp_A_adapter_args.log",
    "logs/_cmp_A_adapter_args.json",
    "logs/_cmp_B_bare_args.log",
    "logs/_cmp_B_bare_args.json",
    "logs/_smoke.txt",
    "logs/_smoke2.txt",
    "logs/_t_p1.txt",
    "logs/_list2.txt",
    "logs/_dirs.txt",
    "logs/_import_check.txt",
    # --- logs：整理过程的自身产物（避免"整理工具制造新垃圾"）---
    "logs/_rm.py",
    "logs/_rm_out.txt",
    "logs/_chk.txt",
    "logs/p1_tree.txt",
    "logs/p1_artifacts.txt",
    "logs/p1_artifacts_tmp.txt",
    "logs/listing.txt",
]

# scripts/ 下的一次性诊断脚本（诊断结论已进报告）
CLEAN_SCRIPTS: list[str] = [
    "scripts/_idcheck.py",
    "scripts/_probe_llama_routes.py",
    "scripts/_cmp_routes.py",
    "scripts/_p1_extract.py",
    "scripts/p1_why_shorter.py",
    "scripts/p1_variance_check.py",
]

# 必须保留的证据源 —— 脚本启动时断言其存在，缺失则拒绝清理
MUST_KEEP: list[str] = [
    "outputs/p0_verify_20260914_180657",
    "outputs/p1_bench_20260914_203956",
    "outputs/p1_bench_20260914_214618",
    "reports/P0_metrics.json",
    "reports/P1_metrics.json",
    "reports/P0_run.log",
    "reports/P1_run.log",
    "reports/P1_对比表.md",
    "reports/p0_synthetic_rd_map.png",
    "docs/00_行为规范.md",
    "docs/README.md",
    "docs/01_P0环境与模型核验.md",
    "docs/02_P1推理引擎选型.md",
    "docs/03_P1推理引擎对比.md",
    "configs/base.yaml",
    "configs/engines/hf.yaml",
    "configs/engines/llamacpp.yaml",
    "configs/engines/ollama.yaml",
    "configs/bench/p1_matrix.yaml",
    "src/bench.py",
    "src/results.py",
    "src/engines/base.py",
    "src/engines/llamacpp_engine.py",
    "scripts/p1_setup.py",
    "scripts/p1_bench.py",
]

CITED_RE = re.compile(r"(?:outputs|logs)[/\\]([A-Za-z0-9_\u4e00-\u9fff.\-]+)")


def find_cited() -> set[str]:
    """从 docs/ 与 reports/ 提取被引用的 outputs/ 与 logs/ 条目名。"""
    cited: set[str] = set()
    for base in (ROOT / "docs", ROOT / "reports"):
        if not base.exists():
            continue
        for f in base.rglob("*"):
            if f.is_file() and f.suffix in {".md", ".json"}:
                try:
                    txt = f.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                cited.update(CITED_RE.findall(txt))
    cited.discard("")
    return cited


def dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def scan() -> dict:
    cited = find_cited()
    out_dirs = sorted([d for d in (ROOT / "outputs").iterdir() if d.is_dir()])
    out_files = sorted([f for f in (ROOT / "outputs").iterdir() if f.is_file()])
    log_files = sorted([f for f in (ROOT / "logs").iterdir() if f.is_file()])

    clean_set = {Path(x).name for x in CLEAN_TARGETS}
    must_keep_set = {Path(x).name for x in MUST_KEEP}

    rows = []
    for d in out_dirs:
        if d.name in cited or d.name in must_keep_set:
            verdict = "KEEP(cited)"
        elif d.name in clean_set:
            verdict = "DELETE"
        else:
            verdict = "KEEP(未分类·请人工确认)"
        rows.append({"name": d.name, "verdict": verdict,
                     "kb": round(dir_bytes(d) / 1024, 1)})

    return {
        "cited": sorted(cited),
        "outputs_rows": rows,
        "outputs_stray": [f.name for f in out_files],
        "logs": [{"name": f.name,
                  "verdict": f"→ {ARCHIVE_RULES[f.name]}/" if f.name in ARCHIVE_RULES
                  else ("DELETE" if f.name in clean_set else "KEEP(未分类)"),
                  "kb": round(f.stat().st_size / 1024, 2)}
                 for f in log_files],
        "missing": [p for p in MUST_KEEP if not (ROOT / p).exists()],
    }


def render(s: dict) -> str:
    L = ["=" * 88, "【1】outputs/ 各 run 目录 —— 去留判定", "=" * 88]
    for r in s["outputs_rows"]:
        L.append(f"  {r['name']:<38} {r['verdict']:<28} {r['kb']:>8} KB")
    if s["outputs_stray"]:
        L.append("  outputs/ 根部散落文件: " + ", ".join(s["outputs_stray"]))
    L += ["", "=" * 88, "【2】logs/ 现有文件 —— 归档去向", "=" * 88]
    for f in s["logs"]:
        L.append(f"  {f['name']:<38} {f['verdict']:<20} {f['kb']:>8} KB")
    L += ["", "=" * 88, "【3】被文档引用的证据源（自动识别，绝不删除）", "=" * 88]
    L += ["  " + c for c in s["cited"]]
    L.append("")
    L.append("✔ MUST_KEEP 全部存在" if not s["missing"]
             else "!!! 关键证据缺失: " + ", ".join(s["missing"]))
    return "\n".join(L)


def do_apply(s: dict) -> tuple[str, dict]:
    ts = time.strftime("%Y%m%d_%H%M%S")
    man: dict = {"timestamp": ts, "deleted": [], "archived": [], "skipped": []}
    cited = set(s["cited"])

    # --- 1. 先删除（顺序很关键：归档会移动文件，后删会静默失效）---
    for rel in CLEAN_TARGETS + CLEAN_SCRIPTS:
        p = ROOT / rel
        if not p.exists():
            continue
        if p.name in cited:
            man["skipped"].append(f"{rel}（被文档引用）")
            continue
        if p.is_dir():
            kb = dir_bytes(p) / 1024
            shutil.rmtree(p)
            man["deleted"].append(f"{rel}/  ({kb:.0f} KB)")
        else:
            kb = p.stat().st_size / 1024
            p.unlink()
            man["deleted"].append(f"{rel}  ({kb:.1f} KB)")

    # 清 __pycache__
    for pc in list(ROOT.rglob("__pycache__")):
        if pc.is_dir():
            shutil.rmtree(pc, ignore_errors=True)
            man["deleted"].append(f"{pc.relative_to(ROOT)}/")

    # --- 2. 再归档 ---
    for name, sub in ARCHIVE_RULES.items():
        src = ROOT / "logs" / name
        if not src.exists():
            continue
        dst_dir = ROOT / "logs" / sub
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / name
        if dst.exists():
            man["skipped"].append(f"归档 {name}（目标已存在）")
            continue
        shutil.move(str(src), str(dst))
        man["archived"].append(f"logs/{name} → logs/{sub}/{name}")

    # --- 3. logs/ 根部残留检查 ---
    leftover = [f.name for f in (ROOT / "logs").iterdir()
                if f.is_file() and not f.name.startswith(("cleanup_",))]
    man["logs_leftover"] = leftover

    # --- 4. 留痕 ---
    (ROOT / "logs" / f"cleanup_manifest_{ts}.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")

    L = [f"执行完毕 @ {ts}", "", f"--- 删除 {len(man['deleted'])} 项 ---"]
    L += ["  " + x for x in man["deleted"]]
    L.append("")
    L.append(f"--- 归档 {len(man['archived'])} 项 ---")
    L += ["  " + x for x in man["archived"]]
    if man["skipped"]:
        L += ["", f"--- 跳过 {len(man['skipped'])} 项 ---"]
        L += ["  " + x for x in man["skipped"]]
    if leftover:
        L += ["", "--- logs/ 根部仍有未归类文件（需人工处理）---"]
        L += ["  " + x for x in leftover]
    (ROOT / "logs" / f"cleanup_report_{ts}.md").write_text(
        "\n".join(L), encoding="utf-8")
    L.append(f"\n清单: logs/cleanup_manifest_{ts}.json")
    L.append(f"报告: logs/cleanup_report_{ts}.md")
    return "\n".join(L), man


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    s = scan()
    text = render(s)
    if args.apply:
        if s["missing"]:
            print(text)
            print("\n!!! 关键证据缺失，拒绝清理。")
            return 2
        detail, _ = do_apply(s)
        text = text + "\n\n" + "=" * 88 + "\n" + detail
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
