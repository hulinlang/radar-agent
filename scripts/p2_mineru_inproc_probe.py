"""P2 · 诊断：**进程内**直接调用 MinerU，抓它真正抛出的异常。

为什么需要这个：
    MinerU 3.x 的 CLI 会先在子进程里起一个本地 FastAPI 服务，再由客户端轮询它。
    服务进程一旦被杀（本项目沙箱实测会在 multiprocessing / 文件清理处杀进程），
    客户端只能看到 `Failed to query task status: 404 Not Found` —— **真正的原因被吞掉了**。
    所以必须绕开 CLI，直接调 `aio_do_parse`，让异常在原地冒出来。

只读 PDF、只写输出目录。
"""

from __future__ import annotations

import asyncio
import inspect
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data_raw" / "机载雷达系统与信息处理_15097299.pdf"
OUT = ROOT / "data_processed" / "corpus" / "mineru_inproc_probe"

from mineru.cli import common as C  # noqa: E402


def main() -> None:
    # 允许从命令行给页码范围（PDF 页序，1-based，闭区间）
    lo, hi = 77, 78
    if len(sys.argv) >= 3:
        lo, hi = int(sys.argv[1]), int(sys.argv[2])

    print("=" * 78)
    print(f"进程内调用 MinerU（PDF 页序 p{lo}-p{hi}，0-based {lo-1}-{hi-1}）")
    print("=" * 78)
    OUT.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        pdf_file_names=[f"probe_p{lo}_{hi}"],
        pdf_bytes_list=[PDF.read_bytes()],
        p_lang_list=["ch"],
        output_dir=str(OUT),
        backend="pipeline",
        f_dump_md=True,
        f_dump_content_list=True,
        f_dump_middle_json=True,
        f_dump_model_output=False,
        f_draw_layout_bbox=False,
        f_draw_span_bbox=False,
        f_make_md_mode=C.MakeMode.MM_MD,
        start_page_id=lo - 1,   # 0-based，闭区间
        end_page_id=hi - 1,
    )
    try:
        asyncio.run(C.aio_do_parse(**kwargs))
        print("\n✓ 进程内调用成功")
    except Exception as exc:  # noqa: BLE001
        print(f"\n✗ 进程内调用抛出：{type(exc).__name__}: {exc}")
        print("\n--- 完整 Traceback ---")
        traceback.print_exc()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
