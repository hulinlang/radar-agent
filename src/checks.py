"""机器可执行的断言框架（对应 00_行为规范 §5.8）。

为什么需要它：**静默错误**（跑得通、数字看着正常但结果是错的）是唯一能毁掉
全部实验数据可信度、又不会自己暴露的错误类型。人工"看日志"不构成可靠防线。
所以把「人恰好注意到」升级成「机器必然报错」。

分级：
- critical：失败则结果不可信 → 脚本以非零退出码结束，不进入下一阶段。
- warning ：可疑，需人工确认；不阻断流程，但必须写入报告。

用法：
    from src.checks import CheckSuite
    ck = CheckSuite(logger)
    ck.check("param_count == 2.13e9", lambda: get_n_params(model) == 2127532032, critical=True)
    ...
    ck.assert_all()      # critical 有失败则抛 CriticalAssertionError
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable


class CriticalAssertionError(RuntimeError):
    """critical 级断言失败时抛出。调用方应让它冒泡到 main，以非零码退出。"""


@dataclass
class CheckResult:
    name: str
    passed: bool
    critical: bool
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "critical": self.critical,
            "detail": self.detail,
        }


@dataclass
class CheckSuite:
    """收集并汇报一组断言。"""

    logger: logging.Logger
    results: list[CheckResult] = field(default_factory=list)

    # ------------------------------------------------------------------
    def check(
        self,
        name: str,
        fn: Callable[[], Any],
        *,
        critical: bool = True,
        expect: Any = True,
        detail: str = "",
    ) -> bool:
        """执行 `fn()` 并断言返回值等于 `expect`。

        异常（而非返回 False）同样视为失败——异常信息会被记录下来，
        因为"断言自己崩了"也必须暴露，不能被静默吞掉。
        """
        try:
            actual = fn()
            passed = bool(actual == expect)
            extra = f"actual={actual!r} expect={expect!r}"
            if detail:
                extra = f"{detail} | {extra}"
            res = CheckResult(name, passed, critical, extra if not passed else detail)
        except Exception as exc:  # noqa: BLE001
            res = CheckResult(
                name, False, critical,
                f"断言执行抛异常: {type(exc).__name__}: {exc}" + (f" | {detail}" if detail else ""),
            )

        self.results.append(res)
        level = "warning" if not critical else "critical"
        if res.passed:
            self.logger.info("  [PASS] %s", name)
        else:
            msg = "  [FAIL][%s] %s | %s" % (level, name, res.detail)
            self.logger.error(msg) if critical else self.logger.warning(msg)
        return res.passed

    # ------------------------------------------------------------------
    def assert_all(self) -> None:
        """汇总裁决。critical 失败 → 抛 CriticalAssertionError。"""
        crit_fail = [r for r in self.results if r.critical and not r.passed]
        warn_fail = [r for r in self.results if not r.critical and not r.passed]
        total = len(self.results)
        self.logger.info(
            "断言汇总: %d/%d 通过（critical 失败 %d，warning 失败 %d）",
            total - len(crit_fail) - len(warn_fail), total, len(crit_fail), len(warn_fail),
        )
        if crit_fail:
            names = "; ".join(r.name for r in crit_fail)
            raise CriticalAssertionError(f"{len(crit_fail)} 条 critical 断言失败: {names}")

    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        return {
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "critical_failed": sum(1 for r in self.results if r.critical and not r.passed),
            "warning_failed": sum(1 for r in self.results if not r.critical and not r.passed),
            "details": [r.as_dict() for r in self.results],
        }
