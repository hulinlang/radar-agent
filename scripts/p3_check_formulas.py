"""验证新增公式：数量、可计算性、数值正确性（对照手算与语料给出的数值例子）。"""
import sys

sys.path.insert(0, r"F:\Qwen3-2B\radar-agent")
from src.dataset import formulas

print("公式总数: %d（原 11 + 新增 %d）" % (len(formulas.REGISTRY), len(formulas.REGISTRY) - 11))
print()
for n in formulas.formula_names():
    f = formulas.REGISTRY[n]
    print("  %-28s out=%-6s %s" % (n, f.output_unit, f.desc))

print()
print("=== 数值抽查（每条都可用手算复核）===")
checks = [
    ("velocity_resolution", {"lam": 0.03, "t": 0.02}, 0.75),
    ("range_from_delay", {"tau": 1e-6}, 150.0),
    ("velocity_from_doppler", {"lam": 0.03, "f_d": 1000.0}, 15.0),
    ("sar_azimuth_resolution", {"D": 2.0}, 1.0),
    ("range_ratio_from_rcs", {"sigma1": 10.0, "sigma2": 1.0}, 10 ** -0.25),
    ("prf_from_period", {"t_r": 1e-4}, 10000.0),
    ("chirp_slope", {"B": 4e9, "t_c": 1e-4}, 4e13),
    ("bandwidth_from_pulse_width", {"tau": 1e-6}, 1e6),
    ("snr_max_matched", {"E": 1e-6, "n0": 4e-21}, 5e14),
    ("adc_sample_rate", {"B": 3e9, "r_max": 200.0, "t_c": 50e-6}, 1.6e8),
    ("max_range_fmcw", {"f_s": 1.6e8, "t_c": 50e-6, "B": 3e9}, 200.0),
]
ok = True
for name, inp, expect in checks:
    r = formulas.evaluate(name, inp)
    good = abs(r["value"] - expect) / max(abs(expect), 1e-30) < 1e-9
    ok = ok and good
    print("  %-28s = %-14.6g %-6s 期望 %-12.6g %s" % (
        name, r["value"], r["unit"], expect, "OK" if good else "*** FAIL ***"))

print()
print("全部数值正确: %s" % ok)
print()
print("=== 与语料给出的数值例子对撞（mmWave fmcw.md「步骤 3」）===")
r = formulas.evaluate("adc_sample_rate", {"B": 3e9, "r_max": 200.0, "t_c": 50e-6})
print("  按文档自给参数 B=3GHz / R_max=200m / T_c=50us 算得 f_s = %.4g Hz" % r["value"])
print("  文档正文写的答案是 16 MHz")
print("  → 程序结果 %.0f MHz %s 文档值" % (
    r["value"] / 1e6, "==" if abs(r["value"] - 1.6e7) < 1e3 else "!="))
print("  反代验证：用 f_s=1.6e8 回算 R_max = %.1f m（应为 200 m）" % formulas.evaluate(
    "max_range_fmcw", {"f_s": 1.6e8, "t_c": 50e-6, "B": 3e9})["value"])
