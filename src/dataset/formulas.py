"""雷达信号处理基础 · 公式注册表（真值的唯一来源）。

为什么要有这个注册表（而不是让人手写答案）：
    规范 §5.2 禁止编造指标，而"人写数值答案"是最容易出错的一环
    （连基线模型都会把 10 MHz 写成 10^8 Hz，见 docs/05 §八）。
    所以数值题的**答案一律由程序算出**，人只负责：
      ① 用人类的语言写"题面"；
      ② 指定用哪条公式、给哪些物理量参数。
    这样答案与题面天然自洽，且可机器复核。

设计要点：
    - 所有量一律 **SI 单位**（m / s / Hz / W / m^2 / 倍），不做单位换算魔法；
    - 参数单位显式声明，便于人工核对；
    - 常数（c / k / T0）给默认值，作者可省略，也可显式覆盖（覆盖会记录进 meta）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

C_LIGHT = 3.0e8  # m/s（教材与工程惯用近似，非 2.99792458e8）
K_BOLTZ = 1.38e-23  # J/K
T0_STD = 290.0  # K


@dataclass(frozen=True)
class Formula:
    name: str
    desc: str
    subdomain: str
    latex: str  # 符号形式的公式
    subst: str  # 代入模板，占位符 <<param>>
    fn: Callable[..., float]
    params: dict[str, str]  # 形参名 -> 单位（"" 表示无量纲）
    defaults: dict[str, float] = field(default_factory=dict)  # 可省略的常数
    output_unit: str = ""


def _f_wavelength(c: float, f0: float) -> float:
    return c / f0


def _f_range_resolution(c: float, B: float) -> float:
    return c / (2.0 * B)


def _f_range_resolution_tau(c: float, tau: float) -> float:
    return c * tau / 2.0


def _f_unambiguous_range(c: float, prf: float) -> float:
    return c / (2.0 * prf)


def _f_unambiguous_velocity(lam: float, prf: float) -> float:
    return lam * prf / 4.0


def _f_doppler_frequency(v: float, lam: float) -> float:
    return 2.0 * v / lam


def _f_doppler_resolution(t_cpi: float) -> float:
    return 1.0 / t_cpi


def _f_pulse_compression_gain(B: float, T: float) -> float:
    return B * T


def _f_pulse_compression_gain_db(B: float, T: float) -> float:
    import math

    return 10.0 * math.log10(B * T)


def _f_coherent_integration_gain_db(n: float) -> float:
    import math

    return 10.0 * math.log10(n)


def _f_noise_power(k: float, T0: float, B: float, F: float) -> float:
    return k * T0 * B * F


def _f_max_detect_range(
    pt: float, g: float, lam: float, sigma: float, pn: float, loss: float
) -> float:
    import math

    return ((pt * g * g * lam * lam * sigma) / ((4.0 * math.pi) ** 3 * pn * loss)) ** 0.25


# ---------------------------------------------------------------------------
# 2026-09-15 新增（D-P2-18）：注册表只有 11 条撑不住 S2 的 ≥80 条 calc
#   ⚠️ 纪律：**每条新公式都必须写明语料出处**（写在下方 REGISTRY 的注释里）。
#      没有出处的式子一律不进注册表 —— 否则等于"我们自己造了一个未经核验的公式"。
# ---------------------------------------------------------------------------


def _f_velocity_resolution(lam: float, t: float) -> float:
    return lam / (2.0 * t)


def _f_range_from_beat(c: float, f_beat: float, t_c: float, B: float) -> float:
    return c * f_beat * t_c / (2.0 * B)


def _f_max_range_fmcw(c: float, f_s: float, t_c: float, B: float) -> float:
    return c * f_s * t_c / (4.0 * B)


def _f_adc_sample_rate(B: float, r_max: float, c: float, t_c: float) -> float:
    return 4.0 * B * r_max / (c * t_c)


def _f_chirp_slope(B: float, t_c: float) -> float:
    return B / t_c


def _f_bandwidth_from_pulse_width(tau: float) -> float:
    return 1.0 / tau


def _f_snr_max_matched(E: float, n0: float) -> float:
    return 2.0 * E / n0


def _f_velocity_from_doppler(lam: float, f_d: float) -> float:
    return lam * f_d / 2.0


def _f_range_from_delay(c: float, tau: float) -> float:
    return c * tau / 2.0


def _f_sar_azimuth_resolution(D: float) -> float:
    return D / 2.0


def _f_range_ratio_from_rcs(sigma1: float, sigma2: float) -> float:
    return (sigma2 / sigma1) ** 0.25


def _f_prf_from_period(t_r: float) -> float:
    return 1.0 / t_r


REGISTRY: dict[str, Formula] = {
    f.name: f
    for f in [
        Formula(
            name="wavelength",
            desc="波长 = 光速 / 载频",
            subdomain="misc",
            latex=r"\lambda = \frac{c}{f_0}",
            subst=r"\lambda = \frac{<<c>>}{<<f0>>}",
            fn=_f_wavelength,
            params={"c": "m/s", "f0": "Hz"},
            defaults={"c": C_LIGHT},
            output_unit="m",
        ),
        Formula(
            name="range_resolution",
            desc="距离分辨率 = 光速 / (2 × 带宽)",
            subdomain="resolution",
            latex=r"\Delta R = \frac{c}{2B}",
            subst=r"\Delta R = \frac{<<c>>}{2 \times <<B>>}",
            fn=_f_range_resolution,
            params={"c": "m/s", "B": "Hz"},
            defaults={"c": C_LIGHT},
            output_unit="m",
        ),
        Formula(
            name="range_resolution_tau",
            desc="距离分辨率（由脉冲宽度）= 光速 × 脉宽 / 2",
            subdomain="resolution",
            latex=r"\Delta R = \frac{c\tau}{2}",
            subst=r"\Delta R = \frac{<<c>> \times <<tau>>}{2}",
            fn=_f_range_resolution_tau,
            params={"c": "m/s", "tau": "s"},
            defaults={"c": C_LIGHT},
            output_unit="m",
        ),
        Formula(
            name="unambiguous_range",
            desc="最大不模糊距离 = 光速 / (2 × 脉冲重复频率)",
            subdomain="ambiguity",
            latex=r"R_u = \frac{c}{2\,\mathrm{PRF}}",
            subst=r"R_u = \frac{<<c>>}{2 \times <<prf>>}",
            fn=_f_unambiguous_range,
            params={"c": "m/s", "prf": "Hz"},
            defaults={"c": C_LIGHT},
            output_unit="m",
        ),
        Formula(
            name="unambiguous_velocity",
            desc="最大不模糊速度 = 波长 × 脉冲重复频率 / 4",
            subdomain="ambiguity",
            latex=r"v_u = \frac{\lambda\,\mathrm{PRF}}{4}",
            subst=r"v_u = \frac{<<lam>> \times <<prf>>}{4}",
            fn=_f_unambiguous_velocity,
            params={"lam": "m", "prf": "Hz"},
            output_unit="m/s",
        ),
        Formula(
            name="doppler_frequency",
            desc="多普勒频移 = 2 × 径向速度 / 波长",
            subdomain="doppler",
            latex=r"f_d = \frac{2v}{\lambda}",
            subst=r"f_d = \frac{2 \times <<v>>}{<<lam>>}",
            fn=_f_doppler_frequency,
            params={"v": "m/s", "lam": "m"},
            output_unit="Hz",
        ),
        Formula(
            name="doppler_resolution",
            desc="多普勒分辨率 = 1 / 相干处理时间",
            subdomain="doppler",
            latex=r"\Delta f_d = \frac{1}{T_{\mathrm{CPI}}}",
            subst=r"\Delta f_d = \frac{1}{<<t_cpi>>}",
            fn=_f_doppler_resolution,
            params={"t_cpi": "s"},
            output_unit="Hz",
        ),
        Formula(
            name="pulse_compression_gain",
            desc="脉冲压缩增益 = 时宽带宽积 BT（倍）",
            subdomain="pulse_compression",
            latex=r"G_{\mathrm{pc}} = B\,T",
            subst=r"G_{\mathrm{pc}} = <<B>> \times <<T>>",
            fn=_f_pulse_compression_gain,
            params={"B": "Hz", "T": "s"},
            output_unit="倍",
        ),
        Formula(
            name="pulse_compression_gain_db",
            desc="脉冲压缩增益（dB）= 10·log10(BT)",
            subdomain="pulse_compression",
            latex=r"G_{\mathrm{pc,dB}} = 10\log_{10}(BT)",
            subst=r"G_{\mathrm{pc,dB}} = 10\log_{10}(<<B>> \times <<T>>)",
            fn=_f_pulse_compression_gain_db,
            params={"B": "Hz", "T": "s"},
            output_unit="dB",
        ),
        Formula(
            name="coherent_integration_gain_db",
            desc="相干积累增益（dB）= 10·log10(N)",
            subdomain="pulse_compression",
            latex=r"G_{\mathrm{ci,dB}} = 10\log_{10}N",
            subst=r"G_{\mathrm{ci,dB}} = 10\log_{10}(<<n>>)",
            fn=_f_coherent_integration_gain_db,
            params={"n": "个"},
            output_unit="dB",
        ),
        Formula(
            name="noise_power",
            desc="接收机噪声功率 = k·T0·B·F（F 为线性噪声系数）",
            subdomain="radar_equation",
            latex=r"P_n = k T_0 B F",
            subst=r"P_n = <<k>> \times <<T0>> \times <<B>> \times <<F>>",
            fn=_f_noise_power,
            params={"k": "J/K", "T0": "K", "B": "Hz", "F": "倍"},
            defaults={"k": K_BOLTZ, "T0": T0_STD},
            output_unit="W",
        ),
        Formula(
            name="max_detect_range",
            desc="雷达方程：最大探测距离（四次方根形式）",
            subdomain="radar_equation",
            latex=(
                r"R_{\max} = \left[\frac{P_t G^2 \lambda^2 \sigma}"
                r"{(4\pi)^3 P_n L}\right]^{1/4}"
            ),
            subst=(
                r"R_{\max} = \left[\frac{<<pt>> \times <<g>>^2 \times <<lam>>^2 \times <<sigma>>}"
                r"{(4\pi)^3 \times <<pn>> \times <<loss>>}\right]^{1/4}"
            ),
            fn=_f_max_detect_range,
            params={
                "pt": "W",
                "g": "倍",
                "lam": "m",
                "sigma": "m^2",
                "pn": "W",
                "loss": "倍",
            },
            output_unit="m",
        ),
        # =====================================================================
        # ↓↓↓ 2026-09-15 新增 12 条（D-P2-18）↓↓↓
        # 每条的【出处】都写在注释里；没有出处的式子不进注册表。
        # =====================================================================
        Formula(
            name="velocity_resolution",
            desc="径向速度分辨率 = 波长 / (2 × 信号持续时间)",
            subdomain="doppler",
            latex=r"\delta_v = \frac{\lambda}{2T}",
            subst=r"\delta_v = \frac{<<lam>>}{2 \times <<t>>}",
            fn=_f_velocity_resolution,
            params={"lam": "m", "t": "s"},
            output_unit="m/s",
        ),  # 【出处】教材 p82 式(2.3.28)：δ_v = λ·δ_fD/2 = λ/(2T)
        Formula(
            name="range_from_beat",
            desc="FMCW 测距 = 光速 × 拍频 × Chirp 周期 / (2 × 扫频带宽)",
            subdomain="resolution",
            latex=r"R = \frac{c\,f_{\mathrm{beat}}\,T_c}{2B}",
            subst=r"R = \frac{<<c>> \times <<f_beat>> \times <<t_c>>}{2 \times <<B>>}",
            fn=_f_range_from_beat,
            params={"c": "m/s", "f_beat": "Hz", "t_c": "s", "B": "Hz"},
            defaults={"c": C_LIGHT},
            output_unit="m",
        ),  # 【出处】mmWave fmcw.md「步骤 4：计算距离」R = c·f_beat·T_c/(2B)
        Formula(
            name="max_range_fmcw",
            desc="FMCW 最大可测距离 = 光速 × 采样率 × Chirp 周期 / (4 × 扫频带宽)",
            subdomain="ambiguity",
            latex=r"R_{\max} = \frac{c\,f_s\,T_c}{4B}",
            subst=r"R_{\max} = \frac{<<c>> \times <<f_s>> \times <<t_c>>}{4 \times <<B>>}",
            fn=_f_max_range_fmcw,
            params={"c": "m/s", "f_s": "Hz", "t_c": "s", "B": "Hz"},
            defaults={"c": C_LIGHT},
            output_unit="m",
        ),  # 【出处】mmWave fmcw.md「为什么有最大距离限制？」R_max = c·f_s·T_c/(4B)
        Formula(
            name="adc_sample_rate",
            desc="FMCW 所需 ADC 采样率 = 4 × 带宽 × 最大距离 / (光速 × Chirp 周期)",
            subdomain="ambiguity",
            latex=r"f_s = \frac{4 B R_{\max}}{c T_c}",
            subst=r"f_s = \frac{4 \times <<B>> \times <<r_max>>}{<<c>> \times <<t_c>>}",
            fn=_f_adc_sample_rate,
            params={"B": "Hz", "r_max": "m", "c": "m/s", "t_c": "s"},
            defaults={"c": C_LIGHT},
            output_unit="Hz",
        ),  # 【出处】mmWave fmcw.md「步骤 3」f_s = 4BR_max/(cT_c)
        #   ⚠️ **该文档此处的答案是错的**：按其自给参数 B=3GHz / R_max=200m / T_c=50μs
        #      算得 f_s = **160 MHz**，而文档正文写 16 MHz（差 10 倍）。
        #      反代验证：f_s=160MHz → R_max = c·f_s·T_c/(4B) = 200 m ✓ 与题设自洽。
        #      → 与教材 p42 的"RCS 减到 1/10 → 距离降 50%"（应为 56.2%）是**同一类问题**：
        #        外部资料也会算错。出题时以本公式为准，不要照抄文档给的数值。
        Formula(
            name="chirp_slope",
            desc="Chirp 调频斜率 = 扫频带宽 / Chirp 周期",
            subdomain="pulse_compression",
            latex=r"S = \frac{B}{T_c}",
            subst=r"S = \frac{<<B>>}{<<t_c>>}",
            fn=_f_chirp_slope,
            params={"B": "Hz", "t_c": "s"},
            output_unit="Hz/s",
        ),  # 【出处】mmWave fmcw.md 参数表：S = B/T_c（文档例子：40 MHz/μs）
        Formula(
            name="bandwidth_from_pulse_width",
            desc="单频矩形脉冲的频谱宽度 = 1 / 脉冲宽度",
            subdomain="resolution",
            latex=r"B = \frac{1}{t_p}",
            subst=r"B = \frac{1}{<<tau>>}",
            fn=_f_bandwidth_from_pulse_width,
            params={"tau": "s"},
            output_unit="Hz",
        ),  # 【出处】教材 p82：单频矩形脉冲"频谱宽度为 1/t_p"
        Formula(
            name="snr_max_matched",
            desc="匹配滤波最大输出信噪比 = 2 × 信号能量 / 噪声单边功率谱密度",
            subdomain="pulse_compression",
            latex=r"\mathrm{SNR}_{\max} = \frac{2E}{N_0}",
            subst=r"\mathrm{SNR}_{\max} = \frac{2 \times <<E>>}{<<n0>>}",
            fn=_f_snr_max_matched,
            params={"E": "J", "n0": "W/Hz"},
            output_unit="倍",
        ),  # 【出处】教材 p76 式(2.2.29)：SNR_max = 2E/N0
        Formula(
            name="velocity_from_doppler",
            desc="径向速度 = 波长 × 多普勒频移 / 2",
            subdomain="doppler",
            latex=r"v = \frac{\lambda f_d}{2}",
            subst=r"v = \frac{<<lam>> \times <<f_d>>}{2}",
            fn=_f_velocity_from_doppler,
            params={"lam": "m", "f_d": "Hz"},
            output_unit="m/s",
        ),  # 【出处】教材 p79：Δν = λ·f_D/2（速度差与多普勒差的对应）
        Formula(
            name="range_from_delay",
            desc="目标距离 = 光速 × 回波时延 / 2",
            subdomain="resolution",
            latex=r"R = \frac{c\tau}{2}",
            subst=r"R = \frac{<<c>> \times <<tau>>}{2}",
            fn=_f_range_from_delay,
            params={"c": "m/s", "tau": "s"},
            defaults={"c": C_LIGHT},
            output_unit="m",
        ),  # 【出处】教材 p79："两个距离上相差 ΔR = cτ/2"
        Formula(
            name="sar_azimuth_resolution",
            desc="合成阵列横向（方位向）距离分辨单元 = 天线孔径 / 2",
            subdomain="resolution",
            latex=r"\Delta R_{\mathrm{az}} = \frac{D}{2}",
            subst=r"\Delta R_{\mathrm{az}} = \frac{<<D>>}{2}",
            fn=_f_sar_azimuth_resolution,
            params={"D": "m"},
            output_unit="m",
        ),  # 【出处】教材 p310："合成阵列……横向距离分辨单元长度为 D/2，与目标距离远近无关"
        Formula(
            name="range_ratio_from_rcs",
            desc="雷达方程：作用距离比 = (RCS 比) 的 4 次方根",
            subdomain="radar_equation",
            latex=r"\frac{R_2}{R_1} = \left(\frac{\sigma_2}{\sigma_1}\right)^{1/4}",
            subst=r"\frac{R_2}{R_1} = \left(\frac{<<sigma2>>}{<<sigma1>>}\right)^{1/4}",
            fn=_f_range_ratio_from_rcs,
            params={"sigma1": "m^2", "sigma2": "m^2"},
            output_unit="倍",
        ),  # 【出处】教材 p42："雷达发现目标的距离与雷达反射截面积的 4 次方根成正比"
        #   ⚠️ 教材 p42 接着写"RCS 减到 1/10 → 距离降为 50%"，与四次方根**不符**（应为 56.2%）；
        #      教材此处是错的/取了近似。出题时以本公式为准，并在 notes 里标出冲突（docs/06 §2.5）。
        Formula(
            name="prf_from_period",
            desc="脉冲重复频率 = 1 / 脉冲重复周期",
            subdomain="ambiguity",
            latex=r"\mathrm{PRF} = \frac{1}{T_r}",
            subst=r"\mathrm{PRF} = \frac{1}{<<t_r>>}",
            fn=_f_prf_from_period,
            params={"t_r": "s"},
            output_unit="Hz",
        ),  # 【出处】教材 p142："因为其脉冲往返时间等于脉冲重复周期，所以 R_u = c·T_r/2"
    ]
}


class FormulaError(ValueError):
    """公式名未知 / 参数缺失或多余。"""


def fmt_latex_num(x: float, sig: int = 4) -> str:
    """把浮点数格式化为便于阅读的 LaTeX 数字。

    规则：先按 sig 位有效数字四舍五入（与判分容差同量级，多写无意义）；
    结果落在 [1e-3, 1e5) 就写普通小数，否则写 m\\times10^{e} 的科学计数。

    例：16.98970004336019 -> "16.99"；75000.0 -> "75000"；3e8 -> "3\\times10^{8}"
    """
    if x == 0:
        return "0"
    import math

    e = math.floor(math.log10(abs(x)))
    q = round(x, -(e - (sig - 1)))
    if q == 0:
        q = x
    if 1e-3 <= abs(q) < 1e5:
        return f"{q:g}"
    e2 = int(math.floor(math.log10(abs(q))))
    m = q / (10.0**e2)
    return rf"{m:g}\times10^{{{e2}}}"


def evaluate(name: str, inputs: dict[str, float]) -> dict:
    """按公式名与输入参数计算真值。

    返回 dict：value / unit / latex / substitution_latex / used / defaults_used
    未知公式名、缺参数、多余参数一律抛 FormulaError（critical 级）。
    """
    if name not in REGISTRY:
        raise FormulaError(
            f"未知公式 {name!r}；可用公式见 formulas.REGISTRY（共 {len(REGISTRY)} 条）"
        )
    f = REGISTRY[name]

    unknown = set(inputs) - set(f.params)
    if unknown:
        raise FormulaError(f"公式 {name} 收到多余参数 {sorted(unknown)}；应为 {sorted(f.params)}")

    missing = set(f.params) - set(inputs) - set(f.defaults)
    if missing:
        raise FormulaError(f"公式 {name} 缺少参数 {sorted(missing)}")

    kwargs: dict[str, float] = {}
    defaults_used: dict[str, float] = {}
    for p in f.params:
        if p in inputs:
            kwargs[p] = float(inputs[p])
        else:
            kwargs[p] = float(f.defaults[p])
            defaults_used[p] = float(f.defaults[p])

    value = float(f.fn(**kwargs))

    subst = f.subst
    for p, val in kwargs.items():
        subst = subst.replace(f"<<{p}>>", fmt_latex_num(val))
    substitution_latex = rf"{subst} = {fmt_latex_num(value)}\ \text{{{f.output_unit}}}"

    return {
        "value": value,
        "unit": f.output_unit,
        "latex": f.latex,
        "substitution_latex": substitution_latex,
        "used": kwargs,
        "defaults_used": defaults_used,
        "desc": f.desc,
        "subdomain": f.subdomain,
    }


def formula_names() -> list[str]:
    return sorted(REGISTRY)
