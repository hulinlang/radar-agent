"""雷达术语口径表（Term Registry）——解决"同一名词在不同体制下含义不同"。

问题（用户 2026-09-15 提出）：
    雷达体制不同，同一个术语的口径就不同。"距离分辨率"在脉冲雷达里 B 是信号带宽，
    在 FMCW 里 B 是**扫频带宽**，而在 SAR 方位向则**与带宽无关**（ΔR_az = D/2）。
    如果数据集不区分体制，题目就没有唯一答案，也无法判断模型是否"体制串味"。

本表的作用（三件事）：
    ① 出题时查表，保证答案口径正确；
    ② 评测时定位"体制串味"错误（`confusable` 字段记录了最常见的错法）；
    ③ 训练/评测分层统计时提供 term × regime 的覆盖依据。

⚠️ **数据等级声明**：条目内容是雷达领域的通行口径，**逐条**标注是否已与语料交叉核对。
   未核对的条目按 docs/05 §9.0 的纪律**不得进入评测集**（schema 会报 `E_TERM_UNVERIFIED`）。

⚠️ **核对粒度是「口径」而不是「术语」**（2026-09-15 改进）：
   原先 `verified` 挂在**术语**上，存在一个静默漏洞 ——
   只要核对过任意一条口径，整个术语就被放行，"同术语换个体制"即可绕过核对。
   现在拆成三层：
     · `Term.universal_verified` —— 普适口径是否核对
     · `RegimeNote.verified`     —— 该**体制**口径是否核对
     · `Term.verified`（派生）    —— 上面**全部**都为真时才为真（保守口径）
   实际判定"本条样本用到的那一层有没有核对"，由 `schema.py` 按 regime 精确判断。

**核对依据的命名约定**（写进 `source` 字段，保证可回溯）：
   · `机载雷达系统与信息处理 <节号> (PDF pN)` —— PDF 页序（1-based）
   · `mmWave_Insight <文件路径>`               —— GitHub matreshka15/mmWave_Insight
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RegimeNote:
    regime: str
    note: str
    formula: str = ""
    verified: bool = False  # 该体制口径是否已与语料交叉核对
    source: str = ""  # 核对依据（见模块头部的命名约定）


@dataclass(frozen=True)
class Term:
    key: str
    zh: str
    en: str
    universal: str
    by_regime: dict[str, RegimeNote] = field(default_factory=dict)
    confusable: list[str] = field(default_factory=list)
    universal_verified: bool = False  # 普适口径是否已核对
    source: str = ""  # 普适口径的核对依据

    @property
    def verified(self) -> bool:
        """保守口径：**普适口径 + 全部已记录体制口径**都核对过才为真。

        精确到"本样本实际用到的那一层"的判断在 `schema.py`（按 regime 取）。
        """
        return self.universal_verified and all(n.verified for n in self.by_regime.values())

    def note(self, regime: str) -> RegimeNote | None:
        return self.by_regime.get(regime)


TERMS: dict[str, Term] = {
    "range_resolution": Term(
        key="range_resolution",
        zh="距离分辨率",
        en="range resolution",
        universal="ΔR = c / (2B)，B 为信号有效带宽",
        universal_verified=True,
        source="机载雷达系统与信息处理 2.3.5 (PDF p82)",
        by_regime={
            "fmcw": RegimeNote(
                "fmcw",
                "形式同为 c/(2B)，但 B 指**扫频带宽**（一个 chirp 的频率变化范围），"
                "不是脉冲宽度的倒数",
                r"\Delta R = \frac{c}{2B},\ B\ \text{为扫频带宽}",
                verified=True,
                source="mmWave_Insight docs/mmwave/fmcw.md",
            ),
            "automotive_mmwave": RegimeNote(
                "automotive_mmwave",
                "同 FMCW：B 为扫频带宽，车载典型值约 1–4 GHz",
                verified=True,
                source="mmWave_Insight docs/mmwave/fmcw.md（4 GHz → 3.75 cm）",
            ),
            "sar": RegimeNote(
                "sar",
                "**分两个方向**：距离向同脉冲压缩 c/(2B)；"
                "**方位向 ΔR_az = D/2**（天线方位尺寸的一半），与带宽无关",
                r"\Delta R_{az} = \frac{D}{2}",
                verified=True,
                source="机载雷达系统与信息处理 8.3.2 (PDF p312)",
            ),
            "airborne_pulse": RegimeNote(
                "airborne_pulse",
                "**分辨率公式本身与地基相同**（c/2B），机载的差别不在公式而在**前提**："
                "平台运动使回波在脉冲间跨距离门移动（**距离走动**），"
                "回波包络可能出现数倍于发射脉宽的变化，必须先做距离校正，"
                "否则「提高分辨率」的前提（同一散射体落在同一距离单元）不成立",
                verified=True,
                source="机载雷达系统与信息处理 8.4.2 (PDF p325)",
            ),
        },
        confusable=[
            "把 SAR 的**方位向**分辨率也写成 c/(2B)（方位向与带宽无关）",
            "把 FMCW 的 B 当成脉冲雷达的脉宽倒数",
            "把机载的 c/2B 当成「和地基完全没差别」（公式同、前提不同）",
        ],
    ),
    "angular_resolution": Term(
        key="angular_resolution",
        zh="角分辨率（波束宽度）",
        en="angular resolution / beamwidth",
        universal="均匀加权线阵：θ_3dB ≈ 0.88 λ / D，D 为孔径尺寸",
        by_regime={
            "phased_array": RegimeNote(
                "phased_array",
                "由阵面尺寸决定；加权可压低旁瓣，**代价是主瓣展宽**（分辨率变差）",
            ),
            "mimo": RegimeNote(
                "mimo",
                "正交波形形成**虚拟阵列**，虚拟阵元数 = N_tx × N_rx，"
                "等效孔径随之增大 —— 这是 MIMO 提升角分辨率的机理",
            ),
            "automotive_mmwave": RegimeNote(
                "automotive_mmwave",
                "实用上受虚拟孔径与阵列标定限制；常用角度 FFT，超分辨需 MUSIC/Capon",
            ),
        },
        confusable=[
            "把 MIMO 的虚拟阵列增益说成「真实孔径变大了」（天线物理尺寸并未改变）",
            "声称「加窗能提高角分辨率」（加窗压旁瓣，通常使主瓣变宽）",
        ],
    ),
    "angle_measurement": Term(
        key="angle_measurement",
        zh="测角方法",
        en="angle estimation",
        universal="相位比较 → 波束形成 → 超分辨（MUSIC / Capon）",
        by_regime={
            "phased_array": RegimeNote(
                "phased_array", "波束扫描；常用**单脉冲（和差波束）**获得高精度角度"
            ),
            "automotive_mmwave": RegimeNote(
                "automotive_mmwave",
                "MIMO 虚拟阵列 + **角度 FFT**（工程主流）；高精度场景用 MUSIC / Capon",
            ),
            "mimo": RegimeNote(
                "mimo", "波形正交分离各收发通道 → 组成虚拟阵列 → 空间谱估计"
            ),
            "sar": RegimeNote(
                "sar",
                "方位向靠**合成孔径相干积累成像**，不是逐脉冲测角（概念上不同层级）",
            ),
        },
        confusable=[
            "用「单脉冲和差波束」描述车载毫米波雷达的测角（车载主流是虚拟阵列 + 角度 FFT）",
            "把 SAR 的方位成像当成逐脉冲测角",
        ],
    ),
    "echo_model": Term(
        key="echo_model",
        zh="回波建模",
        en="echo modeling",
        universal="功率按雷达方程，相干性按相位历史（多普勒）描述",
        by_regime={
            "ground_pulse": RegimeNote(
                "ground_pulse",
                "雷达方程 + 面杂波（后向散射系数 σ0 随擦地角变化）+ 地形遮蔽",
            ),
            "airborne_pulse": RegimeNote(
                "airborne_pulse",
                "在地基基础上叠加**平台运动**：距离走动（range walk）、多普勒走动/展宽；"
                "地面杂波因平台速度产生**空时耦合**（此为 STAP 的由来）",
                verified=True,
                source="机载雷达系统与信息处理 7.6.1 (PDF p282–283)",
            ),
            "spaceborne": RegimeNote(
                "spaceborne",
                "机载基础上再叠加地球自转、超大斜距与星下点几何",
            ),
            "fmcw": RegimeNote(
                "fmcw",
                "用**差频（beat）模型**：回波与发射信号混频得拍频，"
                "f_b 同时含距离项与多普勒项，需 up/down chirp 解耦",
                r"f_b = \frac{2RB}{cT_{chirp}} \pm f_d",
                verified=True,
                source="mmWave_Insight docs/mmwave/fmcw.md",
            ),
            "mimo": RegimeNote(
                "mimo",
                "在各体制基础上增加**正交波形**（TDMA / FDMA / DDMA / CDM）分离通道，"
                "再组成虚拟阵列",
            ),
        },
        confusable=[
            "用 FMCW 的差频模型描述脉冲雷达回波",
            "描述机载回波时只写雷达方程，漏掉平台运动引起的距离/多普勒走动",
        ],
    ),
    "doppler_frequency": Term(
        key="doppler_frequency",
        zh="多普勒频移",
        en="Doppler frequency shift",
        universal="f_d = 2v/λ，v 为径向速度",
        universal_verified=True,
        source="机载雷达系统与信息处理 4.2.2 (PDF p135)",
        by_regime={
            "fmcw": RegimeNote(
                "fmcw",
                "拍频里**同时含距离与多普勒**，不能只看单边 chirp 直接读距离",
                verified=True,
                source="mmWave_Insight docs/mmwave/fmcw.md（f_beat = f_R + f_d）",
            ),
            "sar": RegimeNote(
                "sar",
                "方位向由多普勒历史（多普勒中心 + 调频率）决定，是成像的核心参量",
                verified=True,
                source="机载雷达系统与信息处理 8.3.2 (PDF p312)",
            ),
        },
        confusable=[
            "声称 FMCW 的拍频只反映距离（忽略距离-多普勒耦合）",
        ],
    ),
    "unambiguous_range": Term(
        key="unambiguous_range",
        zh="最大不模糊距离",
        en="maximum unambiguous range",
        universal="R_u = c / (2·PRF)",
        universal_verified=True,
        source="机载雷达系统与信息处理 4.3.1 (PDF p142)",
        by_regime={
            "fmcw": RegimeNote(
                "fmcw",
                "**连续波体制不存在「不模糊距离」这个概念** —— "
                "它是脉冲体制的产物；对应约束变成最大可测拍频（受采样率限制）",
                r"R_{max} = \frac{c\,f_s T_c}{4B}",
                verified=True,
                source="mmWave_Insight docs/mmwave/fmcw.md（最大距离由采样率与 Chirp 时间决定）",
            ),
            "automotive_mmwave": RegimeNote(
                "automotive_mmwave",
                "距离不模糊由 chirp 参数与采样率决定；"
                "**速度**不模糊由 chirp 重复周期（慢时间采样）决定",
                r"v_{max} = \frac{\lambda}{4T_c}",
                verified=True,
                source="mmWave_Insight docs/mmwave/fmcw.md",
            ),
            "airborne_pulse": RegimeNote(
                "airborne_pulse",
                "**R_u 本身同普适式**，但机载的特殊性在于它**不能单独取大**："
                "距离不模糊要求 PRF 低，而多普勒不模糊要求 PRF 高，"
                "下视还要压制主瓣杂波 —— 三者互相冲突，"
                "于是机载雷达按低 / 中 / 高 **三种 PRF 模式**分工取舍："
                "低 PRF 无距离模糊但有多普勒模糊，高 PRF 反过来，中 PRF 两者皆有",
                verified=True,
                source="机载雷达系统与信息处理 4.4.2–4.4.4 (PDF p151–152)",
            ),
        },
        confusable=[
            "给 FMCW 套用 R_u = c/(2·PRF)（该式对连续波体制不成立）",
            "以为机载雷达可以把 R_u 取到任意大（PRF 一降，多普勒模糊与杂波抑制就恶化）",
        ],
    ),
}


def term_keys() -> list[str]:
    return sorted(TERMS)


def by_regime(regime: str) -> dict[str, Term]:
    """列出在该体制下有专门口径的术语。"""
    return {k: t for k, t in TERMS.items() if regime in t.by_regime}


def unverified() -> list[str]:
    """整体尚未核对齐的术语（普适口径或任一体制口径未核对）—— 保守口径。"""
    return sorted(k for k, t in TERMS.items() if not t.verified)


def is_verified_for(term_key: str, regime: str | None) -> bool:
    """**本条样本实际用到的那一层**是否已核对 —— 评测集准入的真正判据。

    universal / unspecified 只要求普适口径已核对；
    具体体制要求该体制的口径存在**且**已核对。
    这样既不会因"某术语别的体制没核对"误伤，也不会有"换个体制就绕过核对"的漏洞。
    """
    t = TERMS[term_key]
    if regime in (None, "universal", "unspecified"):
        return t.universal_verified
    note = t.by_regime.get(regime)
    return bool(note and note.verified)


def render(term_key: str) -> str:
    t = TERMS[term_key]
    mark = lambda ok: "OK " if ok else "未核"  # noqa: E731
    out = [f"【{t.zh}】{t.en}   term_key={t.key}"]
    out.append(f"  普适口径[{mark(t.universal_verified)}]：{t.universal}")
    if t.source:
        out.append(f"      出处：{t.source}")
    for rg, note in t.by_regime.items():
        out.append(f"  · {rg}[{mark(note.verified)}]：{note.note}")
        if note.formula:
            out.append(f"      公式：{note.formula}")
        if note.source:
            out.append(f"      出处：{note.source}")
    if t.confusable:
        out.append("  常见错法（可用于出「体制陷阱题」）：")
        for c in t.confusable:
            out.append(f"    - {c}")
    return "\n".join(out)
