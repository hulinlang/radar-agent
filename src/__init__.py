"""radar-agent 可复用模块包。

模块划分（对应 00_行为规范 §4.1）：
    config.py     配置加载 / 实验目录 / 日志 / 运行时指纹
    checks.py     机器可执行断言框架（§5.8）
    env_probe.py  硬件与系统环境探测（双信源交叉校验）
    modeling.py   Qwen3-VL 加载与推理封装
    vl_probe.py   多模态冒烟测试用的合成图像构造
"""

__all__ = ["config", "checks", "env_probe", "modeling", "vl_probe"]
__version__ = "0.1.0"
