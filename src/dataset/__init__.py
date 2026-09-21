"""P3 数据集层：schema 校验 + 真值计算 + 编译成模型所需的数据块结构。

对外主要入口：
    schema.load_spec / validate_authoring / validate_compiled / validate_batch
    formulas.evaluate / REGISTRY
    compile.compile_file / write_jsonl

命令行入口见 scripts/p3_dataset_build.py。
"""

from . import compile as compile_mod
from . import formulas, schema

__all__ = ["schema", "formulas", "compile_mod"]
