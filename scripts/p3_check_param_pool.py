"""参数池自检：
   ① 每个注册表公式是否都有池；
   ② train / eval 参数组是否重叠（重叠 = 评测泄漏）；
   ③ 每组参数能否被 formulas.evaluate 接受（参数名/数量对不对）。
"""
import sys

import yaml

sys.path.insert(0, r"F:\Qwen3-2B\radar-agent")
from src.dataset import formulas

pool = yaml.safe_load(
    open(r"F:\Qwen3-2B\radar-agent\configs\param_pool.yaml", encoding="utf-8")
)["pools"]

print("参数池公式数: %d　注册表公式数: %d" % (len(pool), len(formulas.REGISTRY)))
missing = [n for n in formulas.REGISTRY if n not in pool]
extra = [n for n in pool if n not in formulas.REGISTRY]
print("注册表里有、池里缺的: %s" % missing)
print("池里有、注册表里没有的: %s" % extra)
print()


def key(d):
    return tuple(sorted((k, float(v)) for k, v in d.items()))


ok = True
for name in sorted(pool):
    if name not in formulas.REGISTRY:
        continue
    halves = pool[name]
    tr, ev = halves.get("train", []), halves.get("eval", [])
    trk, evk = {key(d) for d in tr}, {key(d) for d in ev}
    overlap = trk & evk
    if overlap:
        ok = False
    # 可计算性
    bad = []
    for half, items in (("train", tr), ("eval", ev)):
        for d in items:
            try:
                formulas.evaluate(name, d)
            except Exception as e:
                bad.append("%s:%s" % (half, e))
                ok = False
    flag = "OK" if not overlap and not bad else "FAIL"
    print("  %-28s train %d / eval %d  %s%s%s" % (
        name, len(tr), len(ev), flag,
        ("  重叠 %d 组!" % len(overlap)) if overlap else "",
        ("  计算失败 %s" % bad[:2]) if bad else ""))

print()
print("参数池自检通过（无重叠、全部可算）: %s" % ok)
