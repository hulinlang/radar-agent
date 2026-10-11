#!/usr/bin/env bash
# B10 acceptance: paged KV correctness + memory saving + cross-request prefix reuse
#               + mixed-length continuous batching.
#
#   bash /mnt/f/Qwen3-2B - dsh/radar-agent/cpp/tests/test_b10.sh
#
# GPU-heavy: seeds load Q4_K_M and generate tokens. Do not run concurrently with
# other GPU work (docs/30 §6.3: concurrent regression + eval wedged the GPU).
set -uo pipefail

CPP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# ⚠️ 构建目录按「源码绝对路径的哈希」自动分流。
# 为什么不能只写 $HOME/radar-infer-build：本机存在两份完全相同的项目副本
# （F:\Qwen3-2B 与 F:\Qwen3-2B - dsh），旧 build 目录缓存里的
# CMAKE_HOME_DIRECTORY 指向其中一份，另一份用它会直接
# 「does not match the source ... used to generate cache」而 cmake 失败。
SRC_ID=$(printf '%s' "$CPP_DIR" | cksum | cut -d' ' -f1)
BUILD="${RADAR_INFER_BUILD:-$HOME/radar-infer-build-$SRC_ID}"
MODEL="${RADAR_B10_MODEL:-$HOME/models/qwen3vl-sft-Q4_K_M.gguf}"
TMP=/tmp/radar_b10
pass=0; fail=0
ok()  { echo "PASS: $1"; pass=$((pass+1)); }
bad() { echo "FAIL: $1"; fail=$((fail+1)); }

mkdir -p "$TMP"

echo "=== build ==="
cmake -B "$BUILD" -S "$CPP_DIR" > "$TMP/cmake.log" 2>&1 \
  && cmake --build "$BUILD" -j"$(nproc)" >> "$TMP/cmake.log" 2>&1 \
  || { echo "BUILD FAILED"; tail -20 "$TMP/cmake.log"; exit 1; }
grep -i "warning" "$TMP/cmake.log" > /dev/null && bad "compiler warnings" || ok "build, zero warnings"
BIN="$BUILD/radar-infer"
[ -x "$BIN" ] || { echo "FAIL: $BIN missing"; exit 1; }

# ---------------------------------------------------------------- fixtures
# Generated here (not committed) so the test is self-contained.
python3 - "$TMP" <<'PY'
import sys, pathlib
d = pathlib.Path(sys.argv[1]); d.mkdir(parents=True, exist_ok=True)
PREFIX = ("你是一台机载雷达信号处理领域的专业助手。回答时只依据下面给出的教材与论文内容，不要编造。"
          "所有公式请用 LaTeX 表示。以下为必须遵守的约定：距离分辨率等于光速除以二倍信号带宽；"
          "多普勒频率等于二倍径向速度除以波长；雷达方程中接收功率与目标作用距离的四次方成反比；"
          "脉冲压缩比等于脉冲宽度与信号带宽之积；匹配滤波在加性白噪声下取得最大输出信噪比；"
          "模糊函数关于原点对称。请严格遵守上述约定。")
qs = ["雷达的工作原理是什么？", "脉冲压缩的作用是什么？", "距离分辨率由什么决定？",
      "多普勒频率与速度是什么关系？", "为什么雷达方程里是距离的四次方？",
      "匹配滤波器为什么能提高信噪比？", "模糊函数对称性有什么实际用途？",
      "体杂波强度随哪些雷达参数变化？"]
for i, q in enumerate(qs, 1):
    (d / f"p{i}.txt").write_text(PREFIX + "\n\n问题：" + q + "\n", encoding="utf-8")
BASE = ("雷达通过发射电磁波并接收目标回波来探测目标。距离分辨率等于光速除以二倍带宽。"
        "多普勒频率等于二倍径向速度除以波长。雷达方程中接收功率与作用距离的四次方成反比。"
        "脉冲压缩比等于脉冲宽度与带宽之积。匹配滤波在加性白噪声下取得最大输出信噪比。"
        "模糊函数关于原点对称，因此只需分析一半即可得到完整图形。")
for i, n in enumerate([40, 300, 1200, 3000], 1):
    (d / f"L{i}.txt").write_text((BASE * (n // len(BASE) + 1))[:n] + "\n请据此回答：脉冲压缩的作用是什么？\n",
                                 encoding="utf-8")
# A deliberately short prompt for the memory check: the point there is "a SHORT
# sequence must not reserve kv_cap worth of KV", so the prompt must be short.
(d / "short.txt").write_text("雷达的工作原理是什么？\n", encoding="utf-8")
PY
ok "fixtures generated"

COMMON=(--model "$MODEL" --n-predict 24 --kv-cap 4096 --backend cuda)

# ------------------------------------------- 1) paged-KV memory (DoD#3)
echo "--- 1) memory: paged allocation vs pre-B10 worst case ---"
# ⚠️ 用**短** prompt：要量的正是「短序列不再预留 kv_cap」。
#    若用长 prompt，小池子会真的耗尽（那是正确行为，不是 bug）。
"$BIN" "${COMMON[@]}" --prompt-file "$TMP/short.txt" --num-blocks 0 --ids-out "$TMP/mem_auto.ids" \
    > /dev/null 2> "$TMP/mem_auto.err"
"$BIN" "${COMMON[@]}" --prompt-file "$TMP/short.txt" --num-blocks 8 --ids-out "$TMP/mem_nb8.ids" \
    > /dev/null 2> "$TMP/mem_nb8.err"
# ⚠️ 不能用 grep -oE '[0-9.]+'：它会把标签里的 "B10" 也抓出来（实测踩到）。
alloc=$(sed -nE 's/.*allocated=([0-9.]+) MiB.*/\1/p' "$TMP/mem_nb8.err" | head -1)
worst=$(sed -nE 's/.*pre-B10-equivalent=([0-9.]+) MiB.*/\1/p' "$TMP/mem_nb8.err" | head -1)
meas=$(RADAR_MEM_REPORT=1 "$BIN" "${COMMON[@]}" --prompt-file "$TMP/short.txt" --num-blocks 8 \
       2>&1 >/dev/null | sed -nE 's/.*measured cudaMemGetInfo delta=([0-9.]+).*/\1/p' | head -1)
if [ -n "$alloc" ] && [ -n "$worst" ] && awk "BEGIN{exit !($worst >= $alloc*8)}"; then
  ok "pool shrinks with --num-blocks (${alloc} MiB vs worst ${worst} MiB, >=8x)"
else bad "pool did not shrink (alloc=${alloc:-?} worst=${worst:-?})"; fi
if [ -n "$meas" ] && awk "BEGIN{d=$alloc-$meas; if(d<0)d=-d; exit !(d<=0.5)}"; then
  ok "computed pool size == cudaMemGetInfo delta (${alloc} vs ${meas} MiB)"
else bad "computed vs measured mismatch (${alloc:-?} vs ${meas:-?})"; fi
diff -q "$TMP/mem_auto.ids" "$TMP/mem_nb8.ids" > /dev/null \
  && ok "pool size does not change output ids" || bad "pool size changed output ids"

# ------------------------------- 2) cross-request prefix reuse (DoD#4)
echo "--- 2) cross-request prefix cache ---"
P8=()
for i in 1 2 3 4 5 6 7 8; do P8+=(--prompt-file "$TMP/p$i.txt"); done
rm -f "$TMP/on_"*.ids "$TMP/off_"*.ids
"$BIN" "${COMMON[@]}" --max-batch 8 "${P8[@]}" --ids-out-prefix "$TMP/on_"  > /dev/null 2> "$TMP/on.err"
RADAR_NO_PREFIX_CACHE=1 "$BIN" "${COMMON[@]}" --max-batch 8 "${P8[@]}" \
    --ids-out-prefix "$TMP/off_" > /dev/null 2> "$TMP/off.err"
reused=$(grep -oE 'reused_prompt_tokens=[0-9]+' "$TMP/on.err" | head -1 | cut -d= -f2)
reused_off=$(grep -oE 'reused_prompt_tokens=[0-9]+' "$TMP/off.err" | head -1 | cut -d= -f2)
[ "${reused:-0}" -gt 0 ] && ok "prefix cache reused ${reused} prompt tokens" \
                         || bad "prefix cache reused nothing"
[ "${reused_off:-0}" -eq 0 ] && ok "cache-off control reports 0 reuse" \
                             || bad "RADAR_NO_PREFIX_CACHE did not disable reuse"
same=0
for i in 0 1 2 3 4 5 6 7; do
  diff -q "$TMP/on_$i.ids" "$TMP/off_$i.ids" > /dev/null 2>&1 && same=$((same+1))
done
[ "$same" -eq 8 ] && ok "prefix reuse is numerically inert (8/8 ids identical)" \
                  || bad "prefix reuse changed output ($same/8 identical)"

# ------------------------- 3) mixed-length continuous batching (DoD#5)
echo "--- 3) mixed-length continuous batching ---"
rm -f "$TMP/batch_"*.ids
"$BIN" "${COMMON[@]}" --max-batch 4 --prompt-file "$TMP/L1.txt" --prompt-file "$TMP/L2.txt" \
    --prompt-file "$TMP/L3.txt" --prompt-file "$TMP/L4.txt" \
    --ids-out-prefix "$TMP/batch_" > /dev/null 2> "$TMP/mixed.err"
rc=$?
[ $rc -eq 0 ] && ok "4 mixed-length requests (40..2202 tok) batched without error" \
              || { bad "mixed-length batch rc=$rc"; tail -5 "$TMP/mixed.err"; }
same=0
for i in 1 2 3 4; do
  j=$((i-1))
  "$BIN" "${COMMON[@]}" --prompt-file "$TMP/L$i.txt" --ids-out "$TMP/single_$i.ids" \
      > /dev/null 2>&1
  diff -q "$TMP/batch_$j.ids" "$TMP/single_$i.ids" > /dev/null 2>&1 && same=$((same+1))
done
[ "$same" -eq 4 ] && ok "batch == single-run ids for all 4 lengths" \
                  || bad "batch vs single-run mismatch ($same/4)"

echo "----------------------------------------"
echo "B10 RESULT: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
