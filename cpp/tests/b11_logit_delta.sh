#!/usr/bin/env bash
# B11 / D4: how much did the int8 activation route actually move the logits?
#
# "the generated ids are identical" is a yes/no with no resolution. This measures
# the thing that decides whether an id *could* flip: |Δ logit| between the int8
# and fp32-activation routes on the same prompt, compared against the gap between
# the top-1 and top-2 logits (the margin the perturbation has to overcome).
#
# Requires the CUDA-path RADAR_DEBUG_TOPK dump (added in B11).
#
#   bash /mnt/f/Qwen3-2B - dsh/radar-agent/cpp/tests/b11_logit_delta.sh
set -uo pipefail

CPP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC_ID=$(printf '%s' "$CPP_DIR" | cksum | cut -d' ' -f1)
BUILD="${RADAR_INFER_BUILD:-$HOME/radar-infer-build-$SRC_ID}"
Q4="${RADAR_B11_MODEL:-$HOME/models/qwen3vl-sft-Q4_K_M.gguf}"
BIN="$BUILD/radar-infer"
TMP=/tmp/radar_b11_delta
mkdir -p "$TMP"

python3 - "$TMP" <<'PY'
import pathlib, sys
d = pathlib.Path(sys.argv[1])
prompts = {
    "p1": "雷达的工作原理是什么？",
    "p2": "Explain how a pulsed Doppler radar measures velocity.",
    "p3": "<|im_start|>user\n脉冲压缩的作用是什么？<|im_end|>\n<|im_start|>assistant\n",
    "p4": "1+1=2. 3+5=8. 7+9=",
    "p5": "给定脉冲宽度 1μs，距离分辨率是多少？",
}
for k, v in prompts.items():
    (d / (k + ".txt")).write_text(v, encoding="utf-8")
PY

echo "=== D4: logit displacement, int8 vs fp32 activations (top-5 dumped per step) ==="
printf '%-6s %-7s %-11s %-11s %-11s %-9s %-9s %s\n' \
  prompt steps 'max|dtop1|' 'max|dmarg|' 'min_margA' 'min_margB' ratio flips
for p in p1 p2 p3 p4 p5; do
  RADAR_DEBUG_TOPK=5 "$BIN" --model "$Q4" --prompt-file "$TMP/$p.txt" --backend cuda \
      --n-predict 40 > /dev/null 2> "$TMP/$p.int8.log"
  RADAR_DEBUG_TOPK=5 RADAR_Q4_FP32_ACT=1 "$BIN" --model "$Q4" --prompt-file "$TMP/$p.txt" \
      --backend cuda --n-predict 40 > /dev/null 2> "$TMP/$p.fp32.log"
  grep '^step ' "$TMP/$p.int8.log" > "$TMP/$p.a" || true
  grep '^step ' "$TMP/$p.fp32.log" > "$TMP/$p.b" || true
  awk -v p="$p" '
    FNR==NR {
      split($3, a, ":"); idA[FNR]=a[1]; v1A[FNR]=a[2]+0;
      split($4, b, ":"); v2A[FNR]=b[2]+0;
      na=FNR; next
    }
    FNR<=na {
      split($3, a, ":"); idB=a[1]; v1B=a[2]+0;
      split($4, b, ":"); v2B=b[2]+0;
      d = v1A[FNR]-v1B; if (d<0) d=-d;                 # how far the top-1 score moved
      mA = v1A[FNR]-v2A[FNR]; if (mA<0) mA=0;          # margin inside route A (int8)
      mB = v1B-v2B; if (mB<0) mB=0;                    # margin inside route B (fp32)
      dm = mA-mB; if (dm<0) dm=-dm;                    # did the *margin* move?
      if (d>maxd) maxd=d;
      if (dm>maxdm) maxdm=dm;
      if (mA<minmA || minmA=="") minmA=mA;
      if (mB<minmB || minmB=="") minmB=mB;
      if (idA[FNR]!=idB) flips++;
      n++;
    }
    END {
      if (n==0) { printf "%-6s no logit lines (stale binary? set RADAR_INFER_BUILD)\n", p; exit }
      printf "%-6s %-7d %-11.4g %-11.4g %-11.4g %-9.4g %-9s %s\n", p, n, maxd, maxdm, minmA,
             minmB, (minmA>0 ? sprintf("%.3f", maxd/minmA) : "-"), flips+0,
             (flips+0==0 ? "no flip" : "FLIPPED");
    }' "$TMP/$p.a" "$TMP/$p.b"
done
echo
echo "读法："
echo "  max|dtop1|  = int8 路线把 top-1 的**绝对分数**挪动了多少（最坏一步）"
echo "  max|dmarg|  = int8 把 **top1-top2 间距**改变了多少 —— 这才是会不会翻的依据"
echo "                （扰动是相关的：候选们一起动，绝对位移大、相对间距稳）"
echo "  min_margA/B = 两条路线各自最紧的那一步的间距（越小越接近翻）"
echo "  ratio       = max|dtop1| / min_margA，>1 表示绝对位移超过了间距（近似指标，"
echo "                不等于翻转；翻转要看 flips 列）"
