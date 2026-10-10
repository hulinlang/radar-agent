# B11 层①对拍载荷（committed golden data）

这些文件是 `cpp/tests/int8-ref-test` 用来做 **层① 对拍**的参考载荷：
"我们的 int8 路由 vs 一份独立的 NumPy 实现，逐元素**逐位**一致"。

存在仓库里（而不是每次生成）有两个原因：

1. WSL 里的 `python3` 没有 numpy，而生成脚本用的是 Windows 侧的解释器 ——
   让验收脚本去调一个环境相关的解释器会让 `test_b11.sh` 不可复现。
2. 与 `cpp/tests/golden_forward/*.f32` 同一惯例：**黄金数据入库**，验收只读不生成。

## 内容

每张张量 4 个文件（`<tag> = 张量名去点 + _q4k/_q6k`）：

| 文件 | 含义 |
|---|---|
| `<tag>.x.f32` | 输入激活 `[ncol][in]` fp32（小端） |
| `<tag>.q8.bin` | 量化后的激活：`qs[k/32][ncol][32]` int8，紧跟 `ds[k/32][ncol]`（8B：f32 d + i32 sum_q） |
| `<tag>.y.f32` | 参考输出 `[nrow][ncol]` fp32 |
| `<tag>.meta.txt` | 形状 + 载荷 FNV-1a + 两条误差数量级 |

`ncol = 16`（prefill chunk 宽度），`seed = 0x5EED`。
张量：`blk.0.attn_q` / `attn_k` / `attn_output` / `ffn_up`（Q4_K）、
`blk.0.attn_v` / `ffn_down`（Q6_K）。

## 怎么重新生成

```powershell
# Windows 侧（有 numpy 的解释器）
python cpp/tests/ref_q4_int8.py --gguf <qwen3vl-sft-Q4_K_M.gguf> `
    --tensor blk.0.attn_q.weight --outdir cpp/tests/int8_ref --ncol 16
```

生成后 meta 里的 `*_fnv` 会变，`test_b11.sh` 的输出会把它们打出来 ——
**一旦重新生成，两边必须同步**（这也是为什么 meta 里存 FNV：防止载荷被替换后
验收悄悄对不上）。
