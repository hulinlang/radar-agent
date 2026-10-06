# 阶段 B · B2 详细方案：byte-level BPE 分词器（手写 + transformers 对拍）

> 前置：`docs/20` §4（分词器精读）、§5.2b（手写 vs 接库边界：分词器手写是**学习验证**，
> 生产等价物 = `tokenizers-cpp`，面试话术已锁定）。
> 本文是 B2 的开工前方案，确认后才写码。B2 只做分词/解码，不碰模型。

## 1. 目标与验收（DoD）

**交付物**：`src/tokenizer.{h,cpp}`——从 `GGUFFile` 直接构建的分词器（词表/合并规则/特殊 token 全部从 GGUF 读，零硬编码 Qwen 常量），外加测试可执行 `tokenizer-test`。

**验收清单**（`tests/test_b2.sh` 一键，全绿才算完成）：

| # | 验收项 | 判据 |
|---|---|---|
| 1 | 构建零警告 | `-Wall -Wextra` 无声 |
| 2 | **编码对拍**：100 句覆盖 12 类边界 case | 与 transformers `AutoTokenizer` 产出 id 序列**逐一全等**（100/100） |
| 3 | **解码回环**：decode(encode(x)) == x | 100/100 |
| 4 | 特殊 token 定位 | 含 `<|im_start|>`/`<|im_end|>`/`<|tool_call|>` 的用例，id 出现在正确位置、不被 BPE 拆碎 |
| 5 | 数字规则抽查 | `"1234567"` 等用例的分段符合 `\p{N}{1,3}`（随对拍隐含验证，打印分段备查） |

**对拍基准的选择**：golden 用 **transformers**（Windows `qwen3vl` env，`dir/` 基座的 `AutoTokenizer`）预生成、**提交进仓库**（`cpp/tests/tokenizer_golden.tsv`，100 行），WSL 测试零依赖。
⚠️ 已知风险：llama.cpp 内置 BPE 与 HF 在极端 case 上可能有差异——B4 若发现逐 token 对不上，第一排查点就是分词，届时再加 llama.cpp 侧对拍。本里程碑以 transformers 为准（阶段 A 的 Python 链路用它）。

## 2. 输入：GGUF 里的分词器三件套（B1 已能读出）

| GGUF 键 | 实测值 | 用途 |
|---|---|---|
| `tokenizer.ggml.tokens` | 151936 条字符串 | id ↔ token 字符串（GPT-2 字节映射后的 unicode 形态） |
| `tokenizer.ggml.merges` | 151387 条 `"A B"` | 合并优先级（行号 = rank，越小越优先） |
| `tokenizer.ggml.token_type` | 151936 个 i32 | 1=normal，3=control / 4=user-defined → **特殊 token 判定依据** |
| `tokenizer.ggml.{bos,eos,padding}_token_id` | 151643 / 151645 / 151643 | 引擎起止判定 |
| `tokenizer.ggml.pre` | `"qwen2"` | 预分词规则标识（决定用哪套切分器；未知值报错而非乱猜） |

## 3. 算法三段式

### 3.1 字节 ↔ unicode 映射（GPT-2 式）

可打印字节（`!`–`~`、`¡`–`¬`、`®`–`ÿ`）映射到自身，其余 68 个字节映射到 `U+0100+n`。
编码时：输入 UTF-8 字节流 → 每字节查表得到 mapped 字符串；解码时用逆表。
映射表程序生成（20 行循环），不手抄。

### 3.2 预分词器（本里程碑的技术主体）

qwen2 正则（transformers `Qwen2Tokenizer` 同款），按**交替优先级**逐条尝试：

```
1. (?i:'s|'t|'re|'ve|'m|'ll|'d)     # 英文缩写尾巴，大小写不敏感
2. [^\r\n\p{L}\p{N}]?\p{L}+         # 字母段，可带 1 个非字母非数字前缀字符（如前导空格/制表符）
3. \p{N}{1,3}                       # 数字段，每段 1~3 位
4. \ ?[^\s\p{L}\p{N}]+[\r\n]*       # 标点/符号段，可带前导空格，可吃尾随换行
5. \s*[\r\n]+                       # 含换行的空白段（截至空白串中最后一个 \r/\n）
6. \s+(?!\S)                        # 结尾空白（后面没有非空白字符）
7. \s+                              # 其余空白
```

**C++ 没有支持 `\p{L}`/`\p{N}` 的标准正则**（`std::regex` 的 ECMAScript 模式不含 Unicode 属性），
所以预分词器**手写**：UTF-8 解码出码点序列 → 码点分类 → 按上面 7 条规则的语义逐位置贪心切分。

- **码点分类表**：`tools/gen_unicode_tables.py` 用 Python `unicodedata` 离线生成
  `src/unicode_tables.h`（LETTER = L* 五类、NUMBER = N* 三类、SPACE = White_Space 属性 25 个码点，
  排序区间数组 + 二分查找）。表提交进仓库，脚本只用于再生成。**不引入 ICU/regex 库**。
- 切分语义对齐 regex「最左 + 交替按序 + 分支内贪心」：每个位置从规则 1 开始试，取第一条能匹配的，
  分支内按贪心跳转。规则 5 的 `\s*[\r\n]+` 实现为：取当前位置最大空白串，若含 `\r`/`\n`，
  匹配到串中**最后一个换行符为止**（等价于 regex 回溯结果）。

### 3.3 特殊 token 优先切出

编码入口先扫特殊 token（token_type 3/4 的词表项，按长度降序逐一字面匹配，~20 个，线性扫即可）：
命中则直接出 id、**不经过 BPE**；特殊 token 之间的普通文本段走 3.2 + 3.4。
解码时特殊 token id 原样输出字面字符串。

### 3.4 BPE 合并

每个预分词片段：字节映射 → 先整串查 vocab（高频命中，O(1)）→ 未命中则进入合并循环：
初始符号序列 = mapped 字符串的逐字符；每轮找出**相邻对中 rank 最小**者，从左到右合并该对的所有出现；
直到无可合并对。片段长度通常 <50，O(n²) 重扫实现最简单且语义与 HF Python 参考实现严格一致——
**正确性优先，性能不是本环节目标**（分词占端到端延迟微秒级）。

合并所需两张哈希表构建于构造期：`vocab: string→id`、`merge_rank: (left,right)→rank`。

## 4. 接口与文件

```cpp
class Tokenizer {
public:
    static Tokenizer from_gguf(const GGUFFile& f);   // 读三件套 + 特殊 id；pre!="qwen2" 报错
    std::vector<int32_t> encode(const std::string& utf8) const;
    std::string decode(const std::vector<int32_t>& ids) const;
    int32_t bos() const;  int32_t eos() const;  int32_t pad() const;
    bool is_special(int32_t id) const;
    // 调试用：返回预分词分段（验收 #5 备查）
    std::vector<std::string> pre_tokenize(const std::string& utf8) const;
};
```

```
cpp/
├── src/tokenizer.{h,cpp}        # ~350 行，主体
├── src/unicode_tables.h         # 生成产物（提交）
├── src/tokenizer_test.cpp       # 测试 main：读 golden TSV 逐行对拍
├── tools/gen_unicode_tables.py  # 表生成器（Windows python 跑一次）
└── tests/
    ├── gen_tokenizer_golden.py  # golden 生成器（transformers，Windows 跑一次）
    ├── tokenizer_golden.tsv     # 100 行 golden（提交）：hex(utf8) \t id,id,...
    └── test_b2.sh               # 构建 → 对拍 → 回环 → 特殊 token 断言
```

golden 用 TSV 不用 JSON：hex 编码文本规避一切转义问题，C++ 侧 30 行解析，不引 JSON 库。

## 5. 100 句对拍用例设计（12 类，每类 5~15 句）

| 类 | 构造要点 |
|---|---|
| 中文连续段 | 纯中文句、中英混排、中文+标点 |
| 英文 | 常规句、全大写、带缩写（`it's`/`I'M`/`you're` 大小写变体） |
| 数字 | 1~9 位整数、小数、日期、版本号、千分位 |
| 代码 | C++/Python 片段（缩进、`::`、`->`、注释） |
| emoji | 单 emoji、ZWJ 序列（👨‍👩‍👧）、肤色修饰、国旗 |
| 空白 | 多空格、tab、行尾空格、连续换行、`\r\n`、全角空格 U+3000 |
| 特殊 token | 句首/句中/句尾嵌 `<|im_start|>`、`<|im_end|>`、`<|tool_call|>` |
| 标点符号 | 中英文标点混排、全角标点、数学符号 |
| URL/路径 | http 链接、文件路径、邮箱 |
| 其他文字 | 日文假名、韩文、西里尔、阿拉伯文 |
| 组合字符 | 声调组合符（é 的 NFD 形态）、零宽字符 |
| 极端 | 空串、单字符、超长无空格串（>200 字符）、雷达领域真实问答句 ×5（贴业务） |

## 6. 明确不做

- **不做 chat template**（Jinja）：引擎接收渲染好的 prompt 字符串，模板渲染留在 Python 侧（阶段 A 已有）；
- 不做增量/streaming 解码的半个 UTF-8 字符缓冲（B4 流式输出时再加，接口预留按 token 逐出）；
- 不优化性能（哈希表 + O(n²) 合并即可）；
- 不支持 `pre != "qwen2"` 的其他预分词规则（遇到直接报错）。

## 7. 风险排序

1. **规则 5/6 空白段的 regex 回溯语义**实现偏差 → 对拍集里多空白/混合换行用例兜底；
2. **Unicode 分类表**与 Rust regex（HF fast tokenizer 实际实现）的边界差异（罕见码点）→ 用例覆盖东亚/阿拉伯/组合字符；
3. 特殊 token 与普通文本相邻时的切分顺序（必须先特殊后 BPE）；
4. 字节映射方向用反（症状：英文全对、中文全错，好定位）。

## 8. 工时分解

| 步 | 内容 |
|---|---|
| 1 | `gen_unicode_tables.py` 跑出生成头文件 |
| 2 | `tokenizer.{h,cpp}`：构建（读 GGUF 三件套）→ 预分词 → BPE → 编解码 |
| 3 | `gen_tokenizer_golden.py` 生成 100 句 golden 并提交 |
| 4 | `tokenizer_test.cpp` + `test_b2.sh` 五项验收全绿 |
| 5 | 本文回填实测结果 + git commit |

---

## 9. 实测修正（2026-10-06 写码前，用 `dir/` 基座真实 tokenizer 验证）

1. **Qwen 词表不含多数字 token**：`convert_tokens_to_ids('12'/'123')` 均为 None，
   `"1234567"` 实测编码为 7 个单数字 token `[16,17,18,19,20,21,22]`。
   即预分词规则 `\p{N}{1,3}` 仍会切段（123/456/7），但段内只能拆成单数字。
   → 验收 #5 改为"验证**分段过程**正确"（`pre_tokenize` 调试接口打印），不再期待多数字 token。
2. **字节碎片兜底实例**（必进对拍集）：`encode(" 系统") = [10236, 36097, 54658]`
   = `[空格+系的字节1, 系的字节2~3, 统]`——词表无"空格+系统"组合时 BPE 退回字节级碎片。
   `encode("系统") = [72448]`（整词命中）。这类"带前缀的中文"是手写实现最易错处。
3. **词表 key 空间确认**：vocab key 是**字节映射后的 unicode 形态**（`ĠHello`→21927），
   原始中文字符串直接查不到——印证了 §3.1 字节映射必须先于 vocab 查找。
4. **整串快查（fast path）安全性论证**：词表中的大 token 必为训练时 greedy 合并产物，
   推理时同一 greedy 过程可用同 rank 序列重建它，故"整串在词表 ⟺ 合并可完整收敛"，
   快查不改变结果；若该论证存在反例，100 句对拍兜底。
5. **transformers 带 NFC normalizer，GGUF 链路不带**（对拍时发现的分歧）：`dir/tokenizer.json`
   实测 `normalizer: {"type":"NFC"}`——NFD 输入（如 e+U+0301）会被 transformers 先归一化为 NFC 再分词，
   而 llama.cpp/GGUF 语义不做归一化。**决策：自研引擎对齐 GGUF/llama.cpp 语义（部署目标），
   两个 NFD 用例移出 golden**（替换为 NFC 稳定的等价用例），此分歧记录备查：真实业务中
   输入几乎总是 NFC（键盘/IME 产出），风险可控；若未来要接 transformers 产线的输入，需评估
   是否加归一化层。

---

## 10. 验收结果（2026-10-06 实测，全绿）

`bash cpp/tests/test_b2.sh`（WSL2 gcc）输出 **3 passed, 0 failed**：

| # | 验收项 | 结果 |
|---|---|---|
| 1 | 构建零警告 | ✅（含 B1 回归：test_b1.sh 复跑 9/9 仍全绿） |
| 2 | 编码对拍 100 句（12 类边界 case） | ✅ **encode 100/100 逐 id 全等** |
| 3 | 解码回环 | ✅ decode 100/100 |
| 4 | 特殊 token 定位 | ✅ 8 条例全部正确切出（含句中嵌入、连续两个特殊 token） |

**实现规模**：`tokenizer.{h,cpp}` ~450 行 + 生成的 `unicode_tables.h`（Letter 659 区间 /
Number 137 区间 / Space 10 区间，UCD 15.0）。

**修复记录**：`bos/eos/pad` 三个 id 在 GGUF 里是 UINT32（存入 variant 的 uint64_t 分支），
首版按 int64_t 取触发 `std::get: wrong index for variant`——改为按 ValueType 分派。一次复测通过。

**对拍顺带验证的设计判断**：整串快查 fast path 无反例（§9.4 论证成立）；规则 5/6 空白段
回溯语义与 Rust regex 一致（多空白/混合换行用例全过）；ZWJ/ZWNJ 序列无归一化干扰（保留在 golden 中）。
