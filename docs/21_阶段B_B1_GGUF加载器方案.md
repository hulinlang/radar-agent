# 阶段 B · B1 详细方案：GGUF 加载器（mmap + 解析 + 逐字节对拍）

> 前置：`docs/20_阶段B_B0_GGUF与Qwen3架构精读.md`（格式定义与参数实测）、8 项方案决策（用户已拍板）。
> 本文是 B1 的**开工前方案**，确认后才写码。B1 只做"把文件读进来、读对"，不做任何计算。

## 1. 目标与验收（DoD）

**交付物**：`radar-infer` 内置的 GGUF 解析模块 + 一个 `gguf-dump` 可执行文件，能把 GGUF 文件的 header / 全部 metadata KV / 全部 tensor info / 文件字节对账打印为规范化文本。

**验收清单**（全部实测，全绿才算 B1 完成）：

| # | 验收项 | 判据 |
|---|---|---|
| 1 | 规范化 dump 与 Python `gguf` 包读出**逐字节一致** | 两个 GGUF（f16 + Q4_K_M）各 diff 一次，0 差异 |
| 2 | 文件字节对账 | header + KV + infos + padding + Σtensor_bytes == 文件实际大小，**零字节无法解释** |
| 3 | 张量指针抽查 | `output_norm.weight`（F32）与 `token_embd.weight`（F16）前若干值与 numpy 直读一致 |
| 4 | 负例健壮性 | 截断文件 / 坏 magic / 坏 offset → 干净的报错退出（非崩溃、非 UB） |
| 5 | 构建 | WSL2 gcc 编译零警告（`-Wall -Wextra`） |

## 2. 文件与代码结构

```
cpp/
├── src/
│   ├── ggml_types.h     # ggml type 枚举 + block_size/type_size 表（纯查表，~40 行）
│   ├── gguf.h           # GGUFFile / TensorInfo / GGUFValue 定义与接口
│   ├── gguf.cpp         # mmap + 解析实现（~250 行）
│   └── gguf_dump.cpp    # dump 工具 main（~60 行）
└── tests/
    ├── dump_gguf.py     # Python 侧 golden dump（gguf 包，输出同格式）
    └── test_b1.sh       # 构建 → dump ×2 → diff → 负例 → 抽查 的一键脚本
```

预计 C++ 净代码 ~350 行。`main.cpp` 骨架暂保留，B2 再接入。

## 3. 核心设计

### 3.1 数据结构

```cpp
enum class ValueType : uint32_t {
    UINT8, INT8, UINT16, UINT32=4, INT32, FLOAT32, BOOL, STRING, ARRAY,
    UINT64, INT64, FLOAT64
};  // 枚举值与 GGUF v3 线格式一致，从文件读入后显式校验范围

struct Value {
    ValueType type;
    std::variant<uint64_t, int64_t, double, bool,
                 std::string, std::vector<Value>> data;
};

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> dims;   // 原样保存，最快变化维在前（docs/20 §3.2 纪律）
    uint32_t ggml_type;
    uint64_t offset;              // 相对 data 区起点
    uint64_t n_bytes;             // 由 dims + type 查表算出（见 3.4）
};

class GGUFFile {
public:
    static GGUFFile open(const std::string& path);   // 失败抛 std::runtime_error
    const Value* kv(const std::string& key) const;   // 找不到返回 nullptr
    template<typename T> T kv_or(const std::string& key, T fallback) const;
    const TensorInfo* tensor(const std::string& name) const;
    const void* tensor_data(const TensorInfo&) const; // data_base + offset
    // 只读视图：header 三字段、kv 表、tensor 表、data_base
};
```

设计要点：**零拷贝**——`tensor_data` 返回指向 mmap 区域的裸指针，引擎全程不复制权重；`GGUFFile` 拥有映射句柄，析构时 munmap。

### 3.2 解析器：带边界的游标

```cpp
struct Cursor {
    const uint8_t* p; const uint8_t* end;
    template<typename T> T read();          // memcpy 出小端标量，越界抛异常
    std::string read_string();              // u64 长度 + 字节，长度 > 剩余即抛
};
```

纪律：**每一次读取前先查剩余字节**，所有解析失败都是带文件偏移量的 `runtime_error`，绝无越界读（验收 #4 靠这个）。不假设任何字段默认值——除了 `general.alignment`（缺省 32，这是格式规范写明的唯一缺省）。

### 3.3 mmap 与跨平台

- POSIX（WSL2 开发主环境）：`open → fstat → mmap(PROT_READ, MAP_PRIVATE)`；
- Windows（保持可编译）：`#ifdef _WIN32` 走 `CreateFileMapping/MapViewOfFile`；
- 只读映射，永不写回。

### 3.4 张量字节数计算（验收 #2 对账的关键）

`n_bytes = Π(dims) / block_size(type) × type_size(type)`。`ggml_types.h` 内置查表（本阶段只录入两个实测文件用到的类型，未知类型直接报错而不是猜）：

| type | 名 | block_size | type_size | 出现于 |
|---|---|---|---|---|
| 0 | F32 | 1 | 4 | norm 权重 |
| 1 | F16 | 1 | 2 | 主权重 |
| 8 | Q8_0 | 32 | 34 | Q4_K_M 文件部分张量 |
| 12 | Q4_K | 256 | 144 | Q4_K_M 主权重 |
| 14 | Q6_K | 256 | 210 | Q4_K_M 部分张量 |

（表内容开工时以 llama.cpp `ggml-common.h` 定义为准核对一遍；若实际文件出现表外类型，报错并在方案里补充，不静默跳过。）

### 3.5 对齐处理

data 区起点 = align_up(header + KV + infos 结束位置, alignment)；每个 tensor 实际起点 = data_base + offset（offset 本身已对齐，解析时**校验** `offset % alignment == 0`）。alignment 从 `general.alignment` 读，缺省 32。

## 4. dump 规范化格式（对拍协议）

两侧（C++ / Python）输出**完全相同的文本格式**，逐行：

```
GGUF version=3 tensors=310 kv=31 alignment=32
KV general.architecture = "qwen3vl" (str)
KV qwen3vl.block_count = 28 (u32)
KV qwen3vl.rope.dimension_sections = [24,20,20,0] (i32[], n=4)
...（数组逐元素全量打印；字符串原样不转义；浮点 %.17g）
TENSOR 0 token_embd.weight F16 dims=[2048,151936] offset=0 bytes=622854144
...
ACCOUNT header=24 kv_bytes=N infos_bytes=M padding=P data_bytes=D total=FILESIZE ok
HEAD output_norm.weight [v0,...,v7]      # F32/F16 张量首 8 值，纳入对拍
```

**实现时的一处计划修订**（2026-10-06）：原方案对大字符串/大数组打 sha256 摘要以控制 dump 体积，
实施时改为**全量保真 dump**（tokens/merges/chat_template 全部逐元素输出，单侧约 6 MB）。
理由：全量输出是严格强于哈希的校验（无哈希实现一致性风险），且 Python 侧全程按原始字节
透传，两侧字节流逐一致即可 `cmp` 判零；6 MB 的 diff 输入在 /tmp 下毫无压力。
Python 侧用**独立手写解析器**（仅 stdlib：struct + mmap，不用 gguf 包），构成真正的双实现交叉验证。
浮点用 `%.17g` 保证双精度可逆。

## 5. 负例测试清单（验收 #4）

| 用例 | 构造 | 期望 |
|---|---|---|
| 坏 magic | 前 4 字节改写 | "bad magic" 报错，rc≠0 |
| 截断 | 取文件前 1MB | 解析到越界处干净报错 |
| 张量 offset 越界 | 手改一个 offset 指向文件尾后 | 越界报错 |
| 声明长度爆炸 | string length = 2^60 | 长度>剩余即报错，不分配 |

负例文件由 `test_b1.sh` 用 Python 现场生成到 `/tmp`，不污染仓库。

## 6. 明确不做（防范围蔓延）

- 不做 F16→F32 转换函数（B3 才需要，届时放 `ops`）；
- 不做 Q4_K/Q6_K 反量化（B5；本阶段只需知道它们的字节数）；
- 不做任何前向计算、不建张量抽象（`tensor.h` 留到 B3）；
- 不解析 chat template 语义（当作普通字符串 KV）。

## 7. 工时分解

| 步 | 内容 | 预计 |
|---|---|---|
| 1 | `ggml_types.h` + `gguf.h` 接口 | 短 |
| 2 | `gguf.cpp` mmap + Cursor + KV/info 解析 | 主体 |
| 3 | `gguf_dump.cpp` + `dump_gguf.py` 对格式 | 短 |
| 4 | `test_b1.sh` 五项验收全跑通 + 修零警告 | 主体 |
| 5 | 本方案文档回填实测结果 → docs/21 定稿 + git commit | 短 |

完成后回交：验收五项的实测输出 + commit。

---

## 8. 验收结果（2026-10-06 实测，全绿）

`bash cpp/tests/test_b1.sh`（WSL2 Ubuntu-24.04, gcc）输出 **9 passed, 0 failed**：

| # | 验收项 | 结果 |
|---|---|---|
| 1 | 构建零警告（`-Wall -Wextra`，含 radar-infer 骨架） | ✅ |
| 2 | golden diff：f16（3.44 GB）C++ vs Python 独立解析器 | ✅ **字节级一致**（含 151936 条 tokens、151387 条 merges、chat template、两个 HEAD 抽查） |
| 3 | golden diff：Q4_K_M（1.05 GB） | ✅ 字节级一致 |
| 4 | 字节对账 f16：`header=24 kv=5934184 infos=18253 padding=19 data=3441397760 tensors_sum=3441397760 pad_inside=0 total=3447350240` | ✅ 零字节无法解释，张量区零内部间隙 |
| 5 | 字节对账 Q4_K_M：`data=1101457408 pad_inside=0 total=1107409888` | ✅ |
| 6 | 负例 ×4：bad magic / 截断(1MB) / 声明 2^60 字符串 / 张量 offset 越界 | ✅ 全部干净报错退出（rc=1，stderr 带原因），无崩溃 |

实测顺带确认的两个事实：

- **两个文件的 tensor 排列顺序不同**（f16 首张量是 `token_embd.weight`，Q4_K_M 首张量是 `output_norm.weight`）——loader 按名索引，与顺序无关，这点在 B3 组装模型时同样要遵守。
- **Q4_K_M 实际类型分布**：norm = F32；绝大多数权重 = Q4_K；`token_embd` 与**每 3~5 层间隔的 `attn_v`、`ffn_down`** = Q6_K（如 blk.0/5/8/11/14...）。B5 写反量化时只需 Q4_K + Q6_K 两种 kernel。

**修复记录**：首次构建因声明了 move 语义导致默认构造函数被隐式删除（`GGUFFile() = default` 补回）；测试脚本模型路径多退一层目录（`../models/gguf` 而非 `../../models/gguf`）。均已修复并复测全绿。
