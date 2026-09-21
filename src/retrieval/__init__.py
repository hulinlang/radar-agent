"""P4 检索层：切分 / 索引 / 检索 / 评测。

本包内的模块必须是**无副作用的可复用模块**（§分层铁律）：
- 不读命令行、不写文件、不建目录、不打日志到 stderr
- 只提供函数与数据类，由 `scripts/p4_*.py` 负责参数解析、编排与落盘

子模块：
- `layout_render`  ：MinerU v2 块 → Markdown 文本的渲染与清洗（含公式/图注/表格）
- `chunking`       ：教材的块级版面感知切分主流程
- `paper_chunk`    ：英文文献按 section 切分（D4）
- `mmwave_chunk`   ：第二套语料（Markdown）的围栏状态机切分
- `checks`         ：教材切分的 DoD 断言
- `paper_checks`   ：文献切分的 DoD 断言（P-1..P-10）
- `tokenize`       ：分词器 —— **唯一实现**，索引侧与查询侧都只准调它
- `filters`        ：检索侧过滤规则（Front Matter）+ 图号/表号**精确**索引
- `index`          ：**统一**索引（教材+mmWave+文献）：BM25 + bge-m3 dense + RRF + 图号第三路
- `index_checks`   ：统一索引的 DoD 断言（U-1..U-11）

⚠️ 索引下标有两种语义，**混用会导致张冠李戴且不报错**：
  ① **全局下标**：在完整 chunks 列表里的位置
  ② **索引行号**：剔除 Front Matter 后，在 ids / dense / bm25 里的行号
  两者相差 34 行。跨边界传递下标必须过 `index.build_gid_map()`。
"""
