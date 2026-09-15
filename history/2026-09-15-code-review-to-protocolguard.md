# 从 code review 到 ProtocolGuard 修复：2026-09-15 全过程记录

本文记录 2026-09-15 一天内发生的事：对第一版实现的代码审查、按审查结论做的修复、真实端点的端到端验证、五篇并发实跑及其暴露的问题，以及当天最后一个真实故障（ProtocolGuard）的诊断与修复。写给后续接手这个项目的人（包括未来的自己）。

时间均为本地时间（+08:00）。所有金额/额度相关的数字来自端点返回的 `usage`，不是估算。

## 0. 起点

`main` 上已有 7 个提交（最后一个是 `f695ce4 feat: support local env configuration for API keys`），实现完成但**从未用真实端点跑过**：主链路、补看、校验、渲染、批次编排都在，只有假客户端测试。

项目规则：API key 只从环境变量或本地 `.env` 读取，不写进配置、日志或报告。本文及当天所有产物中都没有出现过 key。

## 1. 代码审查（03:15）

请求是"审查代码，重点看功能上是否真能正常工作"。方法：读完整条链路，并用假 OpenAI 客户端把 `preprocess → read → 补看 → 校验 → 渲染 → 日报索引 → resume` 完整跑几遍（零 API 成本）。

结论：**主链路能跑通，但补看这条 V1 必备路径在常见情况下会把本来成功的阅读判成 `failed` 并丢弃报告。**

### 已验证跑通的

- 全流程无参数跑通，证据图片链接是相对路径且正确（`../../workspace/<id>/pages/page-001.png`）；resume 第二次运行 `skipped=True`；断点指纹按 PDF + config 分目录；本地渲染失败不会重付 API 费用。
- 越界 crop、非法坐标、超预算补看请求会被记原因而不是静默丢弃；Mermaid 只画 `confirmed` 边并做转义；JSON 解析容忍 ` ```json ` 围栏。

### P0：会让成功的结果变成 failed

| # | 问题 | 后果 |
|---|---|---|
| 1 | 补看证据引用裁剪图时校验失败（`semantic_unknown_image_id` 只认 `page-00X`，且没有任何调用方传 `extra_image_ids`／`supplemental_images`） | 整篇 `failed`，已付费的 report.json 被丢弃，resume 不复原 → 重跑再付一次全篇读图 |
| 2 | 补看合并时对未过滤的原始 requests 做 `int(pdf_page)`，遇到 `null`／`"3.5"` 抛 `TypeError` | 一个有效请求 + 一个非法请求同时存在即整篇失败 |
| 3 | 截断／非法 JSON 被 batch 层当可重试错误，重发 3 次全篇图片请求 | 同一确定性失败成本 ×3；且全流程**没有任何地方设置 `max_tokens`**，与计划 §14 冲突 |

### P1：计划要求了但没实现

| # | 问题 |
|---|---|
| 4 | `refinement.max_format_repairs: 1` 全代码 0 命中，只存在于 config.yaml；schema 是 `additionalProperties: false`，模型多写一个字段就直接 `failed`、无报告 |
| 5 | `prompts/refine.md` 是死文件——补看调用用的是 `reader.md`，模型被告知"这是全部输入、请重写完整报告"，实际只收到 2 张原页 + 2 张裁剪图 |
| 6 | 报告头永远写 `Status: running`（状态确定前渲染，之后不更新），同一篇 index 写 `done`、报告写 `running` |
| 7 | 报告缺"输入覆盖情况"，`metadata["input_coverage"]` 从未被传入 |
| 8 | 报告里的 `visual_requests` 被当致命错误（幻觉页码 99、`x0 >= x1` → 整篇 failed），与计划 §3.2"记录原因并保留未解决项"冲突 |
| 9 | 补看阶段产生的问题原因会丢：补看成功时 unresolved 只留在 `ReadResult`，而 batch 从 report.json 判定状态 → 实测 `status=done`、`unresolved_items=[]`，待核查信息静默消失 |

### P2：记账与配置一致性

`job["requests"]` 恒记 1（`ReadResult` 不是 `Mapping`，`isinstance` 分支永不命中）→ 每篇请求预算在 batch 层失真；`failed` 与 `needs_review` 不分（补看全部失败时有可交付的候选报告，应为 `needs_review`）；只要报告里还有 `visual_requests` 就写"本批次尚未执行补看"（即使已经执行过）；校验错误串是 Python dict 的 repr；`output.*`、`input.on_input_over_limit`、`input.include_all_pdf_pages` 等配置键改了不生效（计划 §14 专门提醒过）；`pages.json` 的 `dpi` 记的是请求值而非 `max_pixels` 缩放后的实际值，`printed_page` 恒为 `None`；同一 paper_id 换页数更少的 PDF 时 `pages/` 残留旧图。

### 授权

用户回复"都改都改"——完整授权修复 P0 + P1 + P2。

## 2. 按审查结论修复（03:15–03:41）

- **补图登记**：`Reader` 暴露补图清单（`image_id` + `pdf_page` + crop + dpi + reason），batch 写入 `run_dir/supplemental.json`，校验时传给 `validate_report`；`final.md` 的裁剪图链接和证据页校验都以它为准。
- **合并路径加固**：非法页码／坐标不再抛异常，统一走"记录原因 + 保留未解决项"。
- **重试语义分离**：reader 错误类（config／dependency／input／response）在 batch 层一律不可重试，避免同一确定性失败重发大图请求。
- **格式修复落地**：`max_format_repairs` 生效，纯文本、不带图片、计入每篇请求预算；修复结果仍不合法时保留原始报告（不替换成更模糊的错误）。
- **补看提示词**：新增 `DEFAULT_REFINE_INSTRUCTIONS` + `prompts/refine.md` 真正被加载，"这是补看轮、不是首读"。
- **状态一致性**：先按 unresolved 定状态再渲染，`index.md`／`final.md`／`output/*.md`／`job.json` 四者一致；`failed` 只用于"无法交付合格报告"。
- **配置生效**：未实现的取值直接报错而不是静默忽略（`_check_unsupported_config`）。

## 3. 真实端点验证（03:41–04:17）

三件事一起做：

1. **`max_output_tokens` 放大**：探针实测该端点 `max_tokens` 只约束**可见**输出（上限 32 时可见文本截到 31 字符、`finish_reason=length`，而 `completion_tokens` 记 566）；硬上界 `Range of max_tokens should be [1, 131072]`（200000 被 400 拒绝）。config 定为 131072，并在注释里记录这两个事实。
2. **补看路径实测**：4 张裁剪图在真实 300 DPI 下渲染（整页 300 DPI 会被 `image_max_pixels` 夹到约 167.5 DPI，这正是补看存在的理由）；新增 5 条精确引用表格数字的证据；66/898 字段被改写。
3. **并发实跑五篇**：峰值在飞请求 = 5，五个 `call_started` 落在 0.1 s 内，0 重试、0 个 429，墙钟 18.7 分钟。

## 4. 五篇运行暴露的问题与处置（04:17）

跑完五篇后给出 7 条问题，用户的处置与后续动作：

| # | 问题 | 处置 | 状态 |
|---|---|---|---|
| 1 | 覆盖声明（未引用页面的说明） | 用户否决：未引用页是附录，价值没法与正文比 | 未改 |
| 2 | `[]` 退化响应 | 用户判断是输出上限太低 → 设成端点最大 131072（我按原始证据说明 `finish_reason=stop`、可见输出就是两个字，不是截断） | 已做 |
| 3 | 修复产物缺少来源标注 | 同意 | 已做：`format_repaired` + 报告头 Provenance 行 |
| 4 | 缺输入体量护栏 | 同意 | 已做：`input.max_pages`（默认 80，`null` 关闭），超限在渲染前以 `input_over_limit` 失败 |
| 5 | 不可复现（同输入不同产物） | 不重要 | 未改 |
| 6 | 补看成本/收益 | 先这样 | 未改 |

补充修复（审查之外的发现）：补看轮会把候选报告的 `visual_requests` 原样回抄，导致"补看请求未能执行"的幻影条目；改为按请求键（页码 + crop 坐标）判定是否真的执行过，已执行的归入新前缀 `补看后仍未解决：`，并在 `refine.md` 规则 7 要求模型清空已执行请求。补看轮本身失败时不再丢弃候选报告，降级为 `needs_review` 并记 `补看轮次失败：`。

## 5. ProtocolGuard 故障：诊断、修复、retry（04:44 之后）

五篇里 `Song 等 - 2026 - ProtocolGuard ...` 失败。事件链（旧 run `e6f97a68f9a4a4ba`）：

```
job_started → preprocess_succeeded(18 页) → call_started → repair_started
→ repair_finished → call_finished(format_repaired=true, finish_reason=stop)
→ reader_succeeded → validation_failed
```

模型的可见输出是 `[]`（思考 17,266 tokens ≈ 65k 字符，`finish_reason=stop`，**不是截断**）。`[]` 是合法 JSON 但不是对象 → 被标成"可修复" → 走那一轮纯文本修复。修复轮没有图片、待修内容只有两个字 `[]`，模型只能凭空写一份报告：`evidence = 0`、`claims = 2`、13 个 `evidence_ids` 全空 → 本地校验拒绝 → `report validation failed`。

**根因不是模型偶发退化本身，而是把"零内容"当成了"格式错误"。**

修复（`src/reader.py` + `src/batch.py`）：

- 新增 `_carries_no_report`：空消息、`[]`／`null`／`{}`／任何不含报告必需字段的对象 = 没有内容可修 → `ReaderResponseError(retryable_with_images=True)`，**不消耗修复轮**；同时覆盖 schema 修复路径，`{"a": 1}` 这类能通过解析的空壳也不会被送去"修"。
- 两个标志改为 `ReaderResponseError` 的类字段（`repairable`／`retryable_with_images`，互斥），batch 的 `_retryable` 对退化响应返回 true → 按 `batch.max_retries_per_call` 重走**完整读取**（重传图片并计费），事件里 `retryable: true`。
- 截断（`finish_reason=length`）仍不可修复也不可重试；"有内容但 JSON 坏了"仍走文本修复。
- 新增 6 个测试，含端到端用例：第一次返回 `[]`、第二次成功，并断言事件序列中**没有** `repair_started`。

retry 结果（run `eabff61094323db7`，配置指纹已变所以自动落在新目录，旧失败 run 保留）：第一次调用即成功，`finish_reason: stop`、`format_repaired: false`、无修复轮。

| 项 | 值 |
|---|---|
| 状态 | `needs_review`（2 条模型自标不确定项：Table IV 的 Average 口径、Figure 1 虚线箭头语义） |
| 内容 | 7 claims / 31 evidence / **空 `evidence_ids` 0 个**（对比修复前伪造版：2 claims / 0 evidence / 13 空） |
| 覆盖 | 全篇 18 页（PDF 页序 1-18） |
| 补看 | 2 张裁剪图（第 4、10 页），引用 Table III 数值 |
| 耗时 | 主请求 314.6 s + 补看 188.7 s |
| 用量 | 121,360 tokens（read 43,977 + 35,963；补看 20,336 + 21,084） |

⚠️ 单篇运行会用本次结果重写当日 `index.md`，把另外四篇的行冲掉。已用各自 `job.json` 的记录（`unresolved_items`／`summary`／`output_path`）调 `_write_index` 重建索引，零 API 成本。

## 6. 性能观察（被问到"为什么一次跑这么久"）

| 论文 | 页数 | prompt tokens | 生成 tokens | 其中思考 | 主请求耗时 |
|---|---|---|---|---|---|
| HITS | 9 | 27,667 | 9,631 | 1,383 | ~1 分钟 |
| ProtocolGuard | 18 | 43,977（41,616 是图片） | 35,963 | 24,366 | 314.6 s |
| RFCAudit | 13 | 32,326 | 41,109 | 33,566 | ~5 分钟 |
| CyberGym | 46 | 109,186 | 17,833 | 10,339 | ~5 分钟 |

图片输入（2.3k tokens/页）+ 开启的思考（占总生成量 60–85%）+ 15 章节结构化 JSON，是"比聊天窗口读论文慢"的三个来源；要压缩只能关 `thinking`、降 `render_dpi` 或只送正文页，都各有代价。

## 7. 当前状态与未决事项

- 状态：全部测试通过（43 个，`uv run pytest`），五篇产物在 `output/2026-09-15/`，本地全部改动连同本文一起提交。
- **未决 1**：另外四篇 run 在旧配置指纹下（`input.max_pages` 与 `prompts/refine.md` 规则 7 都是那之后加的）。重跑整个 `inbox/2026-09-15/` 会让四篇全部重新付费读取（约 200k prompt + 100k completion tokens），收益是统一指纹并消掉 MulVul 报告里四条因旧 refine 提示词产生的假"补看请求未能执行"。
- **未决 2**：MulVul 的交付报告来自一轮 schema 修复，但它的 run 早于 `format_repaired` 的实现，job 记录与报告头都没有 provenance 标注；可零成本本地补写（改 job.json + 重渲染）。
- **未决 3**：`schemas/report.schema.json` 的 `$defs/evidence_ids` 可加 `minItems: 1`（五篇真实报告每条 claim 都带证据，只有伪造版是 13 个空值），但这会改变配置指纹，强制全部重读。
- 覆盖声明（问题 1）按用户判断维持现状。

## 8. 复现命令

```bash
uv run pytest -q                                     # 全部测试，零 API 成本
uv run python -m src.batch <pdf> --concurrency 1     # 单篇
uv run python -m src.batch inbox/2026-09-15/ --concurrency 5
git --git-dir=git-data --work-tree=. log --oneline   # 本仓库的元数据在 git-data/
```
