# Paper Reading Agent

按《论文阅读Agent开发计划V5》实现。主路径使用 `qwen3.8-flash`，默认**文本优先**：本地抽取带页码标记的全文送进主阅读请求，整页图片不再上传；模型读不出表格数字、图结构或公式时申请补看，系统按请求渲染 300 DPI 的整页或局部图，再发一轮定向修订。

数据流：

```
PDF → 本地抽取全文（--- PDF_PAGE=n ---）→ 主阅读请求（全文 + report schema）
    → 候选 report + visual_requests → 渲染补看图片 → 定向修订补丁（patch）
    → Python 合并/校验 → Q1–Q7 渲染 → 独立 Mermaid 阶段 → final.md
```

`input.mode: full_page_images` 保留为 A/B 基线：该模式忽略文本，照旧上传全部页面图片。

README 里的路径（`src/`、`prompts/`、`schemas/`、`workspace/`）都相对于项目根目录。

## 快速开始

```bash
uv sync
export DASHSCOPE_API_KEY='...'
export DASHSCOPE_BASE_URL='https://llm-mp7jqtxmeetphq4i.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'
uv run python -m src.batch path/to/paper.pdf --concurrency 1
```

如果当前工具进程无法继承终端环境，也可以在项目根目录创建未纳入 Git 的 `.env`：

```dotenv
DASHSCOPE_API_KEY=你的阿里云 DashScope API Key
DASHSCOPE_BASE_URL=https://llm-mp7jqtxmeetphq4i.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
```

程序只读取本地 `.env`，不会把内容写入日志或报告。`DASHSCOPE_BASE_URL`
环境变量（包括 `.env` 中的值）优先于 `model.default_base_url`；因此只修改
`config.yaml` 并不能覆盖一个仍指向旧域名的环境变量。启动时会校验最终 URL 必须是
带 `http://` 或 `https://` 的绝对地址。

也可以把当天的多篇 PDF 放入 `inbox/YYYY-MM-DD/`，直接把目录传给 batch。API key 只从环境变量读取，不写入配置或日志。

## 产物

每篇论文位于 `workspace/<paper-id>/`：`original.pdf`、页面图片、`pages.json`、`paper.txt`（带 `--- PDF_PAGE=n ---` 标记的全文）、运行原始响应、`report.json`、`final.md` 和事件日志。补看一轮后还会写入 `runs/<id>/crops/` 与 `supplemental.json`（该轮裁剪图的 id、页码、路径和原因），`final.md` 中的裁剪图链接和证据页校验都以它为准。批次入口位于 `output/YYYY-MM-DD/index.md`。

`done` 表示输入完整、结构检查通过并成功渲染；它不表示事实已被独立验证。

`needs_review` 表示存在需要人工核对的条目，具体原因写在 `report.json` 的 `unresolved_items` 中，并按 `final.md` 开头的“待核查事项”逐条列出。以下情况都会进入该列表，`report.json`、`final.md`、job 记录和每日索引四者始终一致：

- 模型自己标记为看不清或无法确认的内容；
- `confirmed: false` 的方法关系（方法图会省略它，因此必须人工核对）；
- 未执行的补看请求（页码不存在、crop 无效或本轮预算已用完），前缀为 `补看请求未能执行：`；
- 补看轮次中被丢弃的、页码非法或没有面积的补看请求；
- 补图看过之后仍然没有答案的请求，前缀为 `补看后仍未解决：`——它与“未能执行”是两件事：图发过了、问题还在，而不是根本没看图。补丁没有把它们写进 `resolved_visual_requests` 时按此归类；
- 补看补丁里无法应用的条目（引用了本轮没发过的图片、页码对不上、index 越界、引用了不存在的节点/边），前缀为 `补看补丁未应用：`：补丁按自然键合并，无法对齐的条目会被拒绝而不是猜着写进报告；
- 补看调用本身失败（端点错误、请求预算耗尽）：此时候选报告照常交付，不会被丢弃，失败原因记为 `补看轮次失败：`；
- 方法图阶段失败或省略了内容：前缀为 `方法图生成失败：` 和 `方法图已省略：`。图只是报告的一个视图，这一阶段失败只损失图，不影响已经付费读出来的报告。

渲染阶段自身的提示（例如节点 ID 被改写）只写入 job 记录，不影响状态。

## 配置

`config.yaml` 中的键按计划 V5 生效。未实现的取值会直接报错而不是被静默忽略，例如 `output.language != zh-CN`、`input.on_input_over_limit != fail_with_reason`、`verification.automated_verifier != false`。

输入模式只有两种实现，且组合必须自洽，否则启动即失败：

| `input.mode` | `extract_text` | `include_all_pdf_pages` | 行为 |
| --- | --- | --- | --- |
| `hybrid_text_visual`（默认） | 必须 `true` | 必须 `false` | 主请求只发全文；页面仍照常渲染，供补看与证据链接使用 |
| `full_page_images` | 忽略 | 必须 `true` | A/B 基线：照旧上传全部页面图片 |

`input.max_text_chars`（默认 400000，`null` 关闭）与 `input.max_pages` 都是发送前的护栏：全文文本和页数超限的论文直接以 `failure_reason: input_over_limit` 失败，不会先渲染图片、也不会等到端点报错才发现。

`refinement.*` 控制补看：`max_visual_rounds`（默认 1）、`max_crop_images`（默认 4，一轮最多渲染几张）、`crop_dpi`（默认 300）、`max_format_repairs`（默认 1）。格式修复配额按模型阶段分别计算：主阅读用掉一次修复后，补看 patch 或方法图仍各自拥有自己的修复机会；所有修复请求仍计入单篇总请求预算。

`diagrams.*` 控制方法图阶段：`enabled`、`max_diagrams`（默认 4）、`reuse_reading_prefix`（默认 true）。开启前缀复用时，方法图调用会把阅读调用的 prompt 前缀（system、阅读指令、全文、report schema）逐字节重放后再追加绘图任务，端点可以按缓存价计费这段前缀；事件日志的 `diagram_stage` 记录 `prefix_reused` 与 `prefix_chars`，便于核算。该阶段只读已确认的报告数据，不重新读 PDF，也不能改动报告里的任何事实。

`model.max_output_tokens` 必须显式设置：端点默认上限常常小于一份完整报告，会把 JSON 静默截断。截断的响应不会被重试，也不会被当作格式问题修复（补内容需要重新上传图片，代价高且通常无效）。当前值 131072 是端点自报的硬上界（`Range of max_tokens should be [1, 131072]`）。

`model.thinking_budget` 可按阶段配置。当前主阅读为 16384，补图为 8192，格式修复和方法图各为 4096；这是实际思考 token 上限，与 `max_output_tokens` 的可见输出上限相互独立。若开启 thinking 却不限制预算，简单的修复或绘图也可能进行很长的推理。

重试只有一层：批处理按 `batch.max_retries_per_call` 重试完整读取，批处理所创建的 `Reader` 不再内部重试，OpenAI SDK 的隐式重试也已关闭。因此 job 中的 `requests` 对应真实请求次数。事件日志会分别记录 `call_started`、`request_attempt_started` 和 `response_received`；主请求实际是否上传图片、文本字符数、thinking budget、响应 ID、用量和每次等待时间都可直接检查。

可见输出为空或只有空壳（`[]`、`null`、`{}`、不含任何报告必需字段的对象）时，既不做格式修复也不直接判失败：修复轮没有可重排的内容，只能凭空写一份报告。这类响应按 `batch.max_retries_per_call` 重走一次完整读取（重新发送全文并计费），事件日志中记为 `reader_failed` 且 `retryable: true`。与截断不同：截断是内容缺失，重发同样的大 prompt 只会再截断一次。

格式修复轮永远只发文本。它拿到的是解析失败的原始响应和 schema，`input.mode=full_page_images` 下也不会重新上传页面图片：修复要解决的是结构问题，重发图片只会再买一次同样的上下文。

`format_repaired: true`（job 记录）与报告头部的 Provenance 行表示这份报告是纯文本修复轮的产物——修复调用不带页面图片，内容可能被重构，因此引用与数字需要逐条对照证据。

模型偶尔会返回不存在的 evidence ID，或把方法边指向未定义节点。这类悬空引用不会被猜测修补，也不会为了通过校验而关闭完整性检查：系统会确定性地删除无效引用，省略无法落地的方法节点、边或步骤，并把原因写入 `unresolved_items`，因此报告仍可交付但状态为 `needs_review`。恢复一个已有 `report.json` 时也会执行同一检查，不会为这种本地可判定的问题重新调用模型。

模型也可能为同一页文本重复生成 evidence ID。若重复条目的来源身份一致，系统会合并其定位信息和引文；若同一 ID 指向不同来源，则为冲突条目生成 `__dupN` 别名并写入待核查事项，保留原引用指向首条证据，避免整篇报告因一个重复 ID 直接失败。

## Git 记录

当前环境中的 `.git` 目录是只读挂载，因此仓库元数据放在项目内的 `git-data/`。使用下面的命令查看历史和状态：

```bash
git --git-dir=git-data --work-tree=. log --oneline
git --git-dir=git-data --work-tree=. status
```

## 开发与校准

回归测试全部使用假客户端和临时生成的 PDF，不消耗 API 额度：

```bash
uv run pytest -q
```

其中端到端用例会真实跑完 preprocess → 主请求 → 补看补丁 → 方法图 → 校验 → 渲染 → 索引，覆盖全文进入主请求、主请求不再上传页面图片、补看按请求渲染指定 page/crop、文本与图片两类证据的校验、Q1–Q7 输出、多张 Mermaid 图与 subgraph，以及原有的重试/格式修复/unresolved_items 行为不回归。`pytest` 声明在 `pyproject.toml` 的 dev 组里：若从项目环境之外解析到 pytest，它无法导入 PyMuPDF，端到端用例会静默跳过而不是失败。

先用一篇已精读论文运行：

```bash
uv run python -m src.batch paper.pdf --concurrency 1
```

然后用 3–5 篇论文建立人工参考答案，比较 `hybrid_text_visual` 与 `full_page_images` 两种输入，记录严重错误、证据支持率、耗时和实际 token 用量。Verifier 默认关闭，只有校准显示有稳定净收益时才加入。

`--date` 只决定 `output/<date>/`，不会强制重新读取论文。需要无条件创建新 run 并再次
调用模型时使用 `--force`（等价别名：`--force-reread`）：

```bash
uv run python -u -m src.batch inbox/2026-09-15/ \
  --date 2026-09-15-v5m2 --concurrency 5 --max-requests-per-paper 8 --force
```

每次 `--force` 都为整个批次生成新的 nonce；该 nonce 与 PDF、配置/源码指纹共同形成新
run ID，因此即使目标日期和已有报告完全相同，也不会复用 `report.json`。这会产生新的
API 请求和费用。
