# Paper Reading Agent

按《论文阅读Agent开发计划V4-最终版》实现的第一版骨架。主路径使用 `qwen3.8-flash`，将 PDF 全部页面逐页渲染后在一次主要请求中发送；模型需要更清晰证据时最多自动补看一轮局部图。

## 快速开始

```bash
uv sync
export DASHSCOPE_API_KEY='...'
export DASHSCOPE_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
uv run python -m src.batch path/to/paper.pdf --concurrency 1
```

如果当前工具进程无法继承终端环境，也可以在项目根目录创建未纳入 Git 的 `.env`：

```dotenv
DASHSCOPE_API_KEY=你的阿里云 DashScope API Key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

程序只读取本地 `.env`，不会把内容写入日志或报告。

也可以把当天的多篇 PDF 放入 `inbox/YYYY-MM-DD/`，直接把目录传给 batch。API key 只从环境变量读取，不写入配置或日志。

## 产物

每篇论文位于 `workspace/<paper-id>/`：页面图片、`pages.json`、运行原始响应、`report.json`、`final.md` 和事件日志。补看一轮后还会写入 `supplemental.json`（该轮裁剪图的 id、页码、路径和原因），`final.md` 中的裁剪图链接和证据页校验都以它为准。批次入口位于 `output/YYYY-MM-DD/index.md`。

`done` 表示输入完整、结构检查通过并成功渲染；它不表示事实已被独立验证。

`needs_review` 表示存在需要人工核对的条目，具体原因写在 `report.json` 的 `unresolved_items` 中，并按 `final.md` 开头的“待核查事项”逐条列出。以下情况都会进入该列表，`report.json`、`final.md`、job 记录和每日索引四者始终一致：

- 模型自己标记为看不清或无法确认的内容；
- `confirmed: false` 的方法关系（方法图会省略它，因此必须人工核对）；
- 未执行的补看请求（页码不存在、crop 无效或本轮预算已用完），前缀为 `补看请求未能执行：`；
- 补看轮次中被丢弃的、页码非法或没有面积的补看请求；
- 补图看过之后仍然没有答案的请求，前缀为 `补看后仍未解决：`——它与“未能执行”是两件事：图发过了、问题还在，而不是根本没看图。模型把候选报告的请求列表原样回抄时按此归类（`refine.md` 规则 7 要求它把已执行的请求清空）；
- 补看调用本身失败（端点错误、请求预算耗尽）：此时候选报告照常交付，不会被丢弃，失败原因记为 `补看轮次失败：`。

渲染阶段自身的提示（例如节点 ID 被改写）只写入 job 记录，不影响状态。

## 配置

`config.yaml` 中的键按计划 V4 生效。未实现的取值会直接报错而不是被静默忽略，例如 `output.language != zh-CN`、`input.include_all_pdf_pages != true`、`input.on_input_over_limit != fail_with_reason`、`verification.automated_verifier != false`。

`model.max_output_tokens` 必须显式设置：端点默认上限常常小于一份完整报告，会把 JSON 静默截断。截断的响应不会被重试，也不会被当作格式问题修复（补内容需要重新上传图片，代价高且通常无效）。当前值 131072 是端点自报的硬上界（`Range of max_tokens should be [1, 131072]`）。

可见输出为空或只有空壳（`[]`、`null`、`{}`、不含任何报告必需字段的对象）时，既不做格式修复也不直接判失败：修复轮没有可重排的内容，只能凭空写一份报告。这类响应按 `batch.max_retries_per_call` 重走一次完整读取（重新上传页面图片并计费），事件日志中记为 `reader_failed` 且 `retryable: true`。与截断不同：截断是内容缺失，重发同样的大 prompt 只会再截断一次。

`input.max_pages`（默认 80，`null` 关闭）是发送前的体量护栏：整篇论文按图片一次性上传，图片占 prompt 的绝大部分，实测 46 页 109k prompt tokens、11–14 页 27k–44k。超过上限的 PDF 直接以 `failure_reason: input_over_limit` 失败，不会先渲染图片、也不会等到端点报错才发现。

`format_repaired: true`（job 记录）与报告头部的 Provenance 行表示这份报告是纯文本修复轮的产物——修复调用不带页面图片，内容可能被重构，因此引用与数字需要逐条对照证据。

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

其中端到端用例会真实跑完 preprocess → 主请求 → 补看 → 校验 → 渲染 → 索引，覆盖补看裁剪图、格式修复和状态推导。`pytest` 声明在 `pyproject.toml` 的 dev 组里：若从项目环境之外解析到 pytest，它无法导入 PyMuPDF，端到端用例会静默跳过而不是失败。

先用一篇已精读论文运行：

```bash
uv run python -m src.batch paper.pdf --concurrency 1
```

然后用 3–5 篇论文建立人工参考答案，比较全页图片和带页码文本加关键图片的输入，记录严重错误、证据支持率、耗时和实际 token 用量。Verifier 默认关闭，只有校准显示有稳定净收益时才加入。
