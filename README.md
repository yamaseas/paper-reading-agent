# Paper Reading Agent

按《论文阅读Agent开发计划V4-最终版》实现的第一版骨架。主路径使用 `qwen3.8-flash`，将 PDF 全部页面逐页渲染后在一次主要请求中发送；模型需要更清晰证据时最多自动补看一轮局部图。

## 快速开始

```bash
uv sync
export DASHSCOPE_API_KEY='...'
export DASHSCOPE_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
uv run python -m src.batch path/to/paper.pdf --concurrency 1
```

也可以把当天的多篇 PDF 放入 `inbox/YYYY-MM-DD/`，直接把目录传给 batch。API key 只从环境变量读取，不写入配置或日志。

## 产物

每篇论文位于 `workspace/<paper-id>/`：页面图片、`pages.json`、运行原始响应、`report.json`、`final.md` 和事件日志。批次入口位于 `output/YYYY-MM-DD/index.md`。

`done` 表示输入完整、结构检查通过并成功渲染；它不表示事实已被独立验证。存在关键证据或清晰度问题时报告会标记 `needs_review`。

## 开发与校准

先用一篇已精读论文运行：

```bash
uv run python -m src.batch paper.pdf --concurrency 1
```

然后用 3–5 篇论文建立人工参考答案，比较全页图片和带页码文本加关键图片的输入，记录严重错误、证据支持率、耗时和实际 token 用量。Verifier 默认关闭，只有校准显示有稳定净收益时才加入。
