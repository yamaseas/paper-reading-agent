# Paper Reading Agent — 开发计划 V3（简化版）

> 目标：每天自动处理约 5 篇论文，生成结构化、可追溯的阅读报告；我负责从中选 1–2 篇精读，并可对任意论文追问。
>
> 约束：
> - 全程一个 API（qwen3.8-max，OpenAI 兼容端点）。
> - 批处理不用 Agent Harness，纯 Python 直连 API。
> - 效果优先，但用校准集数据决定模型，不预设立场。
> - V1 只做最小可用闭环，Verifier / 多模型对比 / 视觉验证都属于后续迭代。

---

## 1. 总体架构

```text
批处理（无状态，纯 Python + OpenAI SDK）          交互（有状态，OpenCode）
                                       
inbox/*.pdf                                workspace/<paper-id>/
    │                                          │
    v                                          v
预处理（PyMuPDF） ──→ paper.md ──→ Reader ──→ final.md ──→ 对任意论文 Q&A
（一次性脚本）              （一次 API 调用）      （paper.md + final.md
                                                塞回 context，多轮对话）
```

分工原则：

- **Python 脚本**：确定性流程 + 一次性 LLM 调用。批处理不需要 agent。
- **OpenCode**：只做交互层——精读时的 Q&A、手动深挖、按需看某页图。
- Reader 是**一次 API 调用**，不是多轮 agent loop：context = paper.md + 选定图片，prompt = 输出 schema。

---

## 2. 模型方案

### V1

| 用途 | 模型 | 说明 |
|---|---|---|
| Reader（批处理） | qwen3.8-max | 唯一接入的模型 |
| Q&A（交互） | qwen3.8-max | 同一个端点，OpenCode 配置一次即可 |

### 后续用校准集决定的事（不预设结论）

- Flash 级模型（Qwen3.8-Flash / GLM5.3-Flash）能否替代 Max 做日常 Reader——2026 代 Flash 不是弱模型，成本差一个数量级，让校准集分数说话。
- 是否需要独立 Verifier 第二遍——先看 V1 错误率。
- 是否需要跨模型交叉验证（Kimi K3 等）。

V1 明确不做：多模型 router、Verifier、GPT/Claude、本地模型。

---

## 3. PDF 输入策略：Text-first + Vision-on-demand

每篇论文的输入 = **完整结构化文本 + 3–8 张关键图**。

- 纯文本会丢 method 图、复杂表格、公式排版，所以保留少量视觉证据；
- 全页转图浪费 token、增加成本、难以定位引用，所以不整篇转图。

视觉证据由预处理脚本固定选出（V1 用启发式规则，不靠模型判断）：

- 含 "Figure 1 / architecture / pipeline / overview" 的页；
- 主结果表所在页；
- 解析失败的复杂表格页；
- 上限 8 页，DPI 160。

---

## 4. PDF 预处理

只用 PyMuPDF，不接第二套 parser（GROBID / Marker / MinerU 等留作解析质量不足时的备选）。

要求：

- 保留页码标记 `<!-- PAGE: n -->`，供 Evidence Index 引用；
- 输出 `paper.md` + `metadata.json` + `figures/`。

---

## 5. 单篇处理流程（V1 全部流程就这么多）

```text
paper.pdf
  → preprocess.py        → paper.md + 选定图片
  → reader.py（1 次 API 调用）→ draft.md
  → render_mermaid.py    → 从 method.json 生成 Mermaid，语法校验失败自动让模型修一轮（最多 2 次）
  → final.md
```

**V1 没有 Verifier。** 可追溯性由 Evidence Index 保证；校准集跑完后再用数据决定要不要加第二遍验证。

---

## 6. Reader 输出格式

每篇 final.md 的结构（Short Summary 置顶，扫一眼就能决定要不要精读）：

```markdown
# <论文标题>

## 摘要（200–300 字）

## Q1 解决什么问题
背景 / 现有不足 / 研究问题 / 声称的贡献
（明确区分"作者声称"与"Reader 解读"）

## Q2 相关工作
| Work | Problem | Method | Difference |

## Q3 方法
- 1–3 段概述（input / output / 核心 insight）
- method.json（节点带 section/page 证据）→ Mermaid 流程图
- 关键实现细节（model / prompt / 工具 / 超参；未报告就写 Not reported，禁止推测）

## Q4 实验
按原文 RQ 组织；主结果表带 Table/Figure/Page 引用

## Q5 局限与展望
作者明确写的 limitation / future work；
LLM 自己的分析单独标记 [LLM Analysis]，不冒充作者观点

## Q6 完整总结

## Evidence Index
| Claim | Evidence |
| 系统有三个阶段 | §3, Figure 2, p.5 |
```

核心原则：**Summary 必须可追溯，而不是"看起来合理"。**

已知风险：Mermaid 语法错和"脑补不存在的边"是最高发问题，所以流程图必须由 method.json 生成、校验、限制修复轮数，不允许模型直接凭印象画。

---

## 7. Q&A（对论文提问）

V1 就支持，实现很轻：

- Q&A session 的 context = `paper.md` + `final.md` + 对话历史，直接塞进 1M context；
- 不重读 PDF，不用 RAG / 向量库；
- 用 OpenCode 在 `workspace/<paper-id>/` 目录下起会话，模型需要看图时自己读 `figures/`。

---

## 8. 项目结构（砍到最少）

```text
paper-reading-agent/
├── inbox/YYYY-MM-DD/*.pdf
├── workspace/<paper-id>/
│   ├── original.pdf
│   ├── paper.md
│   ├── figures/
│   └── final.md
├── output/YYYY-MM-DD/*.md
├── prompts/
│   ├── reader.md          # Reader system prompt + 输出 schema
│   └── method.schema.json
├── src/
│   ├── preprocess.py      # PyMuPDF 抽取 + 选页渲染
│   ├── reader.py          # 一次 API 调用
│   ├── render_mermaid.py  # method.json → Mermaid + 校验修复
│   └── batch.py           # 并发 / 重试 / 日志
└── config.yaml
```

相比 V2 砍掉：独立 Verifier agent、opencode_worker、jobs 状态机（V1 用"失败重跑该篇"就够了）、多份 schema。

---

## 9. 配置

```yaml
model:
  provider: alibaba
  model: qwen3.8-max
  thinking: true

pdf:
  default_dpi: 160
  max_visual_pages: 8

batch:
  concurrency: 3
  max_retries: 2

output:
  language: zh-CN
```

---

## 10. 日志

每篇一行 JSON（batch 跑完追加到日志文件）：

```json
{"paper": "paper-01", "input_tokens": 0, "output_tokens": 0,
 "visual_pages": [4, 8], "latency_s": 0, "status": "done"}
```

用途：成本核算、判断 Flash 替代 Max 能省多少、定位失败模式。

---

## 11. 校准集（决定后续一切迭代的依据）

正式日用前，选 10–20 篇**我已精读过的论文**作为 ground truth（ProtocolGuard、RFCAudit、CyberGym、SWE-bench 等）。

评分项（每项 0–5）：

| 指标 | 关注点 |
|---|---|
| Problem / Contribution 正确性 | 有没有曲解论文动机 |
| Method 组件召回 | 关键组件找到多少 |
| Method 边正确率 | 正确边 / 生成边总数（直接衡量 Mermaid 可靠性） |
| 实验完整性 / 数字准确性 | 数据集、baseline、主结果数字 |
| 幻觉控制 | 有没有编不存在的内容 |
| 证据质量 | Evidence Index 引用是否真实可查 |

校准集要回答的三个问题，按优先级：

1. V1 的错误率是否需要加 Verifier？
2. Flash 能否替代 Max 做日常 Reader？
3. 视觉证据选页规则够不够用？

---

## 12. 开发阶段

### Phase 0 — 最小闭环（目标：1 篇 → 1 份合格报告）

- [ ] qwen3.8-max API 直连跑通
- [ ] PyMuPDF 抽取 + 页码标记
- [ ] Reader prompt + Q1–Q6 输出
- [ ] 用 ProtocolGuard 测试

DoD：生成结果可以替代我对普通论文的第一轮机械阅读。

### Phase 1 — 可追溯性

- [ ] Evidence Index
- [ ] method.json → Mermaid + 校验修复循环
- [ ] 启发式选页 + 图片输入

DoD：Method 和 Experiment 的主要事实都能回到原文找到证据。

### Phase 2 — 批处理 + Q&A

- [ ] 5 篇并发、重试、日志
- [ ] OpenCode 配置为交互 Q&A 前端
- [ ] `python src/batch.py inbox/2026-09-15`

DoD：早上丢 5 篇 PDF，拿到 5 份报告，能对任意一篇追问。

### Phase 3 — 用校准集做决策

- [ ] 跑校准集，按 §11 评分
- [ ] 根据数据决定：要不要 Verifier / 换 Flash / 改进选页规则

---

## 13. 相对 V2 的主要改动

| 改动 | 理由 |
|---|---|
| 删除 Verifier（降为 Phase 3 决策项） | 成本 ×2，价值未经校准集验证；Evidence Index 已提供追溯性 |
| 批处理不走 OpenCode，纯 Python 直连 | 非交互模式有已知权限/稳定性问题；token 统计和 context 组装需要完全可控 |
| OpenCode 只作交互 Q&A 层 | session 管理、多轮、按需看图才是它的价值 |
| 新增 Q&A 支持 | 精读时必然追问；context 复用 paper.md，成本极低 |
| Flash 模型进入校准集对比 | 新一代 Flash 与旗舰差距小、价差约一个数量级，用数据决定 |
| Short Summary 置顶 | 每天 5 份完整报告读不完，摘要先行帮助筛选精读对象 |
