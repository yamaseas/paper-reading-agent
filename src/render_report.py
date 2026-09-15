"""Render a validated paper report into human-readable Markdown.

``report.json`` is the source of truth for the contents of a report.  This
module deliberately contains no model calls and no PDF parsing.  It can
therefore be rerun after a local formatting error without paying for another
reading request.

The renderer accepts the small amount of page metadata produced by
``preprocess.py``.  Image links are calculated relative to the destination
Markdown file, so the same report can be rendered both in a run directory and
in the daily ``output`` directory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_NODE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


@dataclass
class RenderResult:
    """Details about a render operation."""

    output_path: Path
    warnings: list[str] = field(default_factory=list)
    linked_images: int = 0


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ", ".join(_text(item) for item in value)
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _items(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _join_lines(lines: Iterable[str]) -> str:
    """Join Markdown lines while ensuring one terminal newline."""

    return "\n".join(lines).rstrip() + "\n"


def _md_cell(value: Any) -> str:
    """Make arbitrary model text safe inside a Markdown table cell."""

    return _text(value).replace("|", "\\|").replace("\n", "<br>").strip()


def _bullets(values: Any, empty: str = "未提供") -> list[str]:
    entries = [_text(value).strip() for value in _items(values)]
    entries = [entry for entry in entries if entry]
    return [f"- {entry}" for entry in entries] or [f"- {empty}"]


def _evidence_refs(ids: Any) -> str:
    values = [_text(item).strip() for item in _items(ids)]
    values = [item for item in values if item]
    return ", ".join(f"`{item}`" for item in values) or "—"


def _kind_label(kind: Any) -> str:
    return {
        "author_claim": "作者声称",
        "reader_inference": "Reader 推断",
        "llm_analysis": "[LLM Analysis]",
    }.get(_text(kind), _text(kind))


def _safe_mermaid_label(value: Any) -> str:
    """Escape a node label without allowing model text to alter the graph."""

    label = _text(value, "未命名节点").replace("\\", "\\\\")
    label = label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    label = label.replace('"', "&quot;").replace("\r", "").replace("\n", "<br/>")
    return label


def _node_ids(nodes: Sequence[Mapping[str, Any]]) -> tuple[dict[str, str], list[str]]:
    """Return deterministic Mermaid IDs and warnings for malformed IDs."""

    mapping: dict[str, str] = {}
    used: set[str] = set()
    warnings: list[str] = []
    for index, node in enumerate(nodes, start=1):
        original = _text(node.get("id"), f"node_{index}")
        candidate = original if _NODE_ID.fullmatch(original) else f"node_{index}"
        if candidate in used:
            candidate = f"node_{index}"
        while candidate in used:
            candidate += "_"
        if candidate != original:
            warnings.append(f"method node {original!r} was renamed to {candidate!r} for Mermaid")
        used.add(candidate)
        # Duplicate IDs are invalid at the schema level.  Keeping the first
        # mapping makes an invalid report visible instead of drawing a random
        # edge to a different node.
        if original not in mapping:
            mapping[original] = candidate
    return mapping, warnings


def render_mermaid(method: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Generate a deterministic Mermaid flowchart from a method object.

    Only confirmed edges are rendered.  Unconfirmed edges are useful review
    hints, but drawing them as part of the executable-looking pipeline would
    make uncertainty indistinguishable from evidence-backed structure.
    """

    nodes = [item for item in _items(method.get("nodes")) if isinstance(item, Mapping)]
    edges = [item for item in _items(method.get("edges")) if isinstance(item, Mapping)]
    ids, warnings = _node_ids(nodes)
    lines = ["flowchart LR"]
    for index, node in enumerate(nodes, start=1):
        original = _text(node.get("id"), f"node_{index}")
        node_id = ids.get(original)
        if node_id is None:
            continue
        label = _text(node.get("name"), original)
        lines.append(f'    {node_id}["{_safe_mermaid_label(label)}"]')
    for edge_index, edge in enumerate(edges, start=1):
        source = ids.get(_text(edge.get("from")))
        target = ids.get(_text(edge.get("to")))
        if not source or not target:
            warnings.append(f"method edge {edge_index} references an unknown node and was omitted")
            continue
        if edge.get("confirmed") is not True:
            warnings.append(f"method edge {edge_index} is unconfirmed and was omitted from Mermaid")
            continue
        relation = _safe_mermaid_label(edge.get("relation"),)
        lines.append(f'    {source} -->|"{relation}"| {target}')
    if len(lines) == 1:
        lines.append("    empty[\"方法图不适用或暂无证据支持的节点\"]")
    return "\n".join(lines), warnings


def _read_pages(pages_json: Path | None) -> list[Mapping[str, Any]]:
    if pages_json is None or not pages_json.exists():
        return []
    try:
        value = json.loads(pages_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(value, Mapping):
        value = value.get("pages", [])
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _page_image_map(
    report_path: Path,
    pages_json: Path | None = None,
    pages: Sequence[Mapping[str, Any]] | None = None,
    supplemental_images: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Path]:
    """Map page/image IDs to absolute paths.

    Several field names are accepted because pages.json is an interchange
    artifact and early preprocess versions used ``path`` while later versions
    use ``image_path``.  Relative paths are always resolved against
    pages.json's directory.  ``supplemental_images`` adds the crops and
    re-rendered pages of a refinement round, whose paths are usually absolute.
    """

    pages_path = pages_json.resolve() if pages_json is not None else None
    entries = list(pages) if pages is not None else _read_pages(pages_path)
    base = pages_path.parent if pages_path is not None else report_path.parent
    result: dict[str, Path] = {}
    for index, page in enumerate(entries, start=1):
        page_number = page.get("pdf_page", page.get("page", index))
        try:
            page_number = int(page_number)
        except (TypeError, ValueError):
            page_number = index
        image_value = (
            page.get("image_path")
            or page.get("image")
            or page.get("path")
            or page.get("file")
        )
        if not image_value:
            continue
        image_path = Path(_text(image_value))
        if not image_path.is_absolute():
            image_path = base / image_path
        image_path = image_path.resolve()
        identifiers = {
            _text(page.get("id")),
            _text(page.get("image_id")),
            _text(page.get("page_id")),
            f"page-{page_number:03d}",
            f"page-{page_number}",
            str(page_number),
        }
        for identifier in identifiers:
            if identifier:
                result.setdefault(identifier, image_path)
    for image in supplemental_images or []:
        if not isinstance(image, Mapping):
            continue
        value = image.get("path", image.get("image_path"))
        if not value:
            continue
        image_path = Path(_text(value))
        if not image_path.is_absolute():
            image_path = base / image_path
        image_path = image_path.resolve()
        for identifier in (_text(image.get("image_id")), _text(image.get("id")), image_path.name):
            if identifier:
                result.setdefault(identifier, image_path)
    return result


def _image_for_evidence(
    evidence: Mapping[str, Any], report_path: Path, image_map: Mapping[str, Path]
) -> Path | None:
    image_id = _text(evidence.get("image_id")).strip()
    if image_id in image_map:
        return image_map[image_id]
    if image_id:
        candidate = Path(image_id)
        if not candidate.is_absolute():
            candidate = report_path.parent / candidate
        if candidate.exists():
            return candidate.resolve()
    page = evidence.get("pdf_page")
    try:
        page_number = int(page)
    except (TypeError, ValueError):
        return None
    return image_map.get(f"page-{page_number:03d}") or image_map.get(str(page_number))


def _relative_link(target: Path, destination: Path) -> str:
    try:
        relative = os.path.relpath(target, destination.parent)
    except ValueError:
        # Windows drives can make relative paths impossible.  Keep a useful
        # absolute path in the warning rather than emitting a broken link.
        relative = str(target)
    return Path(relative).as_posix()


def _image_markdown(target: Path, destination: Path, alt: str) -> str:
    link = _relative_link(target, destination)
    return f"![{alt}](<{link}>)"


def _render_problem(lines: list[str], problem: Mapping[str, Any]) -> None:
    lines.extend(["## Q1. 解决什么问题", "", "### 背景", _text(problem.get("background"), "未提供"), ""])
    lines.extend(["### 已有不足", _text(problem.get("existing_limitation"), "未提供"), ""])
    lines.extend(["### 研究问题", _text(problem.get("research_question"), "未提供"), ""])
    lines.extend(["### 动机", _text(problem.get("motivation"), "未提供"), ""])
    lines.extend(["### 作者声称的贡献", *_bullets(problem.get("claimed_contributions")), ""])
    lines.append(f"证据：{_evidence_refs(problem.get('evidence_ids'))}")
    lines.append("")


def _render_related_work(lines: list[str], related: Sequence[Any]) -> None:
    lines.extend(["## Q2. 相关工作", ""])
    if not related:
        lines.extend(["材料未提供直接相关工作的结构化信息。", ""])
        return
    lines.extend([
        "| Work | Problem | Method | Difference | Evidence |",
        "| --- | --- | --- | --- | --- |",
    ])
    for item in related:
        if not isinstance(item, Mapping):
            continue
        lines.append("| " + " | ".join([
            _md_cell(item.get("work")),
            _md_cell(item.get("problem")),
            _md_cell(item.get("method")),
            _md_cell(item.get("difference")),
            _md_cell(_evidence_refs(item.get("evidence_ids"))),
        ]) + " |")
    lines.append("")


def _render_method(
    lines: list[str],
    method: Mapping[str, Any],
    include_mermaid: bool,
    include_method_steps: bool = True,
) -> list[str]:
    warnings: list[str] = []
    lines.extend(["## Q3. 方法", ""])
    if method.get("applicable") is False:
        lines.extend(["本文不适用流程型方法图。", ""])
    lines.extend(["### 方法概述", _text(method.get("overview"), "未提供"), ""])
    lines.extend(["### 输入", *_bullets(method.get("inputs")), ""])
    lines.extend(["### 输出", *_bullets(method.get("outputs")), ""])

    if include_mermaid:
        graph, graph_warnings = render_mermaid(method)
        warnings.extend(graph_warnings)
        lines.extend(["### 方法图", "", "```mermaid", graph, "```", ""])

    nodes = [item for item in _items(method.get("nodes")) if isinstance(item, Mapping)]
    if nodes:
        lines.extend(["### 方法节点", "", "| ID | 名称 | 输入 | 操作 | 输出 | 证据 |", "| --- | --- | --- | --- | --- | --- |"])
        for node in nodes:
            lines.append("| " + " | ".join([
                _md_cell(node.get("id")),
                _md_cell(node.get("name")),
                _md_cell(_text(node.get("input"))),
                _md_cell(node.get("operation")),
                _md_cell(_text(node.get("output"))),
                _md_cell(_evidence_refs(node.get("evidence_ids"))),
            ]) + " |")
        lines.append("")

    steps = [item for item in _items(method.get("steps")) if isinstance(item, Mapping)]
    if include_method_steps:
        lines.extend(["### 方法步骤", ""])
        if steps:
            lines.extend(["| Step | Input | Operation | Tool / Model | Output | Why Needed | Evidence |", "| --- | --- | --- | --- | --- | --- | --- |"])
            for step in steps:
                lines.append("| " + " | ".join([
                    _md_cell(step.get("step")),
                    _md_cell(step.get("input")),
                    _md_cell(step.get("operation")),
                    _md_cell(step.get("tool_or_model")),
                    _md_cell(step.get("output")),
                    _md_cell(step.get("why_needed")),
                    _md_cell(_evidence_refs(step.get("evidence_ids"))),
                ]) + " |")
        else:
            lines.append("未提供。")
        lines.append("")

    details = [item for item in _items(method.get("implementation_details")) if isinstance(item, Mapping)]
    lines.extend(["### 关键实现细节", ""])
    if details:
        lines.extend(["| 项目 | 内容 | 证据 |", "| --- | --- | --- |"])
        for detail in details:
            lines.append("| " + " | ".join([
                _md_cell(detail.get("name")),
                _md_cell(detail.get("value")),
                _md_cell(_evidence_refs(detail.get("evidence_ids"))),
            ]) + " |")
    else:
        lines.append("未提供。")
    lines.extend(["", f"方法证据：{_evidence_refs(method.get('evidence_ids'))}", ""])
    return warnings


def _render_experiments(lines: list[str], experiments: Sequence[Any]) -> None:
    lines.extend(["## Q4. 实验", ""])
    if not experiments:
        lines.extend(["未提供结构化实验信息。", ""])
        return
    for index, experiment in enumerate(experiments, start=1):
        if not isinstance(experiment, Mapping):
            continue
        lines.extend([f"### 实验 {index}", ""])
        fields = [
            ("研究问题 / 实验目的", "research_question"),
            ("数据集", "dataset"),
            ("样本规模", "sample_size"),
            ("Baseline", "baselines"),
            ("模型", "models"),
            ("指标", "metrics"),
            ("设置", "settings"),
        ]
        for label, key in fields:
            value = experiment.get(key)
            display = _text(value)
            if isinstance(value, list):
                display = _text(value)
            lines.append(f"**{label}：** {display or '未提供'}")
        lines.append("")
        lines.append("**主要结果：**")
        lines.extend(_bullets(experiment.get("main_results")))
        lines.append(f"证据：{_evidence_refs(experiment.get('evidence_ids'))}")
        lines.append("")


def _render_claims(lines: list[str], claims: Sequence[Any]) -> None:
    lines.extend(["## Claims", ""])
    if not claims:
        lines.extend(["未提供。", ""])
        return
    lines.extend(["| ID | 内容 | 归属 | Evidence |", "| --- | --- | --- | --- |"])
    for claim in claims:
        if not isinstance(claim, Mapping):
            continue
        lines.append("| " + " | ".join([
            _md_cell(claim.get("id")),
            _md_cell(claim.get("text")),
            _md_cell(_kind_label(claim.get("kind"))),
            _md_cell(_evidence_refs(claim.get("evidence_ids"))),
        ]) + " |")
    lines.append("")


def _render_evidence(
    lines: list[str],
    evidence: Sequence[Any],
    report_path: Path,
    image_map: Mapping[str, Path],
) -> int:
    lines.extend(["## Evidence Index", ""])
    if not evidence:
        lines.extend(["未提供。", ""])
        return 0
    lines.extend(["| ID | PDF 页 | Section | Figure / Table | Locator | Quote | Image |", "| --- | ---: | --- | --- | --- | --- | --- |"])
    linked = 0
    for item in evidence:
        if not isinstance(item, Mapping):
            continue
        image_path = _image_for_evidence(item, report_path, image_map)
        image = "—"
        if image_path is not None:
            image = f"[page image]({_relative_link(image_path, report_path)})"
            linked += 1
        lines.append("| " + " | ".join([
            _md_cell(item.get("id")),
            _md_cell(item.get("pdf_page")),
            _md_cell(item.get("section")),
            _md_cell(item.get("figure_or_table")),
            _md_cell(item.get("locator")),
            _md_cell(item.get("quote")),
            _md_cell(image),
        ]) + " |")
    lines.append("")
    return linked


def _render_unresolved(lines: list[str], report: Mapping[str, Any]) -> None:
    """Render the pending-review list that the header status refers to."""

    items = [_text(item).strip() for item in _items(report.get("unresolved_items"))]
    items = [item for item in items if item]
    lines.extend(["## 待核查事项", ""])
    lines.extend([f"- {item}" for item in items] or ["无。"])
    lines.append("")


def render_report(
    report: Mapping[str, Any],
    output_path: str | os.PathLike[str],
    *,
    pages_json: str | os.PathLike[str] | None = None,
    pages: Sequence[Mapping[str, Any]] | None = None,
    supplemental_images: Sequence[Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
    include_mermaid: bool = True,
    include_method_steps: bool = True,
    include_evidence_index: bool = True,
) -> RenderResult:
    """Render ``report`` to ``output_path`` atomically.

    ``pages_json`` should be supplied whenever the destination is outside the
    run directory.  The paths in it are interpreted relative to pages.json,
    and are then converted into links relative to the Markdown destination.
    ``supplemental_images`` describes the crops supplied during one visual
    refinement round so their evidence entries link to the crop itself.
    """

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    pages_path = Path(pages_json).expanduser() if pages_json is not None else None
    image_map = _page_image_map(destination, pages_path, pages, supplemental_images)
    head: list[str] = []
    title = _text(report.get("title"), report.get("paper_id", "Paper report"))
    head.extend([f"# {title}", ""])
    paper_id = _text(report.get("paper_id"), "unknown")
    head.append(f"- **Paper ID:** `{paper_id}`")
    if metadata:
        status = metadata.get("status")
        if status:
            head.append(f"- **Status:** `{_text(status)}`")
        coverage = metadata.get("input_coverage") or metadata.get("coverage")
        if coverage:
            head.append(f"- **Input coverage:** {_text(coverage)}")
        run_id = metadata.get("run_id")
        if run_id:
            head.append(f"- **Run:** `{_text(run_id)}`")
        if metadata.get("format_repaired"):
            head.append(
                "- **Provenance:** 本报告由一轮纯文本结构修复生成（修复时没有重新提供页面图片），"
                "内容可能被重构；引用与数字请对照证据逐条核对。"
            )
    head.append("")

    # The pending-review list belongs in the opening block: the status in the
    # header is only meaningful next to the reasons for it.
    _render_unresolved(head, report)

    # The body is rendered separately so the opening block can be assembled
    # before it, and so the render warnings it produces stay out of the file.
    lines: list[str] = ["## 摘要", "", _text(report.get("short_summary"), "未提供"), ""]

    problem = report.get("problem")
    _render_problem(lines, problem if isinstance(problem, Mapping) else {})
    _render_related_work(lines, _items(report.get("related_work")))
    method = report.get("method")
    warnings = _render_method(
        lines,
        method if isinstance(method, Mapping) else {},
        include_mermaid,
        include_method_steps,
    )
    _render_experiments(lines, _items(report.get("experiments")))

    lines.extend(["## Q5. 局限与展望", "", "### 作者明确报告的局限 / 未来工作", ""])
    limitations = _items(report.get("authors_limitations"))
    if limitations:
        for item in limitations:
            if isinstance(item, Mapping):
                lines.append(f"- {_text(item.get('text'), '未提供')}（证据：{_evidence_refs(item.get('evidence_ids'))}）")
            else:
                lines.append(f"- {_text(item)}")
    else:
        lines.append("未提供。")
    lines.extend(["", "### Reader 分析", ""])
    analysis = _items(report.get("reader_analysis"))
    if analysis:
        for item in analysis:
            if isinstance(item, Mapping):
                lines.append(f"- **[LLM Analysis]** {_text(item.get('text'), '未提供')}（证据：{_evidence_refs(item.get('evidence_ids'))}）")
            else:
                lines.append(f"- **[LLM Analysis]** {_text(item)}")
    else:
        lines.append("未提供。")
    lines.append("")

    lines.extend(["## Q6. 完整总结", "", _text(report.get("full_summary"), "未提供"), ""])
    _render_claims(lines, _items(report.get("claims")))

    visual_requests = [item for item in _items(report.get("visual_requests")) if isinstance(item, Mapping)]
    lines.extend(["## 待补看请求", ""])
    if visual_requests:
        lines.extend(["| PDF 页 | 原因 | Crop |", "| ---: | --- | --- |"])
        for request in visual_requests:
            crop = request.get("crop")
            lines.append("| " + " | ".join([
                _md_cell(request.get("pdf_page")),
                _md_cell(request.get("reason")),
                _md_cell(crop if crop is not None else "整页"),
            ]) + " |")
    else:
        lines.append("无。")
    lines.append("")

    linked = 0
    if include_evidence_index:
        linked = _render_evidence(lines, _items(report.get("evidence")), destination, image_map)

    lines = [*head, *lines, "", "---", "", "由 `report.json` 确定性生成；`done` 不代表事实已经独立核查。", ""]
    content = _join_lines(lines)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=str(destination.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return RenderResult(output_path=destination, warnings=warnings, linked_images=linked)


def render_report_file(
    report_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
    *,
    pages_json: str | os.PathLike[str] | None = None,
    metadata_path: str | os.PathLike[str] | None = None,
    include_mermaid: bool = True,
) -> RenderResult:
    report_file = Path(report_path).expanduser().resolve()
    report = json.loads(report_file.read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise ValueError("report.json root must be an object")
    destination = Path(output_path).expanduser() if output_path is not None else report_file.with_name("final.md")
    metadata: Mapping[str, Any] | None = None
    if metadata_path is not None:
        metadata_value = json.loads(Path(metadata_path).expanduser().read_text(encoding="utf-8"))
        if isinstance(metadata_value, Mapping):
            metadata = metadata_value
    return render_report(report, destination, pages_json=pages_json, metadata=metadata, include_mermaid=include_mermaid)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a report.json as Markdown")
    parser.add_argument("report", type=Path, help="path to report.json")
    parser.add_argument("--output", type=Path, help="destination Markdown path (default: sibling final.md)")
    parser.add_argument("--pages", type=Path, help="pages.json used to resolve evidence image links")
    parser.add_argument("--metadata", type=Path, help="optional metadata/job JSON shown in the header")
    parser.add_argument("--no-mermaid", action="store_true", help="omit the Mermaid method graph")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = render_report_file(
        args.report,
        args.output,
        pages_json=args.pages,
        metadata_path=args.metadata,
        include_mermaid=not args.no_mermaid,
    )
    for warning in result.warnings:
        print(f"warning: {warning}")
    print(result.output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

