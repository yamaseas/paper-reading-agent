from __future__ import annotations

import json
from pathlib import Path

import openai
import pytest

from src.batch import BatchError, _batch_date
from src.preprocess import make_paper_id
from src.reader import Reader, ReaderConfig, ReaderConfigError, _safe_error, parse_json_object
from src.render_report import render_mermaid, render_report
from src.validate import validate_report


def sample_report() -> dict:
    # E1 is read from the page-marked full text, E2 from a page image: both
    # source types have to survive validation, rendering and Mermaid.
    evidence = [
        {
            "id": "E1",
            "source_type": "text",
            "pdf_page": 1,
            "source_id": "page-001-text",
            "section": "3 Method",
            "figure_or_table": "Figure 1",
            "locator": "overview",
            "quote": "Input is transformed into output.",
        },
        {
            "id": "E2",
            "source_type": "image",
            "pdf_page": 2,
            "source_id": "page-002",
            "section": "4 Experiments",
            "figure_or_table": "Table 1",
            "locator": "main result row",
            "quote": "Accuracy: 90%.",
        },
    ]
    return {
        "schema_version": "1",
        "paper_id": "demo",
        "title": "A Demonstration Paper",
        "short_summary": "A short summary.",
        "problem": {
            "background": "Background.",
            "existing_limitation": "Limitation.",
            "research_question": "Question?",
            "motivation": "Motivation.",
            "claimed_contributions": ["Contribution."],
            "evidence_ids": ["E1"],
        },
        "related_work": [],
        "method": {
            "applicable": True,
            "overview": "A two-stage method.",
            "inputs": ["input"],
            "outputs": ["output"],
            "intermediate_artifacts": [
                {
                    "name": "middle",
                    "description": "The representation Stage A passes to Stage B.",
                    "evidence_ids": ["E1"],
                }
            ],
            "tools_and_models": [
                {"name": "Tool", "role": "Transforms the input.", "evidence_ids": ["E1"]}
            ],
            "nodes": [
                {
                    "id": "A",
                    "name": "Stage A",
                    "input": ["input"],
                    "operation": "Transform",
                    "output": ["middle"],
                    "evidence_ids": ["E1"],
                },
                {
                    "id": "B",
                    "name": "Stage B",
                    "input": ["middle"],
                    "operation": "Predict",
                    "output": ["output"],
                    "evidence_ids": ["E1"],
                },
            ],
            "edges": [
                {"from": "A", "to": "B", "relation": "data_flow", "confirmed": True, "evidence_ids": ["E1"]},
                {"from": "B", "to": "A", "relation": "dependency", "confirmed": False, "evidence_ids": ["E1"]},
            ],
            "steps": [
                {
                    "step": "1",
                    "input": "input",
                    "operation": "Transform",
                    "tool_or_model": "Tool",
                    "output": "middle",
                    "why_needed": "Required.",
                    "evidence_ids": ["E1"],
                }
            ],
            "feedback_loops": [
                {
                    "description": "Stage B sends a correction back to Stage A.",
                    "node_ids": ["A", "B"],
                    "evidence_ids": ["E1"],
                }
            ],
            "decisions": [
                {
                    "condition": "Is the output valid?",
                    "branches": ["yes：输出", "no：回到 Stage A"],
                    "evidence_ids": ["E1"],
                }
            ],
            "implementation_details": [],
            "evidence_ids": ["E1"],
        },
        "experiments": [
            {
                "purpose": "Show that the two stages beat the baseline.",
                "research_question": "RQ1",
                "dataset": "Demo",
                "sample_size": "10",
                "baselines": ["Baseline"],
                "models": ["Model"],
                "metrics": ["Accuracy"],
                "settings": "Default",
                "main_results": ["90% accuracy"],
                "conclusion": "The method beats the baseline on the demo set.",
                "evidence_ids": ["E2"],
            }
        ],
        "future_work": {
            "authors_limitations": [{"text": "Only one dataset was used.", "evidence_ids": ["E2"]}],
            "authors_future_work": [{"text": "Scale the evaluation up.", "evidence_ids": ["E2"]}],
            "open_questions": [
                {"text": "Does the gain survive a stronger baseline?", "evidence_ids": ["E2"]}
            ],
            "research_directions": [
                {"text": "Ablate the two stages separately.", "evidence_ids": ["E1"]}
            ],
        },
        "reader_analysis": [
            {"text": "The two stages look independently ablatable.", "evidence_ids": ["E1"]}
        ],
        "full_summary": "A full summary.",
        "reading_guide": {
            "key_sections": [
                {"target": "3 Method", "text": "Read the two-stage design closely.", "evidence_ids": ["E1"]}
            ],
            "key_concepts": [
                {"target": "Stage B", "text": "Understand what Stage B predicts.", "evidence_ids": ["E1"]}
            ],
            "open_questions": [
                {"target": "", "text": "Where does the middle representation come from?", "evidence_ids": ["E1"]}
            ],
            "reproduction_notes": [
                {"target": "4 Experiments", "text": "The settings are one sentence long.", "evidence_ids": ["E2"]}
            ],
            "related_directions": [
                {"target": "", "text": "Compare with other two-stage pipelines.", "evidence_ids": ["E2"]}
            ],
        },
        "claims": [{"id": "C1", "text": "The method has two stages.", "kind": "author_claim", "evidence_ids": ["E1"]}],
        "evidence": evidence,
        "diagrams": sample_diagrams(),
        "visual_requests": [],
        "unresolved_items": [],
    }


def sample_diagrams() -> list[dict]:
    """Two diagrams: a phased overview with a feedback loop, and a second view.

    Written by hand in the shape the diagram stage produces, so the renderer is
    exercised on phase subgraphs, decision/loop/artifact shapes and a dashed
    feedback edge without a model call.
    """

    return [
        {
            "id": "overview",
            "title": "两阶段总览",
            "type": "flowchart",
            "direction": "LR",
            "nodes": [
                {"id": "Input", "label": "输入", "kind": "input", "group": "准备阶段", "evidence_ids": ["E1"]},
                {"id": "StageA", "label": "Stage A", "kind": "component", "group": "准备阶段", "evidence_ids": ["E1"]},
                {"id": "Middle", "label": "middle", "kind": "artifact", "group": "", "evidence_ids": ["E1"]},
                {"id": "StageB", "label": "Stage B", "kind": "component", "group": "推理阶段", "evidence_ids": ["E1"]},
                {"id": "Gate", "label": "输出是否有效？", "kind": "decision", "group": "", "evidence_ids": ["E1"]},
                {"id": "Retry", "label": "回到 Stage A", "kind": "loop", "group": "", "evidence_ids": ["E1"]},
                {"id": "Out", "label": "输出", "kind": "output", "group": "推理阶段", "evidence_ids": ["E1"]},
            ],
            "edges": [
                {"from": "Input", "to": "StageA", "label": "raw", "relation": "data_flow", "confirmed": True, "evidence_ids": ["E1"]},
                {"from": "StageA", "to": "Middle", "label": "produces", "relation": "data_flow", "confirmed": True, "evidence_ids": ["E1"]},
                {"from": "Middle", "to": "StageB", "label": "consumes", "relation": "data_flow", "confirmed": True, "evidence_ids": ["E1"]},
                {"from": "StageB", "to": "Gate", "label": "candidate", "relation": "control_flow", "confirmed": True, "evidence_ids": ["E1"]},
                {"from": "Gate", "to": "Retry", "label": "no", "relation": "feedback", "confirmed": True, "evidence_ids": ["E1"]},
                {"from": "Retry", "to": "StageA", "label": "iterate", "relation": "feedback", "confirmed": True, "evidence_ids": ["E1"]},
                {"from": "Gate", "to": "Out", "label": "yes", "relation": "control_flow", "confirmed": True, "evidence_ids": ["E1"]},
            ],
        },
        {
            "id": "evaluation",
            "title": "评估流程",
            "type": "flowchart",
            "direction": "TB",
            "nodes": [
                {"id": "Data", "label": "Demo", "kind": "environment", "group": "", "evidence_ids": ["E2"]},
                {"id": "Metric", "label": "Accuracy", "kind": "oracle", "group": "", "evidence_ids": ["E2"]},
            ],
            "edges": [
                {"from": "Data", "to": "Metric", "label": "is scored by", "relation": "dependency", "confirmed": True, "evidence_ids": ["E2"]},
            ],
        },
    ]


def pages_manifest() -> dict:
    return {
        "total_pages": 2,
        "pages": [
            {"id": "page-001", "image_id": "page-001", "pdf_page": 1, "path": "pages/page-001.png"},
            {"id": "page-002", "image_id": "page-002", "pdf_page": 2, "path": "pages/page-002.png"},
        ],
    }


def test_valid_report_and_page_semantics() -> None:
    result = validate_report(sample_report(), pages=pages_manifest())
    assert result.valid, result.as_dict()


def test_invalid_evidence_and_edge_are_reported() -> None:
    report = sample_report()
    report["method"]["edges"][0]["to"] = "missing"
    report["claims"][0]["evidence_ids"] = ["MISSING"]
    result = validate_report(report, pages=pages_manifest())
    assert not result.valid
    messages = "\n".join(issue.message for issue in result.errors)
    assert "unknown method node" in messages
    assert "unknown evidence id" in messages


def test_mermaid_omits_unconfirmed_edges_and_escapes_labels() -> None:
    graph, warnings = render_mermaid(sample_report()["method"])
    assert "A -->|\"data_flow\"| B" in graph
    assert "B -->" not in graph
    assert any("unconfirmed" in warning for warning in warnings)


def test_render_report_writes_evidence_links(tmp_path: Path) -> None:
    pages = pages_manifest()
    pages_path = tmp_path / "pages.json"
    pages_path.write_text(json.dumps(pages), encoding="utf-8")
    for page in pages["pages"]:
        image = tmp_path / page["path"]
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"placeholder")
    crop = tmp_path / "crops" / "crop-01.png"
    crop.parent.mkdir(parents=True, exist_ok=True)
    crop.write_bytes(b"crop")
    report = sample_report()
    # A crop is read from an image that is not in the stable page manifest, so
    # its link has to come from the refinement round that produced it.
    report["evidence"][1].update({"source_type": "image", "source_id": "crop-01"})
    output = tmp_path / "final.md"
    result = render_report(
        report,
        output,
        pages_json=pages_path,
        supplemental_images=[{"image_id": "crop-01", "pdf_page": 2, "path": str(crop)}],
    )
    text = output.read_text(encoding="utf-8")
    assert result.output_path == output.resolve()
    assert "## Evidence Index" in text
    # A text source links to the page its quote came from; a crop to the crop.
    assert "原文页" in text
    assert "补看图" in text
    assert "crop-01.png" in text
    assert result.linked_images == 2
    assert "```mermaid" in text


def test_text_and_image_evidence_both_pass_validation() -> None:
    """Both evidence sources are first class: neither is a legacy fallback."""

    report = validate_report(sample_report(), pages=pages_manifest())
    assert report.valid, report.as_dict()
    sources = {item["source_type"] for item in sample_report()["evidence"]}
    assert sources == {"text", "image"}


def test_text_evidence_naming_the_wrong_page_is_rejected() -> None:
    report = sample_report()
    report["evidence"][0]["source_id"] = "page-002-text"  # evidence says pdf_page 1
    result = validate_report(report, pages=pages_manifest())
    assert not result.valid
    assert any("names PDF page 2" in issue.message for issue in result.errors)


def test_legacy_image_id_evidence_is_still_checked() -> None:
    """A report written before evidence had a generic source keeps its check."""

    report = sample_report()
    report["evidence"][0].pop("source_type")
    report["evidence"][0].pop("source_id")
    report["evidence"][0]["image_id"] = "page-002"  # evidence says pdf_page 1
    result = validate_report(report, pages=pages_manifest())
    assert any("belongs to PDF page 2" in issue.message for issue in result.errors)


def test_final_markdown_answers_q1_to_q7(tmp_path: Path) -> None:
    output = tmp_path / "final.md"
    render_report(sample_report(), output)
    text = output.read_text(encoding="utf-8")
    headings = [
        "## Q1: 这篇论文试图解决什么问题？",
        "## Q2: 有哪些相关研究？",
        "## Q3: 论文如何解决这个问题？",
        "## Q4: 论文做了哪些实验？",
        "## Q5: 有什么可以进一步探索的点？",
        "## Q6: 总结一下论文的主要内容",
        "## Q7: 想要进一步了解论文",
    ]
    positions = [text.index(heading) for heading in headings]
    assert positions == sorted(positions), "the seven questions must be in order"
    # Q5 keeps the author's own words apart from the reader's analysis.
    q5 = text.split(headings[4], 1)[1].split(headings[5], 1)[0]
    assert "作者明确提出的 Future Work" in q5
    assert "**[LLM Analysis]**" in q5


def test_q7_is_not_empty_and_not_a_second_copy_of_q6(tmp_path: Path) -> None:
    """Q6 summarises the paper; Q7 says what to do next about it."""

    report = sample_report()
    output = tmp_path / "final.md"
    render_report(report, output)
    text = output.read_text(encoding="utf-8")
    q6 = text.split("## Q6: 总结一下论文的主要内容", 1)[1].split("### 关键论断（claims）", 1)[0]
    q7 = text.split("## Q7: 想要进一步了解论文", 1)[1].split("## Evidence Index", 1)[0]

    assert q6.strip(), "Q6 must carry the summary"
    assert "未提供。" not in q7, "Q7 must be filled in"
    for key, entry in report["reading_guide"].items():
        assert entry[0]["text"] in q7, f"{key} is missing from Q7"
    assert report["full_summary"] not in q7, "Q7 must not repeat Q6"
    assert report["reading_guide"]["key_sections"][0]["text"] not in q6


def test_every_diagram_is_rendered_with_its_subgraphs() -> None:
    from src.render_report import render_diagram

    diagrams = sample_diagrams()
    rendered = []
    for diagram in diagrams:
        graph, warnings = render_diagram(diagram)
        assert not warnings, warnings
        rendered.append(graph)

    overview = rendered[0]
    # A phase becomes a subgraph instead of one flat line.  The subgraph id
    # stays ASCII (Mermaid identifiers are), the title carries the phase name.
    assert 'subgraph group_1["准备阶段"]' in overview
    assert 'subgraph group_2["推理阶段"]' in overview and overview.count("    end") == 2
    # Shapes carry the kind: stadium input, parallelogram artifact, rhombus
    # decision, circle loop, subroutine output.
    assert 'Input(["输入"])' in overview
    assert 'Middle[/"middle"/]' in overview
    assert 'Gate{"输出是否有效？"}' in overview
    assert 'Retry(("回到 Stage A"))' in overview
    assert 'Out[["输出"]]' in overview
    # A feedback loop is dashed: it must not look like a straight-line hand-off.
    assert 'Gate -.->|"no · feedback"| Retry' in overview
    assert 'Retry -.->|"iterate · feedback"| StageA' in overview
    assert 'Gate -->|"yes · control_flow"| Out' in overview
    # The second diagram keeps its own direction and relation.
    assert "flowchart TB" in rendered[1]
    assert 'Data -.->|"is scored by · dependency"| Metric' in rendered[1]


def test_unconfirmed_diagram_edges_are_omitted_not_drawn() -> None:
    from src.render_report import render_diagram

    diagram = sample_diagrams()[0]
    diagram["edges"].append(
        {
            "from": "Out",
            "to": "Middle",
            "label": "guess",
            "relation": "dependency",
            "confirmed": False,
            "evidence_ids": ["E1"],
        }
    )
    graph, warnings = render_diagram(diagram)
    assert "Out -.->" not in graph
    assert any("unconfirmed" in warning for warning in warnings)


def test_the_renderer_lists_the_diagrams_of_a_report(tmp_path: Path) -> None:
    output = tmp_path / "final.md"
    render_report(sample_report(), output)
    text = output.read_text(encoding="utf-8")
    assert text.count("```mermaid") == 2, "one block per diagram"
    assert "#### 图 1：两阶段总览" in text
    assert "#### 图 2：评估流程" in text


def test_a_report_without_diagrams_falls_back_to_the_method_graph(tmp_path: Path) -> None:
    """A report written before the diagram stage still gets one picture."""

    report = sample_report()
    report.pop("diagrams")
    output = tmp_path / "final.md"
    render_report(report, output, include_mermaid=True)
    text = output.read_text(encoding="utf-8")
    assert text.count("```mermaid") == 1
    assert "#### 图 1" not in text


def test_reader_config_and_json_parser(tmp_path: Path) -> None:
    config = ReaderConfig.from_mapping({"model": {"name": "qwen3.8-flash"}})
    assert config.model == "qwen3.8-flash"
    assert parse_json_object("```json\n{\"ok\": true}\n```") == {"ok": True}
    image = tmp_path / "page.png"
    image.write_bytes(b"png")


def test_reader_config_rejects_a_base_url_without_http_scheme(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "dashscope.aliyuncs.com/compatible-mode/v1")
    with pytest.raises(ReaderConfigError, match=r"absolute HTTP\(S\) URL"):
        ReaderConfig.from_mapping({"model": {"name": "qwen3.8-flash"}})


def test_reader_disables_the_sdk_retry_layer(monkeypatch) -> None:
    captured: dict = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://example.test/v1")
    Reader(ReaderConfig.from_mapping({"model": {"name": "qwen3.8-flash"}})).client
    assert captured["max_retries"] == 0


def test_fake_reader_sends_all_pages_once(tmp_path: Path) -> None:
    class Completions:
        def __init__(self) -> None:
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            class Choice:
                finish_reason = "stop"
                message = type("Message", (), {"content": json.dumps(sample_report())})()
            return type("Response", (), {"id": "r1", "choices": [Choice()], "usage": {"total_tokens": 3}})()

    completions = Completions()
    fake_client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
    image1 = tmp_path / "page-001.png"
    image2 = tmp_path / "page-002.png"
    image1.write_bytes(b"one")
    image2.write_bytes(b"two")
    reader = Reader(
        ReaderConfig.from_mapping({"model": {"name": "qwen3.8-flash"}, "refinement": {"max_visual_rounds": 0}}),
        client=fake_client,
    )
    result = reader.read("demo", [{"pdf_page": 1, "path": image1}, {"pdf_page": 2, "path": image2}], schema={})
    assert result.report["paper_id"] == "demo"
    assert len(completions.calls) == 1
    content = completions.calls[0]["messages"][1]["content"]
    assert sum(item.get("type") == "image_url" for item in content) == 2


def test_a_chained_error_keeps_the_reason_that_is_not_in_the_outer_message(monkeypatch) -> None:
    # What the SDK says about a dead connection is "Connection error." and
    # nothing else.  The reason it gives up on is one level down, and a log
    # that drops it cannot tell a proxy outage from a bad base URL.
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-secret-value")
    error = RuntimeError("Connection error.")
    error.__cause__ = OSError("sk-secret-value: [Errno 111] Connection refused")
    message = _safe_error(error)
    assert message == (
        "RuntimeError: Connection error. <- "
        "OSError: [REDACTED]: [Errno 111] Connection refused"
    )


def test_a_cause_cycle_does_not_hang_the_error_renderer() -> None:
    first = RuntimeError("outer")
    second = RuntimeError("inner")
    first.__cause__ = second
    second.__cause__ = first
    assert _safe_error(first) == "RuntimeError: outer <- RuntimeError: inner"


def test_batch_date_prefers_explicit_or_input_date(tmp_path: Path) -> None:
    dated = tmp_path / "2026-09-15"
    dated.mkdir()
    assert _batch_date(dated) == "2026-09-15"
    assert _batch_date(tmp_path / "paper.pdf", "2026-01-02") == "2026-01-02"


def test_a_labelled_batch_date_is_a_directory_name_not_a_date(tmp_path: Path) -> None:
    # A re-run of the same day has to land beside the first one rather than on
    # top of it, so the label is allowed; it is still echoed back verbatim and
    # a directory called ``2026-09-15-v5`` is not mistaken for a dated input.
    assert _batch_date(tmp_path / "paper.pdf", "2026-09-15-v5") == "2026-09-15-v5"
    labelled = tmp_path / "2026-09-15-v5"
    labelled.mkdir()
    assert _batch_date(labelled) != "2026-09-15-v5"
    for bad in ("2026-9-15", "v5-2026-09-15", "2026-09-15/../x", "2026-09-15 "):
        with pytest.raises(BatchError):
            _batch_date(tmp_path / "paper.pdf", bad)


# --- paper identifiers ---------------------------------------------------


def test_a_library_stem_becomes_title_authors_year() -> None:
    stem = "Song 等 - 2026 - ProtocolGuard Detecting Protocol Non-compliance Bugs"
    assert make_paper_id(f"{stem}.pdf") == (
        "ProtocolGuard-Detecting-Protocol-Non-compliance-Bugs-Song-2026"
    )
    assert make_paper_id("Zheng 等 - 2025 - RFCAudit AI Agent.pdf") == "RFCAudit-AI-Agent-Zheng-2025"
    assert make_paper_id("Smith et al. - 2024 - Deep Learning for Cats.pdf") == (
        "Deep-Learning-for-Cats-Smith-et-al-2024"
    )
    # An author list without a year keeps the author at the end.
    assert make_paper_id("Wu 等 - MulVul Retrieval-augmented Detection.pdf") == (
        "MulVul-Retrieval-augmented-Detection-Wu"
    )


def test_a_stem_that_is_not_a_library_export_is_left_alone() -> None:
    for stem in ("paper", "just-a-hand-named-file", "probe-2024"):
        assert make_paper_id(f"{stem}.pdf") == stem


def test_an_explicit_paper_id_is_not_reordered() -> None:
    assert make_paper_id("Song 等 - 2026 - A Title.pdf", paper_id="A Title - Song 2026") == (
        "A-Title-Song-2026"
    )
