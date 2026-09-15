from __future__ import annotations

import json
from pathlib import Path

from src.batch import _batch_date
from src.reader import Reader, ReaderConfig, parse_json_object
from src.render_report import render_mermaid, render_report
from src.validate import validate_report


def sample_report() -> dict:
    evidence = [
        {
            "id": "E1",
            "pdf_page": 1,
            "section": "3 Method",
            "figure_or_table": "Figure 1",
            "locator": "overview",
            "quote": "Input is transformed into output.",
            "image_id": "page-001",
        },
        {
            "id": "E2",
            "pdf_page": 2,
            "section": "4 Experiments",
            "figure_or_table": "Table 1",
            "locator": "main result row",
            "quote": "Accuracy: 90%.",
            "image_id": "page-002",
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
            "implementation_details": [],
            "evidence_ids": ["E1"],
        },
        "experiments": [
            {
                "research_question": "RQ1",
                "dataset": "Demo",
                "sample_size": "10",
                "baselines": ["Baseline"],
                "models": ["Model"],
                "metrics": ["Accuracy"],
                "settings": "Default",
                "main_results": ["90% accuracy"],
                "evidence_ids": ["E2"],
            }
        ],
        "authors_limitations": [],
        "reader_analysis": [],
        "full_summary": "A full summary.",
        "claims": [{"id": "C1", "text": "The method has two stages.", "kind": "author_claim", "evidence_ids": ["E1"]}],
        "evidence": evidence,
        "visual_requests": [],
        "unresolved_items": [],
    }


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
    output = tmp_path / "final.md"
    result = render_report(sample_report(), output, pages_json=pages_path)
    text = output.read_text(encoding="utf-8")
    assert result.output_path == output.resolve()
    assert "## Evidence Index" in text
    assert "page image" in text
    assert "```mermaid" in text


def test_reader_config_and_json_parser(tmp_path: Path) -> None:
    config = ReaderConfig.from_mapping({"model": {"name": "qwen3.8-flash"}})
    assert config.model == "qwen3.8-flash"
    assert parse_json_object("```json\n{\"ok\": true}\n```") == {"ok": True}
    image = tmp_path / "page.png"
    image.write_bytes(b"png")


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


def test_batch_date_prefers_explicit_or_input_date(tmp_path: Path) -> None:
    dated = tmp_path / "2026-09-15"
    dated.mkdir()
    assert _batch_date(dated) == "2026-09-15"
    assert _batch_date(tmp_path / "paper.pdf", "2026-01-02") == "2026-01-02"
