"""Regression tests for the failures found in the first review pass.

Each test here maps to a concrete defect: a refinement round that lost its
crops, a crash while collecting the follow-up images, a paid retry of a
truncated response, a rendered status that never left "running", and config
keys that were read but never applied.
"""

from __future__ import annotations

import json
import re
import sys
import threading
from pathlib import Path

import pytest

from src.batch import (
    BatchOptions,
    _config_fingerprint,
    _coverage_text,
    _requests_made,
    _retryable,
    process_one,
    run_batch,
)
from src.reader import (
    ReadResult,
    Reader,
    ReaderAPIError,
    ReaderConfig,
    ReaderConfigError,
    ReaderResponseError,
    normalize_report_schema_shape,
    reconcile_cross_references,
    schema_errors,
)
from src.render_report import render_report
from src.validate import validate_report
from tests.test_local import pages_manifest, sample_diagrams, sample_report

PROJECT = Path(__file__).resolve().parents[1]


class FakeResponse:
    def __init__(self, content: str, index: int = 1, finish_reason: str = "stop") -> None:
        self.id = f"resp-{index}"
        self.choices = [
            type(
                "Choice",
                (),
                {"finish_reason": finish_reason, "message": type("Msg", (), {"content": content})()},
            )()
        ]
        self.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}


class FakeClient:
    """Returns a scripted sequence of payloads and records every request."""

    def __init__(self, payloads: list, finish_reason: str = "stop") -> None:
        self.payloads = payloads
        self.calls: list[dict] = []
        self.finish_reason = finish_reason
        client = self

        class Completions:
            def create(inner, **kwargs):
                client.calls.append(kwargs)
                index = min(len(client.calls) - 1, len(client.payloads) - 1)
                payload = client.payloads[index]
                text = payload if isinstance(payload, str) else json.dumps(payload)
                return FakeResponse(text, len(client.calls), client.finish_reason)

        self.chat = type("Chat", (), {"completions": Completions()})()


def make_pdf(path: Path, pages: int = 3) -> Path:
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open()
    for index in range(pages):
        page = document.new_page()
        page.insert_text((72, 72), f"Page {index + 1}", fontsize=12)
    document.save(path)
    document.close()
    return path


def diagram_payload() -> dict:
    """The shape the separate Mermaid stage answers with."""

    return {"diagrams": sample_diagrams()}


def empty_patch(**overrides) -> dict:
    """A refinement patch that changes nothing.

    The refinement call returns a patch rather than a report, so every section
    is present and empty unless a test fills it.
    """

    patch = {
        "evidence_updates": [],
        "claim_updates": [],
        "method_updates": [],
        "experiment_updates": [],
        "resolved_visual_requests": [],
        "unresolved_items_add": [],
    }
    patch.update(overrides)
    return patch


def page_images(tmp_path: Path, count: int = 3) -> list[dict]:
    images = []
    for number in range(1, count + 1):
        path = tmp_path / f"page-{number:03d}.png"
        path.write_bytes(b"image")
        images.append({"pdf_page": number, "path": path})
    return images


def batch_options(tmp_path: Path, **overrides) -> BatchOptions:
    values = {
        "config_path": PROJECT / "config.yaml",
        "workspace_root": tmp_path / "workspace",
        "output_root": tmp_path / "output",
        "concurrency": 1,
        "max_retries": 2,
        "max_requests_per_paper": 6,
        "resume": True,
        "retry_backoff_s": 0.0,
        "batch_date": "2026-09-15",
    }
    values.update(overrides)
    return BatchOptions(**values)


# --- refinement images ---------------------------------------------------


def test_crop_evidence_is_accepted_with_supplemental_image_ids() -> None:
    report = sample_report()
    report["evidence"][0].update({"source_type": "image", "source_id": "crop-01"})
    rejected = validate_report(report, pages=pages_manifest())
    assert not rejected.valid, "a crop id must not pass without being declared"

    accepted = validate_report(
        report,
        pages=pages_manifest(),
        image_ids={"crop-01": 1},
    )
    assert accepted.valid, accepted.as_dict()


def test_crop_evidence_from_the_wrong_page_is_still_rejected() -> None:
    report = sample_report()
    report["evidence"][0].update(
        {"source_type": "image", "source_id": "crop-01"}  # evidence says pdf_page 1
    )
    result = validate_report(report, pages=pages_manifest(), image_ids={"crop-01": 2})
    assert not result.valid
    assert any("belongs to PDF page 2" in issue.message for issue in result.errors)


def test_refinement_reports_its_crops_and_does_not_crash_on_bad_requests(tmp_path: Path) -> None:
    candidate = sample_report()
    candidate["unresolved_items"] = ["Figure 3 caption was unreadable"]
    candidate["visual_requests"] = [
        {"pdf_page": 2, "reason": "ok", "crop": {"x0": 0.1, "y0": 0.1, "x1": 0.5, "y1": 0.5}},
        {"pdf_page": None, "reason": "malformed page", "crop": None},
        {"pdf_page": "2.5", "reason": "malformed page", "crop": None},
    ]
    # The refinement answer is a patch, and this patch changes nothing: the
    # item the first round set aside still has to reach the file, and the
    # malformed requests must be recorded rather than crash the round.
    fake = FakeClient([candidate, empty_patch()])
    images = page_images(tmp_path)
    crop_dir = tmp_path / "crops"

    def renderer(*, pdf_page, crop, dpi, source_page=None):
        crop_dir.mkdir(exist_ok=True)
        target = crop_dir / f"crop-{pdf_page:02d}.png"
        target.write_bytes(b"crop")
        return {"pdf_page": pdf_page, "path": target, "image_id": f"crop-{pdf_page:02d}", "crop": crop}

    reader = Reader(ReaderConfig.from_mapping({"model": {"name": "m"}}), client=fake)
    result = reader.read("demo", images, renderer=renderer)

    # The malformed requests are recorded instead of raising a TypeError, and
    # they are recorded before the schema check, so they never cost a repair.
    assert any("非法补看页码" in item for item in result.unresolved_items)
    assert len(fake.calls) == 2, "malformed requests must not trigger a format repair"
    assert [item["image_id"] for item in result.supplemental_images] == ["crop-02"]
    assert result.supplemental_images[0]["pdf_page"] == 2
    # The delivered report carries both the old and the new unresolved items.
    delivered = result.report["unresolved_items"]
    assert any("非法补看页码" in item for item in delivered)
    assert "Figure 3 caption was unreadable" in delivered
    assert "Figure 3 caption was unreadable" in result.unresolved_items
    # Only the accepted request is re-sent, and the images really are attached.
    second = fake.calls[1]["messages"][1]["content"]
    labels = [item["text"] for item in second if item.get("type") == "text" and item["text"].startswith("PDF_PAGE")]
    assert labels == ["PDF_PAGE=2 IMAGE_ID=page-002", "PDF_PAGE=2 IMAGE_ID=crop-02"]


def test_refinement_uses_the_refine_prompt_not_the_reader_prompt(tmp_path: Path) -> None:
    candidate = sample_report()
    candidate["visual_requests"] = [{"pdf_page": 2, "reason": "r", "crop": None}]
    fake = FakeClient([candidate, empty_patch()])
    images = page_images(tmp_path)

    def renderer(*, pdf_page, crop, dpi, source_page=None):
        target = tmp_path / f"recrop-{pdf_page}.png"
        target.write_bytes(b"crop")
        return {"pdf_page": pdf_page, "path": target, "image_id": f"crop-{pdf_page:02d}"}

    reader = Reader(
        ReaderConfig.from_mapping({"model": {"name": "m"}}),
        client=fake,
        prompt_path=PROJECT / "prompts" / "reader.md",
        refine_prompt_path=PROJECT / "prompts" / "refine.md",
    )
    reader.read("demo", images, renderer=renderer)
    first_text = fake.calls[0]["messages"][1]["content"][0]["text"]
    second_text = fake.calls[1]["messages"][1]["content"][0]["text"]
    assert "定向修订器" in second_text
    assert "定向修订器" not in first_text
    # The refinement call must not claim the whole paper was re-sent, and it
    # must say what it is asking for: a patch, not a second report.
    assert "complete input for this call" not in second_text
    assert "do not treat" in second_text
    assert "补丁（patch）" in second_text
    # The refinement call is a patch call, so the patch schema is what it is
    # asked to satisfy -- not the report schema.
    assert '"evidence_updates"' in second_text
    assert '"resolved_visual_requests"' in second_text


def test_a_patch_is_merged_into_the_candidate_instead_of_replacing_it(tmp_path: Path) -> None:
    """The refinement round may not rewrite parts of the report it did not see."""

    candidate = sample_report()
    candidate["visual_requests"] = [{"pdf_page": 2, "reason": "r", "crop": None}]
    fake = FakeClient(
        [
            candidate,
            empty_patch(
                claim_updates=[{"id": "C1", "text": "The method has two stages, verified."}],
                method_updates=[
                    {"target": "node", "id": "A", "operation": "Transform, then normalise"},
                    {"target": "edge", "from": "B", "to": "A", "confirmed": True, "evidence_ids": ["E1"]},
                ],
                unresolved_items_add=["Still unknown: the exact sample size"],
            ),
        ]
    )
    images = page_images(tmp_path)

    def renderer(*, pdf_page, crop, dpi, source_page=None):
        target = tmp_path / f"crop-{pdf_page:02d}.png"
        target.write_bytes(b"crop")
        return {"pdf_page": pdf_page, "path": target, "image_id": f"crop-{pdf_page:02d}"}

    reader = Reader(ReaderConfig.from_mapping({"model": {"name": "m"}}), client=fake)
    result = reader.read("demo", images, renderer=renderer)
    report = result.report

    # The fields the patch sets are merged ...
    assert report["claims"][0]["text"] == "The method has two stages, verified."
    assert report["method"]["nodes"][0]["operation"] == "Transform, then normalise"
    assert report["method"]["edges"][1]["confirmed"] is True
    # ... and everything the patch does not mention is left exactly as it was.
    assert report["title"] == candidate["title"]
    assert report["evidence"] == candidate["evidence"]
    assert report["experiments"] == candidate["experiments"]
    assert report["method"]["steps"] == candidate["method"]["steps"]
    assert report["claims"][0]["kind"] == "author_claim"
    assert report["unresolved_items"] == [
        "Still unknown: the exact sample size",
        "补看后仍未解决：PDF 第 2 页（r）",
    ], report["unresolved_items"]
    # The candidate itself is never mutated.
    assert candidate["claims"][0]["text"] == "The method has two stages."
    assert candidate["method"]["edges"][1]["confirmed"] is False


def test_a_patch_entry_that_cannot_be_applied_is_refused_not_guessed(tmp_path: Path) -> None:
    """A patch that cites an image this round never sent must not be merged."""

    candidate = sample_report()
    candidate["visual_requests"] = [{"pdf_page": 2, "reason": "r", "crop": None}]
    fake = FakeClient(
        [
            candidate,
            empty_patch(
                evidence_updates=[
                    {
                        "id": "E9",
                        "source_type": "image",
                        "pdf_page": 2,
                        "source_id": "crop-99",  # never rendered in this round
                        "section": "4 Experiments",
                        "figure_or_table": "Table 4",
                        "locator": "row 3",
                        "quote": "97%",
                    }
                ],
                experiment_updates=[{"index": 7, "conclusion": "out of range"}],
            ),
        ]
    )
    images = page_images(tmp_path)

    def renderer(*, pdf_page, crop, dpi, source_page=None):
        target = tmp_path / f"crop-{pdf_page:02d}.png"
        target.write_bytes(b"crop")
        return {"pdf_page": pdf_page, "path": target, "image_id": f"crop-{pdf_page:02d}"}

    reader = Reader(ReaderConfig.from_mapping({"model": {"name": "m"}}), client=fake)
    result = reader.read("demo", images, renderer=renderer)

    assert "E9" not in [item["id"] for item in result.report["evidence"]]
    assert any(item.startswith("补看补丁未应用：") for item in result.unresolved_items)
    assert len(result.report["experiments"]) == 1


def test_a_visual_request_renders_the_page_and_crop_it_asked_for(tmp_path: Path) -> None:
    """The request names a page and an area; the follow-up is cut from both."""

    requested = {"x0": 0.2, "y0": 0.3, "x1": 0.8, "y1": 0.6}
    candidate = sample_report()
    candidate["visual_requests"] = [
        {"pdf_page": 2, "reason": "Table 1 digits are too small", "crop": requested}
    ]
    fake = FakeClient([candidate, empty_patch()])
    images = page_images(tmp_path)
    seen: list[dict] = []

    def renderer(*, pdf_page, crop, dpi, source_page=None):
        seen.append({"pdf_page": pdf_page, "crop": crop, "dpi": dpi})
        target = tmp_path / f"crop-{pdf_page:02d}.png"
        target.write_bytes(b"crop")
        return {"pdf_page": pdf_page, "path": target, "image_id": f"crop-{pdf_page:02d}"}

    config = ReaderConfig.from_mapping({"model": {"name": "m"}, "refinement": {"crop_dpi": 300}})
    reader = Reader(config, client=fake)
    result = reader.read("probe", images, renderer=renderer)

    assert seen == [{"pdf_page": 2, "crop": requested, "dpi": 300}]
    assert [item["image_id"] for item in result.supplemental_images] == ["crop-02"]
    assert result.supplemental_images[0]["pdf_page"] == 2
    # The page it was cut from, the crop itself and the reason all reach the
    # follow-up call, each labelled with the image identity it names.
    second = fake.calls[1]["messages"][1]["content"]
    labels = [
        item["text"]
        for item in second
        if item.get("type") == "text" and item["text"].startswith("PDF_PAGE")
    ]
    assert labels == ["PDF_PAGE=2 IMAGE_ID=page-002", "PDF_PAGE=2 IMAGE_ID=crop-02"]
    assert sum(item["type"] == "image_url" for item in second) == 2
    assert "Table 1 digits are too small" in second[0]["text"]
    # Nothing outside the requested page is re-sent.
    assert "PDF_PAGE=1" not in second[0]["text"]


# --- text-first input ----------------------------------------------------


def test_the_main_call_sends_the_page_marked_text_and_no_image(tmp_path: Path) -> None:
    text = "--- PDF_PAGE=1 ---\nAbstract text.\n\n--- PDF_PAGE=2 ---\nMethod text."
    fake = FakeClient([sample_report()])
    reader = Reader(ReaderConfig.from_mapping({"model": {"name": "m"}}), client=fake)
    reader.read("demo", page_images(tmp_path), text=text)

    content = fake.calls[0]["messages"][1]["content"]
    assert len(content) == 1, "a text-first call must not carry a second part"
    assert content[0]["type"] == "text"
    assert "--- PDF_PAGE=1 ---\nAbstract text." in content[0]["text"]
    assert "--- PDF_PAGE=2 ---\nMethod text." in content[0]["text"]
    assert '"diagram_edge"' not in content[0]["text"], (
        "the main schema must not expose diagram-only label fields"
    )
    assert not any(item["type"] == "image_url" for item in content)


def test_the_full_page_images_mode_ignores_the_text_it_is_handed(tmp_path: Path) -> None:
    """The A/B baseline keeps uploading every page even when text is supplied."""

    config = ReaderConfig.from_mapping(
        {"model": {"name": "m"}, "input": {"mode": "full_page_images"}}
    )
    fake = FakeClient([sample_report()])
    reader = Reader(config, client=fake)
    reader.read("demo", page_images(tmp_path), text="--- PDF_PAGE=1 ---\nignored")

    content = fake.calls[0]["messages"][1]["content"]
    assert sum(item["type"] == "image_url" for item in content) == 3
    assert "ignored" not in content[0]["text"]


def test_an_unknown_input_mode_is_refused() -> None:
    with pytest.raises(Exception):
        ReaderConfig.from_mapping({"model": {"name": "m"}, "input": {"mode": "docling"}})


# --- the diagram stage ---------------------------------------------------


def test_the_diagram_stage_reuses_the_reading_prefix(tmp_path: Path) -> None:
    """The new call replays the reading prefix so the endpoint can cache it."""

    text = "--- PDF_PAGE=1 ---\nAbstract text."
    fake = FakeClient([sample_report(), diagram_payload()])
    reader = Reader(ReaderConfig.from_mapping({"model": {"name": "m"}}), client=fake)
    result = reader.read("demo", page_images(tmp_path), text=text)
    diagrams = reader.generate_diagrams("demo", result.report, title="A Demonstration Paper")

    assert len(diagrams.diagrams) == 2
    reading_prefix = fake.calls[0]["messages"][1]["content"][0]["text"]
    diagram_parts = fake.calls[1]["messages"][1]["content"]
    assert len(diagram_parts) == 1, "the diagram stage must not upload a page image"
    assert diagram_parts[0]["text"].startswith(reading_prefix), "the prefix was not replayed"
    assert "Grounded report data" in diagram_parts[0]["text"]
    assert diagrams.calls[0].call_type == "diagram"
    assert not diagrams.unresolved_items


def test_the_diagram_stage_can_skip_the_prefix_replay(tmp_path: Path) -> None:
    text = "--- PDF_PAGE=1 ---\nAbstract text."
    fake = FakeClient([sample_report(), diagram_payload()])
    config = ReaderConfig.from_mapping(
        {"model": {"name": "m"}, "diagrams": {"reuse_reading_prefix": False}}
    )
    reader = Reader(config, client=fake)
    result = reader.read("demo", page_images(tmp_path), text=text)
    reader.generate_diagrams("demo", result.report)

    diagram_text = fake.calls[1]["messages"][1]["content"][0]["text"]
    assert "Abstract text." not in diagram_text, "the self-contained call resends no paper"
    assert "Grounded report data" in diagram_text


def test_the_diagram_stage_leaves_a_report_it_cannot_draw_alone(tmp_path: Path) -> None:
    """A drawing failure costs the diagram, never the reading."""

    fake = FakeClient([sample_report(), "not json at all", "still not json"])
    config = ReaderConfig.from_mapping(
        {"model": {"name": "m"}, "batch": {"max_retries_per_call": 0}}
    )
    reader = Reader(config, client=fake)
    result = reader.read("demo", page_images(tmp_path), text="--- PDF_PAGE=1 ---\nA.")
    before = json.loads(json.dumps(result.report))
    diagrams = reader.generate_diagrams("demo", result.report)

    assert diagrams.diagrams == []
    assert any(item.startswith("方法图生成失败：") for item in diagrams.unresolved_items)
    assert result.report == before, "a failed stage must not touch the report"


def test_the_diagram_stage_cannot_add_an_ungrounded_edge() -> None:
    from src.reader import sanitize_diagrams

    report = sample_report()
    payload = {
        "diagrams": [
            {
                "id": "d1",
                "title": "invented",
                "type": "flowchart",
                "nodes": [
                    {"id": "Good", "label": "Stage A", "kind": "component", "evidence_ids": ["E1"]},
                    {"id": "Bad", "label": "未知", "kind": "not_a_kind", "evidence_ids": ["E9"]},
                ],
                "edges": [
                    {
                        "from": "Good",
                        "to": "Bad",
                        "label": "hallucinated",
                        "relation": "data_flow",
                        "confirmed": True,
                        "evidence_ids": ["E9"],
                    },
                    {
                        "from": "Bad",
                        "to": "Good",
                        "label": "unconfirmed",
                        "relation": "feedback",
                        "confirmed": False,
                        "evidence_ids": ["E1"],
                    },
                ],
            }
        ]
    }
    diagrams, problems = sanitize_diagrams(payload, report)

    assert len(diagrams) == 1
    evidence = {node["id"]: node["evidence_ids"] for node in diagrams[0]["nodes"]}
    assert evidence == {"Good": ["E1"], "Bad": []}
    assert diagrams[0]["nodes"][1]["kind"] == "component", "an unknown kind is downgraded"
    assert diagrams[0]["edges"] == [], "an ungrounded edge must not be drawn"
    assert len(problems) == 3, problems


def test_the_diagram_stage_respects_its_ceiling() -> None:
    from src.reader import sanitize_diagrams

    report = sample_report()
    node = {"id": "N", "label": "N", "kind": "component", "evidence_ids": ["E1"]}
    payload = {
        "diagrams": [
            {"id": f"d{index}", "title": f"图 {index}", "type": "flowchart", "nodes": [dict(node)], "edges": []}
            for index in range(3)
        ]
    }
    diagrams, problems = sanitize_diagrams(payload, report, max_diagrams=2)

    assert [item["id"] for item in diagrams] == ["d0", "d1"]
    assert any("上限" in item for item in problems)


# --- response handling ---------------------------------------------------


def test_malformed_json_is_repaired_with_a_text_only_call(tmp_path: Path) -> None:
    fake = FakeClient(["{not json at all", sample_report()])
    reader = Reader(ReaderConfig.from_mapping({"model": {"name": "m"}}), client=fake)
    result = reader.read("demo", page_images(tmp_path))

    assert result.report["title"] == sample_report()["title"]
    assert len(fake.calls) == 2
    repair_content = fake.calls[1]["messages"][1]["content"]
    assert all(item.get("type") == "text" for item in repair_content), "repair must not resend pages"
    assert "本次不提供任何页面图片" in repair_content[0]["text"]


def test_diagram_gets_its_own_format_repair_budget(tmp_path: Path) -> None:
    """A main-report repair must not consume the diagram stage's allowance."""

    broken_report = sample_report()
    del broken_report["experiments"][0]["models"]
    broken_diagram = json.loads(json.dumps(diagram_payload()))
    del broken_diagram["diagrams"][0]["edges"][0]["evidence_ids"]
    fake = FakeClient(
        [broken_report, sample_report(), broken_diagram, diagram_payload()]
    )
    reader = Reader(ReaderConfig.from_mapping({"model": {"name": "m"}}), client=fake)

    reading = reader.read("demo", page_images(tmp_path), text="--- PDF_PAGE=1 ---\ntext")
    diagrams = reader.generate_diagrams("demo", reading.report)

    assert len(fake.calls) == 4
    assert reading.calls[0].format_repaired is True
    assert diagrams.calls[0].format_repaired is True
    assert diagrams.diagrams


def test_schema_violation_is_repaired_without_images(tmp_path: Path) -> None:
    # A missing required object cannot be invented locally, so this still
    # spends the text-only repair round.  Extra forbidden keys no longer do.
    broken = sample_report()
    del broken["method"]
    fake = FakeClient([broken, sample_report()])
    reader = Reader(ReaderConfig.from_mapping({"model": {"name": "m"}}), client=fake)
    result = reader.read("demo", page_images(tmp_path))

    assert "method" in result.report
    assert len(fake.calls) == 2
    assert all(item.get("type") == "text" for item in fake.calls[1]["messages"][1]["content"])


def test_repair_budget_is_respected(tmp_path: Path) -> None:
    broken = sample_report()
    del broken["method"]
    still_broken = sample_report()
    del still_broken["claims"]
    fake = FakeClient([broken, still_broken, sample_report()])
    config = ReaderConfig.from_mapping(
        {"model": {"name": "m"}, "refinement": {"max_format_repairs": 1}}
    )
    reader = Reader(config, client=fake)
    with pytest.raises(ReaderResponseError, match="still violates the schema"):
        reader.read("demo", page_images(tmp_path))

    # One repair attempt, then fail before visual refinement or diagram calls;
    # the model is not asked again and an invalid report is never propagated.
    assert len(fake.calls) == 2


def test_truncated_output_is_not_retried_and_not_repaired(tmp_path: Path) -> None:
    fake = FakeClient([sample_report()], finish_reason="length")
    reader = Reader(ReaderConfig.from_mapping({"model": {"name": "m"}}), client=fake)
    with pytest.raises(ReaderResponseError):
        reader.read("demo", page_images(tmp_path))
    assert len(fake.calls) == 1, "a truncated response is missing content, not misformatted"


def test_only_payloads_with_no_report_content_are_substance_free() -> None:
    from src.reader import _carries_no_report

    schema = json.loads((PROJECT / "schemas" / "report.schema.json").read_text())
    empty = (
        "",
        "   ",
        "[]",
        "[{}]",
        "null",
        "12",
        '"a string"',
        "{}",
        '{"report": {}}',
        '{"a": 1}',
        "```json\n[]\n```",
    )
    for payload in empty:
        assert _carries_no_report(payload, schema) is True, payload
    # Real content, however malformed, stays repairable.
    content = ("{not json at all", '{"title": "x"', "Here is the report: {\"schema_version\": \"1\"}")
    for payload in content:
        assert _carries_no_report(payload, schema) is False, payload


def test_a_substance_free_response_is_not_repaired(tmp_path: Path) -> None:
    """`[]` holds nothing: a text-only repair could only invent a report."""

    for payload in ("[]", ""):
        fake = FakeClient([payload])
        reader = Reader(ReaderConfig.from_mapping({"model": {"name": "m"}}), client=fake)
        with pytest.raises(ReaderResponseError) as excinfo:
            reader.read("demo", page_images(tmp_path))

        assert len(fake.calls) == 1, f"a repair round was spent on {payload!r}"
        assert excinfo.value.repairable is False
        assert excinfo.value.retryable_with_images is True


def test_an_object_with_no_report_fields_is_not_repaired_either(tmp_path: Path) -> None:
    """`{"a": 1}` parses, so it reaches the schema repair — which must skip it."""

    fake = FakeClient([{"a": 1}])
    reader = Reader(ReaderConfig.from_mapping({"model": {"name": "m"}}), client=fake)
    with pytest.raises(ReaderResponseError) as excinfo:
        reader.read("demo", page_images(tmp_path))

    assert len(fake.calls) == 1, "the schema repair must not write a report from nothing"
    assert excinfo.value.retryable_with_images is True


def test_output_limit_and_thinking_are_sent(tmp_path: Path) -> None:
    fake = FakeClient([sample_report()])
    config = ReaderConfig.from_mapping(
        {"model": {"name": "m", "max_output_tokens": 4096, "thinking": True}}
    )
    Reader(config, client=fake).read("demo", page_images(tmp_path))
    assert fake.calls[0]["max_tokens"] == 4096
    assert fake.calls[0]["extra_body"] == {
        "enable_thinking": True,
        "thinking_budget": 16384,
    }


def test_thinking_budget_can_be_set_per_stage(tmp_path: Path) -> None:
    broken = sample_report()
    del broken["method"]
    fake = FakeClient([broken, sample_report(), diagram_payload()])
    config = ReaderConfig.from_mapping(
        {
            "model": {
                "name": "m",
                "thinking": True,
                "thinking_budget": {
                    "read": 12000,
                    "format_repair": 2000,
                    "diagram": 3000,
                },
            }
        }
    )
    reader = Reader(config, client=fake)
    result = reader.read("demo", page_images(tmp_path))
    reader.generate_diagrams("demo", result.report)

    assert [call["extra_body"]["thinking_budget"] for call in fake.calls] == [
        12000,
        2000,
        3000,
    ]


def test_thinking_budget_rejects_an_unknown_stage() -> None:
    with pytest.raises(ReaderConfigError, match="unknown stages: typo"):
        ReaderConfig.from_mapping(
            {"model": {"name": "m", "thinking_budget": {"typo": 1000}}}
        )


def test_unusable_visual_requests_are_recorded_not_repaired() -> None:
    from src.reader import _sanitize_visual_requests

    report = sample_report()
    report["visual_requests"] = [
        {"pdf_page": 2, "reason": "keep me", "crop": None},
        {"pdf_page": None, "reason": "null page", "crop": None},
        {"pdf_page": 2.5, "reason": "float page", "crop": None},
        {"pdf_page": 3, "reason": "no area", "crop": {"x0": 0.5, "y0": 0.5, "x1": 0.5, "y1": 0.9}},
    ]
    _sanitize_visual_requests(report)
    assert [item["reason"] for item in report["visual_requests"]] == ["keep me"]
    messages = report["unresolved_items"]
    assert len(messages) == 3
    assert any("None" in item for item in messages)
    assert any("2.5" in item for item in messages)
    assert any("无有效面积" in item for item in messages)
    # Sanitizing a report that is otherwise fine makes it schema-valid.
    schema = json.loads((PROJECT / "schemas" / "report.schema.json").read_text())
    assert schema_errors(report, schema) == []


def test_dangling_cross_references_are_omitted_and_recorded() -> None:
    report = sample_report()
    report["claims"][0]["evidence_ids"] = ["MISSING"]
    report["method"]["nodes"][0]["evidence_ids"] = ["E1", "MISSING"]
    report["method"]["edges"].append(
        {
            "from": "A",
            "to": "missing",
            "relation": "data_flow",
            "confirmed": True,
            "evidence_ids": ["E1"],
        }
    )

    reconcile_cross_references(report)
    once = json.dumps(report, ensure_ascii=False, sort_keys=True)
    reconcile_cross_references(report)

    assert json.dumps(report, ensure_ascii=False, sort_keys=True) == once
    assert report["claims"][0]["evidence_ids"] == []
    assert report["method"]["nodes"][0]["evidence_ids"] == ["E1"]
    assert all(edge["to"] != "missing" for edge in report["method"]["edges"])
    assert any(item.startswith("证据引用已移除：") for item in report["unresolved_items"])
    assert any(item.startswith("未落地的方法项已省略：") for item in report["unresolved_items"])
    assert validate_report(report, pages=pages_manifest()).valid


def test_schema_errors_helper_returns_readable_paths() -> None:
    schema = json.loads((PROJECT / "schemas" / "report.schema.json").read_text())
    broken = sample_report()
    del broken["claims"]
    errors = schema_errors(broken, schema)
    assert any("'claims' is a required property" in error for error in errors)
    assert errors and errors[0].startswith("$")


def test_schema_shape_normalization_fixes_rfc_audit_footguns() -> None:
    """Method edges must not carry diagram labels; guide items need target."""

    schema = json.loads((PROJECT / "schemas" / "report.schema.json").read_text())
    report = sample_report()
    report["method"]["edges"][0]["label"] = "Provides Hierarchical Semantic Index"
    del report["reading_guide"]["open_questions"][0]["target"]
    del report["reading_guide"]["related_directions"][0]["target"]

    fixes = normalize_report_schema_shape(report, schema)

    assert "label" not in report["method"]["edges"][0]
    assert report["reading_guide"]["open_questions"][0]["target"] == ""
    assert report["reading_guide"]["related_directions"][0]["target"] == ""
    assert any("label" in item for item in fixes)
    assert any("target" in item for item in fixes)
    assert schema_errors(report, schema) == []
    # Idempotent: a second pass must not invent new notes or change shape.
    assert normalize_report_schema_shape(report, schema) == []


def test_schema_shape_normalization_migrates_components_to_nodes() -> None:
    """The RFCAudit alias must be preserved instead of deleted."""

    schema = json.loads((PROJECT / "schemas" / "report.schema.json").read_text())
    report = sample_report()
    expected_nodes = report["method"].pop("nodes")
    report["method"]["components"] = expected_nodes

    fixes = normalize_report_schema_shape(report, schema)

    assert "components" not in report["method"]
    assert report["method"]["nodes"] == expected_nodes
    assert any("components" in item and "nodes" in item for item in fixes)
    assert schema_errors(report, schema) == []


def test_schema_requires_grounding_for_method_graph_items() -> None:
    schema = json.loads((PROJECT / "schemas" / "report.schema.json").read_text())
    report = sample_report()
    report["method"]["nodes"][0]["evidence_ids"] = []
    report["method"]["edges"][0]["evidence_ids"] = []
    report["method"]["steps"][0]["evidence_ids"] = []

    errors = schema_errors(report, schema)

    assert any("method/nodes/0/evidence_ids" in item for item in errors)
    assert any("method/edges/0/evidence_ids" in item for item in errors)
    assert any("method/steps/0/evidence_ids" in item for item in errors)


def test_schema_repair_sees_method_items_before_semantic_cleanup(tmp_path: Path) -> None:
    """Components, edges and steps must survive long enough to be repaired."""

    broken = sample_report()
    broken["method"]["components"] = broken["method"].pop("nodes")
    for step in broken["method"]["steps"]:
        step.pop("evidence_ids")
    fake = FakeClient([broken, sample_report()])
    reader = Reader(ReaderConfig.from_mapping({"model": {"name": "m"}}), client=fake)

    result = reader.read("demo", page_images(tmp_path))

    repair_text = fake.calls[1]["messages"][1]["content"][0]["text"]
    assert '"nodes"' in repair_text
    assert '"edges"' in repair_text
    assert '"steps"' in repair_text
    assert '"components"' not in repair_text
    assert len(result.report["method"]["nodes"]) == len(sample_report()["method"]["nodes"])
    assert len(result.report["method"]["edges"]) == len(sample_report()["method"]["edges"])
    assert len(result.report["method"]["steps"]) == len(sample_report()["method"]["steps"])


def test_schema_shape_normalization_does_not_invent_missing_objects() -> None:
    schema = json.loads((PROJECT / "schemas" / "report.schema.json").read_text())
    report = sample_report()
    del report["method"]

    fixes = normalize_report_schema_shape(report, schema)

    assert "method" not in report, "missing required objects must stay missing"
    assert fixes == []
    assert schema_errors(report, schema), "still invalid, so a repair round is still needed"


def test_local_shape_fix_avoids_a_repair_round(tmp_path: Path) -> None:
    """A report that only needs mechanical coercions must not spend a repair call."""

    schema_broken = sample_report()
    schema_broken["method"]["edges"][0]["label"] = "flow"
    del schema_broken["reading_guide"]["open_questions"][0]["target"]
    fake = FakeClient([schema_broken])
    reader = Reader(ReaderConfig.from_mapping({"model": {"name": "m"}}), client=fake)

    result = reader.read("demo", page_images(tmp_path))

    assert len(fake.calls) == 1, "local normalize must clear the schema errors"
    assert result.calls[0].format_repaired is False
    assert "label" not in result.report["method"]["edges"][0]
    assert result.report["reading_guide"]["open_questions"][0]["target"] == ""


def test_a_repair_is_accepted_after_local_shape_normalization(tmp_path: Path) -> None:
    """Repair output that is one mechanical fix away from valid must be kept."""

    broken = sample_report()
    del broken["method"]
    almost = sample_report()
    almost["method"]["edges"][0]["label"] = "still has a diagram label"
    del almost["reading_guide"]["reproduction_notes"][0]["target"]
    fake = FakeClient([broken, almost])
    reader = Reader(ReaderConfig.from_mapping({"model": {"name": "m"}}), client=fake)

    result = reader.read("demo", page_images(tmp_path))

    assert len(fake.calls) == 2
    assert result.calls[0].format_repaired is True
    assert "method" in result.report
    assert "label" not in result.report["method"]["edges"][0]
    assert result.report["reading_guide"]["reproduction_notes"][0]["target"] == ""
    assert schema_errors(
        result.report, json.loads((PROJECT / "schemas" / "report.schema.json").read_text())
    ) == []


def test_schema_repair_that_drops_method_content_is_rejected(tmp_path: Path) -> None:
    """Schema validity must not turn a non-empty method graph into a smaller one."""

    broken = sample_report()
    del broken["experiments"][0]["models"]
    regressed = sample_report()
    regressed["method"]["nodes"] = regressed["method"]["nodes"][:1]
    fake = FakeClient([broken, regressed])
    reader = Reader(ReaderConfig.from_mapping({"model": {"name": "m"}}), client=fake)

    with pytest.raises(ReaderResponseError, match="still violates the schema"):
        reader.read("demo", page_images(tmp_path))

    assert len(fake.calls) == 2, "a rejected repair must not continue to later stages"


# --- batch behaviour -----------------------------------------------------


def test_response_errors_are_not_retried_by_the_batch() -> None:
    assert _retryable(ReaderResponseError("Model response is not valid JSON")) is False
    assert _retryable(TimeoutError("read timed out")) is True
    assert _retryable(RuntimeError("429 rate limited")) is True


def test_a_substance_free_response_is_retried_by_the_batch() -> None:
    empty = ReaderResponseError("Model response JSON holds no report content")
    empty.retryable_with_images = True
    assert _retryable(empty) is True, "nothing can be repaired, so a fresh read is the only way"


def test_request_count_uses_every_logical_call() -> None:
    assert _requests_made(ReadResult(report={}, calls=[object(), object()])) == 2
    assert _requests_made(ReadResult(report={}, calls=[object()], requests_made=3)) == 3
    assert _requests_made({"requests_made": 3}) == 3
    assert _requests_made(None) == 1


def test_a_failed_reader_exposes_the_number_of_sent_requests(tmp_path: Path) -> None:
    class FailingClient:
        class Completions:
            def create(self, **kwargs):
                raise TimeoutError("read timed out")

        chat = type("Chat", (), {"completions": Completions()})()

    reader = Reader(
        ReaderConfig.from_mapping(
            {"model": {"name": "m"}, "batch": {"max_retries_per_call": 0}}
        ),
        client=FailingClient(),
    )
    with pytest.raises(ReaderAPIError) as excinfo:
        reader.read("demo", page_images(tmp_path))

    assert excinfo.value.requests_made == 1


def test_pipeline_source_changes_the_run_fingerprint(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source = project / "src"
    source.mkdir(parents=True)
    config_path = project / "config.yaml"
    config_path.write_text("model: {}\n", encoding="utf-8")
    module = source / "reader.py"
    module.write_text("VERSION = 1\n", encoding="utf-8")

    before = _config_fingerprint({"model": {}}, config_path, runtime_root=project)
    module.write_text("VERSION = 2\n", encoding="utf-8")
    after = _config_fingerprint({"model": {}}, config_path, runtime_root=project)

    assert before != after


def test_an_interrupted_job_is_not_left_running(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "paper.pdf", pages=1)
    options = batch_options(tmp_path)
    stop_event = threading.Event()
    stop_event.set()

    result = process_one(pdf, options, {}, "test-fingerprint", stop_event)

    assert result.status == "interrupted"
    job = json.loads(result.job_path.read_text(encoding="utf-8"))
    assert job["status"] == "interrupted"
    assert job["error"] == "interrupted by user"


def test_coverage_text_describes_the_input(tmp_path: Path) -> None:
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    (paper_dir / "metadata.json").write_text(
        json.dumps(
            {
                "total_pages": 3,
                "input_coverage": {"pdf_pages": [1, 2, 3], "complete": True},
            }
        ),
        encoding="utf-8",
    )
    text = _coverage_text(paper_dir)
    assert "全篇 3 页" in text and "未纳入 PDF 之外的补充材料" in text


def test_render_options_follow_the_config() -> None:
    from src.batch import _render_options

    assert _render_options({}) == {
        "include_mermaid": True,
        "include_method_steps": True,
        "include_evidence_index": True,
    }
    options = _render_options({"output": {"include_mermaid": False, "include_evidence_index": False}})
    assert options["include_mermaid"] is False
    assert options["include_evidence_index"] is False


def test_unsupported_config_fails_loudly() -> None:
    from src.batch import _check_unsupported_config

    _check_unsupported_config({})
    with pytest.raises(Exception):
        _check_unsupported_config({"output": {"language": "en-US"}})
    with pytest.raises(Exception):
        _check_unsupported_config({"verification": {"automated_verifier": True}})
    # The input mode decides whether every page image is uploaded, so an
    # unknown mode or a combination that cannot be honoured stops the run
    # instead of being read as a default.
    with pytest.raises(Exception):
        _check_unsupported_config({"input": {"mode": "layout_parser"}})
    with pytest.raises(Exception):
        _check_unsupported_config({"input": {"include_all_pdf_pages": "yes"}})
    with pytest.raises(Exception):
        # Uploading every page under the text-first mode is a contradiction.
        _check_unsupported_config({"input": {"include_all_pdf_pages": True}})
    with pytest.raises(Exception):
        # Text-first without text has no input at all.
        _check_unsupported_config({"input": {"extract_text": False}})
    # The A/B baseline is the mirror image: it must send every page.
    _check_unsupported_config(
        {"input": {"mode": "full_page_images", "include_all_pdf_pages": True}}
    )
    with pytest.raises(Exception):
        _check_unsupported_config({"input": {"mode": "full_page_images"}})


def test_evidence_index_and_steps_can_be_omitted(tmp_path: Path) -> None:
    pages_path = tmp_path / "pages.json"
    pages_path.write_text(json.dumps(pages_manifest()), encoding="utf-8")
    for page in pages_manifest()["pages"]:
        image = tmp_path / page["path"]
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"placeholder")
    output = tmp_path / "final.md"
    render_report(
        sample_report(),
        output,
        pages_json=pages_path,
        include_evidence_index=False,
        include_method_steps=False,
    )
    text = output.read_text(encoding="utf-8")
    assert "## Evidence Index" not in text
    assert "### 方法步骤" not in text
    assert "```mermaid" in text


# --- end to end ----------------------------------------------------------


def test_batch_end_to_end_writes_status_coverage_and_crop_links(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "probe.pdf", pages=2)

    crop = {"x0": 0.1, "y0": 0.1, "x1": 0.6, "y1": 0.5}
    candidate = sample_report()
    candidate["paper_id"] = "probe"
    candidate["visual_requests"] = [
        {"pdf_page": 1, "reason": "digits unclear", "crop": crop}
    ]
    # The refinement round answers with a patch.  A paper can only be "done"
    # when every relation is evidence-backed, so the patch confirms the one
    # relation the first round had to leave out.
    refined = empty_patch(
        evidence_updates=[
            {
                "id": "E3",
                "source_type": "image",
                "pdf_page": 1,
                "source_id": "crop-01",
                "section": "4 Experiments",
                "figure_or_table": "Table 1",
                "locator": "row 2",
                "quote": "91% accuracy",
            }
        ],
        experiment_updates=[
            {"index": 0, "main_results": ["91% accuracy"], "conclusion": "The crop was readable."}
        ],
        method_updates=[{"target": "edge", "from": "B", "to": "A", "confirmed": True}],
        resolved_visual_requests=[{"pdf_page": 1, "crop": crop}],
    )

    fake = FakeClient([candidate, refined, diagram_payload()])
    original = Reader._make_client
    Reader._make_client = lambda self: fake  # type: ignore[assignment]
    try:
        results = run_batch(pdf, batch_options(tmp_path))
    finally:
        Reader._make_client = original  # type: ignore[assignment]

    assert [result.status for result in results] == ["done"], results[0].error
    output = results[0].output_path
    assert output is not None and output.exists()
    text = output.read_text(encoding="utf-8")

    # A report that cites a crop no longer fails validation.
    assert "crop-01" in text
    assert "91% accuracy" in text
    # The diagrams came from the separate stage and both are drawn.
    assert text.count("```mermaid") == 2
    assert "#### 图 1：两阶段总览" in text
    # The header states the real status and the input coverage.
    assert "- **Status:** `done`" in text
    assert "全篇 2 页" in text
    # Nothing is pending, and the pending list says so in the opening block.
    assert text.index("## 待核查事项") < text.index("## 摘要")
    assert "## 待核查事项\n\n无。" in text
    # Crop and page links point at files that exist.
    links = re.findall(r"\]\(([^)]+)\)", text)
    assert links, "the report must link to the evidence it cites"
    for target in links:
        assert (output.parent / target).resolve().exists(), target
    # The reading call, the refinement patch and the diagram stage are all
    # accounted for in the job record.
    job = json.loads((results[0].job_path).read_text(encoding="utf-8"))
    assert job["requests"] == 3, job
    assert job["output_coverage"].startswith("全篇 2 页")


def test_the_batch_reads_the_text_and_never_uploads_the_pages(tmp_path: Path) -> None:
    """The main call is text-first; the pages stay local until a crop is asked for."""

    pdf = make_pdf(tmp_path / "probe.pdf", pages=2)
    report = sample_report()
    report["paper_id"] = "probe"
    fake = FakeClient([report, diagram_payload()])
    original = Reader._make_client
    Reader._make_client = lambda self: fake  # type: ignore[assignment]
    try:
        results = run_batch(pdf, batch_options(tmp_path))
    finally:
        Reader._make_client = original  # type: ignore[assignment]

    assert results[0].status != "failed", results[0].error
    first = fake.calls[0]["messages"][1]["content"]
    assert all(item["type"] == "text" for item in first), "a page image was uploaded"
    prompt = first[0]["text"]
    assert "--- PDF_PAGE=1 ---" in prompt and "--- PDF_PAGE=2 ---" in prompt
    assert "Page 1" in prompt and "Page 2" in prompt, "the extracted text is missing"
    # The pages are still rendered: a crop and every evidence link need them.
    workspace = tmp_path / "workspace" / "probe"
    assert len(list(workspace.glob("pages/page-*.png"))) == 2


def test_batch_marks_the_paper_needs_review_when_a_crop_cannot_be_rendered(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "probe.pdf", pages=2)
    candidate = sample_report()
    candidate["paper_id"] = "probe"
    # Nothing in the report asks for a crop that cannot be produced, so the
    # unresolved item has to come from the refinement bookkeeping itself.
    candidate["visual_requests"] = [{"pdf_page": 99, "reason": "hallucinated page", "crop": None}]
    fake = FakeClient([candidate, sample_report()])
    original = Reader._make_client
    Reader._make_client = lambda self: fake  # type: ignore[assignment]
    try:
        results = run_batch(pdf, batch_options(tmp_path))
    finally:
        Reader._make_client = original  # type: ignore[assignment]

    assert results[0].status == "needs_review"
    assert results[0].unresolved_items
    text = results[0].output_path.read_text(encoding="utf-8")
    assert "待核查" in text


def test_unconfirmed_relations_are_listed_where_the_status_is_derived(tmp_path: Path) -> None:
    """A needs_review header must never sit above an empty pending list."""

    pdf = make_pdf(tmp_path / "probe.pdf", pages=2)
    report = sample_report()
    report["paper_id"] = "probe"
    assert report["method"]["edges"][1]["confirmed"] is False
    fake = FakeClient([report])
    original = Reader._make_client
    Reader._make_client = lambda self: fake  # type: ignore[assignment]
    try:
        results = run_batch(pdf, batch_options(tmp_path))
    finally:
        Reader._make_client = original  # type: ignore[assignment]

    result = results[0]
    assert result.status == "needs_review"
    # report.json, the delivered Markdown and the returned status agree.
    stored = json.loads(result.report_path.read_text(encoding="utf-8"))
    listed = [item for item in stored["unresolved_items"] if item.startswith("未确认的方法关系：")]
    assert len(listed) == 1, stored["unresolved_items"]
    assert "Stage B" in listed[0] and "dependency" in listed[0]

    text = result.output_path.read_text(encoding="utf-8")
    pending = text.split("## 待核查事项")[1].split("##")[0]
    assert listed[0] in pending
    assert "无。" not in pending
    assert listed[0] in result.unresolved_items


def test_derived_items_do_not_linger_after_refinement(tmp_path: Path) -> None:
    """A stale derived item must not outlive the state it described."""

    candidate = sample_report()
    candidate["paper_id"] = "probe"
    request = {"pdf_page": 2, "reason": "edge unclear", "crop": None}
    candidate["visual_requests"] = [request]
    # A patch that confirms the relation and closes the request, and that
    # echoes back the stale derived items the first round wrote.
    refined = empty_patch(
        method_updates=[{"target": "edge", "from": "B", "to": "A", "confirmed": True}],
        resolved_visual_requests=[dict(request)],
        unresolved_items_add=[
            "未确认的方法关系：Stage B → Stage A（dependency）未获证据确认，已从方法图中省略",
            "补看请求未能执行：PDF 第 2 页（edge unclear）",
        ],
    )
    fake = FakeClient([candidate, refined])
    images = page_images(tmp_path)

    def renderer(*, pdf_page, crop, dpi, source_page=None):
        target = tmp_path / f"crop-{pdf_page:02d}.png"
        target.write_bytes(b"crop")
        return {"pdf_page": pdf_page, "path": target, "image_id": f"crop-{pdf_page:02d}"}

    reader = Reader(ReaderConfig.from_mapping({"model": {"name": "m"}}), client=fake)
    result = reader.read("demo", images, renderer=renderer)

    assert result.report["method"]["edges"][1]["confirmed"] is True
    assert result.report["visual_requests"] == []
    assert result.unresolved_items == []
    assert result.report["unresolved_items"] == []


def test_a_resolved_request_that_was_never_rendered_is_refused(tmp_path: Path) -> None:
    """Only the requests this round looked at may be closed by its patch."""

    candidate = sample_report()
    candidate["paper_id"] = "probe"
    candidate["visual_requests"] = [
        {"pdf_page": 2, "reason": "r", "crop": None},
        {"pdf_page": 3, "reason": "never sent", "crop": None},
    ]
    patch = empty_patch(
        resolved_visual_requests=[
            {"pdf_page": 2, "crop": None},
            {"pdf_page": 3, "crop": None},
        ]
    )
    fake = FakeClient([candidate, patch])
    images = page_images(tmp_path)

    def renderer(*, pdf_page, crop, dpi, source_page=None):
        if pdf_page == 3:
            raise RuntimeError("renderer refused page 3")
        target = tmp_path / f"crop-{pdf_page:02d}.png"
        target.write_bytes(b"crop")
        return {"pdf_page": pdf_page, "path": target, "image_id": f"crop-{pdf_page:02d}"}

    reader = Reader(ReaderConfig.from_mapping({"model": {"name": "m"}}), client=fake)
    result = reader.read("probe", images, renderer=renderer)

    assert any(
        item.startswith("补看补丁未应用：") and "第 3 页" in item for item in result.unresolved_items
    ), result.unresolved_items
    assert [item["pdf_page"] for item in result.report["visual_requests"]] == [3]
    assert any(item.startswith("补看请求未能执行：PDF 第 3 页") for item in result.unresolved_items)


def test_a_fulfilled_request_is_not_reported_as_pending(tmp_path: Path) -> None:
    """A crop that was just sent is not a request that failed to run.

    The patch left the request in the candidate, so the round looked at it and
    the question survived the look.  Read as pending, that would turn a
    successful refinement round into a phantom "未能执行" item.
    """

    request = {
        "pdf_page": 2,
        "reason": "Table 4 numbers are too small to read",
        "crop": {"x0": 0.1, "y0": 0.2, "x1": 0.9, "y1": 0.8},
    }
    candidate = sample_report()
    candidate["paper_id"] = "probe"
    candidate["visual_requests"] = [request]
    fake = FakeClient([candidate, empty_patch()])
    images = page_images(tmp_path)

    def renderer(*, pdf_page, crop, dpi, source_page=None):
        target = tmp_path / f"crop-{pdf_page:02d}.png"
        target.write_bytes(b"crop")
        return {"pdf_page": pdf_page, "path": target, "image_id": f"crop-{pdf_page:02d}"}

    reader = Reader(ReaderConfig.from_mapping({"model": {"name": "m"}}), client=fake)
    result = reader.read("probe", images, renderer=renderer)

    assert len(fake.calls) == 2, "the follow-up call must have been made"
    assert result.supplemental_images, "the crop must have been rendered and sent"
    assert not [
        item for item in result.unresolved_items if item.startswith("补看请求未能执行：")
    ], result.unresolved_items
    assert [item for item in result.unresolved_items if item.startswith("补看后仍未解决：")] == [
        "补看后仍未解决：PDF 第 2 页（Table 4 numbers are too small to read）"
    ]


def test_a_request_that_never_got_an_image_is_still_pending(tmp_path: Path) -> None:
    """The two bookkeeping outcomes must stay distinguishable in one round."""

    candidate = sample_report()
    candidate["paper_id"] = "probe"
    candidate["visual_requests"] = [
        {"pdf_page": 2, "reason": "Table 4 numbers are too small", "crop": None},
        {"pdf_page": 3, "reason": "Figure 2 legend needs a closer look", "crop": None},
    ]
    fake = FakeClient([candidate, empty_patch()])
    images = page_images(tmp_path)

    def renderer(*, pdf_page, crop, dpi, source_page=None):
        if pdf_page == 3:
            raise RuntimeError("page 3 could not be rendered")
        target = tmp_path / f"crop-{pdf_page:02d}.png"
        target.write_bytes(b"crop")
        return {"pdf_page": pdf_page, "path": target, "image_id": f"crop-{pdf_page:02d}"}

    reader = Reader(ReaderConfig.from_mapping({"model": {"name": "m"}}), client=fake)
    result = reader.read("probe", images, renderer=renderer)

    assert [item for item in result.unresolved_items if item.startswith("补看请求未能执行：")] == [
        "补看请求未能执行：PDF 第 3 页（Figure 2 legend needs a closer look）"
    ]
    assert [item for item in result.unresolved_items if item.startswith("补看后仍未解决：")] == [
        "补看后仍未解决：PDF 第 2 页（Table 4 numbers are too small）"
    ]


def test_a_failed_follow_up_keeps_the_candidate_report(tmp_path: Path) -> None:
    """The follow-up is an improvement, not the paper: it must not be fatal."""

    candidate = sample_report()
    candidate["paper_id"] = "probe"
    candidate["visual_requests"] = [
        {"pdf_page": 2, "reason": "Table 4 numbers are too small", "crop": None}
    ]

    class FlakyClient:
        """Answer the first call, then fail every follow-up."""

        def __init__(self, payload: dict) -> None:
            self.payload = payload
            self.calls: list[dict] = []
            outer = self

            class Completions:
                def create(inner, **kwargs):
                    outer.calls.append(kwargs)
                    if len(outer.calls) > 1:
                        raise RuntimeError("endpoint unavailable")
                    return FakeResponse(json.dumps(outer.payload), 1)

            self.chat = type("Chat", (), {"completions": Completions()})()

    client = FlakyClient(candidate)
    images = page_images(tmp_path)

    def renderer(*, pdf_page, crop, dpi, source_page=None):
        target = tmp_path / f"crop-{pdf_page:02d}.png"
        target.write_bytes(b"crop")
        return {"pdf_page": pdf_page, "path": target, "image_id": f"crop-{pdf_page:02d}"}

    config = ReaderConfig.from_mapping(
        {"model": {"name": "m"}, "batch": {"max_retries_per_call": 0}}
    )
    reader = Reader(config, client=client)
    result = reader.read("probe", images, renderer=renderer)

    assert len(client.calls) == 2, "the follow-up must have been attempted"
    assert result.report["title"] == candidate["title"], "the candidate was discarded"
    assert [call.call_type for call in result.calls] == ["read"]
    assert [item for item in result.unresolved_items if item.startswith("补看轮次失败：")], (
        result.unresolved_items
    )
    assert "补看请求未能执行：PDF 第 2 页（Table 4 numbers are too small）" in result.unresolved_items


def test_a_repair_produced_report_says_so(tmp_path: Path) -> None:
    """A report that no image was in front of must be labelled as such."""

    fake = FakeClient(["this is not JSON at all", sample_report()])
    reader = Reader(ReaderConfig.from_mapping({"model": {"name": "m"}}), client=fake)
    result = reader.read("demo", page_images(tmp_path))

    assert len(fake.calls) == 2, "the repair round must have run"
    assert result.calls[0].format_repaired is True
    assert result.calls[0].as_dict()["format_repaired"] is True


def test_a_schema_repair_is_labelled_too(tmp_path: Path) -> None:
    """The schema-repair path restructures content just as the unparsable one does."""

    broken = sample_report()
    del broken["method"]  # a schema violation, not a parse failure
    fake = FakeClient([broken, sample_report()])
    reader = Reader(ReaderConfig.from_mapping({"model": {"name": "m"}}), client=fake)
    result = reader.read("demo", page_images(tmp_path))

    assert len(fake.calls) == 2, "the schema repair must have run"
    assert result.requests_made == 2
    assert result.calls[0].format_repaired is True
    assert "method" in result.report


def test_a_repair_whose_result_is_rejected_leaves_the_label_off(tmp_path: Path) -> None:
    """A rejected repair must fail before downstream paid stages run."""

    broken = sample_report()
    del broken["method"]
    still_broken = sample_report()
    del still_broken["claims"]
    fake = FakeClient([broken, still_broken])
    reader = Reader(ReaderConfig.from_mapping({"model": {"name": "m"}}), client=fake)
    with pytest.raises(ReaderResponseError, match="still violates the schema"):
        reader.read("demo", page_images(tmp_path))

    assert len(fake.calls) == 2


def test_a_clean_report_is_not_labelled_as_repaired(tmp_path: Path) -> None:
    fake = FakeClient([sample_report()])
    reader = Reader(ReaderConfig.from_mapping({"model": {"name": "m"}}), client=fake)
    result = reader.read("demo", page_images(tmp_path))

    assert result.calls[0].format_repaired is False
    assert result.calls[0].as_dict()["format_repaired"] is False


def test_the_repair_provenance_reaches_the_record_and_the_report(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "paper.pdf", pages=3)
    fake = FakeClient(["not JSON", sample_report()])
    original = Reader._make_client
    Reader._make_client = lambda self: fake  # type: ignore[assignment]
    try:
        results = run_batch(pdf, batch_options(tmp_path))
    finally:
        Reader._make_client = original  # type: ignore[assignment]

    result = results[0]
    job = json.loads(result.job_path.read_text(encoding="utf-8"))
    assert job["format_repaired"] is True, job
    text = result.output_path.read_text(encoding="utf-8")
    assert "**Provenance:**" in text, "the delivered report hides its provenance"
    assert "纯文本结构修复" in text


def test_the_batch_retries_a_substance_free_response_with_the_text(tmp_path: Path) -> None:
    """The ProtocolGuard failure: `[]` after a long reasoning pass.

    It must not be repaired (a repair invented a whole report once), and it
    must not end the paper either: the second attempt re-sends the full text.
    """

    pdf = make_pdf(tmp_path / "paper.pdf", pages=2)
    report = sample_report()
    report["paper_id"] = "paper"
    fake = FakeClient(["[]", report, diagram_payload()])
    original = Reader._make_client
    Reader._make_client = lambda self: fake  # type: ignore[assignment]
    try:
        results = run_batch(pdf, batch_options(tmp_path))
    finally:
        Reader._make_client = original  # type: ignore[assignment]

    result = results[0]
    assert result.status in {"done", "needs_review"}, result.error
    assert result.error is None, "a retried-then-successful paper keeps no open error"
    assert len(fake.calls) == 3, "the empty response was not retried"
    for call in fake.calls[:2]:
        parts = call["messages"][1]["content"]
        assert all(item["type"] == "text" for item in parts), (
            "every attempt must carry the full text and no page image"
        )
        assert "--- PDF_PAGE=1 ---" in parts[0]["text"]
    events = _events(result)
    names = [event.get("event") or event.get("kind") for event in events]
    assert "repair_started" not in names, "there was nothing to repair"
    failures = [event for event in events if event.get("kind") == "reader_failed"]
    assert [event["retryable"] for event in failures] == [True], failures
    assert names == [
        "job_started",
        "preprocess_succeeded",
        "call_started",
        "request_attempt_started",
        "response_received",
        "call_failed",
        "reader_failed",
        "call_started",
        "request_attempt_started",
        "response_received",
        "call_finished",
        "call_started",
        "request_attempt_started",
        "response_received",
        "call_finished",
        "diagram_stage",
        "reader_succeeded",
        "job_completed",
    ], names
    read_starts = [
        event
        for event in events
        if event.get("event") == "call_started" and event.get("call_type") == "read"
    ]
    assert all(event["input_mode"] == "page_marked_text" for event in read_starts)
    assert all(event["pages"] == [] and event["images"] == [] for event in read_starts)
    assert all(event["available_pages"] == [1, 2] for event in read_starts)
    assert all(event["text_chars"] > 0 for event in read_starts)


def _events(result) -> list[dict]:
    path = (result.job_path).parent / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_a_resumed_run_renders_without_reading_the_paper_again(tmp_path: Path) -> None:
    """A report on disk whose output file is gone must not be paid for twice."""

    pdf = make_pdf(tmp_path / "paper.pdf", pages=2)
    report = sample_report()
    report["paper_id"] = "paper"
    fake = FakeClient([report])
    original = Reader._make_client
    Reader._make_client = lambda self: fake  # type: ignore[assignment]
    try:
        first = run_batch(pdf, batch_options(tmp_path))
        assert first[0].output_path is not None and first[0].output_path.exists()
        calls_after_first = len(fake.calls)
        # The output can go missing on its own (a rename, a failed render)
        # while the paid report.json stays behind.
        first[0].output_path.unlink()

        second = run_batch(pdf, batch_options(tmp_path))
    finally:
        Reader._make_client = original  # type: ignore[assignment]

    assert len(fake.calls) == calls_after_first, "the resume re-read the paper"
    assert second[0].status in {"done", "needs_review"}, second[0].error
    assert second[0].skipped is False
    assert second[0].output_path.exists()


def test_force_reread_creates_a_fresh_run_and_calls_the_reader_again(
    tmp_path: Path, monkeypatch
) -> None:
    pdf = make_pdf(tmp_path / "paper.pdf", pages=2)
    calls: list[str] = []

    def fake_reader(**kwargs):
        calls.append(str(kwargs["paper_id"]))
        report = sample_report()
        report["paper_id"] = str(kwargs["paper_id"])
        return ReadResult(report=report, requests_made=1)

    monkeypatch.setattr("src.batch._reader_callable", lambda: fake_reader)
    options = batch_options(tmp_path, force_reread=True)

    first = run_batch(pdf, options)[0]
    second = run_batch(pdf, options)[0]

    assert calls == ["paper", "paper"]
    assert first.job_path.parent != second.job_path.parent
    assert first.skipped is False and second.skipped is False
    first_job = json.loads(first.job_path.read_text(encoding="utf-8"))
    second_job = json.loads(second.job_path.read_text(encoding="utf-8"))
    assert first_job["forced_reread"] is True
    assert second_job["forced_reread"] is True
    assert first_job["force_nonce"] != second_job["force_nonce"]


def _portable_config(tmp_path: Path, **overrides) -> Path:
    """Write a config that carries its schema and prompts, as the real one does.

    ``schemas/`` and ``prompts/`` are resolved next to the config file, so a
    config dropped into a bare directory fails validation with a missing-schema
    error before it can test anything else.
    """

    import shutil
    import yaml

    config = yaml.safe_load((PROJECT / "config.yaml").read_text(encoding="utf-8"))
    for key, value in overrides.items():
        section, _, name = key.partition(".")
        config.setdefault(section, {})[name] = value
    for directory in ("schemas", "prompts"):
        shutil.copytree(PROJECT / directory, tmp_path / directory, dirs_exist_ok=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return config_path


def test_an_oversized_pdf_fails_before_any_image_is_rendered(tmp_path: Path) -> None:
    """The page ceiling must stop the paper, not discover the limit via the endpoint."""

    pdf = make_pdf(tmp_path / "long.pdf", pages=4)
    config_path = _portable_config(tmp_path, **{"input.max_pages": 2})

    fake = FakeClient([sample_report()])
    original = Reader._make_client
    Reader._make_client = lambda self: fake  # type: ignore[assignment]
    try:
        results = run_batch(pdf, batch_options(tmp_path, config_path=config_path))
    finally:
        Reader._make_client = original  # type: ignore[assignment]

    result = results[0]
    assert result.status == "failed"
    assert "input_over_limit" in str(result.error)
    assert fake.calls == [], "an oversized paper must not be sent to the model"
    workspace = tmp_path / "workspace" / _paper_id_of(result)
    assert not list(workspace.glob("pages/page-*.png")), "the pages were rendered anyway"
    job = json.loads(result.job_path.read_text(encoding="utf-8"))
    assert job["failure_reason"] == "input_over_limit"


def test_an_over_long_text_fails_before_the_first_paid_request(tmp_path: Path) -> None:
    """The text ceiling is a cost guard, not a discovery made by the endpoint."""

    pdf = make_pdf(tmp_path / "long.pdf", pages=2)
    config_path = _portable_config(tmp_path, **{"input.max_text_chars": 10})

    fake = FakeClient([sample_report()])
    original = Reader._make_client
    Reader._make_client = lambda self: fake  # type: ignore[assignment]
    try:
        results = run_batch(pdf, batch_options(tmp_path, config_path=config_path))
    finally:
        Reader._make_client = original  # type: ignore[assignment]

    result = results[0]
    assert result.status == "failed"
    assert "input_over_limit" in str(result.error)
    assert "max_text_chars" in str(result.error)
    assert fake.calls == [], "an over-long paper must not be sent to the model"
    job = json.loads(result.job_path.read_text(encoding="utf-8"))
    assert job["failure_reason"] == "input_over_limit"


def test_the_ab_baseline_mode_uploads_every_page_through_the_batch(tmp_path: Path) -> None:
    """`full_page_images` stays a working A/B baseline: text in, text ignored."""

    pdf = make_pdf(tmp_path / "probe.pdf", pages=2)
    config_path = _portable_config(
        tmp_path, **{"input.mode": "full_page_images", "input.include_all_pdf_pages": True}
    )

    report = sample_report()
    report["paper_id"] = "probe"
    fake = FakeClient([report, diagram_payload()])
    original = Reader._make_client
    Reader._make_client = lambda self: fake  # type: ignore[assignment]
    try:
        results = run_batch(pdf, batch_options(tmp_path, config_path=config_path))
    finally:
        Reader._make_client = original  # type: ignore[assignment]

    assert results[0].status != "failed", results[0].error
    parts = fake.calls[0]["messages"][1]["content"]
    assert sum(item["type"] == "image_url" for item in parts) == 2
    assert "--- PDF_PAGE=1 ---" not in parts[0]["text"]
    # The diagram stage is self-contained here: an image-only read has no
    # replayable text prefix.
    assert "--- PDF_PAGE=1 ---" not in fake.calls[1]["messages"][1]["content"][0]["text"]


def _paper_id_of(result) -> str:
    from src.batch import _paper_id

    return _paper_id(result.pdf_path)


def test_the_page_ceiling_can_be_disabled(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "long.pdf", pages=4)
    config_path = _portable_config(tmp_path, **{"input.max_pages": None})

    report = sample_report()
    report["paper_id"] = "long"
    fake = FakeClient([report])
    original = Reader._make_client
    Reader._make_client = lambda self: fake  # type: ignore[assignment]
    try:
        results = run_batch(pdf, batch_options(tmp_path, config_path=config_path))
    finally:
        Reader._make_client = original  # type: ignore[assignment]

    assert results[0].status != "failed", results[0].error
    assert fake.calls, "with the guard off the paper must be sent"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
