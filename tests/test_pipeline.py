"""Regression tests for the failures found in the first review pass.

Each test here maps to a concrete defect: a refinement round that lost its
crops, a crash while collecting the follow-up images, a paid retry of a
truncated response, a rendered status that never left "running", and config
keys that were read but never applied.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.batch import (
    BatchOptions,
    _coverage_text,
    _requests_made,
    _retryable,
    run_batch,
)
from src.reader import (
    ReadResult,
    Reader,
    ReaderConfig,
    ReaderResponseError,
    schema_errors,
)
from src.render_report import render_report
from src.validate import validate_report
from tests.test_local import pages_manifest, sample_report

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
    report["evidence"][0]["image_id"] = "crop-01"
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
    report["evidence"][0]["image_id"] = "crop-01"  # evidence says pdf_page 1
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
    # The refinement answer forgets the item the first round set aside, and
    # drops the malformed requests; both facts still have to reach the file.
    refined = sample_report()
    fake = FakeClient([candidate, refined])
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
    fake = FakeClient([candidate, sample_report()])
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
    # The refinement call must not claim the whole paper was re-sent.
    assert "complete input for this call" not in second_text
    assert "do not treat" in second_text


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


def test_schema_violation_is_repaired_without_images(tmp_path: Path) -> None:
    broken = sample_report()
    broken["reader_notes"] = "extra field the schema forbids"
    fake = FakeClient([broken, sample_report()])
    reader = Reader(ReaderConfig.from_mapping({"model": {"name": "m"}}), client=fake)
    result = reader.read("demo", page_images(tmp_path))

    assert "reader_notes" not in result.report
    assert len(fake.calls) == 2
    assert all(item.get("type") == "text" for item in fake.calls[1]["messages"][1]["content"])


def test_repair_budget_is_respected(tmp_path: Path) -> None:
    broken = sample_report()
    broken["reader_notes"] = "extra"
    fake = FakeClient([broken, broken, broken])
    config = ReaderConfig.from_mapping(
        {"model": {"name": "m"}, "refinement": {"max_format_repairs": 1}}
    )
    reader = Reader(config, client=fake)
    result = reader.read("demo", page_images(tmp_path))

    # One repair attempt, then the still-invalid report is returned as-is for
    # the batch validator to reject; the model is not asked again.
    assert len(fake.calls) == 2
    assert "reader_notes" in result.report


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
    assert fake.calls[0]["extra_body"] == {"enable_thinking": True}


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


def test_schema_errors_helper_returns_readable_paths() -> None:
    schema = json.loads((PROJECT / "schemas" / "report.schema.json").read_text())
    broken = sample_report()
    del broken["claims"]
    errors = schema_errors(broken, schema)
    assert any("'claims' is a required property" in error for error in errors)
    assert errors and errors[0].startswith("$")


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
    assert _requests_made({"requests_made": 3}) == 3
    assert _requests_made(None) == 1


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
    with pytest.raises(Exception):
        _check_unsupported_config({"input": {"include_all_pdf_pages": False}})


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

    candidate = sample_report()
    candidate["paper_id"] = "probe"
    candidate["visual_requests"] = [
        {"pdf_page": 1, "reason": "digits unclear", "crop": {"x0": 0.1, "y0": 0.1, "x1": 0.6, "y1": 0.5}}
    ]
    refined = sample_report()
    refined["paper_id"] = "probe"
    # A paper can only be "done" when every relation is evidence-backed.
    for edge in refined["method"]["edges"]:
        edge["confirmed"] = True
    refined["evidence"].append(
        {
            "id": "E3",
            "pdf_page": 1,
            "section": "4 Experiments",
            "figure_or_table": "Table 1",
            "locator": "row 2",
            "quote": "",
            "image_id": "crop-01",
        }
    )
    refined["experiments"][0]["evidence_ids"].append("E3")

    fake = FakeClient([candidate, refined])
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
    # The header states the real status and the input coverage.
    assert "- **Status:** `done`" in text
    assert "全篇 2 页" in text
    # Nothing is pending, and the pending list says so in the opening block.
    assert text.index("## 待核查事项") < text.index("## 摘要")
    assert "## 待核查事项\n\n无。" in text
    # Crop and page links point at files that exist.
    for link in (item for item in text.split("](") if item.startswith("<")):
        target = link[1:].split(">")[0]
        assert (output.parent / target).resolve().exists(), target
    # The refinement round is accounted for in the job record.
    job = json.loads((results[0].job_path).read_text(encoding="utf-8"))
    assert job["requests"] == 2
    assert job["output_coverage"].startswith("全篇 2 页")


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
    candidate["visual_requests"] = [{"pdf_page": 2, "reason": "edge unclear", "crop": None}]
    refined = sample_report()
    refined["paper_id"] = "probe"
    for edge in refined["method"]["edges"]:
        edge["confirmed"] = True
    # The prompt tells the model to preserve unresolved_items, so the model
    # echoing the first round's derived items back is a realistic answer.
    refined["unresolved_items"] = [
        "未确认的方法关系：Stage B → Stage A（dependency）未获证据确认，已从方法图中省略",
        "补看请求未能执行：PDF 第 2 页（edge unclear）",
    ]
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


def test_a_fulfilled_request_is_not_reported_as_pending(tmp_path: Path) -> None:
    """A crop that was just sent is not a request that failed to run.

    The refine prompt tells the model to leave untouched parts of the candidate
    alone, so it echoes the request list it was handed.  Read as pending, that
    echo turned a successful refinement round into phantom "未能执行" items.
    """

    request = {
        "pdf_page": 2,
        "reason": "Table 4 numbers are too small to read",
        "crop": {"x0": 0.1, "y0": 0.2, "x1": 0.9, "y1": 0.8},
    }
    candidate = sample_report()
    candidate["paper_id"] = "probe"
    candidate["visual_requests"] = [request]
    refined = sample_report()
    refined["paper_id"] = "probe"
    refined["visual_requests"] = [json.loads(json.dumps(request))]
    fake = FakeClient([candidate, refined])
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


def test_a_new_request_after_refinement_is_still_reported_as_pending(tmp_path: Path) -> None:
    """A request for a different region was not executed and must say so."""

    candidate = sample_report()
    candidate["paper_id"] = "probe"
    candidate["visual_requests"] = [
        {"pdf_page": 2, "reason": "Table 4 numbers are too small", "crop": None}
    ]
    refined = sample_report()
    refined["paper_id"] = "probe"
    refined["visual_requests"] = [
        {"pdf_page": 2, "reason": "Table 4 numbers are too small", "crop": None},
        {"pdf_page": 3, "reason": "Figure 2 legend needs a closer look", "crop": None},
    ]
    fake = FakeClient([candidate, refined])
    images = page_images(tmp_path)

    def renderer(*, pdf_page, crop, dpi, source_page=None):
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
    assert result.calls[0].format_repaired is True
    assert "method" in result.report


def test_a_repair_whose_result_is_rejected_leaves_the_label_off(tmp_path: Path) -> None:
    """A repair round that ran but was thrown away must not claim the report."""

    broken = sample_report()
    del broken["method"]
    still_broken = sample_report()
    del still_broken["claims"]
    fake = FakeClient([broken, still_broken])
    reader = Reader(ReaderConfig.from_mapping({"model": {"name": "m"}}), client=fake)
    result = reader.read("demo", page_images(tmp_path))

    assert len(fake.calls) == 2
    assert result.calls[0].format_repaired is False
    assert "method" not in result.report, "the original parsed report must be kept"


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


def test_the_batch_retries_a_substance_free_response_with_the_images(tmp_path: Path) -> None:
    """The ProtocolGuard failure: `[]` after a long reasoning pass.

    It must not be repaired (a repair invented a whole report once), and it
    must not end the paper either: the second attempt resends the pages.
    """

    pdf = make_pdf(tmp_path / "paper.pdf", pages=2)
    report = sample_report()
    report["paper_id"] = "paper"
    fake = FakeClient(["[]", report])
    original = Reader._make_client
    Reader._make_client = lambda self: fake  # type: ignore[assignment]
    try:
        results = run_batch(pdf, batch_options(tmp_path))
    finally:
        Reader._make_client = original  # type: ignore[assignment]

    result = results[0]
    assert result.status in {"done", "needs_review"}, result.error
    assert len(fake.calls) == 2, "the empty response was not retried"
    for call in fake.calls:
        parts = call["messages"][1]["content"]
        assert any(item["type"] == "image_url" for item in parts), (
            "every attempt must carry the page images"
        )
    events = _events(result)
    names = [event.get("event") or event.get("kind") for event in events]
    assert "repair_started" not in names, "there was nothing to repair"
    failures = [event for event in events if event.get("kind") == "reader_failed"]
    assert [event["retryable"] for event in failures] == [True], failures
    assert names == [
        "job_started",
        "preprocess_succeeded",
        "call_started",
        "call_failed",
        "reader_failed",
        "call_started",
        "call_finished",
        "reader_succeeded",
        "job_completed",
    ], names


def _events(result) -> list[dict]:
    path = (result.job_path).parent / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
