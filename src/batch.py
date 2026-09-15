"""Batch orchestration for the paper-reading pipeline.

The batch layer owns durable state and local artifacts.  PDF preparation and
model reading live in :mod:`src.preprocess` and :mod:`src.reader`; keeping the
orchestration here means a failed Markdown render can be resumed without
repeating a paid model request.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
import re
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from .render_report import render_report


DATE_RE = re.compile(r"^20\d{2}-\d{2}-\d{2}$")
# ``--date`` accepts the plain date plus an optional suffix, so a re-run of the
# same day can be written next to the first one (``2026-09-15-v5``) instead of
# overwriting it.  The suffix is only ever a directory label: nothing parses a
# batch date back out of the output tree, so ``DATE_RE`` stays strict where a
# date really is a date (detecting a dated input directory).
DATE_LABEL_RE = re.compile(r"^20\d{2}-\d{2}-\d{2}(-[A-Za-z0-9._]+)*$")
# The page ceiling applied when ``input.max_pages`` is absent.  Long conference
# papers with appendices run past 40 pages; 80 leaves room for those while
# still stopping a thesis-sized PDF before its images are rendered and sent.
DEFAULT_MAX_PAGES = 80
REPORT_KEYS = {
    "schema_version",
    "paper_id",
    "title",
    "short_summary",
    "problem",
    "related_work",
    "method",
    "experiments",
    "future_work",
    "reader_analysis",
    "full_summary",
    "reading_guide",
    "claims",
    "evidence",
    "visual_requests",
    "unresolved_items",
}
# ``diagrams`` is deliberately not in the set above: it is optional in the
# schema and is added by the diagram stage after the report is written, so a
# report without it is still a complete report.
INPUT_MODES = ("hybrid_text_visual", "full_page_images")


class BatchError(RuntimeError):
    """An error that should be recorded in job.json."""


class BatchPermanentError(BatchError):
    """An error that a retry cannot fix, such as an exhausted request budget."""


class BatchInterruptedError(BatchError):
    """Internal signal used to persist a user-requested interruption."""


@dataclass(frozen=True)
class BatchOptions:
    config_path: Path
    workspace_root: Path
    output_root: Path
    concurrency: int
    max_retries: int
    max_requests_per_paper: int
    resume: bool
    retry_backoff_s: float
    batch_date: str
    force_reread: bool = False


@dataclass
class PaperResult:
    paper_id: str
    pdf_path: Path
    status: str
    title: str = ""
    summary: str = ""
    report_path: Path | None = None
    output_path: Path | None = None
    job_path: Path | None = None
    unresolved_items: list[str] | None = None
    error: str | None = None
    skipped: bool = False


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"cannot encode {type(value).__name__}")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, default=_json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _append_jsonl(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(event), ensure_ascii=False, default=_json_default) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - requirements install path
        raise BatchError("PyYAML is required to read config.yaml") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _deep_get(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_default)


def _fingerprint_file(digest: Any, path: Path, label: str) -> None:
    """Hash one labelled file so equal byte concatenations cannot collide."""

    if not path.exists():
        return
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(path.read_bytes())
    digest.update(b"\0")


def _config_fingerprint(
    config: Mapping[str, Any],
    config_path: Path,
    *,
    runtime_root: Path | None = None,
) -> str:
    """Identify every input that can change a run's model-visible result.

    The old fingerprint covered configuration, schema and prompts but not the
    executing pipeline.  Consequently ``--resume`` could reuse a failed or
    stale run after a code fix.  Include the Python sources and dependency
    lock now so a retry after changing the reader gets a fresh run directory.
    """

    digest = hashlib.sha256()
    digest.update(_stable_json(config).encode("utf-8"))
    _fingerprint_file(digest, config_path, "config.yaml")
    schema_path = config_path.parent / "schemas" / "report.schema.json"
    _fingerprint_file(digest, schema_path, "schemas/report.schema.json")
    for prompt in ("reader.md", "refine.md", "diagram.md"):
        prompt_path = config_path.parent / "prompts" / prompt
        _fingerprint_file(digest, prompt_path, f"prompts/{prompt}")

    project_root = runtime_root or Path(__file__).resolve().parents[1]
    source_dir = project_root / "src"
    for source_path in sorted(source_dir.glob("*.py")):
        _fingerprint_file(digest, source_path, f"src/{source_path.name}")
    for dependency_file in ("pyproject.toml", "uv.lock"):
        _fingerprint_file(digest, project_root / dependency_file, dependency_file)
    return digest.hexdigest()


def _paper_id(pdf_path: Path) -> str:
    """Use preprocess' naming policy when available, with a safe fallback."""

    try:
        from .preprocess import make_paper_id

        return make_paper_id(pdf_path)
    except Exception:
        candidate = re.sub(r"[^A-Za-z0-9._-]+", "-", pdf_path.stem).strip(".-_")
        return candidate or f"paper-{hashlib.sha256(str(pdf_path).encode()).hexdigest()[:10]}"


def _pdfs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise BatchError(f"input file is not a PDF: {input_path}")
        return [input_path.resolve()]
    if input_path.is_dir():
        return sorted(
            (item.resolve() for item in input_path.rglob("*.pdf") if item.is_file()),
            key=lambda item: item.as_posix().lower(),
        )
    raise BatchError(f"input path does not exist: {input_path}")


def _call_with_supported_kwargs(function: Callable[..., Any], kwargs: Mapping[str, Any]) -> Any:
    """Call a project hook while tolerating small API naming differences."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(**kwargs)
    parameters = signature.parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return function(**kwargs)
    selected = {name: value for name, value in kwargs.items() if name in parameters}
    missing: list[str] = []
    for name, parameter in parameters.items():
        if parameter.default is inspect.Parameter.empty and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ) and name not in selected:
            missing.append(name)
    if missing:
        raise BatchError(f"{function.__module__}.{function.__name__} requires unsupported parameters: {', '.join(missing)}")
    return function(**selected)


def _prepare_pdf(pdf_path: Path, paper_dir: Path, config: Mapping[str, Any]) -> Mapping[str, Any]:
    from .preprocess import preprocess_pdf

    kwargs = {
        "pdf_path": pdf_path,
        "workspace_dir": paper_dir,
        "paper_id": paper_dir.name,
        "render_dpi": _deep_get(config, "input", "render_dpi", default=160),
        "image_format": _deep_get(config, "input", "image_format", default="png"),
        "image_max_pixels": _deep_get(config, "input", "image_max_pixels", default=None),
        "extract_text": _deep_get(config, "input", "extract_text", default=False),
    }
    result = _call_with_supported_kwargs(preprocess_pdf, kwargs)
    if not isinstance(result, Mapping):
        raise BatchError("preprocess_pdf must return a mapping")
    return result


def _reader_callable() -> Callable[..., Any]:
    try:
        from . import reader
    except ImportError as exc:
        raise BatchError("src.reader is not available") from exc
    for name in ("read_paper", "run_reader", "run", "process_paper", "read"):
        function = getattr(reader, name, None)
        if callable(function):
            return function
    # The production reader intentionally exposes a stateful Reader because
    # it owns the request budget and visual-refinement session.  Adapt that
    # class here instead of forcing the CLI to know about SDK details.
    reader_class = getattr(reader, "Reader", None)
    if reader_class is not None:
        def read_with_reader(**kwargs: Any) -> Any:
            config = kwargs.pop("config", None)
            event_recorder = kwargs.pop("event_recorder", None)
            schema_path = kwargs.pop("schema_path", None)
            prompt_path = kwargs.pop("prompt_path", None)
            refine_prompt_path = kwargs.pop("refine_prompt_path", None)
            instance = _call_with_supported_kwargs(
                reader_class,
                {
                    "config": config,
                    "event_recorder": event_recorder,
                    "schema_path": schema_path,
                    "prompt_path": prompt_path,
                    "refine_prompt_path": refine_prompt_path,
                    "diagram_prompt_path": kwargs.get("diagram_prompt_path"),
                },
            )
            read_parameters = {
                "paper_id": kwargs.get("paper_id"),
                "pages": kwargs.get("pages", ()),
                "text": kwargs.get("text"),
                "metadata": kwargs.get("metadata"),
                "title": kwargs.get("title", ""),
                "prompt": kwargs.get("prompt"),
                "schema": kwargs.get("schema"),
                "renderer": kwargs.get("renderer"),
                "original_pages": kwargs.get("original_pages", kwargs.get("pages", ())),
            }
            result = instance.read(**read_parameters)
            return _attach_diagrams(
                result,
                instance,
                paper_id=str(kwargs.get("paper_id") or ""),
                title=str(kwargs.get("title") or ""),
            )

        return read_with_reader
    raise BatchError("src.reader exposes none of read_paper/run_reader/run/process_paper/read")


def _paper_metadata(preparation: Mapping[str, Any]) -> dict[str, Any]:
    """The paper metadata a reading call is given, taken from preprocess.

    Deliberately small: it is part of the prompt, and nearly all of it is
    visible in the text itself.  What it adds is the source filename and the
    total page count -- the one thing extracted text cannot show, because a
    page that failed to render is a page that never appears in it.
    """

    source = preparation.get("metadata")
    source = source if isinstance(source, Mapping) else {}
    metadata: dict[str, Any] = {}
    for key in ("paper_id", "filename", "total_pages"):
        value = source.get(key)
        if value is not None:
            metadata[key] = value
    coverage = source.get("input_coverage")
    if isinstance(coverage, Mapping) and coverage.get("complete") is False:
        # A paper whose pages did not all render must not be read as complete.
        metadata["input_coverage_complete"] = False
    return metadata


def _attach_diagrams(result: Any, reader: Any, *, paper_id: str, title: str) -> Any:
    """Run the separate Mermaid stage and merge it into a reader result.

    The stage is a second, text-only call that reads the grounded report data
    rather than the PDF, so it cannot move a fact -- only draw one.  A failure
    here costs the diagrams and nothing else, so it is recorded as an
    unresolved item instead of being raised: the report itself is already
    complete, validated and paid for.
    """

    report = getattr(result, "report", None)
    generate = getattr(reader, "generate_diagrams", None)
    if not isinstance(report, dict) or not callable(generate):
        return result
    if getattr(getattr(reader, "config", None), "diagrams_enabled", True) is False:
        return result
    try:
        diagrams = generate(paper_id or str(report.get("paper_id", "")), report, title=title)
    except Exception as exc:  # noqa: BLE001 - a drawing failure is not a reading failure
        try:
            from .reader import DIAGRAM_FAILED_PREFIX
        except ImportError:  # pragma: no cover - reader is part of this project
            DIAGRAM_FAILED_PREFIX = "方法图生成失败："
        _merge_unresolved(report, [f"{DIAGRAM_FAILED_PREFIX}{type(exc).__name__}: {exc}"])
        _resync_unresolved(result, report)
        return result

    drawn = getattr(diagrams, "diagrams", None)
    if isinstance(drawn, Sequence) and not isinstance(drawn, (str, bytes)):
        report["diagrams"] = [dict(item) for item in drawn if isinstance(item, Mapping)]
    for attribute in ("calls", "raw_responses"):
        extra = getattr(diagrams, attribute, None)
        existing = getattr(result, attribute, None)
        if isinstance(extra, Sequence) and isinstance(existing, list):
            existing.extend(extra)
    if hasattr(result, "requests_made"):
        total = getattr(reader, "requests_made", None)
        if isinstance(total, int) and not isinstance(total, bool):
            result.requests_made = total
    _merge_unresolved(report, getattr(diagrams, "unresolved_items", None))
    _resync_unresolved(result, report)
    return result


def _resync_unresolved(result: Any, report: Mapping[str, Any]) -> None:
    """Keep the returned result's review list equal to the report's."""

    items = report.get("unresolved_items")
    if isinstance(items, list) and hasattr(result, "unresolved_items"):
        result.unresolved_items = [str(item) for item in items]


def _invoke_reader(
    pdf_path: Path,
    paper_dir: Path,
    run_dir: Path,
    preparation: Mapping[str, Any],
    config: Mapping[str, Any],
    max_requests: int,
) -> Any:
    function = _reader_callable()
    # Reader page paths in pages.json are relative to the paper workspace;
    # model input requires absolute paths when the batch is launched elsewhere.
    page_values: list[Any] = []
    for page in preparation.get("pages", []):
        if not isinstance(page, Mapping):
            page_values.append(page)
            continue
        normalised = dict(page)
        value = normalised.get("path", normalised.get("image_path"))
        if value is not None and not Path(str(value)).is_absolute():
            absolute = paper_dir / str(value)
            normalised["path"] = str(absolute)
            if normalised.get("image_path") is not None:
                normalised["image_path"] = str(absolute)
        page_values.append(normalised)

    def crop_renderer(*, pdf_page: int, crop: Any, dpi: float, source_page: Any = None) -> Any:
        from .preprocess import render_crop

        crop_dir = run_dir / "crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        index = len(list(crop_dir.glob("crop-*.png"))) + 1
        target = crop_dir / f"crop-{index:02d}.png"
        source_pdf = preparation.get("original_pdf", paper_dir / "original.pdf")
        return render_crop(
            source_pdf,
            pdf_page,
            crop,
            target,
            dpi=dpi,
            image_format="png",
            max_pixels=_deep_get(config, "input", "image_max_pixels", default=None),
        )
    # The page-marked full text is the input of a text-first reading.  It is
    # read here rather than inside the reader so that the batch layer owns the
    # file locations, and so a missing paper.txt is a visible failure instead
    # of a silently image-only read.
    text: str | None = None
    text_value = preparation.get("text_path")
    if text_value:
        text_path = Path(str(text_value))
        if not text_path.is_absolute():
            text_path = paper_dir / text_path
        if text_path.exists():
            text = text_path.read_text(encoding="utf-8")
        elif _deep_get(config, "input", "mode", default="hybrid_text_visual") == "hybrid_text_visual":
            raise BatchError(
                f"input.extract_text=true 但缺少全文文本 {text_path}；"
                "请重新预处理该 PDF，或关闭 hybrid 模式"
            )

    reader_config = copy.deepcopy(dict(config))
    batch_config = reader_config.setdefault("batch", {})
    if isinstance(batch_config, Mapping):
        batch_config = dict(batch_config)
        batch_config["max_requests_per_paper"] = max(1, int(max_requests))
        # Batch owns retries so that the durable job counter and retry policy
        # have one source of truth.  Direct Reader users still get the
        # configured per-call retries; nesting both policies could exceed the
        # per-paper request budget without the batch layer knowing it.
        batch_config["max_retries_per_call"] = 0
        reader_config["batch"] = batch_config
    kwargs = {
        "pdf_path": pdf_path,
        "paper_path": pdf_path,
        "original_pdf": preparation.get("original_pdf", pdf_path),
        "workspace_dir": paper_dir,
        "workspace": paper_dir,
        "run_dir": run_dir,
        "output_dir": run_dir,
        "preparation": preparation,
        "preprocess_result": preparation,
        "prepared": preparation,
        "pages": page_values,
        "paper_pages": page_values,
        "text": text,
        "metadata": _paper_metadata(preparation),
        "title": "",
        "renderer": crop_renderer,
        "original_pages": page_values,
        "event_recorder": None,
        "schema_path": Path(__file__).resolve().parents[1] / "schemas" / "report.schema.json",
        "prompt_path": Path(__file__).resolve().parents[1] / "prompts" / "reader.md",
        "refine_prompt_path": Path(__file__).resolve().parents[1] / "prompts" / "refine.md",
        "diagram_prompt_path": Path(__file__).resolve().parents[1] / "prompts" / "diagram.md",
        "pages_json": preparation.get("pages_path", paper_dir / "pages.json"),
        "config": reader_config,
        "max_requests": max_requests,
        "max_requests_per_paper": max_requests,
        "paper_id": preparation.get("paper_id", paper_dir.name),
    }
    try:
        from .reader import JsonlEventRecorder

        kwargs["event_recorder"] = JsonlEventRecorder(run_dir / "events.jsonl")
    except (ImportError, AttributeError):
        pass
    return _call_with_supported_kwargs(function, kwargs)


def _extract_report(value: Any, run_dir: Path) -> tuple[dict[str, Any], Mapping[str, Any]]:
    """Normalize reader return values while retaining optional usage metadata."""

    metadata: Mapping[str, Any] = {}
    candidate: Any = value
    if isinstance(value, Mapping):
        metadata = value
        for key in ("report", "report_json", "result"):
            if isinstance(value.get(key), Mapping):
                candidate = value[key]
                break
        else:
            candidate = value
    elif hasattr(value, "report"):
        candidate = getattr(value, "report")
        raw_metadata = getattr(value, "metadata", {})
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        calls = getattr(value, "calls", None)
        requests_made = getattr(value, "requests_made", None)
        if isinstance(requests_made, int) and not isinstance(requests_made, bool):
            metadata["requests_made"] = requests_made
        if isinstance(calls, Sequence):
            metadata["calls"] = [
                call.as_dict() if hasattr(call, "as_dict") else call
                for call in calls
            ]
        unresolved = getattr(value, "unresolved_items", None)
        if isinstance(unresolved, Sequence) and not isinstance(unresolved, (str, bytes)):
            metadata["unresolved_items"] = list(unresolved)
        supplemental = getattr(value, "supplemental_images", None)
        if isinstance(supplemental, Sequence) and not isinstance(supplemental, (str, bytes)):
            metadata["supplemental_images"] = [
                dict(item) if isinstance(item, Mapping) else item for item in supplemental
            ]
        raw_responses = getattr(value, "raw_responses", None)
        if isinstance(raw_responses, Sequence):
            raw_dir = run_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            for index, response in enumerate(raw_responses, start=1):
                payload: Any = response
                if hasattr(response, "model_dump"):
                    try:
                        payload = response.model_dump(mode="json")
                    except Exception:
                        payload = response.model_dump()
                elif hasattr(response, "to_dict"):
                    payload = response.to_dict()
                elif not isinstance(response, (Mapping, list, str, int, float, bool, type(None))):
                    payload = {"repr": repr(response)}
                try:
                    _atomic_json(raw_dir / f"response-{index:02d}.json", payload)
                except (TypeError, ValueError):
                    _atomic_json(raw_dir / f"response-{index:02d}.json", {"repr": repr(response)})
    elif isinstance(value, (str, os.PathLike)):
        candidate = _load_json(Path(value))
    if isinstance(candidate, Mapping) and REPORT_KEYS.intersection(candidate.keys()):
        report = dict(candidate)
        _merge_unresolved(report, metadata.get("unresolved_items"))
        return report, metadata
    report_path = run_dir / "report.json"
    if report_path.exists():
        loaded = _load_json(report_path)
        if isinstance(loaded, Mapping):
            report = dict(loaded)
            _merge_unresolved(report, metadata.get("unresolved_items"))
            return report, metadata
    raise BatchError("reader did not return a report object or create run/report.json")


def _merge_unresolved(report: dict[str, Any], items: Any) -> None:
    """Keep refinement problems inside the report that is written to disk.

    Status and the daily index are derived from ``report.json``, so an
    unresolved item that lives only in the reader's return value would be
    silently dropped from the delivered artifact.
    """

    if not items:
        return
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return
    existing = report.get("unresolved_items")
    if not isinstance(existing, list):
        existing = []
        report["unresolved_items"] = existing
    for item in items:
        text = str(item)
        if text and text not in existing:
            existing.append(text)


def _validate_report(
    report: Mapping[str, Any],
    config_path: Path,
    pages_path: Path | None = None,
    supplemental_images: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[str], list[str]]:
    """Return ``(errors, warnings)`` for a report using src.validate when present.

    Crops and re-rendered pages from the refinement round are passed as an
    image-id mapping so evidence that cites a crop is checked against the page
    it was cropped from instead of being rejected as an unknown image.
    """

    image_ids: dict[str, Any] | None = None
    if supplemental_images:
        image_ids = {
            str(item.get("image_id") or item.get("id")): item.get("pdf_page")
            for item in supplemental_images
            if isinstance(item, Mapping) and (item.get("image_id") or item.get("id"))
        }

    try:
        from . import validate
    except ImportError:
        validate = None  # type: ignore
    if validate is not None:
        for name in ("validate_report", "validate"):
            function = getattr(validate, name, None)
            if not callable(function):
                continue
            kwargs = {
                "report": report,
                "data": report,
                "report_path": None,
                "schema_path": config_path.parent / "schemas" / "report.schema.json",
                "pages": pages_path,
                "image_ids": image_ids,
            }
            try:
                result = _call_with_supported_kwargs(function, kwargs)
            except TypeError:
                continue
            # ``ValidationResult`` is a Mapping as well as an object, so the
            # structured form has to be recognised before the mapping form.
            if hasattr(result, "errors") and hasattr(result, "valid"):
                if bool(getattr(result, "valid")):
                    return [], [str(item) for item in getattr(result, "warnings", [])]
                errors = [str(item) for item in getattr(result, "errors", [])]
                warnings = [str(item) for item in getattr(result, "warnings", [])]
                return errors, warnings
            if result is True or result is None:
                return [], []
            if isinstance(result, Mapping):
                errors = result.get("errors", result.get("issues", []))
                if not errors and result.get("valid") is True:
                    return [], [str(item) for item in result.get("warnings", [])]
                return [str(item) for item in (errors if isinstance(errors, list) else [errors])], []
            if isinstance(result, (list, tuple, set)):
                return [str(item) for item in result], []
            if isinstance(result, str):
                return [result], []
            return [], []
    schema_path = config_path.parent / "schemas" / "report.schema.json"
    try:
        import jsonschema  # type: ignore

        schema = _load_json(schema_path)
        validator = jsonschema.Draft202012Validator(schema)
        errors = [
            error.message
            for error in sorted(validator.iter_errors(report), key=lambda error: list(error.path))
        ]
        return errors, []
    except (ImportError, FileNotFoundError, json.JSONDecodeError):
        required = REPORT_KEYS - set(report)
        return (
            [f"missing required report fields: {', '.join(sorted(required))}"] if required else [],
            [],
        )


def _retryable(error: BaseException) -> bool:
    """Decide whether re-sending the whole paper request could help.

    A malformed or truncated model response is deliberately excluded: the same
    prompt with the same images reproduces the same failure, and each attempt
    pays for every page image again.  The reader already repairs malformed JSON
    with a text-only call, so a response error that reaches this layer is not
    worth a full re-read.

    The exception is a response with no report content at all (`[]` after a long
    reasoning pass, or an empty message).  Nothing can be reformatted there, and
    the failure is a sampling accident rather than a property of the input, so
    re-sending the images is the one thing that can still work.
    """

    if isinstance(error, BatchPermanentError):
        return False
    try:
        from .reader import (
            ReaderConfigError,
            ReaderDependencyError,
            ReaderInputError,
            ReaderResponseError,
        )
    except ImportError:  # pragma: no cover - reader is part of this project
        pass
    else:
        if isinstance(error, (ReaderConfigError, ReaderDependencyError, ReaderInputError, ReaderResponseError)):
            # getattr, not attribute access: the three other error types carry
            # no such flag.
            return bool(getattr(error, "retryable_with_images", False))
    text = str(error).lower()
    permanent_markers = (
        "401",
        "403",
        "authentication",
        "api key",
        "input_over_limit",
        "input over limit",
        "context length",
        "range of input length",
        "exceeds the maximum",
        "unsupported parameter",
        "invalid parameter",
        "requires unsupported parameters",
        "request budget exhausted",
    )
    return not any(marker in text for marker in permanent_markers)


def _max_pages(config: Mapping[str, Any]) -> int | None:
    """Page ceiling for a single paper, or ``None`` when the guard is off.

    In ``full_page_images`` mode every page is uploaded as an image and images
    are almost the whole prompt: 46 pages measured 109k prompt tokens (~2.4k
    per page), against 27k-44k for the 11-14 page papers in the same batch.  In
    ``hybrid_text_visual`` mode the images no longer reach the endpoint, but the
    ceiling still applies: every page is rendered, indexed and linked, and a
    thesis-sized PDF should fail with a reason rather than quietly become a
    very expensive local render.
    """

    raw = _deep_get(config, "input", "max_pages", default=DEFAULT_MAX_PAGES)
    if raw is None:
        return None
    value = _as_positive_int(raw, "input.max_pages")
    return value


def _max_text_chars(config: Mapping[str, Any]) -> int | None:
    """Character ceiling for the extracted full text, or ``None`` when off."""

    raw = _deep_get(config, "input", "max_text_chars", default=None)
    if raw is None:
        return None
    return _as_positive_int(raw, "input.max_text_chars")


def _check_text_limit(preparation: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    """Stop an over-long full text before it becomes the whole prompt.

    A text-first reading sends the entire extracted text as its prompt, so a
    book-length PDF would be rejected by the provider as a context-length
    error.  Failing here names the real reason, with the real size, before any
    request is paid for.
    """

    limit = _max_text_chars(config)
    text_value = preparation.get("text_path")
    if limit is None or not text_value:
        return
    text_path = Path(str(text_value))
    if not text_path.exists():
        return
    try:
        size = len(text_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise BatchError(f"无法读取全文文本 {text_path}：{exc}") from exc
    if size > limit:
        raise BatchPermanentError(
            f"input_over_limit: 全文 {size} 字符，超过 input.max_text_chars={limit}；"
            "请拆分 PDF 或调高上限后重跑"
        )


def _as_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BatchError(f"{name} must be a positive integer or null, got {value!r}")
    return value


def _pdf_page_count(pdf_path: Path) -> int | None:
    """Count the pages of a PDF without rendering any of them."""

    try:
        import pymupdf
    except ImportError:  # pragma: no cover - depends on the environment
        return None
    try:
        with pymupdf.open(pdf_path) as document:
            return int(document.page_count)
    except Exception:  # noqa: BLE001 - the reader will report an unreadable PDF
        return None


def _input_over_limit(error: BaseException) -> bool:
    """Recognise an endpoint rejection caused by the size of the input."""

    text = str(error).lower()
    markers = (
        "context length",
        "range of input length",
        "input length",
        "too many tokens",
        "input_over_limit",
        "input over limit",
        "exceeds the maximum",
        "maximum context",
    )
    return any(marker in text for marker in markers)


def _format_repaired(metadata: Mapping[str, Any]) -> bool:
    """Whether any call of this run delivered a repair-round report."""

    calls = metadata.get("calls")
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
        return bool(metadata.get("format_repaired"))
    return any(
        bool(call.get("format_repaired"))
        for call in calls
        if isinstance(call, Mapping)
    )


def _requests_made(value: Any) -> int:
    """Count the provider requests a reader call actually issued.

    New reader results and errors expose the exact count, including repairs
    and failed sends.  The call-list fallback keeps compatibility with older
    or injected readers that only record successful logical calls.
    """

    direct = getattr(value, "requests_made", None)
    if isinstance(direct, int) and not isinstance(direct, bool) and direct >= 1:
        return direct
    calls = getattr(value, "calls", None)
    if isinstance(calls, Sequence) and not isinstance(calls, (str, bytes)):
        return max(1, len(calls))
    if isinstance(value, Mapping):
        for key in ("requests_made", "requests", "request_count"):
            raw = value.get(key)
            if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 1:
                return raw
        nested = value.get("calls")
        if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
            return max(1, len(nested))
    return 1


def _write_supplemental(run_dir: Path, images: Any) -> None:
    """Record the crops of a refinement round next to the run's report."""

    if not isinstance(images, Sequence) or isinstance(images, (str, bytes)):
        return
    entries = [dict(item) for item in images if isinstance(item, Mapping)]
    if not entries:
        return
    _atomic_json(
        run_dir / "supplemental.json",
        {"schema_version": "1", "images": entries},
    )


def _read_supplemental(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "supplemental.json"
    if not path.exists():
        return []
    try:
        value = _load_json(path)
    except (json.JSONDecodeError, OSError):
        return []
    images = value.get("images", []) if isinstance(value, Mapping) else []
    return [dict(item) for item in images if isinstance(item, Mapping)] if isinstance(images, list) else []


def _render_options(config: Mapping[str, Any]) -> dict[str, Any]:
    """Translate ``output.*`` into renderer arguments."""

    return {
        "include_mermaid": bool(_deep_get(config, "output", "include_mermaid", default=True)),
        "include_method_steps": bool(
            _deep_get(config, "output", "include_method_steps", default=True)
        ),
        "include_evidence_index": bool(
            _deep_get(config, "output", "include_evidence_index", default=True)
        ),
    }


def _coverage_text(paper_dir: Path) -> str:
    """Summarise what was actually fed to the reader, from metadata.json."""

    metadata_path = paper_dir / "metadata.json"
    if not metadata_path.exists():
        return "未记录输入覆盖情况"
    try:
        metadata = _load_json(metadata_path)
    except (json.JSONDecodeError, OSError):
        return "未记录输入覆盖情况"
    if not isinstance(metadata, Mapping):
        return "未记录输入覆盖情况"
    coverage = metadata.get("input_coverage")
    total = metadata.get("total_pages")
    if not isinstance(coverage, Mapping):
        return f"全篇 {total} 页" if isinstance(total, int) else "未记录输入覆盖情况"
    pages = coverage.get("pdf_pages")
    numbers = [item for item in pages if isinstance(item, int)] if isinstance(pages, list) else []
    if not numbers:
        return f"全篇 {total} 页" if isinstance(total, int) else "未记录输入覆盖情况"
    if numbers == list(range(1, len(numbers) + 1)):
        text = f"全篇 {len(numbers)} 页（PDF 页序 1-{len(numbers)}）"
    else:
        text = f"{len(numbers)} 页（PDF 页序 {numbers[0]}-{numbers[-1]}，不连续）"
    if coverage.get("complete") is False:
        text += "；输入覆盖不完整"
    text += "；未纳入 PDF 之外的补充材料"
    return text


def _read_job(path: Path) -> dict[str, Any]:
    try:
        value = _load_json(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _update_job(path: Path, job: dict[str, Any], **updates: Any) -> None:
    job.update(updates)
    job["updated_at"] = _now()
    _atomic_json(path, job)


def _event(run_dir: Path, kind: str, **fields: Any) -> None:
    _append_jsonl(run_dir / "events.jsonl", {"time": _now(), "kind": kind, **fields})


def _result_from_job(pdf_path: Path, job_path: Path, job: Mapping[str, Any], *, skipped: bool = True) -> PaperResult:
    return PaperResult(
        paper_id=str(job.get("paper_id", pdf_path.stem)),
        pdf_path=pdf_path,
        status=str(job.get("status", "failed")),
        title=str(job.get("title", "")),
        summary=str(job.get("summary", "")),
        report_path=Path(job["report_path"]) if job.get("report_path") else None,
        output_path=Path(job["output_path"]) if job.get("output_path") else None,
        job_path=job_path,
        unresolved_items=list(job.get("unresolved_items", [])) if isinstance(job.get("unresolved_items"), list) else [],
        error=str(job["error"]) if job.get("error") else None,
        skipped=skipped,
    )


def _raise_if_interrupted(stop_event: threading.Event | None) -> None:
    if stop_event is not None and stop_event.is_set():
        raise BatchInterruptedError("interrupted by user")


def process_one(
    pdf_path: Path,
    options: BatchOptions,
    config: Mapping[str, Any],
    config_fingerprint: str,
    stop_event: threading.Event | None = None,
    run_nonce: str | None = None,
) -> PaperResult:
    paper_id = _paper_id(pdf_path)
    paper_dir = options.workspace_root / paper_id
    paper_dir.mkdir(parents=True, exist_ok=True)
    try:
        pdf_digest = _file_digest(pdf_path)
    except OSError as exc:
        return PaperResult(paper_id, pdf_path, "failed", error=str(exc))
    run_identity = f"{pdf_digest}:{config_fingerprint}"
    if run_nonce is not None:
        run_identity += f":force:{run_nonce}"
    run_fingerprint = hashlib.sha256(run_identity.encode("utf-8")).hexdigest()[:16]
    run_dir = paper_dir / "runs" / run_fingerprint
    job_path = run_dir / "job.json"
    report_path = run_dir / "report.json"
    output_date = options.batch_date
    output_path = options.output_root / output_date / f"{paper_id}.md"
    job = _read_job(job_path)
    if (
        options.resume
        and job.get("status") in {"done", "needs_review"}
        and report_path.exists()
        and output_path.exists()
    ):
        return _result_from_job(pdf_path, job_path, job)
    job = {
        **job,
        "schema_version": "1",
        "paper_id": paper_id,
        "pdf_path": str(pdf_path),
        "pdf_sha256": pdf_digest,
        "run_id": run_fingerprint,
        "run_dir": str(run_dir),
        "status": "running",
        "phase": job.get("phase", "preprocess"),
        "config_fingerprint": config_fingerprint,
        "forced_reread": run_nonce is not None,
        "force_nonce": run_nonce,
        "requests": int(job.get("requests", 0) or 0),
        "attempts": int(job.get("attempts", 0) or 0),
        "started_at": job.get("started_at", _now()),
        "error": None,
        "report_path": str(report_path),
        "output_path": str(output_path),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    _update_job(job_path, job)
    _event(
        run_dir,
        "job_started",
        paper_id=paper_id,
        run_id=run_fingerprint,
        forced_reread=run_nonce is not None,
    )

    try:
        _raise_if_interrupted(stop_event)
        # Checked before anything is rendered: the endpoint rejects an oversized
        # input anyway, but by then the images have been rendered and the reason
        # arrives as an opaque provider error.
        if not report_path.exists():
            page_limit = _max_pages(config)
            page_count = _pdf_page_count(pdf_path) if page_limit is not None else None
            if page_count is not None and page_count > page_limit:
                raise BatchPermanentError(
                    "input_over_limit: PDF 有 "
                    f"{page_count} 页，超过 input.max_pages={page_limit}；"
                    "整篇会作为图片一次性上传，请拆分或调高上限后重跑"
                )
        # A report left by an interrupted run is enough to resume from local
        # validation/rendering; no paid reader call is made in that case.
        # A run that resumes from a report on disk has no in-memory reader
        # result, so the provenance the earlier run wrote to job.json is the
        # only thing left to render from.  A fresh read overwrites this below.
        reader_metadata: Mapping[str, Any] = {
            "format_repaired": bool(job.get("format_repaired"))
        }
        preparation: Mapping[str, Any]
        if report_path.exists():
            preparation = {
                "paper_id": paper_id,
                "original_pdf": paper_dir / "original.pdf",
                "pages": _load_json(paper_dir / "pages.json").get("pages", []) if (paper_dir / "pages.json").exists() else [],
                "pages_path": paper_dir / "pages.json",
            }
            _update_job(job_path, job, phase="validate")
        else:
            _update_job(job_path, job, phase="preprocess")
            preparation = _prepare_pdf(pdf_path, paper_dir, config)
            _event(run_dir, "preprocess_succeeded", pages=len(preparation.get("pages", [])))
            _raise_if_interrupted(stop_event)
            # Checked here, not at send time: the text exists only after
            # preprocessing, and an over-long one must fail before the first
            # paid request rather than as an opaque provider error.
            _check_text_limit(preparation, config)
            _update_job(job_path, job, phase="read")
            reader_result: Any = None
            last_error: BaseException | None = None
            total_attempts = options.max_retries + 1
            for attempt in range(1, total_attempts + 1):
                _raise_if_interrupted(stop_event)
                if int(job.get("requests", 0)) >= options.max_requests_per_paper:
                    raise BatchPermanentError(
                        "max_requests_per_paper reached before reader call"
                    )
                job["attempts"] = int(job.get("attempts", 0)) + 1
                _update_job(job_path, job)
                try:
                    reader_result = _invoke_reader(
                        pdf_path,
                        paper_dir,
                        run_dir,
                        preparation,
                        config,
                        options.max_requests_per_paper - int(job.get("requests", 0)),
                    )
                    request_count = _requests_made(reader_result)
                    job["requests"] = int(job.get("requests", 0)) + request_count
                    _update_job(job_path, job)
                    _event(run_dir, "reader_succeeded", attempt=attempt, requests_made=request_count)
                    _raise_if_interrupted(stop_event)
                    break
                except BatchInterruptedError:
                    raise
                except Exception as exc:  # noqa: BLE001 - persist the exact failure for resume
                    last_error = exc
                    request_count = _requests_made(exc)
                    job["requests"] = int(job.get("requests", 0)) + request_count
                    _event(
                        run_dir,
                        "reader_failed",
                        attempt=attempt,
                        error=str(exc),
                        retryable=_retryable(exc),
                        requests_made=request_count,
                    )
                    _raise_if_interrupted(stop_event)
                    if attempt >= total_attempts or not _retryable(exc):
                        raise
                    _update_job(job_path, job, error=f"attempt {attempt}: {exc}")
                    time.sleep(min(60.0, options.retry_backoff_s * (2 ** (attempt - 1))))
            if reader_result is None and last_error is not None:
                raise last_error
            report, reader_metadata = _extract_report(reader_result, run_dir)
            _write_supplemental(run_dir, reader_metadata.get("supplemental_images"))
            _atomic_json(report_path, report)
            _update_job(job_path, job, phase="validate")

        _raise_if_interrupted(stop_event)
        report = _load_json(report_path)
        if not isinstance(report, Mapping):
            raise BatchError("report.json root must be an object")
        # Cross-reference cleanup is deterministic rather than a model call.
        # Apply it again on resume so reports written just before this guard
        # was introduced can continue at validation without paying to reread
        # the paper.
        from .reader import normalize_report_schema_shape, reconcile_cross_references

        report = dict(report)
        before_reconcile = _stable_json(report)
        reconcile_cross_references(report)
        after_reconcile = _stable_json(report)
        # Mechanical schema coercions (drop forbidden keys, fill empty-string
        # required fields) run here as well as in the reader so a resumed
        # report.json from an older run can still reach render without a paid
        # reread.
        schema_path = options.config_path.parent / "schemas" / "report.schema.json"
        shape_fixes: list[str] = []
        try:
            shape_schema = _load_json(schema_path) if schema_path.exists() else None
        except Exception:  # noqa: BLE001 - unusable schema must not block validate
            shape_schema = None
        if isinstance(shape_schema, Mapping):
            shape_fixes = normalize_report_schema_shape(report, shape_schema)
        if after_reconcile != before_reconcile or shape_fixes:
            _atomic_json(report_path, report)
            if after_reconcile != before_reconcile:
                _event(run_dir, "cross_references_reconciled")
            if shape_fixes:
                _event(
                    run_dir,
                    "schema_shape_normalized",
                    fixes=shape_fixes,
                    source="batch_validate",
                )
        supplemental = _read_supplemental(run_dir)
        issues, validation_warnings = _validate_report(
            report, options.config_path, paper_dir / "pages.json", supplemental
        )
        if issues:
            _update_job(job_path, job, status="failed", phase="validate", validation_errors=issues, error="report validation failed")
            _event(run_dir, "validation_failed", errors=issues)
            return PaperResult(paper_id, pdf_path, "failed", title=str(report.get("title", "")), report_path=report_path, job_path=job_path, error="report validation failed")

        _raise_if_interrupted(stop_event)
        pages_path = paper_dir / "pages.json"
        _update_job(job_path, job, phase="render")
        unresolved = (
            [str(item) for item in report.get("unresolved_items", [])]
            if isinstance(report.get("unresolved_items"), list)
            else []
        )
        # Requests that were never executed are listed by the reader itself
        # (PENDING_REQUEST_PREFIX), so report.json stays the single source of
        # the review reasons that this status is derived from.

        output_options = _render_options(config)
        render_kwargs = {
            "pages_json": pages_path,
            "supplemental_images": supplemental,
            **output_options,
        }
        # The status is derived from the report alone, so the header, the
        # pending list below it, the job record and the daily index always
        # agree.  Render warnings are diagnostics about the drawing itself
        # (a Mermaid-safe node ID, for instance) and are only recorded.
        status = "needs_review" if unresolved else "done"
        # A report that came out of a text-only repair round was restructured --
        # or, when the repair payload carried no substance, written -- without
        # any page image in front of the model.  That provenance has to travel
        # with the report, because nothing in the report itself shows it.
        repaired = _format_repaired(reader_metadata)
        metadata = {
            "status": status,
            "run_id": run_fingerprint,
            "input_coverage": _coverage_text(paper_dir),
            "format_repaired": repaired,
        }
        run_render = render_report(
            report, run_dir / "final.md", metadata=metadata, **render_kwargs
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_render = render_report(
            report, output_path, metadata=metadata, **render_kwargs
        )
        # Both copies render the same report, so their warnings are identical.
        warnings = list(run_render.warnings)
        title = str(report.get("title", ""))
        summary = str(report.get("short_summary", ""))
        _update_job(
            job_path,
            job,
            status=status,
            phase="complete",
            title=title,
            summary=summary,
            unresolved_items=unresolved,
            warnings=warnings,
            validation_warnings=validation_warnings,
            validation_errors=[],
            output_coverage=metadata["input_coverage"],
            format_repaired=repaired,
            completed_at=_now(),
            # A near miss (a retried attempt that failed before the one that
            # worked) stays in the event log; it is not an open error.
            error=None,
        )
        _event(
            run_dir,
            "job_completed",
            status=status,
            linked_images=output_render.linked_images,
            supplemental_images=len(supplemental),
        )
        return PaperResult(paper_id, pdf_path, status, title, summary, report_path, output_path, job_path, unresolved)
    except BatchInterruptedError as exc:
        _update_job(
            job_path,
            job,
            status="interrupted",
            phase=job.get("phase", "unknown"),
            error=str(exc),
            interrupted_at=_now(),
        )
        _event(run_dir, "job_interrupted", error=str(exc))
        return PaperResult(
            paper_id,
            pdf_path,
            "interrupted",
            report_path=report_path if report_path.exists() else None,
            job_path=job_path,
            error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 - batch must continue with other papers
        error = f"{type(exc).__name__}: {exc}"
        updates: dict[str, Any] = {
            "status": "failed",
            "phase": job.get("phase", "unknown"),
            "error": error,
            "traceback": traceback.format_exc(),
        }
        if _input_over_limit(exc):
            # The plan requires a distinct, visible reason instead of a silent
            # page drop or a truncated reading presented as complete.
            updates["failure_reason"] = "input_over_limit"
            updates["error"] = f"input_over_limit: {error}"
        _update_job(job_path, job, **updates)
        _event(run_dir, "job_failed", error=error)
        return PaperResult(paper_id, pdf_path, "failed", report_path=report_path if report_path.exists() else None, job_path=job_path, error=str(updates["error"]))


def _batch_date(input_path: Path, explicit: str | None = None) -> str:
    if explicit:
        if not DATE_LABEL_RE.fullmatch(explicit):
            raise BatchError("--date must use YYYY-MM-DD or YYYY-MM-DD-<label>")
        return explicit
    candidates = [input_path.name]
    if input_path.is_file():
        candidates.append(input_path.parent.name)
    for value in candidates:
        if DATE_RE.fullmatch(value):
            return value
    return datetime.now().astimezone().strftime("%Y-%m-%d")


def _write_index(results: Sequence[PaperResult], path: Path) -> None:
    lines = ["# Paper Reading Batch", "", f"Generated: {_now()}", "", "| Paper | Status | Summary | Review items |", "| --- | --- | --- | --- |"]
    for result in sorted(results, key=lambda item: item.paper_id.lower()):
        paper = result.paper_id
        if result.output_path is not None and result.output_path.exists():
            try:
                link = Path(os.path.relpath(result.output_path, path.parent)).as_posix()
                paper = f"[{paper}](<{link}>)"
            except ValueError:
                pass
        review = "；".join(result.unresolved_items or []) or "—"
        lines.append("| " + " | ".join([
            paper.replace("|", "\\|"),
            result.status,
            str(result.summary).replace("|", "\\|").replace("\n", "<br>"),
            review.replace("|", "\\|").replace("\n", "<br>"),
        ]) + " |")
    content = "\n".join(lines).rstrip() + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def run_batch(input_path: str | os.PathLike[str], options: BatchOptions) -> list[PaperResult]:
    source = Path(input_path).expanduser().resolve()
    pdfs = _pdfs(source)
    config = _load_yaml(options.config_path)
    config_fingerprint = _config_fingerprint(config, options.config_path)
    if options.concurrency < 1:
        raise BatchError("concurrency must be at least 1")
    stop_event = threading.Event()
    # One nonce is shared by the whole forced batch.  The PDF digest still
    # makes each paper's run ID distinct, while a new invocation always gets a
    # new run directory and therefore cannot see an existing report.json.
    run_nonce = uuid4().hex if options.force_reread else None
    worker = lambda pdf: process_one(
        pdf,
        options,
        config,
        config_fingerprint,
        stop_event,
        run_nonce,
    )
    results: list[PaperResult] = []
    try:
        if options.concurrency == 1 or len(pdfs) <= 1:
            results = [worker(pdf) for pdf in pdfs]
        else:
            executor = ThreadPoolExecutor(
                max_workers=options.concurrency, thread_name_prefix="paper"
            )
            futures: dict[Future[PaperResult], Path] = {executor.submit(worker, pdf): pdf for pdf in pdfs}
            try:
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as exc:  # defensive: process_one normally captures errors
                        pdf = futures[future]
                        results.append(PaperResult(_paper_id(pdf), pdf, "failed", error=f"{type(exc).__name__}: {exc}"))
            except KeyboardInterrupt:
                stop_event.set()
                raise
            finally:
                executor.shutdown(wait=not stop_event.is_set(), cancel_futures=stop_event.is_set())
    except KeyboardInterrupt:
        stop_event.set()
        raise
    index_path = options.output_root / options.batch_date / "index.md"
    _write_index(results, index_path)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process one paper or a directory of PDFs")
    parser.add_argument("input", type=Path, help="PDF file or directory containing PDFs")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--workspace", type=Path, default=Path("workspace"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--max-requests-per-paper", type=int)
    parser.add_argument("--backoff", type=float, default=2.0, help="initial retry backoff in seconds")
    parser.add_argument("--date", help="output batch directory name (YYYY-MM-DD or YYYY-MM-DD-<label>)")
    parser.add_argument("--resume", dest="resume", action="store_true", default=None)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument(
        "--force",
        "--force-reread",
        dest="force_reread",
        action="store_true",
        help="create a fresh run and call the model even when report.json already exists",
    )
    return parser


def _check_unsupported_config(config: Mapping[str, Any]) -> None:
    """Refuse configuration that V1 silently cannot honour.

    The plan states that a setting which exists must take effect.  Anything
    outside the implemented V1 behaviour fails loudly here instead of being
    read, ignored, and mistaken for an applied policy.
    """

    language = _deep_get(config, "output", "language", default="zh-CN")
    if str(language) != "zh-CN":
        raise BatchError(
            f"output.language={language!r} is not implemented; V1 writes zh-CN reports only"
        )
    # The input mode decides whether the reading call uploads every page image
    # or sends the extracted text, so an unknown value or a contradictory
    # combination has to stop the run instead of being read as a default: the
    # difference is the entire cost of the batch.
    mode = str(_deep_get(config, "input", "mode", default="hybrid_text_visual")).strip()
    if mode not in INPUT_MODES:
        raise BatchError(f"input.mode must be one of {', '.join(INPUT_MODES)}; got {mode!r}")
    include_all = _deep_get(config, "input", "include_all_pdf_pages", default=False)
    if include_all not in (True, False):
        raise BatchError(f"input.include_all_pdf_pages must be true or false; got {include_all!r}")
    if mode == "hybrid_text_visual":
        if include_all is True:
            raise BatchError(
                "input.include_all_pdf_pages=true 与 input.mode=hybrid_text_visual 冲突："
                "hybrid 模式不下发整页图片；请改为 false，或把 mode 切到 full_page_images 做 A/B"
            )
        if _deep_get(config, "input", "extract_text", default=True) is not True:
            raise BatchError(
                "input.mode=hybrid_text_visual 需要 input.extract_text=true，"
                "否则主阅读请求既没有图片也没有全文"
            )
    elif include_all is not True:
        raise BatchError(
            "input.mode=full_page_images 需要 input.include_all_pdf_pages=true；"
            "该模式是整个 A/B 基线，不允许静默地少发页面"
        )
    over_limit = _deep_get(config, "input", "on_input_over_limit", default="fail_with_reason")
    if str(over_limit) != "fail_with_reason":
        raise BatchError(
            f"input.on_input_over_limit={over_limit!r} is not implemented; V1 only supports "
            "'fail_with_reason'"
        )
    if _deep_get(config, "verification", "automated_verifier", default=False) is not False:
        raise BatchError(
            "verification.automated_verifier=true is not implemented in V1; enable it only after "
            "calibration shows a reproducible net benefit"
        )
    # Validated here as well as at use: a malformed ceiling should stop the run
    # once, not fail every paper in it with the same message.
    _max_pages(config)
    _max_text_chars(config)


def options_from_args(args: argparse.Namespace) -> BatchOptions:
    config = _load_yaml(args.config)
    _check_unsupported_config(config)
    concurrency = args.concurrency if args.concurrency is not None else int(_deep_get(config, "batch", "concurrency", default=2))
    max_retries = args.max_retries if args.max_retries is not None else int(_deep_get(config, "batch", "max_retries_per_call", default=2))
    max_requests = args.max_requests_per_paper if args.max_requests_per_paper is not None else int(_deep_get(config, "batch", "max_requests_per_paper", default=6))
    resume = args.resume if args.resume is not None else bool(_deep_get(config, "batch", "resume", default=True))
    if concurrency < 1 or max_retries < 0 or max_requests < 1:
        raise BatchError("concurrency must be >= 1, max retries >= 0, and max requests >= 1")
    return BatchOptions(
        config_path=args.config.expanduser().resolve(),
        workspace_root=args.workspace.expanduser().resolve(),
        output_root=args.output.expanduser().resolve(),
        concurrency=concurrency,
        max_retries=max_retries,
        max_requests_per_paper=max_requests,
        resume=resume,
        retry_backoff_s=max(0.0, float(args.backoff)),
        batch_date=_batch_date(args.input.expanduser().resolve(), args.date),
        force_reread=bool(args.force_reread),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        options = options_from_args(args)
        results = run_batch(args.input, options)
    except KeyboardInterrupt:
        print("interrupted by user", file=sys.stderr)
        return 130
    except BatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    done = sum(result.status == "done" for result in results)
    review = sum(result.status == "needs_review" for result in results)
    failed = sum(result.status == "failed" for result in results)
    for result in sorted(results, key=lambda item: item.paper_id.lower()):
        message = f"{result.paper_id}: {result.status}"
        if result.error:
            message += f" ({result.error})"
        print(message)
    print(f"completed={done} needs_review={review} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
