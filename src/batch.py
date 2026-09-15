"""Batch orchestration for the paper-reading pipeline.

The batch layer owns durable state and local artifacts.  PDF preparation and
model reading live in :mod:`src.preprocess` and :mod:`src.reader`; keeping the
orchestration here means a failed Markdown render can be resumed without
repeating a paid model request.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import sys
import tempfile
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .render_report import render_report


DATE_RE = re.compile(r"^20\d{2}-\d{2}-\d{2}$")
REPORT_KEYS = {
    "schema_version",
    "paper_id",
    "title",
    "short_summary",
    "problem",
    "related_work",
    "method",
    "experiments",
    "authors_limitations",
    "reader_analysis",
    "full_summary",
    "claims",
    "evidence",
    "visual_requests",
    "unresolved_items",
}


class BatchError(RuntimeError):
    """An error that should be recorded in job.json."""


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


def _config_fingerprint(config: Mapping[str, Any], config_path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(_stable_json(config).encode("utf-8"))
    if config_path.exists():
        digest.update(config_path.read_bytes())
    schema_path = config_path.parent / "schemas" / "report.schema.json"
    if schema_path.exists():
        digest.update(schema_path.read_bytes())
    for prompt in ("reader.md", "refine.md"):
        prompt_path = config_path.parent / "prompts" / prompt
        if prompt_path.exists():
            digest.update(prompt_path.read_bytes())
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
    raise BatchError("src.reader exposes none of read_paper/run_reader/run/process_paper/read")


def _invoke_reader(
    pdf_path: Path,
    paper_dir: Path,
    run_dir: Path,
    preparation: Mapping[str, Any],
    config: Mapping[str, Any],
    max_requests: int,
) -> Any:
    function = _reader_callable()
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
        "pages": preparation.get("pages", []),
        "pages_json": preparation.get("pages_path", paper_dir / "pages.json"),
        "config": config,
        "max_requests": max_requests,
        "max_requests_per_paper": max_requests,
        "paper_id": preparation.get("paper_id", paper_dir.name),
    }
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
        metadata = getattr(value, "metadata", {}) if isinstance(getattr(value, "metadata", {}), Mapping) else {}
    elif isinstance(value, (str, os.PathLike)):
        candidate = _load_json(Path(value))
    if isinstance(candidate, Mapping) and REPORT_KEYS.intersection(candidate.keys()):
        return dict(candidate), metadata
    report_path = run_dir / "report.json"
    if report_path.exists():
        loaded = _load_json(report_path)
        if isinstance(loaded, Mapping):
            return dict(loaded), metadata
    raise BatchError("reader did not return a report object or create run/report.json")


def _validate_report(report: Mapping[str, Any], config_path: Path) -> list[str]:
    """Use src.validate when present, with a schema-only fallback."""

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
            }
            try:
                result = _call_with_supported_kwargs(function, kwargs)
            except TypeError:
                continue
            if result is True or result is None:
                return []
            if isinstance(result, Mapping):
                errors = result.get("errors", result.get("issues", []))
                if not errors and result.get("valid") is True:
                    return []
                return [str(item) for item in (errors if isinstance(errors, list) else [errors])]
            if isinstance(result, (list, tuple, set)):
                return [str(item) for item in result]
            if isinstance(result, str):
                return [result]
            return []
    schema_path = config_path.parent / "schemas" / "report.schema.json"
    try:
        import jsonschema  # type: ignore

        schema = _load_json(schema_path)
        validator = jsonschema.Draft202012Validator(schema)
        return [error.message for error in sorted(validator.iter_errors(report), key=lambda error: list(error.path))]
    except (ImportError, FileNotFoundError, json.JSONDecodeError):
        required = REPORT_KEYS - set(report)
        return [f"missing required report fields: {', '.join(sorted(required))}"] if required else []


def _retryable(error: BaseException) -> bool:
    text = str(error).lower()
    permanent_markers = (
        "401",
        "403",
        "authentication",
        "api key",
        "input_over_limit",
        "input over limit",
        "unsupported parameter",
        "invalid parameter",
        "requires unsupported parameters",
    )
    return not any(marker in text for marker in permanent_markers)


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


def process_one(pdf_path: Path, options: BatchOptions, config: Mapping[str, Any], config_fingerprint: str) -> PaperResult:
    paper_id = _paper_id(pdf_path)
    paper_dir = options.workspace_root / paper_id
    paper_dir.mkdir(parents=True, exist_ok=True)
    try:
        pdf_digest = _file_digest(pdf_path)
    except OSError as exc:
        return PaperResult(paper_id, pdf_path, "failed", error=str(exc))
    run_fingerprint = hashlib.sha256(
        f"{pdf_digest}:{config_fingerprint}".encode("utf-8")
    ).hexdigest()[:16]
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
        "requests": int(job.get("requests", 0) or 0),
        "attempts": int(job.get("attempts", 0) or 0),
        "started_at": job.get("started_at", _now()),
        "error": None,
        "report_path": str(report_path),
        "output_path": str(output_path),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    _update_job(job_path, job)
    _event(run_dir, "job_started", paper_id=paper_id, run_id=run_fingerprint)

    try:
        # A report left by an interrupted run is enough to resume from local
        # validation/rendering; no paid reader call is made in that case.
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
            _update_job(job_path, job, phase="read")
            reader_result: Any = None
            last_error: BaseException | None = None
            total_attempts = options.max_retries + 1
            for attempt in range(1, total_attempts + 1):
                if int(job.get("requests", 0)) >= options.max_requests_per_paper:
                    raise BatchError("max_requests_per_paper reached before reader call")
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
                    metadata = reader_result if isinstance(reader_result, Mapping) else {}
                    request_count = metadata.get("requests_made", metadata.get("request_count", 1)) if isinstance(metadata, Mapping) else 1
                    try:
                        request_count = max(1, int(request_count))
                    except (TypeError, ValueError):
                        request_count = 1
                    job["requests"] = int(job.get("requests", 0)) + request_count
                    _event(run_dir, "reader_succeeded", attempt=attempt, requests_made=request_count)
                    break
                except Exception as exc:  # noqa: BLE001 - persist the exact failure for resume
                    last_error = exc
                    job["requests"] = int(job.get("requests", 0)) + 1
                    _event(run_dir, "reader_failed", attempt=attempt, error=str(exc), retryable=_retryable(exc))
                    if attempt >= total_attempts or not _retryable(exc):
                        raise
                    _update_job(job_path, job, error=f"attempt {attempt}: {exc}")
                    time.sleep(min(60.0, options.retry_backoff_s * (2 ** (attempt - 1))))
            if reader_result is None and last_error is not None:
                raise last_error
            report, reader_metadata = _extract_report(reader_result, run_dir)
            _atomic_json(report_path, report)
            _update_job(job_path, job, phase="validate")

        report = _load_json(report_path)
        if not isinstance(report, Mapping):
            raise BatchError("report.json root must be an object")
        issues = _validate_report(report, options.config_path)
        if issues:
            _update_job(job_path, job, status="failed", phase="validate", validation_errors=issues, error="report validation failed")
            _event(run_dir, "validation_failed", errors=issues)
            return PaperResult(paper_id, pdf_path, "failed", title=str(report.get("title", "")), report_path=report_path, job_path=job_path, error="report validation failed")

        pages_path = paper_dir / "pages.json"
        render_metadata = {"status": "running", "run_id": run_fingerprint}
        _update_job(job_path, job, phase="render")
        run_render = render_report(report, run_dir / "final.md", pages_json=pages_path, metadata=render_metadata)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_render = render_report(report, output_path, pages_json=pages_path, metadata={"status": "running", "run_id": run_fingerprint})
        unresolved = [str(item) for item in report.get("unresolved_items", [])] if isinstance(report.get("unresolved_items"), list) else []
        if report.get("visual_requests"):
            unresolved = unresolved or ["报告请求了局部补看，但本批次尚未执行补看"]
        status = "needs_review" if unresolved or run_render.warnings or output_render.warnings else "done"
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
            warnings=run_render.warnings + output_render.warnings,
            validation_errors=[],
            completed_at=_now(),
        )
        _event(run_dir, "job_completed", status=status, linked_images=run_render.linked_images)
        return PaperResult(paper_id, pdf_path, status, title, summary, report_path, output_path, job_path, unresolved)
    except Exception as exc:  # noqa: BLE001 - batch must continue with other papers
        error = f"{type(exc).__name__}: {exc}"
        _update_job(job_path, job, status="failed", phase=job.get("phase", "unknown"), error=error, traceback=traceback.format_exc())
        _event(run_dir, "job_failed", error=error)
        return PaperResult(paper_id, pdf_path, "failed", report_path=report_path if report_path.exists() else None, job_path=job_path, error=error)


def _batch_date(input_path: Path, explicit: str | None = None) -> str:
    if explicit:
        if not DATE_RE.fullmatch(explicit):
            raise BatchError("--date must use YYYY-MM-DD")
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
    worker = lambda pdf: process_one(pdf, options, config, config_fingerprint)
    results: list[PaperResult] = []
    if options.concurrency == 1 or len(pdfs) <= 1:
        results = [worker(pdf) for pdf in pdfs]
    else:
        with ThreadPoolExecutor(max_workers=options.concurrency, thread_name_prefix="paper") as executor:
            futures: dict[Future[PaperResult], Path] = {executor.submit(worker, pdf): pdf for pdf in pdfs}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:  # defensive: process_one normally captures errors
                    pdf = futures[future]
                    results.append(PaperResult(_paper_id(pdf), pdf, "failed", error=f"{type(exc).__name__}: {exc}"))
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
    parser.add_argument("--date", help="output batch date (YYYY-MM-DD)")
    parser.add_argument("--resume", dest="resume", action="store_true", default=None)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser


def options_from_args(args: argparse.Namespace) -> BatchOptions:
    config = _load_yaml(args.config)
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
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        options = options_from_args(args)
        results = run_batch(args.input, options)
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

