"""Reader client for the paper-reading pipeline.

The module deliberately contains only the model-facing part of the pipeline.
PDF rendering and report validation live in the neighbouring modules.  A
``Reader`` accepts already rendered page images, sends all pages in one
vision request, and can make one bounded follow-up request for images asked
for in ``visual_requests``.

The OpenAI import is lazy.  This keeps preprocessing and local validation
usable on machines where the optional API dependency has not been installed,
while still producing an actionable error when an API client is needed.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Protocol, Sequence
from uuid import uuid4


DEFAULT_MODEL = "qwen3.8-flash"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "report.schema.json"


def load_env_file(path: str | os.PathLike[str] | None = None) -> Path | None:
    """Load a small project-local ``.env`` file without overriding exports.

    A dependency-free loader keeps API keys out of ``config.yaml`` while
    making the CLI usable when a parent shell does not propagate its
    environment into the tool process.  Existing environment variables win.
    Only simple ``KEY=value`` and ``export KEY=value`` lines are accepted.
    """

    candidate = Path(path) if path is not None else Path.cwd() / ".env"
    if not candidate.is_file():
        return None
    try:
        lines = candidate.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ReaderConfigError(f"Could not read environment file {candidate}: {exc}") from exc
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        name, raw_value = line.split("=", 1)
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or name in os.environ:
            continue
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[name] = value
    return candidate


class ReaderError(RuntimeError):
    """Base exception for errors raised by this module."""


class ReaderDependencyError(ReaderError):
    """A runtime dependency required for the selected operation is missing."""


class ReaderConfigError(ReaderError):
    """Configuration is missing or invalid."""


class ReaderInputError(ReaderError):
    """Page image input or a visual request is invalid."""


class ReaderAPIError(ReaderError):
    """The compatible chat-completions endpoint failed."""


class ReaderResponseError(ReaderError):
    """The endpoint returned an unusable or truncated response."""


class EventRecorder(Protocol):
    """Small protocol used by batch.py to persist call events immediately."""

    def record(self, event: Mapping[str, Any]) -> None:
        ...


class JsonlEventRecorder:
    """Append reader events to an ``events.jsonl`` file.

    The recorder writes one JSON object per call event and flushes after every
    event.  It intentionally never receives image bytes or API credentials.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(event), ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()

    # ``record_event`` is useful for callers which use that naming convention.
    record_event = record


@dataclass(frozen=True)
class ReaderConfig:
    """Configuration translated from the project's nested ``config.yaml``."""

    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key_env: str = "DASHSCOPE_API_KEY"
    thinking: bool | None = True
    response_mode: str = "json_object"
    request_timeout_s: float = 600.0
    render_dpi: int = 160
    crop_dpi: int = 300
    max_visual_rounds: int = 1
    max_crop_images: int = 4
    max_retries_per_call: int = 2
    max_requests_per_paper: int = 6

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ReaderConfig":
        """Build a config from the same nested mapping used by ``config.yaml``.

        Unknown keys are intentionally ignored.  The YAML file is application
        configuration rather than a direct passthrough of provider options.
        """

        load_env_file()
        value = value or {}
        model_cfg = _mapping(value.get("model"))
        input_cfg = _mapping(value.get("input"))
        refinement_cfg = _mapping(value.get("refinement"))
        batch_cfg = _mapping(value.get("batch"))

        model = str(model_cfg.get("name", DEFAULT_MODEL)).strip()
        if not model:
            raise ReaderConfigError("model.name must be a non-empty model identifier")

        base_env = str(model_cfg.get("base_url_env", "DASHSCOPE_BASE_URL"))
        base_url = os.environ.get(base_env) or str(
            model_cfg.get("default_base_url", model_cfg.get("base_url", DEFAULT_BASE_URL))
        ).strip()
        if not base_url:
            raise ReaderConfigError(
                f"No API base URL configured; set {base_env} or model.default_base_url"
            )

        thinking = model_cfg.get("thinking", True)
        if thinking is not None and not isinstance(thinking, bool):
            raise ReaderConfigError("model.thinking must be true, false, or null")

        return cls(
            model=model,
            base_url=base_url,
            api_key_env=str(model_cfg.get("api_key_env", "DASHSCOPE_API_KEY")),
            thinking=thinking,
            response_mode=str(model_cfg.get("response_mode", "json_object")),
            request_timeout_s=_positive_float(
                model_cfg.get("request_timeout_s", 600), "model.request_timeout_s"
            ),
            render_dpi=_positive_int(input_cfg.get("render_dpi", 160), "input.render_dpi"),
            crop_dpi=_positive_int(refinement_cfg.get("crop_dpi", 300), "refinement.crop_dpi"),
            max_visual_rounds=_nonnegative_int(
                refinement_cfg.get("max_visual_rounds", 1), "refinement.max_visual_rounds"
            ),
            max_crop_images=_nonnegative_int(
                refinement_cfg.get("max_crop_images", 4), "refinement.max_crop_images"
            ),
            max_retries_per_call=_nonnegative_int(
                batch_cfg.get("max_retries_per_call", 2), "batch.max_retries_per_call"
            ),
            max_requests_per_paper=_positive_int(
                batch_cfg.get("max_requests_per_paper", 6), "batch.max_requests_per_paper"
            ),
        )

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str] = "config.yaml") -> "ReaderConfig":
        """Load a project YAML config with a clear dependency error."""

        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ReaderDependencyError(
                "PyYAML is required to read config.yaml. Install project dependencies "
                "with `python -m pip install -r requirements.txt`."
            ) from exc

        config_path = Path(path)
        if not config_path.exists():
            raise ReaderConfigError(f"Configuration file does not exist: {config_path}")
        load_env_file(config_path.parent / ".env")
        try:
            with config_path.open("r", encoding="utf-8") as stream:
                loaded = yaml.safe_load(stream) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ReaderConfigError(f"Could not read configuration {config_path}: {exc}") from exc
        if not isinstance(loaded, Mapping):
            raise ReaderConfigError(f"Configuration root must be a mapping: {config_path}")
        return cls.from_mapping(loaded)


@dataclass(frozen=True)
class PageImage:
    """A rendered PDF page used as a vision input."""

    pdf_page: int
    path: Path
    image_id: str | None = None
    width: int | None = None
    height: int | None = None

    def stable_id(self) -> str:
        return self.image_id or f"page-{self.pdf_page:03d}"


@dataclass
class CallRecord:
    """Metadata from one logical API call (including all retry attempts)."""

    call_type: str
    request_id: str
    attempts: int = 0
    usage: dict[str, Any] = field(default_factory=dict)
    response_id: str | None = None
    finish_reason: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    duration_s: float | None = None
    error_type: str | None = None
    error: str | None = None
    provided_pages: list[int] = field(default_factory=list)
    provided_images: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "call_type": self.call_type,
            "request_id": self.request_id,
            "attempts": self.attempts,
            "usage": self.usage,
            "response_id": self.response_id,
            "finish_reason": self.finish_reason,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_s": self.duration_s,
            "error_type": self.error_type,
            "error": self.error,
            "provided_pages": self.provided_pages,
            "provided_images": self.provided_images,
        }


@dataclass
class ReadResult:
    """Result of a main call and optional one-round visual refinement."""

    report: dict[str, Any]
    calls: list[CallRecord] = field(default_factory=list)
    raw_responses: list[Any] = field(default_factory=list)
    unresolved_items: list[str] = field(default_factory=list)

    @property
    def visual_requests(self) -> list[dict[str, Any]]:
        requests = self.report.get("visual_requests", [])
        return requests if isinstance(requests, list) else []

    def as_dict(self) -> dict[str, Any]:
        return {
            "report": self.report,
            "calls": [call.as_dict() for call in self.calls],
            "unresolved_items": list(self.unresolved_items),
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


@dataclass
class _ResponseResult:
    report: dict[str, Any]
    response: Any
    record: CallRecord


Renderer = Callable[..., Any]


DEFAULT_READER_INSTRUCTIONS = """You are a careful scientific paper reader. Read every supplied PDF page in order and return exactly one JSON object matching the supplied report schema.

Write the report in Simplified Chinese. Answer Q1 through Q6 in the schema. Distinguish author claims, reader inferences, and your own analysis. Every important fact, method node, method edge, and reported result must cite evidence IDs that point to a real supplied PDF page. Never invent a page, quote, number, baseline, method step, or relation. Use an empty string or an explicit unresolved item when the page is unreadable or the provided material does not contain the information; use “Not reported” only when the supplied material clearly covers the relevant section and the authors do not report it.

Keep table numbers with their metric, dataset, baseline or method, setting, unit, and denominator. For method edges, say whether the relation is data flow, control flow, or dependency and set confirmed=false if the supplied pages do not support it. If a formula, table, or figure cannot be read reliably, add at most four focused visual_requests with a 1-based PDF page and an optional normalized crop [x0, y0, x1, y1]. Do not ask for pages that are already clear. Return no Markdown fences and no commentary outside the JSON object."""


class Reader:
    """Call Qwen3.8-Flash through an OpenAI-compatible endpoint."""

    def __init__(
        self,
        config: ReaderConfig | Mapping[str, Any] | str | os.PathLike[str] | None = None,
        *,
        client: Any | None = None,
        event_recorder: EventRecorder | Callable[[Mapping[str, Any]], None] | None = None,
        schema_path: str | os.PathLike[str] | None = None,
        prompt_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.config = _coerce_config(config)
        self._client = client
        self.event_recorder = event_recorder
        self.schema_path = Path(schema_path) if schema_path else DEFAULT_SCHEMA_PATH
        self.prompt_path = Path(prompt_path) if prompt_path else None
        self._request_count = 0

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._make_client()
        return self._client

    def _make_client(self) -> Any:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ReaderDependencyError(
                "The OpenAI Python SDK is required for model calls. Install project "
                "dependencies with `python -m pip install -r requirements.txt`."
            ) from exc

        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise ReaderConfigError(
                f"Missing API key: set the {self.config.api_key_env} environment variable "
                "before starting a reader run."
            )
        try:
            return OpenAI(
                api_key=api_key,
                base_url=self.config.base_url,
                timeout=self.config.request_timeout_s,
            )
        except Exception as exc:
            raise ReaderConfigError(f"Could not create OpenAI-compatible client: {exc}") from exc

    def read(
        self,
        paper_id: str,
        pages: Sequence[PageImage | Mapping[str, Any] | str | os.PathLike[str]],
        *,
        title: str = "",
        prompt: str | None = None,
        schema: Mapping[str, Any] | None = None,
        renderer: Renderer | None = None,
        original_pages: Sequence[PageImage | Mapping[str, Any] | str | os.PathLike[str]] | None = None,
    ) -> ReadResult:
        """Read all pages once and optionally perform one visual follow-up.

        ``renderer`` is intentionally injected because page rendering belongs
        to ``preprocess.py``.  It may accept ``(pdf_page, crop, dpi)`` or the
        equivalent keyword arguments and may return a path, ``PageImage``, or
        a mapping containing ``path`` and ``pdf_page``.
        """

        normalized_pages = normalize_pages(pages)
        if not normalized_pages:
            raise ReaderInputError("At least one rendered PDF page is required")
        self._request_count = 0
        first = self._call_report(
            call_type="read",
            paper_id=paper_id,
            pages=normalized_pages,
            title=title,
            prompt=prompt,
            schema=schema,
        )
        result = ReadResult(
            report=first.report,
            calls=[first.record],
            raw_responses=[first.response],
        )

        requests = _visual_requests(first.report)
        if not requests or self.config.max_visual_rounds < 1:
            return result

        available = normalize_pages(original_pages or normalized_pages)
        refined, unresolved = self._refine_once(
            paper_id=paper_id,
            candidate=first.report,
            requests=requests,
            available_pages=available,
            renderer=renderer,
            title=title,
            prompt=prompt,
            schema=schema,
        )
        result.unresolved_items.extend(unresolved)
        if refined is None:
            _append_unresolved_to_report(result.report, unresolved)
            return result
        result.report = refined.report
        result.calls.append(refined.record)
        result.raw_responses.append(refined.response)
        remaining = _visual_requests(result.report)
        if remaining:
            unresolved_remaining = [
                f"补看后模型仍请求 PDF 第 {item.get('pdf_page', '?')} 页：{item.get('reason', '')}".strip()
                for item in remaining
            ]
            result.unresolved_items.extend(unresolved_remaining)
            _append_unresolved_to_report(result.report, unresolved_remaining)
        return result

    # Explicit aliases make the small interface convenient for batch.py and
    # for scripts written before the ReadResult wrapper was introduced.
    read_paper = read
    read_document = read

    def read_report(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Convenience wrapper returning only the final report JSON."""

        return self.read(*args, **kwargs).report

    def refine(
        self,
        paper_id: str,
        candidate: Mapping[str, Any],
        requests: Sequence[Mapping[str, Any]] | None = None,
        *,
        pages: Sequence[PageImage | Mapping[str, Any] | str | os.PathLike[str]] = (),
        renderer: Renderer | None = None,
        title: str = "",
        prompt: str | None = None,
        schema: Mapping[str, Any] | None = None,
    ) -> ReadResult:
        """Run a single explicit visual refinement round.

        This method is useful when batch.py has already persisted the first
        candidate and wants to schedule the follow-up separately.
        """

        if self.config.max_visual_rounds < 1:
            return ReadResult(report=dict(candidate))
        available = normalize_pages(pages)
        selected = list(requests) if requests is not None else _visual_requests(candidate)
        refined, unresolved = self._refine_once(
            paper_id=paper_id,
            candidate=dict(candidate),
            requests=selected,
            available_pages=available,
            renderer=renderer,
            title=title,
            prompt=prompt,
            schema=schema,
        )
        if refined is None:
            report = dict(candidate)
            _append_unresolved_to_report(report, unresolved)
            return ReadResult(report=report, unresolved_items=unresolved)
        result = ReadResult(
            report=refined.report,
            calls=[refined.record],
            raw_responses=[refined.response],
            unresolved_items=unresolved,
        )
        remaining = _visual_requests(result.report)
        if remaining:
            more = [
                f"补看后模型仍请求 PDF 第 {item.get('pdf_page', '?')} 页：{item.get('reason', '')}".strip()
                for item in remaining
            ]
            result.unresolved_items.extend(more)
            _append_unresolved_to_report(result.report, more)
        return result

    def _refine_once(
        self,
        *,
        paper_id: str,
        candidate: Mapping[str, Any],
        requests: Sequence[Mapping[str, Any]],
        available_pages: Sequence[PageImage],
        renderer: Renderer | None,
        title: str,
        prompt: str | None,
        schema: Mapping[str, Any] | None,
    ) -> tuple[_ResponseResult | None, list[str]]:
        by_page = {page.pdf_page: page for page in available_pages}
        supplemental: list[PageImage] = []
        unresolved: list[str] = []
        for request in requests:
            if len(supplemental) >= self.config.max_crop_images:
                unresolved.append(
                    f"超出补看预算（最多 {self.config.max_crop_images} 张）："
                    f"PDF 第 {request.get('pdf_page', '?')} 页"
                )
                continue
            valid, reason = validate_visual_request(request, set(by_page))
            if not valid:
                unresolved.append(reason)
                continue
            page_number = int(request["pdf_page"])
            crop = request.get("crop")
            if renderer is None:
                unresolved.append(
                    f"缺少补图渲染器，无法查看 PDF 第 {page_number} 页：{request.get('reason', '')}"
                )
                continue
            try:
                rendered = _call_renderer(
                    renderer,
                    pdf_page=page_number,
                    crop=crop,
                    dpi=self.config.crop_dpi,
                    source_page=by_page[page_number],
                )
                if rendered is None:
                    raise ReaderInputError("renderer returned no image")
                image = normalize_page(rendered, default_pdf_page=page_number)
                if image.pdf_page != page_number:
                    raise ReaderInputError(
                        f"renderer returned page {image.pdf_page}, expected page {page_number}"
                    )
                supplemental.append(image)
            except ReaderError as exc:
                unresolved.append(f"PDF 第 {page_number} 页补图失败：{exc}")
            except Exception as exc:
                unresolved.append(f"PDF 第 {page_number} 页补图失败：{type(exc).__name__}: {exc}")

        if not supplemental:
            return None, unresolved
        try:
            refined = self._call_report(
                call_type="visual_refinement",
                paper_id=paper_id,
                pages=supplemental,
                title=title,
                prompt=prompt,
                schema=schema,
                candidate=candidate,
                visual_reasons=[str(item.get("reason", "")) for item in requests],
                original_pages=[by_page[int(item["pdf_page"])] for item in requests if int(item.get("pdf_page", 0)) in by_page],
            )
            return refined, unresolved
        except ReaderError:
            raise

    def _call_report(
        self,
        *,
        call_type: str,
        paper_id: str,
        pages: Sequence[PageImage],
        title: str,
        prompt: str | None,
        schema: Mapping[str, Any] | None,
        candidate: Mapping[str, Any] | None = None,
        visual_reasons: Sequence[str] = (),
        original_pages: Sequence[PageImage] = (),
    ) -> _ResponseResult:
        if self._request_count >= self.config.max_requests_per_paper:
            raise ReaderAPIError(
                f"Request budget exhausted for {paper_id}: maximum "
                f"{self.config.max_requests_per_paper} requests per paper"
            )
        messages = self._build_messages(
            paper_id=paper_id,
            pages=pages,
            title=title,
            prompt=prompt,
            schema=schema,
            candidate=candidate,
            visual_reasons=visual_reasons,
            original_pages=original_pages,
        )
        request_id = uuid4().hex
        record = CallRecord(
            call_type=call_type,
            request_id=request_id,
            provided_pages=sorted({page.pdf_page for page in pages}),
            provided_images=[page.stable_id() for page in pages],
        )
        started = time.monotonic()
        record.started_at = _now()
        self._emit(
            {
                "event": "call_started",
                "call_type": call_type,
                "request_id": request_id,
                "paper_id": paper_id,
                "model": self.config.model,
                "pages": record.provided_pages,
                "images": record.provided_images,
                "started_at": record.started_at,
            }
        )

        last_error: Exception | None = None
        max_attempts = self.config.max_retries_per_call + 1
        for attempt in range(1, max_attempts + 1):
            record.attempts = attempt
            self._request_count += 1
            try:
                response = self._create_completion(messages)
                # Capture provider metadata even when the content is
                # truncated or malformed, so the event log explains why the
                # logical call was rejected.
                record.response_id = _response_attr(response, "id")
                record.finish_reason = _finish_reason(response)
                record.usage = _usage_dict(_response_attr(response, "usage"))
                report = self._parse_response(response)
                record.ended_at = _now()
                record.duration_s = round(time.monotonic() - started, 3)
                self._emit(
                    {
                        "event": "call_finished",
                        "call_type": call_type,
                        "request_id": request_id,
                        "model": self.config.model,
                        "attempts": attempt,
                        "response_id": record.response_id,
                        "finish_reason": record.finish_reason,
                        "usage": record.usage,
                        "duration_s": record.duration_s,
                        "ended_at": record.ended_at,
                    }
                )
                return _ResponseResult(report=report, response=response, record=record)
            except ReaderResponseError as exc:
                # Malformed/truncated output is a failed logical call.  It is
                # generally not fixed by resending the same large prompt.
                record.error_type = type(exc).__name__
                record.error = _safe_error(exc)
                record.ended_at = _now()
                record.duration_s = round(time.monotonic() - started, 3)
                self._emit(
                    {
                        "event": "call_failed",
                        "call_type": call_type,
                        "request_id": request_id,
                        "model": self.config.model,
                        "attempts": attempt,
                        "response_id": record.response_id,
                        "finish_reason": record.finish_reason,
                        "usage": record.usage,
                        "error_type": record.error_type,
                        "error": record.error,
                        "retryable": False,
                        "duration_s": record.duration_s,
                        "ended_at": record.ended_at,
                    }
                )
                raise
            except Exception as exc:
                last_error = exc
                retryable = _is_retryable(exc)
                if not retryable or attempt >= max_attempts:
                    record.error_type = type(exc).__name__
                    record.error = _safe_error(exc)
                    record.ended_at = _now()
                    record.duration_s = round(time.monotonic() - started, 3)
                    self._emit(
                        {
                            "event": "call_failed",
                            "call_type": call_type,
                            "request_id": request_id,
                            "model": self.config.model,
                            "attempts": attempt,
                            "error_type": record.error_type,
                            "error": record.error,
                            "retryable": retryable,
                            "duration_s": record.duration_s,
                            "ended_at": record.ended_at,
                        }
                    )
                    raise ReaderAPIError(
                        f"{call_type} call failed after {attempt} attempt(s): {_safe_error(exc)}"
                    ) from exc
                self._emit(
                    {
                        "event": "call_retry",
                        "call_type": call_type,
                        "request_id": request_id,
                        "model": self.config.model,
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "error": _safe_error(exc),
                    }
                )
                time.sleep(min(2.0, 0.25 * (2 ** (attempt - 1))))

        # The loop either returns or raises, but retaining this guard makes a
        # future change to retry policy fail loudly rather than return nothing.
        raise ReaderAPIError(f"{call_type} call failed: {last_error}") from last_error

    def _create_completion(self, messages: list[dict[str, Any]]) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
        }
        if self.config.response_mode.lower() in {"json", "json_object", "auto_validated_json", "auto"}:
            kwargs["response_format"] = {"type": "json_object"}
        if self.config.thinking is not None:
            # DashScope's OpenAI-compatible endpoint accepts provider-specific
            # options through extra_body.  Keep this translation here so the
            # rest of the pipeline remains provider-neutral.
            kwargs["extra_body"] = {"enable_thinking": self.config.thinking}
        try:
            return self.client.chat.completions.create(**kwargs)
        except TypeError as exc:
            # Some small fake clients and older compatible SDK adapters do not
            # accept extra_body.  Retry the same request without that optional
            # provider extension; do not hide endpoint errors generally.
            if "extra_body" in kwargs and "extra_body" in str(exc):
                kwargs.pop("extra_body", None)
                return self.client.chat.completions.create(**kwargs)
            raise

    def _build_messages(
        self,
        *,
        paper_id: str,
        pages: Sequence[PageImage],
        title: str,
        prompt: str | None,
        schema: Mapping[str, Any] | None,
        candidate: Mapping[str, Any] | None,
        visual_reasons: Sequence[str],
        original_pages: Sequence[PageImage],
    ) -> list[dict[str, Any]]:
        schema_obj = schema if schema is not None else self._load_schema()
        instructions = prompt if prompt is not None else self._load_prompt()
        schema_text = json.dumps(schema_obj, ensure_ascii=False, separators=(",", ":")) if schema_obj else ""
        coverage = ", ".join(str(page.pdf_page) for page in pages)
        text_parts = [
            instructions,
            f"paper_id: {paper_id}",
            f"title: {title}" if title else "title: (read from supplied pages)",
            f"The following images are the complete input for this call. PDF pages: {coverage}.",
            "Treat each PDF_PAGE label immediately before an image as authoritative evidence identity.",
        ]
        if schema_text:
            text_parts.append("Return one JSON object conforming to this JSON Schema:\n" + schema_text)
        if candidate is not None:
            text_parts.extend(
                [
                    "This is a visual refinement round. Revise the candidate report only where the new images add evidence. Preserve correct content and return the complete report object again.",
                    "Reasons for requesting visual review:\n" + "\n".join(f"- {reason}" for reason in visual_reasons if reason),
                    "Candidate report JSON:\n" + json.dumps(candidate, ensure_ascii=False),
                ]
            )

        content: list[dict[str, Any]] = [{"type": "text", "text": "\n\n".join(text_parts)}]
        # Original pages make the refinement question auditable while keeping
        # the follow-up small.  De-duplicate pages if a crop is a whole page.
        seen: set[str] = set()
        for page in list(original_pages) + list(pages):
            image_id = page.stable_id()
            if image_id in seen:
                continue
            seen.add(image_id)
            content.append({"type": "text", "text": f"PDF_PAGE={page.pdf_page} IMAGE_ID={image_id}"})
            content.append({"type": "image_url", "image_url": {"url": image_data_uri(page.path)}})
        return [
            {
                "role": "system",
                "content": "Return valid JSON only. All factual claims must be traceable to supplied images.",
            },
            {"role": "user", "content": content},
        ]

    def _load_schema(self) -> Mapping[str, Any] | None:
        if not self.schema_path.exists():
            return None
        try:
            with self.schema_path.open("r", encoding="utf-8") as stream:
                value = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise ReaderConfigError(f"Could not read report schema {self.schema_path}: {exc}") from exc
        if not isinstance(value, Mapping):
            raise ReaderConfigError(f"Report schema root must be an object: {self.schema_path}")
        return value

    def _load_prompt(self) -> str:
        path = self.prompt_path
        if path is None:
            candidate = Path.cwd() / "prompts" / "reader.md"
            path = candidate if candidate.exists() else None
        if path is None:
            return DEFAULT_READER_INSTRUCTIONS
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ReaderConfigError(f"Could not read reader prompt {path}: {exc}") from exc
        return value or DEFAULT_READER_INSTRUCTIONS

    def _parse_response(self, response: Any) -> dict[str, Any]:
        finish_reason = _finish_reason(response)
        if finish_reason in {"length", "max_tokens", "content_filter"}:
            raise ReaderResponseError(
                f"Model response ended with finish_reason={finish_reason}; report is not complete"
            )
        content = _message_content(response)
        if content is None or not str(content).strip():
            raise ReaderResponseError("Model response did not contain message.content JSON")
        try:
            parsed = parse_json_object(str(content))
        except ValueError as exc:
            raise ReaderResponseError(f"Model response is not valid JSON: {exc}") from exc
        if "report" in parsed and isinstance(parsed["report"], Mapping) and len(parsed) == 1:
            parsed = dict(parsed["report"])
        return parsed

    def _emit(self, event: Mapping[str, Any]) -> None:
        recorder = self.event_recorder
        if recorder is None:
            return
        try:
            if callable(recorder):
                recorder(event)
            elif hasattr(recorder, "record"):
                recorder.record(event)  # type: ignore[attr-defined]
            elif hasattr(recorder, "record_event"):
                recorder.record_event(event)  # type: ignore[attr-defined]
            elif hasattr(recorder, "append"):
                recorder.append(dict(event))  # type: ignore[attr-defined]
            else:
                raise TypeError("event_recorder must be callable or expose record(event)")
        except Exception as exc:
            # A logging failure must be visible, but must never turn a paid
            # successful model response into an apparent reading failure.
            # The batch layer can also monitor stderr if it needs strict logs.
            _ = exc


def normalize_page(
    value: PageImage | Mapping[str, Any] | str | os.PathLike[str], *, default_pdf_page: int | None = None
) -> PageImage:
    """Normalize a preprocess page record or image path into ``PageImage``."""

    if isinstance(value, PageImage):
        page = value
    elif isinstance(value, (str, os.PathLike)):
        if default_pdf_page is None:
            raise ReaderInputError(f"A page number is required for image path: {value}")
        page = PageImage(pdf_page=default_pdf_page, path=Path(value))
    elif isinstance(value, Mapping):
        path_value = value.get("path", value.get("image_path", value.get("file")))
        page_value = value.get("pdf_page", value.get("page", default_pdf_page))
        if path_value is None or page_value is None:
            raise ReaderInputError("Page record must contain path and pdf_page")
        page = PageImage(
            pdf_page=_as_int(page_value, "pdf_page"),
            path=Path(path_value),
            image_id=(str(value["image_id"]) if value.get("image_id") is not None else None),
            width=_optional_int(value.get("width")),
            height=_optional_int(value.get("height")),
        )
    else:
        raise ReaderInputError(f"Unsupported page record: {type(value).__name__}")
    if page.pdf_page < 1:
        raise ReaderInputError(f"pdf_page must be 1 or greater, got {page.pdf_page}")
    if not page.path.is_file():
        raise ReaderInputError(f"Rendered page image does not exist: {page.path}")
    return page


def normalize_pages(
    values: Sequence[PageImage | Mapping[str, Any] | str | os.PathLike[str]],
) -> list[PageImage]:
    pages = [normalize_page(value) for value in values]
    pages.sort(key=lambda item: item.pdf_page)
    seen: set[int] = set()
    for page in pages:
        if page.pdf_page in seen:
            raise ReaderInputError(f"Duplicate rendered PDF page: {page.pdf_page}")
        seen.add(page.pdf_page)
    return pages


def image_data_uri(path: str | os.PathLike[str]) -> str:
    """Read an image and return a data URI suitable for OpenAI vision input."""

    image_path = Path(path)
    if not image_path.is_file():
        raise ReaderInputError(f"Image file does not exist: {image_path}")
    try:
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    except OSError as exc:
        raise ReaderInputError(f"Could not read image file {image_path}: {exc}") from exc
    mime, _ = mimetypes.guess_type(image_path.name)
    if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
        mime = "image/png"
    return f"data:{mime};base64,{encoded}"


def parse_json_object(value: str) -> dict[str, Any]:
    """Parse JSON from a provider response, accepting one Markdown fence."""

    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, count=1)
        text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no JSON object found")
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(str(exc)) from exc
    if not isinstance(parsed, dict):
        raise ValueError("top-level JSON value must be an object")
    return parsed


def validate_visual_request(
    request: Mapping[str, Any], available_pages: set[int] | Sequence[int] | None = None
) -> tuple[bool, str]:
    """Check a model visual request before any image is rendered or sent."""

    if not isinstance(request, Mapping):
        return False, "visual_request 不是对象"
    page_value = request.get("pdf_page")
    if isinstance(page_value, bool) or not isinstance(page_value, int) or page_value < 1:
        return False, f"非法补看页码：{page_value!r}"
    if available_pages is not None and page_value not in set(available_pages):
        return False, f"补看页码不在输入材料中：PDF 第 {page_value} 页"
    crop = request.get("crop")
    if crop is None:
        return True, ""
    if not isinstance(crop, Mapping):
        return False, f"PDF 第 {page_value} 页的 crop 必须是对象或 null"
    names = ("x0", "y0", "x1", "y1")
    values: list[float] = []
    for name in names:
        value = crop.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False, f"PDF 第 {page_value} 页的 crop.{name} 不是数字"
        if not 0 <= float(value) <= 1:
            return False, f"PDF 第 {page_value} 页的 crop.{name} 超出 [0, 1]"
        values.append(float(value))
    if values[0] >= values[2] or values[1] >= values[3]:
        return False, f"PDF 第 {page_value} 页的 crop 坐标无有效面积"
    return True, ""


def _coerce_config(value: ReaderConfig | Mapping[str, Any] | str | os.PathLike[str] | None) -> ReaderConfig:
    if value is None:
        default = Path.cwd() / "config.yaml"
        return ReaderConfig.from_yaml(default) if default.exists() else ReaderConfig()
    if isinstance(value, ReaderConfig):
        return value
    if isinstance(value, Mapping):
        return ReaderConfig.from_mapping(value)
    return ReaderConfig.from_yaml(value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ReaderInputError(f"{field_name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ReaderInputError(f"{field_name} must be an integer") from exc
    return result


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return _as_int(value, "image dimension")


def _positive_int(value: Any, field_name: str) -> int:
    result = _as_int(value, field_name)
    if result < 1:
        raise ReaderConfigError(f"{field_name} must be at least 1")
    return result


def _nonnegative_int(value: Any, field_name: str) -> int:
    result = _as_int(value, field_name)
    if result < 0:
        raise ReaderConfigError(f"{field_name} must be at least 0")
    return result


def _positive_float(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ReaderConfigError(f"{field_name} must be a number") from exc
    if result <= 0:
        raise ReaderConfigError(f"{field_name} must be greater than 0")
    return result


def _message_content(response: Any) -> str | None:
    choices = _response_attr(response, "choices")
    if not choices:
        return None
    choice = choices[0]
    message = _response_attr(choice, "message")
    content = _response_attr(message, "content")
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            text = _response_attr(item, "text")
            if text is not None:
                chunks.append(str(text))
        return "".join(chunks) or None
    return str(content) if content is not None else None


def _response_attr(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _finish_reason(response: Any) -> str | None:
    choices = _response_attr(response, "choices") or []
    if not choices:
        return None
    reason = _response_attr(choices[0], "finish_reason")
    return str(reason) if reason is not None else None


def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, Mapping):
        value = dict(usage)
    elif hasattr(usage, "model_dump"):
        try:
            value = dict(usage.model_dump())
        except Exception:
            value = {}
    else:
        value = {
            name: getattr(usage, name)
            for name in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens")
            if getattr(usage, name, None) is not None
        }
    # Preserve provider-specific nested cached-token fields; callers can
    # calculate cost only from fields the endpoint actually returned.
    return value


def _visual_requests(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = report.get("visual_requests", [])
    return [dict(value) for value in values if isinstance(value, Mapping)] if isinstance(values, list) else []


def _append_unresolved_to_report(report: MutableMapping[str, Any], items: Iterable[str]) -> None:
    existing = report.get("unresolved_items")
    if not isinstance(existing, list):
        existing = []
        report["unresolved_items"] = existing
    for item in items:
        if item and item not in existing:
            existing.append(item)


def _call_renderer(
    renderer: Renderer,
    *,
    pdf_page: int,
    crop: Mapping[str, Any] | None,
    dpi: int,
    source_page: PageImage,
) -> Any:
    """Support the two renderer call styles used by preprocess adapters."""

    try:
        return renderer(pdf_page=pdf_page, crop=crop, dpi=dpi, source_page=source_page)
    except TypeError as first_error:
        try:
            return renderer(pdf_page, crop, dpi)
        except TypeError:
            raise first_error


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (ReaderInputError, ReaderConfigError, ReaderResponseError, ReaderDependencyError)):
        return False
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status == 408 or status == 409 or status == 429 or status >= 500
    return isinstance(exc, (TimeoutError, ConnectionError, OSError))


def _safe_error(exc: Exception) -> str:
    message = str(exc).replace(os.environ.get("DASHSCOPE_API_KEY", "\0"), "[REDACTED]")
    return message[:2000]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "CallRecord",
    "DEFAULT_MODEL",
    "EventRecorder",
    "JsonlEventRecorder",
    "PageImage",
    "ReadResult",
    "Reader",
    "ReaderAPIError",
    "ReaderConfig",
    "ReaderConfigError",
    "ReaderDependencyError",
    "ReaderError",
    "ReaderInputError",
    "ReaderResponseError",
    "image_data_uri",
    "load_env_file",
    "normalize_page",
    "normalize_pages",
    "parse_json_object",
    "validate_visual_request",
]
