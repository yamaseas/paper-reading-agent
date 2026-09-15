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
from typing import (
    Any,
    Callable,
    Container,
    Iterable,
    Mapping,
    MutableMapping,
    Protocol,
    Sequence,
)
from uuid import uuid4


DEFAULT_MODEL = "qwen3.8-flash"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "report.schema.json"
# Marks the unresolved items this module derives itself, so they can be
# recomputed from the report instead of accumulating across rounds.
UNCONFIRMED_EDGE_PREFIX = "未确认的方法关系："
PENDING_REQUEST_PREFIX = "补看请求未能执行："
# A request the refinement round did execute but that came back in the revised
# report anyway: the crop was looked at and the question is still open, so
# "pending" would misdescribe it.
REFINED_UNRESOLVED_PREFIX = "补看后仍未解决："
# Items under these prefixes are derived by this module and are recomputed
# from the report at hand, so they must never be carried across rounds.
DERIVED_PREFIXES = (
    UNCONFIRMED_EDGE_PREFIX,
    PENDING_REQUEST_PREFIX,
    REFINED_UNRESOLVED_PREFIX,
)


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
    """The endpoint returned an unusable or truncated response.

    Two flags decide what to do next, and they are mutually exclusive:

    ``repairable``
        The raw text is real content that one text-only repair round may be
        able to re-emit as valid JSON (``raw_content`` carries it).
    ``retryable_with_images``
        The visible output held no report content at all — an empty message,
        ``[]``, ``{}``.  There is nothing to reformat, so the repair round is
        skipped and only a fresh attempt with the page images can help.
    """

    repairable: bool = False
    retryable_with_images: bool = False
    raw_content: str | None = None


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
    max_output_tokens: int | None = None
    render_dpi: int = 160
    crop_dpi: int = 300
    max_visual_rounds: int = 1
    max_crop_images: int = 4
    max_format_repairs: int = 1
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

        # An unset output limit leaves the endpoint default in place, which is
        # usually too small for a full report and silently truncates the JSON.
        raw_output_tokens = model_cfg.get("max_output_tokens", model_cfg.get("max_tokens"))
        max_output_tokens = (
            None
            if raw_output_tokens is None
            else _positive_int(raw_output_tokens, "model.max_output_tokens")
        )

        return cls(
            model=model,
            base_url=base_url,
            api_key_env=str(model_cfg.get("api_key_env", "DASHSCOPE_API_KEY")),
            thinking=thinking,
            response_mode=str(model_cfg.get("response_mode", "json_object")),
            request_timeout_s=_positive_float(
                model_cfg.get("request_timeout_s", 600), "model.request_timeout_s"
            ),
            max_output_tokens=max_output_tokens,
            render_dpi=_positive_int(input_cfg.get("render_dpi", 160), "input.render_dpi"),
            crop_dpi=_positive_int(refinement_cfg.get("crop_dpi", 300), "refinement.crop_dpi"),
            max_visual_rounds=_nonnegative_int(
                refinement_cfg.get("max_visual_rounds", 1), "refinement.max_visual_rounds"
            ),
            max_crop_images=_nonnegative_int(
                refinement_cfg.get("max_crop_images", 4), "refinement.max_crop_images"
            ),
            max_format_repairs=_nonnegative_int(
                refinement_cfg.get("max_format_repairs", 1), "refinement.max_format_repairs"
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
    # Set when the delivered report came out of a text-only repair round: the
    # content was restructured (or, if the repair payload carried no substance,
    # generated) without any page image, which a reader of the report cannot
    # tell from the report itself.
    format_repaired: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "call_type": self.call_type,
            "request_id": self.request_id,
            "attempts": self.attempts,
            "format_repaired": self.format_repaired,
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
    # Crops and re-rendered pages sent during the refinement round.  They are
    # not part of the stable page manifest, so downstream validation and
    # rendering must be told about them explicitly.
    supplemental_images: list[dict[str, Any]] = field(default_factory=list)

    @property
    def visual_requests(self) -> list[dict[str, Any]]:
        requests = self.report.get("visual_requests", [])
        return requests if isinstance(requests, list) else []

    def as_dict(self) -> dict[str, Any]:
        return {
            "report": self.report,
            "calls": [call.as_dict() for call in self.calls],
            "unresolved_items": list(self.unresolved_items),
            "supplemental_images": [dict(item) for item in self.supplemental_images],
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


@dataclass
class _ResponseResult:
    report: dict[str, Any]
    response: Any
    record: CallRecord


Renderer = Callable[..., Any]


DEFAULT_REFINE_INSTRUCTIONS = """You are revising one candidate paper-reading report. You receive the candidate report JSON, the original pages that were unclear, and sharper crops of the areas in question.

Revise only the content that the newly supplied images actually change: numbers, table or figure readings, method relations, and the evidence entries behind them. Keep every other part of the candidate report unchanged, including wording, ordering, and evidence IDs that the new images do not touch. Do not re-read the whole paper from these few pages and do not treat pages that were not re-sent as missing material.

Add new evidence entries with new IDs whose image_id is one of the images supplied in this call. If the new images still do not settle a question, keep the previous wording, set the value aside, and list the item in unresolved_items. Return one JSON object matching the supplied report schema and nothing else."""


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
        refine_prompt_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.config = _coerce_config(config)
        self._client = client
        self.event_recorder = event_recorder
        self.schema_path = Path(schema_path) if schema_path else DEFAULT_SCHEMA_PATH
        self.prompt_path = Path(prompt_path) if prompt_path else None
        self.refine_prompt_path = Path(refine_prompt_path) if refine_prompt_path else None
        self._request_count = 0
        self._format_repairs_used = 0

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
        self._format_repairs_used = 0
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
            return _finalize(result)

        available = normalize_pages(original_pages or normalized_pages)
        # A refinement round returns a whole report, so anything the first
        # round set aside has to be carried across that replacement: an
        # uncertainty that vanishes from the file is an uncertainty the reader
        # of the report can no longer see.
        carried = _carryable_unresolved(first.report)
        # The refinement round has its own instructions: it revisits a few
        # pages of an already written report instead of reading a whole paper.
        refine_prompt = prompt if prompt is not None else self._load_refine_prompt()
        try:
            refined, unresolved, supplemental = self._refine_once(
                paper_id=paper_id,
                candidate=first.report,
                requests=requests,
                available_pages=available,
                renderer=renderer,
                title=title,
                prompt=refine_prompt,
                schema=schema,
            )
        except ReaderError as exc:
            # The first round already produced a complete, schema-valid report.
            # A follow-up that fails is a missing improvement, not a missing
            # paper, so the candidate is delivered with the failure recorded
            # instead of being thrown away and paid for twice on a retry.
            _append_unresolved_to_report(result.report, [f"补看轮次失败：{exc}"])
            _normalize_report(result.report)
            return _finalize(result)
        executed = _executed_request_keys(supplemental)
        if refined is None:
            _append_unresolved_to_report(result.report, unresolved)
            _normalize_report(result.report, executed)
            return _finalize(result)
        result.unresolved_items.extend(unresolved)
        result.report = refined.report
        result.calls.append(refined.record)
        result.raw_responses.append(refined.response)
        result.supplemental_images = supplemental
        _merge_carried(result, carried)
        # Keep the revised report self-describing: a reader of report.json
        # alone must be able to see why something was left unverified.
        _append_unresolved_to_report(result.report, unresolved)
        _normalize_report(result.report, executed)
        return _finalize(result)

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
            return _finalize(ReadResult(report=dict(candidate)))
        available = normalize_pages(pages)
        selected = list(requests) if requests is not None else _visual_requests(candidate)
        carried = _carryable_unresolved(candidate)
        refine_prompt = prompt if prompt is not None else self._load_refine_prompt()
        refined, unresolved, supplemental = self._refine_once(
            paper_id=paper_id,
            candidate=dict(candidate),
            requests=selected,
            available_pages=available,
            renderer=renderer,
            title=title,
            prompt=refine_prompt,
            schema=schema,
        )
        # `_refine_once` only returns no report when no crop could be rendered,
        # so the executed set is empty on that path.
        executed = _executed_request_keys(supplemental)
        if refined is None:
            report = dict(candidate)
            _append_unresolved_to_report(report, unresolved)
            _normalize_report(report, executed)
            return _finalize(ReadResult(report=report))
        result = ReadResult(
            report=refined.report,
            calls=[refined.record],
            raw_responses=[refined.response],
            supplemental_images=supplemental,
        )
        _merge_carried(result, carried)
        _append_unresolved_to_report(result.report, unresolved)
        _normalize_report(result.report, executed)
        return _finalize(result)

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
    ) -> tuple[_ResponseResult | None, list[str], list[dict[str, Any]]]:
        by_page = {page.pdf_page: page for page in available_pages}
        supplemental: list[PageImage] = []
        supplemental_records: list[dict[str, Any]] = []
        # Only the requests that survived validation are allowed into the
        # follow-up call.  Re-deriving pages from the raw model requests here
        # would let a malformed entry (a null page, a string page) raise
        # instead of being recorded as unresolved.
        accepted: list[tuple[int, dict[str, Any]]] = []
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
                supplemental_records.append(
                    {
                        "image_id": image.stable_id(),
                        "pdf_page": page_number,
                        "path": str(image.path),
                        "crop": dict(crop) if isinstance(crop, Mapping) else None,
                        "dpi": float(self.config.crop_dpi),
                        "reason": str(request.get("reason", "")),
                    }
                )
                accepted.append((page_number, dict(request)))
            except ReaderError as exc:
                unresolved.append(f"PDF 第 {page_number} 页补图失败：{exc}")
            except Exception as exc:
                unresolved.append(f"PDF 第 {page_number} 页补图失败：{type(exc).__name__}: {exc}")

        if not supplemental:
            return None, unresolved, supplemental_records
        refined = self._call_report(
            call_type="visual_refinement",
            paper_id=paper_id,
            pages=supplemental,
            title=title,
            prompt=prompt,
            schema=schema,
            candidate=candidate,
            visual_reasons=[str(request.get("reason", "")) for _, request in accepted],
            original_pages=[by_page[page_number] for page_number, _ in accepted],
        )
        return refined, unresolved, supplemental_records

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
        schema_obj = schema if schema is not None else self._load_schema()
        messages = self._build_messages(
            paper_id=paper_id,
            pages=pages,
            title=title,
            prompt=prompt,
            schema=schema_obj,
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
            response: Any = None
            try:
                response = self._create_completion(messages)
                # Capture provider metadata even when the content is
                # truncated or malformed, so the event log explains why the
                # logical call was rejected.
                record.response_id = _response_attr(response, "id")
                record.finish_reason = _finish_reason(response)
                record.usage = _usage_dict(_response_attr(response, "usage"))
                report = self._parse_response(response, schema_obj)
                # Reconciled before the schema check: an unusable visual
                # request must be recorded, not repaired away.
                _normalize_report(report)
                parsed = report
                report = self._repair_schema_errors(
                    report, schema=schema_obj, paper_id=paper_id, call_type=call_type
                )
                # Identity, not a counter: a repair round that ran but whose
                # result was rejected leaves the parsed report in place, and
                # that report must not be labelled as repaired.
                record.format_repaired = report is not parsed
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
                        "format_repaired": record.format_repaired,
                        "duration_s": record.duration_s,
                        "ended_at": record.ended_at,
                    }
                )
                return _ResponseResult(report=report, response=response, record=record)
            except ReaderResponseError as exc:
                # Malformed output is not fixed by resending the same large
                # prompt, but it is often fixed by one text-only repair round
                # that never re-uploads the page images.  An output that holds
                # no report content is the opposite case: no repair is possible
                # (see _parse_response), so it is left to the caller's retry,
                # which does pay for the images again.
                retryable = bool(getattr(exc, "retryable_with_images", False))
                repaired = (
                    self._repair_unparsable(
                        exc, schema=schema_obj, paper_id=paper_id, call_type=call_type
                    )
                    if response is not None
                    else None
                )
                if repaired is not None:
                    record.ended_at = _now()
                    record.duration_s = round(time.monotonic() - started, 3)
                    record.format_repaired = True
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
                            "format_repaired": True,
                            "duration_s": record.duration_s,
                            "ended_at": record.ended_at,
                        }
                    )
                    return _ResponseResult(report=repaired, response=response, record=record)
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
                        "retryable": retryable,
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

    def _repair_schema_errors(
        self,
        report: dict[str, Any],
        *,
        schema: Mapping[str, Any] | None,
        paper_id: str,
        call_type: str,
    ) -> dict[str, Any]:
        """Give the model one text-only chance to fix a structurally invalid report.

        Only shape is repaired here.  Semantic checks (unknown evidence IDs,
        dangling method edges, numbers that contradict a page) are deliberately
        left to ``validate.py`` and to human review, and truncated output is
        never treated as a formatting problem.
        """

        errors = schema_errors(report, schema)
        if not errors:
            return report
        repaired = self._request_repair(
            reason="模型返回的 JSON 不符合 report schema，请修正结构。",
            errors=errors,
            payload=json.dumps(report, ensure_ascii=False),
            schema=schema,
            paper_id=paper_id,
            call_type=call_type,
        )
        if repaired is None:
            return report
        if schema_errors(repaired, schema):
            # A repair that is still invalid is not an improvement; keep the
            # original so the failure is reported against the real output.
            return report
        # The schema cannot express every problem (a crop with no area passes
        # it), so the repaired report is reconciled as well.
        _normalize_report(repaired)
        return repaired

    def _repair_unparsable(
        self,
        error: ReaderResponseError,
        *,
        schema: Mapping[str, Any] | None,
        paper_id: str,
        call_type: str,
    ) -> dict[str, Any] | None:
        """Try one text-only repair of a response whose JSON could not be parsed."""

        raw = error.raw_content
        if not error.repairable or not isinstance(raw, str) or not raw.strip():
            return None
        repaired = self._request_repair(
            reason="上一次输出不是可解析的 JSON 对象，请把同样的内容重新输出为合法 JSON。",
            errors=[_safe_error(error)],
            payload=raw,
            schema=schema,
            paper_id=paper_id,
            call_type=call_type,
        )
        if repaired is None:
            return None
        if schema_errors(repaired, schema):
            return None
        _normalize_report(repaired)
        return repaired

    def _request_repair(
        self,
        *,
        reason: str,
        errors: Sequence[str],
        payload: str,
        schema: Mapping[str, Any] | None,
        paper_id: str,
        call_type: str,
    ) -> dict[str, Any] | None:
        """Send one repair-sized request without any page image.

        The repair call is a real request: it is counted against the per-paper
        budget and is skipped once the repair budget is used up.  A failed
        repair never replaces the original error with a vaguer one.
        """

        if self._format_repairs_used >= self.config.max_format_repairs:
            return None
        if self._request_count >= self.config.max_requests_per_paper:
            self._emit(
                {
                    "event": "repair_skipped",
                    "call_type": call_type,
                    "paper_id": paper_id,
                    "reason": "request budget exhausted",
                }
            )
            return None
        self._format_repairs_used += 1
        self._request_count += 1

        parts = [
            "本次不提供任何页面图片，只做结构修复。不要新增、删除或改写事实、数字、"
            "证据条目与方法关系；只修正 JSON 结构与字段类型，其它内容保持原样。",
            f"paper_id: {paper_id}",
            reason,
            "本地校验发现的问题：\n" + "\n".join(f"- {item}" for item in errors[:20]),
        ]
        if schema:
            parts.append(
                "必须符合的 JSON Schema：\n"
                + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
            )
        parts.append("待修复的内容：\n" + payload[:80_000])

        request_id = uuid4().hex
        started = time.monotonic()
        self._emit(
            {
                "event": "repair_started",
                "call_type": call_type,
                "request_id": request_id,
                "paper_id": paper_id,
                "model": self.config.model,
                "errors": list(errors[:20]),
            }
        )
        try:
            response = self._create_completion(
                [
                    {
                        "role": "system",
                        "content": "Return valid JSON only. Do not add content that was not already present.",
                    },
                    {"role": "user", "content": [{"type": "text", "text": "\n\n".join(parts)}]},
                ]
            )
            content = _message_content(response)
            if content is None or not str(content).strip():
                raise ReaderResponseError("repair response contained no content")
            repaired = parse_json_object(str(content))
        except Exception as exc:  # noqa: BLE001 - repair failure is not fatal
            self._emit(
                {
                    "event": "repair_failed",
                    "call_type": call_type,
                    "request_id": request_id,
                    "paper_id": paper_id,
                    "error_type": type(exc).__name__,
                    "error": _safe_error(exc),
                }
            )
            return None
        self._emit(
            {
                "event": "repair_finished",
                "call_type": call_type,
                "request_id": request_id,
                "paper_id": paper_id,
                "duration_s": round(time.monotonic() - started, 3),
                "usage": _usage_dict(_response_attr(response, "usage")),
            }
        )
        return repaired

    def _create_completion(self, messages: list[dict[str, Any]]) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
        }
        if self.config.max_output_tokens is not None:
            # Leave enough room for a complete report.  The endpoint default is
            # frequently smaller than one report, which truncates the JSON.
            kwargs["max_tokens"] = self.config.max_output_tokens
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
        ]
        if candidate is None:
            text_parts.append(
                f"The following images are the complete input for this call. PDF pages: {coverage}."
            )
        else:
            text_parts.append(
                "This is a visual refinement round, not a first reading. The following original "
                f"pages and enlarged crops are re-sent for verification only: PDF pages {coverage}. "
                "The remaining pages of this paper were supplied in the first call; do not treat "
                "them as missing material, and do not rewrite content that these images do not "
                "change. Keep the candidate report complete: return every section, not only the "
                "revised parts."
            )
        text_parts.append(
            "Treat each PDF_PAGE label immediately before an image as authoritative evidence identity. "
            "IMAGE_ID values name the exact image supplied, including crops; an evidence entry may "
            "cite the crop it was read from."
        )
        if schema_text:
            text_parts.append("Return one JSON object conforming to this JSON Schema:\n" + schema_text)
        if candidate is not None:
            text_parts.extend(
                [
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

    def _load_refine_prompt(self) -> str:
        """Load the refinement instructions, falling back to the built-in ones."""

        path = self.refine_prompt_path
        if path is None and self.prompt_path is not None:
            sibling = self.prompt_path.parent / "refine.md"
            path = sibling if sibling.exists() else None
        if path is None:
            candidate = Path.cwd() / "prompts" / "refine.md"
            path = candidate if candidate.exists() else None
        if path is None:
            return DEFAULT_REFINE_INSTRUCTIONS
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ReaderConfigError(f"Could not read refine prompt {path}: {exc}") from exc
        return value or DEFAULT_REFINE_INSTRUCTIONS

    def _parse_response(
        self, response: Any, schema: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        finish_reason = _finish_reason(response)
        if finish_reason in {"length", "max_tokens", "content_filter"}:
            # Truncated output is missing content, not misformatted content;
            # a repair round would only ask the model to invent the rest.
            raise ReaderResponseError(
                f"Model response ended with finish_reason={finish_reason}; report is not complete"
            )
        content = _message_content(response)
        if content is None or not str(content).strip():
            # Nothing visible at all, so there is no payload to reformat.
            raise _no_report_error("Model response did not contain message.content JSON")
        try:
            parsed = parse_json_object(str(content))
        except ValueError as exc:
            if _carries_no_report(str(content), schema):
                raise _no_report_error(f"Model response is not valid JSON: {exc}") from exc
            error = ReaderResponseError(f"Model response is not valid JSON: {exc}")
            # The raw text is the only repair input available for this case.
            error.raw_content = str(content)
            error.repairable = True
            raise error from exc
        if "report" in parsed and isinstance(parsed["report"], Mapping) and len(parsed) == 1:
            parsed = dict(parsed["report"])
        if _carries_no_report(parsed, schema):
            # A JSON object that holds nothing report-shaped is the same
            # accident as `[]`, and the schema repair would write the report
            # from nothing the same way.
            raise _no_report_error("Model response JSON holds no report content")
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


def _strip_fence(value: str) -> str:
    """Drop the one Markdown fence providers like to wrap JSON in."""

    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, count=1)
        text = text.strip()
    return text


def _no_report_error(message: str) -> ReaderResponseError:
    """A response error that only a fresh attempt with the images can fix."""

    error = ReaderResponseError(message)
    error.retryable_with_images = True
    return error


def _carries_no_report(value: Any, schema: Mapping[str, Any] | None) -> bool:
    """True when a payload cannot be a report that merely needs reformatting.

    Separates "malformed report" from "no report at all".  Text that is not
    parseable JSON is *not* substance-free: the raw text is real content, and a
    repair round can re-emit it as JSON.  Anything else that shares no key with
    the report's required fields is substance-free, and repairing it would mean
    writing a report from nothing.  Accepts raw text or an already-parsed value.
    """

    if isinstance(value, Mapping):
        parsed: Any = value
    else:
        text = _strip_fence(str(value)).strip()
        if not text:
            return True
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return False
    if isinstance(parsed, Mapping) and len(parsed) == 1 and isinstance(parsed.get("report"), Mapping):
        # The same unwrapping _parse_response does, so `{"report": {}}` counts
        # as empty rather than as one filled-in field.
        parsed = parsed["report"]
    if not isinstance(parsed, Mapping):
        return True
    required = schema.get("required") if isinstance(schema, Mapping) else None
    if not isinstance(required, Sequence) or not required:
        return not parsed
    return not (set(parsed) & {str(key) for key in required})


def parse_json_object(value: str) -> dict[str, Any]:
    """Parse JSON from a provider response, accepting one Markdown fence."""

    text = _strip_fence(value)
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


def schema_errors(report: Any, schema: Mapping[str, Any] | None) -> list[str]:
    """Return local JSON-Schema problems for a report, or ``[]`` when unseen.

    The check is best-effort: without ``jsonschema`` or a schema the caller
    simply keeps whatever the model returned, and the batch-level validator
    reports the problem in the usual way.
    """

    if not schema:
        return []
    try:
        from jsonschema import Draft202012Validator  # type: ignore
    except ImportError:  # pragma: no cover - dependency is installed with the project
        return []
    try:
        validator = Draft202012Validator(dict(schema))
        found = sorted(validator.iter_errors(report), key=lambda item: list(item.path))
    except Exception:  # noqa: BLE001 - an unusable schema must not break reading
        return []
    errors: list[str] = []
    for error in found[:20]:
        location = "$" + "".join(f"/{part}" for part in error.path)
        errors.append(f"{location}: {error.message}")
    return errors


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


def _sanitize_visual_requests(report: MutableMapping[str, Any]) -> None:
    """Move visual requests that can never be rendered into ``unresolved_items``.

    A request with a null or non-integer page, or with a crop that has no area,
    is not something a later round can act on -- and it is also a schema
    violation, so leaving it in the report would buy a whole-report format
    repair that is allowed to drop the entry.  Dropping it here keeps the fact
    that the model wanted a closer look, at no extra request.

    Shape only: whether the page really exists is decided later, against the
    pages actually available to the follow-up call.
    """

    values = report.get("visual_requests")
    if not isinstance(values, list):
        return
    kept: list[Any] = []
    rejected: list[str] = []
    for value in values:
        valid, reason = validate_visual_request(value, None)
        if valid:
            kept.append(value)
        else:
            rejected.append(f"已忽略无法执行的补看请求：{reason}")
    if not rejected:
        return
    report["visual_requests"] = kept
    _append_unresolved_to_report(report, rejected)


def _append_unresolved_to_report(report: MutableMapping[str, Any], items: Iterable[str]) -> None:
    existing = report.get("unresolved_items")
    if not isinstance(existing, list):
        existing = []
        report["unresolved_items"] = existing
    for item in items:
        if item and item not in existing:
            existing.append(item)


def _mappings(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _unresolved_texts(report: Mapping[str, Any]) -> list[str]:
    values = report.get("unresolved_items")
    if not isinstance(values, list):
        return []
    return [text for text in (str(value) for value in values) if text]


def _merge_carried(result: ReadResult, items: Sequence[str]) -> None:
    """Keep pre-refinement unresolved items in the delivered report and result."""

    if not items:
        return
    _append_unresolved_to_report(result.report, items)
    for item in items:
        if item not in result.unresolved_items:
            result.unresolved_items.append(item)


def _carryable_unresolved(report: Mapping[str, Any]) -> list[str]:
    """Items worth carrying into a refined report.

    Derived items are left out: they are recomputed from the refined report, so
    carrying them would re-flag a relation the model just confirmed, or a
    request the refinement round has since executed.
    """

    return [item for item in _unresolved_texts(report) if not item.startswith(DERIVED_PREFIXES)]


def _record_unconfirmed_edges(report: MutableMapping[str, Any]) -> None:
    """List every relation the method diagram had to leave out.

    The reader prompt requires a relation without evidence to be marked
    ``confirmed: false`` *and* listed in ``unresolved_items``; the diagram then
    omits it.  Recomputing that list from the report keeps the promise when the
    model forgets, and keeps it correct when a refinement round confirms the
    relation after all.
    """

    method = report.get("method")
    names: dict[str, str] = {}
    if isinstance(method, Mapping):
        for node in _mappings(method.get("nodes")):
            node_id = node.get("id")
            if isinstance(node_id, str):
                names[node_id] = str(node.get("name") or node_id)
    derived: list[str] = []
    for edge in _mappings(method.get("edges") if isinstance(method, Mapping) else None):
        # Mirrors render_mermaid: anything that is not confirmed is omitted.
        if edge.get("confirmed") is True:
            continue
        source = str(edge.get("from", "?"))
        target = str(edge.get("to", "?"))
        derived.append(
            f"{UNCONFIRMED_EDGE_PREFIX}{names.get(source, source)} → {names.get(target, target)}"
            f"（{edge.get('relation', 'relation')}）未获证据确认，已从方法图中省略"
        )
    _recompute_derived(report, UNCONFIRMED_EDGE_PREFIX, derived)


def _finalize(result: ReadResult) -> ReadResult:
    """Make ``unresolved_items`` agree with the report that is returned."""

    result.unresolved_items = _unresolved_texts(result.report)
    return result


def _request_key(request: Mapping[str, Any]) -> tuple[Any, ...]:
    """Identify a crop request closely enough to recognise an echoed copy.

    The refinement prompt tells the model to output the whole report and to
    leave untouched parts alone, so a request that was just executed tends to
    come back verbatim.  Comparing page and crop -- with crops rounded, because
    a model may echo ``0.5200000000000001`` for ``0.52`` -- lets the reader
    tell that echo apart from a genuinely new request.
    """

    page = request.get("pdf_page")
    crop = request.get("crop")
    if not isinstance(crop, Mapping):
        return (page, None)
    coordinates: list[Any] = []
    for name in ("x0", "y0", "x1", "y1"):
        value = crop.get(name)
        coordinates.append(
            round(float(value), 4)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else str(value)
        )
    return (page, tuple(coordinates))


def _executed_request_keys(supplemental: Sequence[Mapping[str, Any]]) -> set[tuple[Any, ...]]:
    """Keys of the requests a refinement round actually rendered an image for."""

    return {_request_key(item) for item in supplemental}


def _request_line(item: Mapping[str, Any]) -> str:
    return (
        f"PDF 第 {item.get('pdf_page', '?')} 页"
        f"（{_text_value(item.get('reason')) or '未说明原因'}）"
    )


def _record_pending_requests(
    report: MutableMapping[str, Any], executed: Container[Any] = ()
) -> None:
    """List every visual request still sitting in the report without an image.

    A request that reached the delivered report was never looked at: the
    refinement round either rewrites the list or leaves it behind.  Recording
    them here is what keeps ``needs_review`` from appearing above an empty
    pending list.

    ``executed`` holds the keys of the requests this round did render.  A
    request that comes back in the revised report although its crop was just
    sent is not pending -- it was looked at and the question survived the look
    -- so it is reported as unresolved instead of unexecuted.  Without that
    distinction the model's habit of echoing the candidate's request list would
    turn a successful refinement round into four phantom "未能执行" items.
    """

    pending: list[str] = []
    still_open: list[str] = []
    for item in _mappings(report.get("visual_requests")):
        line = _request_line(item)
        if _request_key(item) in executed:
            still_open.append(f"{REFINED_UNRESOLVED_PREFIX}{line}")
        else:
            pending.append(f"{PENDING_REQUEST_PREFIX}{line}")
    _recompute_derived(report, PENDING_REQUEST_PREFIX, pending)
    _recompute_derived(report, REFINED_UNRESOLVED_PREFIX, still_open)


def _recompute_derived(report: MutableMapping[str, Any], prefix: str, derived: Sequence[str]) -> None:
    """Replace one group of derived items with a freshly computed list.

    Replacing rather than appending keeps a stale item from outliving the state
    it described -- a relation the refinement round confirmed, or a request it
    executed -- including when the model echoes the old list back verbatim.
    """

    existing = _unresolved_texts(report)
    kept = [item for item in existing if not item.startswith(prefix)]
    if len(kept) == len(existing) and not derived:
        return
    report["unresolved_items"] = kept + [item for item in derived if item not in kept]


def _text_value(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _normalize_report(
    report: MutableMapping[str, Any], executed: Container[Any] = ()
) -> None:
    """Reconcile a freshly parsed report with what the rest of the pipeline needs."""

    _sanitize_visual_requests(report)
    _record_unconfirmed_edges(report)
    _record_pending_requests(report, executed)


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
    "DEFAULT_READER_INSTRUCTIONS",
    "DEFAULT_REFINE_INSTRUCTIONS",
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
    "schema_errors",
    "validate_visual_request",
]
