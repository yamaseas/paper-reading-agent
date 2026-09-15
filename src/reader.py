"""Reader client for the paper-reading pipeline.

The module deliberately contains only the model-facing part of the pipeline.
PDF rendering and report validation live in the neighbouring modules.

The reader is **text-first**: when page-marked full text is supplied, that text
is the primary input of the main call, and page images are only rendered and
sent when the model asks for a closer look through ``visual_requests``.  A
caller that supplies no text keeps the older image-only behaviour, which is
what an A/B comparison between the two input modes needs.

Three model-facing stages share one request budget, one retry policy, one
format-repair budget and one event log:

``read``
    The main reading call, plus at most one bounded visual follow-up.  The
    follow-up returns a *patch* rather than a second report, so a crop that
    changes one table cell cannot rewrite the sections it was not asked about.
``generate_diagrams``
    A separate Mermaid stage.  It reads the grounded report data (and, when
    prefix reuse is on, replays the main call's prompt prefix) instead of
    re-reading the PDF, and it never runs inside the first reading call.
``_request_repair``
    A text-only schema repair, which never re-uploads any image.

The OpenAI import is lazy.  This keeps preprocessing and local validation
usable on machines where the optional API dependency has not been installed,
while still producing an actionable error when an API client is needed.
"""

from __future__ import annotations

import base64
import copy
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
from urllib.parse import urlsplit
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
# A patch entry the deterministic merge refused to apply.  Recorded rather than
# raised: the candidate report is already complete, so an unusable patch entry
# is a missing improvement, not a missing paper.
PATCH_REJECTED_PREFIX = "补看补丁未应用："
DANGLING_EVIDENCE_PREFIX = "证据引用已移除："
DANGLING_METHOD_PREFIX = "未落地的方法项已省略："
# Diagram-stage bookkeeping.  These are computed once from the final diagrams,
# so -- unlike DERIVED_PREFIXES -- they are never recomputed and must be
# carried like any other unresolved item.
DIAGRAM_FAILED_PREFIX = "方法图生成失败："
DIAGRAM_OMITTED_PREFIX = "方法图已省略："
# Items under these prefixes are derived by this module and are recomputed
# from the report at hand, so they must never be carried across rounds.
DERIVED_PREFIXES = (
    UNCONFIRMED_EDGE_PREFIX,
    PENDING_REQUEST_PREFIX,
    REFINED_UNRESOLVED_PREFIX,
    DANGLING_EVIDENCE_PREFIX,
    DANGLING_METHOD_PREFIX,
)

# One system message for every call this module makes.  The reading call and
# the diagram call must start with the same tokens for the endpoint's prefix
# cache to serve the second one, so this is a constant rather than per-call
# text.
READER_SYSTEM_MESSAGE = (
    "Return valid JSON only. All factual claims must be traceable to the supplied material."
)

# The refinement patch is applied to the candidate report by this module, so
# its shape is fixed here rather than in the report schema.
PATCH_SECTIONS = (
    "evidence_updates",
    "claim_updates",
    "method_updates",
    "experiment_updates",
    "resolved_visual_requests",
    "unresolved_items_add",
)
_EVIDENCE_SOURCE_TYPES = ("text", "image")
_EVIDENCE_FIELDS = (
    "source_type",
    "pdf_page",
    "source_id",
    "section",
    "figure_or_table",
    "locator",
    "quote",
)
_CLAIM_KINDS = ("author_claim", "reader_inference", "llm_analysis")
_METHOD_EDGE_RELATIONS = (
    "data_flow",
    "control_flow",
    "dependency",
    "feedback",
    "parallel",
)
INPUT_MODES = ("hybrid_text_visual", "full_page_images")
_DIAGRAM_NODE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_DIAGRAM_KINDS = (
    "input",
    "output",
    "component",
    "artifact",
    "decision",
    "loop",
    "oracle",
    "external",
    "environment",
    "stage",
)
THINKING_BUDGET_STAGES = ("read", "visual_refinement", "format_repair", "diagram")
DEFAULT_THINKING_BUDGETS = {
    "read": 16384,
    "visual_refinement": 8192,
    "format_repair": 4096,
    "diagram": 4096,
}


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

    requests_made: int = 0


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
    thinking_budgets: Mapping[str, int] = field(
        default_factory=lambda: dict(DEFAULT_THINKING_BUDGETS)
    )
    response_mode: str = "json_object"
    request_timeout_s: float = 600.0
    max_output_tokens: int | None = None
    input_mode: str = "hybrid_text_visual"
    render_dpi: int = 160
    crop_dpi: int = 300
    max_visual_rounds: int = 1
    max_crop_images: int = 4
    max_format_repairs: int = 1
    max_retries_per_call: int = 2
    max_requests_per_paper: int = 6
    diagrams_enabled: bool = True
    max_diagrams: int = 4
    reuse_reading_prefix: bool = True

    def text_first(self) -> bool:
        """Whether page-marked text may replace the page images of a read.

        ``full_page_images`` is the old A/B baseline: it keeps every page
        image and ignores any text it is handed.
        """

        return self.input_mode != "full_page_images"

    def thinking_budget_for(self, call_type: str) -> int | None:
        """Return the bounded reasoning budget for one model-facing stage."""

        if self.thinking is not True:
            return None
        value = self.thinking_budgets.get(call_type)
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else None

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
        diagram_cfg = _mapping(value.get("diagrams"))

        model = str(model_cfg.get("name", DEFAULT_MODEL)).strip()
        if not model:
            raise ReaderConfigError("model.name must be a non-empty model identifier")

        base_env = str(model_cfg.get("base_url_env", "DASHSCOPE_BASE_URL"))
        base_url = (
            os.environ.get(base_env)
            or str(model_cfg.get("default_base_url", model_cfg.get("base_url", DEFAULT_BASE_URL)))
        ).strip()
        if not base_url:
            raise ReaderConfigError(
                f"No API base URL configured; set {base_env} or model.default_base_url"
            )
        try:
            parsed_base_url = urlsplit(base_url)
        except ValueError as exc:
            raise ReaderConfigError(
                f"{base_env} or model.default_base_url must be an absolute HTTP(S) URL"
            ) from exc
        if parsed_base_url.scheme not in {"http", "https"} or not parsed_base_url.netloc:
            raise ReaderConfigError(
                f"{base_env} or model.default_base_url must be an absolute HTTP(S) URL"
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
            thinking_budgets=_thinking_budgets(model_cfg.get("thinking_budget")),
            response_mode=str(model_cfg.get("response_mode", "json_object")),
            request_timeout_s=_positive_float(
                model_cfg.get("request_timeout_s", 600), "model.request_timeout_s"
            ),
            max_output_tokens=max_output_tokens,
            input_mode=_input_mode(input_cfg.get("mode", "hybrid_text_visual")),
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
            diagrams_enabled=_bool(diagram_cfg.get("enabled", True), "diagrams.enabled"),
            max_diagrams=_positive_int(diagram_cfg.get("max_diagrams", 4), "diagrams.max_diagrams"),
            reuse_reading_prefix=_bool(
                diagram_cfg.get("reuse_reading_prefix", True), "diagrams.reuse_reading_prefix"
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
    requests_made: int = 0
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
            "requests_made": self.requests_made,
            "unresolved_items": list(self.unresolved_items),
            "supplemental_images": [dict(item) for item in self.supplemental_images],
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


@dataclass
class DiagramResult:
    """Result of the separate Mermaid stage.

    The stage deliberately returns diagrams rather than a report: it must not
    be able to change a fact, only to draw the ones that are already grounded.
    """

    diagrams: list[dict[str, Any]] = field(default_factory=list)
    calls: list[CallRecord] = field(default_factory=list)
    raw_responses: list[Any] = field(default_factory=list)
    requests_made: int = 0
    unresolved_items: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "diagrams": [dict(item) for item in self.diagrams],
            "calls": [call.as_dict() for call in self.calls],
            "requests_made": self.requests_made,
            "unresolved_items": list(self.unresolved_items),
        }


@dataclass
class _ResponseResult:
    """One logical call's parsed payload plus the metadata it produced."""

    payload: dict[str, Any]
    response: Any
    record: CallRecord


@dataclass
class _RefineOutcome:
    """What one visual refinement round produced.

    ``report`` is ``None`` when no crop could be rendered at all, in which case
    the candidate is still the best report available.
    """

    report: dict[str, Any] | None = None
    problems: list[str] = field(default_factory=list)
    supplemental: list[dict[str, Any]] = field(default_factory=list)
    record: CallRecord | None = None
    response: Any = None


Renderer = Callable[..., Any]


DEFAULT_REFINE_INSTRUCTIONS = """You are revising one candidate paper-reading report. You receive the candidate report JSON, the original pages that were unclear, and sharper crops of the areas in question. You return a PATCH, not a report.

Return exactly one JSON object with these six arrays and nothing else:
- evidence_updates: evidence entries to add or correct. Each entry needs an "id". An id that already exists in the candidate is merged field by field; a new id must carry every field (source_type, pdf_page, source_id, section, figure_or_table, locator, quote). New image evidence must use a source_id that was supplied in this call.
- claim_updates: claims to add or correct, keyed by "id"; a new claim also needs "text" and "kind".
- method_updates: entries with a "target" of "method", "node", "edge" or "step". A node is keyed by "id", an edge by "from" and "to", a step by "step". Only the fields you list are changed.
- experiment_updates: experiments to correct, keyed by the 0-based "index" into the candidate's experiments array.
- resolved_visual_requests: requests from the candidate that these images have now settled, keyed by "pdf_page" and "crop". A request you did not receive an image for cannot be resolved here.
- unresolved_items_add: new sentences for unresolved_items.

Rules:
1. Only record what the newly supplied images actually show: numbers, table cells, figure readings, method relations, and the evidence behind them.
2. Never restate or rewrite anything the new images do not change. Do not re-read the whole paper from these few pages, and do not treat pages that were not re-sent as missing material.
3. If the new images still do not settle a question, add nothing for it and say so in unresolved_items_add. Never guess a value.
4. Only add a method edge with confirmed=true when the supplied images show that relation; anything you cannot confirm stays out of the patch.
5. Return no Markdown fences and no commentary outside the JSON object."""


DEFAULT_READER_INSTRUCTIONS = """You are a careful scientific paper reader. You receive the paper's page-marked full text, and for some papers page images as well. Return exactly one JSON object matching the supplied report schema.

Write the report in Simplified Chinese. Answer Q1 through Q7 in the schema. Distinguish author claims, reader inferences, and your own analysis. Every important fact, method node, method edge, and reported result must cite evidence IDs that point to a real supplied page. Never invent a page, quote, number, baseline, method step, or relation. Use an empty string or an explicit unresolved item when the material is unreadable or does not contain the information; use “Not reported” only when the supplied material clearly covers the relevant section and the authors do not report it.

The full text is extracted automatically, so reading order inside a page is approximate and equations, tables and figures are often garbled or missing entirely. Anything you cannot judge reliably from the text — a table's numbers, a figure's structure, an equation's symbols, the layout that decides which column follows which — must trigger a visual_requests entry rather than a guess. Ask for at most four focused crops, each with a 1-based PDF page and an optional normalized crop [x0, y0, x1, y1] in 0..1 page coordinates. Do not ask for material the text already states plainly.

Cover each question with what the paper actually supports:
- Q1: background, the gap in existing work, the research problem, the motivation, and the authors' claimed contributions.
- Q2: related work grouped by research category, each entry stating the work, the problem it addresses, its method, and how this paper differs.
- Q3: input, overall idea, workflow, intermediate artifacts, tools and models, feedback loops, decisions and branches, and output. Put every core processing component in the exact `method.nodes` field; never emit a `method.components` field. Every method node, edge, and step must carry at least one non-empty `evidence_ids` entry that resolves to this report's evidence array. Method edges do not have a `label` field.
- Q4: one entry per experiment with its purpose, dataset and sample size, baselines, model or system, metrics, settings, main result and conclusion. When the paper gives no explicit RQ, leave research_question empty instead of inventing an RQ number.
- Q5: authors_limitations and authors_future_work are the authors' own words; open_questions and research_directions are your analysis and must be marked as such by where they are placed.
- Q6: a complete, self-contained summary of the paper.
- Q7: what a researcher should look at next to understand, reproduce or extend this paper — the sections/figures/tables worth close reading, concepts to look up, questions worth asking, what reproduction requires, and what to read next. Do not repeat Q6.

Evidence entries state where a fact came from: source_type "text" quotes the page-marked full text with source_id "page-NNN-text"; source_type "image" cites a supplied image with its page or crop id as source_id. Choose the evidence array's ids first, then copy those ids byte-for-byte into every evidence_ids reference; never compose a similar-looking id separately. Keep table numbers with their metric, dataset, baseline or method, setting, unit, and denominator. For method edges, say whether the relation is data flow, control flow, or dependency and set confirmed=false when the material does not support it; an edge with confirmed=false is omitted from every diagram and listed for review.

Do not write Mermaid yourself: a separate stage draws the diagrams from the entries you confirm. Return no Markdown fences and no commentary outside the JSON object."""


DEFAULT_DIAGRAM_INSTRUCTIONS = """You are drawing the method diagrams of one already-read paper. You receive the grounded report data that a previous reading pass produced, and you return exactly one JSON object with a "diagrams" array.

Draw between 1 and 4 diagrams. Return an empty array only when the paper genuinely has no process to draw.

Every node and edge you draw must already exist in the grounded report data and must carry the evidence_ids that support it. Change nothing, correct nothing, and add no step, tool, number or relation that the report data does not already contain. If the report data is too thin for a diagram, draw fewer diagrams rather than filling the gaps yourself.

Draw what the material actually supports — do not flatten a rich method into a straight line:
- the input and output artifacts of the whole method;
- the core processing components;
- intermediate artifacts that pass between steps;
- branches and decisions, with the condition on the decision node and the outcome on each outgoing edge;
- iteration and feedback loops, drawn as an edge that returns to an earlier node;
- parallel paths, drawn as separate edges leaving the same node;
- the execution or verification oracle that decides whether a step succeeded;
- external tools, models and services;
- the environment the method runs in;
- phases such as training, inference and evaluation, which belong in "group" so they render as subgraphs.

A researcher who reads only your diagrams should be able to retell the method's execution flow. Never drop a meaningful feedback loop, branch, oracle or intermediate artifact just to keep the picture simple.

Node "kind" selects the shape: input, output, component, artifact, decision, loop, oracle, external, environment, stage. Node "group" is the phase name; nodes sharing a group are rendered inside one subgraph. Edge "relation" is one of data_flow, control_flow, dependency, feedback, parallel. Only set an edge's confirmed to true when the report data supports the relation; an unconfirmed edge is dropped and reported for review.

Return no Markdown fences and no commentary outside the JSON object."""


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
        diagram_prompt_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.config = _coerce_config(config)
        self._client = client
        self.event_recorder = event_recorder
        self.schema_path = Path(schema_path) if schema_path else DEFAULT_SCHEMA_PATH
        self.prompt_path = Path(prompt_path) if prompt_path else None
        self.refine_prompt_path = Path(refine_prompt_path) if refine_prompt_path else None
        self.diagram_prompt_path = Path(diagram_prompt_path) if diagram_prompt_path else None
        self._request_count = 0
        self._format_repairs_used: dict[str, int] = {}
        # The user-message prefix of the last text-first reading call.  The
        # diagram stage replays it verbatim so the endpoint's prefix cache can
        # serve it instead of charging for the whole paper a second time.
        self._last_reading_prefix: str | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._make_client()
        return self._client

    @property
    def requests_made(self) -> int:
        """Number of HTTP model requests issued by this Reader session."""

        return self._request_count

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
                # Retry only in ``_call_json`` (or in batch.py when batch owns
                # the retry policy).  The SDK otherwise retries
                # twice invisibly, so a configured 600-second timeout can
                # occupy one apparent attempt for roughly 30 minutes and the
                # durable request counter no longer describes actual sends.
                max_retries=0,
            )
        except Exception as exc:
            raise ReaderConfigError(f"Could not create OpenAI-compatible client: {exc}") from exc

    def read(
        self,
        paper_id: str,
        pages: Sequence[PageImage | Mapping[str, Any] | str | os.PathLike[str]],
        *,
        text: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        title: str = "",
        prompt: str | None = None,
        schema: Mapping[str, Any] | None = None,
        renderer: Renderer | None = None,
        original_pages: Sequence[PageImage | Mapping[str, Any] | str | os.PathLike[str]] | None = None,
    ) -> ReadResult:
        """Read the paper once and optionally perform one visual follow-up.

        ``text`` is the page-marked full text from ``preprocess.py``.  When it
        is supplied -- and ``input.mode`` is not the ``full_page_images`` A/B
        baseline -- it becomes the complete input of the main call and no page
        image is uploaded; the pages are still rendered, because a follow-up
        crop and every evidence link need them.

        ``renderer`` is intentionally injected because page rendering belongs
        to ``preprocess.py``.  It may accept ``(pdf_page, crop, dpi)`` or the
        equivalent keyword arguments and may return a path, ``PageImage``, or
        a mapping containing ``path`` and ``pdf_page``.
        """

        normalized_pages = normalize_pages(pages)
        if not normalized_pages:
            raise ReaderInputError("At least one rendered PDF page is required")
        self._request_count = 0
        self._format_repairs_used = {}
        effective_text = text if self.config.text_first() else None
        first = self._call_report(
            call_type="read",
            paper_id=paper_id,
            pages=normalized_pages,
            text=effective_text,
            metadata=metadata,
            title=title,
            prompt=prompt,
            schema=schema,
        )
        result = ReadResult(
            report=first.payload,
            calls=[first.record],
            raw_responses=[first.response],
        )

        requests = _visual_requests(first.payload)
        if not requests or self.config.max_visual_rounds < 1:
            return _finalize(result, self._request_count)

        available = normalize_pages(original_pages or normalized_pages)
        # A refinement round merges a patch into the candidate, so anything the
        # first round set aside has to be carried explicitly: an uncertainty
        # that vanishes from the file is an uncertainty the reader of the
        # report can no longer see.
        carried = _carryable_unresolved(first.payload)
        # The refinement round has its own instructions: it revisits a few
        # pages of an already written report instead of reading a whole paper.
        refine_prompt = prompt if prompt is not None else self._load_refine_prompt()
        try:
            outcome = self._refine_once(
                paper_id=paper_id,
                candidate=first.payload,
                requests=requests,
                available_pages=available,
                renderer=renderer,
                title=title,
                prompt=refine_prompt,
            )
        except ReaderError as exc:
            # The first round already produced a complete, schema-valid report.
            # A follow-up that fails is a missing improvement, not a missing
            # paper, so the candidate is delivered with the failure recorded
            # instead of being thrown away and paid for twice on a retry.
            _append_unresolved_to_report(result.report, [f"补看轮次失败：{exc}"])
            _normalize_report(result.report)
            return _finalize(result, self._request_count)
        executed = _executed_request_keys(outcome.supplemental)
        if outcome.report is None:
            _append_unresolved_to_report(result.report, outcome.problems)
            _normalize_report(result.report, executed)
            return _finalize(result, self._request_count)
        result.report = outcome.report
        if outcome.record is not None:
            result.calls.append(outcome.record)
            result.raw_responses.append(outcome.response)
        result.supplemental_images = outcome.supplemental
        _merge_carried(result, carried)
        # Keep the revised report self-describing: a reader of report.json
        # alone must be able to see why something was left unverified.
        _append_unresolved_to_report(result.report, outcome.problems)
        _normalize_report(result.report, executed)
        return _finalize(result, self._request_count)

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
        candidate and wants to schedule the follow-up separately.  The result
        carries the candidate with the returned patch applied; the candidate
        itself is never mutated.
        """

        if self.config.max_visual_rounds < 1:
            return _finalize(
                ReadResult(report=copy.deepcopy(dict(candidate))), self._request_count
            )
        available = normalize_pages(pages)
        selected = list(requests) if requests is not None else _visual_requests(candidate)
        carried = _carryable_unresolved(candidate)
        refine_prompt = prompt if prompt is not None else self._load_refine_prompt()
        outcome = self._refine_once(
            paper_id=paper_id,
            candidate=dict(candidate),
            requests=selected,
            available_pages=available,
            renderer=renderer,
            title=title,
            prompt=refine_prompt,
        )
        # `_refine_once` only returns no report when no crop could be rendered,
        # so the executed set is empty on that path.
        executed = _executed_request_keys(outcome.supplemental)
        if outcome.report is None:
            report = copy.deepcopy(dict(candidate))
            _append_unresolved_to_report(report, outcome.problems)
            _normalize_report(report, executed)
            return _finalize(ReadResult(report=report), self._request_count)
        result = ReadResult(
            report=outcome.report,
            calls=[outcome.record] if outcome.record is not None else [],
            raw_responses=[outcome.response] if outcome.record is not None else [],
            supplemental_images=outcome.supplemental,
        )
        _merge_carried(result, carried)
        _append_unresolved_to_report(result.report, outcome.problems)
        _normalize_report(result.report, executed)
        return _finalize(result, self._request_count)

    def generate_diagrams(
        self,
        paper_id: str,
        report: Mapping[str, Any],
        *,
        title: str = "",
        prompt: str | None = None,
        schema: Mapping[str, Any] | None = None,
    ) -> DiagramResult:
        """Draw the report's method diagrams in a separate call.

        The stage reads only the grounded report data -- confirmed nodes,
        steps, artifacts, conditions, loops and the evidence behind them -- and
        never re-reads the PDF.  It cannot change a fact: its output is a
        ``diagrams`` array that is sanitized against the report before it is
        returned, and a diagram that cites an evidence entry the report does
        not have is dropped rather than drawn.

        A failure here costs the diagrams, not the paper: the report is already
        complete and grounded, so the error is returned as an unresolved item.
        """

        if not self.config.diagrams_enabled:
            return DiagramResult()
        stage_request_start = self._request_count
        report_schema = schema if schema is not None else self._load_schema()
        instructions = prompt if prompt is not None else self._load_diagram_prompt()
        schema_obj = diagram_response_schema(report_schema)
        prefix = self._last_reading_prefix if self.config.reuse_reading_prefix else None
        messages = self._build_diagram_messages(
            paper_id=paper_id,
            report=report,
            instructions=instructions,
            schema_obj=schema_obj,
            prefix=prefix,
        )
        try:
            response = self._call_json(
                call_type="diagram",
                paper_id=paper_id,
                messages=messages,
                schema=schema_obj,
            )
        except ReaderError as exc:
            return DiagramResult(
                requests_made=self._request_count - stage_request_start,
                unresolved_items=[f"{DIAGRAM_FAILED_PREFIX}{exc}"],
            )

        diagrams, problems = sanitize_diagrams(
            response.payload, report, max_diagrams=self.config.max_diagrams
        )
        if not diagrams and not problems:
            problems.append(f"{DIAGRAM_OMITTED_PREFIX}模型没有返回任何方法图")
        self._emit(
            {
                "event": "diagram_stage",
                "paper_id": paper_id,
                "at": _now(),
                "diagrams": len(diagrams),
                # Recorded so the cost of this stage can be audited: a reused
                # prefix is charged at the provider's cached-input rate.
                "prefix_reused": prefix is not None,
                "prefix_chars": len(prefix or ""),
            }
        )
        return DiagramResult(
            diagrams=diagrams,
            calls=[response.record],
            raw_responses=[response.response],
            requests_made=self._request_count - stage_request_start,
            unresolved_items=problems,
        )

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
    ) -> _RefineOutcome:
        by_page = {page.pdf_page: page for page in available_pages}
        supplemental: list[PageImage] = []
        supplemental_records: list[dict[str, Any]] = []
        # Only the requests that survived validation are allowed into the
        # follow-up call.  Re-deriving pages from the raw model requests here
        # would let a malformed entry (a null page, a string page) raise
        # instead of being recorded as unresolved.
        accepted: list[tuple[int, dict[str, Any]]] = []
        problems: list[str] = []
        for request in requests:
            if len(supplemental) >= self.config.max_crop_images:
                problems.append(
                    f"超出补看预算（最多 {self.config.max_crop_images} 张）："
                    f"PDF 第 {request.get('pdf_page', '?')} 页"
                )
                continue
            valid, reason = validate_visual_request(request, set(by_page))
            if not valid:
                problems.append(reason)
                continue
            page_number = int(request["pdf_page"])
            crop = request.get("crop")
            if renderer is None:
                problems.append(
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
                problems.append(f"PDF 第 {page_number} 页补图失败：{exc}")
            except Exception as exc:
                problems.append(f"PDF 第 {page_number} 页补图失败：{type(exc).__name__}: {exc}")

        if not supplemental:
            return _RefineOutcome(problems=problems, supplemental=supplemental_records)

        response = self._call_patch(
            paper_id=paper_id,
            candidate=candidate,
            pages=supplemental,
            original_pages=[by_page[page_number] for page_number, _ in accepted],
            visual_reasons=[str(request.get("reason", "")) for _, request in accepted],
            title=title,
            prompt=prompt,
        )
        # The images this round actually sent: an evidence entry that claims to
        # be visual has to name one of them, or it points at nothing.
        supplied = {str(record["image_id"]) for record in supplemental_records}
        supplied.update(f"page-{int(record['pdf_page']):03d}" for record in supplemental_records)
        merged, rejected = _apply_patch(
            candidate,
            response.payload,
            executed=_executed_request_keys(supplemental_records),
            supplied=supplied,
        )
        return _RefineOutcome(
            report=merged,
            problems=[*problems, *rejected],
            supplemental=supplemental_records,
            record=response.record,
            response=response.response,
        )

    def _call_report(
        self,
        *,
        call_type: str,
        paper_id: str,
        pages: Sequence[PageImage],
        text: str | None,
        metadata: Mapping[str, Any] | None,
        title: str,
        prompt: str | None,
        schema: Mapping[str, Any] | None,
    ) -> _ResponseResult:
        """Run one reading call for a whole paper."""

        schema_obj = schema if schema is not None else self._load_schema()
        messages = self._build_reading_messages(
            paper_id=paper_id,
            pages=pages,
            text=text,
            metadata=metadata,
            title=title,
            prompt=prompt,
            schema_obj=schema_obj,
        )
        return self._call_json(
            call_type=call_type,
            paper_id=paper_id,
            messages=messages,
            schema=schema_obj,
            # In hybrid mode the rendered pages remain available for a later
            # crop, but the main request uploads no image at all.  Keep those
            # two facts separate in the event log.
            pages=() if text is not None else pages,
            available_pages=pages,
            input_mode="page_marked_text" if text is not None else "full_page_images",
            # Invalid visual requests are sanitized before schema repair so
            # they are recorded locally rather than spending a repair call.
            # Destructive cross-reference cleanup waits until shape is valid.
            preprocess=_sanitize_visual_requests,
            postprocess=_normalize_report,
        )

    def _call_patch(
        self,
        *,
        paper_id: str,
        candidate: Mapping[str, Any],
        pages: Sequence[PageImage],
        original_pages: Sequence[PageImage],
        visual_reasons: Sequence[str],
        title: str,
        prompt: str | None,
    ) -> _ResponseResult:
        """Run the refinement call, which returns a patch rather than a report."""

        schema_obj = refine_patch_schema()
        messages = self._build_patch_messages(
            paper_id=paper_id,
            candidate=candidate,
            pages=pages,
            original_pages=original_pages,
            visual_reasons=visual_reasons,
            title=title,
            prompt=prompt,
            schema_obj=schema_obj,
        )
        return self._call_json(
            call_type="visual_refinement",
            paper_id=paper_id,
            messages=messages,
            schema=schema_obj,
            pages=list(original_pages) + list(pages),
            available_pages=original_pages,
            input_mode="visual_refinement_images",
        )

    def _call_json(
        self,
        *,
        call_type: str,
        paper_id: str,
        messages: list[dict[str, Any]],
        schema: Mapping[str, Any] | None,
        pages: Sequence[PageImage] = (),
        available_pages: Sequence[PageImage] = (),
        input_mode: str = "grounded_report",
        preprocess: Callable[[dict[str, Any]], None] | None = None,
        postprocess: Callable[[dict[str, Any]], None] | None = None,
    ) -> _ResponseResult:
        """Send one logical call and return its parsed, repaired JSON payload.

        Budget, retry, truncation and format-repair policy live here so that
        every stage is charged and logged the same way.  Schema shape is
        repaired before ``postprocess`` performs destructive semantic cleanup:
        otherwise an alias such as ``method.components`` can make valid edges
        look dangling and erase the very material a repair needs to preserve.
        """

        if self._request_count >= self.config.max_requests_per_paper:
            error = ReaderAPIError(
                f"Request budget exhausted for {paper_id}: maximum "
                f"{self.config.max_requests_per_paper} requests per paper"
            )
            error.requests_made = self._request_count
            raise error
        request_id = uuid4().hex
        record = CallRecord(
            call_type=call_type,
            request_id=request_id,
            provided_pages=sorted({page.pdf_page for page in pages}),
            provided_images=[page.stable_id() for page in pages],
        )
        started = time.monotonic()
        record.started_at = _now()
        thinking_budget = self.config.thinking_budget_for(call_type)
        self._emit(
            {
                "event": "call_started",
                "call_type": call_type,
                "request_id": request_id,
                "paper_id": paper_id,
                "model": self.config.model,
                "input_mode": input_mode,
                "text_chars": _message_text_chars(messages),
                "pages": record.provided_pages,
                "images": record.provided_images,
                "available_pages": sorted({page.pdf_page for page in available_pages}),
                "thinking_budget": thinking_budget,
                "request_timeout_s": self.config.request_timeout_s,
                "started_at": record.started_at,
            }
        )

        last_error: Exception | None = None
        max_attempts = self.config.max_retries_per_call + 1
        for attempt in range(1, max_attempts + 1):
            if self._request_count >= self.config.max_requests_per_paper:
                error = ReaderAPIError(
                    f"Request budget exhausted for {paper_id}: maximum "
                    f"{self.config.max_requests_per_paper} requests per paper"
                )
                error.requests_made = self._request_count
                raise error
            record.attempts = attempt
            self._request_count += 1
            request_sequence = self._request_count
            attempt_started = time.monotonic()
            self._emit(
                {
                    "event": "request_attempt_started",
                    "call_type": call_type,
                    "request_id": request_id,
                    "paper_id": paper_id,
                    "model": self.config.model,
                    "attempt": attempt,
                    "request_sequence": request_sequence,
                    "thinking_budget": thinking_budget,
                    "request_timeout_s": self.config.request_timeout_s,
                    "started_at": _now(),
                }
            )
            response: Any = None
            try:
                response = self._create_completion(messages, call_type=call_type)
                # Capture provider metadata even when the content is
                # truncated or malformed, so the event log explains why the
                # logical call was rejected.
                record.response_id = _response_attr(response, "id")
                record.finish_reason = _finish_reason(response)
                record.usage = _usage_dict(_response_attr(response, "usage"))
                self._emit(
                    {
                        "event": "response_received",
                        "call_type": call_type,
                        "request_id": request_id,
                        "paper_id": paper_id,
                        "model": self.config.model,
                        "attempt": attempt,
                        "request_sequence": request_sequence,
                        "response_id": record.response_id,
                        "finish_reason": record.finish_reason,
                        "usage": record.usage,
                        "duration_s": round(time.monotonic() - attempt_started, 3),
                        "received_at": _now(),
                    }
                )
                payload = self._parse_response(response, schema)
                if preprocess is not None:
                    preprocess(payload)
                parsed = payload
                payload = self._repair_schema_errors(
                    payload,
                    schema=schema,
                    paper_id=paper_id,
                    call_type=call_type,
                    postprocess=None,
                )
                remaining_errors = schema_errors(payload, schema)
                if remaining_errors:
                    self._emit(
                        {
                            "event": "schema_repair_rejected",
                            "call_type": call_type,
                            "paper_id": paper_id,
                            "errors": remaining_errors[:20],
                        }
                    )
                    raise ReaderResponseError(
                        "Model response still violates the schema after format repair: "
                        + "; ".join(remaining_errors[:5])
                    )
                if postprocess is not None:
                    postprocess(payload)
                # Identity, not a counter: a repair round that ran but whose
                # result was rejected leaves the parsed report in place, and
                # that report must not be labelled as repaired.
                record.format_repaired = payload is not parsed
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
                return _ResponseResult(payload=payload, response=response, record=record)
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
                        exc,
                        schema=schema,
                        paper_id=paper_id,
                        call_type=call_type,
                        postprocess=postprocess,
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
                    return _ResponseResult(payload=repaired, response=response, record=record)
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
                exc.requests_made = self._request_count
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
                    error = ReaderAPIError(
                        f"{call_type} call failed after {attempt} attempt(s): {_safe_error(exc)}"
                    )
                    error.requests_made = self._request_count
                    raise error from exc
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
        error = ReaderAPIError(f"{call_type} call failed: {last_error}")
        error.requests_made = self._request_count
        raise error from last_error

    def _repair_schema_errors(
        self,
        payload: dict[str, Any],
        *,
        schema: Mapping[str, Any] | None,
        paper_id: str,
        call_type: str,
        postprocess: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Give the model one text-only chance to fix structurally invalid output.

        Only shape is repaired here.  Semantic checks (unknown evidence IDs,
        dangling method edges, numbers that contradict a page) are deliberately
        left to ``validate.py`` and to human review, and truncated output is
        never treated as a formatting problem.

        Before spending a repair request, known mechanical violations are
        normalised locally.  A repair result is also normalised before it is
        accepted, so a payload that is one dropped ``label`` or one filled
        ``target`` away from valid is not thrown away.
        """

        if schema:
            local_fixes = normalize_report_schema_shape(payload, schema)
            if local_fixes:
                self._emit(
                    {
                        "event": "schema_shape_normalized",
                        "call_type": call_type,
                        "paper_id": paper_id,
                        "fixes": local_fixes,
                        "source": "original",
                    }
                )
        errors = schema_errors(payload, schema)
        if not errors:
            return payload
        repaired = self._request_repair(
            reason="模型返回的 JSON 不符合本次要求的 schema，请修正结构。",
            errors=errors,
            payload=json.dumps(payload, ensure_ascii=False),
            schema=schema,
            paper_id=paper_id,
            call_type=call_type,
        )
        if repaired is None:
            return payload
        if schema:
            repair_fixes = normalize_report_schema_shape(repaired, schema)
            if repair_fixes:
                self._emit(
                    {
                        "event": "schema_shape_normalized",
                        "call_type": call_type,
                        "paper_id": paper_id,
                        "fixes": repair_fixes,
                        "source": "repair",
                    }
                )
        if schema_errors(repaired, schema):
            # A repair that is still invalid is not an improvement; keep the
            # original so the failure is reported against the real output.
            return payload
        regressions = _repair_regressions(payload, repaired)
        if regressions:
            self._emit(
                {
                    "event": "repair_rejected",
                    "call_type": "format_repair",
                    "parent_call_type": call_type,
                    "paper_id": paper_id,
                    "reason": "content_regression",
                    "regressions": regressions,
                }
            )
            return payload
        # The schema cannot express every problem (a crop with no area passes
        # it), so the repaired payload is reconciled as well.
        if postprocess is not None:
            postprocess(repaired)
        return repaired

    def _repair_unparsable(
        self,
        error: ReaderResponseError,
        *,
        schema: Mapping[str, Any] | None,
        paper_id: str,
        call_type: str,
        postprocess: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any] | None:
        """Try one text-only repair of a response whose JSON could not be parsed."""

        raw = getattr(error, "raw_content", None)
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
        if schema:
            repair_fixes = normalize_report_schema_shape(repaired, schema)
            if repair_fixes:
                self._emit(
                    {
                        "event": "schema_shape_normalized",
                        "call_type": call_type,
                        "paper_id": paper_id,
                        "fixes": repair_fixes,
                        "source": "unparsable_repair",
                    }
                )
        if schema_errors(repaired, schema):
            return None
        if postprocess is not None:
            postprocess(repaired)
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

        repairs_used = self._format_repairs_used.get(call_type, 0)
        if repairs_used >= self.config.max_format_repairs:
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
        self._format_repairs_used[call_type] = repairs_used + 1
        self._request_count += 1

        parts = [
            "本次不提供任何页面图片，只做结构修复。不要新增、删除或改写事实、数字、"
            "证据条目与方法关系；只修正 JSON 结构与字段类型，其它内容保持原样。",
            "结构修复不得减少任何已有数组的条目数。若输入中的 method.components 已被"
            "迁移为 method.nodes，必须保留每个 node、edge 和 step，并为它们填写至少一个"
            "已经存在于 evidence 数组中的 evidence id；禁止用空 evidence_ids 规避校验。",
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
        messages = [
            {
                "role": "system",
                "content": "Return valid JSON only. Do not add content that was not already present.",
            },
            {"role": "user", "content": [{"type": "text", "text": "\n\n".join(parts)}]},
        ]
        thinking_budget = self.config.thinking_budget_for("format_repair")
        self._emit(
            {
                "event": "repair_started",
                "call_type": "format_repair",
                "parent_call_type": call_type,
                "request_id": request_id,
                "paper_id": paper_id,
                "model": self.config.model,
                "errors": list(errors[:20]),
                "input_mode": "repair_text",
                "text_chars": _message_text_chars(messages),
                "pages": [],
                "images": [],
                "thinking_budget": thinking_budget,
                "request_timeout_s": self.config.request_timeout_s,
                "started_at": _now(),
            }
        )
        self._emit(
            {
                "event": "request_attempt_started",
                "call_type": "format_repair",
                "parent_call_type": call_type,
                "request_id": request_id,
                "paper_id": paper_id,
                "model": self.config.model,
                "attempt": 1,
                "request_sequence": self._request_count,
                "thinking_budget": thinking_budget,
                "request_timeout_s": self.config.request_timeout_s,
                "started_at": _now(),
            }
        )
        try:
            response = self._create_completion(messages, call_type="format_repair")
            self._emit(
                {
                    "event": "response_received",
                    "call_type": "format_repair",
                    "parent_call_type": call_type,
                    "request_id": request_id,
                    "paper_id": paper_id,
                    "model": self.config.model,
                    "attempt": 1,
                    "request_sequence": self._request_count,
                    "response_id": _response_attr(response, "id"),
                    "finish_reason": _finish_reason(response),
                    "usage": _usage_dict(_response_attr(response, "usage")),
                    "duration_s": round(time.monotonic() - started, 3),
                    "received_at": _now(),
                }
            )
            content = _message_content(response)
            if content is None or not str(content).strip():
                raise ReaderResponseError("repair response contained no content")
            repaired = parse_json_object(str(content))
        except Exception as exc:  # noqa: BLE001 - repair failure is not fatal
            self._emit(
                {
                    "event": "repair_failed",
                    "call_type": "format_repair",
                    "parent_call_type": call_type,
                    "request_id": request_id,
                    "paper_id": paper_id,
                    "error_type": type(exc).__name__,
                    "error": _safe_error(exc),
                    "duration_s": round(time.monotonic() - started, 3),
                    "ended_at": _now(),
                }
            )
            return None
        self._emit(
            {
                "event": "repair_finished",
                "call_type": "format_repair",
                "parent_call_type": call_type,
                "request_id": request_id,
                "paper_id": paper_id,
                "duration_s": round(time.monotonic() - started, 3),
                "usage": _usage_dict(_response_attr(response, "usage")),
                "ended_at": _now(),
            }
        )
        return repaired

    def _create_completion(self, messages: list[dict[str, Any]], *, call_type: str) -> Any:
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
            extra_body: dict[str, Any] = {"enable_thinking": self.config.thinking}
            thinking_budget = self.config.thinking_budget_for(call_type)
            if thinking_budget is not None:
                extra_body["thinking_budget"] = thinking_budget
            kwargs["extra_body"] = extra_body
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

    def _build_reading_messages(
        self,
        *,
        paper_id: str,
        pages: Sequence[PageImage],
        text: str | None,
        metadata: Mapping[str, Any] | None,
        title: str,
        prompt: str | None,
        schema_obj: Mapping[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Build the main reading call.

        With ``text`` the call carries the page-marked full text and no image
        at all; without it the call falls back to the pre-text-first layout of
        one labelled image per page, which is what the A/B baseline needs.
        """

        instructions = prompt if prompt is not None else self._load_prompt()
        prefix = _reading_prefix_text(
            paper_id=paper_id,
            title=title,
            instructions=instructions,
            metadata=metadata,
            text=text,
            pages=pages,
            # The main reader must not see the diagram-only ``label`` field:
            # it repeatedly copied that field into method edges even though
            # those edges deliberately use a different contract.
            schema_text=_schema_text(_reading_schema_for_prompt(schema_obj)),
        )
        # The diagram stage replays this exact prefix so the endpoint's cache
        # can serve it.  An image-only reading is not replayable (the images
        # would have to be re-uploaded), so it is not remembered either.
        self._last_reading_prefix = prefix if text is not None else None

        content: list[dict[str, Any]] = [{"type": "text", "text": prefix}]
        if text is None:
            for page in pages:
                content.append(
                    {"type": "text", "text": f"PDF_PAGE={page.pdf_page} IMAGE_ID={page.stable_id()}"}
                )
                content.append({"type": "image_url", "image_url": {"url": image_data_uri(page.path)}})
        return [
            {"role": "system", "content": READER_SYSTEM_MESSAGE},
            {"role": "user", "content": content},
        ]

    def _build_patch_messages(
        self,
        *,
        paper_id: str,
        candidate: Mapping[str, Any],
        pages: Sequence[PageImage],
        original_pages: Sequence[PageImage],
        visual_reasons: Sequence[str],
        title: str,
        prompt: str | None,
        schema_obj: Mapping[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Build the refinement call, which returns a patch for the candidate."""

        instructions = prompt if prompt is not None else self._load_refine_prompt()
        schema_text = _schema_text(schema_obj)
        coverage = ", ".join(str(page.pdf_page) for page in pages)
        text_parts = [
            instructions,
            f"paper_id: {paper_id}",
            f"title: {title}" if title else "title: (already read in the first call)",
            "This is a visual refinement round, not a first reading. The original pages and "
            f"enlarged crops that follow are re-sent for verification only: PDF pages {coverage}. "
            "The rest of the paper was supplied in the first call; do not treat it as missing "
            "material, and do not restate anything these images do not change.",
            "Treat each PDF_PAGE label immediately before an image as authoritative evidence "
            "identity. IMAGE_ID values name the exact image supplied, including crops; a new "
            "evidence entry may cite the crop it was read from.",
        ]
        if schema_text:
            text_parts.append("Return one JSON object conforming to this JSON Schema:\n" + schema_text)
        reasons = "\n".join(f"- {reason}" for reason in visual_reasons if reason)
        text_parts.append(
            "Reasons for requesting visual review:\n" + (reasons or "- (none recorded)")
        )
        text_parts.append("Candidate report JSON:\n" + json.dumps(candidate, ensure_ascii=False))

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
            {"role": "system", "content": READER_SYSTEM_MESSAGE},
            {"role": "user", "content": content},
        ]

    def _build_diagram_messages(
        self,
        *,
        paper_id: str,
        report: Mapping[str, Any],
        instructions: str,
        schema_obj: Mapping[str, Any] | None,
        prefix: str | None,
    ) -> list[dict[str, Any]]:
        """Build the Mermaid stage.

        With ``prefix`` the call reuses the reading call's prompt prefix,
        including the page-marked full text and the report schema, and appends
        the diagram task after it: the endpoint caches the shared prefix, so
        the text is not paid for twice.  Without it the call is small and
        self-contained and only carries the grounded report data.
        """

        schema_text = _schema_text(schema_obj)
        task_parts = [instructions, f"paper_id: {paper_id}"]
        if prefix:
            task_parts.append(
                "Everything above belongs to the reading call that has already happened: the "
                "page-marked full text and the report schema it was asked for. The report that "
                "call produced is quoted below. Do not read the paper again, do not re-derive a "
                "fact from it, and do not return a report: draw the diagrams from the grounded "
                "report data below and return one JSON object conforming to the diagram schema "
                "that follows."
            )
        if schema_text:
            task_parts.append("Return one JSON object conforming to this JSON Schema:\n" + schema_text)
        task_parts.append(
            "Grounded report data:\n"
            + json.dumps(_diagram_source(report), ensure_ascii=False, sort_keys=True)
        )
        task = "\n\n".join(task_parts)
        return [
            {"role": "system", "content": READER_SYSTEM_MESSAGE},
            {"role": "user", "content": [{"type": "text", "text": f"{prefix}\n\n{task}" if prefix else task}]},
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

    def _load_diagram_prompt(self) -> str:
        """Load the diagram-stage instructions, falling back to the built-in ones."""

        path = self.diagram_prompt_path
        if path is None and self.prompt_path is not None:
            sibling = self.prompt_path.parent / "diagram.md"
            path = sibling if sibling.exists() else None
        if path is None:
            candidate = Path.cwd() / "prompts" / "diagram.md"
            path = candidate if candidate.exists() else None
        if path is None:
            return DEFAULT_DIAGRAM_INSTRUCTIONS
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ReaderConfigError(f"Could not read diagram prompt {path}: {exc}") from exc
        return value or DEFAULT_DIAGRAM_INSTRUCTIONS

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


def _resolve_schema_node(schema: Mapping[str, Any], node: Any) -> Any:
    """Follow a single ``#/$defs/...`` reference inside one schema document."""

    if not isinstance(node, Mapping) or "$ref" not in node:
        return node
    ref = node.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
        return node
    name = ref.rsplit("/", 1)[-1]
    defs = schema.get("$defs")
    if isinstance(defs, Mapping):
        return defs.get(name, {})
    return {}


def normalize_report_schema_shape(
    report: MutableMapping[str, Any],
    schema: Mapping[str, Any] | None,
) -> list[str]:
    """Mechanically coerce a report toward the JSON Schema without inventing facts.

    Only two local fixes are applied:

    - drop object keys forbidden by ``additionalProperties: false`` (the
      method-edge / diagram-edge ``label`` confusion is the common case);
    - fill a missing required *string* property with ``""`` when that is a
      legal empty value (``guide_item.target`` is the common case).

    Missing required objects, arrays, booleans, or enum-backed strings are
    left alone: inventing them would fabricate structure the model never
    produced.  Every change is returned as a readable note so the event log
    can record it.  The operation mutates ``report`` in place and is
    idempotent.
    """

    if not schema or not isinstance(report, MutableMapping):
        return []
    fixes: list[str] = []

    def walk(value: Any, schema_node: Any, path: str) -> None:
        schema_node = _resolve_schema_node(schema, schema_node)
        if not isinstance(schema_node, Mapping):
            return
        if isinstance(value, MutableMapping):
            properties = schema_node.get("properties")
            if not isinstance(properties, Mapping):
                return
            # An earlier Q3 prompt called method nodes "components", and real
            # responses consequently used that prose term as a JSON key. This
            # migration preserves the objects verbatim before the generic
            # additional-properties cleanup would otherwise delete them.
            if (
                path == "$.method"
                and "components" in value
                and "nodes" in properties
                and (
                    "nodes" not in value
                    or (isinstance(value.get("nodes"), list) and not value.get("nodes"))
                )
                and isinstance(value.get("components"), list)
            ):
                value["nodes"] = value.pop("components")
                fixes.append("本地结构归一：$.method.components 已迁移为 $.method.nodes")
            if schema_node.get("additionalProperties") is False:
                allowed = set(properties)
                for key in list(value):
                    if key not in allowed:
                        del value[key]
                        fixes.append(f"本地结构归一：{path}.{key} 不在 schema 中，已删除")
            required = schema_node.get("required")
            if isinstance(required, list):
                for key in required:
                    if key in value or key not in properties:
                        continue
                    prop = _resolve_schema_node(schema, properties.get(key))
                    if not isinstance(prop, Mapping):
                        continue
                    if prop.get("type") != "string":
                        continue
                    enum = prop.get("enum")
                    if isinstance(enum, list) and "" not in enum:
                        continue
                    value[key] = ""
                    fixes.append(f"本地结构归一：{path}.{key} 缺失，已填空字符串")
            for key, item in value.items():
                if key in properties:
                    walk(item, properties[key], f"{path}.{key}")
        elif isinstance(value, list):
            items = schema_node.get("items")
            if items is not None:
                for index, item in enumerate(value):
                    walk(item, items, f"{path}[{index}]")

    walk(report, schema, "$")
    return fixes


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


def _input_mode(value: Any) -> str:
    """Validate ``input.mode`` rather than silently falling back.

    The mode decides whether the main call uploads every page image, so a
    typo in it must stop the run: quietly reading it as the default would
    change the cost of every paper in the batch.
    """

    mode = str(value).strip() if isinstance(value, str) else ""
    if mode not in INPUT_MODES:
        raise ReaderConfigError(
            f"input.mode must be one of {', '.join(INPUT_MODES)}; got {value!r}"
        )
    return mode


def _thinking_budgets(value: Any) -> dict[str, int]:
    """Validate the project-level per-stage wrapper for DashScope budgets.

    A scalar applies to every stage for convenience.  A mapping may override
    individual stages while retaining the conservative defaults for omitted
    ones.  Unknown stage names are rejected because a typo would silently
    restore Qwen3.8's much larger provider default.
    """

    if value is None:
        return dict(DEFAULT_THINKING_BUDGETS)
    if isinstance(value, int) and not isinstance(value, bool):
        budget = _positive_int(value, "model.thinking_budget")
        return {stage: budget for stage in THINKING_BUDGET_STAGES}
    if not isinstance(value, Mapping):
        raise ReaderConfigError(
            "model.thinking_budget must be an integer or a per-stage mapping"
        )
    unknown = sorted(str(key) for key in value if key not in THINKING_BUDGET_STAGES)
    if unknown:
        raise ReaderConfigError(
            "model.thinking_budget contains unknown stages: " + ", ".join(unknown)
        )
    budgets = dict(DEFAULT_THINKING_BUDGETS)
    for stage in THINKING_BUDGET_STAGES:
        if stage in value:
            budgets[stage] = _positive_int(
                value[stage], f"model.thinking_budget.{stage}"
            )
    return budgets


def _bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        raise ReaderConfigError(f"{field_name} must be true or false")
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ReaderConfigError(f"{field_name} must be true or false")


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


def _message_text_chars(messages: Sequence[Mapping[str, Any]]) -> int:
    """Count text sent in a request without inspecting or logging its content."""

    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
            for part in content:
                if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                    total += len(part["text"])
    return total


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


def _finalize(result: ReadResult, requests_made: int | None = None) -> ReadResult:
    """Make ``unresolved_items`` agree with the report that is returned."""

    if requests_made is not None:
        result.requests_made = requests_made
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


def _schema_text(schema: Mapping[str, Any] | None) -> str:
    """Render a schema for a prompt, compactly and deterministically."""

    if not schema:
        return ""
    try:
        return json.dumps(schema, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError):
        return ""


def _reading_schema_for_prompt(
    schema: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Remove diagram-only definitions from the schema shown to the reader.

    Diagrams are produced by a separate grounded stage. Keeping their
    ``label``-bearing edge definition in the main prompt caused the model to
    leak ``label`` into ``method.edges``, where it is forbidden. Validation
    still uses the complete report schema after diagrams are attached.
    """

    if not schema:
        return schema
    prompt_schema = copy.deepcopy(dict(schema))
    properties = prompt_schema.get("properties")
    if isinstance(properties, MutableMapping):
        properties.pop("diagrams", None)
    definitions = prompt_schema.get("$defs")
    if isinstance(definitions, MutableMapping):
        for name in ("diagram", "diagram_node", "diagram_edge"):
            definitions.pop(name, None)
    return prompt_schema


_REPAIR_COLLECTION_PATHS = (
    ("related_work",),
    ("experiments",),
    ("claims",),
    ("evidence",),
    ("method", "nodes"),
    ("method", "edges"),
    ("method", "steps"),
    ("method", "intermediate_artifacts"),
    ("method", "tools_and_models"),
    ("method", "feedback_loops"),
    ("method", "decisions"),
    ("method", "implementation_details"),
    ("future_work", "authors_limitations"),
    ("future_work", "authors_future_work"),
    ("future_work", "open_questions"),
    ("future_work", "research_directions"),
    ("reading_guide", "key_sections"),
    ("reading_guide", "key_concepts"),
    ("reading_guide", "open_questions"),
    ("reading_guide", "reproduction_notes"),
    ("reading_guide", "related_directions"),
)


def _nested_value(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = value
    for name in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(name)
    return current


def _repair_regressions(
    original: Mapping[str, Any], repaired: Mapping[str, Any]
) -> list[str]:
    """Reject a structural repair that silently drops report content.

    A repair may reshape entries, but it was explicitly instructed not to
    delete facts. Collection cardinality is a conservative, deterministic
    guard against the observed failure where two components and six steps
    became empty arrays merely to satisfy the schema.
    """

    regressions: list[str] = []
    for path in _REPAIR_COLLECTION_PATHS:
        before = _nested_value(original, path)
        after = _nested_value(repaired, path)
        if not isinstance(before, list) or not before:
            continue
        after_count = len(after) if isinstance(after, list) else 0
        if after_count < len(before):
            regressions.append(
                f"$.{'.'.join(path)} 从 {len(before)} 项减少到 {after_count} 项"
            )
    return regressions


def _metadata_text(metadata: Mapping[str, Any] | None) -> str:
    if not isinstance(metadata, Mapping) or not metadata:
        return ""
    try:
        return json.dumps(dict(metadata), ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return ""


def _reading_prefix_text(
    *,
    paper_id: str,
    title: str,
    instructions: str,
    metadata: Mapping[str, Any] | None,
    text: str | None,
    pages: Sequence[PageImage],
    schema_text: str,
) -> str:
    """The exact user-message prefix every text-first stage starts with.

    Pure in its arguments, and that is the point: the diagram stage replays
    this string byte for byte so the endpoint's prefix cache can serve it, and
    a prefix that differed by one character would be charged again in full.
    """

    parts = [instructions, f"paper_id: {paper_id}"]
    parts.append(f"title: {title}" if title else "title: (read from the supplied material)")
    metadata_text = _metadata_text(metadata)
    if metadata_text:
        parts.append("Paper metadata:\n" + metadata_text)
    if text is not None:
        parts.append(
            "The complete extracted text of the paper follows, page by page, each page "
            "introduced by a --- PDF_PAGE=N --- marker. Read the paper from this text."
        )
        parts.append(text)
    else:
        coverage = ", ".join(str(page.pdf_page) for page in pages)
        parts.append(
            f"The following images are the complete input for this call. PDF pages: {coverage}."
        )
        parts.append(
            "Treat each PDF_PAGE label immediately before an image as authoritative evidence "
            "identity. IMAGE_ID values name the exact image supplied, including crops; an "
            "evidence entry may cite the crop it was read from."
        )
    if schema_text:
        parts.append("Return one JSON object conforming to this JSON Schema:\n" + schema_text)
    return "\n\n".join(parts)


# The only method fields a diagram may be drawn from.  Reported results are
# deliberately absent: a diagram shows how the method works, not what it
# scored, and the experiments are passed in their own compact form below.
_DIAGRAM_METHOD_FIELDS = (
    "applicable",
    "overview",
    "inputs",
    "outputs",
    "intermediate_artifacts",
    "tools_and_models",
    "nodes",
    "edges",
    "steps",
    "feedback_loops",
    "decisions",
    "implementation_details",
)


def _diagram_source(report: Mapping[str, Any]) -> dict[str, Any]:
    """The grounded subset of a report that the diagram stage may read.

    Only facts that already survived reading are handed over, together with
    the evidence entries they cite: enough to draw the method, and nothing
    that would let the stage re-decide the paper instead of drawing it.
    """

    method = report.get("method")
    method = dict(method) if isinstance(method, Mapping) else {}
    grounded: dict[str, Any] = {
        "title": _text_value(report.get("title")),
        "method": {name: method[name] for name in _DIAGRAM_METHOD_FIELDS if name in method},
        "experiments": [
            {
                "index": index,
                "purpose": experiment.get("purpose", ""),
                "dataset": experiment.get("dataset", ""),
                "main_results": experiment.get("main_results", []),
                "conclusion": experiment.get("conclusion", ""),
                "evidence_ids": experiment.get("evidence_ids", []),
            }
            for index, experiment in enumerate(_mappings(report.get("experiments")))
        ],
        "unresolved_items": _unresolved_texts(report),
    }
    wanted = _referenced_evidence_ids(grounded)
    grounded["evidence"] = [
        {
            "id": entry.get("id"),
            "source_type": entry.get("source_type"),
            "pdf_page": entry.get("pdf_page"),
            "section": entry.get("section"),
            "figure_or_table": entry.get("figure_or_table"),
        }
        for entry in _mappings(report.get("evidence"))
        if str(entry.get("id")) in wanted
    ]
    return grounded


def _referenced_evidence_ids(value: Any) -> set[str]:
    """Every evidence id mentioned anywhere inside a nested report fragment."""

    found: set[str] = set()
    stack: list[Any] = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            for key, item in current.items():
                if key == "evidence_ids" and isinstance(item, list):
                    found.update(str(entry) for entry in item)
                elif isinstance(item, (Mapping, list)):
                    stack.append(item)
        elif isinstance(current, list):
            stack.extend(item for item in current if isinstance(item, (Mapping, list)))
    return found


def _artifact_defs() -> dict[str, Any]:
    """The nested method-entry shapes a patch is allowed to append."""

    evidence_ids = {"type": "array", "items": {"type": "string"}}
    return {
        "artifact": {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "description", "evidence_ids"],
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "evidence_ids": evidence_ids,
            },
        },
        "tool_model": {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "role", "evidence_ids"],
            "properties": {
                "name": {"type": "string"},
                "role": {"type": "string"},
                "evidence_ids": evidence_ids,
            },
        },
        "feedback_loop": {
            "type": "object",
            "additionalProperties": False,
            "required": ["description", "node_ids", "evidence_ids"],
            "properties": {
                "description": {"type": "string"},
                "node_ids": {"type": "array", "items": {"type": "string"}},
                "evidence_ids": evidence_ids,
            },
        },
        "decision": {
            "type": "object",
            "additionalProperties": False,
            "required": ["condition", "branches", "evidence_ids"],
            "properties": {
                "condition": {"type": "string"},
                "branches": {"type": "array", "items": {"type": "string"}},
                "evidence_ids": evidence_ids,
            },
        },
        "implementation_detail": {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "value", "evidence_ids"],
            "properties": {
                "name": {"type": "string"},
                "value": {"type": "string"},
                "evidence_ids": evidence_ids,
            },
        },
    }


def refine_patch_schema() -> dict[str, Any]:
    """The schema of a refinement patch.

    The patch never enters the report -- this module merges it -- so its shape
    lives here rather than in ``report.schema.json``.  Every array is required
    so that a missing section is a visible contract violation instead of a
    silently ignored update.
    """

    defs = _artifact_defs()
    defs.update(
        {
            "evidence_update": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id"],
                "properties": {
                    "id": {"type": "string"},
                    "source_type": {"type": "string", "enum": list(_EVIDENCE_SOURCE_TYPES)},
                    "pdf_page": {"type": "integer", "minimum": 1},
                    "source_id": {"type": "string"},
                    "section": {"type": "string"},
                    "figure_or_table": {"type": "string"},
                    "locator": {"type": "string"},
                    "quote": {"type": "string"},
                },
            },
            "claim_update": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id"],
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "kind": {"type": "string", "enum": list(_CLAIM_KINDS)},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
            "method_update": {
                "type": "object",
                "additionalProperties": False,
                "required": ["target"],
                "properties": {
                    "target": {"type": "string", "enum": ["method", "node", "edge", "step"]},
                    "id": {"type": "string"},
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "step": {"type": "string"},
                    "name": {"type": "string"},
                    "input": {"type": ["string", "array"], "items": {"type": "string"}},
                    "operation": {"type": "string"},
                    "output": {"type": ["string", "array"], "items": {"type": "string"}},
                    "tool_or_model": {"type": "string"},
                    "why_needed": {"type": "string"},
                    "relation": {"type": "string", "enum": list(_METHOD_EDGE_RELATIONS)},
                    "confirmed": {"type": "boolean"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                    "overview": {"type": "string"},
                    "applicable": {"type": "boolean"},
                    "inputs": {"type": "array", "items": {"type": "string"}},
                    "outputs": {"type": "array", "items": {"type": "string"}},
                    "intermediate_artifacts": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/artifact"},
                    },
                    "tools_and_models": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/tool_model"},
                    },
                    "feedback_loops": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/feedback_loop"},
                    },
                    "decisions": {"type": "array", "items": {"$ref": "#/$defs/decision"}},
                    "implementation_details": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/implementation_detail"},
                    },
                },
            },
            "experiment_update": {
                "type": "object",
                "additionalProperties": False,
                "required": ["index"],
                "properties": {
                    "index": {"type": "integer", "minimum": 0},
                    "purpose": {"type": "string"},
                    "research_question": {"type": "string"},
                    "dataset": {"type": "string"},
                    "sample_size": {"type": "string"},
                    "baselines": {"type": "array", "items": {"type": "string"}},
                    "models": {"type": "array", "items": {"type": "string"}},
                    "metrics": {"type": "array", "items": {"type": "string"}},
                    "settings": {"type": "string"},
                    "main_results": {"type": "array", "items": {"type": "string"}},
                    "conclusion": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
            "resolved_request": {
                "type": "object",
                "additionalProperties": False,
                "required": ["pdf_page"],
                "properties": {
                    "pdf_page": {"type": "integer", "minimum": 1},
                    "crop": {
                        "type": ["object", "null"],
                        "additionalProperties": False,
                        "required": ["x0", "y0", "x1", "y1"],
                        "properties": {
                            "x0": {"type": "number"},
                            "y0": {"type": "number"},
                            "x1": {"type": "number"},
                            "y1": {"type": "number"},
                        },
                    },
                    "note": {"type": "string"},
                },
            },
        }
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": list(PATCH_SECTIONS),
        "properties": {
            "evidence_updates": {"type": "array", "items": {"$ref": "#/$defs/evidence_update"}},
            "claim_updates": {"type": "array", "items": {"$ref": "#/$defs/claim_update"}},
            "method_updates": {"type": "array", "items": {"$ref": "#/$defs/method_update"}},
            "experiment_updates": {"type": "array", "items": {"$ref": "#/$defs/experiment_update"}},
            "resolved_visual_requests": {
                "type": "array",
                "items": {"$ref": "#/$defs/resolved_request"},
            },
            "unresolved_items_add": {"type": "array", "items": {"type": "string"}},
        },
        "$defs": defs,
    }


_DIAGRAM_DEF_NAMES = ("diagram", "diagram_node", "diagram_edge", "evidence_ids")


def _diagram_defs() -> dict[str, Any]:
    """Built-in diagram definitions, used only when there is no schema file."""

    evidence_ids = {
        "type": "array",
        "minItems": 1,
        "items": {"type": "string", "minLength": 1},
    }
    return {
        "evidence_ids": evidence_ids,
        "diagram": {
            "type": "object",
            "additionalProperties": False,
            "required": ["id", "title", "type", "nodes", "edges"],
            "properties": {
                "id": {"type": "string"},
                "title": {"type": "string"},
                "type": {"type": "string", "enum": ["flowchart"]},
                "direction": {"type": "string", "enum": ["LR", "TB"]},
                "nodes": {"type": "array", "items": {"$ref": "#/$defs/diagram_node"}},
                "edges": {"type": "array", "items": {"$ref": "#/$defs/diagram_edge"}},
            },
        },
        "diagram_node": {
            "type": "object",
            "additionalProperties": False,
            "required": ["id", "label", "kind", "evidence_ids"],
            "properties": {
                "id": {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_]*$"},
                "label": {"type": "string"},
                "kind": {"type": "string", "enum": list(_DIAGRAM_KINDS)},
                "group": {"type": "string"},
                "evidence_ids": evidence_ids,
            },
        },
        "diagram_edge": {
            "type": "object",
            "additionalProperties": False,
            "required": ["from", "to", "label", "relation", "confirmed", "evidence_ids"],
            "properties": {
                "from": {"type": "string"},
                "to": {"type": "string"},
                "label": {"type": "string"},
                "relation": {"type": "string", "enum": list(_METHOD_EDGE_RELATIONS)},
                "confirmed": {"type": "boolean"},
                "evidence_ids": evidence_ids,
            },
        },
    }


def diagram_response_schema(report_schema: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The schema of the diagram stage's output.

    The report schema already defines diagrams, so its definitions are reused
    whenever it is available: the stage and the report cannot then drift
    apart.  Only the diagram definitions are copied, because the whole report
    schema would be a large bill for a stage that needs four shapes.
    """

    available = report_schema.get("$defs") if isinstance(report_schema, Mapping) else None
    available = dict(available) if isinstance(available, Mapping) else {}
    defs = {name: available[name] for name in _DIAGRAM_DEF_NAMES if name in available}
    for name, value in _diagram_defs().items():
        defs.setdefault(name, value)
    # Diagram generation is stricter than storing an old report: every shape
    # produced now must be grounded. Override the report's compatibility
    # definition so a repair cannot satisfy the contract with empty arrays.
    defs["evidence_ids"] = _diagram_defs()["evidence_ids"]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["diagrams"],
        "properties": {"diagrams": {"type": "array", "items": {"$ref": "#/$defs/diagram"}}},
        "$defs": defs,
    }


def sanitize_diagrams(
    payload: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    max_diagrams: int = 4,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep the diagrams that are drawable from the report, drop the rest.

    Every node and edge is checked against the evidence the report actually
    holds, so a stage that invents a citation loses the edge instead of
    putting it in front of a reader.  Nothing here raises: a diagram is a view
    of the report, and a broken view is a missing view, not a broken paper.
    """

    known = {
        str(entry.get("id"))
        for entry in _mappings(report.get("evidence"))
        if entry.get("id")
    }
    diagrams: list[dict[str, Any]] = []
    problems: list[str] = []
    seen_ids: set[str] = set()
    values = payload.get("diagrams")
    candidates = list(values) if isinstance(values, list) else []
    if values is not None and not isinstance(values, list):
        problems.append(f"{DIAGRAM_OMITTED_PREFIX}diagrams 不是数组，已忽略")
        candidates = []
    for position, value in enumerate(candidates):
        if not isinstance(value, Mapping):
            problems.append(f"{DIAGRAM_OMITTED_PREFIX}第 {position + 1} 张图不是 JSON 对象")
            continue
        title = _text_value(value.get("title")) or f"方法图 {position + 1}"
        if len(diagrams) >= max_diagrams:
            problems.append(f"{DIAGRAM_OMITTED_PREFIX}超出 {max_diagrams} 张上限，忽略了「{title}」")
            continue
        diagram_id = _text_value(value.get("id")) or f"diagram-{position + 1}"
        if diagram_id in seen_ids:
            problems.append(f"{DIAGRAM_OMITTED_PREFIX}图 id 重复，忽略了「{title}」")
            continue
        seen_ids.add(diagram_id)

        nodes: list[dict[str, Any]] = []
        node_ids: set[str] = set()
        for raw_node in _mappings(value.get("nodes")):
            node_id = raw_node.get("id")
            if (
                not isinstance(node_id, str)
                or not _DIAGRAM_NODE_ID.match(node_id)
                or node_id in node_ids
            ):
                problems.append(f"{DIAGRAM_OMITTED_PREFIX}「{title}」忽略了 id 非法或重复的节点")
                continue
            node_ids.add(node_id)
            evidence = _known_evidence_ids(raw_node.get("evidence_ids"), known)
            if not evidence:
                problems.append(f"{DIAGRAM_OMITTED_PREFIX}「{title}」的节点 {node_id} 没有可用证据")
            kind = raw_node.get("kind")
            nodes.append(
                {
                    "id": node_id,
                    "label": _text_value(raw_node.get("label")) or node_id,
                    "kind": kind if kind in _DIAGRAM_KINDS else "component",
                    "group": _text_value(raw_node.get("group")),
                    "evidence_ids": evidence,
                }
            )
        if not nodes:
            problems.append(f"{DIAGRAM_OMITTED_PREFIX}「{title}」没有任何可用节点，整张图已省略")
            continue

        edges: list[dict[str, Any]] = []
        for raw_edge in _mappings(value.get("edges")):
            source = _text_value(raw_edge.get("from"))
            target = _text_value(raw_edge.get("to"))
            label = f"{source or '?'} → {target or '?'}"
            if source not in node_ids or target not in node_ids:
                problems.append(f"{DIAGRAM_OMITTED_PREFIX}「{title}」的边 {label} 引用了不存在的节点")
                continue
            evidence = _known_evidence_ids(raw_edge.get("evidence_ids"), known)
            confirmed = raw_edge.get("confirmed")
            # `confirmed` is the model's own flag and evidence is the ground
            # truth, so a flag that is simply missing is taken as confirmed
            # when the edge does carry the evidence for it.  An explicit
            # `false` is honoured: the model is allowed to say "do not draw
            # this yet".  An edge whose evidence the report does not hold is
            # dropped whatever the flag says: otherwise `confirmed: true`
            # would be a way around grounding.
            if confirmed is False or not evidence:
                problems.append(f"{DIAGRAM_OMITTED_PREFIX}「{title}」的边 {label} 未获证据确认")
                continue
            relation = raw_edge.get("relation")
            edges.append(
                {
                    "from": source,
                    "to": target,
                    "label": _text_value(raw_edge.get("label")),
                    "relation": relation if relation in _METHOD_EDGE_RELATIONS else "data_flow",
                    "confirmed": True,
                    "evidence_ids": evidence,
                }
            )
        direction = value.get("direction")
        diagrams.append(
            {
                "id": diagram_id,
                "title": title,
                "type": "flowchart",
                "direction": direction if direction in ("LR", "TB") else "LR",
                "nodes": nodes,
                "edges": edges,
            }
        )
    return diagrams, problems


def _known_evidence_ids(value: Any, known: Container[str]) -> list[str]:
    """The evidence ids of a diagram entry that the report can actually back."""

    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item) in known]


def _section_entries(container: MutableMapping[str, Any], key: str) -> list[Any]:
    value = container.get(key)
    if not isinstance(value, list):
        value = []
        container[key] = value
    return value


def _patch_entries(value: Any) -> list[Any]:
    """The entries of one patch array, keeping non-objects so they are refused."""

    return list(value) if isinstance(value, list) else []


def _text_list(value: Any) -> list[str] | None:
    """A list of strings, or ``None`` when the value cannot be one."""

    if not isinstance(value, list):
        return None
    items: list[str] = []
    for item in value:
        if isinstance(item, str):
            items.append(item)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            items.append(str(item))
        else:
            return None
    return items


def _structured_list(
    value: Any, required: Sequence[str], list_fields: Sequence[str] = ()
) -> list[dict[str, Any]] | None:
    """A list of method entries that carry every required key."""

    if not isinstance(value, list):
        return None
    items: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            return None
        if [name for name in required if name not in item]:
            return None
        entry = {name: item[name] for name in required}
        for name in list_fields:
            texts = _text_list(entry.get(name))
            if texts is None:
                return None
            entry[name] = texts
        items.append(entry)
    return items


def _collect_fields(
    value: Mapping[str, Any], spec: Mapping[str, str]
) -> tuple[dict[str, Any], str | None]:
    """Validate the fields a patch entry sets, one kind at a time."""

    fields: dict[str, Any] = {}
    for name, kind in spec.items():
        if name not in value:
            continue
        raw = value[name]
        if kind == "text":
            if not isinstance(raw, str):
                return {}, f"{name} 必须是字符串"
            fields[name] = raw
        elif kind == "text_list":
            items = _text_list(raw)
            if items is None:
                return {}, f"{name} 必须是字符串数组"
            fields[name] = items
        elif kind == "text_or_list":
            if isinstance(raw, str):
                fields[name] = [raw]
            else:
                items = _text_list(raw)
                if items is None:
                    return {}, f"{name} 必须是字符串或字符串数组"
                fields[name] = items
        elif kind == "text_from_list":
            # The report keeps a method step's input and output as one string
            # while a node's are lists, so a patch written for either shape is
            # folded into the shape the target entry actually has.
            if isinstance(raw, str):
                fields[name] = raw
            else:
                items = _text_list(raw)
                if items is None:
                    return {}, f"{name} 必须是字符串"
                fields[name] = "；".join(items)
        elif kind == "bool":
            if not isinstance(raw, bool):
                return {}, f"{name} 必须是布尔值"
            fields[name] = raw
        elif kind == "relation":
            if raw not in _METHOD_EDGE_RELATIONS:
                return {}, f"relation 必须是 {'/'.join(_METHOD_EDGE_RELATIONS)}"
            fields[name] = raw
        else:  # pragma: no cover - a spec typo, not a model error
            raise ReaderConfigError(f"Unknown patch field kind: {kind}")
    return fields, None


_NODE_PATCH_SPEC = {
    "name": "text",
    "input": "text_or_list",
    "operation": "text",
    "output": "text_or_list",
    "evidence_ids": "text_list",
}
_STEP_PATCH_SPEC = {
    "input": "text_from_list",
    "operation": "text",
    "tool_or_model": "text",
    "output": "text_from_list",
    "why_needed": "text",
    "evidence_ids": "text_list",
}
_EDGE_PATCH_SPEC = {
    "relation": "relation",
    "confirmed": "bool",
    "evidence_ids": "text_list",
}
_METHOD_PATCH_SPEC = {
    "overview": "text",
    "applicable": "bool",
    "inputs": "text_list",
    "outputs": "text_list",
}
_EXPERIMENT_PATCH_SPEC = {
    "purpose": "text",
    "research_question": "text",
    "dataset": "text",
    "sample_size": "text",
    "baselines": "text_list",
    "models": "text_list",
    "metrics": "text_list",
    "settings": "text",
    "main_results": "text_list",
    "conclusion": "text",
    "evidence_ids": "text_list",
}
# name -> (required keys, keys that hold lists)
_METHOD_APPEND_SPECS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "intermediate_artifacts": (("name", "description", "evidence_ids"), ("evidence_ids",)),
    "tools_and_models": (("name", "role", "evidence_ids"), ("evidence_ids",)),
    "feedback_loops": (("description", "node_ids", "evidence_ids"), ("node_ids", "evidence_ids")),
    "decisions": (("condition", "branches", "evidence_ids"), ("branches", "evidence_ids")),
    "implementation_details": (("name", "value", "evidence_ids"), ("evidence_ids",)),
}
_NODE_REQUIRED = ("id", "name", "input", "operation", "output", "evidence_ids")
_STEP_REQUIRED = ("step", "input", "operation", "tool_or_model", "output", "why_needed", "evidence_ids")
_EDGE_REQUIRED = ("relation", "confirmed", "evidence_ids")


def _reject(rejected: list[str], section: str, position: int, reason: str) -> None:
    rejected.append(f"{PATCH_REJECTED_PREFIX}{section}[{position}]：{reason}")


def _apply_patch(
    candidate: Mapping[str, Any],
    patch: Mapping[str, Any],
    *,
    executed: Container[Any] = (),
    supplied: Container[str] = (),
) -> tuple[dict[str, Any], list[str]]:
    """Merge a refinement patch into a candidate report.

    Deterministic and idempotent: every entry is matched to an existing entry
    by a natural key, an entry that cannot be applied is refused rather than
    guessed at, and nothing outside the patch is touched.  The candidate is
    never mutated, so the first-round report stays available as the fallback.
    """

    merged = copy.deepcopy(dict(candidate))
    rejected: list[str] = []
    _apply_evidence_updates(merged, patch.get("evidence_updates"), supplied, rejected)
    _apply_claim_updates(merged, patch.get("claim_updates"), rejected)
    _apply_method_updates(merged, patch.get("method_updates"), rejected)
    _apply_experiment_updates(merged, patch.get("experiment_updates"), rejected)
    _apply_resolved_requests(merged, patch.get("resolved_visual_requests"), executed, rejected)
    _append_unresolved_to_report(merged, _text_list(patch.get("unresolved_items_add")) or [])
    return merged, rejected


def _evidence_field_problems(
    fields: Mapping[str, Any], *, new: bool, supplied: Container[str]
) -> list[str]:
    problems: list[str] = []
    for name, value in fields.items():
        if name == "source_type":
            if value not in _EVIDENCE_SOURCE_TYPES:
                problems.append("source_type 必须是 " + " 或 ".join(_EVIDENCE_SOURCE_TYPES))
        elif name == "pdf_page":
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                problems.append("pdf_page 必须是 >= 1 的整数")
        elif name == "source_id":
            if not isinstance(value, str) or not value.strip():
                problems.append("source_id 必须是非空字符串")
        elif not isinstance(value, str):
            problems.append(f"{name} 必须是字符串")
    missing = [name for name in _EVIDENCE_FIELDS if name not in fields] if new else []
    if missing:
        problems.append("新增证据缺少字段：" + "、".join(missing))
    if not problems and new and supplied and fields.get("source_type") == "image":
        # A visual claim has to name an image that was actually sent in this
        # round; otherwise the "evidence" points at nothing a reader can open.
        source_id = str(fields.get("source_id"))
        if source_id not in supplied:
            problems.append(f"source_id {source_id} 不在本轮补看的图片里")
    return problems


def _apply_evidence_updates(
    report: MutableMapping[str, Any],
    values: Any,
    supplied: Container[str],
    rejected: list[str],
) -> None:
    entries = _section_entries(report, "evidence")
    by_id = {
        str(entry.get("id")): entry
        for entry in entries
        if isinstance(entry, MutableMapping) and entry.get("id")
    }
    for position, value in enumerate(_patch_entries(values)):
        if not isinstance(value, Mapping):
            _reject(rejected, "evidence_updates", position, "条目不是 JSON 对象")
            continue
        evidence_id = _text_value(value.get("id"))
        if not evidence_id:
            _reject(rejected, "evidence_updates", position, "缺少 id")
            continue
        fields = {name: value[name] for name in _EVIDENCE_FIELDS if name in value}
        if not fields:
            _reject(rejected, "evidence_updates", position, f"{evidence_id} 没有给出任何字段")
            continue
        existing = by_id.get(evidence_id)
        problems = _evidence_field_problems(fields, new=existing is None, supplied=supplied)
        if problems:
            _reject(
                rejected, "evidence_updates", position, f"{evidence_id}：" + "；".join(problems)
            )
            continue
        if existing is None:
            entry = {"id": evidence_id, **fields}
            entries.append(entry)
            by_id[evidence_id] = entry
        else:
            existing.update(fields)


def _apply_claim_updates(
    report: MutableMapping[str, Any], values: Any, rejected: list[str]
) -> None:
    entries = _section_entries(report, "claims")
    by_id = {
        str(entry.get("id")): entry
        for entry in entries
        if isinstance(entry, MutableMapping) and entry.get("id")
    }
    for position, value in enumerate(_patch_entries(values)):
        if not isinstance(value, Mapping):
            _reject(rejected, "claim_updates", position, "条目不是 JSON 对象")
            continue
        claim_id = _text_value(value.get("id"))
        if not claim_id:
            _reject(rejected, "claim_updates", position, "缺少 id")
            continue
        fields = {name: value[name] for name in ("text", "kind") if name in value}
        if "kind" in fields and fields["kind"] not in _CLAIM_KINDS:
            _reject(rejected, "claim_updates", position, f"kind 必须是 {'/'.join(_CLAIM_KINDS)}")
            continue
        if "text" in fields and not isinstance(fields["text"], str):
            _reject(rejected, "claim_updates", position, "text 必须是字符串")
            continue
        if "evidence_ids" in value:
            evidence = _text_list(value["evidence_ids"])
            if evidence is None:
                _reject(rejected, "claim_updates", position, "evidence_ids 必须是字符串数组")
                continue
            fields["evidence_ids"] = evidence
        if not fields:
            _reject(rejected, "claim_updates", position, f"{claim_id} 没有给出任何字段")
            continue
        existing = by_id.get(claim_id)
        missing = [name for name in ("text", "kind", "evidence_ids") if name not in fields]
        if existing is None and missing:
            _reject(
                rejected,
                "claim_updates",
                position,
                f"新增 claim {claim_id} 缺少字段：" + "、".join(missing),
            )
            continue
        if existing is None:
            entry = {"id": claim_id, **fields}
            entries.append(entry)
            by_id[claim_id] = entry
        else:
            existing.update(fields)


def _apply_method_updates(
    report: MutableMapping[str, Any], values: Any, rejected: list[str]
) -> None:
    method = report.get("method")
    if not isinstance(method, MutableMapping):
        method = {}
        report["method"] = method
    for position, value in enumerate(_patch_entries(values)):
        if not isinstance(value, Mapping):
            _reject(rejected, "method_updates", position, "条目不是 JSON 对象")
            continue
        target = _text_value(value.get("target"))
        if target == "method":
            _apply_method_target(method, value, position, rejected)
        elif target == "node":
            _apply_node_target(method, value, position, rejected)
        elif target == "edge":
            _apply_edge_target(method, value, position, rejected)
        elif target == "step":
            _apply_step_target(method, value, position, rejected)
        else:
            _reject(
                rejected, "method_updates", position, "target 必须是 method / node / edge / step"
            )


def _append_method_entries(
    method: MutableMapping[str, Any],
    name: str,
    values: Any,
    position: int,
    rejected: list[str],
) -> bool:
    required, list_fields = _METHOD_APPEND_SPECS[name]
    items = _structured_list(values, required, list_fields)
    if items is None:
        _reject(rejected, "method_updates", position, f"{name} 的条目缺少必填字段")
        return False
    entries = _section_entries(method, name)
    seen = {
        json.dumps(entry, ensure_ascii=False, sort_keys=True, default=str)
        for entry in entries
        if isinstance(entry, Mapping)
    }
    for item in items:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        entries.append(item)
    return True


def _apply_method_target(
    method: MutableMapping[str, Any],
    value: Mapping[str, Any],
    position: int,
    rejected: list[str],
) -> None:
    fields, problem = _collect_fields(value, _METHOD_PATCH_SPEC)
    if problem:
        _reject(rejected, "method_updates", position, problem)
        return
    appends: list[str] = []
    for name in _METHOD_APPEND_SPECS:
        if name not in value:
            continue
        if not _append_method_entries(method, name, value[name], position, rejected):
            return
        appends.append(name)
    if not fields and not appends:
        _reject(rejected, "method_updates", position, "target=method 没有给出任何可应用字段")
        return
    method.update(fields)


def _apply_node_target(
    method: MutableMapping[str, Any],
    value: Mapping[str, Any],
    position: int,
    rejected: list[str],
) -> None:
    node_id = _text_value(value.get("id"))
    if not node_id:
        _reject(rejected, "method_updates", position, "target=node 缺少 id")
        return
    nodes = _section_entries(method, "nodes")
    existing = next(
        (
            node
            for node in nodes
            if isinstance(node, MutableMapping) and str(node.get("id")) == node_id
        ),
        None,
    )
    fields, problem = _collect_fields(value, _NODE_PATCH_SPEC)
    if problem:
        _reject(rejected, "method_updates", position, problem)
        return
    if not fields:
        _reject(rejected, "method_updates", position, f"node {node_id} 没有给出任何字段")
        return
    if existing is None:
        missing = [name for name in _NODE_REQUIRED if name not in fields]
        if missing:
            _reject(
                rejected,
                "method_updates",
                position,
                f"新增 node {node_id} 缺少字段：" + "、".join(missing),
            )
            return
        nodes.append({"id": node_id, **fields})
        return
    existing.update(fields)


def _apply_edge_target(
    method: MutableMapping[str, Any],
    value: Mapping[str, Any],
    position: int,
    rejected: list[str],
) -> None:
    source = _text_value(value.get("from"))
    target = _text_value(value.get("to"))
    if not source or not target:
        _reject(rejected, "method_updates", position, "target=edge 需要 from 和 to")
        return
    edges = _section_entries(method, "edges")
    existing = next(
        (
            edge
            for edge in edges
            if isinstance(edge, MutableMapping)
            and str(edge.get("from")) == source
            and str(edge.get("to")) == target
        ),
        None,
    )
    fields, problem = _collect_fields(value, _EDGE_PATCH_SPEC)
    if problem:
        _reject(rejected, "method_updates", position, problem)
        return
    if not fields:
        _reject(rejected, "method_updates", position, f"edge {source} → {target} 没有给出任何字段")
        return
    if existing is None:
        missing = [name for name in _EDGE_REQUIRED if name not in fields]
        if missing:
            _reject(
                rejected,
                "method_updates",
                position,
                f"新增 edge {source} → {target} 缺少字段：" + "、".join(missing),
            )
            return
        edges.append({"from": source, "to": target, **fields})
        return
    existing.update(fields)


def _apply_step_target(
    method: MutableMapping[str, Any],
    value: Mapping[str, Any],
    position: int,
    rejected: list[str],
) -> None:
    step_name = _text_value(value.get("step"))
    if not step_name:
        _reject(rejected, "method_updates", position, "target=step 缺少 step")
        return
    steps = _section_entries(method, "steps")
    existing = next(
        (
            step
            for step in steps
            if isinstance(step, MutableMapping) and str(step.get("step")) == step_name
        ),
        None,
    )
    fields, problem = _collect_fields(value, _STEP_PATCH_SPEC)
    if problem:
        _reject(rejected, "method_updates", position, problem)
        return
    if not fields:
        _reject(rejected, "method_updates", position, f"step {step_name} 没有给出任何字段")
        return
    if existing is None:
        missing = [name for name in _STEP_REQUIRED if name not in fields]
        if missing:
            _reject(
                rejected,
                "method_updates",
                position,
                f"新增 step {step_name} 缺少字段：" + "、".join(missing),
            )
            return
        steps.append({"step": step_name, **fields})
        return
    existing.update(fields)


def _apply_experiment_updates(
    report: MutableMapping[str, Any], values: Any, rejected: list[str]
) -> None:
    experiments = _section_entries(report, "experiments")
    for position, value in enumerate(_patch_entries(values)):
        if not isinstance(value, Mapping):
            _reject(rejected, "experiment_updates", position, "条目不是 JSON 对象")
            continue
        index = value.get("index")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(experiments)
        ):
            _reject(
                rejected,
                "experiment_updates",
                position,
                f"index {index!r} 不在候选报告的 {len(experiments)} 个实验范围内",
            )
            continue
        experiment = experiments[index]
        if not isinstance(experiment, MutableMapping):
            _reject(rejected, "experiment_updates", position, f"第 {index} 个实验不是 JSON 对象")
            continue
        fields, problem = _collect_fields(value, _EXPERIMENT_PATCH_SPEC)
        if problem:
            _reject(rejected, "experiment_updates", position, problem)
            continue
        if not fields:
            _reject(rejected, "experiment_updates", position, f"实验 {index} 没有给出任何字段")
            continue
        experiment.update(fields)


def _apply_resolved_requests(
    report: MutableMapping[str, Any],
    values: Any,
    executed: Container[Any],
    rejected: list[str],
) -> None:
    requests = _section_entries(report, "visual_requests")
    for position, value in enumerate(_patch_entries(values)):
        if not isinstance(value, Mapping):
            _reject(rejected, "resolved_visual_requests", position, "条目不是 JSON 对象")
            continue
        page = value.get("pdf_page", "?")
        key = _request_key(value)
        if key not in executed:
            # A request whose crop was never rendered cannot have been
            # answered by this round, so the model is only allowed to close
            # the ones it was shown.
            _reject(
                rejected,
                "resolved_visual_requests",
                position,
                f"PDF 第 {page} 页的请求本轮没有执行，不能标记为已解决",
            )
            continue
        remaining = [
            item
            for item in requests
            if not (isinstance(item, Mapping) and _request_key(item) == key)
        ]
        if len(remaining) == len(requests):
            _reject(
                rejected,
                "resolved_visual_requests",
                position,
                f"候选报告里没有 PDF 第 {page} 页的这条请求",
            )
            continue
        requests[:] = remaining


def _normalize_report(
    report: MutableMapping[str, Any], executed: Container[Any] = ()
) -> None:
    """Reconcile a freshly parsed report with what the rest of the pipeline needs."""

    _sanitize_visual_requests(report)
    reconcile_cross_references(report)
    _record_unconfirmed_edges(report)
    _record_pending_requests(report, executed)


def reconcile_cross_references(report: MutableMapping[str, Any]) -> None:
    """Remove model-produced references that point at no report object.

    A dangling ID carries no evidence and cannot be repaired locally by
    guessing what the model intended.  Preserve the surrounding content when
    it still has grounded references, omit ungrounded method graph elements,
    and record every omission for human review.  This operation is
    deterministic and idempotent, so batch resume may also apply it to a
    report written by an older run.
    """

    known_evidence = {
        str(entry.get("id"))
        for entry in _mappings(report.get("evidence"))
        if isinstance(entry.get("id"), str) and str(entry.get("id")).strip()
    }
    evidence_problems: list[str] = []

    def prune(value: Any, path: str) -> None:
        if isinstance(value, MutableMapping):
            for key, item in list(value.items()):
                child_path = f"{path}.{key}"
                if key == "evidence_ids" and isinstance(item, list):
                    kept: list[Any] = []
                    for evidence_id in item:
                        if isinstance(evidence_id, str) and evidence_id in known_evidence:
                            kept.append(evidence_id)
                        else:
                            evidence_problems.append(
                                f"{DANGLING_EVIDENCE_PREFIX}{child_path} 引用了不存在的 "
                                f"{evidence_id!r}"
                            )
                    value[key] = kept
                elif isinstance(item, (MutableMapping, list)):
                    prune(item, child_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, (MutableMapping, list)):
                    prune(item, f"{path}[{index}]")

    prune(report, "$")
    _append_unresolved_to_report(report, evidence_problems)

    method = report.get("method")
    if not isinstance(method, MutableMapping):
        return

    method_problems: list[str] = []
    nodes = method.get("nodes")
    if isinstance(nodes, list):
        grounded_nodes: list[Any] = []
        for index, node in enumerate(nodes):
            if isinstance(node, Mapping) and not node.get("evidence_ids"):
                method_problems.append(
                    f"{DANGLING_METHOD_PREFIX}$.method.nodes[{index}] "
                    f"{node.get('id')!r} 没有可解析的证据"
                )
                continue
            grounded_nodes.append(node)
        method["nodes"] = grounded_nodes
    node_ids = {
        str(node.get("id"))
        for node in _mappings(method.get("nodes"))
        if isinstance(node.get("id"), str)
    }

    edges = method.get("edges")
    if isinstance(edges, list):
        grounded_edges: list[Any] = []
        for index, edge in enumerate(edges):
            if not isinstance(edge, Mapping):
                grounded_edges.append(edge)
                continue
            source = edge.get("from")
            target = edge.get("to")
            reason = ""
            if source not in node_ids or target not in node_ids:
                reason = f"端点 {source!r} → {target!r} 未定义"
            elif not edge.get("evidence_ids"):
                reason = "没有可解析的证据"
            if reason:
                method_problems.append(
                    f"{DANGLING_METHOD_PREFIX}$.method.edges[{index}] {reason}"
                )
                continue
            grounded_edges.append(edge)
        method["edges"] = grounded_edges

    steps = method.get("steps")
    if isinstance(steps, list):
        grounded_steps: list[Any] = []
        for index, step in enumerate(steps):
            if isinstance(step, Mapping) and not step.get("evidence_ids"):
                method_problems.append(
                    f"{DANGLING_METHOD_PREFIX}$.method.steps[{index}] 没有可解析的证据"
                )
                continue
            grounded_steps.append(step)
        method["steps"] = grounded_steps
    _append_unresolved_to_report(report, method_problems)


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
    """Render an exception with its cause chain, minus the API key.

    The SDK's own message for a failed connection is the four-word
    "Connection error."; whether it was a refused proxy, a DNS failure or a
    reset TLS handshake lives in the chained exception.  Recording only the
    outer message leaves a failed run undiagnosable -- five papers once died
    in the same twelve seconds and the log said nothing but those four words.
    """

    key = os.environ.get("DASHSCOPE_API_KEY")
    chain: list[str] = []
    seen: set[int] = set()
    error: BaseException | None = exc
    while error is not None and id(error) not in seen and len(chain) < 4:
        seen.add(id(error))
        text = str(error).strip()
        if key:
            text = text.replace(key, "[REDACTED]")
        label = f"{type(error).__name__}: {text}" if text else type(error).__name__
        if not chain or chain[-1] != label:
            chain.append(label)
        error = error.__cause__ or error.__context__
    return " <- ".join(chain)[:2000]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "CallRecord",
    "DEFAULT_DIAGRAM_INSTRUCTIONS",
    "DEFAULT_MODEL",
    "DEFAULT_READER_INSTRUCTIONS",
    "DEFAULT_REFINE_INSTRUCTIONS",
    "DIAGRAM_FAILED_PREFIX",
    "DIAGRAM_OMITTED_PREFIX",
    "DiagramResult",
    "EventRecorder",
    "INPUT_MODES",
    "JsonlEventRecorder",
    "PATCH_REJECTED_PREFIX",
    "PATCH_SECTIONS",
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
    "diagram_response_schema",
    "image_data_uri",
    "load_env_file",
    "normalize_report_schema_shape",
    "normalize_page",
    "normalize_pages",
    "parse_json_object",
    "refine_patch_schema",
    "reconcile_cross_references",
    "sanitize_diagrams",
    "schema_errors",
    "validate_visual_request",
]
