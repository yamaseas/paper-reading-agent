"""Schema and semantic checks for generated paper-reading reports.

JSON Schema catches missing fields and wrong primitive types.  It cannot tell
whether a page exists in the current PDF, whether a method edge points to a
real node, or whether an evidence reference is valid.  The semantic checks in
this module cover those cross-document invariants without pretending to judge
the factual correctness of an LLM's interpretation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover - handled with a useful validation error
    Draft202012Validator = None  # type: ignore[assignment,misc]


DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "report.schema.json"


@dataclass(frozen=True)
class ValidationIssue:
    """One validation finding.

    ``severity`` is either ``error`` or ``warning``.  Warnings document a
    quality concern but do not make a report structurally undeliverable.
    """

    path: str
    message: str
    code: str = "validation_error"
    severity: str = "error"

    def as_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "message": self.message,
            "code": self.code,
            "severity": self.severity,
        }

    def __str__(self) -> str:
        location = self.path or "$"
        return f"{location}: {self.message}"


@dataclass
class ValidationResult(Mapping[str, Any]):
    """Result returned by :func:`validate_report`."""

    issues: list[ValidationIssue]
    schema_checked: bool = True
    semantic_checked: bool = True

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def schema_valid(self) -> bool:
        return not any(issue.code.startswith("schema_") for issue in self.errors)

    @property
    def semantic_valid(self) -> bool:
        return not any(issue.code.startswith("semantic_") for issue in self.errors)

    @property
    def valid(self) -> bool:
        return not self.errors

    def __bool__(self) -> bool:
        return self.valid

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "schema_valid": self.schema_valid,
            "semantic_valid": self.semantic_valid,
            "errors": [issue.as_dict() for issue in self.errors],
            "warnings": [issue.as_dict() for issue in self.warnings],
        }

    # Expose the result as a read-only mapping as well as an object.  This is
    # convenient for batch callers that support validators returning either a
    # dictionary or a richer result object.
    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def __iter__(self):
        return iter(self.as_dict())

    def __len__(self) -> int:
        return len(self.as_dict())

    def raise_for_errors(self) -> None:
        if self.errors:
            details = "\n".join(str(issue) for issue in self.errors)
            raise ValueError(f"report validation failed:\n{details}")


def _json_path(parts: Iterable[Any]) -> str:
    """Format a jsonschema path as a compact, useful JSONPath-like string."""

    result = "$"
    for part in parts:
        if isinstance(part, int):
            result += f"[{part}]"
        else:
            text = str(part)
            if text.isidentifier():
                result += f".{text}"
            else:
                result += f"[{json.dumps(text, ensure_ascii=False)}]"
    return result


def _load_json(path: str | Path) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def _load_schema(schema_path: str | Path | None) -> Mapping[str, Any]:
    path = Path(schema_path) if schema_path is not None else DEFAULT_SCHEMA_PATH
    schema = _load_json(path)
    if not isinstance(schema, Mapping):
        raise ValueError(f"schema must be a JSON object: {path}")
    return schema


def _issue(
    issues: list[ValidationIssue],
    path: str,
    message: str,
    *,
    code: str,
    severity: str = "error",
) -> None:
    issues.append(ValidationIssue(path=path, message=message, code=code, severity=severity))


def _normalise_page_manifest(
    pages: Any,
    page_count: int | None = None,
    extra_image_ids: Iterable[str] | Mapping[str, Any] | None = None,
) -> tuple[set[int] | None, set[str] | None, dict[str, int]]:
    """Extract page and image identities from a pages manifest or list."""

    if isinstance(pages, (str, Path)):
        pages = _load_json(pages)
    entries: Any = pages
    manifest_total: int | None = None
    if isinstance(pages, Mapping):
        entries = pages.get("pages", [])
        raw_total = pages.get("total_pages")
        if isinstance(raw_total, int) and raw_total >= 0:
            manifest_total = raw_total
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        entries = []

    page_numbers: set[int] = set()
    image_ids: set[str] = set()
    image_to_page: dict[str, int] = {}
    for entry in entries:
        if isinstance(entry, int) and not isinstance(entry, bool):
            page_numbers.add(entry)
            continue
        if not isinstance(entry, Mapping):
            continue
        raw_page = entry.get("pdf_page")
        page: int | None = None
        if isinstance(raw_page, int) and not isinstance(raw_page, bool):
            page = raw_page
            page_numbers.add(page)
        for key in ("id", "image_id"):
            value = entry.get(key)
            if isinstance(value, str) and value:
                image_ids.add(value)
                if page is not None:
                    image_to_page[value] = page
        # Accepting paths here helps validate manifests produced by older
        # development versions, while evidence should still use image IDs.
        for key in ("path", "image_path"):
            value = entry.get(key)
            if isinstance(value, str) and value:
                image_ids.add(value)
                image_ids.add(Path(value).name)
                if page is not None:
                    image_to_page[value] = page
                    image_to_page[Path(value).name] = page

    # Cropped images are deliberately not added to ``pages.json`` because
    # that manifest describes the stable full-page inputs.  Callers can pass
    # the crop IDs (or a manifest's supplemental_images list) when validating
    # a refined report.
    if isinstance(pages, Mapping):
        supplemental = pages.get("supplemental_images", [])
        if isinstance(supplemental, Sequence) and not isinstance(supplemental, (str, bytes)):
            for value in supplemental:
                if isinstance(value, str) and value:
                    image_ids.add(value)
                elif isinstance(value, Mapping):
                    for key in ("id", "image_id", "path", "image_path"):
                        item = value.get(key)
                        if isinstance(item, str) and item:
                            image_ids.add(item)
                            image_ids.add(Path(item).name)
    if extra_image_ids is not None:
        # A mapping is ``image_id -> pdf_page``: crops and re-rendered pages
        # that were supplied during visual refinement belong to a real page but
        # never appear in the stable page manifest.  Accepting a plain iterable
        # keeps the older call style working.
        items = (
            extra_image_ids.items()
            if isinstance(extra_image_ids, Mapping)
            else ((value, None) for value in extra_image_ids)
        )
        for value, page in items:
            if not isinstance(value, str) or not value:
                continue
            image_ids.add(value)
            image_ids.add(Path(value).name)
            if isinstance(page, int) and not isinstance(page, bool):
                image_to_page.setdefault(value, page)
                image_to_page.setdefault(Path(value).name, page)

    if not page_numbers:
        total = page_count if page_count is not None else manifest_total
        if total is not None and total >= 0:
            page_numbers = set(range(1, total + 1))
    elif page_count is not None:
        # A caller-provided count is authoritative for bounds, but preserve
        # the explicit manifest set so missing pages remain detectable.
        page_numbers.update(range(1, page_count + 1))
    return (
        page_numbers if page_numbers else None,
        image_ids if image_ids else None,
        image_to_page,
    )


def _walk_evidence_references(value: Any, path: str = "$") -> Iterable[tuple[str, Any]]:
    """Yield every field named ``evidence_ids`` in a report."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}" if str(key).isidentifier() else f"{path}[{key!r}]"
            if key == "evidence_ids":
                yield child_path, child
            else:
                yield from _walk_evidence_references(child, child_path)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            yield from _walk_evidence_references(child, f"{path}[{index}]")


def _check_semantics(
    report: Mapping[str, Any],
    issues: list[ValidationIssue],
    *,
    pages: Any = None,
    page_count: int | None = None,
    image_ids: Iterable[str] | Mapping[str, Any] | None = None,
) -> None:
    available_pages, available_images, image_to_page = _normalise_page_manifest(
        pages, page_count, image_ids
    )

    evidence = report.get("evidence")
    evidence_by_id: dict[str, Mapping[str, Any]] = {}
    if isinstance(evidence, list):
        for index, item in enumerate(evidence):
            path = f"$.evidence[{index}]"
            if not isinstance(item, Mapping):
                continue
            evidence_id = item.get("id")
            if not isinstance(evidence_id, str) or not evidence_id.strip():
                continue
            if evidence_id in evidence_by_id:
                _issue(
                    issues,
                    f"{path}.id",
                    f"duplicate evidence id {evidence_id!r}",
                    code="semantic_duplicate_evidence_id",
                )
            else:
                evidence_by_id[evidence_id] = item
            pdf_page = item.get("pdf_page")
            if isinstance(pdf_page, int) and not isinstance(pdf_page, bool) and available_pages is not None:
                if pdf_page not in available_pages:
                    _issue(
                        issues,
                        f"{path}.pdf_page",
                        f"PDF page {pdf_page} is not present in the input page manifest",
                        code="semantic_page_out_of_range",
                    )
            image_id = item.get("image_id")
            if isinstance(image_id, str) and image_id:
                if available_images is not None and image_id not in available_images:
                    _issue(
                        issues,
                        f"{path}.image_id",
                        f"image id {image_id!r} is not present in the input page manifest",
                        code="semantic_unknown_image_id",
                    )
                mapped_page = image_to_page.get(image_id)
                if (
                    mapped_page is not None
                    and isinstance(pdf_page, int)
                    and mapped_page != pdf_page
                ):
                    _issue(
                        issues,
                        f"{path}.image_id",
                        f"image id {image_id!r} belongs to PDF page {mapped_page}, not {pdf_page}",
                        code="semantic_image_page_mismatch",
                    )

    # Every reference, including those on method nodes and experimental
    # results, must resolve to an evidence entry from this report.
    for path, references in _walk_evidence_references(report):
        if not isinstance(references, list):
            continue  # The schema validator reports the type error.
        for index, evidence_id in enumerate(references):
            if not isinstance(evidence_id, str) or evidence_id not in evidence_by_id:
                _issue(
                    issues,
                    f"{path}[{index}]",
                    f"unknown evidence id {evidence_id!r}",
                    code="semantic_unknown_evidence_id",
                )

    claims = report.get("claims")
    claim_ids: set[str] = set()
    if isinstance(claims, list):
        for index, claim in enumerate(claims):
            if not isinstance(claim, Mapping):
                continue
            claim_id = claim.get("id")
            if not isinstance(claim_id, str) or not claim_id.strip():
                continue
            if claim_id in claim_ids:
                _issue(
                    issues,
                    f"$.claims[{index}].id",
                    f"duplicate claim id {claim_id!r}",
                    code="semantic_duplicate_claim_id",
                )
            claim_ids.add(claim_id)

    method = report.get("method")
    if isinstance(method, Mapping):
        nodes = method.get("nodes")
        node_ids: set[str] = set()
        if isinstance(nodes, list):
            for index, node in enumerate(nodes):
                if not isinstance(node, Mapping):
                    continue
                node_id = node.get("id")
                if not isinstance(node_id, str) or not node_id.strip():
                    continue
                if node_id in node_ids:
                    _issue(
                        issues,
                        f"$.method.nodes[{index}].id",
                        f"duplicate method node id {node_id!r}",
                        code="semantic_duplicate_node_id",
                    )
                node_ids.add(node_id)
                if not isinstance(node.get("evidence_ids"), list) or not node.get("evidence_ids"):
                    _issue(
                        issues,
                        f"$.method.nodes[{index}].evidence_ids",
                        "every method node must carry at least one evidence id",
                        code="semantic_node_without_evidence",
                    )
        edges = method.get("edges")
        if isinstance(edges, list):
            for index, edge in enumerate(edges):
                if not isinstance(edge, Mapping):
                    continue
                source = edge.get("from")
                target = edge.get("to")
                if isinstance(source, str) and source not in node_ids:
                    _issue(
                        issues,
                        f"$.method.edges[{index}].from",
                        f"unknown method node id {source!r}",
                        code="semantic_unknown_node_id",
                    )
                if isinstance(target, str) and target not in node_ids:
                    _issue(
                        issues,
                        f"$.method.edges[{index}].to",
                        f"unknown method node id {target!r}",
                        code="semantic_unknown_node_id",
                    )
                relation = edge.get("relation")
                if not isinstance(relation, str) or not relation.strip():
                    _issue(
                        issues,
                        f"$.method.edges[{index}].relation",
                        "method edges must state data flow, control flow, or dependency",
                        code="semantic_empty_edge_relation",
                    )
                if not isinstance(edge.get("evidence_ids"), list) or not edge.get("evidence_ids"):
                    _issue(
                        issues,
                        f"$.method.edges[{index}].evidence_ids",
                        "every method edge must carry at least one evidence id",
                        code="semantic_edge_without_evidence",
                    )
        steps = method.get("steps")
        if isinstance(steps, list):
            for index, step in enumerate(steps):
                if isinstance(step, Mapping) and (
                    not isinstance(step.get("evidence_ids"), list) or not step.get("evidence_ids")
                ):
                    _issue(
                        issues,
                        f"$.method.steps[{index}].evidence_ids",
                        "every method step must carry at least one evidence id",
                        code="semantic_step_without_evidence",
                    )

    visual_requests = report.get("visual_requests")
    if isinstance(visual_requests, list):
        for index, request in enumerate(visual_requests):
            if not isinstance(request, Mapping):
                continue
            page = request.get("pdf_page")
            if isinstance(page, int) and not isinstance(page, bool) and available_pages is not None:
                if page not in available_pages:
                    # A page that is not in the input is not executed: the
                    # request is recorded and the paper still delivers the
                    # report it did manage to produce.
                    _issue(
                        issues,
                        f"$.visual_requests[{index}].pdf_page",
                        f"PDF page {page} is not present in the input page manifest",
                        code="semantic_visual_page_out_of_range",
                        severity="warning",
                    )
            crop = request.get("crop")
            if crop is None:
                continue
            if not isinstance(crop, Mapping):
                continue
            try:
                values = tuple(float(crop[name]) for name in ("x0", "y0", "x1", "y1"))
            except (KeyError, TypeError, ValueError):
                continue  # JSON Schema reports missing/non-number fields.
            if not all(math.isfinite(value) for value in values):
                _issue(
                    issues,
                    f"$.visual_requests[{index}].crop",
                    "crop coordinates must be finite",
                    code="semantic_invalid_crop",
                    severity="warning",
                )
            elif not (0 <= values[0] < values[2] <= 1 and 0 <= values[1] < values[3] <= 1):
                _issue(
                    issues,
                    f"$.visual_requests[{index}].crop",
                    "crop must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1",
                    code="semantic_invalid_crop",
                    severity="warning",
                )


def validate_report(
    report: Mapping[str, Any] | Any,
    *,
    schema_path: str | Path | None = None,
    pages: Any = None,
    page_count: int | None = None,
    image_ids: Iterable[str] | Mapping[str, Any] | None = None,
    check_schema: bool = True,
    check_semantics: bool = True,
) -> ValidationResult:
    """Validate a report object against schema and input-document invariants.

    ``pages`` may be a pages manifest path, the manifest object itself, or its
    list of page entries.  Pass it whenever page/image references need to be
    checked; without it, report-local evidence and graph invariants are still
    checked.

    ``image_ids`` lists images supplied outside the stable manifest, such as
    the crops and re-rendered pages from one visual-refinement round.  Pass a
    mapping of ``image_id -> pdf_page`` so their evidence entries are checked
    against the right page instead of being rejected as unknown.
    """

    issues: list[ValidationIssue] = []
    if check_schema:
        if Draft202012Validator is None:
            _issue(
                issues,
                "$",
                "jsonschema is required for report schema validation",
                code="schema_dependency_missing",
            )
        else:
            try:
                schema = _load_schema(schema_path)
                validator = Draft202012Validator(schema)
                for error in sorted(validator.iter_errors(report), key=lambda item: list(item.path)):
                    _issue(
                        issues,
                        _json_path(error.path),
                        error.message,
                        code="schema_validation_error",
                    )
            except (ValueError, OSError) as exc:
                _issue(issues, "$", str(exc), code="schema_load_error")
    if check_semantics and isinstance(report, Mapping):
        try:
            _check_semantics(
                report, issues, pages=pages, page_count=page_count, image_ids=image_ids
            )
        except (ValueError, TypeError) as exc:
            _issue(issues, "$", f"could not complete semantic checks: {exc}", code="semantic_check_error")
    elif check_semantics:
        _issue(issues, "$", "report must be a JSON object", code="semantic_not_object")
    return ValidationResult(
        issues=issues,
        schema_checked=check_schema,
        semantic_checked=check_semantics,
    )


def _discover_pages_manifest(report_path: Path) -> Path | None:
    """Find the workspace pages manifest for ``runs/<id>/report.json``."""

    candidates = [
        report_path.parent / "pages.json",
        report_path.parent.parent / "pages.json",
        report_path.parent.parent.parent / "pages.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def validate_report_file(
    report_path: str | Path,
    *,
    schema_path: str | Path | None = None,
    pages: Any = None,
    page_count: int | None = None,
    image_ids: Iterable[str] | Mapping[str, Any] | None = None,
    check_schema: bool = True,
    check_semantics: bool = True,
) -> ValidationResult:
    """Load and validate ``report.json``, discovering nearby ``pages.json``."""

    path = Path(report_path)
    try:
        report = _load_json(path)
    except ValueError as exc:
        return ValidationResult(
            issues=[ValidationIssue("$", str(exc), code="report_load_error")],
            schema_checked=check_schema,
            semantic_checked=check_semantics,
        )
    if pages is None:
        pages = _discover_pages_manifest(path)
    try:
        return validate_report(
            report,
            schema_path=schema_path,
            pages=pages,
            page_count=page_count,
            image_ids=image_ids,
            check_schema=check_schema,
            check_semantics=check_semantics,
        )
    except ValueError as exc:
        return ValidationResult(
            issues=[ValidationIssue("$", str(exc), code="validation_load_error")],
            schema_checked=check_schema,
            semantic_checked=check_semantics,
        )


# Explicit aliases make the intended use clear to callers that only want one
# layer of checks, while keeping one implementation of the invariants.
validate_structure = validate_report
validate_report_json = validate_report_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a paper-reading report JSON")
    parser.add_argument("report", type=Path)
    parser.add_argument("--schema", type=Path, default=None)
    parser.add_argument("--pages", type=Path, default=None)
    parser.add_argument("--page-count", type=int, default=None)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--no-schema", action="store_true")
    parser.add_argument("--no-semantics", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = validate_report_file(
        args.report,
        schema_path=args.schema,
        pages=args.pages,
        page_count=args.page_count,
        check_schema=not args.no_schema,
        check_semantics=not args.no_semantics,
    )
    if args.as_json:
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    else:
        if result.valid:
            print(f"valid: {args.report}")
        for issue in result.issues:
            print(f"{issue.severity}: {issue}")
    return 0 if result.valid else 1


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    sys.exit(main())


__all__ = [
    "DEFAULT_SCHEMA_PATH",
    "ValidationIssue",
    "ValidationResult",
    "validate_report",
    "validate_report_file",
    "validate_report_json",
    "validate_structure",
]
