"""PDF preprocessing for the paper-reading pipeline.

The reader uses page images as its primary input.  This module keeps all PDF
handling in one place and deliberately imports PyMuPDF lazily: commands that
only validate or render an existing manifest remain usable when the optional
PDF dependency is not installed.

The public entry point is :func:`preprocess_pdf`.  It creates the stable
``original.pdf``, ``pages/``, ``metadata.json`` and ``pages.json`` artifacts
described in the project plan.  ``render_crop`` and
``render_visual_requests`` are used by the one-round visual refinement pass.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class PreprocessError(RuntimeError):
    """Raised when a PDF cannot be prepared for reading."""


def _require_fitz() -> Any:
    """Import PyMuPDF only when a PDF operation is actually requested."""

    try:
        import pymupdf as fitz  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        try:
            # Legacy alias retained for older PyMuPDF releases.
            import fitz  # type: ignore
        except ImportError:
            raise PreprocessError(
                "PyMuPDF is required for PDF preprocessing. Install it with "
                "`pip install PyMuPDF`."
            ) from exc
    return fitz


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of *path* without loading it into memory."""

    source = Path(path)
    if not source.is_file():
        raise PreprocessError(f"PDF file does not exist: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def make_paper_id(pdf_path: str | os.PathLike[str], paper_id: str | None = None) -> str:
    """Return a filesystem-friendly paper identifier.

    An explicit ID is preserved after validation.  For an omitted ID the PDF
    stem is normalised, which makes the identifier stable across runs while
    avoiding path separators and control characters.
    """

    candidate = paper_id if paper_id is not None else Path(pdf_path).stem
    candidate = str(candidate).strip()
    if not candidate:
        raise PreprocessError("paper_id must not be empty")
    normalised = re.sub(r"[^A-Za-z0-9._-]+", "-", candidate).strip(".-_")
    if not normalised:
        raise PreprocessError(f"paper_id has no usable filename characters: {candidate!r}")
    return normalised


def _normalise_image_format(image_format: str) -> str:
    value = str(image_format).lower().lstrip(".")
    if value not in {"png", "jpg", "jpeg"}:
        raise PreprocessError(f"unsupported image format {image_format!r}; use png or jpeg")
    return "jpg" if value == "jpeg" else value


def _atomic_write_json(path: Path, value: Any) -> None:
    """Write JSON atomically so a half-written manifest cannot look complete."""

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
            json.dump(value, handle, ensure_ascii=False, indent=2)
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


def _page_rect(page: Any) -> Any:
    """Get the visible page rectangle from a PyMuPDF page."""

    try:
        return page.rect
    except Exception as exc:  # pragma: no cover - defensive for fake pages
        raise PreprocessError("PyMuPDF returned a page without a rectangle") from exc


def _printed_page_labels(document: Any) -> list[str | None]:
    """Return the printed page label for every page, when the PDF declares one.

    The label comes from the PDF page-label structure, not from guessing at
    text on the page.  PDFs without labels return ``None`` for every page,
    which keeps the field honest instead of inventing a number.
    """

    getter = getattr(document, "get_page_labels", None)
    if not callable(getter):
        return []
    try:
        labels = getter()
    except Exception:  # pragma: no cover - malformed label trees are not fatal
        return []
    if not isinstance(labels, list):
        return []
    total = len(document)
    result: list[str | None] = [None] * total
    try:
        for entry in labels:
            if not isinstance(entry, Mapping):
                continue
            start = int(entry.get("startpage", 0) or 0)
            first = int(entry.get("firstpagenum", 1) or 1)
            prefix = str(entry.get("prefix", "") or "")
            style = str(entry.get("style", "D") or "D")
            end = total
            for other in labels:
                if isinstance(other, Mapping) and int(other.get("startpage", -1) or -1) > start:
                    end = min(end, int(other.get("startpage", total) or total))
            for index in range(max(0, start), min(end, total)):
                number = first + (index - start)
                if style.upper() in {"D", ""}:
                    result[index] = f"{prefix}{number}"
                elif style.upper() == "R" or style.upper() == "r":
                    result[index] = f"{prefix}{_roman(number, style.islower())}"
                else:
                    result[index] = None
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return []
    return result


def _roman(value: int, lower: bool = False) -> str:
    if not 0 < value < 4000:
        return str(value)
    numerals = (
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
        (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    )
    parts: list[str] = []
    remaining = value
    for amount, numeral in numerals:
        while remaining >= amount:
            parts.append(numeral)
            remaining -= amount
    text = "".join(parts)
    return text.lower() if lower else text


def _normalise_crop(crop: Mapping[str, Any] | Sequence[Any] | None) -> tuple[float, float, float, float] | None:
    """Validate and normalise a visible-page crop in [0, 1] coordinates."""

    if crop is None:
        return None
    if isinstance(crop, Mapping):
        names = ("x0", "y0", "x1", "y1")
        if any(name not in crop for name in names):
            raise PreprocessError("crop must contain x0, y0, x1 and y1")
        values = tuple(crop[name] for name in names)
    elif isinstance(crop, Sequence) and not isinstance(crop, (str, bytes)) and len(crop) == 4:
        values = tuple(crop)
    else:
        raise PreprocessError("crop must be null, a mapping, or a four-item sequence")

    try:
        coordinates = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise PreprocessError("crop coordinates must be numbers") from exc
    if not all(math.isfinite(value) for value in coordinates):
        raise PreprocessError("crop coordinates must be finite")
    x0, y0, x1, y1 = coordinates
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise PreprocessError("crop must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    return coordinates


def _render_pixmap(
    page: Any,
    fitz: Any,
    *,
    dpi: float,
    crop: tuple[float, float, float, float] | None,
    max_pixels: int | None,
) -> Any:
    """Render a page or normalised page crop to a PyMuPDF pixmap.

    The normalised rectangle is calculated against ``page.rect`` rather than
    the PDF media box.  This follows the visible, rotated page coordinates
    exposed by PyMuPDF and keeps crop requests independent of PDF units.
    """

    if not math.isfinite(float(dpi)) or float(dpi) <= 0:
        raise PreprocessError("dpi must be a positive finite number")
    rect = _page_rect(page)
    if crop is None:
        clip = rect
    else:
        x0, y0, x1, y1 = crop
        clip = fitz.Rect(
            rect.x0 + rect.width * x0,
            rect.y0 + rect.height * y0,
            rect.x0 + rect.width * x1,
            rect.y0 + rect.height * y1,
        )

    requested_dpi = float(dpi)
    if max_pixels is not None:
        try:
            pixel_budget = int(max_pixels)
        except (TypeError, ValueError) as exc:
            raise PreprocessError("max_pixels must be an integer") from exc
        if pixel_budget <= 0:
            raise PreprocessError("max_pixels must be positive")
        estimated = max(1.0, float(clip.width) * requested_dpi / 72.0) * max(
            1.0, float(clip.height) * requested_dpi / 72.0
        )
        if estimated > pixel_budget:
            requested_dpi *= math.sqrt(pixel_budget / estimated)

    matrix = fitz.Matrix(requested_dpi / 72.0, requested_dpi / 72.0)
    try:
        return page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
    except TypeError:
        # A small compatibility fallback for older PyMuPDF releases and the
        # simple fake pages used by downstream tests.
        return page.get_pixmap(matrix=matrix, clip=clip)


def _save_pixmap(pixmap: Any, output_path: Path, image_format: str) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pixmap.save(str(output_path))
    except Exception as exc:
        raise PreprocessError(f"could not save rendered page to {output_path}") from exc
    return int(pixmap.width), int(pixmap.height)


def render_pdf_pages(
    pdf_path: str | os.PathLike[str],
    pages_dir: str | os.PathLike[str],
    *,
    dpi: float = 160,
    image_format: str = "png",
    max_pixels: int | None = None,
) -> list[dict[str, Any]]:
    """Render every PDF page and return the entries written to ``pages.json``.

    Paths in returned entries are relative to the parent of ``pages_dir``.
    This makes manifests portable when a workspace is copied elsewhere.
    """

    fitz = _require_fitz()
    source = Path(pdf_path)
    if not source.is_file():
        raise PreprocessError(f"PDF file does not exist: {source}")
    page_root = Path(pages_dir)
    page_root.mkdir(parents=True, exist_ok=True)
    fmt = _normalise_image_format(image_format)
    suffix = ".jpg" if fmt == "jpg" else ".png"
    try:
        document = fitz.open(str(source))
    except Exception as exc:
        raise PreprocessError(f"could not open PDF: {source}") from exc

    entries: list[dict[str, Any]] = []
    try:
        total_pages = len(document)
        printed_labels = _printed_page_labels(document)
        for index in range(total_pages):
            page = document.load_page(index)
            image_id = f"page-{index + 1:03d}"
            image_path = page_root / f"{image_id}{suffix}"
            crop = None
            pixmap = _render_pixmap(page, fitz, dpi=dpi, crop=crop, max_pixels=max_pixels)
            width, height = _save_pixmap(pixmap, image_path, fmt)
            rect = _page_rect(page)
            entries.append(
                {
                    "id": image_id,
                    "image_id": image_id,
                    "pdf_page": index + 1,
                    "printed_page": printed_labels[index] if index < len(printed_labels) else None,
                    "path": image_path.relative_to(page_root.parent).as_posix(),
                    "image_path": image_path.relative_to(page_root.parent).as_posix(),
                    "width": width,
                    "height": height,
                    "render_width": width,
                    "render_height": height,
                    "page_width": float(rect.width),
                    "page_height": float(rect.height),
                    "rotation": int(getattr(page, "rotation", 0) or 0),
                    # Record the DPI actually used: ``max_pixels`` can lower it,
                    # and a manifest that reports the requested value would
                    # misdescribe the image on disk.
                    "dpi": _effective_dpi(width, rect.width),
                    "requested_dpi": float(dpi),
                    "image_format": fmt,
                    "max_pixels": max_pixels,
                }
            )
    finally:
        close = getattr(document, "close", None)
        if close is not None:
            close()
    _prune_stale_pages(page_root, {entry["id"] for entry in entries}, suffix)
    return entries


def _effective_dpi(pixels: int, points: float) -> float:
    """Convert a rendered width back into the DPI it was produced at."""

    if not points or points <= 0:
        return 0.0
    return round(float(pixels) * 72.0 / float(points), 2)


def _prune_stale_pages(page_root: Path, keep: set[str], suffix: str) -> None:
    """Drop page images left over from an earlier, longer PDF.

    A manifest that says five pages must not sit next to ten images, or a later
    reader can link a page the current PDF no longer contains.
    """

    keep_names = {f"{image_id}{suffix}" for image_id in keep}
    for pattern in ("page-*.png", "page-*.jpg"):
        for stale in page_root.glob(pattern):
            if stale.name not in keep_names:
                try:
                    stale.unlink()
                except OSError:  # pragma: no cover - best effort cleanup
                    pass


def extract_text_with_pages(
    pdf_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
) -> str:
    """Extract coarse text with explicit PDF page markers.

    The result is intentionally labelled as coarse text in the manifest.  It
    helps page lookup and Q&A, but callers should still use page images as the
    evidence source because PDF reading order can be wrong for multi-column
    layouts.
    """

    fitz = _require_fitz()
    source = Path(pdf_path)
    if not source.is_file():
        raise PreprocessError(f"PDF file does not exist: {source}")
    try:
        document = fitz.open(str(source))
    except Exception as exc:
        raise PreprocessError(f"could not open PDF: {source}") from exc
    chunks: list[str] = []
    try:
        for index in range(len(document)):
            page = document.load_page(index)
            try:
                text = page.get_text("text", sort=True)
            except TypeError:
                text = page.get_text("text")
            chunks.append(f"--- PDF_PAGE={index + 1} ---\n{text.rstrip()}\n")
    finally:
        close = getattr(document, "close", None)
        if close is not None:
            close()
    result = "\n".join(chunks)
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(result, encoding="utf-8")
    return result


# A short alias is useful to callers and keeps the CLI-facing name concise.
extract_text = extract_text_with_pages


def render_crop(
    pdf_path: str | os.PathLike[str],
    pdf_page: int,
    crop: Mapping[str, Any] | Sequence[Any] | None,
    output_path: str | os.PathLike[str],
    *,
    dpi: float = 300,
    image_format: str = "png",
    max_pixels: int | None = None,
) -> dict[str, Any]:
    """Render one page or normalised crop for a visual refinement request."""

    fitz = _require_fitz()
    source = Path(pdf_path)
    if not source.is_file():
        raise PreprocessError(f"PDF file does not exist: {source}")
    try:
        page_number = int(pdf_page)
    except (TypeError, ValueError) as exc:
        raise PreprocessError("pdf_page must be a positive integer") from exc
    if page_number < 1:
        raise PreprocessError("pdf_page must be a positive integer")
    normalised = _normalise_crop(crop)
    fmt = _normalise_image_format(image_format)
    target = Path(output_path)
    try:
        document = fitz.open(str(source))
    except Exception as exc:
        raise PreprocessError(f"could not open PDF: {source}") from exc
    try:
        if page_number > len(document):
            raise PreprocessError(
                f"pdf_page {page_number} is outside the PDF (1-{len(document)})"
            )
        page = document.load_page(page_number - 1)
        pixmap = _render_pixmap(
            page, fitz, dpi=dpi, crop=normalised, max_pixels=max_pixels
        )
        width, height = _save_pixmap(pixmap, target, fmt)
        rect = _page_rect(page)
        if normalised is None:
            crop_width_points = float(rect.width)
        else:
            crop_width_points = float(rect.width) * (normalised[2] - normalised[0])
        return {
            "id": target.stem,
            "image_id": target.stem,
            "pdf_page": page_number,
            "path": target.as_posix(),
            "image_path": target.as_posix(),
            "crop": list(normalised) if normalised is not None else None,
            "width": width,
            "height": height,
            "render_width": width,
            "render_height": height,
            "page_width": float(rect.width),
            "page_height": float(rect.height),
            "rotation": int(getattr(page, "rotation", 0) or 0),
            "dpi": _effective_dpi(width, crop_width_points),
            "requested_dpi": float(dpi),
            "image_format": fmt,
            "max_pixels": max_pixels,
        }
    finally:
        close = getattr(document, "close", None)
        if close is not None:
            close()


def render_visual_requests(
    pdf_path: str | os.PathLike[str],
    requests: Iterable[Mapping[str, Any]],
    crops_dir: str | os.PathLike[str],
    *,
    max_crops: int = 4,
    dpi: float = 300,
    image_format: str = "png",
    max_pixels: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Render at most ``max_crops`` valid visual requests.

    Invalid or over-budget requests are returned in ``rejected`` with their
    reason instead of being silently dropped.  The caller can place these
    reasons into ``unresolved_items`` and mark the run ``needs_review``.
    """

    if max_crops < 0:
        raise PreprocessError("max_crops must be non-negative")
    target_dir = Path(crops_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for request_index, request in enumerate(requests):
        if len(rendered) >= max_crops:
            rejected.append(
                {
                    "request_index": request_index,
                    "request": dict(request),
                    "reason": f"crop budget exceeded (maximum {max_crops})",
                }
            )
            continue
        try:
            if not isinstance(request, Mapping):
                raise PreprocessError("visual request must be an object")
            page = request.get("pdf_page")
            if isinstance(page, bool):
                raise PreprocessError("pdf_page must be a positive integer")
            page_number = int(page)
            if page_number < 1:
                raise PreprocessError("pdf_page must be a positive integer")
            # Validate before rendering so bad requests are recorded without
            # opening and rendering a potentially expensive PDF page.
            normalised = _normalise_crop(request.get("crop"))
            suffix = ".jpg" if _normalise_image_format(image_format) == "jpg" else ".png"
            output = target_dir / f"page-{page_number:03d}-crop-{len(rendered) + 1:02d}{suffix}"
            entry = render_crop(
                pdf_path,
                page_number,
                normalised,
                output,
                dpi=dpi,
                image_format=image_format,
                max_pixels=max_pixels,
            )
            entry["request_index"] = request_index
            entry["reason"] = str(request.get("reason", ""))
            rendered.append(entry)
        except (PreprocessError, TypeError, ValueError) as exc:
            rejected.append(
                {
                    "request_index": request_index,
                    "request": dict(request) if isinstance(request, Mapping) else request,
                    "reason": str(exc),
                }
            )
    return {"rendered": rendered, "rejected": rejected}


def preprocess_pdf(
    pdf_path: str | os.PathLike[str],
    workspace_dir: str | os.PathLike[str],
    *,
    paper_id: str | None = None,
    render_dpi: float = 160,
    image_format: str = "png",
    image_max_pixels: int | None = 2_621_440,
    extract_text: bool = False,
    copy_original: bool = True,
) -> dict[str, Any]:
    """Prepare one PDF workspace and write its stable page manifests.

    ``workspace_dir`` is expected to be ``workspace/<paper-id>``.  Existing
    page images are overwritten only for the pages in the current PDF; the
    manifest is atomically replaced after all pages render successfully.
    """

    source = Path(pdf_path)
    if not source.is_file():
        raise PreprocessError(f"PDF file does not exist: {source}")
    workspace = Path(workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)
    resolved_id = make_paper_id(source, paper_id)
    original = workspace / "original.pdf"
    if copy_original:
        if source.resolve() != original.resolve():
            shutil.copy2(source, original)
    else:
        original = source

    fmt = _normalise_image_format(image_format)
    pages = render_pdf_pages(
        original,
        workspace / "pages",
        dpi=render_dpi,
        image_format=fmt,
        max_pixels=image_max_pixels,
    )
    digest = sha256_file(original)
    page_numbers = [entry["pdf_page"] for entry in pages]
    metadata: dict[str, Any] = {
        "schema_version": "1",
        "paper_id": resolved_id,
        "filename": source.name,
        "source_filename": source.name,
        "pdf_sha256": digest,
        "total_pages": len(pages),
        "input_coverage": {
            "pdf_pages": page_numbers,
            "complete": len(page_numbers) == len(set(page_numbers))
            and page_numbers == list(range(1, len(pages) + 1)),
            "source": "original.pdf",
        },
        "render": {
            "dpi": float(render_dpi),
            "image_format": fmt,
            "image_max_pixels": image_max_pixels,
        },
    }
    text_path: Path | None = None
    if extract_text:
        text_path = workspace / "paper.txt"
        extract_text_with_pages(original, text_path)
        metadata["text_path"] = "paper.txt"
        metadata["text_kind"] = "coarse_page_marked_text"

    pages_manifest: dict[str, Any] = {
        "schema_version": "1",
        "paper_id": resolved_id,
        "total_pages": len(pages),
        "pages": pages,
    }
    metadata_path = workspace / "metadata.json"
    pages_path = workspace / "pages.json"
    _atomic_write_json(metadata_path, metadata)
    _atomic_write_json(pages_path, pages_manifest)
    return {
        "paper_id": resolved_id,
        "workspace": workspace,
        "original_pdf": original,
        "metadata": metadata,
        "metadata_path": metadata_path,
        "pages": pages,
        "pages_manifest": pages_manifest,
        "pages_path": pages_path,
        "text_path": text_path,
    }


__all__ = [
    "PreprocessError",
    "extract_text",
    "extract_text_with_pages",
    "make_paper_id",
    "preprocess_pdf",
    "render_crop",
    "render_pdf_pages",
    "render_visual_requests",
    "sha256_file",
]
