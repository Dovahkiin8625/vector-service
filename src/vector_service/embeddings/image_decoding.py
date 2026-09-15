"""Validate and decode base64 image payloads.

Returns raw bytes; per-model decoding to PIL.Image lives in the embedder
subclass (each model has its own preprocess pipeline).
"""
from __future__ import annotations

import base64
import binascii
from typing import Callable, Iterable, Sequence

from vector_service.core.errors import ImageDecodeError, ImageTooLarge, UnsupportedMime
from vector_service.embeddings.image_base import ImageInput


def decode_image(
    b64: str,
    mime: str,
    *,
    max_bytes: int,
    allowed_mime: set[str],
) -> ImageInput:
    """Decode base64 + validate MIME + size.

    Args:
        b64: Base64-encoded image bytes (no data: URI prefix).
        mime: MIME type claim from the caller.
        max_bytes: Maximum decoded byte length.
        allowed_mime: Set of permitted MIME types.

    Returns:
        ``ImageInput`` carrying raw bytes and the verified MIME type.

    Raises:
        UnsupportedMime: ``mime`` not in ``allowed_mime``.
        ImageTooLarge: decoded bytes exceed ``max_bytes``.
        ImageDecodeError: base64 decoding failed (other reasons).
    """
    if mime not in allowed_mime:
        raise UnsupportedMime(
            f"unsupported mime: {mime!r}",
            got=mime,
            allowed=sorted(allowed_mime),
        )
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ImageDecodeError(f"base64 decode failed: {e}") from e
    if len(raw) > max_bytes:
        raise ImageTooLarge(
            f"image too large: {len(raw)} > {max_bytes}",
            got=len(raw),
            max=max_bytes,
        )
    return ImageInput(data=raw, mime=mime)


def decode_batch_or_422(
    items: Iterable,
    *,
    extract: Callable,
    max_bytes: int,
    allowed_mime: set[str],
) -> list[ImageInput]:
    """Decode a batch of images, returning ``ImageInput`` for each.

    On any per-item failure the helper returns a list of
    ``(index, code, message, extras)`` describing every failure so the
    caller can build its own 422 envelope — different routes format
    the envelope slightly differently (single-image routes omit
    ``failed_indices``; batch routes include them) so the helper
    deliberately stays at the decode layer and lets the caller raise.

    Args:
        items: iterable of route-specific items.
        extract: ``(item) -> (b64: str, mime: str)`` callable — every
            route stores ``data`` / ``mime`` under different attribute
            names (``item.data`` for image_embeddings, ``p.data`` for
            multimodal_embeddings, ``(b64, mime)`` already a tuple for
            management), so we accept a callable rather than assume a
            schema.
        max_bytes: per-image byte cap forwarded to ``decode_image``.
        allowed_mime: forwarded to ``decode_image``.

    Returns:
        ``(decoded, failures)`` where ``decoded`` is the list of
        successfully decoded ``ImageInput`` instances in input order
        and ``failures`` is the list of
        ``(index, code, message, extras)`` tuples.
    """
    decoded: list[ImageInput] = []
    failures: list[tuple[int, str, str, dict]] = []
    for i, item in enumerate(items):
        b64, mime = extract(item)
        try:
            decoded.append(
                decode_image(
                    b64, mime,
                    max_bytes=max_bytes, allowed_mime=allowed_mime,
                )
            )
        except UnsupportedMime as e:
            failures.append(
                (i, "unsupported_mime", str(e), {"got": e.got, "allowed": e.allowed})
            )
        except ImageTooLarge as e:
            failures.append(
                (i, "image_too_large", str(e), {"got": e.got, "max": e.max})
            )
        except ImageDecodeError as e:
            failures.append((i, "image_decode_failed", str(e), {}))
    return decoded, failures


def fail_envelope_422(failures: Sequence[tuple[int, str, str, dict]]) -> dict:
    """Build the canonical 422 error envelope body for batch decode
    failures.

    All-failures-share-same-code picks the first; mixed failures
    collapse to ``image_decode_failed`` (the generic catch-all).
    Caller wraps this in ``HTTPException(status_code=422, detail=...)``.
    """
    code = failures[0][1]
    if not all(f[1] == code for f in failures):
        code = "image_decode_failed"
    return {
        "error": {
            "code": code,
            "message": failures[0][2],
            "failed_indices": [f[0] for f in failures],
            "failures": [
                {"index": f[0], "code": f[1], "message": f[2], **f[3]}
                for f in failures
            ],
        }
    }

