"""Validate and decode base64 image payloads.

Returns raw bytes; per-model decoding to PIL.Image lives in the embedder
subclass (each model has its own preprocess pipeline).
"""
from __future__ import annotations

import base64
import binascii

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
