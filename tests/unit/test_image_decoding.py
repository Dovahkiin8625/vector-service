"""decode_image helper: validation, MIME/size limits, error mapping."""
from __future__ import annotations

import base64

import pytest

from vector_service.core.errors import ImageDecodeError, ImageTooLarge, UnsupportedMime
from vector_service.embeddings.image_decoding import decode_image


ALLOWED = {"image/jpeg", "image/png", "image/webp"}


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def test_decode_image_happy_path_returns_bytes_and_mime():
    raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
    data, mime = decode_image(_b64(raw), "image/png", max_bytes=1024, allowed_mime=ALLOWED)
    assert data == raw
    assert mime == "image/png"


def test_decode_image_rejects_unsupported_mime():
    with pytest.raises(UnsupportedMime) as ei:
        decode_image(_b64(b"x"), "image/bmp", max_bytes=1024, allowed_mime=ALLOWED)
    assert ei.value.got == "image/bmp"
    assert "image/png" in ei.value.allowed


def test_decode_image_rejects_oversize():
    raw = b"\x00" * 2048
    with pytest.raises(ImageTooLarge) as ei:
        decode_image(_b64(raw), "image/png", max_bytes=1024, allowed_mime=ALLOWED)
    assert ei.value.got == 2048
    assert ei.value.max == 1024


def test_decode_image_rejects_invalid_base64():
    with pytest.raises(ImageDecodeError) as ei:
        decode_image("!!!not-base64!!!", "image/png", max_bytes=1024, allowed_mime=ALLOWED)
    # Must be ImageDecodeError but NOT UnsupportedMime or ImageTooLarge
    assert not isinstance(ei.value, UnsupportedMime)
    assert not isinstance(ei.value, ImageTooLarge)


def test_decode_image_does_not_touch_pil():
    """The helper must NOT decode bytes into a PIL.Image — that's the
    embedder's job. It only validates and returns raw bytes."""
    raw = b"raw-bytes"
    data, _ = decode_image(_b64(raw), "image/jpeg", max_bytes=1024, allowed_mime=ALLOWED)
    assert isinstance(data, bytes)  # not a PIL.Image
    assert data == raw  # content preserved
