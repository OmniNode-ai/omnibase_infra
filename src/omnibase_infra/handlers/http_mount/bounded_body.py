# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cap-bounded reading of a compressed request body (OMN-26085).

The runtime's aiohttp listeners run with ``auto_decompress=False``, so a request
body reaches the bridge exactly as it came off the socket. Decoding it here, one
bounded step at a time, lets a mount's ``max_body_bytes`` bound the raw stream
and the decoded stream before the mounted app is called: aiohttp's own parser
would otherwise expand up to a read buffer of body into memory ahead of both.
"""

from __future__ import annotations

import zlib
from collections.abc import AsyncIterator, Iterator
from typing import Protocol

_IDENTITY = "identity"
_ZLIB_CODINGS = frozenset({"gzip", "x-gzip", "deflate"})
_GZIP_WBITS = 16 + zlib.MAX_WBITS
_STREAM_STEP = 64 * 1024


class BodyTooLargeError(Exception):
    """The raw or the decoded body went over the cap."""

    def __init__(self, limit: int, size: int) -> None:
        super().__init__(f"body of at least {size} bytes exceeds {limit}")
        self.limit = limit
        self.size = size


class MalformedBodyError(Exception):
    """The body is not valid for its declared ``Content-Encoding``."""


class UnsupportedCodingError(Exception):
    """The ``Content-Encoding`` names a coding this bridge does not decode."""


class ContentStream(Protocol):
    async def read(self, n: int = -1) -> bytes: ...

    async def readany(self) -> bytes: ...


def content_coding(header: str | None) -> str | None:
    """The one coding to decode, or None for an identity body.

    Raises:
        UnsupportedCodingError: A coding other than gzip or deflate, or a stack of codings.
    """
    codings = [c for c in (p.strip().lower() for p in (header or "").split(",")) if c]
    codings = [c for c in codings if c != _IDENTITY]
    if not codings:
        return None
    if len(codings) > 1 or codings[0] not in _ZLIB_CODINGS:
        raise UnsupportedCodingError(", ".join(codings))
    return codings[0]


class BoundedInflater:
    """Incremental gzip/deflate decoder whose decoded output never exceeds a cap.

    ``feed`` asks zlib for at most one byte over the remaining allowance, so no
    more than ``limit + 1`` decoded bytes are ever produced for a capped body.
    Concatenated gzip members are decoded as one stream.
    """

    def __init__(self, coding: str, limit: int | None) -> None:
        self._coding = coding
        self._limit = limit
        self._total = 0
        self._in_member = False
        self._started = False
        self._wbits = zlib.MAX_WBITS if coding == "deflate" else _GZIP_WBITS
        self._inflater = zlib.decompressobj(self._wbits)

    def feed(self, raw: bytes) -> Iterator[bytes]:
        """Decode ``raw``; yield each decoded piece.

        Raises:
            BodyTooLargeError: The decoded total passed the cap.
            MalformedBodyError: ``raw`` is not valid for the coding.
        """
        if raw and not self._started:
            self._started = True
            # A deflate body without the zlib header is raw deflate (as aiohttp reads it).
            if self._coding == "deflate" and raw[0] & 0x0F != 8:
                self._wbits = -zlib.MAX_WBITS
                self._inflater = zlib.decompressobj(self._wbits)
        pending = raw
        while pending:
            self._in_member = True
            room = (
                _STREAM_STEP if self._limit is None else self._limit - self._total + 1
            )
            try:
                piece = self._inflater.decompress(pending, room)
            except zlib.error as exc:
                raise MalformedBodyError(str(exc)) from exc
            if piece:
                self._total += len(piece)
                if self._limit is not None and self._total > self._limit:
                    raise BodyTooLargeError(self._limit, self._total)
                yield piece
            if self._inflater.eof:
                pending = self._inflater.unused_data
                self._in_member = False
                self._inflater = zlib.decompressobj(self._wbits)
            else:
                pending = self._inflater.unconsumed_tail

    def finish(self) -> None:
        """Raise MalformedBodyError when the raw stream ended inside a member."""
        if self._in_member:
            raise MalformedBodyError("compressed body is truncated")


async def decoded_chunks(
    content: ContentStream, inflater: BoundedInflater, raw_limit: int | None
) -> AsyncIterator[bytes]:
    """Yield decoded chunks, reading at most ``raw_limit + 1`` raw bytes.

    Raises:
        BodyTooLargeError: The raw or the decoded body passed the cap.
        MalformedBodyError: The body is not valid for its coding.
    """
    raw_size = 0
    while True:
        if raw_limit is None:
            raw = await content.readany()
        else:
            raw = await content.read(raw_limit - raw_size + 1)
        if not raw:
            inflater.finish()
            return
        raw_size += len(raw)
        if raw_limit is not None and raw_size > raw_limit:
            raise BodyTooLargeError(raw_limit, raw_size)
        for piece in inflater.feed(raw):
            yield piece


async def replay(chunks: list[bytes]) -> AsyncIterator[bytes]:
    """Yield already decoded chunks."""
    for chunk in chunks:
        yield chunk


__all__ = [
    "BodyTooLargeError",
    "BoundedInflater",
    "MalformedBodyError",
    "UnsupportedCodingError",
    "content_coding",
    "decoded_chunks",
    "replay",
]
