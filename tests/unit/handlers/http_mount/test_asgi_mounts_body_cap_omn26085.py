# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A mount's body cap bounds the raw and the decoded request stream (OMN-26085).

Real aiohttp listener (``ServiceHealth``), real bridge, real HTTP requests. The
mounted app is a raw ASGI callable that counts its ``receive`` calls, so a body
refused before dispatch shows zero calls. Two probes on aiohttp's server-side
``StreamReader`` count what the listener itself buffered (``feed_data``) and
what the bridge pulled from it (``read``/``readany``).
"""

from __future__ import annotations

import asyncio
import gzip
import sys
import zlib
from collections.abc import AsyncIterator
from pathlib import Path

import aiohttp
import pytest
from aiohttp import streams, web_protocol

from omnibase_infra.handlers.http_mount.bounded_body import (
    BodyTooLargeError,
    BoundedInflater,
    MalformedBodyError,
    UnsupportedCodingError,
    content_coding,
    decoded_chunks,
)
from tests.unit.handlers.http_mount import asgi_fixtures
from tests.unit.handlers.http_mount.test_asgi_mounts_omn21288 import (
    _bound_port,
    _manifest,
    _mount_block,
    _mounted,
    _server,
)

pytestmark = pytest.mark.unit

CAP = 128
# Pattern chosen so the gzip form is 46 bytes against 275 decoded bytes.
EXPANDING = (bytes((j * 13 + 5) % 94 + 33 for j in range(23)) * 300)[:275]


class _StreamProbe:
    """Bytes the server-side request streams buffered and handed to the bridge."""

    def __init__(self) -> None:
        self.fed = 0
        self.read = 0


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> _StreamProbe:
    stats = _StreamProbe()
    feed, read, readany = (
        streams.StreamReader.feed_data,
        streams.StreamReader.read,
        streams.StreamReader.readany,
    )

    def is_request_stream(stream: streams.StreamReader) -> bool:
        return isinstance(stream._protocol, web_protocol.RequestHandler)

    def counted_feed(self: streams.StreamReader, data: bytes, *args: int) -> None:
        if is_request_stream(self):
            stats.fed += len(data)
        feed(self, data, *args)

    async def counted_read(self: streams.StreamReader, n: int = -1) -> bytes:
        data = await read(self, n)
        if is_request_stream(self):
            stats.read += len(data)
        return data

    async def counted_readany(self: streams.StreamReader) -> bytes:
        data = await readany(self)
        if is_request_stream(self):
            stats.read += len(data)
        return data

    monkeypatch.setattr(streams.StreamReader, "feed_data", counted_feed)
    monkeypatch.setattr(streams.StreamReader, "read", counted_read)
    monkeypatch.setattr(streams.StreamReader, "readany", counted_readany)
    return stats


async def serve_decoded_cap_probe(port_file: Path) -> None:
    """Start the runtime listener for the hosted loopback deploy probe."""
    server = await _server()
    try:
        block = (
            _mount_block("/probe", "create_unread_probe") + f"\n  max_body_bytes: {CAP}"
        )
        await _mounted(server, _manifest(port_file.parent, block))
        port_file.write_text(str(_bound_port(server)), encoding="utf-8")
        await asyncio.Event().wait()
    finally:
        await server.stop()


def _compress(body: bytes, coding: str) -> bytes:
    if coding == "gzip":
        return gzip.compress(body, mtime=0)
    if coding == "deflate":
        return zlib.compress(body)
    compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    return compressor.compress(body) + compressor.flush()


async def _post(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    data: AsyncIterator[bytes] | bytes,
    *,
    encoding: str | None,
    limit: int | None = CAP,
    read_body: bool = False,
) -> tuple[int, list[int], list[bytes]]:
    calls: list[int] = []
    bodies: list[bytes] = []
    monkeypatch.setattr(
        asgi_fixtures,
        "create_app",
        lambda: asgi_fixtures.create_receive_probe(calls, bodies, read_body),
    )
    block = _mount_block("/probe", "create_app")
    if limit is not None:
        block += f"\n  max_body_bytes: {limit}"
    headers = {"Content-Encoding": encoding} if encoding else {}
    server = await _server()
    try:
        await _mounted(server, _manifest(tmp_path, block))
        url = f"http://127.0.0.1:{_bound_port(server)}/probe"
        async with (
            aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session,
            session.post(url, data=data, headers=headers) as response,
        ):
            await response.read()
            return response.status, calls, bodies
    finally:
        await server.stop()


@pytest.mark.parametrize(
    "read_body", [False, True], ids=["app-never-reads", "app-reads"]
)
@pytest.mark.parametrize("coding", ["gzip", "deflate"])
async def test_decoded_body_over_cap_is_refused_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    probe: _StreamProbe,
    coding: str,
    read_body: bool,
) -> None:
    """AC1: a body under the cap on the wire and over it decoded gets 413 at once."""
    data = _compress(EXPANDING, coding)
    assert len(data) < CAP < len(EXPANDING)
    status, calls, bodies = await _post(
        tmp_path, monkeypatch, data, encoding=coding, read_body=read_body
    )
    assert status == 413
    assert calls == []
    assert bodies == []
    assert probe.fed <= CAP + 1
    assert probe.read <= CAP + 1


async def test_a_gzip_bomb_is_never_expanded_past_the_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, probe: _StreamProbe
) -> None:
    data = gzip.compress(b"\0" * (64 * 1024 * 1024), mtime=0)
    assert len(data) < 128 * 1024
    status, calls, _ = await _post(
        tmp_path, monkeypatch, data, encoding="gzip", limit=len(data) + 1
    )
    assert status == 413
    assert calls == []
    assert probe.fed <= len(data) + 1


@pytest.mark.parametrize("encoding", [None, "gzip"], ids=["plain", "gzip"])
async def test_raw_body_over_cap_reads_at_most_the_cap_plus_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    probe: _StreamProbe,
    encoding: str | None,
) -> None:
    """AC2: a chunked raw body over the cap is refused after cap + 1 bytes.

    The client keeps sending after the refusal, so a listener that reads and
    discards the rest of the body (aiohttp's lingering close) is caught too.
    """
    raw = bytes((j * 7919) % 251 for j in range(4096))
    body = gzip.compress(raw, mtime=0) if encoding else raw
    assert len(body) > CAP
    monkeypatch.setattr(
        asgi_fixtures,
        "create_app",
        lambda: asgi_fixtures.create_receive_probe([], [], True),
    )
    block = _mount_block("/probe", "create_app") + f"\n  max_body_bytes: {CAP}"
    server = await _server()
    try:
        await _mounted(server, _manifest(tmp_path, block))
        reader, writer = await asyncio.open_connection("127.0.0.1", _bound_port(server))
        head = "POST /probe HTTP/1.1\r\nHost: probe\r\nTransfer-Encoding: chunked\r\n"
        if encoding:
            head += f"Content-Encoding: {encoding}\r\n"
        writer.write(head.encode() + b"\r\n")
        steps = [body[start : start + 200] for start in range(0, len(body), 200)]
        writer.write(f"{len(steps[0]):x}\r\n".encode() + steps[0] + b"\r\n")
        status_line = await asyncio.wait_for(reader.readline(), timeout=5)
        assert b" 413 " in status_line
        try:
            for step in steps[1:]:
                writer.write(f"{len(step):x}\r\n".encode() + step + b"\r\n")
                await writer.drain()
                await asyncio.sleep(0.02)
        except OSError:
            pass  # the listener closed the connection, as it should
        await asyncio.sleep(0.1)
        writer.close()
        assert probe.read <= CAP + 1
    finally:
        await server.stop()


@pytest.mark.parametrize("coding", [None, "gzip", "deflate", "raw-deflate"])
async def test_a_body_under_the_cap_reaches_the_app_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, coding: str | None
) -> None:
    """AC3: identity and decoded bodies under the cap arrive byte for byte."""
    body = bytes(range(100))
    data = body if coding is None else _compress(body, coding)
    header = {None: None, "raw-deflate": "deflate"}.get(coding, coding)
    status, calls, bodies = await _post(
        tmp_path, monkeypatch, data, encoding=header, read_body=True
    )
    assert status == 200
    assert calls
    assert bodies == [body]


async def test_a_decoded_body_of_exactly_the_cap_is_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = (bytes(range(8)) * CAP)[:CAP]
    data = gzip.compress(body, mtime=0)
    status, _, bodies = await _post(
        tmp_path, monkeypatch, data, encoding="gzip", limit=CAP, read_body=True
    )
    assert status == 200
    assert bodies == [body]


async def test_concatenated_gzip_members_decode_as_one_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = gzip.compress(b"first-", mtime=0) + gzip.compress(b"second", mtime=0)
    status, _, bodies = await _post(
        tmp_path, monkeypatch, data, encoding="gzip", read_body=True
    )
    assert status == 200
    assert bodies == [b"first-second"]


@pytest.mark.parametrize(
    ("data", "encoding", "expected"),
    [
        pytest.param(
            gzip.compress(b"a" * 50, mtime=0)[:-9], "gzip", 400, id="truncated"
        ),
        pytest.param(b"not gzip at all", "gzip", 400, id="not-gzip"),
        pytest.param(b"abc", "br", 415, id="brotli"),
        pytest.param(b"abc", "gzip, br", 415, id="stacked"),
    ],
)
async def test_a_bad_or_unsupported_encoding_is_refused_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    data: bytes,
    encoding: str,
    expected: int,
) -> None:
    status, calls, _ = await _post(tmp_path, monkeypatch, data, encoding=encoding)
    assert status == expected
    assert calls == []


async def test_a_compressed_body_on_a_runtime_route_is_refused(tmp_path: Path) -> None:
    """The listener no longer decodes request bodies: a runtime route says so."""
    server = await _server()
    try:
        url = f"http://127.0.0.1:{_bound_port(server)}/skill"
        async with (
            aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session,
            session.post(
                url,
                data=gzip.compress(b"{}", mtime=0),
                headers={"Content-Encoding": "gzip"},
            ) as response,
        ):
            assert response.status == 415
    finally:
        await server.stop()


class _Stream:
    """A request stream that serves raw bytes in fixed steps."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    async def read(self, n: int = -1) -> bytes:
        taken, self._data = self._data[:n], self._data[n:]
        return taken

    async def readany(self) -> bytes:
        return await self.read(64)


def _decoded_total(inflater: BoundedInflater, data: bytes) -> int:
    total = 0
    try:
        for step in range(0, len(data), 7):
            total += sum(len(piece) for piece in inflater.feed(data[step : step + 7]))
    except BodyTooLargeError as exc:
        assert total <= exc.limit
        raise
    return total


def test_the_inflater_never_produces_more_than_the_cap_plus_one() -> None:
    produced: list[int] = []
    inflater = BoundedInflater("gzip", 100)
    with pytest.raises(BodyTooLargeError) as caught:
        for piece in inflater.feed(gzip.compress(b"\0" * 10_000_000, mtime=0)):
            produced.append(len(piece))
    assert sum(produced) <= 100
    assert caught.value.size == 101


def test_the_inflater_decodes_in_steps_below_the_cap() -> None:
    data = gzip.compress(bytes(range(200)), mtime=0)
    assert _decoded_total(BoundedInflater("gzip", 200), data) == 200
    with pytest.raises(BodyTooLargeError):
        _decoded_total(BoundedInflater("gzip", 199), data)


def test_the_inflater_reports_a_truncated_stream() -> None:
    inflater = BoundedInflater("gzip", None)
    list(inflater.feed(gzip.compress(b"abcdef" * 20, mtime=0)[:-4]))
    with pytest.raises(MalformedBodyError):
        inflater.finish()


@pytest.mark.parametrize(
    ("header", "coding"),
    [
        (None, None),
        ("", None),
        ("identity", None),
        ("GZIP", "gzip"),
        ("identity, gzip", "gzip"),
        ("x-gzip", "x-gzip"),
        ("deflate", "deflate"),
    ],
)
def test_content_coding_accepts_one_zlib_coding(
    header: str | None, coding: str | None
) -> None:
    assert content_coding(header) == coding


@pytest.mark.parametrize("header", ["br", "zstd", "gzip, deflate", "compress"])
def test_content_coding_refuses_the_rest(header: str) -> None:
    with pytest.raises(UnsupportedCodingError):
        content_coding(header)


async def test_decoded_chunks_stops_reading_past_the_raw_cap() -> None:
    stream = _Stream(bytes(1000))
    chunks = decoded_chunks(stream, BoundedInflater("gzip", None), 100)
    with pytest.raises(BodyTooLargeError) as caught:
        async for _ in chunks:
            pass
    assert caught.value.size == 101
    assert len(stream._data) == 1000 - 101


if __name__ == "__main__":
    sys.stdout.buffer.write(gzip.compress(EXPANDING, mtime=0))
