# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A TCP listener that speaks just enough Kafka to refuse a SASL login (OMN-19452).

The real aiokafka client connects to this listener, negotiates API versions,
completes the SASL handshake and sends its first authenticate frame, and the
listener answers that frame with the broker error ``SASL_AUTHENTICATION_FAILED``
(code 58). Nothing between the test and the refusal is mocked: the client under
test is the one production builds, so what it does with the refusal -- which is
to log it and then raise a generic "Unable to bootstrap" -- is the behaviour the
code under test has to classify.

Every name and value here is synthetic.
"""

from __future__ import annotations

import asyncio
import struct
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from aiokafka.protocol.admin import (
    ApiVersionResponse_v0,
    SaslAuthenticateResponse_v0,
    SaslHandShakeResponse_v1,
)

__all__ = ["SASL_MECHANISM", "serve_sasl_refusing_broker"]

SASL_MECHANISM = "SCRAM-SHA-256"

_API_KEY_SASL_HANDSHAKE = 17
_API_KEY_API_VERSIONS = 18
_API_KEY_SASL_AUTHENTICATE = 36
_SASL_AUTHENTICATION_FAILED = 58

# Only the three keys the connection handshake touches. Advertising nothing else
# is deliberate: a client that asks this broker for anything past the
# handshake has gone further than a refused login should allow.
_ADVERTISED_API_VERSIONS = [
    (_API_KEY_SASL_HANDSHAKE, 0, 1),
    (_API_KEY_API_VERSIONS, 0, 0),
    (_API_KEY_SASL_AUTHENTICATE, 0, 0),
]


def _response_body(api_key: int) -> bytes:
    if api_key == _API_KEY_API_VERSIONS:
        return ApiVersionResponse_v0(0, _ADVERTISED_API_VERSIONS).encode()
    if api_key == _API_KEY_SASL_HANDSHAKE:
        return SaslHandShakeResponse_v1(0, [SASL_MECHANISM]).encode()
    if api_key == _API_KEY_SASL_AUTHENTICATE:
        return SaslAuthenticateResponse_v0(
            _SASL_AUTHENTICATION_FAILED, "login refused by the fake broker", b""
        ).encode()
    raise AssertionError(f"the fake broker was asked for api key {api_key}")


async def _serve_connection(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    try:
        while True:
            (size,) = struct.unpack(">i", await reader.readexactly(4))
            frame = await reader.readexactly(size)
            api_key, _api_version, correlation_id = struct.unpack(">hhi", frame[:8])
            body = _response_body(api_key)
            payload = struct.pack(">i", correlation_id) + body
            writer.write(struct.pack(">i", len(payload)) + payload)
            await writer.drain()
    except (asyncio.IncompleteReadError, ConnectionError):
        return
    finally:
        writer.close()


@contextmanager
def serve_sasl_refusing_broker() -> Iterator[str]:
    """Serve on an ephemeral loopback port from a thread; yield ``host:port``.

    A thread rather than the caller's loop because the code under test drives
    its own ``asyncio.run`` from a synchronous entry point, which cannot share
    a loop with a listener.
    """
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    box: dict[str, asyncio.Server] = {}

    async def _start() -> None:
        box["server"] = await asyncio.start_server(_serve_connection, "127.0.0.1", 0)
        ready.set()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_start())
        loop.run_forever()

    thread = threading.Thread(target=_run, name="fake-sasl-broker", daemon=True)
    thread.start()
    assert ready.wait(timeout=10), "the fake broker did not start listening"
    server = box["server"]
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"127.0.0.1:{port}"
    finally:

        async def _stop() -> None:
            server.close()
            await server.wait_closed()

        asyncio.run_coroutine_threadsafe(_stop(), loop).result(timeout=10)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        loop.close()
