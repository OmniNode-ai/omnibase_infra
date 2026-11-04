# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime-side client of the deploy agent's listener broker (OMN-22377, task T1).

Wire shape (chosen here; the broker lane T3 must match it):

* Transport: a unix ``SOCK_STREAM`` connection to
  ``<handover dir>/broker.sock`` (the dir is mounted at ``/run/onex-handover``).
  One connection is one attachment session: it is opened by ``get`` and its
  end, however it ends, removes the session from the broker's table.
* Framing: one JSON object per line (UTF-8, ``\\n`` terminated), both ways.
* ``{"op": "get", "pool": "<pool>"}`` -> one reply line ``{"ok": true}`` sent in
  the SAME ``sendmsg`` as one ``SCM_RIGHTS`` descriptor (the listening socket).
  The broker records the session as PROVISIONAL: it counts for nothing in the
  detach decision. An unknown pool or an absent listener answers
  ``{"ok": false, "error": "<reason>"}`` with no descriptor.
* ``{"op": "confirmed"}`` -> ``{"ok": true}``: the runtime's site is accepting;
  the session becomes CONFIRMED.
* ``{"op": "detach"}`` -> ``{"ok": true, "answer": "granted" | "wait" | "alone"}``
  (rule A6). ``granted`` marks the grant outstanding on that session until the
  connection ends; the runtime closes the connection after its drain (A3 step 4).
* The runtime never half-closes: closing the connection is the only goodbye.
"""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import Callable
from pathlib import Path

from omnibase_infra.enums.enum_detach_answer import EnumDetachAnswer

BROKER_SOCKET_NAME = "broker.sock"
_MAX_LINE = 65536


class BrokerUnavailableError(Exception):
    """The broker is absent, refused, or its session ended."""


def unix_connector(handover_dir: Path) -> Callable[[], socket.socket]:
    """Connector to the broker socket under the handover mount."""

    def connect() -> socket.socket:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(str(handover_dir / BROKER_SOCKET_NAME))
        except OSError:
            sock.close()
            raise
        return sock

    return connect


class BrokerSession:
    """One attachment session: a held broker connection."""

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._buffer = b""

    def close(self) -> None:
        """End the session (the broker removes it from its table)."""
        self._sock.close()

    async def _request(self, payload: dict[str, str]) -> dict[str, object]:
        loop = asyncio.get_running_loop()
        try:
            await loop.sock_sendall(
                self._sock, json.dumps(payload).encode("utf-8") + b"\n"
            )
            while b"\n" not in self._buffer:
                chunk = await loop.sock_recv(self._sock, _MAX_LINE)
                if not chunk:
                    raise BrokerUnavailableError("broker session ended")
                self._buffer += chunk
        except OSError as exc:
            raise BrokerUnavailableError(f"broker session failed: {exc}") from exc
        line, _, self._buffer = self._buffer.partition(b"\n")
        reply = json.loads(line)
        if not isinstance(reply, dict) or not reply.get("ok"):
            raise BrokerUnavailableError(f"broker refused: {reply!r}")
        return reply

    async def confirmed(self) -> None:
        """Tell the broker this runtime's site is accepting."""
        await self._request({"op": "confirmed"})

    async def detach(self) -> EnumDetachAnswer:
        """Ask for permission to stop accepting (A6)."""
        reply = await self._request({"op": "detach"})
        return EnumDetachAnswer(str(reply.get("answer")))


def _fetch_blocking(
    connect: Callable[[], socket.socket], pool: str, timeout: float
) -> tuple[BrokerSession, socket.socket]:
    sock = connect()
    try:
        sock.settimeout(timeout)
        sock.sendall(json.dumps({"op": "get", "pool": pool}).encode("utf-8") + b"\n")
        data, fds, _flags, _addr = socket.recv_fds(sock, _MAX_LINE, 1)
        reply = json.loads(data.split(b"\n", 1)[0]) if data else {}
        if not reply.get("ok") or len(fds) != 1:
            for fd in fds:
                socket.socket(fileno=fd).close()
            raise BrokerUnavailableError(
                f"broker did not hand over {pool!r}: {reply!r}"
            )
        listener = socket.socket(fileno=fds[0])
        sock.setblocking(False)
        session = BrokerSession(sock)
        # Bytes after the first line belong to later replies.
        session._buffer = data.partition(b"\n")[2]
        return session, listener
    except (OSError, ValueError) as exc:
        sock.close()
        raise BrokerUnavailableError(f"broker fetch failed: {exc}") from exc
    except BrokerUnavailableError:
        sock.close()
        raise


async def fetch_listener(
    connect: Callable[[], socket.socket], pool: str, timeout: float
) -> tuple[BrokerSession, socket.socket]:
    """Fetch the pool's listening socket; the returned session is PROVISIONAL."""
    fetch = asyncio.ensure_future(
        asyncio.to_thread(_fetch_blocking, connect, pool, timeout)
    )
    try:
        return await asyncio.shield(fetch)
    except asyncio.CancelledError:
        # The thread cannot be interrupted: close what it hands back late, so a
        # cancelled attach leaves neither a descriptor nor a provisional session.
        fetch.add_done_callback(_close_late_fetch)
        raise


def _close_late_fetch(
    fetch: asyncio.Future[tuple[BrokerSession, socket.socket]],
) -> None:
    if fetch.cancelled() or fetch.exception() is not None:
        return
    session, listener = fetch.result()
    listener.close()
    session.close()


__all__: list[str] = [
    "BROKER_SOCKET_NAME",
    "BrokerUnavailableError",
    "BrokerSession",
    "fetch_listener",
    "unix_connector",
]
