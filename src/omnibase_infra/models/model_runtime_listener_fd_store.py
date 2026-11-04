# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The listening sockets the deploy agent holds, by pool name (OMN-22379)."""

from __future__ import annotations

import socket
import threading

from pydantic import BaseModel, ConfigDict, PrivateAttr


class ModelRuntimeListenerFdStore(BaseModel):
    """Pool name to listening socket; the handler borrows, never assigns."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _sockets: dict[str, socket.socket] = PrivateAttr(default_factory=dict)

    def keep(self, pool: str, sock: socket.socket) -> None:
        """Hold ``sock`` for ``pool``, closing any socket held before."""
        with self._lock:
            previous = self._sockets.get(pool)
            self._sockets[pool] = sock
        if previous is not None and previous is not sock:
            previous.close()

    def socket_for(self, pool: str) -> socket.socket | None:
        with self._lock:
            return self._sockets.get(pool)

    def pools(self) -> tuple[str, ...]:
        """The held pools, in the order they were first kept."""
        with self._lock:
            return tuple(self._sockets)

    def release_all(self) -> tuple[tuple[str, socket.socket], ...]:
        """Give up every held socket (the caller closes them)."""
        with self._lock:
            held = tuple(self._sockets.items())
            self._sockets.clear()
        return held


__all__: list[str] = ["ModelRuntimeListenerFdStore"]
