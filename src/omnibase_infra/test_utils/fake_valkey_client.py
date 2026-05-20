# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pure-in-memory Valkey/Redis client for tests (OMN-9265).

Drop-in replacement for ``redis.asyncio.Redis`` used by
``RuntimeScheduler`` (and any other code that calls ``ping``, ``get``,
``set``, or ``aclose`` on a Redis client) without requiring a live
Valkey/Redis process.

Packaging note: ships in the ``omnibase_infra.test_utils`` wheel so
downstream test suites can import it from their own conftests.
"""

from __future__ import annotations

from redis.exceptions import ConnectionError as RedisConnectionError


class FakeValkeyClient:
    """In-memory ``redis.asyncio.Redis``-compatible test double.

    Supports the subset of the Redis client interface used by
    ``RuntimeScheduler``: ``ping``, ``get``, ``set``, and ``aclose``.

    Concurrent coroutine calls are safe; all operations are atomic
    at the asyncio level.

    Constructor flags for error-path coverage:

    ``ping_error``: if set to an exception instance, ``ping()`` raises it.
    ``get_error``: if set, ``get()`` raises it.
    ``set_error``: if set, ``set()`` raises it.
    ``closed``: start in the already-closed state (``ping`` raises
        ``RedisConnectionError``).

    Attributes:
        store: Mapping of key → value (both str).
        ping_call_count: Number of ``ping`` calls received.
        get_call_count: Number of ``get`` calls received.
        set_call_count: Number of ``set`` calls received.
        aclose_call_count: Number of ``aclose`` calls received.
    """

    def __init__(
        self,
        initial: dict[str, str] | None = None,
        *,
        ping_error: Exception | None = None,
        get_error: Exception | None = None,
        set_error: Exception | None = None,
        closed: bool = False,
    ) -> None:
        self.store: dict[str, str] = dict(initial) if initial else {}
        self._ping_error = ping_error
        self._get_error = get_error
        self._set_error = set_error
        self._closed = closed

        self.ping_call_count: int = 0
        self.get_call_count: int = 0
        self.set_call_count: int = 0
        self.aclose_call_count: int = 0

    async def ping(self) -> bool:
        """Simulate a ``PING``.

        Raises ``RedisConnectionError`` if the client is closed or
        ``ping_error`` was set at construction time.
        """
        self.ping_call_count += 1
        if self._closed:
            raise RedisConnectionError("FakeValkeyClient: client is closed")
        if self._ping_error is not None:
            raise self._ping_error
        return True

    async def get(self, key: str) -> str | None:
        """Return the value stored at *key*, or ``None`` if absent.

        Raises ``get_error`` if one was set at construction time.
        """
        self.get_call_count += 1
        if self._get_error is not None:
            raise self._get_error
        return self.store.get(key)

    async def set(self, key: str, value: str) -> bool:
        """Store *value* at *key*.

        Raises ``set_error`` if one was set at construction time.
        """
        self.set_call_count += 1
        if self._set_error is not None:
            raise self._set_error
        self.store[key] = value
        return True

    async def aclose(self) -> None:
        """Close the client (idempotent)."""
        self.aclose_call_count += 1
        self._closed = True


__all__: list[str] = ["FakeValkeyClient"]
