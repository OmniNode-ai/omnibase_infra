# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""In-memory keyed store for handlers that hold test or lifecycle state.

A handler keeps no mutable state of its own: it receives a ``StoreKeyedMemory``
and reads and writes through it. Handlers sharing a store use its lock to
serialize compound operations on a single event loop. Direct synchronous store
operations do not acquire the lock; callers must not share it between threads.
"""

from __future__ import annotations

import asyncio
from collections.abc import Hashable


class StoreKeyedMemory[K: Hashable, V]:
    """Dictionary-backed values keyed by ``K``."""

    def __init__(self) -> None:
        """Initialize an empty store."""
        self._items: dict[K, V] = {}
        self._lock = asyncio.Lock()

    @property
    def lock(self) -> asyncio.Lock:
        """Return the shared coroutine lock for compound store operations."""
        return self._lock

    def get(self, key: K) -> V | None:
        """Return the value stored under ``key``, or None when absent."""
        return self._items.get(key)

    def has(self, key: K) -> bool:
        """Return whether ``key`` is stored, even when its value is None."""
        return key in self._items

    def put(self, key: K, value: V) -> None:
        """Store ``value`` under ``key``, replacing any previous value."""
        self._items[key] = value

    def evict(self, key: K) -> V | None:
        """Remove and return the value stored under ``key``, or None when absent."""
        return self._items.pop(key, None)

    def reset(self) -> None:
        """Drop every value."""
        self._items.clear()

    def keys(self) -> tuple[K, ...]:
        """Return every stored key."""
        return tuple(self._items)

    def snapshot(self) -> tuple[V, ...]:
        """Return every stored value."""
        return tuple(self._items.values())

    def count(self) -> int:
        """Return the number of stored values."""
        return len(self._items)


__all__ = ["StoreKeyedMemory"]
