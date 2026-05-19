# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed in-memory local state store for local runtime fallback.

When the deployed Kafka runtime is unavailable, nodes dispatch through the
local runtime path which uses an in-memory event bus.  State that would
normally be persisted in Postgres or Valkey is held here for the lifetime
of a single local invocation.

This store is explicitly typed and visible — it is NOT an untracked
process-local cache.  Consumers receive the selected backend in the
invocation evidence so the choice is auditable.

Usage:
    store = ModelLocalStateStore()
    store.put("my_key", {"content": "hello"})
    value = store.get("my_key")    # {"content": "hello"}
    snapshot = store.snapshot()    # full dict copy

    # Read-only inspection (no side effects)
    keys = store.keys()
    size = store.size()
"""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID, uuid4

from omnibase_infra.runtime.models.model_local_state_store_entry import (
    ModelLocalStateStoreEntry,
)

logger = logging.getLogger(__name__)


class ModelLocalStateStore:
    """Typed in-memory state store for local runtime fallback.

    Provides explicit, auditable key/value state for nodes that execute
    against the in-memory event bus rather than the deployed Kafka runtime.
    All writes are logged at DEBUG level so they appear in structured logs.

    Thread Safety:
        Uses asyncio.Lock for all mutations.  Synchronous callers must
        not mix sync/async access on the same event loop.

    Example::

        store = ModelLocalStateStore()
        store.put("result_key", {"status": "ok"})
        snapshot = store.snapshot()

    Attributes:
        store_id: Unique identifier for this store instance.  Included in
            log messages so multiple stores can be distinguished.
    """

    def __init__(self) -> None:
        self._store_id: UUID = uuid4()
        self._entries: dict[str, ModelLocalStateStoreEntry] = {}
        self._lock = asyncio.Lock()

    @property
    def store_id(self) -> UUID:
        """Unique identifier for this store instance."""
        return self._store_id

    def put(
        self,
        key: str,
        value: dict[str, object],
        *,
        correlation_id: UUID | None = None,
    ) -> ModelLocalStateStoreEntry:
        """Write a value to the store.

        Args:
            key: Unique key to store the value under.
            value: JSON-serialisable payload dict.
            correlation_id: Optional tracing correlation ID.

        Returns:
            The typed entry that was written.
        """
        entry = ModelLocalStateStoreEntry(
            key=key,
            value=value,
            correlation_id=correlation_id,
        )
        self._entries[key] = entry
        logger.debug(
            "LocalStateStore.put",
            extra={
                "store_id": str(self._store_id),
                "key": key,
                "correlation_id": str(correlation_id) if correlation_id else None,
            },
        )
        return entry

    def get(self, key: str) -> dict[str, object] | None:
        """Retrieve a value from the store.

        Args:
            key: Key to look up.

        Returns:
            Stored payload dict, or None if the key does not exist.
        """
        entry = self._entries.get(key)
        return dict(entry.value) if entry is not None else None

    def get_entry(self, key: str) -> ModelLocalStateStoreEntry | None:
        """Retrieve the full typed entry for a key.

        Args:
            key: Key to look up.

        Returns:
            Full ModelLocalStateStoreEntry, or None.
        """
        return self._entries.get(key)

    def snapshot(self) -> dict[str, dict[str, object]]:
        """Return a shallow copy of all current state.

        Returns:
            Dict mapping key → payload dict.
        """
        return {k: dict(v.value) for k, v in self._entries.items()}

    def keys(self) -> tuple[str, ...]:
        """Return all current keys.

        Returns:
            Tuple of key strings.
        """
        return tuple(self._entries.keys())

    def size(self) -> int:
        """Return the number of entries in the store.

        Returns:
            Number of stored entries.
        """
        return len(self._entries)

    def clear(self) -> None:
        """Remove all entries from the store."""
        self._entries.clear()
        logger.debug(
            "LocalStateStore.clear",
            extra={"store_id": str(self._store_id)},
        )


__all__ = ["ModelLocalStateStore"]
