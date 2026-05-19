# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pure-in-memory ``ProtocolPostgresAdapter`` for tests (OMN-9265).

Used by unit/integration tests that exercise code depending on a
``ProtocolPostgresAdapter`` without standing up a real PostgreSQL instance.

Packaging note: ships in the ``omnibase_infra.test_utils`` wheel so
downstream test suites can import it from their own conftests.
"""

from __future__ import annotations

import asyncio
from uuid import UUID

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.primitives import ModelSemVer
from omnibase_infra.models.model_backend_result import ModelBackendResult


class FakePostgresAdapter:
    """In-memory ``ProtocolPostgresAdapter`` backed by dicts.

    Fully implements ``upsert`` and ``deactivate``.  Concurrent coroutine
    calls are safe: an ``asyncio.Lock`` guards all mutations.

    Constructor accepts an optional ``fail_on_upsert`` flag and
    ``fail_on_deactivate`` flag for error-path testing.

    Attributes:
        records: Mapping of ``node_id`` → latest upsert kwargs, present
            only for active records.
        deactivated: Set of ``node_id`` values that have been deactivated.
        upsert_call_count: Number of ``upsert`` calls received.
        deactivate_call_count: Number of ``deactivate`` calls received.
    """

    def __init__(
        self,
        *,
        fail_on_upsert: bool = False,
        fail_on_deactivate: bool = False,
    ) -> None:
        self.records: dict[UUID, dict[str, object]] = {}
        self.deactivated: set[UUID] = set()
        self.upsert_call_count: int = 0
        self.deactivate_call_count: int = 0
        self._fail_on_upsert = fail_on_upsert
        self._fail_on_deactivate = fail_on_deactivate
        self._lock: asyncio.Lock = asyncio.Lock()

    async def upsert(
        self,
        node_id: UUID,
        node_type: EnumNodeKind,
        node_version: ModelSemVer,
        endpoints: dict[str, str],
        metadata: dict[str, str],
    ) -> ModelBackendResult:
        """Store or update a node registration record in memory.

        Returns a success ``ModelBackendResult`` unless ``fail_on_upsert``
        was set at construction time, in which case an error result is returned.
        """
        async with self._lock:
            self.upsert_call_count += 1
            if self._fail_on_upsert:
                return ModelBackendResult(
                    success=False,
                    error="FakePostgresAdapter: forced upsert failure",
                    backend_id="fake-postgres",
                )
            self.records[node_id] = {
                "node_id": node_id,
                "node_type": node_type,
                "node_version": node_version,
                "endpoints": dict(endpoints),
                "metadata": dict(metadata),
            }
            self.deactivated.discard(node_id)
            return ModelBackendResult(
                success=True,
                backend_id="fake-postgres",
            )

    async def deactivate(self, node_id: UUID) -> ModelBackendResult:
        """Soft-delete a node registration record.

        Returns a success ``ModelBackendResult`` unless
        ``fail_on_deactivate`` was set at construction time.
        """
        async with self._lock:
            self.deactivate_call_count += 1
            if self._fail_on_deactivate:
                return ModelBackendResult(
                    success=False,
                    error="FakePostgresAdapter: forced deactivate failure",
                    backend_id="fake-postgres",
                )
            self.deactivated.add(node_id)
            self.records.pop(node_id, None)
            return ModelBackendResult(
                success=True,
                backend_id="fake-postgres",
            )


__all__: list[str] = ["FakePostgresAdapter"]
