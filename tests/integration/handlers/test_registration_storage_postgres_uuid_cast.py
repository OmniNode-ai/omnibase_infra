# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration regression guard for OMN-9041.

Exercises ``HandlerRegistrationStoragePostgres.query_registrations`` end-to-end
against a real PostgreSQL instance. Ensures that whatever concrete type asyncpg
returns for the VARCHAR ``node_id`` column (``str`` on some asyncpg versions,
``asyncpg.pgproto.pgproto.UUID`` on others) is cast to stdlib ``uuid.UUID``
without raising ``AttributeError``.

Context:
    OMN-9041 — production crash in ``UUID(row["node_id"])``. Fix: wrap with
    ``str()`` to coerce to canonical hex before handing to stdlib UUID.

Skip policy:
    Skipped when PostgreSQL is not available (no ``POSTGRES_HOST`` /
    ``OMNIBASE_INFRA_DB_URL``). Enabled in CI via test splits that run the
    full suite with Postgres configured.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypedDict
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.handlers.registration_storage.handler_registration_storage_postgres import (
    HandlerRegistrationStoragePostgres,
)
from omnibase_infra.handlers.registration_storage.models import (
    ModelDeleteRegistrationRequest,
)
from omnibase_infra.nodes.node_registration_storage_effect.models import (
    ModelRegistrationRecord,
    ModelStorageQuery,
)
from tests.helpers.util_postgres import PostgresConfig

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


_postgres_config = PostgresConfig.from_env()
POSTGRES_AVAILABLE = _postgres_config.is_configured


class _PostgresConfigDict(TypedDict):
    host: str
    port: int
    database: str
    user: str
    password: str


def _resolve_postgres_config() -> _PostgresConfigDict:
    return {
        "host": _postgres_config.host or "localhost",
        "port": _postgres_config.port,
        "database": _postgres_config.database or "omnibase_infra",
        "user": _postgres_config.user,
        "password": _postgres_config.password or "",
    }


@pytest.mark.skipif(
    not POSTGRES_AVAILABLE,
    reason="PostgreSQL not available (set OMNIBASE_INFRA_DB_URL or POSTGRES_HOST+POSTGRES_PASSWORD)",
)
class TestRegistrationStoragePostgresUuidCast:
    """Live-DB regression guard for OMN-9041 UUID cast."""

    @pytest.fixture
    async def handler(
        self,
    ) -> AsyncGenerator[HandlerRegistrationStoragePostgres, None]:
        pg = _resolve_postgres_config()
        handler = HandlerRegistrationStoragePostgres(
            container=MagicMock(spec=ModelONEXContainer),
            host=pg["host"],
            port=pg["port"],
            database=pg["database"],
            user=pg["user"],
            password=pg["password"],
        )
        yield handler
        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_query_round_trip_produces_stdlib_uuid(
        self,
        handler: HandlerRegistrationStoragePostgres,
    ) -> None:
        """Store + query round-trip returns ``record.node_id`` as stdlib ``uuid.UUID``.

        Pre-fix (OMN-9041) this round-trip crashes with
        ``InfraConnectionError('PostgreSQL query failed: AttributeError')``
        when asyncpg decodes the VARCHAR column as its pgproto UUID type.
        Post-fix the cast succeeds regardless of the row value's concrete type.
        """
        correlation_id = uuid4()
        original_node_id = uuid4()

        record = ModelRegistrationRecord(
            node_id=original_node_id,
            node_type=EnumNodeKind.EFFECT,
            node_version=ModelSemVer.parse("1.0.0"),
            capabilities=["omn-9041.regression"],
            endpoints={},
            metadata={"ticket": "OMN-9041"},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            correlation_id=correlation_id,
        )

        try:
            store_result = await handler.store_registration(
                record=record,
                correlation_id=correlation_id,
            )
            assert store_result.success, (
                f"store_registration failed: {store_result.error}"
            )

            query_result = await handler.query_registrations(
                query=ModelStorageQuery(node_id=original_node_id),
                correlation_id=correlation_id,
            )
            assert query_result.success, (
                f"query_registrations failed: {query_result.error}"
            )
            assert len(query_result.records) == 1, (
                "Expected exactly one record for our just-stored node_id"
            )

            fetched = query_result.records[0]
            assert isinstance(fetched.node_id, UUID), (
                f"node_id must be stdlib uuid.UUID after cast, got {type(fetched.node_id).__name__}"
            )
            assert fetched.node_id == original_node_id
        finally:
            await handler.delete_registration(
                ModelDeleteRegistrationRequest(
                    node_id=original_node_id,
                    correlation_id=correlation_id,
                )
            )
