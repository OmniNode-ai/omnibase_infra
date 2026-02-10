# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Pytest fixtures for Event Ledger integration tests.

This module provides fixtures for testing:
- HandlerLedgerAppend idempotent writes
- HandlerLedgerQuery correlation_id lookups
- E2E pipeline from event to database

Fixtures connect to real PostgreSQL (configured via environment variables)
and provide cleanup of test data after each test.
"""

from __future__ import annotations

import base64
import logging
import os
from collections.abc import AsyncGenerator, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)

# Marker for all tests in this directory
pytestmark = [pytest.mark.postgres]


def _get_postgres_dsn() -> str | None:
    """Build PostgreSQL DSN from environment variables.

    Primary source: ``OMNIBASE_INFRA_DB_URL``.
    Fallback: individual ``POSTGRES_*`` env vars.

    Returns:
        DSN string if configuration is available, None otherwise.
    """
    db_url = os.getenv("OMNIBASE_INFRA_DB_URL")
    if db_url:
        # Basic validation: ensure the user-provided DSN is well-formed
        from urllib.parse import urlparse

        parsed = urlparse(db_url)
        if parsed.scheme not in ("postgresql", "postgres"):
            raise ValueError(
                f"OMNIBASE_INFRA_DB_URL has invalid scheme '{parsed.scheme}'. "
                "Expected 'postgresql://' or 'postgres://'."
            )
        return db_url

    host = os.getenv("POSTGRES_HOST")
    password = os.getenv("POSTGRES_PASSWORD")

    if not host or not password:
        return None

    from urllib.parse import quote_plus

    # 5436 = external port on remote infra server (192.168.86.200)
    port = os.getenv("POSTGRES_PORT", "5436")
    user = os.getenv("POSTGRES_USER", "postgres")

    # URL-encode credentials to handle special characters (@, :, /, %, etc.)
    encoded_user = quote_plus(user, safe="")
    encoded_password = quote_plus(password, safe="")

    return (
        f"postgresql://{encoded_user}:{encoded_password}@{host}:{port}/omnibase_infra"
    )


@pytest.fixture
def postgres_dsn() -> str:
    """Get PostgreSQL DSN or skip test if not configured.

    Returns:
        PostgreSQL DSN string.

    Raises:
        pytest.skip: If PostgreSQL is not configured.
    """
    dsn = _get_postgres_dsn()
    if dsn is None:
        pytest.skip(
            "PostgreSQL not configured (set OMNIBASE_INFRA_DB_URL or POSTGRES_HOST+POSTGRES_PASSWORD)"
        )
    return dsn


@pytest.fixture
async def postgres_pool(postgres_dsn: str) -> AsyncGenerator[asyncpg.Pool, None]:
    """Create a PostgreSQL connection pool for ledger tests.

    This fixture creates a dedicated pool for integration tests
    and closes it after the test completes.

    Yields:
        asyncpg.Pool connected to the test database.
    """
    import asyncpg

    pool = await asyncpg.create_pool(postgres_dsn, min_size=1, max_size=5, timeout=10.0)
    try:
        yield pool
    finally:
        await pool.close()


@pytest.fixture
async def cleanup_event_ledger(
    postgres_pool: asyncpg.Pool,
) -> AsyncGenerator[list[UUID | None], None]:
    """Track and cleanup ledger entries created during tests.

    This fixture provides a list that tests can append ledger_entry_ids to.
    After the test completes, all tracked entries are deleted.

    Note:
        The list accepts ``UUID | None`` because duplicate appends return
        ``ledger_entry_id=None``. None values are filtered during cleanup.

    Usage:
        async def test_something(cleanup_event_ledger, ...):
            result = await handler.append(payload)
            cleanup_event_ledger.append(result.ledger_entry_id)
            # ... assertions ...
            # Cleanup happens automatically after test

    Yields:
        List to collect ledger_entry_ids for cleanup (None values filtered).
    """
    entry_ids: list[UUID | None] = []

    yield entry_ids

    # Cleanup tracked entries
    if entry_ids:
        async with postgres_pool.acquire() as conn:
            # Filter out None values (duplicates don't return an ID)
            valid_ids = [str(eid) for eid in entry_ids if eid is not None]
            if valid_ids:
                await conn.execute(
                    "DELETE FROM event_ledger WHERE ledger_entry_id = ANY($1::uuid[])",
                    valid_ids,
                )
                logger.debug("Cleaned up %d ledger entries", len(valid_ids))


@pytest.fixture
async def db_handler(
    postgres_dsn: str, mock_container: MagicMock
) -> AsyncGenerator[HandlerDb, None]:
    """Create and initialize a HandlerDb for ledger tests.

    Yields:
        Initialized HandlerDb connected to test database.
    """
    from omnibase_infra.handlers.handler_db import HandlerDb

    handler = HandlerDb(mock_container)
    await handler.initialize({"dsn": postgres_dsn})

    try:
        yield handler
    finally:
        await handler.shutdown()


@pytest.fixture
async def ledger_append_handler(
    db_handler: HandlerDb, mock_container: MagicMock
) -> AsyncGenerator[HandlerLedgerAppend, None]:
    """Create and initialize a HandlerLedgerAppend for tests.

    Yields:
        Initialized HandlerLedgerAppend ready for append operations.
    """
    from omnibase_infra.nodes.node_ledger_write_effect.handlers.handler_ledger_append import (
        HandlerLedgerAppend,
    )

    handler = HandlerLedgerAppend(mock_container, db_handler)
    await handler.initialize({})

    try:
        yield handler
    finally:
        await handler.shutdown()


@pytest.fixture
async def ledger_query_handler(
    db_handler: HandlerDb, mock_container: MagicMock
) -> AsyncGenerator[HandlerLedgerQuery, None]:
    """Create and initialize a HandlerLedgerQuery for tests.

    Yields:
        Initialized HandlerLedgerQuery ready for query operations.
    """
    from omnibase_infra.nodes.node_ledger_write_effect.handlers.handler_ledger_query import (
        HandlerLedgerQuery,
    )

    handler = HandlerLedgerQuery(mock_container, db_handler)
    await handler.initialize({})

    try:
        yield handler
    finally:
        await handler.shutdown()


@pytest.fixture
def sample_ledger_payload() -> ModelPayloadLedgerAppend:
    """Create a sample ModelPayloadLedgerAppend for testing.

    Returns:
        A valid payload with unique Kafka position and test data.
    """
    from omnibase_infra.nodes.reducers.models.model_payload_ledger_append import (
        ModelPayloadLedgerAppend,
    )

    # Generate unique Kafka position to avoid conflicts
    unique_offset = int(uuid4().int % (2**62))  # Large unique offset

    return ModelPayloadLedgerAppend(
        topic="test.integration.ledger.events.v1",
        partition=0,
        kafka_offset=unique_offset,
        event_key=base64.b64encode(b"test-key").decode("ascii"),
        event_value=base64.b64encode(b'{"test": "data", "node_id": "test-123"}').decode(
            "ascii"
        ),
        correlation_id=uuid4(),
        envelope_id=uuid4(),
        event_type="TestEvent",
        source="ledger-integration-test",
        event_timestamp=datetime.now(UTC),
        onex_headers={"test_header": "test_value"},
    )


@pytest.fixture
def make_ledger_payload() -> Callable[..., ModelPayloadLedgerAppend]:
    """Factory fixture to create ledger payloads with custom parameters.

    Returns:
        Callable that creates ModelPayloadLedgerAppend with given overrides.

    Usage:
        async def test_custom_payload(make_ledger_payload):
            payload = make_ledger_payload(
                topic="custom.topic.v1",
                correlation_id=my_correlation_id,
            )
    """
    from omnibase_infra.nodes.reducers.models.model_payload_ledger_append import (
        ModelPayloadLedgerAppend,
    )

    def _make(**overrides: Any) -> ModelPayloadLedgerAppend:
        # Generate unique Kafka position
        unique_offset = int(uuid4().int % (2**62))

        defaults = {
            "topic": "test.integration.ledger.events.v1",
            "partition": 0,
            "kafka_offset": unique_offset,
            "event_key": base64.b64encode(b"test-key").decode("ascii"),
            "event_value": base64.b64encode(b'{"test": "data"}').decode("ascii"),
            "correlation_id": uuid4(),
            "envelope_id": uuid4(),
            "event_type": "TestEvent",
            "source": "ledger-integration-test",
            "event_timestamp": datetime.now(UTC),
            "onex_headers": {},
        }

        # Apply overrides
        defaults.update(overrides)

        return ModelPayloadLedgerAppend(**defaults)

    return _make


# Re-export TYPE_CHECKING imports for type hints
if TYPE_CHECKING:
    from omnibase_infra.handlers.handler_db import HandlerDb
    from omnibase_infra.nodes.node_ledger_write_effect.handlers.handler_ledger_append import (
        HandlerLedgerAppend,
    )
    from omnibase_infra.nodes.node_ledger_write_effect.handlers.handler_ledger_query import (
        HandlerLedgerQuery,
    )
    from omnibase_infra.nodes.reducers.models.model_payload_ledger_append import (
        ModelPayloadLedgerAppend,
    )
