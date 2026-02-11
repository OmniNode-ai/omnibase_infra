# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Pytest fixtures for injection effectiveness golden path integration tests.

This module provides fixtures for testing:
- WriterInjectionEffectivenessPostgres metric writes
- LedgerSinkInjectionEffectivenessPostgres ledger entries
- End-to-end correlation ID traceability

Fixtures connect to real PostgreSQL (configured via environment variables)
and provide cleanup of test data after each test.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import quote_plus
from uuid import UUID, uuid4

import pytest

if TYPE_CHECKING:
    import asyncpg

    from omnibase_infra.services.observability.injection_effectiveness.ledger_sink_postgres import (
        LedgerSinkInjectionEffectivenessPostgres,
    )
    from omnibase_infra.services.observability.injection_effectiveness.writer_postgres import (
        WriterInjectionEffectivenessPostgres,
    )

logger = logging.getLogger(__name__)

# NOTE: pytestmark in conftest.py does NOT propagate to test files in the same
# directory. Each test module must declare its own pytestmark. See:
# tests/integration/conftest.py for the canonical documentation of this behavior.


def _get_postgres_dsn() -> str | None:
    """Build PostgreSQL DSN from environment variables."""
    host = os.getenv("POSTGRES_HOST")
    password = os.getenv("POSTGRES_PASSWORD")

    if not host or not password:
        return None

    port = os.getenv("POSTGRES_PORT", "5436")
    database = os.getenv("POSTGRES_DATABASE", "omninode_bridge")
    user = os.getenv("POSTGRES_USER", "postgres")

    return f"postgresql://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{database}"


@pytest.fixture
def postgres_dsn() -> str:
    """Get PostgreSQL DSN or skip test if not configured."""
    dsn = _get_postgres_dsn()
    if dsn is None:
        pytest.skip(
            "PostgreSQL not configured (missing POSTGRES_HOST or POSTGRES_PASSWORD)"
        )
    return dsn


@pytest.fixture
async def postgres_pool(postgres_dsn: str) -> AsyncGenerator[asyncpg.Pool, None]:
    """Create a PostgreSQL connection pool for injection effectiveness tests."""
    import asyncpg

    pool = await asyncpg.create_pool(postgres_dsn, min_size=1, max_size=5, timeout=10.0)
    try:
        yield pool
    finally:
        await pool.close()


@pytest.fixture
async def metrics_writer(
    postgres_pool: asyncpg.Pool,
) -> WriterInjectionEffectivenessPostgres:
    """Create a WriterInjectionEffectivenessPostgres for tests."""
    from omnibase_infra.services.observability.injection_effectiveness.writer_postgres import (
        WriterInjectionEffectivenessPostgres,
    )

    return WriterInjectionEffectivenessPostgres(
        pool=postgres_pool,
        circuit_breaker_threshold=5,
        circuit_breaker_reset_timeout=60.0,
    )


@pytest.fixture
async def ledger_sink(
    postgres_pool: asyncpg.Pool,
) -> LedgerSinkInjectionEffectivenessPostgres:
    """Create a LedgerSinkInjectionEffectivenessPostgres for tests."""
    from omnibase_infra.services.observability.injection_effectiveness.ledger_sink_postgres import (
        LedgerSinkInjectionEffectivenessPostgres,
    )

    return LedgerSinkInjectionEffectivenessPostgres(
        pool=postgres_pool,
        circuit_breaker_threshold=5,
        circuit_breaker_reset_timeout=60.0,
    )


@pytest.fixture
async def cleanup_injection_test_data(
    postgres_pool: asyncpg.Pool,
) -> AsyncGenerator[dict[str, list[UUID]], None]:
    """Track and cleanup injection effectiveness test data after each test.

    Yields a dict with lists to track session_ids and ledger_entry_ids for cleanup.

    Cleanup order matters for FK constraints:
    1. pattern_hit_rates (no FK dependency)
    2. latency_breakdowns (FK → injection_effectiveness.session_id)
    3. injection_effectiveness (parent table)
    4. event_ledger (independent table)
    """
    tracker: dict[str, list[UUID]] = {
        "session_ids": [],
        "ledger_entry_ids": [],
        "pattern_ids": [],
    }

    yield tracker

    # Cleanup all tracked test data (order: children before parents for FK safety)
    async with postgres_pool.acquire() as conn:
        # Delete pattern_hit_rates for test patterns
        if tracker["pattern_ids"]:
            valid_pattern_ids = [str(pid) for pid in tracker["pattern_ids"]]
            await conn.execute(
                "DELETE FROM pattern_hit_rates WHERE pattern_id = ANY($1::uuid[])",
                valid_pattern_ids,
            )
            logger.debug("Cleaned up %d pattern_hit_rates rows", len(valid_pattern_ids))

        # Delete latency_breakdowns and injection_effectiveness (FK order: children first)
        if tracker["session_ids"]:
            valid_session_ids = [str(sid) for sid in tracker["session_ids"]]
            await conn.execute(
                "DELETE FROM latency_breakdowns WHERE session_id = ANY($1::uuid[])",
                valid_session_ids,
            )
            logger.debug(
                "Cleaned up latency_breakdowns for %d sessions", len(valid_session_ids)
            )
            await conn.execute(
                "DELETE FROM injection_effectiveness WHERE session_id = ANY($1::uuid[])",
                valid_session_ids,
            )
            logger.debug(
                "Cleaned up %d injection_effectiveness rows", len(valid_session_ids)
            )

        # Delete event_ledger entries
        if tracker["ledger_entry_ids"]:
            valid_ledger_ids = [
                str(lid) for lid in tracker["ledger_entry_ids"] if lid is not None
            ]
            if valid_ledger_ids:
                await conn.execute(
                    "DELETE FROM event_ledger WHERE ledger_entry_id = ANY($1::uuid[])",
                    valid_ledger_ids,
                )
                logger.debug("Cleaned up %d event_ledger rows", len(valid_ledger_ids))


@pytest.fixture
def make_context_utilization_event() -> Callable[
    ..., tuple[dict[str, Any], UUID, UUID]
]:
    """Factory fixture to create ModelContextUtilizationEvent with defaults."""
    from omnibase_infra.services.observability.injection_effectiveness.models.model_pattern_utilization import (
        ModelPatternUtilization,
    )

    def _make(**overrides: Any) -> tuple[dict[str, Any], UUID, UUID]:
        pattern_id_1 = overrides.pop("pattern_id_1", uuid4())
        pattern_id_2 = overrides.pop("pattern_id_2", uuid4())

        defaults: dict[str, Any] = {
            "session_id": uuid4(),
            "correlation_id": uuid4(),
            "cohort": "treatment",
            "cohort_identity_type": "session_id",
            "total_injected_tokens": 1500,
            "patterns_injected": 2,
            "utilization_score": 0.85,
            "utilization_method": "identifier_match",
            "injected_identifiers_count": 20,
            "reused_identifiers_count": 17,
            "pattern_utilizations": (
                ModelPatternUtilization(
                    pattern_id=pattern_id_1,
                    utilization_score=0.9,
                    utilization_method="identifier_match",
                ),
                ModelPatternUtilization(
                    pattern_id=pattern_id_2,
                    utilization_score=0.8,
                    utilization_method="identifier_match",
                ),
            ),
            "created_at": datetime.now(UTC),
        }
        defaults.update(overrides)
        return defaults, pattern_id_1, pattern_id_2

    return _make
