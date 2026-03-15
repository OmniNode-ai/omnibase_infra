# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for writer_postgres.py schema tolerance (OMN-4076).

Tests:
    - All-valid batch: all events pass through and are written.
    - One invalid event: invalid event skipped, valid events still written.
    - All-invalid batch: returns 0 without raising.

All tests are unit-level — no real PostgreSQL required (asyncpg pool is mocked).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.services.observability.skill_lifecycle.writer_postgres import (
    _REQUIRED_COMPLETED_FIELDS,
    _REQUIRED_STARTED_FIELDS,
    _validate_event_fields,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_started_event(**overrides: object) -> dict[str, object]:
    """Return a minimal valid skill-started event dict."""
    base: dict[str, object] = {
        "event_id": "evt-001",
        "run_id": "run-001",
        "skill_name": "test-skill",
        "repo_id": "omniclaude",
        "correlation_id": "corr-001",
        "emitted_at": datetime.now(UTC).isoformat(),
    }
    base.update(overrides)
    return base


def _make_completed_event(**overrides: object) -> dict[str, object]:
    """Return a minimal valid skill-completed event dict."""
    base: dict[str, object] = {
        "event_id": "evt-002",
        "run_id": "run-001",
        "skill_name": "test-skill",
        "repo_id": "omniclaude",
        "correlation_id": "corr-001",
        "status": "completed",
        "emitted_at": datetime.now(UTC).isoformat(),
    }
    base.update(overrides)
    return base


def _make_pool(executemany_result: Any = None) -> MagicMock:
    """Return a mock asyncpg Pool with acquire() context manager."""
    conn = AsyncMock()
    conn.executemany = AsyncMock(return_value=executemany_result)

    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    return pool


# ---------------------------------------------------------------------------
# _validate_event_fields unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_validate_event_fields_all_valid() -> None:
    """Valid event with all required fields returns True."""
    event = _make_started_event()
    assert _validate_event_fields(event, _REQUIRED_STARTED_FIELDS, "test") is True


@pytest.mark.unit
def test_validate_event_fields_missing_key(caplog: pytest.LogCaptureFixture) -> None:
    """Event missing a required field returns False and logs WARNING with missing fields."""
    import logging

    event = _make_started_event()
    del event["run_id"]

    with caplog.at_level(logging.WARNING):
        result = _validate_event_fields(
            event, _REQUIRED_STARTED_FIELDS, "write_started"
        )

    assert result is False
    # The WARNING message should be emitted; check the log record's extra data
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) == 1
    record = warning_records[0]
    assert "run_id" in record.__dict__.get("missing_fields", [])
    assert record.__dict__.get("context") == "write_started"


@pytest.mark.unit
def test_validate_event_fields_multiple_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Event missing multiple required fields returns False; missing_fields logged."""
    import logging

    event: dict[str, object] = {"skill_name": "only-this"}

    with caplog.at_level(logging.WARNING):
        result = _validate_event_fields(
            event, _REQUIRED_STARTED_FIELDS, "write_started"
        )

    assert result is False
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) == 1
    record = warning_records[0]
    missing = record.__dict__.get("missing_fields", [])
    # event_id, run_id, repo_id, correlation_id, emitted_at are all missing
    assert "event_id" in missing
    assert "run_id" in missing


# ---------------------------------------------------------------------------
# WriterSkillLifecyclePostgres integration-unit tests (mocked pool)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_write_started_all_valid() -> None:
    """All-valid batch: all events written, returns batch size."""
    from omnibase_infra.services.observability.skill_lifecycle.writer_postgres import (
        WriterSkillLifecyclePostgres,
    )

    pool = _make_pool()
    writer = WriterSkillLifecyclePostgres(pool)

    events = [
        _make_started_event(event_id=f"evt-{i:03d}", run_id=f"run-{i:03d}")
        for i in range(3)
    ]
    count = await writer.write_started(events)

    assert count == 3
    pool.acquire.return_value.__aenter__.return_value.executemany.assert_called_once()
    call_args = pool.acquire.return_value.__aenter__.return_value.executemany.call_args
    rows = call_args[0][1]
    assert len(rows) == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_write_started_one_invalid_skipped() -> None:
    """One invalid event is skipped; valid events are still written."""
    from omnibase_infra.services.observability.skill_lifecycle.writer_postgres import (
        WriterSkillLifecyclePostgres,
    )

    pool = _make_pool()
    writer = WriterSkillLifecyclePostgres(pool)

    valid_event = _make_started_event(event_id="evt-valid")
    invalid_event: dict[str, object] = {
        "skill_name": "old-schema-only"
    }  # missing required keys

    count = await writer.write_started([valid_event, invalid_event])

    assert count == 1
    call_args = pool.acquire.return_value.__aenter__.return_value.executemany.call_args
    rows = call_args[0][1]
    assert len(rows) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_write_started_all_invalid_returns_zero() -> None:
    """All-invalid batch returns 0 without raising KeyError or calling executemany."""
    from omnibase_infra.services.observability.skill_lifecycle.writer_postgres import (
        WriterSkillLifecyclePostgres,
    )

    pool = _make_pool()
    writer = WriterSkillLifecyclePostgres(pool)

    invalid_events: list[dict[str, object]] = [
        {"skill_name": "old-schema"},
        {"event_type": "started"},
    ]
    count = await writer.write_started(invalid_events)

    assert count == 0
    pool.acquire.return_value.__aenter__.return_value.executemany.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_write_completed_all_valid() -> None:
    """All-valid completed batch: all events written, returns batch size."""
    from omnibase_infra.services.observability.skill_lifecycle.writer_postgres import (
        WriterSkillLifecyclePostgres,
    )

    pool = _make_pool()
    writer = WriterSkillLifecyclePostgres(pool)

    events = [
        _make_completed_event(event_id=f"evt-{i:03d}", run_id=f"run-{i:03d}")
        for i in range(2)
    ]
    count = await writer.write_completed(events)

    assert count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_write_completed_one_invalid_skipped() -> None:
    """One invalid completed event skipped; valid event still written."""
    from omnibase_infra.services.observability.skill_lifecycle.writer_postgres import (
        WriterSkillLifecyclePostgres,
    )

    pool = _make_pool()
    writer = WriterSkillLifecyclePostgres(pool)

    valid_event = _make_completed_event(event_id="evt-valid")
    invalid_event: dict[str, object] = {"skill_name": "no-status-or-run-id"}

    count = await writer.write_completed([valid_event, invalid_event])

    assert count == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_write_completed_all_invalid_returns_zero() -> None:
    """All-invalid completed batch returns 0 without raising."""
    from omnibase_infra.services.observability.skill_lifecycle.writer_postgres import (
        WriterSkillLifecyclePostgres,
    )

    pool = _make_pool()
    writer = WriterSkillLifecyclePostgres(pool)

    count = await writer.write_completed([{"partial": "data"}])

    assert count == 0
    pool.acquire.return_value.__aenter__.return_value.executemany.assert_not_called()
