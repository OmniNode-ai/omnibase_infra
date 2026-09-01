# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Offline query-shape tests for the first-effect PostgreSQL adapter.

These fakes do not prove PostgreSQL concurrency. The integration test applies
the migration and exercises the adapter against a throwaway PostgreSQL cluster.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import UUID

import asyncpg
import pytest

from omnibase_infra.errors.repository import RepositoryExecutionError
from omnibase_infra.runtime.first_effect_ledger import (
    EnumFirstEffectLedgerState,
    FirstEffectLedgerConflictError,
    ModelFirstEffectAuthorizationRequest,
    PostgresFirstEffectLedger,
)

_AUTH = "a" * 64
_NONCE = "b" * 64
_REQUEST = "c" * 64
_MANIFEST = "d" * 64
_EVIDENCE = "e" * 64
_RECEIPT = "f" * 64
_CORRELATION = UUID("00000000-0000-0000-0000-000000000001")


def _row(
    state: EnumFirstEffectLedgerState, version: int, /, *, evidence: str | None = None
) -> dict[str, object]:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return {
        "authorization_digest": _AUTH,
        "nonce_digest": _NONCE,
        "correlation_id": _CORRELATION,
        "request_digest": _REQUEST,
        "manifest_hash": _MANIFEST,
        "state": state.value,
        "publish_evidence_hash": evidence,
        "terminal_receipt_hash": _RECEIPT
        if state is EnumFirstEffectLedgerState.TERMINAL_OBSERVED
        else None,
        "issued_at": now,
        "preflight_consumed_at": now
        if state is not EnumFirstEffectLedgerState.ISSUED
        else None,
        "publishing_at": now
        if state
        not in (
            EnumFirstEffectLedgerState.ISSUED,
            EnumFirstEffectLedgerState.PREFLIGHT_CONSUMED,
        )
        else None,
        "published_unknown_at": now
        if state is EnumFirstEffectLedgerState.PUBLISHED_UNKNOWN
        else None,
        "terminal_observed_at": now
        if state is EnumFirstEffectLedgerState.TERMINAL_OBSERVED
        else None,
        "blocked_at": now if state is EnumFirstEffectLedgerState.BLOCKED else None,
        "updated_at": now,
        "version": version,
    }


class _Acquire:
    def __init__(self, connection: _FakeConnection) -> None:
        self._connection = connection

    async def __aenter__(self) -> _FakeConnection:
        return self._connection

    async def __aexit__(self, *args: object) -> bool:
        return False


class _FakeConnection:
    def __init__(self, rows: list[dict[str, object] | BaseException | None]) -> None:
        self._rows = rows
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def fetchrow(self, sql: str, *values: object) -> dict[str, object] | None:
        self.calls.append((sql, values))
        result = self._rows.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class _FakePool:
    def __init__(self, connection: _FakeConnection) -> None:
        self._connection = connection

    def acquire(self) -> _Acquire:
        return _Acquire(self._connection)


def _adapter(connection: _FakeConnection) -> PostgresFirstEffectLedger:
    async def _factory() -> _FakePool:
        return _FakePool(connection)

    return PostgresFirstEffectLedger(pool_factory=_factory)  # type: ignore[arg-type]


@pytest.mark.unit
def test_issue_persists_only_redacted_identity_and_rejects_duplicate() -> None:
    connection = _FakeConnection([_row(EnumFirstEffectLedgerState.ISSUED, 0), None])
    adapter = _adapter(connection)
    request = ModelFirstEffectAuthorizationRequest(
        authorization_digest=_AUTH,
        nonce_digest=_NONCE,
        correlation_id=_CORRELATION,
        request_digest=_REQUEST,
        manifest_hash=_MANIFEST,
    )

    issued = asyncio.run(adapter.issue(request))
    assert issued.state is EnumFirstEffectLedgerState.ISSUED
    with pytest.raises(RuntimeError, match="already bound"):
        asyncio.run(adapter.issue(request))
    sql, values = connection.calls[0]
    assert "ON CONFLICT DO NOTHING" in sql
    assert "payload" not in sql.lower()
    assert values == (_AUTH, _NONCE, _CORRELATION, _REQUEST, _MANIFEST)


@pytest.mark.unit
def test_competing_nonce_consumption_uses_the_compare_and_swap_query_shape() -> None:
    """The real concurrency property is covered by the ephemeral-PG test."""
    connection = _FakeConnection(
        [_row(EnumFirstEffectLedgerState.PREFLIGHT_CONSUMED, 1), None]
    )
    adapter = _adapter(connection)

    async def _race() -> list[object]:
        return list(
            await asyncio.gather(
                adapter.consume_preflight(
                    authorization_digest=_AUTH, expected_version=0
                ),
                adapter.consume_preflight(
                    authorization_digest=_AUTH, expected_version=0
                ),
            )
        )

    results = asyncio.run(_race())
    assert sum(result is not None for result in results) == 1
    assert len(connection.calls) == 2
    for sql, values in connection.calls:
        assert "AND version = $5" in sql
        assert "AND state = ANY($6::text[])" in sql
        assert values[4] == 0
        assert values[5] == ["ISSUED"]


@pytest.mark.unit
def test_publishing_start_is_an_observation_not_publish_permission() -> None:
    connection = _FakeConnection([_row(EnumFirstEffectLedgerState.PUBLISHING, 2)])
    first_process = _adapter(connection)
    published = asyncio.run(
        first_process.record_publishing_started(
            authorization_digest=_AUTH, expected_version=1
        )
    )
    assert published is not None

    restart_connection = _FakeConnection([None])
    restarted_process = _adapter(restart_connection)
    assert (
        asyncio.run(
            restarted_process.record_publishing_started(
                authorization_digest=_AUTH, expected_version=2
            )
        )
        is None
    )
    sql, values = restart_connection.calls[0]
    assert values[5] == ["PREFLIGHT_CONSUMED"]
    assert "PUBLISHING" not in values[5]
    assert "version = version + 1" in sql


@pytest.mark.unit
def test_illegal_transition_fails_closed_at_the_cas_boundary() -> None:
    connection = _FakeConnection([None])
    adapter = _adapter(connection)

    assert (
        asyncio.run(
            adapter.record_publishing_started(
                authorization_digest=_AUTH, expected_version=0
            )
        )
        is None
    )
    sql, values = connection.calls[0]
    assert values[5] == ["PREFLIGHT_CONSUMED"]
    assert "AND state = ANY($6::text[])" in sql


@pytest.mark.unit
def test_published_unknown_only_recovers_through_terminal_observation() -> None:
    connection = _FakeConnection(
        [
            _row(EnumFirstEffectLedgerState.PUBLISHED_UNKNOWN, 3, evidence=_EVIDENCE),
            _row(EnumFirstEffectLedgerState.TERMINAL_OBSERVED, 4, evidence=_EVIDENCE),
        ]
    )
    adapter = _adapter(connection)

    observed = asyncio.run(
        adapter.observe_terminal(
            authorization_digest=_AUTH,
            expected_version=3,
            publish_evidence_hash=_EVIDENCE,
            terminal_receipt_hash=_RECEIPT,
        )
    )
    assert observed is not None
    assert observed.state is EnumFirstEffectLedgerState.TERMINAL_OBSERVED
    sql, values = connection.calls[1]
    assert values[5] == ["PUBLISHING", "PUBLISHED_UNKNOWN"]
    assert "terminal_receipt_hash" in sql


@pytest.mark.unit
def test_terminal_retry_reads_the_committed_record_after_a_lost_response() -> None:
    connection = _FakeConnection(
        [
            _row(EnumFirstEffectLedgerState.TERMINAL_OBSERVED, 4, evidence=_EVIDENCE),
            None,
            _row(EnumFirstEffectLedgerState.TERMINAL_OBSERVED, 4, evidence=_EVIDENCE),
        ]
    )
    adapter = _adapter(connection)

    recovered = asyncio.run(
        adapter.observe_terminal(
            authorization_digest=_AUTH,
            expected_version=3,
            publish_evidence_hash=_EVIDENCE,
            terminal_receipt_hash=_RECEIPT,
        )
    )

    assert recovered is not None
    assert recovered.state is EnumFirstEffectLedgerState.TERMINAL_OBSERVED
    assert recovered.version == 4
    assert "SELECT" in connection.calls[2][0]


@pytest.mark.unit
def test_terminal_retry_with_different_evidence_is_a_conflict() -> None:
    committed = _row(
        EnumFirstEffectLedgerState.TERMINAL_OBSERVED, 4, evidence=_EVIDENCE
    )
    committed["terminal_receipt_hash"] = "a" * 64
    adapter = _adapter(
        _FakeConnection(
            [
                _row(
                    EnumFirstEffectLedgerState.TERMINAL_OBSERVED, 4, evidence=_EVIDENCE
                ),
                None,
                committed,
            ]
        )
    )

    with pytest.raises(FirstEffectLedgerConflictError, match="committed evidence"):
        asyncio.run(
            adapter.observe_terminal(
                authorization_digest=_AUTH,
                expected_version=3,
                publish_evidence_hash=_EVIDENCE,
                terminal_receipt_hash=_RECEIPT,
            )
        )


@pytest.mark.unit
def test_published_unknown_different_publish_evidence_conflicts_before_update() -> None:
    connection = _FakeConnection(
        [_row(EnumFirstEffectLedgerState.PUBLISHED_UNKNOWN, 3, evidence=_EVIDENCE)]
    )
    adapter = _adapter(connection)

    with pytest.raises(FirstEffectLedgerConflictError, match="publish evidence"):
        asyncio.run(
            adapter.observe_terminal(
                authorization_digest=_AUTH,
                expected_version=3,
                publish_evidence_hash="0" * 64,
                terminal_receipt_hash=_RECEIPT,
            )
        )

    assert len(connection.calls) == 1
    assert "SELECT" in connection.calls[0][0]


@pytest.mark.unit
def test_exact_immutable_evidence_trigger_maps_to_ledger_conflict() -> None:
    connection = _FakeConnection(
        [
            _row(EnumFirstEffectLedgerState.PUBLISHED_UNKNOWN, 3, evidence=_EVIDENCE),
            asyncpg.RaiseError(
                "first-effect authorization evidence and transition timestamps are immutable"
            ),
        ]
    )
    adapter = _adapter(connection)

    with pytest.raises(FirstEffectLedgerConflictError, match="publish evidence"):
        asyncio.run(
            adapter.observe_terminal(
                authorization_digest=_AUTH,
                expected_version=3,
                publish_evidence_hash=_EVIDENCE,
                terminal_receipt_hash=_RECEIPT,
            )
        )


@pytest.mark.unit
def test_other_postgres_error_is_not_mapped_to_ledger_conflict() -> None:
    connection = _FakeConnection(
        [
            _row(EnumFirstEffectLedgerState.PUBLISHED_UNKNOWN, 3, evidence=_EVIDENCE),
            asyncpg.DataError(
                "first-effect authorization evidence and transition timestamps are immutable"
            ),
        ]
    )
    adapter = _adapter(connection)

    with pytest.raises(RepositoryExecutionError, match="observe_terminal failed"):
        asyncio.run(
            adapter.observe_terminal(
                authorization_digest=_AUTH,
                expected_version=3,
                publish_evidence_hash=_EVIDENCE,
                terminal_receipt_hash=_RECEIPT,
            )
        )


@pytest.mark.unit
def test_repeated_block_returns_the_committed_block() -> None:
    connection = _FakeConnection([None, _row(EnumFirstEffectLedgerState.BLOCKED, 4)])
    adapter = _adapter(connection)

    blocked = asyncio.run(adapter.block(authorization_digest=_AUTH, expected_version=3))

    assert blocked is not None
    assert blocked.state is EnumFirstEffectLedgerState.BLOCKED
    assert "SELECT" in connection.calls[1][0]
