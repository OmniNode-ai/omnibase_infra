# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the state_io async CAS adapter (OMN-14208).

Covers CAS conflict + retry (via the shared ``retry_on_optimistic_conflict``
helper), seed idempotence, the ``recover_stale_rows`` TTL boundary, and the
ContextVar "set-for-unknown-correlation_id" semantics (G3: an empty/missing
row is still a *set* ContextVar value, distinguishing "state_io active, no
row yet" from "state_io not active on this dispatch").

The asyncpg pool/connection are faked (no real Postgres) — these tests
exercise the adapter's SQL-shaping and control flow, not real database I/O.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import asyncpg
import pytest

from omnibase_infra.errors.repository import (
    RepositoryContractError,
    RepositoryExecutionError,
)
from omnibase_infra.runtime.state_io.state_store_adapter import (
    CONTEXTVAR_STATE_IO_ROWS,
    StateStoreAdapter,
)


class _FakeAcquireCtx:
    def __init__(self, conn: _FakeConnection) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeConnection:
        return self._conn

    async def __aexit__(self, *exc_info: object) -> bool:
        return False


class _FakeConnection:
    """Minimal asyncpg.Connection stand-in: queue of canned responses."""

    def __init__(
        self,
        *,
        fetchrow_results: list[object] | None = None,
        execute_results: list[str] | None = None,
    ) -> None:
        self._fetchrow_results = list(fetchrow_results or [])
        self._execute_results = list(execute_results or [])
        self.fetchrow_calls: list[tuple[str, tuple[object, ...]]] = []
        self.execute_calls: list[tuple[str, tuple[object, ...]]] = []

    async def fetchrow(self, sql: str, *params: object) -> object:
        self.fetchrow_calls.append((sql, params))
        if not self._fetchrow_results:
            return None
        return self._fetchrow_results.pop(0)

    async def execute(self, sql: str, *params: object) -> str:
        self.execute_calls.append((sql, params))
        if not self._execute_results:
            return "UPDATE 0"
        return self._execute_results.pop(0)


class _FakePool:
    def __init__(self, conn: _FakeConnection) -> None:
        self._conn = conn

    def acquire(self) -> _FakeAcquireCtx:
        return _FakeAcquireCtx(self._conn)


def _adapter_with_fake_pool(conn: _FakeConnection) -> StateStoreAdapter:
    adapter = StateStoreAdapter(
        "postgresql://user:pass@host:5432/omnibase_infra",
        table="delegation_workflow_state",
    )
    adapter._pool = _FakePool(conn)
    return adapter


# ---------------------------------------------------------------------------
# Tests: __init__ table name validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_init_rejects_invalid_table_name() -> None:
    with pytest.raises(RepositoryContractError, match="Invalid state_io table name"):
        StateStoreAdapter("postgresql://x", table="bad; drop table users;--")


@pytest.mark.unit
def test_init_accepts_valid_table_name() -> None:
    adapter = StateStoreAdapter("postgresql://x", table="delegation_workflow_state")
    assert adapter is not None


# ---------------------------------------------------------------------------
# Tests: load
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_load_returns_none_when_no_row() -> None:
    conn = _FakeConnection(fetchrow_results=[None])
    adapter = _adapter_with_fake_pool(conn)
    result = asyncio.run(adapter.load("cid-1"))
    assert result is None


@pytest.mark.unit
def test_load_returns_payload_and_version() -> None:
    conn = _FakeConnection(
        fetchrow_results=[{"payload_json": '{"state": "RECEIVED"}', "version": 3}]
    )
    adapter = _adapter_with_fake_pool(conn)
    result = asyncio.run(adapter.load("cid-1"))
    assert result == ('{"state": "RECEIVED"}', 3)


@pytest.mark.unit
def test_load_wraps_postgres_error() -> None:
    class _RaisingConnection(_FakeConnection):
        async def fetchrow(self, sql: str, *params: object) -> object:
            raise asyncpg.PostgresError("connection reset")

    adapter = _adapter_with_fake_pool(_RaisingConnection())
    with pytest.raises(RepositoryExecutionError, match="state_io load failed"):
        asyncio.run(adapter.load("cid-1"))


# ---------------------------------------------------------------------------
# Tests: seed idempotence
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_seed_returns_true_on_first_insert() -> None:
    conn = _FakeConnection(fetchrow_results=[{"correlation_id": "cid-1"}])
    adapter = _adapter_with_fake_pool(conn)
    won = asyncio.run(
        adapter.seed(
            "cid-1",
            tenant_id="acme",
            state="RECEIVED",
            in_flight=True,
            payload_json="{}",
        )
    )
    assert won is True


@pytest.mark.unit
def test_seed_returns_false_on_conflict_idempotent() -> None:
    """A second seed() for the same correlation_id (ON CONFLICT DO NOTHING) must
    report the loss (no row returned) rather than silently overwrite."""
    conn = _FakeConnection(fetchrow_results=[None])
    adapter = _adapter_with_fake_pool(conn)
    won = asyncio.run(
        adapter.seed(
            "cid-1",
            tenant_id="acme",
            state="RECEIVED",
            in_flight=True,
            payload_json="{}",
        )
    )
    assert won is False


# ---------------------------------------------------------------------------
# Tests: CAS conflict + retry (task: "stale version -> row_count 0 -> retry
# re-invokes closure")
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cas_update_returns_zero_row_count_on_conflict() -> None:
    conn = _FakeConnection(execute_results=["UPDATE 0"])
    adapter = _adapter_with_fake_pool(conn)
    row_count = asyncio.run(
        adapter.cas_update(
            "cid-1",
            tenant_id="acme",
            state="COMPLETED",
            in_flight=False,
            payload_json="{}",
            expected_version=5,
        )
    )
    assert row_count == 0


@pytest.mark.unit
def test_cas_update_returns_one_on_success() -> None:
    conn = _FakeConnection(execute_results=["UPDATE 1"])
    adapter = _adapter_with_fake_pool(conn)
    row_count = asyncio.run(
        adapter.cas_update(
            "cid-1",
            tenant_id="acme",
            state="COMPLETED",
            in_flight=False,
            payload_json="{}",
            expected_version=5,
        )
    )
    assert row_count == 1
    # expected_version is bound as the final positional param ($6).
    _sql, params = conn.execute_calls[0]
    assert params[-1] == 5


@pytest.mark.unit
def test_cas_update_conflict_retries_and_re_invokes_closure() -> None:
    """A stale-version CAS (row_count 0) must retry — reloading and re-running
    the caller's closure — until a fresh version's CAS succeeds."""
    from omnibase_infra.utils.util_retry_optimistic import retry_on_optimistic_conflict

    attempts = 0

    async def _load_and_cas() -> int:
        nonlocal attempts
        attempts += 1
        # First attempt loses the race (stale version); second succeeds.
        return 0 if attempts == 1 else 1

    result = asyncio.run(
        retry_on_optimistic_conflict(
            _load_and_cas,
            check_conflict=lambda row_count: row_count == 0,
            max_retries=3,
            initial_backoff=0.0,
        )
    )
    assert result == 1
    assert attempts == 2, "closure must be re-invoked exactly once on conflict"


# ---------------------------------------------------------------------------
# Tests: recover_stale_rows TTL boundary
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_recover_stale_rows_uses_default_ttl() -> None:
    conn = _FakeConnection(execute_results=["UPDATE 0"])
    adapter = _adapter_with_fake_pool(conn)
    with patch.dict("os.environ", {}, clear=False):
        import os

        os.environ.pop("DELEGATION_STALE_TTL_SECONDS", None)
        recovered = asyncio.run(adapter.recover_stale_rows())
    assert recovered == 0
    _sql, params = conn.execute_calls[0]
    assert params[0] == 900


@pytest.mark.unit
def test_recover_stale_rows_uses_env_override() -> None:
    conn = _FakeConnection(execute_results=["UPDATE 0"])
    adapter = _adapter_with_fake_pool(conn)
    with patch.dict("os.environ", {"DELEGATION_STALE_TTL_SECONDS": "60"}):
        asyncio.run(adapter.recover_stale_rows())
    _sql, params = conn.execute_calls[0]
    assert params[0] == 60


@pytest.mark.unit
def test_recover_stale_rows_explicit_arg_wins_over_env() -> None:
    conn = _FakeConnection(execute_results=["UPDATE 0"])
    adapter = _adapter_with_fake_pool(conn)
    with patch.dict("os.environ", {"DELEGATION_STALE_TTL_SECONDS": "60"}):
        asyncio.run(adapter.recover_stale_rows(ttl_seconds=30))
    _sql, params = conn.execute_calls[0]
    assert params[0] == 30


@pytest.mark.unit
def test_recover_stale_rows_returns_recovered_count() -> None:
    conn = _FakeConnection(execute_results=["UPDATE 3"])
    adapter = _adapter_with_fake_pool(conn)
    recovered = asyncio.run(adapter.recover_stale_rows(ttl_seconds=900))
    assert recovered == 3


# ---------------------------------------------------------------------------
# Tests: ContextVar set-for-unknown-correlation_id (G3)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_contextvar_defaults_unset() -> None:
    """Outside any state_io-wired dispatch, the ContextVar stays at its unset
    default (None) — distinguishing "state_io inactive" from "state_io active,
    no row yet"."""
    assert CONTEXTVAR_STATE_IO_ROWS.get() is None


@pytest.mark.unit
def test_stateful_callback_sets_contextvar_for_missing_row() -> None:
    """A dispatch for a correlation_id with NO existing row must still SET the
    ContextVar (to a dict mapping cid -> (None, 0)) rather than leaving it at
    its unset default — an empty/missing row is distinct from "not active"."""
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        _make_stateful_dispatch_callback,
    )

    observed: list[Any] = []

    class _ObservingHandler:
        async def handle(self, envelope: object) -> None:
            observed.append(CONTEXTVAR_STATE_IO_ROWS.get())

    class _FakeAdapter:
        async def load(self, cid: str) -> None:
            return None

    with (
        patch.dict("os.environ", {"OMNIBASE_INFRA_DB_URL": "postgresql://x"}),
        patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=object,
        ),
        patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring.StateStoreAdapter",
            return_value=_FakeAdapter(),
        ),
    ):
        callback = _make_stateful_dispatch_callback(
            _ObservingHandler(),
            None,
            {
                "database": "omnibase_infra",
                "table": "delegation_workflow_state",
                "key": "correlation_id",
                "codec": {"module": "tests", "name": "_ObservingHandler"},
            },
        )

        class _Envelope:
            payload = {"correlation_id": "11111111-1111-1111-1111-111111111111"}

        asyncio.run(callback(_Envelope()))  # type: ignore[arg-type]

    assert len(observed) == 1
    cid = "11111111-1111-1111-1111-111111111111"
    assert observed[0] == {cid: (None, 0)}

    # After the dispatch completes, the ContextVar is reset to its prior
    # (unset) value — no leakage across dispatches.
    assert CONTEXTVAR_STATE_IO_ROWS.get() is None
