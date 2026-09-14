# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Connection-free SQL and redaction checks for the injected claim adapter."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime

import asyncpg
import pytest

from omnibase_infra.runtime.action_authorization_claim import (
    EnumActionAuthorizationClaimOutcome,
    ModelActionAuthorizationClaimRequest,
    PostgresActionAuthorizationClaim,
)


def _request() -> ModelActionAuthorizationClaimRequest:
    return ModelActionAuthorizationClaimRequest(
        authorization_id="action-auth-12345678-1234-1234-1234-123456789abc",
        ticket_id="OMN-17462",
        contract_path="contracts/OMN-17462.yaml",
        contract_commit_sha="a" * 40,
        contract_sha256="sha256:" + "b" * 64,
        action_id="postgres-push-lane-bootstrap",
        source_sha="c" * 40,
        artifact_sha256="sha256:" + "d" * 64,
        target_database="rsd_push_lanes",
        target_schema="push_lanes",
        target_service="rsd_push_lane_broker",
        target_principal="rsd_push_lane_broker",
        execute_enabled=False,
        issuer="operator-governance",
        nonce="e" * 64,
        issued_at=datetime(2030, 1, 1, 10, tzinfo=UTC),
        expires_at=datetime(2030, 1, 1, 11, tzinfo=UTC),
        one_time_use=True,
        reason="bounded execute-disabled bootstrap verification",
    )


class _Acquire(AbstractAsyncContextManager):
    def __init__(self, connection: _Connection) -> None:
        self._connection = connection

    async def __aenter__(self) -> _Connection:
        return self._connection

    async def __aexit__(self, *args: object) -> bool:
        return False


class _Connection:
    def __init__(self, results: Iterator[object]) -> None:
        self._results = results
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def fetchrow(self, sql: str, *values: object) -> object:
        self.calls.append((sql, values))
        result = next(self._results)
        if isinstance(result, BaseException):
            raise result
        return result


class _Pool:
    def __init__(self, connection: _Connection) -> None:
        self._connection = connection
        self.closed = False

    def acquire(self) -> _Acquire:
        return _Acquire(self._connection)

    async def close(self) -> None:
        self.closed = True


def _adapter(connection: _Connection) -> PostgresActionAuthorizationClaim:
    pool = _Pool(connection)

    async def pool_factory() -> _Pool:
        return pool

    return PostgresActionAuthorizationClaim(pool_factory=pool_factory)  # type: ignore[arg-type]


@pytest.mark.unit
def test_claim_maps_a_database_error_to_closed_redacted_error() -> None:
    connection = _Connection(iter([asyncpg.DataError("database detail")]))

    result = asyncio.run(_adapter(connection).claim(_request()))

    assert result.outcome is EnumActionAuthorizationClaimOutcome.ERROR
    assert result.state is None
    assert result.redacted_receipt_digest is None


@pytest.mark.unit
def test_claim_uses_the_dedicated_atomic_function() -> None:
    request = _request()
    connection = _Connection(
        iter(
            [
                {
                    "outcome": "CLAIMED",
                    "state": "CLAIMED",
                    "version": 1,
                    "redacted_receipt_digest": request.redacted_receipt_digest,
                }
            ]
        )
    )

    result = asyncio.run(_adapter(connection).claim(request))

    assert result.outcome is EnumActionAuthorizationClaimOutcome.CLAIMED
    sql, values = connection.calls[0]
    assert "claim_action_authorization" in sql
    assert "FROM action_authorization_claim" in sql
    assert "register_action_authorization" not in sql
    assert "first_effect" not in sql
    assert len(values) == 21
    assert values[14] == request.nonce_digest
    assert request.nonce not in values
    assert values[19] == request.request_digest
    assert values[20] == request.redacted_receipt_digest
