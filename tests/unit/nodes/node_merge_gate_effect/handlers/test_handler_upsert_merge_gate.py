# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerUpsertMergeGate.

Tests validate:
  - Successful upsert (insert) returns ModelBackendResult success=True
  - Idempotent upsert (update) returns success=True
  - QUARANTINE decision triggers Linear ticket creation
  - PASS/WARN decisions do NOT trigger Linear ticket
  - Linear not configured: quarantine skipped with warning, upsert still succeeds
  - Linear API failure: logged but upsert still succeeds
  - TimeoutError / InfraConnectionError / InfraAuthenticationError handling

Related Tickets:
    - OMN-3140: NodeMergeGateEffect + migration
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest

from omnibase_infra.enums import EnumPostgresErrorCode
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
)
from omnibase_infra.nodes.node_merge_gate_effect.handlers.handler_upsert_merge_gate import (
    HandlerUpsertMergeGate,
)
from omnibase_infra.nodes.node_merge_gate_effect.models.model_merge_gate_result import (
    ModelMergeGateResult,
    ModelMergeGateViolation,
)

# =============================================================================
# Mock pool factory
# =============================================================================


def create_mock_pool_with_conn(conn: AsyncMock) -> MagicMock:
    """Create a mock pool returning a specific connection."""
    pool = MagicMock()
    mock_acquire = AsyncMock()
    mock_acquire.__aenter__ = AsyncMock(return_value=conn)
    mock_acquire.__aexit__ = AsyncMock(return_value=None)
    pool.acquire = MagicMock(return_value=mock_acquire)
    return pool


def make_gate_payload(
    *,
    decision: str = "PASS",
    tier: str = "tier-a",
    violations: list[ModelMergeGateViolation] | None = None,
) -> ModelMergeGateResult:
    """Create a test ModelMergeGateResult."""
    return ModelMergeGateResult(
        gate_id=uuid4(),
        pr_ref="OmniNode-ai/omnibase_infra#42",
        head_sha="abc123def456",
        base_sha="000111222333",
        decision=decision,  # type: ignore[arg-type]
        tier=tier,  # type: ignore[arg-type]
        violations=violations or [],
        run_id=uuid4(),
        correlation_id=uuid4(),
        run_fingerprint="fp-test-12345",
        decided_at=datetime.now(tz=UTC),
    )


# =============================================================================
# Tests: successful upsert
# =============================================================================


class TestHandlerUpsertMergeGateSuccess:
    """Test successful merge gate decision upserts."""

    @pytest.mark.asyncio
    async def test_insert_returns_success(self) -> None:
        """Successful insert returns ModelBackendResult with success=True."""
        conn = AsyncMock()
        # was_insert = True (new row)
        conn.fetchrow = AsyncMock(return_value={"was_insert": True})
        pool = create_mock_pool_with_conn(conn)

        handler = HandlerUpsertMergeGate(pool)
        payload = make_gate_payload(decision="PASS")
        correlation_id = uuid4()

        result = await handler.handle(payload, correlation_id)

        assert result.success is True
        assert result.backend_id == "postgres"
        # effective_cid = payload.correlation_id or correlation_id
        # payload has its own correlation_id, so that takes precedence
        assert result.correlation_id == payload.correlation_id
        assert result.error is None
        assert result.error_code is None
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_insert_uses_handler_correlation_id_when_payload_has_none(
        self,
    ) -> None:
        """When payload.correlation_id is None, handler's correlation_id is used."""
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"was_insert": True})
        pool = create_mock_pool_with_conn(conn)

        handler = HandlerUpsertMergeGate(pool)
        payload = ModelMergeGateResult(
            gate_id=uuid4(),
            pr_ref="test/repo#1",
            head_sha="abc123",
            base_sha="def456",
            decision="PASS",
            tier="tier-a",
            decided_at=datetime.now(tz=UTC),
            correlation_id=None,
        )
        handler_cid = uuid4()

        result = await handler.handle(payload, handler_cid)

        assert result.success is True
        assert result.correlation_id == handler_cid

    @pytest.mark.asyncio
    async def test_idempotent_update_returns_success(self) -> None:
        """ON CONFLICT DO UPDATE (was_insert=False) still returns success=True."""
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"was_insert": False})
        pool = create_mock_pool_with_conn(conn)

        handler = HandlerUpsertMergeGate(pool)
        payload = make_gate_payload(decision="WARN")
        result = await handler.handle(payload, uuid4())

        assert result.success is True
        assert result.backend_id == "postgres"


# =============================================================================
# Tests: QUARANTINE -> Linear ticket
# =============================================================================


class TestHandlerUpsertMergeGateQuarantine:
    """Test QUARANTINE decision -> Linear ticket creation."""

    @pytest.mark.asyncio
    async def test_quarantine_opens_linear_ticket(self) -> None:
        """QUARANTINE decision calls Linear GraphQL to create a ticket."""
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"was_insert": True})
        pool = create_mock_pool_with_conn(conn)

        violations = [
            ModelMergeGateViolation(
                rule_code="RRH-1001",
                severity="FAIL",
                message="Working tree is dirty",
            ),
        ]
        payload = make_gate_payload(decision="QUARANTINE", violations=violations)

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(
            return_value={
                "data": {
                    "issueCreate": {
                        "success": True,
                        "issue": {
                            "id": "issue-123",
                            "identifier": "OMN-9999",
                            "url": "https://linear.app/omninode/issue/OMN-9999",
                        },
                    }
                }
            }
        )

        with (
            patch.dict(
                "os.environ",
                {"LINEAR_API_KEY": "test-key", "LINEAR_TEAM_ID": "team-123"},
            ),
            patch(
                "omnibase_infra.nodes.node_merge_gate_effect.handlers."
                "handler_upsert_merge_gate.httpx.AsyncClient"
            ) as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            handler = HandlerUpsertMergeGate(pool)
            result = await handler.handle(payload, uuid4())

        assert result.success is True
        # Verify Linear API was called
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert "QUARANTINE" in call_kwargs[1]["json"]["variables"]["title"]

    @pytest.mark.asyncio
    async def test_pass_does_not_open_linear_ticket(self) -> None:
        """PASS decision does NOT call Linear."""
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"was_insert": True})
        pool = create_mock_pool_with_conn(conn)

        payload = make_gate_payload(decision="PASS")

        with patch(
            "omnibase_infra.nodes.node_merge_gate_effect.handlers."
            "handler_upsert_merge_gate.httpx.AsyncClient"
        ) as mock_client_cls:
            handler = HandlerUpsertMergeGate(pool)
            result = await handler.handle(payload, uuid4())

        assert result.success is True
        mock_client_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_quarantine_update_does_not_open_linear_ticket(self) -> None:
        """QUARANTINE re-evaluation (was_insert=False) skips Linear ticket (idempotent)."""
        conn = AsyncMock()
        # was_insert = False -> this is an update, not a fresh insert
        conn.fetchrow = AsyncMock(return_value={"was_insert": False})
        pool = create_mock_pool_with_conn(conn)

        payload = make_gate_payload(decision="QUARANTINE")

        with patch(
            "omnibase_infra.nodes.node_merge_gate_effect.handlers."
            "handler_upsert_merge_gate.httpx.AsyncClient"
        ) as mock_client_cls:
            handler = HandlerUpsertMergeGate(pool)
            result = await handler.handle(payload, uuid4())

        assert result.success is True
        # Linear should NOT have been called on update
        mock_client_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_quarantine_without_linear_config_still_succeeds(self) -> None:
        """QUARANTINE without LINEAR_API_KEY skips ticket, upsert still succeeds."""
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"was_insert": True})
        pool = create_mock_pool_with_conn(conn)

        payload = make_gate_payload(decision="QUARANTINE")

        with patch.dict(
            "os.environ",
            {"LINEAR_API_KEY": "", "LINEAR_TEAM_ID": ""},
            clear=False,
        ):
            handler = HandlerUpsertMergeGate(pool)
            result = await handler.handle(payload, uuid4())

        assert result.success is True

    @pytest.mark.asyncio
    async def test_quarantine_linear_failure_still_succeeds(self) -> None:
        """Linear API failure during QUARANTINE does not fail the upsert."""
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"was_insert": True})
        pool = create_mock_pool_with_conn(conn)

        payload = make_gate_payload(decision="QUARANTINE")

        with (
            patch.dict(
                "os.environ",
                {"LINEAR_API_KEY": "test-key", "LINEAR_TEAM_ID": "team-123"},
            ),
            patch(
                "omnibase_infra.nodes.node_merge_gate_effect.handlers."
                "handler_upsert_merge_gate.httpx.AsyncClient"
            ) as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_client_cls.return_value = mock_client

            handler = HandlerUpsertMergeGate(pool)
            result = await handler.handle(payload, uuid4())

        # Upsert succeeded even though Linear failed
        assert result.success is True


# =============================================================================
# Tests: error handling
# =============================================================================


class TestHandlerUpsertMergeGateErrors:
    """Test error handling for HandlerUpsertMergeGate."""

    @pytest.mark.asyncio
    async def test_timeout_error_returns_timeout_code(self) -> None:
        """TimeoutError returns success=False with TIMEOUT_ERROR code."""
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(side_effect=TimeoutError("timed out"))
        pool = create_mock_pool_with_conn(conn)

        handler = HandlerUpsertMergeGate(pool)
        result = await handler.handle(make_gate_payload(), uuid4())

        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.TIMEOUT_ERROR
        assert result.backend_id == "postgres"

    @pytest.mark.asyncio
    async def test_connection_error_returns_connection_code(self) -> None:
        """InfraConnectionError returns success=False with CONNECTION_ERROR code."""
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(side_effect=InfraConnectionError("refused"))
        pool = create_mock_pool_with_conn(conn)

        handler = HandlerUpsertMergeGate(pool)
        result = await handler.handle(make_gate_payload(), uuid4())

        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.CONNECTION_ERROR
        assert result.backend_id == "postgres"

    @pytest.mark.asyncio
    async def test_auth_error_returns_auth_code(self) -> None:
        """InfraAuthenticationError returns success=False with AUTH_ERROR code."""
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(side_effect=InfraAuthenticationError("bad creds"))
        pool = create_mock_pool_with_conn(conn)

        handler = HandlerUpsertMergeGate(pool)
        result = await handler.handle(make_gate_payload(), uuid4())

        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.AUTH_ERROR
        assert result.backend_id == "postgres"

    @pytest.mark.asyncio
    async def test_generic_exception_returns_unknown_code(self) -> None:
        """Unexpected exception returns success=False with UNKNOWN_ERROR code."""
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(side_effect=RuntimeError("boom"))
        pool = create_mock_pool_with_conn(conn)

        handler = HandlerUpsertMergeGate(pool)
        result = await handler.handle(make_gate_payload(), uuid4())

        assert result.success is False
        assert result.error_code == EnumPostgresErrorCode.UNKNOWN_ERROR
        assert result.backend_id == "postgres"


# =============================================================================
# Tests: model validation
# =============================================================================


class TestModelMergeGateResult:
    """Test ModelMergeGateResult payload validation."""

    def test_valid_payload_construction(self) -> None:
        """Valid payload fields are accepted."""
        payload = make_gate_payload(decision="QUARANTINE", tier="tier-b")
        assert payload.decision == "QUARANTINE"
        assert payload.tier == "tier-b"

    def test_violations_default_to_empty(self) -> None:
        """violations defaults to an empty list."""
        payload = ModelMergeGateResult(
            gate_id=uuid4(),
            pr_ref="test/repo#1",
            head_sha="abc",
            base_sha="def",
            decision="PASS",
            tier="tier-a",
            decided_at=datetime.now(tz=UTC),
        )
        assert payload.violations == []

    def test_frozen_model_is_immutable(self) -> None:
        """Model is frozen -- attribute assignment raises."""
        payload = make_gate_payload()
        with pytest.raises(Exception):  # ValidationError from Pydantic
            payload.decision = "QUARANTINE"  # type: ignore[misc]


# =============================================================================
# Tests: tier-a-contract-gate profile
# =============================================================================


class TestTierAContractGateProfile:
    """Test the tier-a-contract-gate RRH profile."""

    def test_profile_exists_in_registry(self) -> None:
        """tier-a-contract-gate profile is registered."""
        from omnibase_infra.nodes.node_rrh_validate_compute.profiles import (
            PROFILES,
            get_profile,
        )

        assert "tier-a-contract-gate" in PROFILES
        profile = get_profile("tier-a-contract-gate")
        assert profile.name == "tier-a-contract-gate"

    def test_profile_has_correct_rule_count(self) -> None:
        """Profile has all 13 rules defined."""
        from omnibase_infra.nodes.node_rrh_validate_compute.profiles import (
            get_profile,
        )

        profile = get_profile("tier-a-contract-gate")
        assert len(profile.rules) == 13


__all__: list[str] = [
    "TestHandlerUpsertMergeGateSuccess",
    "TestHandlerUpsertMergeGateQuarantine",
    "TestHandlerUpsertMergeGateErrors",
    "TestModelMergeGateResult",
    "TestTierAContractGateProfile",
]
