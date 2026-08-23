# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerSavingsCorrelation.

Ports the assertions from the deleted legacy
``services/observability/savings_estimation`` test suite (ServiceSavingsEstimator
/ConfigSavingsEstimation) onto the Postgres-projection-backed correlator.

Ticket: OMN-16293
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

pytestmark = pytest.mark.unit

from omnibase_infra.nodes.node_savings_estimation_compute.handlers.handler_savings_correlation import (
    EnumCatchSeverity,
    HandlerSavingsCorrelation,
    InjectionRow,
    LlmCallRow,
    ValidatorCatchRow,
    _build_effectiveness_entries,
    _classify_severity,
    _compute_validator_catch_savings,
    _resolve_counterfactual,
)
from omnibase_infra.nodes.node_savings_estimation_compute.models.model_savings_correlation_batch_command import (
    ModelSavingsCorrelationBatchCommand,
)


@pytest.fixture
def mock_pool() -> MagicMock:
    """Mock asyncpg.Pool with a connection supporting acquire()."""
    pool = MagicMock()
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=None)
    conn.execute = AsyncMock(return_value="INSERT 0 1")

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire = MagicMock(return_value=ctx)
    pool._test_conn = conn
    return pool


@pytest.fixture
def mock_publisher() -> AsyncMock:
    return AsyncMock(return_value=True)


# ---------------------------------------------------------------------------
# Pure math helpers — formulas ported unchanged from the legacy consumer.
# ---------------------------------------------------------------------------


class TestPureHelpers:
    def test_classify_severity_maps_known_aliases(self) -> None:
        assert _classify_severity("error") == EnumCatchSeverity.CRITICAL
        assert _classify_severity("WARNING") == EnumCatchSeverity.MAJOR
        assert _classify_severity("something-unknown") == EnumCatchSeverity.MINOR

    def test_resolve_counterfactual_never_returns_none(self) -> None:
        assert _resolve_counterfactual("claude-sonnet-4") == "claude-opus-4-6"
        assert _resolve_counterfactual("claude-opus-4-6") == "claude-opus-4-6"
        # Unknown model still resolves to *something* non-empty — the
        # downstream omnimarket consumer DLQs on an empty
        # model_cloud_baseline (OMN-14533), so this must never be falsy.
        resolved = _resolve_counterfactual("some-unknown-model")
        assert resolved == "some-unknown-model"
        assert resolved

    def test_compute_validator_catch_savings_empty(self) -> None:
        assert _compute_validator_catch_savings([]) == (0.0, 0, 0.0)

    def test_compute_validator_catch_savings_diminishing_returns(self) -> None:
        usd, tokens, confidence = _compute_validator_catch_savings(
            [EnumCatchSeverity.CRITICAL, EnumCatchSeverity.CRITICAL]
        )
        # catch 1 = full weight (0.50, 2000), catch 2 = 1/1.3 weight
        assert usd == pytest.approx(0.50 + 0.50 / 1.3, rel=1e-6)
        assert tokens == 2000 + int(2000 / 1.3)
        assert confidence == pytest.approx(0.7)

    def test_build_effectiveness_entries_from_injection_only(self) -> None:
        entries = _build_effectiveness_entries(
            [InjectionRow(tokens_injected=120, patterns_count=2)],
            [],
            [],
            has_session_outcome=True,
        )
        assert len(entries) == 1
        assert entries[0].tokens_saved == 120
        assert entries[0].patterns_count == 2

    def test_build_effectiveness_entries_falls_back_to_llm_only(self) -> None:
        entries = _build_effectiveness_entries(
            [],
            [
                LlmCallRow(
                    model_id="claude-sonnet-4",
                    prompt_tokens=1000,
                    completion_tokens=200,
                )
            ],
            [],
            has_session_outcome=True,
        )
        assert len(entries) == 1
        assert entries[0].tokens_saved == 0
        assert entries[0].model_tier.value == "sonnet"

    def test_build_effectiveness_entries_validator_catch_only_requires_outcome(
        self,
    ) -> None:
        no_outcome = _build_effectiveness_entries(
            [],
            [],
            [ValidatorCatchRow(severity=EnumCatchSeverity.MAJOR)],
            has_session_outcome=False,
        )
        assert no_outcome == ()

        with_outcome = _build_effectiveness_entries(
            [],
            [],
            [ValidatorCatchRow(severity=EnumCatchSeverity.MAJOR)],
            has_session_outcome=True,
        )
        assert len(with_outcome) == 1

    def test_build_effectiveness_entries_empty_when_no_signal(self) -> None:
        assert _build_effectiveness_entries([], [], [], has_session_outcome=True) == ()


# ---------------------------------------------------------------------------
# Ingest — one INSERT per event, no buffering.
# ---------------------------------------------------------------------------


class TestIngest:
    @pytest.mark.asyncio
    async def test_ingest_injection_event_inserts_row(
        self, mock_pool: MagicMock
    ) -> None:
        handler = HandlerSavingsCorrelation(pool=mock_pool)
        await handler.ingest_injection_event(
            {"session_id": "s1", "tokens_injected": 100, "patterns_count": 3}
        )
        conn = mock_pool._test_conn
        # conn.execute is called twice: once by set_statement_timeout's
        # "SET LOCAL ..." and once for the INSERT — assert on the INSERT,
        # the last call.
        sql, *params = conn.execute.call_args.args
        assert "INSERT INTO omninode_internal.savings_injection_signals" in sql
        assert params == ["s1", 100, 3]

    @pytest.mark.asyncio
    async def test_ingest_injection_event_skips_empty_session(
        self, mock_pool: MagicMock
    ) -> None:
        handler = HandlerSavingsCorrelation(pool=mock_pool)
        await handler.ingest_injection_event({"session_id": "", "tokens_injected": 100})
        mock_pool._test_conn.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ingest_injection_event_skips_non_positive_tokens(
        self, mock_pool: MagicMock
    ) -> None:
        handler = HandlerSavingsCorrelation(pool=mock_pool)
        await handler.ingest_injection_event({"session_id": "s1", "tokens_injected": 0})
        mock_pool._test_conn.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ingest_validator_catch_event_derives_source_from_topic(
        self, mock_pool: MagicMock
    ) -> None:
        handler = HandlerSavingsCorrelation(pool=mock_pool)
        await handler.ingest_validator_catch_event(
            "onex.evt.omniclaude.pattern-enforcement.v1",
            {"session_id": "s1", "severity": "critical", "validator_type": "lint"},
        )
        conn = mock_pool._test_conn
        sql, *params = conn.execute.call_args.args
        assert "INSERT INTO omninode_internal.savings_validator_catch_signals" in sql
        assert params == ["s1", "critical", "lint", "pattern-enforcement"]

    @pytest.mark.asyncio
    async def test_ingest_validator_catch_event_skips_empty_session(
        self, mock_pool: MagicMock
    ) -> None:
        handler = HandlerSavingsCorrelation(pool=mock_pool)
        await handler.ingest_validator_catch_event(
            "onex.evt.omniclaude.validator-catch.v1", {"session_id": ""}
        )
        mock_pool._test_conn.execute.assert_not_awaited()


# ---------------------------------------------------------------------------
# Correlate — periodic batch step.
# ---------------------------------------------------------------------------


class TestRunCorrelationBatch:
    @pytest.mark.asyncio
    async def test_finalizes_and_publishes_non_dispatch_session(
        self, mock_pool: MagicMock, mock_publisher: AsyncMock
    ) -> None:
        """Mirrors the deleted non-dispatch-projection integration tests:
        an llm-call + injection + session-outcome session produces an
        estimate whose local/cloud/savings cost fields are consistent."""
        conn = mock_pool._test_conn

        async def fetch_side_effect(sql: str, *args: object) -> list[dict[str, object]]:
            if "candidate_sessions" in sql:
                return [{"session_id": "session-non-dispatch"}]
            if "savings_injection_signals" in sql:
                return [{"tokens_injected": 300, "patterns_count": 3}]
            if "savings_validator_catch_signals" in sql:
                return []
            if "llm_call_metrics" in sql:
                return [
                    {
                        "model_id": "claude-sonnet-4",
                        "prompt_tokens": 5000,
                        "completion_tokens": 1000,
                    }
                ]
            return []

        async def fetchrow_side_effect(
            sql: str, *args: object
        ) -> dict[str, object] | None:
            if "session_outcomes" in sql:
                return {"outcome": "success"}
            return None

        conn.fetch = AsyncMock(side_effect=fetch_side_effect)
        conn.fetchrow = AsyncMock(side_effect=fetchrow_side_effect)

        handler = HandlerSavingsCorrelation(pool=mock_pool, publisher=mock_publisher)
        output = await handler.run_correlation_batch(
            ModelSavingsCorrelationBatchCommand(correlation_id=uuid4())
        )

        assert output.sessions_finalized == 1
        assert output.errors == ()
        mock_publisher.assert_awaited_once()
        _, kwargs = mock_publisher.call_args
        payload = kwargs["payload"]
        assert payload["session_id"] == "session-non-dispatch"
        assert payload["model_local"] == "claude-sonnet-4"
        assert payload["model_cloud_baseline"] == "claude-opus-4-6"
        assert payload["local_cost_usd"] == payload["actual_cost_usd"]
        assert payload["cloud_cost_usd"] == pytest.approx(
            payload["actual_cost_usd"] + payload["estimated_total_savings_usd"]
        )
        assert payload["savings_usd"] == payload["estimated_total_savings_usd"]
        # counterfactual_model_id must never be None/empty — the downstream
        # omnimarket consumer DLQs the event otherwise (OMN-14533).
        assert payload["counterfactual_model_id"]

    @pytest.mark.asyncio
    async def test_no_ready_sessions_publishes_nothing(
        self, mock_pool: MagicMock, mock_publisher: AsyncMock
    ) -> None:
        handler = HandlerSavingsCorrelation(pool=mock_pool, publisher=mock_publisher)
        output = await handler.run_correlation_batch(
            ModelSavingsCorrelationBatchCommand(correlation_id=uuid4())
        )
        assert output.sessions_finalized == 0
        mock_publisher.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_one_bad_session_does_not_abort_the_tick(
        self, mock_pool: MagicMock, mock_publisher: AsyncMock
    ) -> None:
        conn = mock_pool._test_conn

        async def fetch_side_effect(sql: str, *args: object) -> list[dict[str, object]]:
            if "candidate_sessions" in sql:
                return [{"session_id": "broken-session"}]
            raise RuntimeError("boom")

        conn.fetch = AsyncMock(side_effect=fetch_side_effect)

        handler = HandlerSavingsCorrelation(pool=mock_pool, publisher=mock_publisher)
        output = await handler.run_correlation_batch(
            ModelSavingsCorrelationBatchCommand(correlation_id=uuid4())
        )
        assert output.sessions_finalized == 0
        assert len(output.errors) == 1
        assert "broken-session" in output.errors[0]
        mock_publisher.assert_not_awaited()
