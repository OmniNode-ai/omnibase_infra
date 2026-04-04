# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Canary integration test: full in-memory build loop cycle.

Verifies the orchestrator drives the reducer through all phases,
invokes compute/effect handlers, and produces a valid result.

Related:
    - OMN-7323: Canary integration test
    - OMN-7324: Wire real build loop handlers
    - OMN-5113: Autonomous Build Loop epic
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest

from omnibase_infra.enums.enum_build_loop_phase import EnumBuildLoopPhase
from omnibase_infra.nodes.node_autonomous_loop_orchestrator.handlers.handler_loop_orchestrator import (
    HandlerLoopOrchestrator,
)
from omnibase_infra.nodes.node_autonomous_loop_orchestrator.models.model_loop_orchestrator import (
    ModelLoopStartCommand,
)


@pytest.mark.unit
class TestBuildLoopIntegration:
    """Full in-memory loop integration tests."""

    @pytest.mark.asyncio
    async def test_dry_run_single_cycle(self):
        """A dry-run single cycle should complete successfully through all phases."""
        orchestrator = HandlerLoopOrchestrator()
        command = ModelLoopStartCommand(
            correlation_id=uuid4(),
            max_cycles=1,
            skip_closeout=False,
            dry_run=True,
            requested_at=datetime.now(tz=UTC),
        )

        result = await orchestrator.handle(command)

        assert result.cycles_completed == 1
        assert result.cycles_failed == 0
        assert len(result.cycle_summaries) == 1
        summary = result.cycle_summaries[0]
        assert summary.final_phase == EnumBuildLoopPhase.COMPLETE
        assert summary.cycle_number == 1

    @pytest.mark.asyncio
    async def test_dry_run_skip_closeout(self):
        """Skip closeout should go IDLE -> VERIFYING directly."""
        orchestrator = HandlerLoopOrchestrator()
        command = ModelLoopStartCommand(
            correlation_id=uuid4(),
            max_cycles=1,
            skip_closeout=True,
            dry_run=True,
            requested_at=datetime.now(tz=UTC),
        )

        result = await orchestrator.handle(command)

        assert result.cycles_completed == 1
        assert result.cycles_failed == 0
        assert result.cycle_summaries[0].final_phase == EnumBuildLoopPhase.COMPLETE

    @pytest.mark.asyncio
    async def test_dry_run_multiple_cycles(self):
        """Multiple cycles should all complete in dry-run mode."""
        orchestrator = HandlerLoopOrchestrator()
        command = ModelLoopStartCommand(
            correlation_id=uuid4(),
            max_cycles=3,
            skip_closeout=True,
            dry_run=True,
            requested_at=datetime.now(tz=UTC),
        )

        result = await orchestrator.handle(command)

        assert result.cycles_completed == 3
        assert result.cycles_failed == 0
        assert len(result.cycle_summaries) == 3
        for i, summary in enumerate(result.cycle_summaries):
            assert summary.final_phase == EnumBuildLoopPhase.COMPLETE

    @pytest.mark.asyncio
    async def test_correlation_id_propagation(self):
        """Correlation ID should propagate through all summaries."""
        cid = uuid4()
        orchestrator = HandlerLoopOrchestrator()
        command = ModelLoopStartCommand(
            correlation_id=cid,
            max_cycles=1,
            dry_run=True,
            requested_at=datetime.now(tz=UTC),
        )

        result = await orchestrator.handle(command)

        assert result.correlation_id == cid
        assert result.cycle_summaries[0].correlation_id == cid


def _make_linear_response(tickets: list[dict]) -> dict:
    """Build a mock Linear GraphQL response."""
    return {"data": {"issues": {"nodes": tickets}}}


_SAMPLE_LINEAR_TICKETS = [
    {
        "id": "abc-123",
        "identifier": "OMN-9001",
        "title": "Implement new handler for build dispatch",
        "priority": 1,
        "description": "Wire the build dispatch handler to process targets",
        "state": {"name": "Backlog", "type": "backlog"},
        "labels": {"nodes": [{"name": "automation"}, {"name": "handler"}]},
    },
    {
        "id": "abc-124",
        "identifier": "OMN-9002",
        "title": "Investigate spike: evaluate caching strategy",
        "priority": 3,
        "description": "Research and evaluate caching approaches",
        "state": {"name": "Backlog", "type": "backlog"},
        "labels": {"nodes": [{"name": "research"}]},
    },
    {
        "id": "abc-125",
        "identifier": "OMN-9003",
        "title": "Fix broken test in node_rsd_fill",
        "priority": 2,
        "description": "Unit test fails after model refactor",
        "state": {"name": "Todo", "type": "unstarted"},
        "labels": {"nodes": [{"name": "bug"}, {"name": "test"}]},
    },
]


@pytest.mark.unit
class TestBuildLoopLinearIntegration:
    """Tests with mocked Linear API responses flowing through the full pipeline."""

    @pytest.mark.asyncio
    async def test_no_linear_api_key_graceful_degradation(
        self,
    ):
        """Without LINEAR_API_KEY, cycle completes with zero tickets."""
        orchestrator = HandlerLoopOrchestrator()
        command = ModelLoopStartCommand(
            correlation_id=uuid4(),
            max_cycles=1,
            skip_closeout=True,
            dry_run=True,
            requested_at=datetime.now(tz=UTC),
        )

        result = await orchestrator.handle(command)

        assert result.cycles_completed == 1
        assert result.cycles_failed == 0
        summary = result.cycle_summaries[0]
        assert summary.final_phase == EnumBuildLoopPhase.COMPLETE
        assert summary.tickets_filled == 0
        assert summary.tickets_classified == 0
        assert summary.tickets_dispatched == 0

    @pytest.mark.asyncio
    async def test_mocked_linear_tickets_flow_through_pipeline(
        self,
    ):
        """Mocked Linear tickets should flow through fill -> classify -> dispatch."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = _make_linear_response(_SAMPLE_LINEAR_TICKETS)

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            orchestrator = HandlerLoopOrchestrator(linear_api_key="lin_api_test_key")
            command = ModelLoopStartCommand(
                correlation_id=uuid4(),
                max_cycles=1,
                skip_closeout=True,
                dry_run=True,
                requested_at=datetime.now(tz=UTC),
            )

            result = await orchestrator.handle(command)

        assert result.cycles_completed == 1
        assert result.cycles_failed == 0
        summary = result.cycle_summaries[0]
        assert summary.final_phase == EnumBuildLoopPhase.COMPLETE
        # 3 tickets fetched, all 3 selected (max_tickets=5)
        assert summary.tickets_filled == 3
        # All 3 classified
        assert summary.tickets_classified == 3
        # OMN-9001 (handler/implement) and OMN-9003 (fix/test) are AUTO_BUILDABLE
        # OMN-9002 (investigate/spike/research/evaluate) is NEEDS_ARCH_DECISION
        assert summary.tickets_dispatched == 2

    @pytest.mark.asyncio
    async def test_linear_api_failure_graceful_degradation(
        self,
    ):
        """Linear API failure should not crash the loop — graceful empty result."""
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.ConnectError("connection refused")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            orchestrator = HandlerLoopOrchestrator(linear_api_key="lin_api_test_key")
            command = ModelLoopStartCommand(
                correlation_id=uuid4(),
                max_cycles=1,
                skip_closeout=True,
                dry_run=True,
                requested_at=datetime.now(tz=UTC),
            )

            result = await orchestrator.handle(command)

        assert result.cycles_completed == 1
        assert result.cycles_failed == 0
        summary = result.cycle_summaries[0]
        assert summary.final_phase == EnumBuildLoopPhase.COMPLETE
        assert summary.tickets_filled == 0

    @pytest.mark.asyncio
    async def test_classification_filters_non_buildable(
        self,
    ):
        """Only AUTO_BUILDABLE tickets should reach the build dispatch phase."""
        # All tickets are architecture/research — none should be AUTO_BUILDABLE
        arch_tickets = [
            {
                "id": "xyz-1",
                "identifier": "OMN-8001",
                "title": "Investigate database architecture spike",
                "priority": 2,
                "description": "Research and evaluate new DB options",
                "state": {"name": "Backlog", "type": "backlog"},
                "labels": {"nodes": [{"name": "research"}]},
            },
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = _make_linear_response(arch_tickets)

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            orchestrator = HandlerLoopOrchestrator(linear_api_key="lin_api_test_key")
            command = ModelLoopStartCommand(
                correlation_id=uuid4(),
                max_cycles=1,
                skip_closeout=True,
                dry_run=True,
                requested_at=datetime.now(tz=UTC),
            )

            result = await orchestrator.handle(command)

        assert result.cycles_completed == 1
        summary = result.cycle_summaries[0]
        assert summary.tickets_filled == 1
        assert summary.tickets_classified == 1
        # Architecture ticket should NOT be dispatched
        assert summary.tickets_dispatched == 0
