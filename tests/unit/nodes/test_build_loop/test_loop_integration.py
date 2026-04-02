# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Canary integration test: full in-memory build loop cycle.

Verifies the orchestrator drives the reducer through all phases,
invokes compute/effect handlers, and produces a valid result.

Related:
    - OMN-7323: Canary integration test
    - OMN-5113: Autonomous Build Loop epic
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

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
