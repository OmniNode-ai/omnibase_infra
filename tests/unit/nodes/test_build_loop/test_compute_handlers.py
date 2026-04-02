# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for build loop COMPUTE handlers.

Related:
    - OMN-7314, OMN-7315: Compute nodes
    - OMN-7323: Canary integration test
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.enums.enum_buildability import EnumBuildability
from omnibase_infra.nodes.node_rsd_fill_compute.handlers.handler_rsd_fill import (
    HandlerRsdFill,
)
from omnibase_infra.nodes.node_rsd_fill_compute.models.model_scored_ticket import (
    ModelScoredTicket,
)
from omnibase_infra.nodes.node_ticket_classify_compute.handlers.handler_ticket_classify import (
    HandlerTicketClassify,
)
from omnibase_infra.nodes.node_ticket_classify_compute.models.model_ticket_for_classification import (
    ModelTicketForClassification,
)


def _ticket(ticket_id: str, rsd: float, priority: int = 2) -> ModelScoredTicket:
    return ModelScoredTicket(
        ticket_id=ticket_id,
        title=f"Ticket {ticket_id}",
        rsd_score=rsd,
        priority=priority,
    )


@pytest.mark.unit
class TestHandlerRsdFill:
    """Tests for RSD fill compute handler."""

    @pytest.mark.asyncio
    async def test_selects_top_n(self):
        handler = HandlerRsdFill()
        tickets = (
            _ticket("OMN-1", 5.0),
            _ticket("OMN-2", 9.0),
            _ticket("OMN-3", 7.0),
            _ticket("OMN-4", 3.0),
            _ticket("OMN-5", 8.0),
        )
        result = await handler.handle(
            correlation_id=uuid4(),
            scored_tickets=tickets,
            max_tickets=3,
        )
        assert result.total_selected == 3
        assert result.total_candidates == 5
        ids = [t.ticket_id for t in result.selected_tickets]
        assert ids == ["OMN-2", "OMN-5", "OMN-3"]

    @pytest.mark.asyncio
    async def test_deterministic_tiebreak(self):
        """Same RSD score: lower priority number wins, then ticket_id ASC."""
        handler = HandlerRsdFill()
        tickets = (
            _ticket("OMN-B", 5.0, priority=2),
            _ticket("OMN-A", 5.0, priority=2),
            _ticket("OMN-C", 5.0, priority=1),  # urgent
        )
        result = await handler.handle(
            correlation_id=uuid4(),
            scored_tickets=tickets,
            max_tickets=3,
        )
        ids = [t.ticket_id for t in result.selected_tickets]
        assert ids == ["OMN-C", "OMN-A", "OMN-B"]

    @pytest.mark.asyncio
    async def test_empty_input(self):
        handler = HandlerRsdFill()
        result = await handler.handle(
            correlation_id=uuid4(),
            scored_tickets=(),
            max_tickets=5,
        )
        assert result.total_selected == 0
        assert result.selected_tickets == ()


@pytest.mark.unit
class TestHandlerTicketClassify:
    """Tests for ticket classify compute handler."""

    @pytest.mark.asyncio
    async def test_auto_buildable(self):
        handler = HandlerTicketClassify()
        tickets = (
            ModelTicketForClassification(
                ticket_id="OMN-1",
                title="Add build loop node",
                description="Create the node_rsd_fill_compute handler",
            ),
        )
        result = await handler.handle(correlation_id=uuid4(), tickets=tickets)
        assert result.total_auto_buildable == 1
        assert result.classifications[0].buildability == EnumBuildability.AUTO_BUILDABLE

    @pytest.mark.asyncio
    async def test_blocked_ticket(self):
        handler = HandlerTicketClassify()
        tickets = (
            ModelTicketForClassification(
                ticket_id="OMN-2",
                title="Waiting on external vendor approval",
                description="Blocked by third-party dependency",
            ),
        )
        result = await handler.handle(correlation_id=uuid4(), tickets=tickets)
        assert result.classifications[0].buildability == EnumBuildability.BLOCKED

    @pytest.mark.asyncio
    async def test_arch_decision(self):
        handler = HandlerTicketClassify()
        tickets = (
            ModelTicketForClassification(
                ticket_id="OMN-3",
                title="Evaluate architecture for new pipeline",
                description="RFC needed for design decision",
            ),
        )
        result = await handler.handle(correlation_id=uuid4(), tickets=tickets)
        assert (
            result.classifications[0].buildability
            == EnumBuildability.NEEDS_ARCH_DECISION
        )

    @pytest.mark.asyncio
    async def test_skip_terminal_state(self):
        handler = HandlerTicketClassify()
        tickets = (
            ModelTicketForClassification(
                ticket_id="OMN-4",
                title="Something normal",
                description="",
                state="Done",
            ),
        )
        result = await handler.handle(correlation_id=uuid4(), tickets=tickets)
        assert result.classifications[0].buildability == EnumBuildability.SKIP

    @pytest.mark.asyncio
    async def test_mixed_classification(self):
        handler = HandlerTicketClassify()
        tickets = (
            ModelTicketForClassification(
                ticket_id="OMN-10",
                title="Add new handler",
                description="implement node",
            ),
            ModelTicketForClassification(
                ticket_id="OMN-11",
                title="Blocked by vendor",
                description="external dependency",
            ),
            ModelTicketForClassification(
                ticket_id="OMN-12",
                title="Investigate spike on auth",
                description="research tradeoff",
            ),
        )
        result = await handler.handle(correlation_id=uuid4(), tickets=tickets)
        assert result.total_auto_buildable == 1
        assert result.total_skipped == 2
        buildabilities = [c.buildability for c in result.classifications]
        assert EnumBuildability.AUTO_BUILDABLE in buildabilities
        assert EnumBuildability.BLOCKED in buildabilities
        assert EnumBuildability.NEEDS_ARCH_DECISION in buildabilities
