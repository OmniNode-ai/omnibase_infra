# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for HandlerRsdScoreCalculate - the core RSD 5-factor algorithm."""

from __future__ import annotations

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_rsd_data_fetch_effect.models.model_agent_request_data import (
    ModelAgentRequestData,
)
from omnibase_infra.nodes.node_rsd_data_fetch_effect.models.model_dependency_edge import (
    ModelDependencyEdge,
)
from omnibase_infra.nodes.node_rsd_data_fetch_effect.models.model_plan_override_data import (
    ModelPlanOverrideData,
)
from omnibase_infra.nodes.node_rsd_data_fetch_effect.models.model_ticket_data import (
    ModelTicketData,
)
from omnibase_infra.nodes.node_rsd_score_compute.handlers.handler_rsd_score_calculate import (
    HandlerRsdScoreCalculate,
    _calculate_agent_utility,
    _calculate_dependency_distance,
    _calculate_failure_surface,
    _calculate_time_decay,
    _calculate_user_weighting,
)
from omnibase_infra.nodes.node_rsd_score_compute.models.model_rsd_factor_weights import (
    ModelRsdFactorWeights,
)


@pytest.mark.unit
class TestHandlerRsdScoreCalculate:
    """Tests for the core RSD scoring handler."""

    @pytest.fixture
    def handler(self) -> HandlerRsdScoreCalculate:
        return HandlerRsdScoreCalculate()

    @pytest.fixture
    def weights(self) -> ModelRsdFactorWeights:
        return ModelRsdFactorWeights()

    @pytest.mark.asyncio
    async def test_empty_tickets(self, handler: HandlerRsdScoreCalculate):
        """Empty ticket list returns empty result."""
        result = await handler.handle(
            correlation_id=uuid4(),
            tickets=(),
            dependency_edges=(),
            agent_requests=(),
            plan_overrides=(),
            weights=ModelRsdFactorWeights(),
        )
        assert result.ticket_scores == ()
        assert result.ranked_ticket_ids == ()

    @pytest.mark.asyncio
    async def test_single_ticket_scores_between_0_and_1(
        self, handler: HandlerRsdScoreCalculate
    ):
        """A single ticket should produce a score in [0, 1]."""
        ticket = ModelTicketData(
            ticket_id="T-1",
            title="Fix validation bug",
            priority="high",
            created_at=datetime.now() - timedelta(days=14),
        )
        result = await handler.handle(
            correlation_id=uuid4(),
            tickets=(ticket,),
            dependency_edges=(),
            agent_requests=(),
            plan_overrides=(),
            weights=ModelRsdFactorWeights(),
        )
        assert len(result.ticket_scores) == 1
        score = result.ticket_scores[0]
        assert 0.0 <= score.final_score <= 1.0
        assert len(score.factors) == 5

    @pytest.mark.asyncio
    async def test_ranking_order(self, handler: HandlerRsdScoreCalculate):
        """Tickets should be ranked by descending score."""
        tickets = (
            ModelTicketData(
                ticket_id="low",
                title="Minor tweak",
                priority="low",
                created_at=datetime.now(),
            ),
            ModelTicketData(
                ticket_id="critical",
                title="Critical security validation fix",
                priority="critical",
                created_at=datetime.now() - timedelta(days=100),
                tags=("security", "compliance"),
            ),
        )
        result = await handler.handle(
            correlation_id=uuid4(),
            tickets=tickets,
            dependency_edges=(),
            agent_requests=(),
            plan_overrides=(),
            weights=ModelRsdFactorWeights(),
        )
        assert result.ranked_ticket_ids[0] == "critical"
        assert result.ranked_ticket_ids[1] == "low"

    @pytest.mark.asyncio
    async def test_factor_weights_sum(self, handler: HandlerRsdScoreCalculate):
        """Factor weighted scores should sum to the final score."""
        ticket = ModelTicketData(
            ticket_id="T-1", priority="medium", created_at=datetime.now()
        )
        result = await handler.handle(
            correlation_id=uuid4(),
            tickets=(ticket,),
            dependency_edges=(),
            agent_requests=(),
            plan_overrides=(),
            weights=ModelRsdFactorWeights(),
        )
        score = result.ticket_scores[0]
        factor_sum = sum(f.weighted_score for f in score.factors)
        assert abs(score.final_score - max(0.0, min(1.0, factor_sum))) < 1e-9

    @pytest.mark.asyncio
    async def test_dependency_boost(self, handler: HandlerRsdScoreCalculate):
        """Ticket blocking others should score higher on dependency factor."""
        blocker = ModelTicketData(ticket_id="blocker", priority="medium")
        blocked = ModelTicketData(ticket_id="blocked", priority="medium")
        edges = (
            ModelDependencyEdge(source_id="blocker", target_id="blocked"),
            ModelDependencyEdge(source_id="blocker", target_id="T-3"),
            ModelDependencyEdge(source_id="blocker", target_id="T-4"),
        )
        result = await handler.handle(
            correlation_id=uuid4(),
            tickets=(blocker, blocked),
            dependency_edges=edges,
            agent_requests=(),
            plan_overrides=(),
            weights=ModelRsdFactorWeights(),
        )
        scores = {s.ticket_id: s for s in result.ticket_scores}
        blocker_dep = next(
            f
            for f in scores["blocker"].factors
            if f.factor_name == "dependency_distance"
        )
        blocked_dep = next(
            f
            for f in scores["blocked"].factors
            if f.factor_name == "dependency_distance"
        )
        assert blocker_dep.raw_score > blocked_dep.raw_score


@pytest.mark.unit
class TestDependencyDistance:
    """Tests for the dependency distance scoring function."""

    def test_no_edges(self):
        ticket = ModelTicketData(ticket_id="T-1")
        score = _calculate_dependency_distance(ticket, [], {}, {"T-1"})
        # No edges means low dependency score
        assert 0.0 <= score <= 1.0

    def test_many_dependents_high_score(self):
        ticket = ModelTicketData(ticket_id="T-1")
        edges = [
            ModelDependencyEdge(source_id="T-1", target_id=f"T-{i}")
            for i in range(2, 7)
        ]
        edges_by_source = {"T-1": edges}
        score = _calculate_dependency_distance(ticket, edges, edges_by_source, {"T-1"})
        assert score > 0.5

    def test_cycle_safety(self):
        """Cycles in dependency graph should not cause infinite recursion."""
        ticket = ModelTicketData(ticket_id="T-1")
        edges = [
            ModelDependencyEdge(source_id="T-1", target_id="T-2"),
            ModelDependencyEdge(source_id="T-2", target_id="T-1"),
        ]
        edges_by_source = {"T-1": [edges[0]], "T-2": [edges[1]]}
        score = _calculate_dependency_distance(
            ticket, [edges[0]], edges_by_source, {"T-1", "T-2"}
        )
        assert 0.0 <= score <= 1.0


@pytest.mark.unit
class TestFailureSurface:
    """Tests for the failure surface scoring function."""

    def test_no_keywords(self):
        ticket = ModelTicketData(ticket_id="T-1", title="Add feature")
        score = _calculate_failure_surface(ticket)
        assert score == 0.0

    def test_validator_keywords_boost(self):
        ticket = ModelTicketData(
            ticket_id="T-1",
            title="Fix validation check",
            description="Update the validator audit",
        )
        score = _calculate_failure_surface(ticket)
        assert score > 0.0

    def test_critical_tags_boost(self):
        ticket = ModelTicketData(
            ticket_id="T-1",
            title="Update module",
            tags=("security", "compliance"),
        )
        score = _calculate_failure_surface(ticket)
        assert score > 0.0

    def test_critical_priority_multiplier(self):
        base = ModelTicketData(
            ticket_id="T-1",
            title="Fix validation",
            priority="medium",
        )
        critical = ModelTicketData(
            ticket_id="T-2",
            title="Fix validation",
            priority="critical",
        )
        assert _calculate_failure_surface(critical) > _calculate_failure_surface(base)


@pytest.mark.unit
class TestTimeDecay:
    """Tests for the time decay scoring function."""

    def test_no_created_at(self):
        ticket = ModelTicketData(ticket_id="T-1", created_at=None)
        assert _calculate_time_decay(ticket) == 0.5

    def test_new_ticket_low_score(self):
        ticket = ModelTicketData(ticket_id="T-1", created_at=datetime.now())
        score = _calculate_time_decay(ticket)
        assert score < 0.2

    def test_old_ticket_high_score(self):
        ticket = ModelTicketData(
            ticket_id="T-1",
            created_at=datetime.now() - timedelta(days=120),
        )
        score = _calculate_time_decay(ticket)
        assert score > 0.8

    def test_monotonically_increasing(self):
        """Older tickets should always score higher."""
        scores = []
        for days in [0, 7, 30, 60, 90, 120]:
            ticket = ModelTicketData(
                ticket_id="T-1",
                created_at=datetime.now() - timedelta(days=days),
            )
            scores.append(_calculate_time_decay(ticket))
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1]


@pytest.mark.unit
class TestAgentUtility:
    """Tests for the agent utility scoring function."""

    def test_no_requests(self):
        assert _calculate_agent_utility([]) == 0.0

    def test_single_active_request(self):
        reqs = [
            ModelAgentRequestData(
                agent_id="a1", ticket_id="T-1", priority_boost=0.5, is_active=True
            )
        ]
        score = _calculate_agent_utility(reqs)
        assert score > 0.0

    def test_inactive_requests_ignored(self):
        reqs = [ModelAgentRequestData(agent_id="a1", ticket_id="T-1", is_active=False)]
        assert _calculate_agent_utility(reqs) == 0.0

    def test_multiple_agents_higher(self):
        single = [
            ModelAgentRequestData(agent_id="a1", ticket_id="T-1", priority_boost=0.5)
        ]
        multi = [
            ModelAgentRequestData(agent_id=f"a{i}", ticket_id="T-1", priority_boost=0.5)
            for i in range(3)
        ]
        assert _calculate_agent_utility(multi) > _calculate_agent_utility(single)


@pytest.mark.unit
class TestUserWeighting:
    """Tests for the user weighting scoring function."""

    def test_priority_mapping(self):
        for priority, expected_min in [
            ("critical", 0.9),
            ("high", 0.7),
            ("medium", 0.4),
            ("low", 0.2),
        ]:
            ticket = ModelTicketData(ticket_id="T-1", priority=priority)
            score = _calculate_user_weighting(ticket, [])
            assert score >= expected_min, f"{priority} scored {score}"

    def test_active_override_applies(self):
        ticket = ModelTicketData(ticket_id="T-1", priority="low")
        override = ModelPlanOverrideData(
            ticket_id="T-1",
            override_score=90.0,
            timestamp=datetime.now(),
            is_active=True,
        )
        score = _calculate_user_weighting(ticket, [override])
        assert score > 0.8

    def test_expired_override_ignored(self):
        ticket = ModelTicketData(ticket_id="T-1", priority="low")
        override = ModelPlanOverrideData(
            ticket_id="T-1",
            override_score=90.0,
            timestamp=datetime.now() - timedelta(days=1),
            expires_at=datetime.now() - timedelta(hours=1),
            is_active=True,
        )
        score = _calculate_user_weighting(ticket, [override])
        # Should fall back to base priority score for "low"
        assert score < 0.5

    def test_old_override_decays(self):
        ticket = ModelTicketData(ticket_id="T-1", priority="medium")
        recent = ModelPlanOverrideData(
            ticket_id="T-1",
            override_score=90.0,
            timestamp=datetime.now(),
            is_active=True,
        )
        old = ModelPlanOverrideData(
            ticket_id="T-1",
            override_score=90.0,
            timestamp=datetime.now() - timedelta(days=30),
            is_active=True,
        )
        recent_score = _calculate_user_weighting(ticket, [recent])
        old_score = _calculate_user_weighting(ticket, [old])
        assert recent_score > old_score
