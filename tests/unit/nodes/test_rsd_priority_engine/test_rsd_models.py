# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for RSD priority engine models."""

from __future__ import annotations

from datetime import datetime
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
from omnibase_infra.nodes.node_rsd_data_fetch_effect.models.model_rsd_data_fetch_request import (
    ModelRsdDataFetchRequest,
)
from omnibase_infra.nodes.node_rsd_data_fetch_effect.models.model_rsd_data_fetch_result import (
    ModelRsdDataFetchResult,
)
from omnibase_infra.nodes.node_rsd_data_fetch_effect.models.model_ticket_data import (
    ModelTicketData,
)
from omnibase_infra.nodes.node_rsd_orchestrator.models.model_rsd_command import (
    ModelRsdCommand,
)
from omnibase_infra.nodes.node_rsd_orchestrator.models.model_rsd_result import (
    ModelRsdResult,
)
from omnibase_infra.nodes.node_rsd_score_compute.models.model_rsd_factor_score import (
    ModelRsdFactorScore,
)
from omnibase_infra.nodes.node_rsd_score_compute.models.model_rsd_factor_weights import (
    ModelRsdFactorWeights,
)
from omnibase_infra.nodes.node_rsd_score_compute.models.model_rsd_score_input import (
    ModelRsdScoreInput,
)
from omnibase_infra.nodes.node_rsd_score_compute.models.model_rsd_score_result import (
    ModelRsdScoreResult,
)
from omnibase_infra.nodes.node_rsd_score_compute.models.model_rsd_ticket_score import (
    ModelRsdTicketScore,
)
from omnibase_infra.nodes.node_rsd_state_reducer.models.model_rsd_rank_change import (
    ModelRsdRankChange,
)
from omnibase_infra.nodes.node_rsd_state_reducer.models.model_rsd_score_snapshot import (
    ModelRsdScoreSnapshot,
)
from omnibase_infra.nodes.node_rsd_state_reducer.models.model_rsd_state import (
    ModelRsdState,
)


@pytest.mark.unit
class TestRsdModels:
    """Tests that all RSD models can be instantiated and are frozen."""

    def test_data_fetch_request(self):
        cid = uuid4()
        req = ModelRsdDataFetchRequest(correlation_id=cid, ticket_ids=("T-1", "T-2"))
        assert req.correlation_id == cid
        assert len(req.ticket_ids) == 2
        with pytest.raises(Exception):
            req.ticket_ids = ("T-3",)  # type: ignore[misc]

    def test_ticket_data(self):
        t = ModelTicketData(ticket_id="T-1", title="Fix bug", priority="high")
        assert t.status == "open"
        assert t.tags == ()

    def test_dependency_edge(self):
        e = ModelDependencyEdge(source_id="T-1", target_id="T-2")
        assert e.edge_type == "depends_on"
        assert e.weight == 1.0

    def test_agent_request_data(self):
        r = ModelAgentRequestData(agent_id="agent-1", ticket_id="T-1")
        assert r.is_active is True
        assert r.priority_boost == 0.0

    def test_plan_override_data(self):
        o = ModelPlanOverrideData(
            ticket_id="T-1", override_score=80.0, previous_score=50.0
        )
        assert o.is_active is True

    def test_data_fetch_result(self):
        r = ModelRsdDataFetchResult(correlation_id=uuid4())
        assert r.success is True
        assert r.tickets == ()

    def test_factor_weights_defaults(self):
        w = ModelRsdFactorWeights()
        total = (
            w.dependency_distance
            + w.failure_surface
            + w.time_decay
            + w.agent_utility
            + w.user_weighting
        )
        assert abs(total - 1.0) < 1e-9

    def test_score_input(self):
        inp = ModelRsdScoreInput(
            correlation_id=uuid4(),
            tickets=(ModelTicketData(ticket_id="T-1"),),
        )
        assert len(inp.tickets) == 1

    def test_factor_score(self):
        f = ModelRsdFactorScore(
            factor_name="time_decay",
            raw_score=0.6,
            weight=0.15,
            weighted_score=0.09,
        )
        assert f.factor_name == "time_decay"

    def test_ticket_score(self):
        ts = ModelRsdTicketScore(
            ticket_id="T-1",
            final_score=0.75,
            factors=(
                ModelRsdFactorScore(
                    factor_name="dep",
                    raw_score=1.0,
                    weight=0.4,
                    weighted_score=0.4,
                ),
            ),
        )
        assert ts.final_score == 0.75

    def test_score_result(self):
        sr = ModelRsdScoreResult(
            correlation_id=uuid4(),
            ticket_scores=(),
            ranked_ticket_ids=(),
        )
        assert sr.ranked_ticket_ids == ()

    def test_rsd_state(self):
        s = ModelRsdState(correlation_id=uuid4())
        assert s.workflow_state == "pending"
        assert s.total_cycles == 0

    def test_rank_change(self):
        rc = ModelRsdRankChange(
            ticket_id="T-1",
            previous_rank=3,
            new_rank=1,
            score_delta=0.15,
        )
        assert rc.new_rank < rc.previous_rank

    def test_score_snapshot(self):
        ss = ModelRsdScoreSnapshot(
            calculated_at=datetime.now(),
            ticket_scores=(),
            ranked_ticket_ids=(),
        )
        assert ss.ticket_scores == ()

    def test_rsd_command(self):
        cmd = ModelRsdCommand(correlation_id=uuid4(), ticket_ids=("T-1", "T-2", "T-3"))
        assert len(cmd.ticket_ids) == 3

    def test_rsd_result(self):
        r = ModelRsdResult(
            correlation_id=uuid4(),
            ticket_scores=(),
            ranked_ticket_ids=(),
        )
        assert r.success is True
