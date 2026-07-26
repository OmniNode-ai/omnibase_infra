# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-14828 — HandlerRsdScoreCalculate canonical def-B dispatch proof.

RED-against-EXISTS-but-WRONG proof for the canonical def-B flip (hand-flip,
OMN-14781 path) under the canonical-shape ratchet epic OMN-14355.

Before this ticket ``HandlerRsdScoreCalculate.handle`` was declared
``handle(self, correlation_id, tickets, dependency_edges, agent_requests,
plan_overrides, weights) -> ModelRsdScoreResult`` — a MULTI-POSITIONAL signature.
It exposed a callable ``handle`` (so it was NOT entrypoint-less) but the shared
runtime adapter ``_resolve_def_b_input_model_type`` returns ``None`` for it
(more than one positional parameter), so the dispatcher would hand the handler
the RAW materialized envelope instead of a validated ``ModelRsdScoreInput`` and
crash on the first missing positional argument. The canonical-shape ratchet
classified the node ``nonadaptable`` (baselined in ``NON_CANONICAL``).

The flip retypes the entrypoint to the canonical def-B shape
``handle(self, request: ModelRsdScoreInput) -> ModelRsdScoreResult`` and unpacks
the six fields from ``request`` at the top; the 5-factor RSD scoring business
logic (``_count_downstream`` / ``_calculate_dependency_distance`` /
``_calculate_failure_surface`` / ``_calculate_time_decay`` /
``_calculate_agent_utility`` / ``_calculate_user_weighting``) is preserved
byte-identical base_ref<->HEAD, which the ratchet re-derives from git (the
``.handflip.json`` proof).

``test_handle_is_canonical_def_b_typed_entrypoint`` is the RED discriminator: it
asserts the REAL runtime helper resolves ``ModelRsdScoreInput`` from the
entrypoint signature — FALSE on the pre-flip multi-positional tree, TRUE on the
flip. The parametrized dispatch tests drive the REAL production
``_make_dispatch_callback`` over the SELECTED input corpus and assert a SUCCESS
dispatch carrying the computed ``ModelRsdScoreResult``; that corpus is the exact
set bound (by ``input_hash``) into the adequacy receipt and the hand-flip proof
under ``scripts/ci/adequacy_receipts/``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus
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
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _handler_accepts_event_envelope,
    _make_dispatch_callback,
    _resolve_def_b_input_model_type,
)

pytestmark = [pytest.mark.unit]

# Fixed, absolute inputs so the SELECTED corpus hashes reproducibly — the
# adequacy receipt + hand-flip proof pin these exact payloads via input_hash.
_CID_A = UUID("aaaaaaaa-0000-0000-0000-000000000001")
_CID_B = UUID("bbbbbbbb-0000-0000-0000-000000000002")
_CID_C = UUID("cccccccc-0000-0000-0000-000000000003")
_CID_D = UUID("dddddddd-0000-0000-0000-000000000004")


def _c1_rich() -> ModelRsdScoreInput:
    """Multi-ticket graph: dependency bottleneck + failure keywords + agents + override."""
    tickets = (
        ModelTicketData(
            ticket_id="OMN-A",
            title="Critical security validation fix",
            description="audit the validator and replay the trace log",
            priority="critical",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
            tags=("security", "compliance"),
        ),
        ModelTicketData(
            ticket_id="OMN-B",
            title="Add integration test coverage",
            description="unit and e2e spec work",
            priority="high",
            created_at=datetime(2026, 3, 1, tzinfo=UTC),
        ),
        ModelTicketData(
            ticket_id="OMN-C",
            title="Minor tweak",
            priority="low",
            created_at=datetime(2026, 6, 1, tzinfo=UTC),
        ),
    )
    edges = (
        ModelDependencyEdge(source_id="OMN-A", target_id="OMN-B"),
        ModelDependencyEdge(source_id="OMN-A", target_id="OMN-C"),
        ModelDependencyEdge(source_id="OMN-A", target_id="OMN-D"),
        ModelDependencyEdge(source_id="OMN-B", target_id="OMN-C"),
    )
    agents = (
        ModelAgentRequestData(
            agent_id="a1", ticket_id="OMN-A", priority_boost=0.8, is_active=True
        ),
        ModelAgentRequestData(
            agent_id="a2", ticket_id="OMN-A", priority_boost=0.4, is_active=True
        ),
        ModelAgentRequestData(
            agent_id="a3", ticket_id="OMN-B", priority_boost=0.5, is_active=False
        ),
    )
    overrides = (
        ModelPlanOverrideData(
            ticket_id="OMN-C",
            override_score=95.0,
            timestamp=datetime(2026, 6, 25, tzinfo=UTC),
            is_active=True,
        ),
    )
    return ModelRsdScoreInput(
        correlation_id=_CID_A,
        tickets=tickets,
        dependency_edges=edges,
        agent_requests=agents,
        plan_overrides=overrides,
        weights=ModelRsdFactorWeights(),
    )


def _c2_old_boost() -> ModelRsdScoreInput:
    """Single very old ticket: exercises the >90-day time-decay boost branch."""
    return ModelRsdScoreInput(
        correlation_id=_CID_B,
        tickets=(
            ModelTicketData(
                ticket_id="OMN-OLD",
                title="Refactor module",
                priority="medium",
                created_at=datetime(2024, 1, 1, tzinfo=UTC),
            ),
        ),
        weights=ModelRsdFactorWeights(),
    )


def _c3_empty() -> ModelRsdScoreInput:
    """No tickets: exercises the empty-result / no-ranking branch."""
    return ModelRsdScoreInput(
        correlation_id=_CID_C,
        tickets=(),
        weights=ModelRsdFactorWeights(),
    )


def _c4_overrides() -> ModelRsdScoreInput:
    """Override decay + expiry + null-created_at branches of user_weighting/time_decay."""
    tickets = (
        ModelTicketData(
            ticket_id="OMN-OV1",
            title="Tune weighting",
            priority="low",
            created_at=None,
        ),
        ModelTicketData(
            ticket_id="OMN-OV2",
            title="Expired override ticket",
            priority="minimal",
            created_at=datetime(2026, 6, 20, tzinfo=UTC),
        ),
    )
    overrides = (
        ModelPlanOverrideData(
            ticket_id="OMN-OV1",
            override_score=90.0,
            timestamp=datetime(2026, 5, 1, tzinfo=UTC),
            is_active=True,
        ),
        ModelPlanOverrideData(
            ticket_id="OMN-OV2",
            override_score=88.0,
            timestamp=datetime(2026, 6, 1, tzinfo=UTC),
            expires_at=datetime(2026, 6, 2, tzinfo=UTC),
            is_active=True,
        ),
    )
    return ModelRsdScoreInput(
        correlation_id=_CID_D,
        tickets=tickets,
        plan_overrides=overrides,
        weights=ModelRsdFactorWeights(),
    )


# The candidate pool — shared verbatim with the adequacy-receipt recorder driver so
# the recorded selected_input_hashes match what these tests actually drive.
CANDIDATE_BUILDERS = {
    "C1_rich_graph": _c1_rich,
    "C2_old_boost": _c2_old_boost,
    "C3_empty": _c3_empty,
    "C4_overrides": _c4_overrides,
}
CANDIDATES: list[ModelRsdScoreInput] = [b() for b in CANDIDATE_BUILDERS.values()]
_CASE_IDS = list(CANDIDATE_BUILDERS.keys())
_EXPECTED_TICKET_COUNT = {
    "C1_rich_graph": 3,
    "C2_old_boost": 1,
    "C3_empty": 0,
    "C4_overrides": 2,
}


@pytest.fixture
def handler() -> HandlerRsdScoreCalculate:
    return HandlerRsdScoreCalculate()


@pytest.mark.unit
def test_handle_is_canonical_def_b_typed_entrypoint(
    handler: HandlerRsdScoreCalculate,
) -> None:
    """RED discriminator: the entrypoint is the canonical def-B typed shape.

    Pre-flip (multi-positional ``handle``) the runtime resolves NO typed input
    model (returns ``None``) — this test is RED there. Post-flip the runtime
    resolves ``ModelRsdScoreInput`` and the core does not accept a raw envelope.
    """
    assert _resolve_def_b_input_model_type(handler.handle) is ModelRsdScoreInput, (
        "handle() is not an adaptable def-B typed entrypoint — the runtime would "
        "hand it the raw envelope instead of a validated ModelRsdScoreInput."
    )
    # The core does NOT accept a raw envelope; the envelope boundary is the
    # shared runtime adapter (definition B, C-core).
    assert _handler_accepts_event_envelope(handler.handle) is False


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("case_id", _CASE_IDS, ids=_CASE_IDS)
async def test_real_dispatch_callback_computes_scores(case_id: str) -> None:
    """LOAD-BEARING: a real ModelRsdScoreInput dispatched through the REAL
    auto-wiring callback reaches the def-B ``handle`` and yields a SUCCESS
    dispatch carrying the computed ``ModelRsdScoreResult``.

    The contract declares ``operation_match`` (no ``event_model``), so this
    exercises the untyped def-B coercion arm: the adapter passes the validated
    ``ModelRsdScoreInput`` typed model, not the raw envelope dict.
    """
    handler = HandlerRsdScoreCalculate()
    payload = CANDIDATE_BUILDERS[case_id]()
    callback = _make_dispatch_callback(handler, None)
    envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
        payload=payload,
        correlation_id=uuid4(),
        event_type="ModelRsdScoreInput",
    )

    result = await callback(envelope)

    assert result is not None, "Dispatch produced no result — the handler never ran."
    assert result.status is EnumDispatchStatus.SUCCESS, (
        f"Expected SUCCESS dispatch status, got {result.status!r}."
    )
    # A COMPUTE handler's bare BaseModel return is classified as a single output event.
    assert len(result.output_events) == 1
    score_result = result.output_events[0]
    assert isinstance(score_result, ModelRsdScoreResult)
    expected_n = _EXPECTED_TICKET_COUNT[case_id]
    assert len(score_result.ticket_scores) == expected_n
    assert len(score_result.ranked_ticket_ids) == expected_n
    assert set(score_result.ranked_ticket_ids) == {t.ticket_id for t in payload.tickets}
    for ticket_score in score_result.ticket_scores:
        assert 0.0 <= ticket_score.final_score <= 1.0
        assert len(ticket_score.factors) == 5


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rich_graph_ranks_bottleneck_ticket_first() -> None:
    """Behavior anchor: the critical, dependency-bottleneck ticket outranks the
    trivial low-priority one — invariant of the preserved scoring logic and
    independent of wall-clock."""
    handler = HandlerRsdScoreCalculate()
    result = await handler.handle(_c1_rich())
    ranked = result.ranked_ticket_ids
    assert ranked[0] == "OMN-A"
    assert ranked.index("OMN-A") < ranked.index("OMN-C")


@pytest.mark.unit
def test_missing_typed_entrypoint_is_the_red() -> None:
    """Documents the exact RED the flip closes.

    A handler whose ``handle`` takes multiple positional parameters is not an
    adaptable def-B typed entrypoint — the runtime resolves no input model, so it
    would hand the handler the raw envelope. This guards against silent
    regression of the multi-positional non-canonical shape through the REAL
    runtime helper.
    """

    class _LegacyShape:
        async def handle(
            self,
            correlation_id: object,
            tickets: object,
            weights: object,
        ) -> None:
            raise AssertionError("unreachable")

    assert _resolve_def_b_input_model_type(_LegacyShape().handle) is None
