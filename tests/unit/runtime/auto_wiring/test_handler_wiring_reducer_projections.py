# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Reducer projection-aware dispatch-adapter tests (OMN-14598).

Before this fix, ``_normalize_handler_result`` read only ``.events`` /
``.intents`` / ``.result`` off a ``ModelHandlerOutput`` and never ``.projections``.
A def-B REDUCER return therefore lost its projections two ways:

  - a ``ModelHandlerOutput.for_reducer(projections=(...))`` had its projections
    read by NO branch and silently dropped (``HandlerCodingAgentFsm`` emits TWO
    such projections per fold, both dropped);
  - a bare typed model / ``Sequence`` return from a reducer fell through to the
    ``output_events`` / fan-out-event branches, misclassifying a projection as an
    EVENT (which the REDUCER output contract forbids), while
    ``projection_intents`` stayed empty so ``DispatchResultApplier``'s projection
    sink never fired.

These tests pin the projection-aware branch: a REDUCER return (as
``ModelHandlerOutput.for_reducer`` projections, a bare typed model, or a
``Sequence`` of models) populates ``ModelDispatchResult.projection_intents`` —
NOT ``output_events`` — while non-reducer bare/typed returns keep their existing
event classification.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import BaseModel

from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _build_projection_intents,
    _coerce_projection_models,
    _derive_projector_key,
    _node_kind_from_node_type,
    _normalize_handler_result,
)


class ModelProjectionA(BaseModel):
    value: str


class ModelProjectionB(BaseModel):
    count: int


def _envelope() -> dict[str, object]:
    """A minimal transport envelope carrying a valid hex correlation_id."""
    return {
        "payload": {"correlation_id": str(uuid4())},
        "partition_key": None,
        "correlation_id": str(uuid4()),
        "event_type": "onex.evt.test.reducer.v1",
        "envelope_id": str(uuid4()),
    }


@pytest.mark.unit
class TestReducerHandlerOutputProjections:
    def test_for_reducer_projections_populate_projection_intents(self) -> None:
        """ModelHandlerOutput.for_reducer projections -> projection_intents."""
        output = ModelHandlerOutput.for_reducer(
            input_envelope_id=uuid4(),
            correlation_id=uuid4(),
            handler_id="test-reducer",
            projections=(ModelProjectionA(value="x"), ModelProjectionB(count=3)),
        )

        result = _normalize_handler_result(output, _envelope(), "ModelReducerAdvance")

        assert result is not None
        # Projections routed to projection_intents, NOT output_events.
        assert len(result.projection_intents) == 2
        assert result.output_events == []
        assert result.output_intents == ()
        assert result.output_count == 2
        keys = {pi.projector_key for pi in result.projection_intents}
        assert keys == {"projection_a", "projection_b"}
        # event_type carries the folded message type.
        assert all(
            pi.event_type == "ModelReducerAdvance" for pi in result.projection_intents
        )
        # The projection model itself is carried as the intent envelope.
        envelopes = {type(pi.envelope) for pi in result.projection_intents}
        assert envelopes == {ModelProjectionA, ModelProjectionB}

    def test_for_reducer_projections_classified_regardless_of_node_kind(self) -> None:
        """A ModelHandlerOutput self-declares REDUCER; no node_kind hint needed."""
        output = ModelHandlerOutput.for_reducer(
            input_envelope_id=uuid4(),
            correlation_id=uuid4(),
            handler_id="test-reducer",
            projections=(ModelProjectionA(value="y"),),
        )
        # handler_node_kind intentionally omitted (None).
        result = _normalize_handler_result(output, _envelope(), "Evt")
        assert result is not None
        assert len(result.projection_intents) == 1
        assert result.output_events == []


@pytest.mark.unit
class TestReducerBareAndSequenceReturns:
    def test_bare_reducer_model_becomes_projection_intent(self) -> None:
        """A def-B REDUCER bare typed model -> a single projection_intent."""
        result = _normalize_handler_result(
            ModelProjectionA(value="z"),
            _envelope(),
            "ModelReducerAdvance",
            EnumNodeKind.REDUCER,
        )
        assert result is not None
        assert len(result.projection_intents) == 1
        assert result.output_events == []
        assert result.output_count == 1
        assert result.projection_intents[0].projector_key == "projection_a"

    def test_reducer_sequence_return_becomes_projection_intents(self) -> None:
        """A def-B REDUCER multi-projection Sequence -> N projection_intents."""
        result = _normalize_handler_result(
            (ModelProjectionA(value="a"), ModelProjectionB(count=1)),
            _envelope(),
            "ModelReducerAdvance",
            EnumNodeKind.REDUCER,
        )
        assert result is not None
        assert len(result.projection_intents) == 2
        assert result.output_events == []
        assert result.output_count == 2

    def test_reducer_empty_sequence_yields_no_projections(self) -> None:
        """An empty no-op fold Sequence yields SUCCESS with no outputs."""
        result = _normalize_handler_result(
            [],
            _envelope(),
            "ModelReducerAdvance",
            EnumNodeKind.REDUCER,
        )
        assert result is not None
        assert result.projection_intents == ()
        assert result.output_events == []
        assert result.output_count == 0


@pytest.mark.unit
class TestNonReducerReturnsUnchanged:
    def test_bare_model_without_node_kind_stays_event(self) -> None:
        """Backward-compat: no node_kind hint -> bare model classified as EVENT."""
        result = _normalize_handler_result(
            ModelProjectionA(value="e"), _envelope(), "Evt"
        )
        assert result is not None
        assert result.output_events == [ModelProjectionA(value="e")]
        assert result.projection_intents == ()

    def test_effect_bare_model_stays_event(self) -> None:
        """An EFFECT bare model return stays an EVENT, not a projection."""
        result = _normalize_handler_result(
            ModelProjectionA(value="f"),
            _envelope(),
            "Evt",
            EnumNodeKind.EFFECT,
        )
        assert result is not None
        assert len(result.output_events) == 1
        assert result.projection_intents == ()


@pytest.mark.unit
class TestNodeKindAndProjectorKeyHelpers:
    def test_node_kind_from_node_type_mapping(self) -> None:
        assert _node_kind_from_node_type("REDUCER_GENERIC") is EnumNodeKind.REDUCER
        assert _node_kind_from_node_type("EFFECT_GENERIC") is EnumNodeKind.EFFECT
        assert _node_kind_from_node_type("COMPUTE_GENERIC") is EnumNodeKind.COMPUTE
        assert (
            _node_kind_from_node_type("ORCHESTRATOR_GENERIC")
            is EnumNodeKind.ORCHESTRATOR
        )
        assert _node_kind_from_node_type("reducer_generic") is EnumNodeKind.REDUCER
        assert _node_kind_from_node_type("SOMETHING_ELSE") is None
        assert _node_kind_from_node_type(None) is None
        assert _node_kind_from_node_type("") is None

    def test_derive_projector_key_strips_model_and_snake_cases(self) -> None:
        assert _derive_projector_key(ModelProjectionA(value="q")) == "projection_a"

    def test_coerce_projection_models(self) -> None:
        single = ModelProjectionA(value="s")
        assert _coerce_projection_models(single) == (single,)
        seq = (single, ModelProjectionB(count=2))
        assert _coerce_projection_models(seq) == seq
        # Non-model / non-sequence returns nothing.
        assert _coerce_projection_models("not-a-model") == ()
        assert _coerce_projection_models(42) == ()

    def test_build_projection_intents_falls_back_to_class_name_event_type(self) -> None:
        corr = uuid4()
        intents = _build_projection_intents((ModelProjectionA(value="c"),), corr, None)
        assert len(intents) == 1
        assert intents[0].event_type == "ModelProjectionA"
        assert intents[0].correlation_id == corr


@pytest.mark.unit
class TestFsmReducerCrossBoundary:
    async def test_fsm_reducer_output_flows_through_adapter(self) -> None:
        """Cross-boundary regression: the REAL HandlerCodingAgentFsm's two
        ``for_reducer`` projections now reach ``projection_intents`` through the
        dispatch adapter instead of being silently dropped (OMN-14598).
        """
        from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
        from omnibase_infra.models.coding_agent.enum_agent_sandbox import (
            EnumAgentSandbox,
        )
        from omnibase_infra.models.coding_agent.enum_coding_agent import EnumCodingAgent
        from omnibase_infra.models.coding_agent.enum_coding_agent_event_kind import (
            EnumCodingAgentEventKind,
        )
        from omnibase_infra.models.coding_agent.enum_coding_agent_fsm_state import (
            EnumCodingAgentFsmState,
        )
        from omnibase_infra.models.coding_agent.model_coding_agent_event import (
            ModelCodingAgentEvent,
        )
        from omnibase_infra.models.coding_agent.model_coding_agent_fsm_state import (
            ModelCodingAgentFsmState,
        )
        from omnibase_infra.nodes.node_coding_agent_fsm_reducer.handlers.handler_coding_agent_fsm import (
            HandlerCodingAgentFsm,
        )

        corr = uuid4()
        handler = HandlerCodingAgentFsm()
        payload = {
            "state": ModelCodingAgentFsmState(
                correlation_id=corr,
                agent=EnumCodingAgent.CLAUDE,
                sandbox=EnumAgentSandbox.READ_ONLY,
                current_state=EnumCodingAgentFsmState.IDLE,
            ).model_dump(mode="json"),
            "event": ModelCodingAgentEvent(
                correlation_id=corr,
                kind=EnumCodingAgentEventKind.INVOKE_REQUESTED,
            ).model_dump(mode="json"),
        }
        envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
            payload=payload,
            correlation_id=corr,
            event_type="onex.evt.omnibase-infra.coding-agent-fsm-advance.v1",
        )

        output = await handler.handle(envelope)
        # The handler emits exactly two reducer projections (state + trace).
        assert len(output.projections) == 2

        # Feed the real handler output through the adapter as the runtime would.
        result = _normalize_handler_result(
            output, envelope, "ModelCodingAgentFsmAdvance", EnumNodeKind.REDUCER
        )
        assert result is not None
        # Previously ZERO; now both projections are carried as projection_intents.
        assert len(result.projection_intents) == 2
        assert result.output_events == []
        keys = {pi.projector_key for pi in result.projection_intents}
        assert keys == {"coding_agent_fsm_state", "coding_agent_trace_projection"}
