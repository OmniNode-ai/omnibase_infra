# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""REDUCER declared-published-event dispatch classification (OMN-14794).

Background — the live delegation drop this pins:

OMN-14598 made ``_normalize_handler_result`` classify EVERY bare-model / Sequence
return from a REDUCER (``node_type: REDUCER_GENERIC``) as ``projections[]``. That is
correct for a pure FSM fold, but the delegation routing reducer
(``node_delegation_routing_reducer``) is a REDUCER that ALSO declares a
``published_events`` entry — ``event_type: RoutingDecision`` ->
``onex.evt.omnibase-infra.routing-decision.v1`` — and returns a
``ModelRoutingDecision`` to be PUBLISHED as an event.

Under the un-refined OMN-14598 rule that decision was captured into
``projection_intents`` and NEVER published as an event. The delegation orchestrator's
``handle_routing_decision`` therefore never fired, routing-decision.v1 stayed empty
(high-watermark flat), and the workflow stalled at RECEIVED — the uncommitted
delegation-request offset redelivering every session timeout. This was reproduced and
then fixed LIVE on the stability-test runtime by editing exactly this branch:
excluding the declared-event return from the REDUCER->projection capture advanced the
FSM RECEIVED->ROUTED->COMPLETED and moved the routing-decision.v1 HW by exactly one.

These tests pin the durable generalization of that hotpatch: a REDUCER return whose
model class is a declared published event (short-name membership in the contract's
``published_events`` keys) is routed to ``output_events`` (emitted), while every other
REDUCER return keeps the OMN-14598 projection classification.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest
from pydantic import BaseModel

from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.services.service_handler_resolver import ServiceHandlerResolver
from omnibase_core.services.service_local_handler_ownership_query import (
    ServiceLocalHandlerOwnershipQuery,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _is_declared_published_event_model,
    _normalize_handler_result,
    _prepare_handler_wiring,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)


class ModelDelegationRoutingRequest(BaseModel):
    """Stand-in for the reducer's contract-declared command payload."""

    correlation_id: str
    prompt: str


class ModelRoutingDecision(BaseModel):
    """Stand-in for the delegation reducer's declared published event.

    Its contract maps ``event_type: RoutingDecision`` (the ``Model``-stripped
    short-name) to ``onex.evt.omnibase-infra.routing-decision.v1``.
    """

    selected_agent: str
    confidence_score: float


class ModelDelegationTraceProjection(BaseModel):
    """A genuine reducer projection (NOT in published_events) — stays a projection."""

    correlation_id: str
    note: str


# The short-name key set as ``load_published_events_map`` would return it for the
# reducer's contract: {event_type -> topic}.keys() -> {"RoutingDecision"}.
_PUBLISHED = frozenset({"RoutingDecision"})


class HandlerDelegationRoutingReducer:
    def handle(self, command: ModelDelegationRoutingRequest) -> ModelRoutingDecision:
        return ModelRoutingDecision(
            selected_agent=f"agent-for-{command.prompt}",
            confidence_score=1.0,
        )


def _envelope() -> dict[str, object]:
    """A minimal transport envelope carrying a valid hex correlation_id."""
    return {
        "payload": {"correlation_id": str(uuid4())},
        "partition_key": None,
        "correlation_id": str(uuid4()),
        "event_type": "onex.cmd.omnibase-infra.delegation-routing-request.v1",
        "envelope_id": str(uuid4()),
    }


def _write_reducer_contract_manifest(tmp_path: Path) -> Path:
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(
        "\n".join(
            (
                "name: node_delegation_routing_reducer",
                "node_type: REDUCER_GENERIC",
                "published_events:",
                "  - event_type: RoutingDecision",
                "    topic: onex.evt.omnibase-infra.routing-decision.v1",
            )
        ),
        encoding="utf-8",
    )
    return contract_path


def _reducer_contract(contract_path: Path) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="node_delegation_routing_reducer",
        node_type="REDUCER_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=contract_path,
        entry_point_name="node_delegation_routing_reducer",
        package_name="omnibase_infra",
        event_bus=ModelEventBusWiring(
            subscribe_topics=("onex.cmd.omnibase-infra.delegation-routing-request.v1",),
            publish_topics=("onex.evt.omnibase-infra.routing-decision.v1",),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerDelegationRoutingReducer",
                        module=__name__,
                    ),
                    event_model=ModelHandlerRef(
                        name="ModelDelegationRoutingRequest",
                        module=__name__,
                    ),
                    event_type="omnibase-infra.delegation-routing-request",
                    message_category="COMMAND",
                ),
            ),
        ),
    )


@pytest.mark.unit
class TestReducerDeclaredPublishedEventEmitsEvent:
    @pytest.mark.asyncio
    async def test_prepare_handler_wiring_threads_contract_published_events(
        self, tmp_path: Path
    ) -> None:
        """Production wiring loads the reducer contract map before dispatch.

        This covers the auto-wiring path CodeRabbit called out: a real contract
        manifest supplies ``published_events``, ``_prepare_handler_wiring`` builds
        the runtime dispatcher, and the dispatcher emits the declared reducer
        model instead of capturing it as a projection.
        """
        contract = _reducer_contract(_write_reducer_contract_manifest(tmp_path))
        entry = contract.handler_routing.handlers[0]  # type: ignore[union-attr]
        resolver = ServiceHandlerResolver()
        ownership_query = ServiceLocalHandlerOwnershipQuery(
            local_node_names=frozenset({contract.name})
        )

        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=HandlerDelegationRoutingReducer,
        ):
            prepared = _prepare_handler_wiring(
                contract=contract,
                entry=entry,
                dispatch_engine=None,
                resolver=resolver,
                ownership_query=ownership_query,
                event_bus=object(),
                container=None,
            )

        result = await prepared.dispatcher(
            {
                "payload": {
                    "correlation_id": str(uuid4()),
                    "prompt": "route",
                },
                "partition_key": None,
                "correlation_id": str(uuid4()),
                "event_type": "omnibase-infra.delegation-routing-request",
                "envelope_id": str(uuid4()),
            }
        )

        assert result is not None
        assert len(result.output_events) == 1
        assert isinstance(result.output_events[0], ModelRoutingDecision)
        assert result.output_events[0].selected_agent == "agent-for-route"
        assert result.projection_intents == ()
        assert result.output_count == 1

    def test_reducer_declared_event_without_threading_drops_to_projection(self) -> None:
        """RED / EXISTS-but-WRONG: the pre-fix live drop.

        With no ``published_event_names`` threaded (the un-refined OMN-14598 rule),
        a REDUCER's ``ModelRoutingDecision`` return is captured as a projection and
        NEVER published — ``output_events`` is empty. This is the exact
        resolved-but-never-emitted drop that stalled the delegation FSM at RECEIVED:
        the decision was computed, then dropped to 0 published events.
        """
        decision = ModelRoutingDecision(
            selected_agent="Qwen3.6-35B-A3B", confidence_score=1.0
        )

        result = _normalize_handler_result(
            decision,
            _envelope(),
            "ModelRoutingDecision",
            EnumNodeKind.REDUCER,
            None,  # published_event_names absent -> pre-fix behavior
        )

        assert result is not None
        # The decision was resolved into a projection intent — never emitted.
        assert result.output_events == []
        assert len(result.projection_intents) == 1
        assert result.output_count == 1

    def test_reducer_declared_event_routes_to_output_events(self) -> None:
        """GREEN: the fix — the declared published event IS emitted.

        Threading the contract's ``published_events`` short-names makes the same
        REDUCER return land in ``output_events``, so the applier publishes it to
        routing-decision.v1 and the next command (RECEIVED->ROUTED) fires.
        """
        decision = ModelRoutingDecision(
            selected_agent="Qwen3.6-35B-A3B", confidence_score=1.0
        )

        result = _normalize_handler_result(
            decision,
            _envelope(),
            "ModelRoutingDecision",
            EnumNodeKind.REDUCER,
            _PUBLISHED,  # published_event_names threaded -> post-fix behavior
        )

        assert result is not None
        # The decision is now emitted as an event, NOT captured as a projection.
        assert result.output_events == [decision]
        assert result.projection_intents == ()
        assert result.output_count == 1

    def test_reducer_non_declared_model_still_projects(self) -> None:
        """OMN-14598 preserved: a real reducer projection is NOT reclassified.

        Even with ``published_event_names`` threaded, a REDUCER return whose class
        is NOT a declared published event keeps the projection classification — the
        fix is scoped strictly to declared events.
        """
        projection = ModelDelegationTraceProjection(
            correlation_id=str(uuid4()), note="fold-trace"
        )

        result = _normalize_handler_result(
            projection,
            _envelope(),
            "ModelDelegationTraceProjection",
            EnumNodeKind.REDUCER,
            _PUBLISHED,  # threaded, but this class is not a declared event
        )

        assert result is not None
        assert result.output_events == []
        assert len(result.projection_intents) == 1
        assert result.output_count == 1

    def test_non_reducer_declared_event_stays_event_regardless(self) -> None:
        """A non-REDUCER declared-event return is unaffected — already an event."""
        decision = ModelRoutingDecision(selected_agent="a", confidence_score=0.5)

        result = _normalize_handler_result(
            decision,
            _envelope(),
            "ModelRoutingDecision",
            EnumNodeKind.ORCHESTRATOR,
            _PUBLISHED,
        )

        assert result is not None
        assert result.output_events == [decision]
        assert result.projection_intents == ()


@pytest.mark.unit
class TestIsDeclaredPublishedEventModel:
    def test_matches_model_stripped_short_name(self) -> None:
        decision = ModelRoutingDecision(selected_agent="a", confidence_score=0.5)
        assert _is_declared_published_event_model(decision, _PUBLISHED) is True

    def test_matches_full_class_name(self) -> None:
        decision = ModelRoutingDecision(selected_agent="a", confidence_score=0.5)
        assert (
            _is_declared_published_event_model(
                decision, frozenset({"ModelRoutingDecision"})
            )
            is True
        )

    def test_non_member_model_is_not_declared_event(self) -> None:
        projection = ModelDelegationTraceProjection(correlation_id="c", note="n")
        assert _is_declared_published_event_model(projection, _PUBLISHED) is False

    def test_none_or_empty_set_is_never_declared_event(self) -> None:
        decision = ModelRoutingDecision(selected_agent="a", confidence_score=0.5)
        assert _is_declared_published_event_model(decision, None) is False
        assert _is_declared_published_event_model(decision, frozenset()) is False

    def test_non_basemodel_is_never_declared_event(self) -> None:
        assert (
            _is_declared_published_event_model("RoutingDecision", _PUBLISHED) is False
        )
        assert _is_declared_published_event_model([1, 2], _PUBLISHED) is False
