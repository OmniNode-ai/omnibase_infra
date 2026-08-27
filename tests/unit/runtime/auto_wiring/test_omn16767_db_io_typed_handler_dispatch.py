# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16767 — ``db_io`` must not steal a typed def-B handler into the projection arm.

THE DEFECT. ``_prepare_handler_wiring`` selected the projection dispatch arm on
``contract.db_io.db_tables`` ALONE. But ``db_io`` declares GOVERNED DB ACCESS
(which tables a node touches, under which role); it says nothing about the
handler's dispatch SHAPE. The projection arm's contract is
``handle(input_data: dict[str, object])`` carrying ``_db`` / ``_event_type`` /
``_topic``, so a typed def-B handler routed there was handed the raw
``input_data`` dict and crashed on its first attribute access.

Live on the shared .201 dev lane, 2026-08-27T15:32Z, correlation
``7a300827-0000-4000-8000-000000000001``::

    [ERROR] omnibase_infra.runtime.auto_wiring.handler_wiring:
      Projection handler error: handler=HandlerRoutingIntent
      topic=onex.cmd.omnibase-infra.delegation-routing-request.v1
      error_type=AttributeError error='dict' object has no attribute 'payload'
    [ERROR] ... HandlerRoutingIntent has NO DLQ topic declared in
      contract.event_bus.dlq_topics — routing malformed/erroring event to the
      platform quarantine sink onex.dlq.omnibase-infra.quarantine.v1

No ``ModelRoutingDecision`` was ever produced, so every delegation on the lane
timed out (16 routing requests accepted, zero terminals of any kind).

The trigger was a CONTRACT change, not a runtime one: ``node_delegation_routing_
reducer`` gained a ``db_io`` block for its tenant-overlay table (which the
handler reads through its own resolver, never through ``input_data['_db']``) and
silently changed wiring arms underneath an unchanged handler.

THE FIX under test. A handler declaring a concrete ``BaseModel`` input is a typed
def-B handler and can never be a projection handler — it cannot read
``input_data`` at all. Such a handler takes the TYPED dispatch arm regardless of
``db_io``, and the runtime owes it a validated model.

WHY THIS IS THE REAL SEAM. ``omnibase_infra`` cannot import ``omnimarket`` (a
downstream package), so ``HandlerRoutingIntent`` / ``ModelRoutingIntent`` are
mirrored field-for-field below. Everything else is the production path: the real
``_prepare_handler_wiring`` arm selection, the real shipped topology, the real
dispatcher it returns, and the materialized wire envelope shape the
``MessageDispatchEngine`` hands a dispatcher. A test that called the handler
directly, or that built a callback by hand, would have passed throughout the
outage — the failure lives in wiring, not in the handler.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import patch
from uuid import UUID

import pytest
from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.models.contracts.subcontracts.model_db_ownership_subcontract import (
    ModelDbOwnershipSubcontract,
)
from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_core.models.core.model_deployment_topology import ModelDeploymentTopology
from omnibase_core.services.service_handler_resolver import ServiceHandlerResolver
from omnibase_core.services.service_local_handler_ownership_query import (
    ServiceLocalHandlerOwnershipQuery,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    PreparedWiring,
    _extract_dispatch_payload,
    _make_payload_type_matcher,
    _prepare_handler_wiring,
    _typed_def_b_input_model,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from tests.helpers.application_db_topology import application_topology

_THIS_MODULE = (
    "tests.unit.runtime.auto_wiring.test_omn16767_db_io_typed_handler_dispatch"
)

# The live topic and correlation from the reproduction recorded on OMN-16767.
_TOPIC = "onex.cmd.omnibase-infra.delegation-routing-request.v1"  # onex-topic-allow: verbatim from the live incident trace
_CORRELATION = "7a300827-0000-4000-8000-000000000001"

_PATCH_IMPORT_HANDLER = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class"
)
_PATCH_BUILD_ADAPTER = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring._build_projection_db_adapter"
)


# ---------------------------------------------------------------------------
# Field-for-field mirrors of the omnimarket / omnibase_core production shapes
# ---------------------------------------------------------------------------


class ModelMirrorDelegationRequest(BaseModel):
    """Mirror of ``ModelDelegationRequest`` — the inner domain payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt: str
    task_type: str
    correlation_id: UUID
    max_tokens: int = 2048
    tenant_id: str | None = None


class ModelMirrorRoutingIntent(BaseModel):
    """Mirror of ``omnibase_core.models.delegation.wire.ModelRoutingIntent``.

    Critically it declares its own ``payload`` field, which is what
    ``HandlerRoutingIntent.handle`` dereferences first (``intent.payload
    .task_type``) and therefore what raised ``AttributeError`` when a dict
    arrived instead.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent: str = Field(default="routing_reducer")
    payload: ModelMirrorDelegationRequest
    min_tier_name: str | None = None
    excluded_backend_refs: tuple[str, ...] = Field(default_factory=tuple)


class ModelLenientEnvelopeShaped(BaseModel):
    """Envelope-shaped model that tolerates extras — the unconditional-unwrap trap.

    It VALIDATES a payload carrying an undeclared key, while
    ``_is_registered_input_payload``'s stricter key-containment check refuses to
    claim that same payload — so an unconditional unwrap walks straight past it.
    Module-level because the matcher imports its ``event_model`` by name.
    """

    model_config = ConfigDict(extra="ignore")

    event_type: str
    payload: dict[str, object] = Field(default_factory=dict)


class ModelMirrorRoutingDecision(BaseModel):
    """Mirror of the ``ModelRoutingDecision`` the reducer publishes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    selected_model: str
    correlation_id: UUID


_RECEIVED: list[object] = []


class HandlerMirrorRoutingIntent:
    """Mirror of ``HandlerRoutingIntent``: a typed def-B handler, db_io-declaring.

    Deliberately performs the SAME first action as production —
    ``intent.payload.task_type`` — so a raw dict reproduces the live
    ``AttributeError`` rather than being silently tolerated.
    """

    def handle(self, intent: ModelMirrorRoutingIntent) -> ModelMirrorRoutingDecision:
        _RECEIVED.append(intent)
        return ModelMirrorRoutingDecision(
            selected_model=f"model-for-{intent.payload.task_type}",
            correlation_id=intent.payload.correlation_id,
        )


class HandlerMirrorProjection:
    """A genuine projection handler: consumes the injected ``input_data`` mapping."""

    def handle(self, input_data: dict[str, object]) -> dict[str, object]:
        _RECEIVED.append(dict(input_data))
        return {"rows_upserted": 1}


class HandlerMirrorAsyncTyped:
    """Typed def-B handler whose runtime entrypoint is ``handle_async``."""

    def handle(self, intent: ModelMirrorRoutingIntent) -> ModelMirrorRoutingDecision:
        raise AssertionError("handle_async is the runtime entrypoint, not handle")

    async def handle_async(
        self, intent: ModelMirrorRoutingIntent
    ) -> ModelMirrorRoutingDecision:
        _RECEIVED.append(intent)
        return ModelMirrorRoutingDecision(
            selected_model="async", correlation_id=intent.payload.correlation_id
        )


# ---------------------------------------------------------------------------
# Wire shapes — verbatim structure from the live topic record
# ---------------------------------------------------------------------------


def _routing_intent_wire() -> dict[str, object]:
    """The ModelRoutingIntent as published, per the OMN-16767 trace."""
    return {
        "intent": "routing_reducer",
        "payload": {
            "prompt": "summarize the release notes",
            "task_type": "test",
            "correlation_id": _CORRELATION,
            "max_tokens": 2048,
        },
        "min_tier_name": None,
        "excluded_backend_refs": [],
    }


def _materialized_dispatch(inner: dict[str, object]) -> dict[str, object]:
    """The JSON-safe materialization ``MessageDispatchEngine`` hands a dispatcher."""
    return {
        "payload": {
            "payload": inner,
            "event_type": "omnibase-infra.delegation-routing-request",
            "correlation_id": _CORRELATION,
            "source_tool": "delegation-orchestrator",
        },
        "__bindings": {},
        "__debug_trace": {"topic": _TOPIC, "correlation_id": _CORRELATION},
    }


# ---------------------------------------------------------------------------
# Contract construction — the real db_io block from the live contract
# ---------------------------------------------------------------------------


def _tenant_overlay_table() -> ModelDbTableDeclaration:
    """The exact ``db_io.db_tables`` entry that flipped the arm in production."""
    return ModelDbTableDeclaration(
        name="delegation_routing_tenant_overlay",
        database_ref="application",
        schema="tenant",
        migration="0001_create_delegation_routing_tenant_overlay.sql",
        access="read_write",
        role="tenant_routing_overlay",
    )


# The real contract's ``published_events`` map, which the REDUCER arm reads from
# disk (OMN-14794) to classify a declared-event return as an EVENT rather than a
# projection. Written to a real file so the classification under test is the one
# production performs, not a degraded no-file fallback.
_CONTRACT_YAML = """
name: "node_delegation_routing_reducer_mirror"
node_type: "REDUCER_GENERIC"
event_bus:
  subscribe_topics:
    - "onex.cmd.omnibase-infra.delegation-routing-request.v1"
  publish_topics:
    - "onex.evt.omnibase-infra.routing-decision.v1"
published_events:
  - event_type: "MirrorRoutingDecision"
    topic: "onex.evt.omnibase-infra.routing-decision.v1"
    description: "Routing decision emitted by the mirrored reducer."
"""


@pytest.fixture(scope="module")
def contract_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("omn16767") / "contract.yaml"
    path.write_text(_CONTRACT_YAML)
    return path


def _contract(
    *, with_db_io: bool, contract_path: Path, node_type: str = "REDUCER_GENERIC"
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="node_delegation_routing_reducer_mirror",
        node_type=node_type,
        contract_version=ModelContractVersion(major=0, minor=3, patch=0),
        contract_path=contract_path,
        entry_point_name="node_delegation_routing_reducer_mirror",
        package_name="test-pkg",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(_TOPIC,),
            publish_topics=(),
        ),
        db_io=(
            ModelDbOwnershipSubcontract(db_tables=[_tenant_overlay_table()])
            if with_db_io
            else None
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="operation_match",
            handlers=(_entry(),),
        ),
    )


def _entry(
    *, handler_name: str = "HandlerMirrorRoutingIntent"
) -> ModelHandlerRoutingEntry:
    return ModelHandlerRoutingEntry(
        handler=ModelHandlerRef(name=handler_name, module=_THIS_MODULE),
        event_model=ModelHandlerRef(
            name="ModelMirrorRoutingIntent", module=_THIS_MODULE
        ),
        operation="delegation_routing",
    )


def _prepare(
    handler_cls: type,
    *,
    with_db_io: bool,
    contract_path: Path,
    topology: ModelDeploymentTopology | None = None,
) -> PreparedWiring:
    """Run the REAL wiring seam and return what it prepared."""
    contract = _contract(with_db_io=with_db_io, contract_path=contract_path)
    entry = contract.handler_routing.handlers[0]
    ownership = ServiceLocalHandlerOwnershipQuery(
        local_node_names=frozenset({contract.name})
    )
    with patch(_PATCH_IMPORT_HANDLER, return_value=handler_cls):
        return _prepare_handler_wiring(
            contract=contract,
            entry=entry,
            dispatch_engine=None,
            resolver=ServiceHandlerResolver(),
            ownership_query=ownership,
            event_bus=None,
            container=None,
            topology=topology if topology is not None else application_topology(),
        )


@pytest.fixture(autouse=True)
def _clean_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    _RECEIVED.clear()
    # The projection arm refuses to build without a DSN for every topology
    # binding, so set it — pre-fix this is what let the wrong arm construct at
    # all on the dev lane.
    monkeypatch.setenv("OMNIDASH_ANALYTICS_DB_URL", "postgresql://fixture/omn16767")


# ---------------------------------------------------------------------------
# RED: the live coercion failure, reproduced at the wiring seam
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDbIoDoesNotStealTypedHandlers:
    @pytest.mark.asyncio
    async def test_typed_handler_with_db_io_receives_the_validated_model(
        self, contract_path: Path
    ) -> None:
        """RED before the fix: the handler was handed the projection ``input_data`` dict.

        Pre-fix the dispatcher prepared here was the projection callback, and
        ``handle`` raised ``AttributeError: 'dict' object has no attribute
        'payload'`` — swallowed by the projection arm's ``except Exception``,
        logged, and routed to the quarantine sink. Nothing was ever appended to
        ``_RECEIVED`` as a model.
        """
        prepared = _prepare(
            HandlerMirrorRoutingIntent, with_db_io=True, contract_path=contract_path
        )

        await prepared.dispatcher(
            cast("object", _materialized_dispatch(_routing_intent_wire()))
        )

        assert len(_RECEIVED) == 1, (
            "handler was never reached with a usable argument — the projection "
            "arm swallowed the AttributeError and quarantined the event"
        )
        received = _RECEIVED[0]
        assert isinstance(received, ModelMirrorRoutingIntent), (
            f"runtime handed the handler a {type(received).__name__}, not a "
            "validated ModelMirrorRoutingIntent"
        )
        # The exact dereference that raised in production now succeeds.
        assert received.payload.task_type == "test"
        assert received.payload.correlation_id == UUID(_CORRELATION)
        assert received.min_tier_name is None
        assert received.excluded_backend_refs == ()

    @pytest.mark.asyncio
    async def test_typed_handler_result_is_returned_not_swallowed(
        self, contract_path: Path
    ) -> None:
        """The decision must reach the result applier, not vanish into the DLQ.

        The projection arm always returns ``None`` (it owns its own persistence
        and terminal emission), so even a handler that somehow survived the dict
        would have had its ``ModelRoutingDecision`` dropped — the FSM stall half
        of the outage.
        """
        prepared = _prepare(
            HandlerMirrorRoutingIntent, with_db_io=True, contract_path=contract_path
        )

        result = await prepared.dispatcher(
            cast("object", _materialized_dispatch(_routing_intent_wire()))
        )

        assert result is not None, "typed dispatch result was swallowed"
        events = list(getattr(result, "output_events", ()))
        assert [type(event).__name__ for event in events] == [
            "ModelMirrorRoutingDecision"
        ]

    @pytest.mark.asyncio
    async def test_async_typed_handler_with_db_io_also_takes_the_typed_arm(
        self, contract_path: Path
    ) -> None:
        """Arm selection must read the SAME entrypoint the dispatch will invoke.

        A handler whose runtime entrypoint is ``handle_async`` must be
        classified from ``handle_async``'s annotation; reading ``handle``
        instead would reintroduce the defect for every async node handler.
        """
        prepared = _prepare(
            HandlerMirrorAsyncTyped, with_db_io=True, contract_path=contract_path
        )

        await prepared.dispatcher(
            cast("object", _materialized_dispatch(_routing_intent_wire()))
        )

        assert len(_RECEIVED) == 1
        assert isinstance(_RECEIVED[0], ModelMirrorRoutingIntent)

    @pytest.mark.asyncio
    async def test_typed_handler_without_db_io_is_unchanged(
        self, contract_path: Path
    ) -> None:
        """Control: the same handler on a db_io-free contract always worked.

        This is what the 2026-07-30 matrix run exercised. If this ever fails the
        defect is not the arm selection and these tests are measuring the wrong
        thing.
        """
        prepared = _prepare(
            HandlerMirrorRoutingIntent, with_db_io=False, contract_path=contract_path
        )

        await prepared.dispatcher(
            cast("object", _materialized_dispatch(_routing_intent_wire()))
        )

        assert len(_RECEIVED) == 1
        assert isinstance(_RECEIVED[0], ModelMirrorRoutingIntent)


# ---------------------------------------------------------------------------
# The type-scoping predicate must agree with the dispatch it scopes
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPayloadTypeMatcherAgreesWithDispatch:
    """The second defect, found live only after the first fix was deployed.

    Moving the handler onto the typed arm also gives it a
    ``payload_type_matcher`` (the projection arm has none). The engine consults
    that predicate with what it has in hand — on the Kafka path, the TRANSPORT
    ENVELOPE — while the dispatch callback unwraps to the domain payload first.
    Matching on the un-unwrapped envelope makes the two disagree, and the
    disagreement is silent in the worst direction: the engine reads a rejecting
    matcher as "not my message", which is not an error, so the record is
    consumed and COMMITTED with no handler invoked and nothing in any DLQ.

    Measured on the dev lane after deploying the arm fix:
    ``onex.cmd.omnibase-infra.delegation-routing-request.v1`` advanced 22
    records, its consumer group sat Stable at lag 0, and
    ``onex.evt.omnibase-infra.routing-decision.v1`` never moved off 59 — every
    record silently dropped, while each one validated cleanly as
    ``ModelRoutingIntent`` once unwrapped.
    """

    def test_matcher_accepts_the_envelope_wrapped_domain_payload(self) -> None:
        """RED before the fix: the matcher was asked about the envelope and said no."""
        matcher = _make_payload_type_matcher(
            ModelHandlerRef(name="ModelMirrorRoutingIntent", module=_THIS_MODULE)
        )
        envelope = _materialized_dispatch(_routing_intent_wire())

        # Precondition: the domain payload inside genuinely IS the model, so a
        # rejection here can only be an unwrap disagreement, never bad data.
        assert ModelMirrorRoutingIntent.model_validate(_routing_intent_wire())

        assert matcher(envelope) is True

    def test_matcher_still_accepts_the_bare_domain_payload(self) -> None:
        matcher = _make_payload_type_matcher(
            ModelHandlerRef(name="ModelMirrorRoutingIntent", module=_THIS_MODULE)
        )

        assert matcher(_routing_intent_wire()) is True

    def test_matcher_rejects_a_foreign_payload_in_the_same_envelope(self) -> None:
        """Fail-closed: unwrapping must not turn the predicate into a rubber stamp."""
        matcher = _make_payload_type_matcher(
            ModelHandlerRef(name="ModelMirrorRoutingIntent", module=_THIS_MODULE)
        )
        foreign = _materialized_dispatch({"totally": "different", "shape": 1})

        assert matcher(foreign) is False
        assert matcher.last_validation_detail  # type: ignore[attr-defined]

    def test_payload_as_given_is_tried_before_the_unwrapped_candidate(self) -> None:
        """Order matters: this predicate may only ADD acceptances, never remove one.

        Unwrapping unconditionally is not safe. ``_is_registered_input_payload``
        applies a key-containment check that is deliberately stricter than
        validation, so a model tolerating extra keys can VALIDATE a candidate the
        stop condition refuses to claim — the unwrap then walks past a payload
        that already was the model and lands on its inner payload, which does not
        validate. Unconditional unwrapping inverted a live selection from
        ``success`` to ``no_dispatcher`` for ``HandlerA2ATask`` on
        ``onex.cmd.omnibase-infra.remote-agent-invoke.v1``, caught by the
        committed dispatch-selection oracle.
        """

        matcher = _make_payload_type_matcher(
            ModelHandlerRef(name="ModelLenientEnvelopeShaped", module=_THIS_MODULE)
        )
        # Envelope-shaped AND carrying a key the model does not declare, so the
        # stop condition will not claim it even though it validates.
        as_given = {
            "event_type": "omnibase-infra.remote-agent-invoke",
            "payload": {"task": "probe"},
            "source_tool": "parity-probe",
        }
        assert ModelLenientEnvelopeShaped.model_validate(as_given) is not None
        assert (
            _extract_dispatch_payload(as_given, ModelLenientEnvelopeShaped) != as_given
        ), "precondition: the unwrap does walk past this payload"

        assert matcher(as_given) is True

    def test_prepared_wiring_matcher_accepts_what_its_dispatcher_will_receive(
        self, contract_path: Path
    ) -> None:
        """The whole point: the scoping predicate and the dispatch must agree.

        Both come out of the same ``_prepare_handler_wiring`` call, so this
        asserts the property at the seam rather than on a hand-built matcher.
        """
        prepared = _prepare(
            HandlerMirrorRoutingIntent, with_db_io=True, contract_path=contract_path
        )
        envelope = _materialized_dispatch(_routing_intent_wire())

        matcher = prepared.payload_type_matcher
        assert matcher is not None, "typed arm must type-scope its dispatcher"
        assert matcher(envelope) is True


# ---------------------------------------------------------------------------
# Regression: genuine projection handlers keep their arm and their injection
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProjectionHandlersKeepTheProjectionArm:
    @pytest.mark.asyncio
    async def test_projection_shaped_handler_still_gets_db_injection(
        self, contract_path: Path
    ) -> None:
        """The fix must not de-wire the 30+ real projection nodes.

        A projection handler takes ``input_data`` and depends on ``_db`` /
        ``_event_type`` / ``_topic`` being injected. Narrowing the arm on the
        handler's declared input model must leave that path untouched.
        """
        sentinel = object()
        with patch(_PATCH_BUILD_ADAPTER, return_value=sentinel):
            prepared = _prepare(
                HandlerMirrorProjection, with_db_io=True, contract_path=contract_path
            )
            await prepared.dispatcher(
                cast("object", _materialized_dispatch(_routing_intent_wire()))
            )

        assert len(_RECEIVED) == 1
        input_data = cast("dict[str, object]", _RECEIVED[0])
        assert input_data["_db"] is sentinel
        assert input_data["_topic"] == _TOPIC
        # The projection arm's single-level unwrap is unchanged: it merges
        # the transport layer's own keys, not the domain model.
        assert input_data["payload"] == _routing_intent_wire()

    def test_projection_shaped_handler_declares_no_typed_input_model(self) -> None:
        """Ground truth for the discriminator itself."""
        assert _typed_def_b_input_model(HandlerMirrorProjection()) is None


# ---------------------------------------------------------------------------
# The discriminator: fail-closed in both directions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTypedDefBInputModel:
    def test_typed_handler_resolves_its_model(self) -> None:
        assert (
            _typed_def_b_input_model(HandlerMirrorRoutingIntent())
            is ModelMirrorRoutingIntent
        )

    def test_async_entrypoint_wins_over_sync_handle(self) -> None:
        assert (
            _typed_def_b_input_model(HandlerMirrorAsyncTyped())
            is ModelMirrorRoutingIntent
        )

    def test_untyped_object_annotation_is_not_a_typed_handler(self) -> None:
        class _Untyped:
            def handle(self, request: object) -> None:
                return None

        assert _typed_def_b_input_model(_Untyped()) is None

    def test_envelope_annotation_is_not_a_typed_handler(self) -> None:
        """An envelope handler owns its own coercion; it must keep its arm."""
        from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope

        class _EnvelopeHandler:
            def handle(self, envelope: ModelEventEnvelope[object]) -> None:
                return None

        assert _typed_def_b_input_model(_EnvelopeHandler()) is None

    def test_handler_without_an_entrypoint_is_not_a_typed_handler(self) -> None:
        class _NoEntrypoint:
            pass

        assert _typed_def_b_input_model(_NoEntrypoint()) is None
