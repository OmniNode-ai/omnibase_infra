# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Event Chain Gate — whole chains driven through the REAL dispatch seam (OMN-16774).

Why this suite exists.

OMN-16767 killed the entire delegation chain on the .201 dev lane and **every
repo's CI stayed green**. ``HandlerRoutingIntent`` was handed the projection
arm's raw ``input_data`` dict instead of a validated ``ModelRoutingIntent``, so
it raised ``AttributeError: 'dict' object has no attribute 'payload'`` on its
first dereference, the projection arm swallowed the exception, and every routing
request went to the platform quarantine sink. Zero routing decisions were
produced; every delegation timed out.

Nothing caught it because **nothing drives the real dispatch path per PR**:

* Handler-isolation tests construct the handler and call ``handle()`` directly.
  They never see wiring, so the arm selection that broke is invisible to them.
* ``tests/unit/nodes/node_coding_agent/test_real_dispatch_multitopic_routing.py``
  is the closest existing "real dispatch" suite, but it imports
  ``_make_dispatch_callback`` **directly** — hand-building the callback and
  thereby bypassing ``_prepare_handler_wiring``, which is the exact function the
  defect lived in. That is why it was green throughout the outage.
* omnimarket's ``delegation-regression-nightly.yml`` does drive a real chain, but
  it is nightly and needs a live Redpanda + Postgres lane.

Doctrine rule 5: detection that is not a pre-merge gate is advisory and gets
ignored. This suite is wired as the ``Event Chain Gate`` CI job and registered in
``scripts/ci/ci_summary_gate.py::STRICT_GATE_JOBS``, so it blocks merge.

What "real" means here.

Every hop below is the production object, driven with zero infrastructure:

1. ``EventBusInmemory`` — the default local transport, real publish/subscribe.
2. Raw JSON **bytes** published to the entry topic, exactly the shape Kafka
   delivers. The test never hands the runtime a pre-built typed model.
3. ``EventBusSubcontractWiring.wire_subscriptions`` — the real consumer, whose
   real callback runs the real ``_deserialize_to_envelope``.
4. ``_prepare_handler_wiring`` — the real **arm selection**. This is the seam
   OMN-16767 lived in and the reason this suite exists.
5. ``MessageDispatchEngine.dispatch`` — the real routing/materialization.
6. The real handler.
7. ``DispatchResultApplier.apply`` — the real terminal publish.

The assertions are the two the outage needed and nobody was making: **the
terminal event actually lands on the bus**, and **nothing lands in the
quarantine/DLQ path**. A chain that silently dies into quarantine fails here.

Adding a chain.

Append a ``ChainCase`` row to ``CHAIN_CASES``. Anything that can be wired from a
contract and reached in-process belongs here; a chain whose handlers live in
omnimarket is gated in that repo's mirror of this suite, against the dispatch
seam it owns.

Mirrors.

``omnibase_infra`` cannot import ``omnimarket`` (a downstream package), so the
delegation row mirrors ``ModelRoutingIntent`` / ``HandlerRoutingIntent``
field-for-field, following the precedent set by the OMN-16767 fix lane's seam
test. Everything the mirror touches — wiring, engine, bus, consumer, applier —
is production code. The real omnimarket handlers are driven by the omnimarket
half of this gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ConfigDict

from omnibase_core.models.contracts.subcontracts.model_db_ownership_subcontract import (
    ModelDbOwnershipSubcontract,
)
from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_core.models.contracts.subcontracts.model_event_bus_subcontract import (
    ModelEventBusSubcontract,
)
from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_core.protocols.event_bus.protocol_event_bus_subscriber import (
    ProtocolEventBusSubscriber,
)
from omnibase_core.services.service_handler_resolver import ServiceHandlerResolver
from omnibase_core.services.service_local_handler_ownership_query import (
    ServiceLocalHandlerOwnershipQuery,
)
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    PreparedWiring,
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
from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    EventBusSubcontractWiring,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.runtime.service_dispatch_result_applier import DispatchResultApplier
from omnibase_spi.protocols.runtime import ProtocolDispatchEngine
from tests.helpers.application_db_topology import application_topology

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_THIS_MODULE = "tests.integration.chains.test_event_chain_gate"

# The platform quarantine sink. A chain that dies silently ends up here, which is
# exactly how OMN-16767 hid for weeks behind green CI.
QUARANTINE_TOPIC = "onex.dlq.omnibase-infra.quarantine.v1"  # onex-topic-allow: asserted-empty sink, never produced to by this suite


# ---------------------------------------------------------------------------
# Delegation chain mirrors (see MIRRORS in the module docstring)
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

    ``payload`` is the field ``HandlerRoutingIntent.handle`` dereferences first,
    and the exact dereference that raised ``AttributeError: 'dict' object has no
    attribute 'payload'`` in the live outage.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    payload: ModelMirrorDelegationRequest
    min_tier_name: str | None = None
    excluded_backend_refs: tuple[str, ...] = ()


class ModelMirrorRoutingDecision(BaseModel):
    """Mirror of the reducer's terminal event."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID
    backend_ref: str
    tier_name: str


class HandlerMirrorRoutingIntent:
    """Mirror of ``HandlerRoutingIntent`` — a canonical def-B handler.

    ``handle(request: ModelX) -> ModelY``: a typed-payload core. The runtime owes
    it a validated ``ModelMirrorRoutingIntent``; handed a raw dict it raises on
    ``intent.payload``, which is the whole defect.
    """

    def handle(self, intent: ModelMirrorRoutingIntent) -> ModelMirrorRoutingDecision:
        return ModelMirrorRoutingDecision(
            correlation_id=intent.payload.correlation_id,
            backend_ref=f"backend-for-{intent.payload.task_type}",
            tier_name=intent.min_tier_name or "local",
        )


# ---------------------------------------------------------------------------
# Chain case definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChainCase:
    """One end-to-end chain driven through the real dispatch seam.

    Attributes:
        chain_id: pytest parameter id; also the node name used for the consumer
            group, so a failure names the chain.
        node_type: contract ``node_type`` (drives archetype-specific wiring).
        entry_topic: the topic the wire message is published to.
        terminal_topic: the topic the terminal event must land on.
        contract_yaml: the on-disk contract. Written to a real file because the
            reducer arm reads ``published_events`` from disk to classify a
            declared-event return (OMN-14794); a no-file fallback would test a
            degraded path.
        handler_cls: the handler class the resolver is made to return.
        handler_name / event_model_name: names wired into the routing entry.
        operation: the routing entry's operation.
        wire_payload: the inner domain payload as JSON-safe primitives, exactly
            as it arrives off the wire.
        terminal_type_name: class name the terminal payload must have.
        db_tables: ``db_io.db_tables`` declarations. A NON-EMPTY value is the
            OMN-16767 shape: governed DB access declared by a node whose handler
            is nonetheless typed def-B. Pre-fix this flipped the wiring to the
            projection arm and killed the chain.
        broken_by: open ticket that makes this row fail TODAY, or ``""`` when the
            row must pass. Set it and the row is marked ``xfail(strict=True)``:
            it does not block this PR, but the moment the cited fix lands the row
            XPASSes and CI goes RED, forcing the marker's removal. Never use this
            to silence a NEW failure — a chain that starts failing is a dead
            chain, which is the entire thing this gate exists to catch. The only
            legitimate value is a ticket whose fix is already in flight.
    """

    chain_id: str
    node_type: str
    entry_topic: str
    terminal_topic: str
    contract_yaml: str
    handler_cls: type
    handler_name: str
    event_model_name: str
    operation: str
    wire_payload: dict[str, object]
    terminal_type_name: str
    db_tables: tuple[ModelDbTableDeclaration, ...] = field(default_factory=tuple)
    broken_by: str = ""

    def as_param(self) -> object:
        """Wrap this row for ``parametrize``, applying ``broken_by`` if set."""
        if not self.broken_by:
            return pytest.param(self, id=self.chain_id)
        return pytest.param(
            self,
            id=self.chain_id,
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    f"{self.broken_by}: chain is dead on dev today. STRICT — this "
                    f"turns RED (XPASS) the moment the fix lands, which is the "
                    f"signal to delete this marker."
                ),
            ),
        )


def _tenant_overlay_table() -> ModelDbTableDeclaration:
    """The exact ``db_io.db_tables`` entry that flipped the arm in production.

    ``node_delegation_routing_reducer`` gained this block for its tenant-overlay
    table — which its handler reads through its own resolver, never through
    ``input_data['_db']`` — and silently changed wiring arms underneath an
    unchanged handler. The trigger was a CONTRACT change, not a runtime one,
    which is precisely why a per-PR chain gate is the control that catches it.
    """
    return ModelDbTableDeclaration(
        name="delegation_routing_tenant_overlay",
        database_ref="application",
        schema="tenant",
        migration="0001_create_delegation_routing_tenant_overlay.sql",
        access="read_write",
        role="tenant_routing_overlay",
    )


_DELEGATION_ENTRY_TOPIC = "onex.cmd.omnibase-infra.delegation-routing-request.v1"  # onex-topic-allow: verbatim from the OMN-16767 incident trace
_DELEGATION_TERMINAL_TOPIC = "onex.evt.omnibase-infra.routing-decision.v1"  # onex-topic-allow: verbatim from the OMN-16767 incident trace

_DELEGATION_CONTRACT_YAML = f"""
name: "node_delegation_routing_reducer_chain_gate"
node_type: "REDUCER_GENERIC"
event_bus:
  subscribe_topics:
    - "{_DELEGATION_ENTRY_TOPIC}"
  publish_topics:
    - "{_DELEGATION_TERMINAL_TOPIC}"
published_events:
  - event_type: "ModelMirrorRoutingDecision"
    topic: "{_DELEGATION_TERMINAL_TOPIC}"
    description: "Routing decision emitted by the delegation reducer."
"""


CHAIN_CASES: tuple[ChainCase, ...] = (
    # ---------------------------------------------------------------------
    # The OMN-16767 chain, in the exact shape that died in production: a typed
    # def-B handler on a contract that ALSO declares db_io.db_tables.
    # ---------------------------------------------------------------------
    ChainCase(
        chain_id="delegation-routing-with-db-io",
        node_type="REDUCER_GENERIC",
        entry_topic=_DELEGATION_ENTRY_TOPIC,
        terminal_topic=_DELEGATION_TERMINAL_TOPIC,
        contract_yaml=_DELEGATION_CONTRACT_YAML,
        handler_cls=HandlerMirrorRoutingIntent,
        handler_name="HandlerMirrorRoutingIntent",
        event_model_name="ModelMirrorRoutingIntent",
        operation="delegation_routing",
        wire_payload={
            "payload": {
                "prompt": "summarize the chain gate",
                "task_type": "summarize",
                "correlation_id": "7a300827-0000-4000-8000-000000000001",
                "max_tokens": 2048,
                "tenant_id": None,
            },
            "min_tier_name": None,
            "excluded_backend_refs": [],
        },
        terminal_type_name="ModelMirrorRoutingDecision",
        db_tables=(_tenant_overlay_table(),),
    ),
    # ---------------------------------------------------------------------
    # Control row: the SAME chain with no db_io. Pre-fix this row passed while
    # the row above failed, which is what localizes a failure to the db_io arm
    # selection rather than to the handler, the bus, or the engine.
    # ---------------------------------------------------------------------
    ChainCase(
        chain_id="delegation-routing-no-db-io",
        node_type="REDUCER_GENERIC",
        entry_topic=_DELEGATION_ENTRY_TOPIC,
        terminal_topic=_DELEGATION_TERMINAL_TOPIC,
        contract_yaml=_DELEGATION_CONTRACT_YAML,
        handler_cls=HandlerMirrorRoutingIntent,
        handler_name="HandlerMirrorRoutingIntent",
        event_model_name="ModelMirrorRoutingIntent",
        operation="delegation_routing",
        wire_payload={
            "payload": {
                "prompt": "summarize the chain gate",
                "task_type": "summarize",
                "correlation_id": "7a300827-0000-4000-8000-000000000002",
                "max_tokens": 2048,
                "tenant_id": None,
            },
            "min_tier_name": None,
            "excluded_backend_refs": [],
        },
        terminal_type_name="ModelMirrorRoutingDecision",
        db_tables=(),
    ),
)


# ---------------------------------------------------------------------------
# Harness — every object below is the production one
# ---------------------------------------------------------------------------


@dataclass
class ChainRun:
    """What one chain execution produced, for assertion by the tests."""

    prepared: PreparedWiring
    terminal_messages: list[bytes]
    quarantine_messages: list[bytes]


def _contract(case: ChainCase, contract_path: Path) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=f"node_{case.chain_id.replace('-', '_')}",
        node_type=case.node_type,
        contract_version=ModelContractVersion(major=0, minor=1, patch=0),
        contract_path=contract_path,
        entry_point_name=f"node_{case.chain_id.replace('-', '_')}",
        package_name="omnibase-infra-chain-gate",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(case.entry_topic,),
            publish_topics=(case.terminal_topic,),
        ),
        db_io=(
            ModelDbOwnershipSubcontract(db_tables=list(case.db_tables))
            if case.db_tables
            else None
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="operation_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name=case.handler_name, module=_THIS_MODULE
                    ),
                    event_model=ModelHandlerRef(
                        name=case.event_model_name, module=_THIS_MODULE
                    ),
                    operation=case.operation,
                ),
            ),
        ),
    )


async def _run_chain(case: ChainCase, tmp_path: Path) -> ChainRun:
    """Drive one chain end to end and capture the terminal + quarantine traffic.

    The ONLY seam that is not production here is ``_import_handler_class``, which
    is patched so the resolver returns this module's handler instead of importing
    a package path. Arm selection, engine, bus, consumer, deserializer, and
    applier are all real.
    """
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(case.contract_yaml, encoding="utf-8")

    contract = _contract(case, contract_path)
    assert contract.handler_routing is not None
    entry = contract.handler_routing.handlers[0]

    # Precondition, not decoration. A row declaring db_tables exists to exercise
    # the OMN-16767 shape (governed DB access on a node whose handler is typed
    # def-B). If the resolved omnibase_core parses that block away, the row would
    # still pass — vacuously, gating nothing. Measured divergence: on the CI
    # shared venv this row passes, while locally the same source selects
    # `_make_projection_dispatch_callback` and the chain dies. Fail loudly on the
    # missing precondition rather than bank a free green.
    if case.db_tables:
        assert contract.db_io is not None and contract.db_io.db_tables, (
            f"[{case.chain_id}] gate precondition not met: the row declares "
            f"db_tables but the parsed contract carries no db_io.db_tables, so "
            f"this row is NOT exercising the OMN-16767 arm-selection shape. "
            f"Fix the fixture or the core pin — do not delete the row."
        )

    bus = EventBusInmemory(environment="chain-gate", group="chain-gate")
    await bus.start()

    terminal_messages: list[bytes] = []
    quarantine_messages: list[bytes] = []

    async def _collect_terminal(message: object) -> None:
        terminal_messages.append(cast("bytes", getattr(message, "value", b"")))

    async def _collect_quarantine(message: object) -> None:
        quarantine_messages.append(cast("bytes", getattr(message, "value", b"")))

    # Watch the terminal topic AND the quarantine sink. A chain that dies
    # silently produces nothing on the former and something on the latter.
    await bus.subscribe(
        case.terminal_topic,
        on_message=_collect_terminal,
        group_id=f"chain-gate-terminal-{case.chain_id}",
    )
    await bus.subscribe(
        QUARANTINE_TOPIC,
        on_message=_collect_quarantine,
        group_id=f"chain-gate-quarantine-{case.chain_id}",
    )

    engine = MessageDispatchEngine()

    # ---- THE SEAM UNDER GATE: real arm selection from the real contract ----
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=case.handler_cls,
    ):
        prepared = _prepare_handler_wiring(
            contract=contract,
            entry=entry,
            dispatch_engine=engine,
            resolver=ServiceHandlerResolver(),
            ownership_query=ServiceLocalHandlerOwnershipQuery(
                local_node_names=frozenset({contract.name})
            ),
            event_bus=bus,
            container=None,
            topology=application_topology(),
        )

    engine.register_dispatcher(
        dispatcher_id=prepared.dispatcher_id,
        dispatcher=prepared.dispatcher,
        category=prepared.category,
        message_types=prepared.message_types,
    )
    engine.register_route(
        ModelDispatchRoute(
            route_id=f"{case.chain_id}-route",
            topic_pattern=case.entry_topic,
            message_category=prepared.category,
            # NOTE: the field is ``handler_id``, not ``dispatcher_id``. The model
            # silently drops unknown keys, so a wrong name here does NOT raise —
            # the route just stops binding to this dispatcher and the chain
            # matches on topic+category alone. Caught by mypy, not at runtime.
            handler_id=prepared.dispatcher_id,
        )
    )
    engine.freeze()

    applier = DispatchResultApplier(
        event_bus=cast("ProtocolEventBusLike", bus),
        output_topic=case.terminal_topic,
        output_topic_map={case.terminal_type_name: case.terminal_topic},
        allowed_output_topics=(case.terminal_topic,),
    )

    wiring = EventBusSubcontractWiring(
        event_bus=cast("ProtocolEventBusSubscriber", bus),
        dispatch_engine=cast("ProtocolDispatchEngine", engine),
        environment="chain-gate",
        node_name=case.chain_id,
        service="omnibase-infra",
        version="v1",
        result_applier=applier,
    )
    await wiring.wire_subscriptions(
        ModelEventBusSubcontract(
            version=ModelSemVer(major=1, minor=0, patch=0),
            subscribe_topics=[case.entry_topic],
            publish_topics=[case.terminal_topic],
        ),
        case.chain_id,
    )

    # ---- The wire message, exactly as Kafka delivers it: raw JSON bytes ----
    envelope_json = {
        "payload": case.wire_payload,
        "event_type": case.entry_topic,
        "correlation_id": str(uuid4()),
        "source_tool": "chain-gate",
    }
    await bus.publish(
        case.entry_topic,
        key=None,
        value=json.dumps(envelope_json).encode("utf-8"),
    )

    await wiring.cleanup()
    await bus.close()

    return ChainRun(
        prepared=prepared,
        terminal_messages=terminal_messages,
        quarantine_messages=quarantine_messages,
    )


@pytest.fixture(autouse=True)
def _projection_arm_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the projection arm a DSN so it can construct.

    Pre-fix, this is what let the WRONG arm build at all on the dev lane. Setting
    it here means a pass is a real pass and not an artifact of the projection arm
    failing to construct for an unrelated reason.
    """
    monkeypatch.setenv("OMNIDASH_ANALYTICS_DB_URL", "postgresql://fixture/omn16774")


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", [c.as_param() for c in CHAIN_CASES])
async def test_chain_reaches_terminal_and_never_quarantines(
    case: ChainCase, tmp_path: Path
) -> None:
    """A chain published as raw wire bytes must terminalize and never quarantine.

    This is the assertion pair the OMN-16767 outage needed and nobody was making.
    The chain died into the quarantine sink and produced no terminal; both halves
    are checked here, on the real seam.
    """
    run = await _run_chain(case, tmp_path)

    assert run.prepared.quarantine_reason is None, (
        f"[{case.chain_id}] wiring quarantined the handler before a single "
        f"message was dispatched: {run.prepared.quarantine_reason} "
        f"({run.prepared.quarantine_detail})"
    )

    assert not run.quarantine_messages, (
        f"[{case.chain_id}] {len(run.quarantine_messages)} message(s) landed in "
        f"{QUARANTINE_TOPIC}. This is the OMN-16767 failure signature: the chain "
        f"died silently into the quarantine sink while CI stayed green."
    )

    assert run.terminal_messages, (
        f"[{case.chain_id}] no terminal event reached {case.terminal_topic}. "
        f"The chain did not complete; a delegation on this chain would hang "
        f"until it timed out."
    )

    terminal = json.loads(run.terminal_messages[0].decode("utf-8"))
    payload = terminal.get("payload")
    assert payload is not None, (
        f"[{case.chain_id}] terminal envelope carried no payload: {terminal!r}"
    )


@pytest.mark.parametrize("case", [c.as_param() for c in CHAIN_CASES])
async def test_typed_def_b_handler_is_never_wired_to_the_projection_arm(
    case: ChainCase, tmp_path: Path
) -> None:
    """A handler declaring a concrete BaseModel input takes the TYPED arm.

    The generalization of OMN-16767: ``db_io`` declares governed DB ACCESS, never
    a dispatch SHAPE. A typed def-B handler cannot read ``input_data`` at all, so
    selecting the projection arm for it is always wrong — and always fatal,
    because the projection arm swallows the resulting ``AttributeError`` and
    quarantines instead of failing loudly.

    Asserted through the wiring's own output rather than by inspecting the
    handler, so it holds for every row including ones added later.
    """
    run = await _run_chain(case, tmp_path)

    # The projection arm always returns None (it owns its own persistence and
    # terminal emission), so a terminal on the bus is positive proof the typed
    # arm ran and its result reached the applier.
    assert run.terminal_messages, (
        f"[{case.chain_id}] the typed dispatch result was swallowed — the "
        f"hallmark of the projection arm having been selected for a typed "
        f"def-B handler (OMN-16767)."
    )
