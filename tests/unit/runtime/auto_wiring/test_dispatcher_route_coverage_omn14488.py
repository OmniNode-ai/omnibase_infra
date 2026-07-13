# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-14488 (child of OMN-14208) — route-coverage invariant for type-scoped dispatchers.

INVARIANT: a dispatcher registered with a `payload_type_matcher` (built from a
contract's declared handler `event_model`, OMN-12416) MUST resolve for a **valid
instance of that same event_model** — i.e. registration alone is not enough, the
message must actually ROUTE. A registered-but-unroutable dispatcher (valid payload
still dropped) is a wiring defect and must fail this gate.

WHY THIS GATE EXISTS: the dispatch path (`_find_matching_dispatchers`,
`message_dispatch_engine.py`) drops a type-scoped dispatcher whenever the payload
neither `isinstance`s nor `model_validate`s against its declared `event_model`.
The only prior coverage (`test_plugin_managed_dispatchers_still_registered`)
asserted a dispatcher is *registered* but never dispatched a payload — so a
registered-but-unroutable dispatcher passed green while the live runtime DLQ'd
(`No dispatcher found for category '…' and message type '…'`). This is the exact
trap that hid the OMN-14484 investigation for a full arc. (OMN-14484's live DLQ
itself turned out to be a *harness* artifact — a synthetic publisher used the wrong
model — not a product bug; the real producers emit valid payloads that dispatch.
This gate captures the reusable lesson, not that harness case.)

This is the SELF-CONTAINED unit gate. Generalizing it to iterate every
`payload_type_match` contract (constructing a valid instance of each declared
event_model) is a follow-up; the cross-boundary producer↔consumer register is
owned separately by the seam-reaping lane.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_core.enums import EnumMessageCategory
from omnibase_core.models.delegation.wire import ModelDelegationRequest
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

_COMMAND_TOPIC = "onex.cmd.omnibase-infra.delegation-request.v1"
_WIRE_EVENT_MODEL_MODULE = "omnibase_core.models.delegation.wire"


class _FakeHandler:
    async def handle(self, envelope: object) -> None:
        return None


def _plugin_managed_command_contract() -> ModelDiscoveredContract:
    """A plugin_managed COMMAND contract whose sole handler declares a real wire
    `event_model` — the delegation-orchestrator shape (auto-wiring owns the
    dispatch route; plugin owns the subscription). A concrete vehicle for the
    general route-coverage invariant."""
    return ModelDiscoveredContract(
        name="node_delegation_orchestrator",
        node_type="ORCHESTRATOR_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/contract.yaml"),
        entry_point_name="node_delegation_orchestrator",
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(_COMMAND_TOPIC,),
            publish_topics=(),
            plugin_managed=True,
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerDelegationWorkflow", module="fake.module"
                    ),
                    event_model=ModelHandlerRef(
                        name="ModelDelegationRequest",
                        module=_WIRE_EVENT_MODEL_MODULE,
                    ),
                    message_category="command",
                    event_type="omnibase-infra.delegation-request",
                    operation="delegation.orchestrate",
                ),
            ),
        ),
    )


async def _wire() -> MessageDispatchEngine:
    contract = _plugin_managed_command_contract()
    manifest = ModelAutoWiringManifest(contracts=(contract,))
    engine = MessageDispatchEngine()
    event_bus = MagicMock(spec=ProtocolEventBusLike)
    event_bus.subscribe = AsyncMock(return_value=AsyncMock())
    # Patch ONLY the handler import — the event_model (wire ModelDelegationRequest)
    # resolves for real via _import_event_model_class, so the payload_type_matcher
    # is the true production matcher.
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=_FakeHandler,
    ):
        report = await wire_from_manifest(
            manifest, engine, event_bus=event_bus, environment="local"
        )
    result = next(
        r for r in report.results if r.contract_name == "node_delegation_orchestrator"
    )
    # Registration precondition: the bug this gate guards is NOT missing wiring.
    assert result.outcome.value == "wired"
    assert len(result.dispatchers_registered) == 1
    assert len(result.routes_registered) == 1
    return engine


@pytest.mark.unit
@pytest.mark.asyncio
async def test_valid_event_model_instance_routes_to_dispatcher() -> None:
    """LOAD-BEARING invariant: a valid instance of the declared event_model MUST
    resolve the registered dispatcher. Registered-but-unroutable (a valid payload
    still dropped by type-scoping / route misconfig) fails here — the coverage the
    registration-only test lacked."""
    engine = await _wire()
    valid = {
        "correlation_id": "00000000-0000-0000-0000-000000000001",
        "emitted_at": "2026-07-12T22:13:56.030240Z",
        "prompt": "do the thing",
        "task_type": "code_generation",
    }
    # Precondition: the payload really is a valid instance of the declared model.
    ModelDelegationRequest.model_validate(valid)
    matches = engine._find_matching_dispatchers(
        topic=_COMMAND_TOPIC,
        category=EnumMessageCategory.COMMAND,
        message_type="ModelDelegationRequest",
        payload=valid,
    )
    assert len(matches) == 1, (
        "A valid instance of the declared event_model MUST resolve the registered "
        "command dispatcher (route coverage) — registered-but-unroutable is a defect."
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_nonconforming_payload_is_dropped_by_type_scoping() -> None:
    """Discriminator: a payload that does NOT conform to the declared event_model is
    dropped by OMN-12416 type-scoping (the matcher must actually discriminate, not
    accept everything) — otherwise the route-coverage assertion above would be
    vacuous. Uses a generic non-conforming payload (not any specific live capture)."""
    engine = await _wire()
    nonconforming = {"not_a": "delegation request", "missing": "everything"}
    # Precondition: this payload is genuinely not a valid instance of the model.
    with pytest.raises(Exception):
        ModelDelegationRequest.model_validate(nonconforming)
    matches = engine._find_matching_dispatchers(
        topic=_COMMAND_TOPIC,
        category=EnumMessageCategory.COMMAND,
        message_type="ModelDelegationRequest",
        payload=nonconforming,
    )
    assert matches == [], (
        "A payload that fails the declared event_model must be type-scoping-dropped "
        "(no dispatcher) — proving the matcher discriminates."
    )
