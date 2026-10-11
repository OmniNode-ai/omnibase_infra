# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration proof for the OMN-17427 rootless-terminal settle shortcut.

On the .201 dev lane, about 270 parentless in-process terminals each spent
the roughly five-second settle budget serially, delaying the chain canary
past its 120-second window. A terminal without a recorded parent cannot
close its declared edge, so the real handler must persist its evidence after
one read through the real wiring, dispatch engine, and in-memory event bus.

The parented, incomplete terminal control must still spend the entire read
budget. It proves the read-once assertion is not vacuous: the harness actually
exercises the settle loop and does not force every dispatch to read once.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, cast
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.handlers.handler_delegation_chain_ledger import (
    HandlerDelegationChainLedger,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models import (
    ModelDeclaredChainHop,
    ModelLedgerChainRow,
    ModelObservedHop,
)
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

_SUBSCRIBE_TOPIC = "onex.evt.omnimarket.delegate-skill-completed.v1"
_CHAIN_TOPICS = (
    "onex.cmd.omnimarket.delegate-skill.v1",
    "onex.cmd.omnibase-infra.delegation-routing-request.v1",
    "onex.evt.omnibase-infra.routing-decision.v1",
    _SUBSCRIBE_TOPIC,
)
_CHAIN = tuple(
    ModelDeclaredChainHop(
        topic=topic,
        parent=None if index == 0 else _CHAIN_TOPICS[index - 1],
    )
    for index, topic in enumerate(_CHAIN_TOPICS)
)
_SETTLE_ATTEMPTS = 5


class HandlerUnderTest(HandlerDelegationChainLedger):
    """Inherit the real handle() and override only its three database edges."""

    observed: ClassVar[tuple[ModelObservedHop, ...]] = ()
    instances: ClassVar[list[HandlerUnderTest]] = []

    def __init__(self) -> None:
        super().__init__(
            cast("Any", object()),
            db_dsn="postgresql://test.invalid/test",
            declared_chain=_CHAIN,
            settle_attempts=_SETTLE_ATTEMPTS,
            settle_delay_seconds=0,
        )
        self.read_count = 0
        self.persisted_rows: list[tuple[ModelLedgerChainRow, ...]] = []
        type(self).instances.append(self)

    async def _ensure_db_ready(self) -> None:
        return None

    async def _read_observed(
        self, correlation_id: UUID
    ) -> tuple[ModelObservedHop, ...]:
        self.read_count += 1
        return type(self).observed

    async def _persist_rows(self, rows: Sequence[ModelLedgerChainRow]) -> None:
        self.persisted_rows.append(tuple(rows))


def _real_contract() -> ModelDiscoveredContract:
    """This node's real shape: bus-triggered effect, no publish_topics."""
    return ModelDiscoveredContract(
        name="node_delegation_chain_ledger_effect",
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=(
            Path(tempfile.gettempdir())
            / "node_delegation_chain_ledger_effect"
            / "contract.yaml"
        ),
        entry_point_name="node_delegation_chain_ledger_effect",
        package_name="omnibase_infra",
        event_bus=ModelEventBusWiring(subscribe_topics=(_SUBSCRIBE_TOPIC,)),
        handler_routing=ModelHandlerRouting(
            routing_strategy="topic_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerUnderTest",
                        module=__name__,
                    ),
                    event_model=ModelHandlerRef(
                        name="ModelDelegationTerminalPayload",
                        module=(
                            "omnibase_infra.nodes."
                            "node_delegation_chain_ledger_effect.models"
                        ),
                    ),
                    topic=_SUBSCRIBE_TOPIC,
                    operation="delegation_chain.record",
                    message_category="event",
                ),
            ),
        ),
    )


async def _dispatch_with_evidence(
    observed: tuple[ModelObservedHop, ...], correlation_id: UUID
) -> HandlerUnderTest:
    """Publish one terminal through the real dispatch path; return its handler."""
    HandlerUnderTest.observed = observed
    HandlerUnderTest.instances = []

    bus = EventBusInmemory(environment="test", group="omn17427-rootless-terminal")
    await bus.start()
    try:
        engine = MessageDispatchEngine()
        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=HandlerUnderTest,
        ):
            await wire_from_manifest(
                ModelAutoWiringManifest(contracts=(_real_contract(),)),
                engine,
                event_bus=bus,
                environment="local",
            )
        engine.freeze()

        envelope = ModelEventEnvelope[object](
            payload={"correlation_id": str(correlation_id), "status": "completed"},
            correlation_id=correlation_id,
            event_type="omnimarket.delegate-skill-completed",
        )
        # Subscriber callbacks are awaited inline, so publish synchronizes dispatch.
        await bus.publish(
            _SUBSCRIBE_TOPIC,
            None,
            envelope.model_dump_json().encode("utf-8"),
            None,
        )
    finally:
        await bus.close()

    assert len(HandlerUnderTest.instances) == 1, (
        "the wiring did not construct exactly one handler, so nothing was "
        f"dispatched: {HandlerUnderTest.instances!r}"
    )
    return HandlerUnderTest.instances[0]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rootless_terminal_is_read_once_through_the_real_wiring() -> None:
    """A rootless terminal persists incomplete evidence after the first read."""
    correlation_id = uuid4()
    observed = (
        ModelObservedHop(
            topic=_SUBSCRIBE_TOPIC,
            envelope_id=uuid4(),
            parent_envelope_id=None,
            correlation_id=correlation_id,
        ),
    )

    handler = await _dispatch_with_evidence(observed, correlation_id)

    assert handler.read_count == 1
    assert len(handler.persisted_rows) == 1
    rows = handler.persisted_rows[0]
    assert rows
    assert any(row.hop == _SUBSCRIBE_TOPIC for row in rows)
    assert not all(row.replay_green for row in rows)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_parented_incomplete_terminal_keeps_its_settle_budget() -> None:
    """Missing middle hops still consume the read budget for a parented terminal."""
    correlation_id = uuid4()
    head_envelope_id = uuid4()
    observed = (
        ModelObservedHop(
            topic=_CHAIN_TOPICS[0],
            envelope_id=head_envelope_id,
            parent_envelope_id=None,
            correlation_id=correlation_id,
        ),
        ModelObservedHop(
            topic=_SUBSCRIBE_TOPIC,
            envelope_id=uuid4(),
            parent_envelope_id=head_envelope_id,
            correlation_id=correlation_id,
        ),
    )

    handler = await _dispatch_with_evidence(observed, correlation_id)

    assert handler.read_count == _SETTLE_ATTEMPTS
    assert len(handler.persisted_rows) == 1
