# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression for OMN-14580: topic_match entries sharing handler + operation
must not collide on one dispatcher ID.

Found live via a genuine cold ``--profile runtime`` bring-up of the .201 dev
lane: ``node_swarm_subtask_state_reducer`` declares ``routing_strategy:
topic_match`` with 5 handler_routing entries, all routing the SAME
``operation: reduce_subtask_state`` to the SAME handler class
(``HandlerSwarmSubtaskState``), differing only by ``topic`` + ``event_model``
(one reducer operation invoked from 5 distinct event sources — a legitimate
contract shape). ``_derive_handler_entry_key`` keyed only on handler name +
operation digest, so all 5 entries produced the identical dispatcher ID;
``_commit_handler_wiring`` calls ``engine.register_dispatcher`` once per entry,
so the 2nd call raised ``ONEX_CORE_064_DUPLICATE_REGISTRATION`` and crashed
``bootstrap()`` — a fatal, boot-crashing defect that only ever surfaces on a
genuine cold bootstrap (a warm targeted-service restart never re-derives/
re-registers from scratch, matching the OMN-14517 visibility pattern).

This drives the actual production path (``wire_from_manifest`` against a real
``MessageDispatchEngine``) with the real 5-entry shape mirrored from
``omnimarket/src/omnimarket/nodes/node_swarm_subtask_state_reducer/contract.yaml``
— not a synthetic single-entry surrogate — so a regression here reproduces the
identical crash the runtime hit.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.auto_wiring import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
    wire_from_manifest,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

pytestmark = pytest.mark.integration

_HANDLER_MODULE = (
    "omnimarket.nodes.node_swarm_subtask_state_reducer.handlers."
    "handler_swarm_subtask_state"
)
_HANDLER_NAME = "HandlerSwarmSubtaskState"
_OPERATION = "reduce_subtask_state"

# (topic, event_model name) pairs copied from the real contract.yaml.
_ENTRIES = (
    ("onex.cmd.omnimarket.delegation-execute.v1", "ModelLlmDelegationCallRequest"),
    (
        "onex.evt.omnimarket.delegation-call-completed.v1",
        "ModelLlmDelegationCompletedEvent",
    ),
    (
        "onex.evt.omnimarket.delegation-escalation-triggered.v1",
        "ModelLlmDelegationEscalationTriggeredEvent",
    ),
    (
        "onex.evt.omnimarket.delegation-all-tiers-failed.v1",
        "ModelLlmDelegationAllTiersFailedEvent",
    ),
    ("onex.evt.omnimarket.swarm-fanout-completed.v1", "ModelSwarmFanoutResult"),
)


class HandlerSwarmSubtaskStateProbe:
    """Zero-arg stand-in — the real handler is also zero-arg constructed."""

    async def handle(self, envelope: object) -> None:
        return None


def _contract() -> ModelDiscoveredContract:
    handlers = tuple(
        ModelHandlerRoutingEntry(
            topic=topic,
            operation=_OPERATION,
            event_model=ModelHandlerRef(
                name=event_model_name, module="omnimarket.models.fake"
            ),
            handler=ModelHandlerRef(name=_HANDLER_NAME, module=_HANDLER_MODULE),
        )
        for topic, event_model_name in _ENTRIES
    )
    return ModelDiscoveredContract(
        name="node_swarm_subtask_state_reducer",
        node_type="REDUCER_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/tmp/omn-14580/contract.yaml"),  # noqa: S108
        entry_point_name="node_swarm_subtask_state_reducer",
        package_name="test-package",
        event_bus=ModelEventBusWiring(
            subscribe_topics=tuple(topic for topic, _ in _ENTRIES),
            publish_topics=(),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="topic_match",
            handlers=handlers,
        ),
    )


@pytest.mark.asyncio
async def test_topic_match_same_operation_wires_five_distinct_dispatchers() -> None:
    """OMN-14580: 5 topic_match entries, same handler + operation, must not
    collide on dispatcher registration."""
    manifest = ModelAutoWiringManifest(contracts=(_contract(),), errors=())
    dispatch_engine = MessageDispatchEngine()
    event_bus = MagicMock(spec=ProtocolEventBusLike)
    event_bus.subscribe = AsyncMock(return_value=AsyncMock())

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=HandlerSwarmSubtaskStateProbe,
    ):
        report = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=dispatch_engine,
            event_bus=event_bus,
            environment="local",
        )

    assert report.total_failed == 0, report.results
    result = report.results[0]
    assert len(result.dispatchers_registered) == 5
    assert len(set(result.dispatchers_registered)) == 5, (
        "all 5 topic_match entries must register distinct dispatcher IDs "
        "(OMN-14580: same handler + operation across 5 topics collided on "
        "one shared ID before the fix)"
    )
    assert len(result.routes_registered) == 5
    assert len(set(result.routes_registered)) == 5
