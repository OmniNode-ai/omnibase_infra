# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The topic-migration executor is reachable through the runtime's own entry point (OMN-21050).

The executor node was PROVEN-IN-TEST only: its handler was exercised by calling
``execute()`` directly, a method the runtime never calls. On the lab dev lane the
node was loaded and its consumer group Stable, yet one DUAL_WRITE command for a
synthetic topic pair never became a lifecycle event: every delivery raised
``Auto-wired handler HandlerTopicMigrationExecutor does not expose a callable
handle() or handle_async() dispatch entrypoint`` and was retried into a failure
terminal on the lifecycle topic.

These tests load the REAL ``contract.yaml`` from disk, wire the REAL handler class
through ``_prepare_handler_wiring`` (the same call the kernel makes), and drive the
resulting dispatcher with the materialized envelope shape the dispatch engine hands
it. Only the two infrastructure collaborators (topic provisioner, lag-observing
drain gate) are replaced, because a unit test has no broker.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest

from omnibase_core.enums.enum_cutover_criterion import EnumCutoverCriterion
from omnibase_core.enums.enum_migration_phase import EnumMigrationPhase
from omnibase_core.models.contracts.model_topic_migration_contract import (
    ModelTopicMigrationContract,
)
from omnibase_core.models.contracts.model_topic_schema_binding import (
    ModelTopicSchemaBinding,
)
from omnibase_core.models.errors import ModelOnexError
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_core.services.service_handler_resolver import ServiceHandlerResolver
from omnibase_core.services.service_local_handler_ownership_query import (
    ServiceLocalHandlerOwnershipQuery,
)
from omnibase_infra.migration.service_drain_proof_gate import (
    ModelDrainProofDecision,
    ServiceDrainProofGate,
)
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.nodes.node_topic_migration_executor_effect.models.model_topic_migration_command import (
    ModelTopicMigrationCommand,
)
from omnibase_infra.nodes.node_topic_migration_executor_effect.models.model_topic_migration_lifecycle_event import (
    ModelTopicMigrationLifecycleEvent,
)
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import _prepare_handler_wiring
from omnibase_infra.runtime.service_dispatch_result_applier import (
    build_contract_result_applier,
)
from omnibase_infra.topics.model_topic_spec import ModelTopicSpec

pytestmark = pytest.mark.unit

_CONTRACT_PATH = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_topic_migration_executor_effect"
    / "contract.yaml"
)
_COMMAND_TOPIC = "onex.cmd.omnibase-infra.topic-migration-execute.v1"
_LIFECYCLE_TOPIC = "onex.evt.omnibase-infra.topic-migration-lifecycle.v1"
_OLD_TOPIC = "onex.evt.labproof.omn21050-old.v1"
_NEW_TOPIC = "onex.evt.labproof.omn21050-new.v1"
_PATCH_DEPS = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring."
    "_build_topic_migration_executor_dependencies"
)


class _RecordingProvisioner:
    def __init__(self) -> None:
        self.specs: list[ModelTopicSpec | None] = []

    async def ensure_topic_exists(
        self,
        topic_name: str,
        config: object | None = None,
        correlation_id: UUID | None = None,
        *,
        spec: ModelTopicSpec | None = None,
    ) -> bool:
        self.specs.append(spec)
        return True


class _FakeGate(ServiceDrainProofGate):
    def __init__(self, *, allowed: bool, residual: int) -> None:
        self._allowed = allowed
        self._residual = residual

    async def evaluate(
        self, contract: ModelTopicMigrationContract
    ) -> ModelDrainProofDecision:
        return ModelDrainProofDecision(
            migration_ticket=contract.ticket,
            old_consumer_group=contract.old_consumer_group,
            old_topic=contract.old_binding.topic,
            retirement_allowed=self._allowed,
            residual_lag=self._residual,
            reason="fake gate decision",
        )


def _binding(topic: str) -> ModelTopicSchemaBinding:
    return ModelTopicSchemaBinding(
        topic=topic,
        event_name="OMN21050_PROOF",
        schema_version=ModelSemVer(major=1, minor=0, patch=0),
    )


def _command(
    *, phase: EnumMigrationPhase, target: EnumMigrationPhase
) -> ModelTopicMigrationCommand:
    return ModelTopicMigrationCommand(
        correlation_id=uuid4(),
        contract=ModelTopicMigrationContract(
            contract_version=ModelSemVer(major=1, minor=0, patch=0),
            ticket="OMN-21050",
            old_binding=_binding(_OLD_TOPIC),
            new_binding=_binding(_NEW_TOPIC),
            old_consumer_group="local.labproof.omn21050-old.consume.v1",
            new_consumer_group="local.labproof.omn21050-new.consume.v1",
            compatibility_window_hours=1,
            cutover_criteria=(EnumCutoverCriterion.OLD_TOPIC_DRAINED,),
            phase=phase,
        ),
        target_phase=target,
        new_topic_partitions=1,
    )


def _wire(
    *, provisioner: _RecordingProvisioner, gate: _FakeGate
) -> Callable[[object], Awaitable[object]]:
    """Prepare the real contract's handler exactly as the kernel does."""
    manifest = discover_contracts_from_paths([_CONTRACT_PATH])
    assert not manifest.errors, manifest.errors
    contract = manifest.contracts[0]
    assert contract.handler_routing is not None
    entry = contract.handler_routing.handlers[0]
    with patch(
        _PATCH_DEPS,
        return_value={"provisioner": provisioner, "drain_proof_gate": gate},
    ):
        prepared = _prepare_handler_wiring(
            contract=contract,
            entry=entry,
            dispatch_engine=None,
            resolver=ServiceHandlerResolver(),
            ownership_query=ServiceLocalHandlerOwnershipQuery(
                local_node_names=frozenset({contract.name})
            ),
            event_bus=None,
            container=None,
        )
    assert prepared.is_skip is False
    assert prepared.dispatcher is not None
    return cast("Callable[[object], Awaitable[object]]", prepared.dispatcher)


def _materialized_dispatch(command: ModelTopicMigrationCommand) -> dict[str, object]:
    """The JSON-safe materialization ``MessageDispatchEngine`` hands a dispatcher."""
    return {
        "payload": {
            "payload": command.model_dump(mode="json"),
            "event_type": "execute_topic_migration",
            "correlation_id": str(command.correlation_id),
        },
        "__bindings": {},
        "__debug_trace": {
            "topic": _COMMAND_TOPIC,
            "correlation_id": str(command.correlation_id),
        },
    }


async def _dispatch(
    dispatcher: Callable[[object], Awaitable[object]],
    command: ModelTopicMigrationCommand,
) -> tuple[Exception | None, object]:
    """Run the dispatcher; return (raised error, result) so a failure is an assertion."""
    try:
        return None, await dispatcher(_materialized_dispatch(command))
    except (ValueError, ModelOnexError) as exc:
        return exc, None


@pytest.mark.asyncio
async def test_dual_write_command_reaches_a_dual_write_lifecycle_event() -> None:
    """RED before the fix: the dispatcher raised ``does not expose a callable handle()``."""
    provisioner = _RecordingProvisioner()
    dispatcher = _wire(
        provisioner=provisioner, gate=_FakeGate(allowed=True, residual=0)
    )
    command = _command(
        phase=EnumMigrationPhase.PLANNED, target=EnumMigrationPhase.DUAL_WRITE
    )

    error, result = await _dispatch(dispatcher, command)

    assert error is None, f"dispatch raised {error!r}"
    events = [
        event
        for event in getattr(result, "output_events", ())
        if isinstance(event, ModelTopicMigrationLifecycleEvent)
    ]
    assert len(events) == 1, f"expected one lifecycle event, got {result!r}"
    event = events[0]
    assert event.phase is EnumMigrationPhase.DUAL_WRITE
    assert event.correlation_id == command.correlation_id
    assert event.new_topic == _NEW_TOPIC
    assert event.new_topic_provisioned is True
    assert [spec.suffix for spec in provisioner.specs if spec is not None] == [
        _NEW_TOPIC
    ]

    # The kernel's applier publishes that event to the contract's lifecycle topic.
    bus = AsyncMock(spec=ProtocolEventBusLike)
    applier = build_contract_result_applier(
        event_bus=bus,
        contract_path=_CONTRACT_PATH,
        publish_topics=[_LIFECYCLE_TOPIC],
    )
    assert isinstance(result, ModelDispatchResult)
    await applier.apply(result, command.correlation_id)
    bus.publish_envelope.assert_awaited_once()
    published = bus.publish_envelope.await_args.kwargs
    assert published["topic"] == _LIFECYCLE_TOPIC
    assert published["envelope"].payload == event


@pytest.mark.asyncio
async def test_cutover_with_residual_lag_emits_no_cutover_lifecycle_event() -> None:
    """The drain-proof gate holds through the runtime entry point (AC3).

    A CUTOVER while the old group still has lag must not produce a lifecycle event
    for phase CUTOVER. The dispatcher refuses it (the kernel turns the raise into a
    failure terminal), it never returns an event that claims the cutover happened.
    """
    dispatcher = _wire(
        provisioner=_RecordingProvisioner(), gate=_FakeGate(allowed=False, residual=4)
    )
    command = _command(
        phase=EnumMigrationPhase.DUAL_READ, target=EnumMigrationPhase.CUTOVER
    )

    error, result = await _dispatch(dispatcher, command)

    assert isinstance(error, ValueError), f"dispatch raised {error!r}"
    assert "cutover blocked by drain-proof gate" in str(error)
    assert not [
        event
        for event in getattr(result, "output_events", ())
        if isinstance(event, ModelTopicMigrationLifecycleEvent)
        and event.phase is EnumMigrationPhase.CUTOVER
    ]
