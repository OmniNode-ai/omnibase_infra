# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler for the topic-migration executor effect (OMN-12623).

Drives a :class:`ModelTopicMigrationContract` forward one phase at a time:

* ``DUAL_WRITE`` provisions the NEW topic (via ``TopicProvisioner`` /
  ``ModelTopicSpec``) and mints the NEW consumer group (via
  ``compute_consumer_group_id``), then emits a lifecycle event.
* ``CUTOVER`` runs the drain-proof gate against the OLD group and refuses to
  advance while residual lag remains.
* every advanced phase emits a :class:`ModelTopicMigrationLifecycleEvent`.

The handler is deterministic given its injected collaborators (provisioner,
drain-proof gate). Phase transitions are validated to be strictly forward — the
executor never moves a migration backward.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import uuid4

from omnibase_core.enums.enum_migration_phase import EnumMigrationPhase
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_node_identity import ModelNodeIdentity
from omnibase_infra.nodes.node_topic_migration_executor_effect.models.model_topic_migration_command import (
    ModelTopicMigrationCommand,
)
from omnibase_infra.nodes.node_topic_migration_executor_effect.models.model_topic_migration_lifecycle_event import (
    ModelTopicMigrationLifecycleEvent,
)
from omnibase_infra.topics.model_topic_spec import ModelTopicSpec
from omnibase_infra.utils.util_consumer_group import compute_consumer_group_id

if TYPE_CHECKING:
    from omnibase_infra.event_bus.service_topic_manager import TopicProvisioner
    from omnibase_infra.migration.service_drain_proof_gate import ServiceDrainProofGate

logger = logging.getLogger(__name__)

# Strict forward ordering of migration phases. The executor only advances along
# this sequence; equal-or-backward target phases are rejected.
_PHASE_ORDER: tuple[EnumMigrationPhase, ...] = (
    EnumMigrationPhase.PLANNED,
    EnumMigrationPhase.DUAL_WRITE,
    EnumMigrationPhase.DUAL_READ,
    EnumMigrationPhase.CUTOVER,
    EnumMigrationPhase.COMPLETE,
)


class HandlerTopicMigrationExecutor:
    """Advances a topic migration one forward phase, emitting a lifecycle event."""

    def __init__(
        self,
        provisioner: TopicProvisioner,
        drain_proof_gate: ServiceDrainProofGate,
        env: str = "dev",
    ) -> None:
        self._provisioner = provisioner
        self._drain_proof_gate = drain_proof_gate
        self._env = env

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def _phase_index(self, phase: EnumMigrationPhase) -> int:
        return _PHASE_ORDER.index(phase)

    def _validate_forward(
        self,
        current: EnumMigrationPhase,
        target: EnumMigrationPhase,
    ) -> None:
        if self._phase_index(target) <= self._phase_index(current):
            raise ValueError(
                f"target_phase {target.value!r} is not strictly forward of the "
                f"contract's current phase {current.value!r}; the executor never "
                "moves a migration backward."
            )

    def _mint_new_group(self, command: ModelTopicMigrationCommand) -> str:
        """Mint the canonical new consumer group from the new topic identity."""
        parsed = command.contract.new_binding.parsed
        identity = ModelNodeIdentity(
            env=self._env,
            service=parsed.service,
            node_name=parsed.event,
            version=f"v{parsed.topic_major}",
        )
        return compute_consumer_group_id(identity)

    async def execute(
        self,
        command: ModelTopicMigrationCommand,
    ) -> ModelTopicMigrationLifecycleEvent:
        """Advance the migration to ``command.target_phase`` and emit the event."""
        contract = command.contract
        self._validate_forward(contract.phase, command.target_phase)

        old_topic = contract.old_binding.topic
        new_topic = contract.new_binding.topic
        old_group = contract.old_consumer_group

        new_provisioned = False
        retirement_allowed = False
        residual_lag = 0
        detail = ""

        if command.target_phase is EnumMigrationPhase.DUAL_WRITE:
            spec = ModelTopicSpec(
                suffix=new_topic,
                partitions=command.new_topic_partitions,
                replication_factor=command.new_topic_replication_factor,
            )
            new_provisioned = await self._provisioner.ensure_topic_exists(spec.suffix)
            detail = (
                f"provisioned new topic {new_topic!r} "
                f"(partitions={command.new_topic_partitions}); "
                f"minted new group {self._mint_new_group(command)!r}; "
                f"dual_publish={command.dual_publish}"
            )

        elif command.target_phase is EnumMigrationPhase.CUTOVER:
            decision = await self._drain_proof_gate.evaluate(contract)
            retirement_allowed = decision.retirement_allowed
            residual_lag = decision.residual_lag
            detail = decision.reason
            if not decision.retirement_allowed:
                raise ValueError(
                    f"cutover blocked by drain-proof gate: {decision.reason}"
                )

        elif command.target_phase is EnumMigrationPhase.COMPLETE:
            # COMPLETE retires the old group; re-assert drain proof at retirement.
            decision = await self._drain_proof_gate.evaluate(contract)
            retirement_allowed = decision.retirement_allowed
            residual_lag = decision.residual_lag
            detail = (
                f"retiring old group {old_group!r} on topic {old_topic!r}: "
                f"{decision.reason}"
            )
            if not decision.retirement_allowed:
                raise ValueError(
                    f"completion blocked by drain-proof gate: {decision.reason}"
                )

        else:
            detail = f"advanced migration to phase {command.target_phase.value!r}"

        return ModelTopicMigrationLifecycleEvent(
            event_id=uuid4(),
            correlation_id=command.correlation_id,
            migration_ticket=contract.ticket,
            old_topic=old_topic,
            new_topic=new_topic,
            old_consumer_group=old_group,
            new_consumer_group=contract.new_consumer_group,
            phase=command.target_phase,
            sequence=self._phase_index(command.target_phase),
            new_topic_provisioned=new_provisioned,
            retirement_allowed=retirement_allowed,
            residual_lag=residual_lag,
            detail=detail,
        )


__all__ = ["HandlerTopicMigrationExecutor"]
