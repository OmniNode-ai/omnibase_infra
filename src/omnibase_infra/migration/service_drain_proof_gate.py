# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Drain-proof gate for topic migrations (OMN-12623).

A topic migration may only retire the OLD consumer group (and stop writing the
old topic) once the old group has fully drained the old topic. This gate makes
that decision explicit and decidable:

* it requires the migration contract to declare the drain-proof requirement
  (``OLD_TOPIC_DRAINED`` cutover criterion + ``drain_proof_required``);
* it observes the old group's lag via :class:`ServiceConsumerLagObserver`;
* it returns a typed decision that BLOCKS retirement while any lag remains and
  ALLOWS it only on observed zero-lag drain proof.

There is no "skip" path: opting out of drain proof is an explicit field on the
contract, validated by the contract itself (OMN-12621). The gate never silently
allows retirement on unobservable lag.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.enums.enum_cutover_criterion import EnumCutoverCriterion
from omnibase_core.models.contracts.model_topic_migration_contract import (
    ModelTopicMigrationContract,
)
from omnibase_infra.migration.models.model_consumer_group_lag import (
    ModelConsumerGroupLag,
)
from omnibase_infra.migration.service_consumer_lag_observer import (
    ServiceConsumerLagObserver,
)

logger = logging.getLogger(__name__)


class ModelDrainProofDecision(BaseModel):
    """Typed outcome of evaluating the drain-proof gate for a migration."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    migration_ticket: str = Field(..., description="Migration contract ticket id")
    old_consumer_group: str = Field(..., description="Old group being evaluated")
    old_topic: str = Field(..., description="Old topic being drained")
    retirement_allowed: bool = Field(
        ...,
        description="True iff the old group may be retired (zero residual lag)",
    )
    residual_lag: int = Field(
        ...,
        ge=0,
        description="Remaining un-consumed messages on the old topic for the group",
    )
    reason: str = Field(..., description="Human-readable rationale for the decision")


class ServiceDrainProofGate:
    """Blocks old-group retirement until the old topic is provably drained."""

    def __init__(self, observer: ServiceConsumerLagObserver) -> None:
        self._observer = observer

    async def evaluate(
        self,
        contract: ModelTopicMigrationContract,
    ) -> ModelDrainProofDecision:
        """Decide whether the migration's old consumer group may be retired.

        Observes the old group's lag on the old topic and gates retirement on a
        zero-lag drain proof. Honors ``contract.drain_proof_required``: when the
        contract explicitly opts out (False), retirement is allowed without a lag
        observation — but the contract validator guarantees that opting out was a
        deliberate, recorded choice.
        """
        old_topic = contract.old_binding.topic
        old_group = contract.old_consumer_group

        if not contract.drain_proof_required:
            return ModelDrainProofDecision(
                migration_ticket=contract.ticket,
                old_consumer_group=old_group,
                old_topic=old_topic,
                retirement_allowed=True,
                residual_lag=0,
                reason=(
                    "drain_proof_required is False on the migration contract; "
                    "retirement allowed without lag observation (explicit opt-out)."
                ),
            )

        # Defensive contract-shape assertion: a drain-proof migration MUST list
        # OLD_TOPIC_DRAINED. The contract validator enforces this too, but the
        # gate refuses to act on a contradictory contract.
        if EnumCutoverCriterion.OLD_TOPIC_DRAINED not in contract.cutover_criteria:
            raise ValueError(
                "drain_proof_required is True but OLD_TOPIC_DRAINED is not a "
                "cutover criterion; refusing to evaluate an inconsistent contract."
            )

        lag: ModelConsumerGroupLag = await self._observer.observe(old_group)
        residual = lag.lag_for_topic(old_topic)

        if not lag.partition_offsets:
            return ModelDrainProofDecision(
                migration_ticket=contract.ticket,
                old_consumer_group=old_group,
                old_topic=old_topic,
                retirement_allowed=False,
                residual_lag=0,
                reason=(
                    f"no committed offsets observed for group {old_group!r}; "
                    "drain is unproven (absence of evidence is not proof of drain)."
                ),
            )

        if residual > 0:
            return ModelDrainProofDecision(
                migration_ticket=contract.ticket,
                old_consumer_group=old_group,
                old_topic=old_topic,
                retirement_allowed=False,
                residual_lag=residual,
                reason=(
                    f"old topic {old_topic!r} still has {residual} un-consumed "
                    f"message(s) for group {old_group!r}; retirement blocked."
                ),
            )

        return ModelDrainProofDecision(
            migration_ticket=contract.ticket,
            old_consumer_group=old_group,
            old_topic=old_topic,
            retirement_allowed=True,
            residual_lag=0,
            reason=(
                f"old topic {old_topic!r} fully drained for group {old_group!r} "
                "(zero residual lag); retirement allowed."
            ),
        )


__all__ = ["ModelDrainProofDecision", "ServiceDrainProofGate"]
