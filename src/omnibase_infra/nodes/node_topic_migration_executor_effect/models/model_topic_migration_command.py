# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed command model for the migration executor (OMN-12623).

The migration executor is driven by a :class:`ModelTopicMigrationContract` from
``omnibase_core`` (OMN-12621). The command wraps that contract with the runtime
parameters the executor needs (correlation id, partition/replication for the new
topic, dual-publish toggle) so the executor's I/O surface is fully typed rather
than argparse-driven.

The executor emits :class:`ModelTopicMigrationLifecycleEvent` for each phase it
advances; that model lives in ``model_topic_migration_lifecycle_event.py``.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.enums.enum_migration_phase import EnumMigrationPhase
from omnibase_core.models.contracts.model_topic_migration_contract import (
    ModelTopicMigrationContract,
)


class ModelTopicMigrationCommand(BaseModel):
    """Command instructing the executor to advance a topic migration.

    The command is immutable. ``target_phase`` is the phase the executor should
    drive the migration *to*; the executor validates the transition is a forward
    move from the contract's current ``phase``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Trace correlation id")
    contract: ModelTopicMigrationContract = Field(
        ..., description="Declarative migration contract being executed"
    )
    target_phase: EnumMigrationPhase = Field(
        ...,
        description="Phase the executor should advance the migration to",
    )
    new_topic_partitions: int = Field(
        default=6,
        ge=1,
        description="Partition count to provision for the new topic",
    )
    new_topic_replication_factor: int = Field(
        default=1,
        ge=1,
        description="Replication factor to provision for the new topic",
    )
    dual_publish: bool = Field(
        default=False,
        description=(
            "Whether producers dual-publish to both topics during the window. "
            "Informational for the executor's emitted lifecycle event."
        ),
    )


__all__ = ["ModelTopicMigrationCommand"]
