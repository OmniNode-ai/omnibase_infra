# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A folded profile's consumer-group handover, declared once (OMN-22714).

The one-runtime fold of a profile with many contracts (effects: ~175) cannot
declare ``consumer_group_handover`` in every contract. The lane's runtime policy
contract declares it once per folded profile; the renderer writes it as
``ONEX_RUNTIME_FOLD_HANDOVERS`` and the auto-wiring gives it to every contract
the folded profile owns that declares none of its own.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.event_bus.model_consumer_group_fence import (
    ModelConsumerGroupFence,
)
from omnibase_infra.models.event_bus.model_consumer_group_handover import (
    ModelConsumerGroupHandover,
)

__all__ = ["ModelFoldHandover"]


class ModelFoldHandover(BaseModel):
    """The retired process of a folded profile, and the rule its commits follow."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    from_instance_id: str = Field(
        ...,
        description=(
            "KAFKA_INSTANCE_ID of the retired container that ran the profile's "
            "contracts (runtime-effects for the effects fold)."
        ),
    )
    from_auto_offset_reset: Literal["earliest", "latest"] = Field(
        ...,
        description="What the retired consumer did on a partition it never committed.",
    )
    handover_rule: Literal["copy_if_uncommitted", "last_ran"] = Field(
        ...,
        description=(
            "last_ran for effects and the worker, whose handlers cannot take a "
            "repeat; stated on every declaration, never defaulted."
        ),
    )
    fence: ModelConsumerGroupFence = Field(default_factory=ModelConsumerGroupFence)

    def to_contract_handover(self) -> ModelConsumerGroupHandover:
        """The per-contract declaration this fold-wide one stands for."""
        return ModelConsumerGroupHandover(
            from_instance_id=self.from_instance_id,
            from_auto_offset_reset=self.from_auto_offset_reset,
            handover_rule=self.handover_rule,
            fence=self.fence,
        )
