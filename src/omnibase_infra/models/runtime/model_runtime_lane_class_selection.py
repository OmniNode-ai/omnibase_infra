# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The class contracts a lane instance's deployment selects (OMN-20867).

Read from ``ONEX_RUNTIME_LANE_CLASS`` and ``ONEX_RUNTIME_LANE_CLASS_CONTRACTS``
by the auto-wiring ownership filter; see
:func:`omnibase_infra.runtime.auto_wiring.profile_ownership.resolve_runtime_lane_class_selection`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.models.contracts.subcontracts.model_runtime_lane_scope import (
    ModelRuntimeLaneScope,
)


class ModelRuntimeLaneClassSelection(BaseModel):
    """The class contracts a lane instance's deployment selects (OMN-20867)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lane_class: str | None = Field(
        default=None,
        description=(
            "The registered lane class this runtime is an instance of, or None "
            "when the deployment selects none or the selection is refused."
        ),
    )
    contracts: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Contract names the instance takes from its class.",
    )
    refusal: str = Field(
        default="",
        description="Why a set selection cannot be honoured; empty when it can.",
    )

    def admits(self, contract_name: str, scope: ModelRuntimeLaneScope) -> bool:
        """Return whether this selection attaches a contract its lane excluded."""
        return (
            self.lane_class is not None
            and contract_name in self.contracts
            and scope.admits(self.lane_class)
        )


__all__ = ["ModelRuntimeLaneClassSelection"]
