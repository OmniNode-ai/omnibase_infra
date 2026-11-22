# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One runtime lane's node-exclusion entry, declared as data (OMN-26324).

An entry names the contracts a lane leaves unattached and the contracts it needs.
:class:`~omnibase_infra.models.runtime.model_runtime_lane_node_exclusion.ModelRuntimeLaneNodeExclusion`
holds one entry per runtime profile.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.models.runtime.model_runtime_lane_use_path import (
    ModelRuntimeLaneUsePath,
)


def _checked_names(value: tuple[str, ...]) -> tuple[str, ...]:
    if any(not name.strip() or "," in name for name in value):
        raise ValueError("contract names must be non-empty and hold no comma")
    if len(set(value)) != len(value):
        raise ValueError("contract names must be unique")
    return value


class ModelRuntimeLaneNodeExclusionEntry(BaseModel):
    """One lane's declaration: what it leaves unattached and what it needs."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    excluded_contracts: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Contract names this lane does not attach.",
    )
    required_contracts: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Contract names this lane needs. Never read at boot: the lane's "
            "regression fails when one is also excluded."
        ),
    )

    delegation_check_paths: tuple[ModelRuntimeLaneUsePath, ...] = Field(
        default_factory=tuple,
        description=(
            "The paths this lane's work uses, as seeds. Never read at boot: the "
            "lane's regression fails when a contract they reach is also excluded."
        ),
    )
    reached_contracts: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "The contracts the declared paths reach through topics, as read from "
            "every shipped contract file. Never read at boot."
        ),
    )

    @field_validator("excluded_contracts", "required_contracts", "reached_contracts")
    @classmethod
    def _names_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _checked_names(value)


__all__ = ["ModelRuntimeLaneNodeExclusionEntry"]
