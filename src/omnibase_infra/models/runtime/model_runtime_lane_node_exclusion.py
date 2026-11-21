# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The contracts a runtime lane leaves unattached, declared as data (OMN-26324).

``contracts/services/runtime_lane_node_exclusion.contract.yaml`` names, per
runtime profile, the contracts that profile does not attach and the contracts it
needs. A profile is a registered runtime lane without its ``compose-`` prefix
(``runtime_policy.contract.yaml`` keys its profiles the same way), so the
declaration spells no lane id. The lane a runtime declares
(``ONEX_RUNTIME_LANE``) selects its profile's entry; the excluded names join ``ONEX_RUNTIME_NODE_EXCLUSION`` in
:class:`~omnibase_infra.models.runtime.model_runtime_node_selection.ModelRuntimeNodeSelection`,
so the auto-wiring ownership filter, the health monitor and the raw ownership
path all read one list. No contract name is a literal in code.

Where the file is read from is the runtime's contracts directory
(``ONEX_CONTRACTS_DIR``). Unset, blank, a missing file, or a lane with no entry
declares nothing: the lane attaches what it attached before. A file that is
present and malformed raises, because a half-read declaration would attach a
different set than the one the repository declares.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Final, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_core.constants.constants_runtime_lanes import REGISTERED_RUNTIME_LANES
from omnibase_infra.models.runtime.model_runtime_lane_use_path import (
    ModelRuntimeLaneUsePath,
)

ENV_CONTRACTS_DIR: Final = "ONEX_CONTRACTS_DIR"
ENV_RUNTIME_LANE: Final = "ONEX_RUNTIME_LANE"

#: The prefix a registered lane carries over its policy profile name.
LANE_PROFILE_PREFIX: Final = "compose-"

#: The declaration's path under the runtime's contracts directory.
RUNTIME_LANE_NODE_EXCLUSION_CONTRACT: Final = Path(
    "services/runtime_lane_node_exclusion.contract.yaml"
)


def lane_profile(lane: str) -> str:
    """The runtime-policy profile a registered lane belongs to."""
    return lane.removeprefix(LANE_PROFILE_PREFIX)


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


class ModelRuntimeLaneNodeExclusion(BaseModel):
    """The declaration file: one entry per registered runtime lane."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    name: Literal["runtime_lane_node_exclusion"]
    version: int = Field(ge=1)
    profiles: dict[str, ModelRuntimeLaneNodeExclusionEntry] = Field(
        default_factory=dict
    )

    @field_validator("profiles")
    @classmethod
    def _profiles_belong_to_a_registered_lane(
        cls, value: dict[str, ModelRuntimeLaneNodeExclusionEntry]
    ) -> dict[str, ModelRuntimeLaneNodeExclusionEntry]:
        known = {lane_profile(lane) for lane in REGISTERED_RUNTIME_LANES}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(
                f"profiles {unknown} belong to no registered runtime lane "
                f"(known profiles {sorted(known)})"
            )
        return value

    @classmethod
    def declaration_path(cls, contracts_dir: Path) -> Path:
        """Where the declaration lives under a contracts directory."""
        return contracts_dir / RUNTIME_LANE_NODE_EXCLUSION_CONTRACT

    @classmethod
    def load(cls, path: Path) -> ModelRuntimeLaneNodeExclusion:
        """Parse the declaration at ``path``; a malformed file raises."""
        stat = path.stat()
        return _load_cached(str(path), stat.st_mtime_ns, stat.st_size)

    def entry_for(self, lane: str | None) -> ModelRuntimeLaneNodeExclusionEntry:
        """The lane's profile entry; a lane whose profile has none declares nothing."""
        if lane is None:
            return ModelRuntimeLaneNodeExclusionEntry()
        return self.profiles.get(
            lane_profile(lane), ModelRuntimeLaneNodeExclusionEntry()
        )


@lru_cache(maxsize=16)
def _load_cached(path: str, mtime_ns: int, size: int) -> ModelRuntimeLaneNodeExclusion:
    # mtime and size key the cache: the raw ownership path asks once per contract.
    del mtime_ns, size
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ModelRuntimeLaneNodeExclusion.model_validate(raw)


def declared_lane_node_exclusion(
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """The contracts the declaration excludes for the lane this runtime declares.

    Empty when the runtime names no contracts directory, the directory holds no
    declaration, the runtime declares no lane, or the lane's profile has no entry.
    """
    source: Mapping[str, str] = os.environ if environ is None else environ
    contracts_dir = (source.get(ENV_CONTRACTS_DIR) or "").strip()
    lane = (source.get(ENV_RUNTIME_LANE) or "").strip().lower()
    if not contracts_dir or not lane:
        return ()
    path = ModelRuntimeLaneNodeExclusion.declaration_path(Path(contracts_dir))
    if not path.is_file():
        return ()
    return ModelRuntimeLaneNodeExclusion.load(path).entry_for(lane).excluded_contracts


__all__ = [
    "ENV_CONTRACTS_DIR",
    "LANE_PROFILE_PREFIX",
    "RUNTIME_LANE_NODE_EXCLUSION_CONTRACT",
    "ModelRuntimeLaneNodeExclusion",
    "ModelRuntimeLaneNodeExclusionEntry",
    "declared_lane_node_exclusion",
    "lane_profile",
]
