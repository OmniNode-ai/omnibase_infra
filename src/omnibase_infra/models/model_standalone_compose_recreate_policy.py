# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The lanes and targets the standalone compose recreate route may touch, as the node contract declares them."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.model_standalone_compose_recreate_target import (
    ModelStandaloneComposeRecreateTarget,
)


class ModelStandaloneComposeRecreatePolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed_lanes: frozenset[str]
    targets: tuple[ModelStandaloneComposeRecreateTarget, ...]

    @classmethod
    def from_contract(cls, contract_path: Path) -> ModelStandaloneComposeRecreatePolicy:
        """Read ``allowed_lanes`` and ``standalone_targets`` from the node contract."""
        contract = yaml.safe_load(contract_path.read_text())
        return cls(
            allowed_lanes=frozenset(str(lane) for lane in contract["allowed_lanes"]),
            targets=tuple(
                ModelStandaloneComposeRecreateTarget.model_validate(item)
                for item in contract["standalone_targets"]
            ),
        )
