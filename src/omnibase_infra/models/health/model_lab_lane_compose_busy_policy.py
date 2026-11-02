# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Settle stages that are not a compose deploy, declared in the node contract.

Related Tickets:
    - OMN-21866: a k3s lab_overlay settle is not a compose deploy
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.model_node_contract_config_block import (
    ModelNodeContractConfigBlock,
)

_NODE_CONTRACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "nodes"
    / "node_lab_lane_compose_busy_compute"
    / "contract.yaml"
)
_POLICY_KEY = "compose_busy_policy"


class ModelLabLaneComposeBusyPolicy(BaseModel):
    """The ``config.compose_busy_policy`` block of node_lab_lane_compose_busy_compute.

    The field is required: a contract that omits it fails to load rather than
    falling back to stages held in code.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    non_deploy_settling_stages: tuple[str, ...] = Field(
        ...,
        description="settling_stage values that mutate the k3s lane, not the compose project.",
    )

    @classmethod
    def from_node_contract(
        cls, contract_path: Path = _NODE_CONTRACT_PATH
    ) -> ModelLabLaneComposeBusyPolicy:
        """Load the policy from the ``config`` block of the node contract."""
        block = ModelNodeContractConfigBlock.load(contract_path, _POLICY_KEY)
        return cls.model_validate(block)


__all__: list[str] = ["ModelLabLaneComposeBusyPolicy"]
