# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Premium-model markers for the model_overkill rule, declared in the node contract.

Related Tickets:
    - OMN-21623: model names leave the analyzer; the node contract declares them
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.model_node_contract_config_block import (
    ModelNodeContractConfigBlock,
)

_NODE_CONTRACT_PATH = Path(__file__).resolve().parents[1] / "contract.yaml"
_POLICY_KEY = "premium_model_policy"


class ModelWastePremiumModelPolicy(BaseModel):
    """Lower-case substrings that identify a premium model identifier.

    The field is required: a contract that omits it fails to load rather than
    falling back to markers held in code.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    markers: tuple[str, ...] = Field(
        ..., min_length=1, description="Lower-case premium model identifier markers"
    )

    @classmethod
    def from_node_contract(
        cls, contract_path: Path = _NODE_CONTRACT_PATH
    ) -> ModelWastePremiumModelPolicy:
        """Load the policy from the ``config`` block of the node contract."""
        block = ModelNodeContractConfigBlock.load(contract_path, _POLICY_KEY)
        return cls.model_validate(block)


__all__: list[str] = ["ModelWastePremiumModelPolicy"]
