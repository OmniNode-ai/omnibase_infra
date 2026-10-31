# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Premium-model markers for the model_overkill rule, declared in the node contract.

Related Tickets:
    - OMN-21623: model names leave the analyzer; the node contract declares them
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

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
        document = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        config = document.get("config") if isinstance(document, dict) else None
        block = config.get(_POLICY_KEY) if isinstance(config, dict) else None
        return cls.model_validate(block)


__all__: list[str] = ["ModelWastePremiumModelPolicy"]
