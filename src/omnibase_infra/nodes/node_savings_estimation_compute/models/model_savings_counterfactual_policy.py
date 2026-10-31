# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Counterfactual model policy for savings estimation, declared in the node contract.

Related Tickets:
    - OMN-21623: model names leave the handler; the node contract declares them
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_savings_estimation_compute.models.model_savings_counterfactual_rule import (
    ModelSavingsCounterfactualRule,
)

_NODE_CONTRACT_PATH = Path(__file__).resolve().parents[1] / "contract.yaml"
_POLICY_KEY = "counterfactual_policy"


class ModelSavingsCounterfactualPolicy(BaseModel):
    """The counterfactual model rules and the fallback model, read from the contract.

    Every field is required: a contract that omits one fails to load rather than
    falling back to a value held in code.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    fallback_model: str = Field(
        ...,
        min_length=1,
        description="Model assumed when a session records none",
    )
    rules: tuple[ModelSavingsCounterfactualRule, ...] = Field(
        ..., min_length=1, description="Ordered rules; the first matching rule wins"
    )

    def resolve(self, actual_model_id: str) -> str:
        """Return the counterfactual model for ``actual_model_id``.

        Never returns an empty value: downstream projection treats an absent or
        empty ``model_cloud_baseline`` as a malformed event (OMN-14533), so an
        unmatched model resolves to itself.
        """
        lower = actual_model_id.lower()
        for rule in self.rules:
            if rule.match in lower:
                return rule.counterfactual_model
        return actual_model_id

    @classmethod
    def from_node_contract(
        cls, contract_path: Path = _NODE_CONTRACT_PATH
    ) -> ModelSavingsCounterfactualPolicy:
        """Load the policy from the ``config`` block of the node contract."""
        document = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        config = document.get("config") if isinstance(document, dict) else None
        block = config.get(_POLICY_KEY) if isinstance(config, dict) else None
        return cls.model_validate(block)


__all__: list[str] = ["ModelSavingsCounterfactualPolicy"]
