# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Full delegation config parsed from routing_tiers.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_delegation_routing_reducer.models.model_routing_tier import (
    ModelRoutingTier,
)
from omnibase_infra.nodes.node_delegation_routing_reducer.models.model_tier_model import (
    ModelTierModel,
)

_DEFAULT_CONFIG_PATH = (
    Path(__file__).parent.parent.parent.parent / "configs" / "routing_tiers.yaml"
)


class ModelDelegationConfig(BaseModel):
    """Full delegation config parsed from routing_tiers.yaml."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tiers: tuple[ModelRoutingTier, ...] = Field(
        default_factory=tuple,
        description="Ordered escalation tiers.",
    )

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> ModelDelegationConfig:
        """Load delegation config from a YAML file.

        Args:
            path: Path to routing_tiers.yaml. Defaults to the bundled config.

        Returns:
            Parsed and validated DelegationConfig.
        """
        config_path = path or _DEFAULT_CONFIG_PATH
        raw = yaml.safe_load(config_path.read_text())
        tiers = []
        for tier_data in raw.get("tiers", []):
            models = []
            for m in tier_data.get("models", []):
                models.append(
                    ModelTierModel(
                        id=m["id"],
                        env_var=m["env_var"],
                        max_context_tokens=m["max_context_tokens"],
                        use_for=tuple(m.get("use_for", [])),
                        fast_path_threshold_tokens=m.get("fast_path_threshold_tokens"),
                    )
                )
            tiers.append(
                ModelRoutingTier(
                    name=tier_data["name"],
                    models=tuple(models),
                    eval_before_accept=tier_data.get("eval_before_accept", False),
                    eval_model=tier_data.get("eval_model"),
                    max_retries=tier_data.get("max_retries", 0),
                )
            )
        return cls(tiers=tuple(tiers))


__all__: list[str] = ["ModelDelegationConfig"]
