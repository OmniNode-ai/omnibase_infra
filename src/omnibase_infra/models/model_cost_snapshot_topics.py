# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cost-snapshot projection topics declared in the registry API node contract.

The cost API routes read the latest projection snapshot keyed by topic. The topic
names live in ``contract.yaml`` under ``config.cost_snapshot_topics``; this model
carries them into the app factory, which injects them into the route handlers.

Ticket: OMN-21736
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

_SNAPSHOT_TOPIC_PATTERN = r"^onex\.snapshot\.[a-z0-9_-]+\.[a-z0-9._-]+\.v[0-9]+$"


class ModelCostSnapshotTopics(BaseModel):
    """Topics whose latest projection snapshot backs the cost API routes.

    Attributes:
        summary: Topic of the cost summary snapshot.
        by_repo: Topic of the cost-by-repo snapshot.
        token_usage: Topic of the token-usage snapshot.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    summary: str = Field(..., pattern=_SNAPSHOT_TOPIC_PATTERN)
    by_repo: str = Field(..., pattern=_SNAPSHOT_TOPIC_PATTERN)
    token_usage: str = Field(..., pattern=_SNAPSHOT_TOPIC_PATTERN)

    @classmethod
    def from_contract(
        cls, contract_path: Path | None = None
    ) -> ModelCostSnapshotTopics:
        """Build the topics from ``config.cost_snapshot_topics`` of the node contract.

        Raises:
            pydantic.ValidationError: If any topic is missing or is not a snapshot topic.
        """
        path = contract_path or (
            Path(__file__).resolve().parents[1]
            / "nodes/node_registry_api_effect/contract.yaml"
        )
        contract = yaml.safe_load(path.read_text(encoding="utf-8"))
        config = contract.get("config") if isinstance(contract, dict) else None
        topics = (
            config.get("cost_snapshot_topics") if isinstance(config, dict) else None
        )
        return cls.model_validate(topics if isinstance(topics, dict) else {})


__all__ = ["ModelCostSnapshotTopics"]
