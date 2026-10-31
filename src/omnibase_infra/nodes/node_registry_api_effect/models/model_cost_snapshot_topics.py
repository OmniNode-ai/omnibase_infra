# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cost-snapshot projection topics declared in the registry API node contract.

The cost API routes read the latest projection snapshot keyed by topic. The topic
names live in ``contract.yaml`` under ``config.cost_snapshot_topics``; this model
carries them into the app factory, which injects them into the route handlers.

Ticket: OMN-21736
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_registry_api_effect.node import (
    load_registry_api_config,
)


class ModelCostSnapshotTopics(BaseModel):
    """Topics whose latest projection snapshot backs the cost API routes.

    Attributes:
        summary: Topic of the cost summary snapshot.
        by_repo: Topic of the cost-by-repo snapshot.
        token_usage: Topic of the token-usage snapshot.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    summary: str = Field(..., min_length=1)
    by_repo: str = Field(..., min_length=1)
    token_usage: str = Field(..., min_length=1)

    @classmethod
    def from_contract(cls) -> ModelCostSnapshotTopics:
        """Build the topics from ``config.cost_snapshot_topics`` of the node contract.

        Raises:
            pydantic.ValidationError: If the contract does not declare every topic.
        """
        return cls.model_validate(
            load_registry_api_config().get("cost_snapshot_topics", {})
        )


__all__ = ["ModelCostSnapshotTopics"]
