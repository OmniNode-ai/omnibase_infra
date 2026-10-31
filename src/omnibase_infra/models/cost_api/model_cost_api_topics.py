# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-declared snapshot topics of the cost API (OMN-21623).

The values live in the ``topics`` block of ``services/cost_api/cost_api_topics.yaml``.
The model has no field defaults and checks each value's shape, so a block that
omits a topic or names a non-snapshot topic fails to load rather than falling
back to a literal.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

_TOPICS_PATH = (
    Path(__file__).resolve().parents[2]
    / "services"
    / "cost_api"
    / "cost_api_topics.yaml"
)

_SNAPSHOT_TOPIC_PATTERN = r"^onex\.snapshot\.[a-z0-9_-]+\.[a-z0-9._-]+\.v[0-9]+$"


class ModelCostApiTopics(BaseModel):
    """Projection snapshot topics whose latest payload the cost API serves.

    Attributes:
        summary: Topic of the cost summary snapshot.
        by_repo: Topic of the cost-by-repository snapshot.
        token_usage: Topic of the token usage snapshot.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    summary: str = Field(pattern=_SNAPSHOT_TOPIC_PATTERN)
    by_repo: str = Field(pattern=_SNAPSHOT_TOPIC_PATTERN)
    token_usage: str = Field(pattern=_SNAPSHOT_TOPIC_PATTERN)

    @classmethod
    def from_file(cls, topics_path: Path) -> ModelCostApiTopics:
        """Load the topics from the ``topics`` mapping of a YAML file.

        Raises:
            ProtocolConfigurationError: If the file has no ``topics`` mapping.
        """
        with topics_path.open(encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}
        block = document.get("topics") if isinstance(document, dict) else None
        if not isinstance(block, dict):
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.FILESYSTEM,
                operation="load_cost_api_topics",
            )
            raise ProtocolConfigurationError(
                f"{topics_path.name} has no topics mapping",
                context=context,
            )
        return cls.model_validate(block)


@cache
def load_cost_api_topics() -> ModelCostApiTopics:
    """Return the snapshot topics declared beside the cost API."""
    return ModelCostApiTopics.from_file(_TOPICS_PATH)


__all__: list[str] = ["ModelCostApiTopics", "load_cost_api_topics"]
