# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed Docker database-consumer catalog projection."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.topology.models.model_deferred_docker_database_consumer import (
    ModelDeferredDockerDatabaseConsumer,
)
from omnibase_infra.topology.models.model_docker_database_consumer import (
    ModelDockerDatabaseConsumer,
)

__all__ = ["ModelDockerDatabaseConsumerCatalog"]


class ModelDockerDatabaseConsumerCatalog(BaseModel):
    """Adapter from Docker service names to topology binding references."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = Field(pattern=r"^[0-9]+\.[0-9]+$")
    environment: str
    consumers: dict[str, ModelDockerDatabaseConsumer]
    deferred_consumers: dict[str, ModelDeferredDockerDatabaseConsumer] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def consumer_sets_do_not_overlap(self) -> ModelDockerDatabaseConsumerCatalog:
        """A service cannot be both classified and held."""
        overlap = sorted(self.consumers.keys() & self.deferred_consumers.keys())
        if overlap:
            raise ValueError(
                f"Docker consumers cannot be mapped and deferred: {overlap}"
            )
        return self
