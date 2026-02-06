# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Top-level configuration for the DependencyMaterializer.

Part of OMN-1976: Contract dependency materialization.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.models.model_http_client_config import (
    ModelHttpClientConfig,
)
from omnibase_infra.runtime.models.model_kafka_producer_config import (
    ModelKafkaProducerConfig,
)
from omnibase_infra.runtime.models.model_postgres_pool_config import (
    ModelPostgresPoolConfig,
)


class ModelMaterializerConfig(BaseModel):
    """Top-level configuration for the DependencyMaterializer."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    postgres: ModelPostgresPoolConfig = Field(
        default_factory=ModelPostgresPoolConfig.from_env,
    )
    kafka: ModelKafkaProducerConfig = Field(
        default_factory=ModelKafkaProducerConfig.from_env,
    )
    http: ModelHttpClientConfig = Field(
        default_factory=ModelHttpClientConfig.from_env,
    )

    @classmethod
    def from_env(cls) -> ModelMaterializerConfig:
        """Create full config from environment."""
        return cls(
            postgres=ModelPostgresPoolConfig.from_env(),
            kafka=ModelKafkaProducerConfig.from_env(),
            http=ModelHttpClientConfig.from_env(),
        )


__all__ = ["ModelMaterializerConfig"]
