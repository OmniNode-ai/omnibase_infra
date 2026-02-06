# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Model for materialized infrastructure resources.

Holds infrastructure resources (asyncpg pools, Kafka producers, HTTP clients)
created by the DependencyMaterializer from contract.dependencies declarations.

Part of OMN-1976: Contract dependency materialization.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ModelMaterializedResources(BaseModel):
    """Container for infrastructure resources materialized from contracts.

    Maps dependency names to their materialized resource instances.
    Multiple names may alias the same underlying resource (deduplication).

    Example:
        >>> resources = ModelMaterializedResources(
        ...     resources={
        ...         "pattern_store": asyncpg_pool,
        ...         "kafka_producer": aiokafka_producer,
        ...     }
        ... )
        >>> pool = resources.get("pattern_store")

    .. versionadded:: 0.x.x
        Part of OMN-1976 contract dependency materialization.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
        arbitrary_types_allowed=True,
    )

    # ONEX_EXCLUDE: any_type - dict holds heterogeneous resource instances
    # (asyncpg.Pool, AIOKafkaProducer, httpx.AsyncClient). Type varies by resource.
    resources: dict[str, Any] = Field(
        default_factory=dict,
        description="Map of dependency names to materialized resource instances",
    )

    # ONEX_EXCLUDE: any_type - returns heterogeneous resource instance
    def get(self, name: str) -> Any:
        """Get a materialized resource by dependency name.

        Args:
            name: The dependency name from contract.yaml

        Returns:
            The materialized resource instance.

        Raises:
            KeyError: If name is not in the materialized resources.
        """
        if name not in self.resources:
            raise KeyError(
                f"Resource '{name}' not found in materialized resources. "
                f"Available: {list(self.resources.keys())}"
            )
        return self.resources[name]

    # ONEX_EXCLUDE: any_type - returns heterogeneous resource instance or default
    def get_optional(self, name: str, default: Any = None) -> Any:
        """Get a resource by name, returning default if not found."""
        return self.resources.get(name, default)

    def has(self, name: str) -> bool:
        """Check if a resource is available."""
        return name in self.resources

    def __len__(self) -> int:
        """Return number of materialized resources."""
        return len(self.resources)

    def __bool__(self) -> bool:
        """Return True if any resources are materialized.

        Warning:
            **Non-standard __bool__ behavior**: Returns True only when
            at least one resource is materialized.
        """
        return len(self.resources) > 0


__all__ = ["ModelMaterializedResources"]
