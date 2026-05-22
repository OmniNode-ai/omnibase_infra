# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Model for materialized infrastructure resources.

Holds infrastructure resources (asyncpg pools, Kafka producers, HTTP clients)
created by the DependencyMaterializer from contract.dependencies declarations.

Part of OMN-1976: Contract dependency materialization.
"""

from __future__ import annotations

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

    .. versionadded:: 0.4.1
        Part of OMN-1976 contract dependency materialization.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
        arbitrary_types_allowed=True,
    )

    resources: dict[str, object] = Field(
        default_factory=dict,
        description="Map of dependency names to materialized resource instances",
    )

    def get(self, name: str) -> object:
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

    def get_optional(self, name: str, default: object | None = None) -> object | None:
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
