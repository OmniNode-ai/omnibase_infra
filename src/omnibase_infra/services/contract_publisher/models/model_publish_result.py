# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Publish Result Model.

Complete result of contract publishing with errors and statistics.

.. versionadded:: 0.3.0
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.services.contract_publisher.models.model_contract_error import (
    ModelContractError,
)
from omnibase_infra.services.contract_publisher.models.model_infra_error import (
    ModelInfraError,
)
from omnibase_infra.services.contract_publisher.models.model_publish_stats import (
    ModelPublishStats,
)


class ModelPublishResult(BaseModel):
    """Complete result of contract publishing operation.

    Contains three categories of information:
    - published: List of handler_ids successfully published
    - contract_errors: Non-fatal contract-level errors
    - infra_errors: Infrastructure errors
    - stats: Publishing statistics

    The __bool__ method returns True if any contracts were published
    successfully, enabling idiomatic conditional checks:

        >>> result = await publisher.publish_all()
        >>> if result:
        ...     print(f"Published {len(result.published)} contracts")
        ... else:
        ...     print("No contracts published")

    Warning:
        This model overrides __bool__ to return True only when contracts
        are published. This differs from standard Pydantic behavior where
        bool(model) always returns True. Use explicit checks if you need
        to verify the model exists rather than its success state.

    Attributes:
        published: List of handler_ids successfully published
        contract_errors: Non-fatal contract-level errors
        infra_errors: Infrastructure errors
        stats: Publishing statistics

    Example:
        >>> result = ModelPublishResult(
        ...     published=["handler.foo", "handler.bar"],
        ...     contract_errors=[],
        ...     infra_errors=[],
        ...     stats=stats,
        ... )
        >>> if result:
        ...     print("Success!")

    .. versionadded:: 0.3.0
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    published: list[str] = Field(
        description="List of handler_ids successfully published"
    )
    contract_errors: list[ModelContractError] = Field(
        default_factory=list,
        description="Non-fatal contract-level errors",
    )
    infra_errors: list[ModelInfraError] = Field(
        default_factory=list,
        description="Infrastructure errors",
    )
    stats: ModelPublishStats = Field(description="Publishing statistics")

    def __bool__(self) -> bool:
        """Return True if any contracts were published successfully.

        Warning:
            This differs from standard Pydantic behavior. The model
            evaluates to False when no contracts are published, even
            if the model itself is valid.

        Returns:
            True if published list is non-empty, False otherwise.
        """
        return bool(self.published)

    @property
    def has_contract_errors(self) -> bool:
        """Check if any contract errors occurred."""
        return len(self.contract_errors) > 0

    @property
    def has_infra_errors(self) -> bool:
        """Check if any infrastructure errors occurred."""
        return len(self.infra_errors) > 0

    @property
    def has_errors(self) -> bool:
        """Check if any errors (contract or infra) occurred."""
        return self.has_contract_errors or self.has_infra_errors


__all__ = ["ModelPublishResult"]
