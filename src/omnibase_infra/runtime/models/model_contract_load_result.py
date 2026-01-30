# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract load result model for tracking individual contract loading.

This module provides the ModelContractLoadResult class for representing
the outcome of loading a single contract.yaml file, including any subcontracts
(handler_routing, operation_bindings) that were loaded from it.

Part of OMN-1519: Runtime contract config loader.

Design Pattern:
    ModelContractLoadResult captures both successful and failed contract
    loading attempts, allowing the runtime to continue loading other
    contracts even if one fails. Errors are collected rather than raised
    immediately.

Thread Safety:
    ModelContractLoadResult is immutable (frozen=True) after creation,
    making it thread-safe for concurrent read access.

Example:
    >>> from pathlib import Path
    >>> from omnibase_infra.runtime.models import ModelContractLoadResult
    >>>
    >>> # Successful load
    >>> result = ModelContractLoadResult.succeeded(
    ...     contract_path=Path("nodes/auth/contract.yaml"),
    ...     handler_routing=routing_subcontract,
    ...     operation_bindings=bindings_subcontract,
    ... )
    >>> result.success
    True
    >>>
    >>> # Failed load
    >>> result = ModelContractLoadResult.failed(
    ...     contract_path=Path("nodes/broken/contract.yaml"),
    ...     error="Invalid YAML syntax",
    ... )
    >>> result.success
    False

.. versionadded:: 0.2.8
    Created as part of OMN-1519.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.bindings import ModelOperationBindingsSubcontract
from omnibase_infra.models.routing import ModelRoutingSubcontract


class ModelContractLoadResult(BaseModel):
    """Result of loading a single contract.yaml file.

    Captures the outcome of loading a contract, including any subcontracts
    that were successfully loaded or any errors that occurred.

    Attributes:
        contract_path: Path to the contract.yaml file that was loaded.
        success: True if the contract was loaded successfully.
        error: Error message if loading failed, empty string otherwise.
        handler_routing: Loaded handler routing subcontract, or None if not present.
        operation_bindings: Loaded operation bindings subcontract, or None if not present.
        correlation_id: Correlation ID for tracing this load operation.

    Example:
        >>> result = ModelContractLoadResult.succeeded(
        ...     contract_path=Path("nodes/auth/contract.yaml"),
        ...     handler_routing=routing,
        ...     operation_bindings=bindings,
        ... )
        >>> result.success
        True
        >>> result.handler_routing is not None
        True
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    contract_path: Path = Field(
        ...,
        description="Path to the contract.yaml file that was loaded.",
    )
    success: bool = Field(
        ...,
        description="True if the contract was loaded successfully.",
    )
    error: str = Field(
        default="",
        description="Error message if loading failed, empty string otherwise.",
    )
    handler_routing: ModelRoutingSubcontract | None = Field(
        default=None,
        description="Loaded handler routing subcontract, or None if not present.",
    )
    operation_bindings: ModelOperationBindingsSubcontract | None = Field(
        default=None,
        description="Loaded operation bindings subcontract, or None if not present.",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID for tracing this load operation.",
    )

    @property
    def has_error(self) -> bool:
        """Check if an error message exists.

        Returns:
            True if error is non-empty, False otherwise.
        """
        return bool(self.error)

    @property
    def has_handler_routing(self) -> bool:
        """Check if handler routing was loaded.

        Returns:
            True if handler_routing is not None.
        """
        return self.handler_routing is not None

    @property
    def has_operation_bindings(self) -> bool:
        """Check if operation bindings were loaded.

        Returns:
            True if operation_bindings is not None.
        """
        return self.operation_bindings is not None

    @classmethod
    def succeeded(
        cls,
        contract_path: Path,
        handler_routing: ModelRoutingSubcontract | None = None,
        operation_bindings: ModelOperationBindingsSubcontract | None = None,
        correlation_id: UUID | None = None,
    ) -> ModelContractLoadResult:
        """Create a successful contract load result.

        Args:
            contract_path: Path to the contract.yaml file.
            handler_routing: Loaded handler routing subcontract, if present.
            operation_bindings: Loaded operation bindings subcontract, if present.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelContractLoadResult indicating success.
        """
        return cls(
            contract_path=contract_path,
            success=True,
            error="",
            handler_routing=handler_routing,
            operation_bindings=operation_bindings,
            correlation_id=correlation_id,
        )

    @classmethod
    def failed(
        cls,
        contract_path: Path,
        error: str,
        correlation_id: UUID | None = None,
    ) -> ModelContractLoadResult:
        """Create a failed contract load result.

        Args:
            contract_path: Path to the contract.yaml file.
            error: Description of the error that occurred.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelContractLoadResult indicating failure with error details.
        """
        return cls(
            contract_path=contract_path,
            success=False,
            error=error,
            handler_routing=None,
            operation_bindings=None,
            correlation_id=correlation_id,
        )

    def __bool__(self) -> bool:
        """Allow using result in boolean context.

        Warning:
            **Non-standard __bool__ behavior**: This model overrides ``__bool__`` to
            return ``True`` only when ``success`` is True.

        Returns:
            True if contract was loaded successfully, False otherwise.
        """
        return self.success

    def __str__(self) -> str:
        """Return a human-readable string representation.

        Returns:
            String describing the load result.
        """
        if self.success:
            parts = []
            if self.has_handler_routing:
                parts.append("handler_routing")
            if self.has_operation_bindings:
                parts.append("operation_bindings")
            loaded = ", ".join(parts) if parts else "no subcontracts"
            return f"ModelContractLoadResult({self.contract_path}: {loaded})"
        return f"ModelContractLoadResult({self.contract_path}: FAILED - {self.error})"


__all__ = ["ModelContractLoadResult"]
