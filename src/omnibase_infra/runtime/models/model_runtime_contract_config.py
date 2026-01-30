# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Runtime contract configuration model for consolidated contract loading.

This module provides the ModelRuntimeContractConfig class for representing
the consolidated result of loading all contract.yaml files at startup.

Part of OMN-1519: Runtime contract config loader.

Design Pattern:
    ModelRuntimeContractConfig aggregates results from loading multiple
    contract.yaml files, providing a single source of truth for all
    handler routing and operation bindings configuration.

Thread Safety:
    ModelRuntimeContractConfig is immutable (frozen=True) after creation,
    making it thread-safe for concurrent read access.

Example:
    >>> from omnibase_infra.runtime.models import ModelRuntimeContractConfig
    >>>
    >>> config = ModelRuntimeContractConfig(
    ...     contract_results=[result1, result2],
    ...     total_contracts_found=5,
    ...     total_contracts_loaded=4,
    ...     total_errors=1,
    ... )
    >>> config.success_rate
    0.8

.. versionadded:: 0.2.8
    Created as part of OMN-1519.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, computed_field

from omnibase_infra.models.bindings import ModelOperationBindingsSubcontract
from omnibase_infra.models.routing import ModelRoutingSubcontract
from omnibase_infra.runtime.models.model_contract_load_result import (
    ModelContractLoadResult,
)


class ModelRuntimeContractConfig(BaseModel):
    """Consolidated runtime contract configuration from all loaded contracts.

    Aggregates the results of loading multiple contract.yaml files,
    providing access to all handler routing and operation bindings
    configuration loaded at startup.

    Attributes:
        contract_results: List of individual contract load results.
        total_contracts_found: Number of contract.yaml files discovered.
        total_contracts_loaded: Number of contracts successfully loaded.
        total_errors: Number of contracts that failed to load.
        correlation_id: Correlation ID for tracing this load session.

    Example:
        >>> config = ModelRuntimeContractConfig(
        ...     contract_results=[result1, result2],
        ...     total_contracts_found=2,
        ...     total_contracts_loaded=2,
        ...     total_errors=0,
        ... )
        >>> config.all_successful
        True
        >>> len(config.handler_routing_configs)
        2
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    contract_results: list[ModelContractLoadResult] = Field(
        default_factory=list,
        description="List of individual contract load results.",
    )
    total_contracts_found: int = Field(
        default=0,
        ge=0,
        description="Number of contract.yaml files discovered.",
    )
    total_contracts_loaded: int = Field(
        default=0,
        ge=0,
        description="Number of contracts successfully loaded.",
    )
    total_errors: int = Field(
        default=0,
        ge=0,
        description="Number of contracts that failed to load.",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID for tracing this load session.",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def all_successful(self) -> bool:
        """Check if all contracts were loaded successfully.

        Returns:
            True if total_errors is 0 and at least one contract was found.
        """
        return self.total_errors == 0 and self.total_contracts_found > 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def success_rate(self) -> float:
        """Calculate the success rate of contract loading.

        Returns:
            Ratio of successfully loaded contracts to total found (0.0-1.0).
            Returns 1.0 if no contracts were found.
        """
        if self.total_contracts_found == 0:
            return 1.0
        return self.total_contracts_loaded / self.total_contracts_found

    @property
    def successful_results(self) -> list[ModelContractLoadResult]:
        """Get all successful contract load results.

        Returns:
            List of ModelContractLoadResult where success is True.
        """
        return [r for r in self.contract_results if r.success]

    @property
    def failed_results(self) -> list[ModelContractLoadResult]:
        """Get all failed contract load results.

        Returns:
            List of ModelContractLoadResult where success is False.
        """
        return [r for r in self.contract_results if not r.success]

    @property
    def handler_routing_configs(self) -> dict[Path, ModelRoutingSubcontract]:
        """Get all loaded handler routing configurations.

        Returns:
            Mapping of contract path to handler routing subcontract.
        """
        return {
            r.contract_path: r.handler_routing
            for r in self.contract_results
            if r.handler_routing is not None
        }

    @property
    def operation_bindings_configs(
        self,
    ) -> dict[Path, ModelOperationBindingsSubcontract]:
        """Get all loaded operation bindings configurations.

        Returns:
            Mapping of contract path to operation bindings subcontract.
        """
        return {
            r.contract_path: r.operation_bindings
            for r in self.contract_results
            if r.operation_bindings is not None
        }

    @property
    def error_messages(self) -> dict[Path, str]:
        """Get all error messages from failed loads.

        Returns:
            Mapping of contract path to error message.
        """
        return {r.contract_path: r.error for r in self.contract_results if r.error}

    def get_routing_for_contract(
        self, contract_path: Path
    ) -> ModelRoutingSubcontract | None:
        """Get handler routing for a specific contract.

        Args:
            contract_path: Path to the contract.yaml file.

        Returns:
            ModelRoutingSubcontract if found, None otherwise.
        """
        for result in self.contract_results:
            if result.contract_path == contract_path:
                return result.handler_routing
        return None

    def get_bindings_for_contract(
        self, contract_path: Path
    ) -> ModelOperationBindingsSubcontract | None:
        """Get operation bindings for a specific contract.

        Args:
            contract_path: Path to the contract.yaml file.

        Returns:
            ModelOperationBindingsSubcontract if found, None otherwise.
        """
        for result in self.contract_results:
            if result.contract_path == contract_path:
                return result.operation_bindings
        return None

    def __bool__(self) -> bool:
        """Allow using config in boolean context.

        Warning:
            **Non-standard __bool__ behavior**: This model overrides ``__bool__`` to
            return ``True`` only when all contracts were loaded successfully.

        Note:
            **Edge case - no contracts found**: Returns ``False`` when
            ``total_contracts_found == 0``, even if ``total_errors == 0``.
            This is intentional: an empty result (no contracts discovered)
            should trigger explicit handling by the caller rather than
            silently proceeding as if everything succeeded.

        Example:
            >>> # Empty config with no errors still returns False
            >>> config = ModelRuntimeContractConfig()
            >>> bool(config)
            False
            >>> config.total_errors
            0
            >>> config.total_contracts_found
            0
            >>>
            >>> # Only returns True when contracts are found AND no errors
            >>> config_with_contracts = ModelRuntimeContractConfig(
            ...     total_contracts_found=1,
            ...     total_contracts_loaded=1,
            ... )
            >>> bool(config_with_contracts)
            True

        Returns:
            True if all contracts loaded successfully (requires at least one
            contract found and zero errors), False otherwise.
        """
        return self.all_successful

    def __str__(self) -> str:
        """Return a human-readable summary.

        Returns:
            String summarizing the load results.
        """
        return (
            f"ModelRuntimeContractConfig("
            f"found={self.total_contracts_found}, "
            f"loaded={self.total_contracts_loaded}, "
            f"errors={self.total_errors}, "
            f"success_rate={self.success_rate:.1%})"
        )


__all__ = ["ModelRuntimeContractConfig"]
