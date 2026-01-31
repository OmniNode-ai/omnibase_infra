# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract Publisher Error Classes.

This module defines error classes specific to contract publishing operations.
All error classes extend from RuntimeHostError to maintain consistency with
ONEX infrastructure error handling patterns.

Error Hierarchy:
    RuntimeHostError (from omnibase_infra.errors)
    └── ContractPublisherError (base contract publishing error)
        ├── ContractSourceNotConfiguredError
        ├── ContractPublishingInfraError
        └── NoContractsFoundError

Related:
    - OMN-1752: Extract ContractPublisher to omnibase_infra
    - ARCH-002: Runtime owns all Kafka plumbing

.. versionadded:: 0.3.0
    Created as part of OMN-1752 (ContractPublisher extraction).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.models.errors import ModelInfraErrorContext

if TYPE_CHECKING:
    from omnibase_infra.services.contract_publisher.models import ModelInfraError


class ContractPublisherError(RuntimeHostError):
    """Base error class for contract publishing errors.

    All contract publishing-specific errors should inherit from this class.
    Provides common context for contract publishing operations.

    Example:
        >>> raise ContractPublisherError("Publishing failed")

    .. versionadded:: 0.3.0
    """

    def __init__(
        self,
        message: str,
        error_code: EnumCoreErrorCode | None = None,
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize ContractPublisherError.

        Args:
            message: Human-readable error message
            error_code: Error code (defaults to OPERATION_FAILED)
            context: Bundled infrastructure context
            **extra_context: Additional context information
        """
        # Default to KAFKA transport type for contract publishing
        if context is None:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="contract_publish",
            )

        super().__init__(
            message=message,
            error_code=error_code or EnumCoreErrorCode.OPERATION_FAILED,
            context=context,
            **extra_context,
        )


class ContractSourceNotConfiguredError(ContractPublisherError):
    """Raised when no contract source is configured.

    This error indicates a configuration problem - the ContractPublisherConfig
    does not have the required source configuration for the selected mode.

    Examples:
        - mode="filesystem" but filesystem_root is None
        - mode="package" but package_module is None
        - mode="composite" but neither source is configured

    Example:
        >>> raise ContractSourceNotConfiguredError(
        ...     mode="filesystem",
        ...     missing_field="filesystem_root",
        ... )

    .. versionadded:: 0.3.0
    """

    def __init__(
        self,
        mode: str,
        missing_field: str,
        message: str | None = None,
    ) -> None:
        """Initialize ContractSourceNotConfiguredError.

        Args:
            mode: The configured mode (filesystem, package, composite)
            missing_field: The field that is missing
            message: Optional custom message
        """
        self.mode = mode
        self.missing_field = missing_field

        default_message = (
            f"Contract source not configured: mode='{mode}' requires '{missing_field}'"
        )

        super().__init__(
            message=message or default_message,
            error_code=EnumCoreErrorCode.CONFIGURATION_ERROR,
            mode=mode,
            missing_field=missing_field,
        )


class ContractPublishingInfraError(ContractPublisherError):
    """Raised when infrastructure fails during contract publishing.

    This error wraps one or more ModelInfraError instances that occurred
    during the publishing process. It is raised when fail_fast=True and
    an infrastructure error occurs.

    Attributes:
        infra_errors: List of infrastructure errors that occurred

    Example:
        >>> from omnibase_infra.services.contract_publisher.models import ModelInfraError
        >>> error = ModelInfraError(
        ...     error_type="publisher_unavailable",
        ...     message="Event bus publisher not available",
        ...     retriable=False,
        ... )
        >>> raise ContractPublishingInfraError([error])

    .. versionadded:: 0.3.0
    """

    def __init__(
        self,
        infra_errors: list[ModelInfraError],
        message: str | None = None,
    ) -> None:
        """Initialize ContractPublishingInfraError.

        Args:
            infra_errors: List of infrastructure errors that occurred
            message: Optional custom message
        """
        self.infra_errors = infra_errors

        # Build message from errors if not provided
        if message is None:
            error_count = len(infra_errors)
            if error_count == 1:
                message = f"Contract publishing failed: {infra_errors[0].message}"
            else:
                message = f"Contract publishing failed with {error_count} infrastructure errors"

        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.OPERATION_FAILED,
            error_count=len(infra_errors),
        )


class NoContractsFoundError(ContractPublisherError):
    """Raised when no contracts are found and allow_zero_contracts=False.

    This error indicates that:
    - Discovery found zero contract files, OR
    - Discovery found files but all failed validation (published_count == 0)

    Both cases result in "no contracts published" which may indicate
    a configuration or deployment problem.

    Attributes:
        source_description: Description of the source that was searched
        discovered_count: Number of contracts discovered (may be > 0 if all failed)
        valid_count: Number of valid contracts (always 0 when raised)

    Example:
        >>> raise NoContractsFoundError(
        ...     source_description="filesystem: /app/contracts",
        ...     discovered_count=0,
        ... )

    .. versionadded:: 0.3.0
    """

    def __init__(
        self,
        source_description: str,
        discovered_count: int = 0,
        valid_count: int = 0,
        message: str | None = None,
    ) -> None:
        """Initialize NoContractsFoundError.

        Args:
            source_description: Description of the source searched
            discovered_count: Number of contracts discovered
            valid_count: Number of valid contracts (should be 0)
            message: Optional custom message
        """
        self.source_description = source_description
        self.discovered_count = discovered_count
        self.valid_count = valid_count

        if message is None:
            if discovered_count == 0:
                message = f"No contracts found in {source_description}"
            else:
                message = (
                    f"No valid contracts: discovered {discovered_count} but "
                    f"all failed validation in {source_description}"
                )

        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.RESOURCE_NOT_FOUND,
            source_description=source_description,
            discovered_count=discovered_count,
            valid_count=valid_count,
        )


__all__ = [
    "ContractPublisherError",
    "ContractSourceNotConfiguredError",
    "ContractPublishingInfraError",
    "NoContractsFoundError",
]
