# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Architecture validation request model for the architecture validator node.

This module defines the input model for requesting architecture validation.
Callers can specify paths to validate, specific rules to check, and
whether to fail fast on the first violation.

Example:
    >>> from omnibase_infra.nodes.architecture_validator.models import (
    ...     ModelArchitectureValidationRequest,
    ... )
    >>> # Validate specific files with all rules
    >>> request = ModelArchitectureValidationRequest(
    ...     paths=["src/mymodule/service.py", "src/mymodule/models/"],
    ... )
    >>> # Validate with specific rules only
    >>> request = ModelArchitectureValidationRequest(
    ...     paths=["src/"],
    ...     rule_ids=["ARCH-001", "ARCH-002"],
    ...     fail_fast=True,
    ... )
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelArchitectureValidationRequest(BaseModel):
    """Request to validate architecture patterns in code.

    This model represents a validation request containing the paths
    to validate, optional rule filters, and execution options.

    Attributes:
        paths: List of paths to validate. Can be files, directories,
            or glob patterns (e.g., "src/**/*.py"). If empty, validation
            may use a default scope or fail depending on implementation.
        rule_ids: Optional list of specific rule IDs to check.
            If empty, all available rules are applied.
        fail_fast: If True, stop validation on the first violation found.
            Useful for quick CI checks. Default is False (check all).

    Example:
        >>> # Validate entire src directory
        >>> request = ModelArchitectureValidationRequest(
        ...     paths=["src/"],
        ... )
        >>> # Validate with specific rules and fail fast
        >>> request = ModelArchitectureValidationRequest(
        ...     paths=["src/omnibase_infra/"],
        ...     rule_ids=["ARCH-001", "ARCH-003"],
        ...     fail_fast=True,
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    paths: list[str] = Field(
        default_factory=list,
        description=(
            "Paths to validate. Can be files, directories, or glob patterns. "
            "If empty, validation scope depends on implementation."
        ),
    )
    rule_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Optional list of specific rule IDs to check (e.g., ['ARCH-001']). "
            "If empty, all available rules are applied."
        ),
    )
    fail_fast: bool = Field(
        default=False,
        description=(
            "If True, stop validation on the first violation found. "
            "Useful for quick CI checks."
        ),
    )

    @property
    def has_paths(self) -> bool:
        """Check if any paths are specified for validation.

        Returns:
            True if paths list is non-empty, False otherwise.
        """
        return len(self.paths) > 0

    @property
    def has_rule_filter(self) -> bool:
        """Check if specific rules are requested.

        Returns:
            True if rule_ids list is non-empty, False otherwise.
        """
        return len(self.rule_ids) > 0


__all__ = ["ModelArchitectureValidationRequest"]
