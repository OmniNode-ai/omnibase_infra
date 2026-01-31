# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract Publisher Configuration Model.

This module defines the configuration model for contract publishing operations.
Configuration is validated at construction time using Pydantic model validators.

Configuration Options:
    mode: Publishing source mode (filesystem, package, composite)
    filesystem_root: Root directory for filesystem mode
    package_module: Module name for package mode
    fail_fast: Whether to raise on infrastructure errors
    allow_zero_contracts: Whether to allow empty publish results
    environment: Environment prefix for Kafka topics

Environment Resolution:
    The environment is resolved with precedence:
    1. config.environment (if provided)
    2. ONEX_ENV environment variable
    3. Default "dev"

Related:
    - OMN-1752: Extract ContractPublisher to omnibase_infra
    - ARCH-002: Runtime owns all Kafka plumbing

.. versionadded:: 0.3.0
    Created as part of OMN-1752 (ContractPublisher extraction).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelContractPublisherConfig(BaseModel):
    """Configuration for contract publishing.

    Defines the source mode and configuration for discovering and
    publishing contracts to Kafka.

    Source Modes:
        filesystem: Discover contracts from a directory tree
        package: Discover contracts from installed package resources
        composite: Merge both sources with conflict detection

    Configuration Rules (enforced by validator):
        - filesystem mode requires filesystem_root
        - package mode requires package_module
        - composite mode requires at least one of filesystem_root or package_module

    Attributes:
        mode: Publishing source mode
        filesystem_root: Root directory for filesystem discovery
        package_module: Module name for package resource discovery
        fail_fast: If True, raise immediately on infrastructure errors
        allow_zero_contracts: If True, allow empty publish results
        environment: Environment prefix for topics (defaults via resolve_environment)

    Example:
        >>> config = ModelContractPublisherConfig(
        ...     mode="filesystem",
        ...     filesystem_root=Path("/app/contracts/handlers"),
        ...     fail_fast=True,
        ...     allow_zero_contracts=False,
        ... )
        >>> env = config.resolve_environment()
        >>> print(f"Publishing to {env}.onex.evt.contract-registered.v1")

    .. versionadded:: 0.3.0
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["filesystem", "package", "composite"] = Field(
        description="Publishing source mode"
    )
    filesystem_root: Path | None = Field(
        default=None,
        description="Root directory for filesystem discovery",
    )
    package_module: str | None = Field(
        default=None,
        description="Module name for package resource discovery (e.g., 'myapp.contracts')",
    )
    fail_fast: bool = Field(
        default=True,
        description="If True, raise immediately on infrastructure errors",
    )
    allow_zero_contracts: bool = Field(
        default=False,
        description="If True, allow empty publish results without raising",
    )
    environment: str | None = Field(
        default=None,
        description="Environment prefix for topics (resolved via resolve_environment)",
    )

    @model_validator(mode="after")
    def validate_source_configured(self) -> Self:
        """Validate that required source fields are configured for the mode.

        Rules:
            - filesystem mode requires filesystem_root
            - package mode requires package_module
            - composite mode requires at least one source

        Returns:
            Self if validation passes

        Raises:
            ValueError: If required source configuration is missing
        """
        match self.mode:
            case "filesystem":
                if not self.filesystem_root:
                    raise ValueError("filesystem mode requires filesystem_root")
            case "package":
                if not self.package_module:
                    raise ValueError("package mode requires package_module")
            case "composite":
                if not self.filesystem_root and not self.package_module:
                    raise ValueError(
                        "composite mode requires at least one source "
                        "(filesystem_root or package_module)"
                    )
        return self

    def resolve_environment(self) -> str:
        """Resolve environment with precedence: config > env var > 'dev'.

        Resolution Order:
            1. self.environment (if provided and non-empty)
            2. ONEX_ENV environment variable (if set and non-empty)
            3. Default "dev"

        Normalization:
            - Whitespace is stripped
            - Trailing dots are removed (to prevent "dev..topic" issues)

        Returns:
            Resolved environment string, normalized

        Example:
            >>> config = ModelContractPublisherConfig(
            ...     mode="filesystem",
            ...     filesystem_root=Path("/app"),
            ...     environment="prod",
            ... )
            >>> config.resolve_environment()
            'prod'

            >>> # With no config.environment and ONEX_ENV=staging
            >>> config2 = ModelContractPublisherConfig(
            ...     mode="filesystem",
            ...     filesystem_root=Path("/app"),
            ... )
            >>> # Returns "staging" if ONEX_ENV is set, else "dev"
        """
        # Priority 1: Explicit config
        if self.environment:
            return self._normalize_environment(self.environment)

        # Priority 2: Environment variable
        env_var = os.getenv("ONEX_ENV", "")
        if env_var:
            return self._normalize_environment(env_var)

        # Priority 3: Default
        return "dev"

    @staticmethod
    def _normalize_environment(value: str) -> str:
        """Normalize environment string.

        Strips whitespace and removes trailing dots to prevent
        topic formatting issues like "dev..topic".

        Args:
            value: Raw environment value

        Returns:
            Normalized environment string
        """
        return value.strip().rstrip(".")


__all__ = ["ModelContractPublisherConfig"]
