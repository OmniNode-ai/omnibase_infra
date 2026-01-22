# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Source Configuration Model.

This module provides ModelHandlerSourceConfig, a Pydantic model for
configuring handler source mode selection at runtime.

The configuration controls how handlers are discovered and loaded:
    - BOOTSTRAP: Hardcoded handlers from _KNOWN_HANDLERS dict (MVP mode)
    - CONTRACT: YAML contracts from handler_contract.yaml files (production)
    - HYBRID: Contract-first with bootstrap fallback per-handler identity

Production hardening features:
    - Bootstrap expiry enforcement: If bootstrap_expires_at is set and now > expires_at,
      the runtime will refuse to start in BOOTSTRAP mode (or force CONTRACT mode)
    - Structured logging of expiry status at startup
    - Override control for hybrid mode handler resolution

.. versionadded:: 0.7.0
    Created as part of OMN-1095 handler source mode configuration.

See Also:
    - HANDLER_PROTOCOL_DRIVEN_ARCHITECTURE.md: Full architecture documentation
    - EnumHandlerSourceMode: Enum defining valid source modes
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_handler_source_mode import EnumHandlerSourceMode


class ModelHandlerSourceConfig(BaseModel):
    """Configuration for handler source mode selection.

    Controls how handlers are discovered and loaded at runtime. This model
    is used by RuntimeHostProcess and related components to determine the
    handler loading strategy.

    Configuration Options:
        - handler_source_mode: Selects the loading strategy (BOOTSTRAP, CONTRACT, HYBRID)
        - allow_bootstrap_override: Controls handler resolution in HYBRID mode
        - bootstrap_expires_at: Production safety - forces CONTRACT after expiry

    Production Hardening:
        When bootstrap_expires_at is set and the current time exceeds it:
        - BOOTSTRAP mode: Runtime refuses to start (safety mechanism)
        - HYBRID mode: Bootstrap fallback disabled, contract-only resolution
        - CONTRACT mode: No effect (already contract-only)

        This prevents accidental deployment with hardcoded handlers in production.

    Attributes:
        handler_source_mode: Handler loading source mode.
            - BOOTSTRAP: Load from hardcoded _KNOWN_HANDLERS dict (MVP)
            - CONTRACT: Load from handler_contract.yaml files (production)
            - HYBRID: Contract-first with bootstrap fallback per-handler identity
            Defaults to CONTRACT as recommended for production.

        allow_bootstrap_override: If True, bootstrap handlers can override
            contract handlers in HYBRID mode. Default is False, meaning
            contract handlers take precedence (inverse of naive HYBRID).
            Has no effect in BOOTSTRAP or CONTRACT modes.

        bootstrap_expires_at: If set and expired, refuse BOOTSTRAP mode and
            force CONTRACT. This is a production safety mechanism to ensure
            hardcoded handlers are not accidentally deployed to production
            after a migration deadline. Set to None to disable expiry checking.

    Example:
        >>> from datetime import datetime, timedelta
        >>> from omnibase_infra.models.handlers import ModelHandlerSourceConfig
        >>> from omnibase_infra.enums import EnumHandlerSourceMode
        >>>
        >>> # Production configuration (recommended)
        >>> config = ModelHandlerSourceConfig(
        ...     handler_source_mode=EnumHandlerSourceMode.CONTRACT,
        ... )
        >>>
        >>> # Migration configuration with safety expiry
        >>> config = ModelHandlerSourceConfig(
        ...     handler_source_mode=EnumHandlerSourceMode.HYBRID,
        ...     bootstrap_expires_at=datetime(2025, 3, 1, 0, 0, 0),
        ... )
        >>>
        >>> # Check if bootstrap is expired
        >>> if config.is_bootstrap_expired:
        ...     print("Bootstrap mode has expired - must use CONTRACT")
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
    )

    handler_source_mode: EnumHandlerSourceMode = Field(
        default=EnumHandlerSourceMode.CONTRACT,
        description="Handler loading source mode: BOOTSTRAP, CONTRACT, or HYBRID",
    )

    allow_bootstrap_override: bool = Field(
        default=False,
        description=(
            "If True, bootstrap handlers can override contract handlers in HYBRID mode. "
            "Default is False (contract handlers take precedence)."
        ),
    )

    bootstrap_expires_at: datetime | None = Field(
        default=None,
        description=(
            "If set and expired, refuse BOOTSTRAP mode and force CONTRACT. "
            "Production safety mechanism for migration deadlines."
        ),
    )

    @property
    def is_bootstrap_expired(self) -> bool:
        """Check if bootstrap mode has expired.

        Returns:
            True if bootstrap_expires_at is set and current time exceeds it,
            False otherwise.

        Note:
            Uses datetime.now() for comparison. For timezone-aware expiry,
            ensure bootstrap_expires_at is set with appropriate timezone.
        """
        if self.bootstrap_expires_at is None:
            return False
        return datetime.now() > self.bootstrap_expires_at

    @property
    def effective_mode(self) -> EnumHandlerSourceMode:
        """Get the effective handler source mode after expiry check.

        If bootstrap_expires_at is set and expired, returns CONTRACT
        regardless of the configured handler_source_mode. Otherwise
        returns the configured mode.

        Returns:
            The effective handler source mode to use at runtime.

        Note:
            This property should be used by runtime components instead of
            directly accessing handler_source_mode to ensure expiry
            enforcement is applied.
        """
        if self.is_bootstrap_expired:
            return EnumHandlerSourceMode.CONTRACT
        return self.handler_source_mode


__all__ = ["ModelHandlerSourceConfig"]
