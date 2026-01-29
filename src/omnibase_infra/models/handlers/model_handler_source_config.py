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

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
            Defaults to HYBRID as recommended for gradual migration.

        allow_bootstrap_override: If True, bootstrap handlers can override
            contract handlers in HYBRID mode. Default is False, meaning
            contract handlers take precedence (inverse of naive HYBRID).
            Has no effect in BOOTSTRAP or CONTRACT modes.
            When parsing from environment or config, string values "true",
            "yes", "1", "on" (case-insensitive) are accepted as truthy.

        bootstrap_expires_at: If set and expired, refuse BOOTSTRAP mode and
            force CONTRACT. This is a production safety mechanism to ensure
            hardcoded handlers are not accidentally deployed to production
            after a migration deadline. Set to None to disable expiry checking.

    Example:
        >>> from datetime import datetime, timezone
        >>> from omnibase_infra.models.handlers import ModelHandlerSourceConfig
        >>> from omnibase_infra.enums import EnumHandlerSourceMode
        >>>
        >>> # Production configuration (recommended)
        >>> config = ModelHandlerSourceConfig(
        ...     handler_source_mode=EnumHandlerSourceMode.CONTRACT,
        ... )
        >>>
        >>> # Migration configuration with safety expiry (must be timezone-aware)
        >>> config = ModelHandlerSourceConfig(
        ...     handler_source_mode=EnumHandlerSourceMode.HYBRID,
        ...     bootstrap_expires_at=datetime(2025, 3, 1, 0, 0, 0, tzinfo=timezone.utc),
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
        default=EnumHandlerSourceMode.HYBRID,
        description="Handler loading source mode: BOOTSTRAP, CONTRACT, or HYBRID. "
        "Defaults to HYBRID (contract-first with bootstrap fallback).",
    )

    allow_bootstrap_override: bool = Field(
        default=False,
        description=(
            "If True, bootstrap handlers can override contract handlers in HYBRID mode. "
            "Default is False (contract handlers take precedence). "
            "When parsing from config, string values 'true', 'yes', '1', 'on' "
            "(case-insensitive) are accepted as truthy."
        ),
    )

    bootstrap_expires_at: datetime | None = Field(
        default=None,
        description=(
            "If set and expired, refuse BOOTSTRAP mode and force CONTRACT. "
            "Production safety mechanism for migration deadlines. "
            "Must be timezone-aware (UTC recommended); naive datetimes are rejected."
        ),
    )

    @field_validator("allow_bootstrap_override", mode="before")
    @classmethod
    def _coerce_allow_bootstrap_override(cls, value: object) -> bool:
        """Coerce string and numeric values to boolean for config file compatibility.

        Environment variables and YAML/JSON config files often represent booleans
        as strings. This validator handles common truthy string representations
        before Pydantic's strict type validation.

        Args:
            value: The raw value to coerce (may be str, bool, int, float, None, or other).

        Returns:
            True if value is a truthy string ("true", "yes", "1", "on") or
            a truthy boolean, False otherwise.

        Type Handling:
            - bool: Passed through unchanged.
            - str: Case-insensitive check for "true", "yes", "1", "on".
            - int/float: 0 and 0.0 are False, all other numbers are True.
            - None: Returns False.
            - Unknown types: Default to False for safety.
        """
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "yes", "1", "on")
        if isinstance(value, (int, float)):
            # Explicit: 0/0.0 = False, any other number = True
            return bool(value)
        # Unknown types default to False for safety
        return False

    @field_validator("bootstrap_expires_at")
    @classmethod
    def _validate_expires_at_timezone(cls, value: datetime | None) -> datetime | None:
        """Validate and normalize bootstrap_expires_at to UTC.

        Args:
            value: The datetime value to validate.

        Returns:
            None if value is None, otherwise the datetime normalized to UTC.

        Raises:
            ValueError: If the datetime is naive (no timezone info).
        """
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(
                "bootstrap_expires_at must be timezone-aware (UTC recommended). "
                "Use datetime.now(timezone.utc) or datetime(..., tzinfo=timezone.utc)."
            )
        return value.astimezone(UTC)

    @property
    def is_bootstrap_expired(self) -> bool:
        """Check if bootstrap mode has expired.

        Returns:
            True if bootstrap_expires_at is set and current time exceeds it,
            False otherwise.

        Note:
            Uses UTC-aware comparison. The bootstrap_expires_at field is
            validated and normalized to UTC at construction time.
        """
        if self.bootstrap_expires_at is None:
            return False
        return datetime.now(UTC) > self.bootstrap_expires_at

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
