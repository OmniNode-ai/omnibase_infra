# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Gateway Configuration Model.

This module provides the Pydantic model for gateway configuration, including
realm routing, runtime identification, and Ed25519 signing keypair paths.

The gateway is responsible for:
    - Signing outbound envelopes with Ed25519 signatures
    - Validating inbound envelope signatures
    - Enforcing realm boundaries for message routing
    - Topic allowlisting for security

Example:
    >>> from pathlib import Path
    >>> config = ModelGatewayConfig(
    ...     realm="dev",
    ...     runtime_id="runtime-dev-001",
    ...     private_key_path=Path("/etc/onex/keys/private.pem"),
    ...     public_key_path=Path("/etc/onex/keys/public.pem"),
    ... )
    >>> print(config.realm)
    'dev'

"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelGatewayConfig(BaseModel):
    """Gateway configuration model.

    Configures the runtime gateway for envelope signing and realm enforcement.
    The gateway acts as the security boundary for all inbound and outbound
    messages in the ONEX runtime.

    Attributes:
        realm: Routing boundary identifier (e.g., "dev", "staging", "prod").
            Messages are routed within realm boundaries to prevent cross-realm
            contamination. Required for gateway operation.
        runtime_id: Unique identifier for this gateway instance (e.g.,
            "runtime-dev-001"). Used for message attribution and debugging.
            Required for gateway operation.
        private_key_path: Path to the Ed25519 private key PEM file for signing
            outbound envelopes. Optional - if not provided, signing is disabled.
        public_key_path: Path to the Ed25519 public key PEM file for verifying
            inbound envelope signatures. Optional - if not provided, verification
            is disabled.
        enabled: Whether gateway signing and validation is enabled. When False,
            the gateway passes messages through without modification. Defaults
            to True.
        allowed_topics: List of topic patterns allowed for publishing. Empty list
            means all topics are allowed. Supports glob patterns (e.g., "events.*").
        reject_unsigned: Whether to reject inbound messages that lack a valid
            signature. When True (default), unsigned messages are rejected.
            When ``public_key_path`` is configured, inbound signatures are
            verified cryptographically. When no public key is configured, the
            runtime still enforces ``reject_unsigned`` by rejecting any
            envelope that *looks* signed but cannot be verified. Set to
            False during migration periods or for development.

    Security Considerations:
        - Private keys should have restricted file permissions (0600)
        - Key paths should be absolute to prevent path traversal
        - Realm names should be validated against a known allowlist in production
        - Topic allowlisting provides defense-in-depth against misconfiguration

    Example:
        >>> # Development configuration (signing disabled)
        >>> dev_config = ModelGatewayConfig(
        ...     realm="dev",
        ...     runtime_id="runtime-local",
        ...     enabled=False,
        ... )

        >>> # Production configuration (full signing)
        >>> from pathlib import Path
        >>> prod_config = ModelGatewayConfig(
        ...     realm="prod",
        ...     runtime_id="runtime-prod-001",
        ...     private_key_path=Path("/etc/onex/keys/private.pem"),
        ...     public_key_path=Path("/etc/onex/keys/public.pem"),
        ...     allowed_topics=["events.*", "commands.*"],
        ...     reject_unsigned=True,
        ... )

    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,  # Support pytest-xdist compatibility
    )

    realm: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Routing boundary identifier (e.g., 'dev', 'staging', 'prod')",
    )
    # Note: runtime_id is intentionally str (not UUID) per OMN-1898 design.
    # It's a human-readable gateway identifier like "runtime-dev-001", not a UUID.
    runtime_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Unique gateway instance identifier (e.g., 'runtime-dev-001')",
    )
    private_key_path: Path | None = Field(
        default=None,
        description="Path to Ed25519 private key PEM file for signing outbound envelopes",
    )
    public_key_path: Path | None = Field(
        default=None,
        description="Path to Ed25519 public key PEM file for verifying inbound signatures",
    )
    enabled: bool = Field(
        default=True,
        description="Whether gateway signing and validation is enabled",
    )
    allowed_topics: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Topic patterns allowed for publishing (empty = allow all)",
    )

    @field_validator("allowed_topics", mode="before")
    @classmethod
    def _coerce_allowed_topics(cls, v: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        """Coerce list inputs to tuple for ``strict=True`` compatibility.

        With ``strict=True``, Pydantic refuses to coerce a ``list`` into a
        ``tuple`` automatically. This validator accepts either type so that
        callers can pass ``["events.*"]`` without triggering a validation
        error.

        Args:
            v: The raw allowed_topics value (list or tuple of strings).

        Returns:
            A tuple of topic pattern strings.
        """
        if isinstance(v, list):
            return tuple(v)
        return v

    reject_unsigned: bool = Field(
        default=True,
        description=(
            "Whether to reject inbound messages lacking valid signatures. "
            "When public_key_path is configured, signatures are verified "
            "cryptographically. Without a public key, signed envelopes "
            "are still rejected because they cannot be verified."
        ),
    )

    @field_validator("private_key_path", "public_key_path", mode="before")
    @classmethod
    def _validate_key_path_is_absolute(cls, v: str | Path | None) -> Path | None:
        """Validate that key paths are absolute to prevent path traversal.

        With ``strict=True``, Pydantic refuses to coerce a ``str`` into a
        ``Path`` automatically. This validator accepts both types, converts
        to ``Path``, and validates that the path is absolute.

        Args:
            v: The path value to validate (str or Path), or None if not
                provided.

        Returns:
            The validated ``Path`` object, or ``None`` if not provided.

        Raises:
            ValueError: If the path is not absolute.
        """
        if v is not None:
            path = Path(v) if not isinstance(v, Path) else v
            if not path.is_absolute():
                msg = (
                    f"Key path must be absolute to prevent path traversal, "
                    f"got relative path: {path}"
                )
                raise ValueError(msg)
            return path
        return v


__all__: list[str] = ["ModelGatewayConfig"]
