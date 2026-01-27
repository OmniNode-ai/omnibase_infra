# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Identity Model for ONEX Infrastructure.

This module provides a typed Pydantic model for uniquely identifying ONEX nodes
within the infrastructure. The identity encapsulates the environment, service,
node name, and version - the four dimensions required to uniquely identify
a node instance.

The model is immutable (frozen) to ensure identity stability throughout a node's
lifecycle. Once created, a node identity cannot be modified.

.. versionadded:: 0.2.6
    Created as part of OMN-1602 typed node identity for introspection.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class ModelNodeIdentity(BaseModel):
    """Typed identity for ONEX infrastructure nodes.

    This model uniquely identifies a node within the ONEX infrastructure using
    four dimensions: environment, service, node name, and version. The same
    node identity can be used with different purposes (e.g., introspection,
    registration, heartbeat) - purpose is passed separately to operations.

    All fields are required and must be non-empty strings without whitespace-only
    values. The model is frozen to ensure identity immutability.

    Attributes:
        env: Environment identifier (e.g., "dev", "staging", "prod").
            Determines which infrastructure deployment the node belongs to.
        service: Service name from the node's contract (e.g., "omniintelligence",
            "omnibridge"). Groups related nodes under a common service boundary.
        node_name: Node name from the contract (e.g., "claude_hook_event_effect",
            "registration_orchestrator"). Uniquely identifies the node within
            its service.
        version: Version string for the node (e.g., "v1", "v2.0.0").
            Enables version-aware routing and registration. While any non-empty
            string is accepted, semver-style prefixed with 'v' is recommended
            for consistency (e.g., 'v1', 'v1.0.0', 'v2.1.3').

    Example:
        >>> identity = ModelNodeIdentity(
        ...     env="dev",
        ...     service="omniintelligence",
        ...     node_name="claude_hook_event_effect",
        ...     version="v1",
        ... )
        >>> identity.env
        'dev'
        >>> identity.service
        'omniintelligence'

        The model is immutable - attempting to modify raises an error:

        >>> identity.env = "prod"  # doctest: +SKIP
        Traceback (most recent call last):
            ...
        pydantic_core._pydantic_core.ValidationError: ...

        Empty or whitespace-only values are rejected:

        >>> ModelNodeIdentity(env="", service="svc", node_name="node", version="v1")
        Traceback (most recent call last):
            ...
        pydantic_core._pydantic_core.ValidationError: ...

    Note:
        The `purpose` field is intentionally NOT included in this model.
        Purpose (e.g., "introspection", "registration") is passed separately
        to operations because the same node identity can be used for multiple
        purposes.

    .. versionadded:: 0.2.6
        Created as part of OMN-1602.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    env: str = Field(
        description="Environment identifier (e.g., 'dev', 'staging', 'prod')",
    )
    service: str = Field(
        description="Service name from the node's contract (e.g., 'omniintelligence')",
    )
    node_name: str = Field(  # pattern-ok: canonical identifier, not a foreign key reference
        description="Node name from the contract (e.g., 'claude_hook_event_effect')",
    )
    version: str = Field(
        description="Version string for the node (e.g., 'v1', 'v2.0.0')",
    )

    @field_validator("env", "service", "node_name", "version", mode="after")
    @classmethod
    def _validate_non_empty(cls, v: str, info: ValidationInfo) -> str:
        """Validate that string fields are non-empty and not whitespace-only.

        Args:
            v: The string value to validate.
            info: Pydantic validation context containing field name.

        Returns:
            The validated string value.

        Raises:
            ValueError: If the value is empty or contains only whitespace.
        """
        field_name = info.field_name
        if not v:
            msg = f"'{field_name}' must not be empty"
            raise ValueError(msg)
        if not v.strip():
            msg = f"'{field_name}' must not contain only whitespace"
            raise ValueError(msg)
        return v


__all__: list[str] = ["ModelNodeIdentity"]
