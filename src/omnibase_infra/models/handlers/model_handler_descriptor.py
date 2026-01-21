# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Descriptor Model for Filesystem Discovery.

This module provides ModelHandlerDescriptor, which represents a discovered handler
from a contract file. Part of OMN-1097 HandlerContractSource filesystem discovery.

The descriptor captures essential metadata about a handler discovered from a
handler_contract.yaml file, suitable for registration and validation.

See Also:
    - HandlerContractSource: Source that discovers these descriptors
    - ModelHandlerContract: Contract model from omnibase_core
    - ModelContractDiscoveryResult: Result model with graceful error handling

.. versionadded:: 0.6.2
    Created as part of OMN-1097 filesystem handler discovery.

.. versionchanged:: 0.6.3
    Changed version field from str to ModelSemVer for better type safety
    and consistency with rest of codebase.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_core.types import JsonType


def _parse_version(v: ModelSemVer | str | dict[str, JsonType]) -> ModelSemVer:
    """Parse version input to ModelSemVer.

    Accepts ModelSemVer instance, string (e.g., "1.0.0"), or dict
    (e.g., {"major": 1, "minor": 0, "patch": 0}).

    This validator enables flexible input while maintaining ModelSemVer
    as the canonical type, supporting:
    - Direct ModelSemVer instances (programmatic use)
    - String versions (from YAML contract files)
    - Dict representations (from JSON/dict sources)

    Dict Validation:
        When a dict is provided, the validator explicitly checks for:
        1. Required keys ("major", "minor", "patch") are present
        2. Required key values are integers (not strings or other types)

        This provides clearer error messages before delegating to model_validate(),
        rather than relying on Pydantic's generic validation errors.

    Raises:
        ValueError: If dict input is missing required keys (major, minor, patch),
            or if required key values are not integers.
    """
    if isinstance(v, ModelSemVer):
        return v
    if isinstance(v, str):
        return ModelSemVer.parse(v)
    if isinstance(v, dict):
        required_keys = {"major", "minor", "patch"}
        if not required_keys.issubset(v.keys()):
            missing = required_keys - v.keys()
            raise ValueError(f"Version dict missing required keys: {missing}")
        # Validate value types before delegating to model_validate()
        for key in required_keys:
            value = v[key]
            if not isinstance(value, int):
                raise ValueError(
                    f"Version '{key}' must be an integer, "
                    f"got {type(value).__name__}: {value!r}"
                )
        return ModelSemVer.model_validate(v)
    # NOTE: Return as-is for unhandled types and let Pydantic's own
    # validation raise an appropriate error with type information.
    return v  # type: ignore[return-value]  # NOTE: Pydantic validation handles unknown types


# Annotated type for version field that accepts string, dict, or ModelSemVer
_VersionField = Annotated[ModelSemVer, BeforeValidator(_parse_version)]

# Type alias for valid ONEX handler kinds
LiteralHandlerKind = Literal["compute", "effect", "reducer", "orchestrator"]


class ModelHandlerDescriptor(BaseModel):
    """Handler descriptor model representing a discovered handler.

    This model captures the essential metadata about a handler discovered
    from a contract file, suitable for registration and validation.

    Attributes:
        handler_id: Unique identifier for the handler.
        name: Human-readable name for the handler.
        version: Semantic version (ModelSemVer). Accepts string, dict, or ModelSemVer.
        handler_kind: Handler kind (compute, effect, reducer, orchestrator).
        input_model: Fully qualified input model class path.
        output_model: Fully qualified output model class path.
        description: Optional description of the handler.
        handler_class: Fully qualified Python class path for dynamic handler import.
        contract_path: Path to the source contract file.

    Example:
        Create with string version (common when loading from YAML):

        >>> descriptor = ModelHandlerDescriptor(
        ...     handler_id="auth.handler",
        ...     name="Auth Handler",
        ...     version="1.0.0",
        ...     handler_kind="compute",
        ...     input_model="auth.models.AuthInput",
        ...     output_model="auth.models.AuthOutput",
        ... )
        >>> descriptor.version.major
        1

        Create with ModelSemVer instance:

        >>> from omnibase_core.models.primitives.model_semver import ModelSemVer
        >>> descriptor = ModelHandlerDescriptor(
        ...     handler_id="auth.handler",
        ...     name="Auth Handler",
        ...     version=ModelSemVer(major=2, minor=1, patch=0, prerelease=("beta", 1)),
        ...     handler_kind="compute",
        ...     input_model="auth.models.AuthInput",
        ...     output_model="auth.models.AuthOutput",
        ... )
        >>> str(descriptor.version)
        '2.1.0-beta.1'

    .. versionadded:: 0.6.2
        Created as part of OMN-1097 filesystem handler discovery.

    .. versionchanged:: 0.6.3
        Changed version field from str to ModelSemVer for better type safety.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    handler_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the handler",
    )
    name: str = Field(
        ...,
        min_length=1,
        description="Human-readable name for the handler",
    )
    version: _VersionField = Field(
        ...,
        description="Semantic version (accepts string, dict, or ModelSemVer instance)",
    )
    handler_kind: LiteralHandlerKind = Field(
        ...,
        description="Handler architectural kind (compute, effect, reducer, orchestrator)",
    )
    input_model: str = Field(
        ...,
        min_length=3,
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)+$",
        description="Fully qualified input model class path (e.g., 'myapp.models.User')",
    )
    output_model: str = Field(
        ...,
        min_length=3,
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)+$",
        description="Fully qualified output model class path (e.g., 'myapp.models.Result')",
    )
    description: str | None = Field(
        default=None,
        description="Optional description of the handler",
    )
    handler_class: str | None = Field(
        default=None,
        min_length=3,
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)+$",
        description="Fully qualified Python class path for dynamic handler import (e.g., 'omnibase_infra.handlers.handler_consul.HandlerConsul')",
    )
    contract_path: str | None = Field(
        default=None,
        description="Path to the source contract file",
    )
    contract_config: dict[str, JsonType] | None = Field(
        default=None,
        description=(
            "Parsed configuration from the handler contract file. "
            "Contains extracted values like security settings, tags, and handler metadata. "
            "Populated during handler discovery when contract_path is set."
        ),
    )


__all__ = ["LiteralHandlerKind", "ModelHandlerDescriptor"]
