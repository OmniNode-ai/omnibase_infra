# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Configuration reference parsing model.

This module provides the ModelConfigRef model for parsing and validating
configuration references that point to external configuration sources.
Handler configs can reference external configuration via the `config_ref` field.

Supported Reference Formats:
    File references (preferred shorthand format)::

        file:configs/handler.yaml              # Relative path (preferred)
        file:/absolute/path/to/config.yaml     # Absolute path
        file://relative/path/config.yaml       # Also supported (backwards compat)

    Environment variable references::

        env:HANDLER_CONFIG_JSON                # JSON config in env var
        env:DATABASE_CONFIG_YAML               # YAML config in env var

    Vault secret references::

        vault:secret/data/handlers/db#config   # Specific field from secret
        vault:secret/data/handlers/db          # Whole secret as JSON/YAML

Security:
    File paths are validated to block dangerous path traversal patterns (multiple
    consecutive ../ sequences). Single parent directory references (../) are allowed
    as legitimate use cases. Full security enforcement happens at the resolution layer
    where resolved paths are validated against the config_dir boundary.
    All schemes are validated against the allowed set.

Design Pattern:
    ModelConfigRefParseResult implements a result pattern where parsing
    never throws exceptions for invalid input. Use the boolean context
    or check `success` to determine if parsing succeeded.

Example:
    >>> from omnibase_infra.runtime.models import ModelConfigRef
    >>>
    >>> # Parse a file reference
    >>> result = ModelConfigRef.parse("file:configs/db.yaml")
    >>> if result:
    ...     print(f"Scheme: {result.config_ref.scheme}")
    ...     print(f"Path: {result.config_ref.path}")
    Scheme: EnumConfigRefScheme.FILE
    Path: configs/db.yaml
    >>>
    >>> # Parse a vault reference with fragment
    >>> result = ModelConfigRef.parse("vault:secret/db#password")
    >>> if result:
    ...     print(f"Fragment: {result.config_ref.fragment}")
    Fragment: password
    >>>
    >>> # Handle invalid input
    >>> result = ModelConfigRef.parse("unknown:path")
    >>> if not result:
    ...     print(f"Error: {result.error_message}")
    Error: Unknown scheme 'unknown'. Supported: file, env, vault

.. versionadded:: 0.8.0
    Initial implementation for OMN-765.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.runtime.enums.enum_config_ref_scheme import EnumConfigRefScheme
from omnibase_infra.runtime.models.model_config_ref_parse_result import (
    ModelConfigRefParseResult,
)

# Pattern to detect dangerous path traversal attempts (multiple consecutive ../)
# Single parent directory references like ../config.yaml are allowed as legitimate use cases.
# Security enforcement happens at the resolution layer where resolved paths are validated
# against the config_dir boundary.
_DANGEROUS_TRAVERSAL_PATTERN = re.compile(r"(\.\.([/\\])\.\.)|(\.\.[/\\]){3,}")


class ModelConfigRef(BaseModel):
    """Parsed configuration reference.

    Represents a structured form of a config_ref string after parsing.
    Use the `parse()` classmethod to create instances from raw strings.

    Attributes:
        scheme: The configuration source type (file, env, vault).
        path: The path portion after the scheme (file path, env var name, or vault path).
        fragment: Optional fragment identifier (for vault:path#field references).

    Examples:
        >>> ref = ModelConfigRef(
        ...     scheme=EnumConfigRefScheme.FILE,
        ...     path="configs/db.yaml"
        ... )
        >>> ref.to_uri()
        'file:configs/db.yaml'
        >>>
        >>> ref = ModelConfigRef(
        ...     scheme=EnumConfigRefScheme.VAULT,
        ...     path="secret/data/db",
        ...     fragment="password"
        ... )
        >>> ref.to_uri()
        'vault:secret/data/db#password'
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    scheme: EnumConfigRefScheme = Field(
        ...,
        description="The configuration source type (file, env, vault).",
    )
    path: str = Field(
        ...,
        min_length=1,
        description="The path portion after the scheme. For file: a filesystem path, "
        "for env: an environment variable name, for vault: a vault path.",
    )
    fragment: str | None = Field(
        default=None,
        description="Optional fragment identifier for vault references (vault:path#field). "
        "Used to extract a specific field from a vault secret.",
    )

    @field_validator("path")
    @classmethod
    def validate_path_no_dangerous_traversal(cls, v: str) -> str:
        """Validate that file paths do not contain dangerous path traversal sequences.

        Args:
            v: The path value to validate.

        Returns:
            The validated path.

        Raises:
            ValueError: If the path contains dangerous path traversal patterns.

        Note:
            Single parent directory references (../) are allowed as legitimate use cases
            for referencing config files in parent directories (e.g., file:../config.yaml).
            Only patterns that could indicate directory escape attacks are blocked
            (e.g., multiple consecutive ../ sequences).

            Full security enforcement happens at the resolution layer where resolved
            paths are validated against the config_dir boundary.
        """
        if _DANGEROUS_TRAVERSAL_PATTERN.search(v):
            raise ValueError(
                f"Dangerous path traversal patterns are not allowed in config paths: {v}"
            )
        return v

    @classmethod
    def parse(cls, config_ref: str) -> ModelConfigRefParseResult:
        """Parse a config_ref string into a structured ModelConfigRef.

        This method never raises exceptions for invalid input. Instead, it returns
        a ModelConfigRefParseResult that can be checked for success.

        Args:
            config_ref: The raw config_ref string to parse. Supported formats:
                - file:path/to/config.yaml (relative path - preferred)
                - file:/absolute/path (absolute path)
                - file://path (also supported for backwards compatibility)
                - env:VAR_NAME (environment variable)
                - vault:path/to/secret (whole vault secret)
                - vault:path/to/secret#field (specific vault field)

        Returns:
            ModelConfigRefParseResult with success=True and config_ref set on success,
            or success=False and error_message set on failure.

        Examples:
            >>> result = ModelConfigRef.parse("file:configs/db.yaml")
            >>> result.success
            True
            >>> result.config_ref.path
            'configs/db.yaml'
            >>>
            >>> result = ModelConfigRef.parse("")
            >>> result.success
            False
            >>> result.error_message
            'Config reference cannot be empty'
        """
        # Handle empty input
        if not config_ref:
            return ModelConfigRefParseResult(
                success=False,
                error_message="Config reference cannot be empty",
            )

        config_ref = config_ref.strip()
        if not config_ref:
            return ModelConfigRefParseResult(
                success=False,
                error_message="Config reference cannot be empty",
            )

        # Find scheme separator
        colon_idx = config_ref.find(":")
        if colon_idx == -1:
            return ModelConfigRefParseResult(
                success=False,
                error_message=f"Invalid config reference format: missing scheme separator ':' in '{config_ref}'",
            )

        scheme_str = config_ref[:colon_idx].lower()
        path_part = config_ref[colon_idx + 1 :]

        # Validate scheme
        try:
            scheme = EnumConfigRefScheme(scheme_str)
        except ValueError:
            valid_schemes = ", ".join(s.value for s in EnumConfigRefScheme)
            return ModelConfigRefParseResult(
                success=False,
                error_message=f"Unknown scheme '{scheme_str}'. Supported: {valid_schemes}",
            )

        # Handle file:// with multiple slashes
        if scheme == EnumConfigRefScheme.FILE:
            # file:///absolute/path -> /absolute/path
            # file://relative/path -> relative/path
            # file:path -> path
            if path_part.startswith("//"):
                # file:// prefix - strip the //
                path_part = path_part[2:]
        elif scheme == EnumConfigRefScheme.ENV:
            # env:VAR_NAME - no special handling needed
            pass
        elif scheme == EnumConfigRefScheme.VAULT:
            # vault:path#fragment - handled below
            pass

        # Handle empty path after scheme
        if not path_part:
            return ModelConfigRefParseResult(
                success=False,
                error_message=f"Missing path after scheme '{scheme_str}:'",
            )

        # Extract fragment for vault references
        fragment: str | None = None
        if scheme == EnumConfigRefScheme.VAULT and "#" in path_part:
            hash_idx = path_part.rfind("#")
            fragment = path_part[hash_idx + 1 :]
            path_part = path_part[:hash_idx]

            if not fragment:
                return ModelConfigRefParseResult(
                    success=False,
                    error_message="Empty fragment after '#' in vault reference",
                )

            if not path_part:
                return ModelConfigRefParseResult(
                    success=False,
                    error_message="Missing vault path before '#' fragment",
                )

        # Validate path for dangerous traversal patterns
        if _DANGEROUS_TRAVERSAL_PATTERN.search(path_part):
            return ModelConfigRefParseResult(
                success=False,
                error_message=f"Dangerous path traversal patterns are not allowed: {path_part}",
            )

        # Create the config ref
        try:
            ref = cls(
                scheme=scheme,
                path=path_part,
                fragment=fragment,
            )
            return ModelConfigRefParseResult(
                success=True,
                config_ref=ref,
            )
        except ValueError as e:
            return ModelConfigRefParseResult(
                success=False,
                error_message=str(e),
            )

    def to_uri(self) -> str:
        """Reconstruct the URI string from the parsed components.

        Returns:
            The config reference as a URI string.

        Examples:
            >>> ref = ModelConfigRef(
            ...     scheme=EnumConfigRefScheme.FILE,
            ...     path="configs/db.yaml"
            ... )
            >>> ref.to_uri()
            'file:configs/db.yaml'
            >>>
            >>> ref = ModelConfigRef(
            ...     scheme=EnumConfigRefScheme.VAULT,
            ...     path="secret/db",
            ...     fragment="password"
            ... )
            >>> ref.to_uri()
            'vault:secret/db#password'
        """
        uri = f"{self.scheme.value}:{self.path}"
        if self.fragment is not None:
            uri = f"{uri}#{self.fragment}"
        return uri


# Rebuild models to resolve forward references from __future__ annotations
ModelConfigRef.model_rebuild()
ModelConfigRefParseResult.model_rebuild()


__all__: list[str] = [
    "ModelConfigRef",
]
