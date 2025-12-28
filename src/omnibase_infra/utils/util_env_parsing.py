# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Environment variable parsing utilities for ONEX infrastructure.

This module provides type-safe environment variable parsing functions with:
    - Comprehensive error handling using ONEX error patterns
    - Range validation with configurable min/max values
    - Transport-aware error context for debugging
    - Security-conscious value redaction in error messages

Environment variable parsing is a common operation across infrastructure handlers
and configuration models. This module centralizes the parsing logic to reduce
code duplication and ensure consistent error handling.

Security Note:
    Invalid values are always redacted in error messages to prevent potential
    exposure of sensitive configuration values in logs or error outputs.

Example:
    >>> from omnibase_infra.utils import parse_env_int, parse_env_float
    >>> from omnibase_infra.enums import EnumInfraTransportType
    >>>
    >>> # Parse an integer with range validation
    >>> pool_size = parse_env_int(
    ...     "ONEX_DB_POOL_SIZE",
    ...     default=5,
    ...     min_value=1,
    ...     max_value=100,
    ...     transport_type=EnumInfraTransportType.DATABASE,
    ...     service_name="db_handler",
    ... )
    >>>
    >>> # Parse a float with range validation
    >>> timeout = parse_env_float(
    ...     "ONEX_HTTP_TIMEOUT",
    ...     default=30.0,
    ...     min_value=0.1,
    ...     max_value=3600.0,
    ...     transport_type=EnumInfraTransportType.HTTP,
    ...     service_name="http_handler",
    ... )
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from omnibase_infra.enums import EnumInfraTransportType

logger = logging.getLogger(__name__)


def parse_env_int(
    env_var: str,
    default: int,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
    transport_type: EnumInfraTransportType,
    service_name: str = "unknown",
) -> int:
    """Parse an integer environment variable with validation and error handling.

    Safely parses an environment variable as an integer, with optional range
    validation and comprehensive error handling following ONEX patterns.

    Args:
        env_var: Name of the environment variable to parse.
        default: Default value to use if env var is not set or out of range.
        min_value: Optional minimum valid value (inclusive). If the parsed value
            is below this, default is returned with a warning.
        max_value: Optional maximum valid value (inclusive). If the parsed value
            is above this, default is returned with a warning.
        transport_type: Required. Infrastructure transport type for error context.
        service_name: Service name for error context. Defaults to "unknown".

    Returns:
        Parsed integer value if valid and within range, otherwise the default.
        - If env var is not set: returns default
        - If parsed value is below min_value: logs warning, returns default
        - If parsed value is above max_value: logs warning, returns default

    Raises:
        ProtocolConfigurationError: If the environment variable contains a value
            that cannot be parsed as an integer. The error includes transport-aware
            context for debugging without exposing the actual invalid value.

    Example:
        >>> import os
        >>> from omnibase_infra.enums import EnumInfraTransportType
        >>> os.environ["MY_POOL_SIZE"] = "10"
        >>> parse_env_int(
        ...     "MY_POOL_SIZE", 5, min_value=1, max_value=100,
        ...     transport_type=EnumInfraTransportType.DATABASE,
        ... )
        10
        >>> parse_env_int(
        ...     "UNSET_VAR", 5,
        ...     transport_type=EnumInfraTransportType.DATABASE,
        ... )  # Returns default
        5

    Note:
        Following ONEX security guidelines, the actual invalid value is never
        exposed in error messages. Only the environment variable name and
        expected type are included.
    """
    # Lazy imports to avoid circular dependency
    from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

    raw_value = os.environ.get(env_var)
    if raw_value is None:
        return default

    try:
        parsed = int(raw_value)
    except ValueError:
        context = ModelInfraErrorContext(
            transport_type=transport_type,
            operation="parse_env_config",
            target_name=service_name,
            correlation_id=uuid4(),
        )
        raise ProtocolConfigurationError(
            f"Invalid value for {env_var} environment variable: expected integer",
            context=context,
            parameter=env_var,
            value="[REDACTED]",
        ) from None

    # Range validation with fallback to default
    if min_value is not None and parsed < min_value:
        logger.warning(
            "Environment variable %s value %d is below minimum %d, using default %d",
            env_var,
            parsed,
            min_value,
            default,
        )
        return default

    if max_value is not None and parsed > max_value:
        logger.warning(
            "Environment variable %s value %d is above maximum %d, using default %d",
            env_var,
            parsed,
            max_value,
            default,
        )
        return default

    return parsed


def parse_env_float(
    env_var: str,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    transport_type: EnumInfraTransportType,
    service_name: str = "unknown",
) -> float:
    """Parse a float environment variable with validation and error handling.

    Safely parses an environment variable as a float, with optional range
    validation and comprehensive error handling following ONEX patterns.

    Args:
        env_var: Name of the environment variable to parse.
        default: Default value to use if env var is not set or out of range.
        min_value: Optional minimum valid value (inclusive). If the parsed value
            is below this, default is returned with a warning.
        max_value: Optional maximum valid value (inclusive). If the parsed value
            is above this, default is returned with a warning.
        transport_type: Required. Infrastructure transport type for error context.
        service_name: Service name for error context. Defaults to "unknown".

    Returns:
        Parsed float value if valid and within range, otherwise the default.
        - If env var is not set: returns default
        - If parsed value is below min_value: logs warning, returns default
        - If parsed value is above max_value: logs warning, returns default

    Raises:
        ProtocolConfigurationError: If the environment variable contains a value
            that cannot be parsed as a float. The error includes transport-aware
            context for debugging without exposing the actual invalid value.

    Example:
        >>> import os
        >>> from omnibase_infra.enums import EnumInfraTransportType
        >>> os.environ["MY_TIMEOUT"] = "30.5"
        >>> parse_env_float(
        ...     "MY_TIMEOUT", 10.0, min_value=0.1, max_value=3600.0,
        ...     transport_type=EnumInfraTransportType.HTTP,
        ... )
        30.5
        >>> parse_env_float(
        ...     "UNSET_VAR", 10.0,
        ...     transport_type=EnumInfraTransportType.HTTP,
        ... )  # Returns default
        10.0

    Note:
        Following ONEX security guidelines, the actual invalid value is never
        exposed in error messages. Only the environment variable name and
        expected type are included.
    """
    # Lazy imports to avoid circular dependency
    from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

    raw_value = os.environ.get(env_var)
    if raw_value is None:
        return default

    try:
        parsed = float(raw_value)
    except ValueError:
        context = ModelInfraErrorContext(
            transport_type=transport_type,
            operation="parse_env_config",
            target_name=service_name,
            correlation_id=uuid4(),
        )
        raise ProtocolConfigurationError(
            f"Invalid value for {env_var} environment variable: expected numeric value",
            context=context,
            parameter=env_var,
            value="[REDACTED]",
        ) from None

    # Range validation with fallback to default
    if min_value is not None and parsed < min_value:
        logger.warning(
            "Environment variable %s value %f is below minimum %f, using default %f",
            env_var,
            parsed,
            min_value,
            default,
        )
        return default

    if max_value is not None and parsed > max_value:
        logger.warning(
            "Environment variable %s value %f is above maximum %f, using default %f",
            env_var,
            parsed,
            max_value,
            default,
        )
        return default

    return parsed


__all__: list[str] = [
    "parse_env_int",
    "parse_env_float",
]
