# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract validation for ONEX runtime configuration.

This module provides contract-level validation for runtime configuration.
It validates configuration dictionaries against the runtime config contract
schema before Pydantic model validation occurs.

Contract validation is a pre-validation step that:
    1. Validates field patterns (e.g., topic names match allowed patterns)
    2. Validates enum constraints (e.g., event_bus.type in allowed values)
    3. Validates numeric ranges (e.g., grace_period_seconds within bounds)
    4. Provides early, actionable error messages before Pydantic validation

This is ADDITIONAL validation on top of Pydantic model validation. The
Pydantic models (ModelRuntimeConfig, etc.) remain the authoritative schema
for type validation and defaults.

Usage:
    >>> from omnibase_infra.runtime.util_validation import validate_runtime_config
    >>> config = {"input_topic": "my-requests", "output_topic": "my-responses"}
    >>> errors = validate_runtime_config(config)
    >>> if errors:
    ...     print(f"Validation errors: {errors}")

    >>> # Or use load_and_validate_config for file-based loading
    >>> from omnibase_infra.runtime.util_validation import load_and_validate_config
    >>> config = load_and_validate_config(Path("config.yaml"))
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

import yaml

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError

# Topic name pattern: alphanumeric, underscores, hyphens, and periods
# This matches Kafka/Redpanda topic naming conventions and ONEX naming
# (e.g., "dev.onex.evt.node-introspection.v1")
TOPIC_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+$")

# Valid event bus types (matches ModelEventBusConfig.type Literal)
VALID_EVENT_BUS_TYPES = frozenset({"inmemory", "kafka"})

# Shutdown grace period bounds (matches ModelShutdownConfig constraints)
MIN_GRACE_PERIOD_SECONDS = 0
MAX_GRACE_PERIOD_SECONDS = 3600  # Max 1 hour to match ModelShutdownConfig


def validate_runtime_config(
    config: Mapping[str, object],
    contract_path: Path | None = None,
) -> list[str]:
    """Validate runtime configuration against contract schema.

    Performs contract-level validation on a configuration dictionary.
    This is a pre-validation step before Pydantic model validation.

    Validation Rules:
        - input_topic: Must be string matching ^[a-zA-Z0-9._-]+$
        - output_topic: Must be string matching ^[a-zA-Z0-9._-]+$
        - consumer_group/group_id: Must be string matching ^[a-zA-Z0-9._-]+$
        - event_bus.type: Must be "inmemory" or "kafka"
        - shutdown.grace_period_seconds: Must be integer 0-3600

    Args:
        config: Configuration dictionary to validate. Can be the raw dict
            loaded from YAML before Pydantic validation.
        contract_path: Optional path to contract file. Reserved for future
            use when we support loading validation rules from contract YAML.
            Currently validation rules are hardcoded to match the Pydantic models.

    Returns:
        List of validation error messages. Empty list if configuration is valid.
        Each error message describes the validation failure in a human-readable
        format suitable for logging or error reporting.

    Example:
        >>> config = {"input_topic": "valid-topic", "output_topic": "also-valid"}
        >>> errors = validate_runtime_config(config)
        >>> assert errors == []

        >>> config = {"input_topic": "invalid topic with spaces"}
        >>> errors = validate_runtime_config(config)
        >>> assert len(errors) == 1
        >>> assert "input_topic" in errors[0]
    """
    errors: list[str] = []

    # Validate topic names if present
    # These fields use the Kafka/Redpanda topic naming pattern
    topic_fields = ["input_topic", "output_topic"]
    for field in topic_fields:
        if field in config:
            value = config[field]
            if not isinstance(value, str):
                errors.append(f"{field} must be a string, got {type(value).__name__}")
            elif not TOPIC_NAME_PATTERN.match(value):
                errors.append(
                    f"{field} must match pattern ^[a-zA-Z0-9._-]+$, got: '{value}'"
                )

    # Validate consumer_group or group_id (alias)
    # Check both field names since YAML may use either
    group_field = None
    if "consumer_group" in config:
        group_field = "consumer_group"
    elif "group_id" in config:
        group_field = "group_id"

    if group_field is not None:
        value = config[group_field]
        if not isinstance(value, str):
            errors.append(f"{group_field} must be a string, got {type(value).__name__}")
        elif not TOPIC_NAME_PATTERN.match(value):
            errors.append(
                f"{group_field} must match pattern ^[a-zA-Z0-9._-]+$, got: '{value}'"
            )

    # Validate event_bus if present
    if "event_bus" in config:
        event_bus = config["event_bus"]
        if not isinstance(event_bus, dict):
            errors.append(
                f"event_bus must be an object, got {type(event_bus).__name__}"
            )
        else:
            # Validate event_bus.type enum
            if "type" in event_bus:
                bus_type = event_bus["type"]
                if not isinstance(bus_type, str):
                    errors.append(
                        f"event_bus.type must be a string, got {type(bus_type).__name__}"
                    )
                elif bus_type not in VALID_EVENT_BUS_TYPES:
                    errors.append(
                        f"event_bus.type must be one of {sorted(VALID_EVENT_BUS_TYPES)}, "
                        f"got: '{bus_type}'"
                    )

            # Validate event_bus.environment is string if present
            if "environment" in event_bus:
                env_value = event_bus["environment"]
                if not isinstance(env_value, str):
                    errors.append(
                        f"event_bus.environment must be a string, "
                        f"got {type(env_value).__name__}"
                    )

            # Validate event_bus.max_history is non-negative integer if present
            if "max_history" in event_bus:
                max_history = event_bus["max_history"]
                if not isinstance(max_history, int) or isinstance(max_history, bool):
                    errors.append(
                        f"event_bus.max_history must be an integer, "
                        f"got {type(max_history).__name__}"
                    )
                elif max_history < 0:
                    errors.append(
                        f"event_bus.max_history must be >= 0, got: {max_history}"
                    )

            # Validate event_bus.circuit_breaker_threshold is positive integer if present
            if "circuit_breaker_threshold" in event_bus:
                threshold = event_bus["circuit_breaker_threshold"]
                if not isinstance(threshold, int) or isinstance(threshold, bool):
                    errors.append(
                        f"event_bus.circuit_breaker_threshold must be an integer, "
                        f"got {type(threshold).__name__}"
                    )
                elif threshold < 1:
                    errors.append(
                        f"event_bus.circuit_breaker_threshold must be >= 1, "
                        f"got: {threshold}"
                    )

    # Validate shutdown if present
    if "shutdown" in config:
        shutdown = config["shutdown"]
        if not isinstance(shutdown, dict):
            errors.append(f"shutdown must be an object, got {type(shutdown).__name__}")
        # Validate shutdown.grace_period_seconds
        elif "grace_period_seconds" in shutdown:
            grace = shutdown["grace_period_seconds"]
            if not isinstance(grace, int) or isinstance(grace, bool):
                errors.append(
                    f"shutdown.grace_period_seconds must be an integer, "
                    f"got {type(grace).__name__}"
                )
            elif grace < MIN_GRACE_PERIOD_SECONDS:
                errors.append(
                    f"shutdown.grace_period_seconds must be >= "
                    f"{MIN_GRACE_PERIOD_SECONDS}, got: {grace}"
                )
            elif grace > MAX_GRACE_PERIOD_SECONDS:
                errors.append(
                    f"shutdown.grace_period_seconds must be <= "
                    f"{MAX_GRACE_PERIOD_SECONDS}, got: {grace}"
                )

    return errors


def load_and_validate_config(
    config_path: Path,
    contract_path: Path | None = None,
) -> dict[str, object]:
    """Load and validate runtime configuration from a YAML file.

    Loads a YAML configuration file and performs contract validation.
    This function is useful for standalone validation without going
    through the full kernel bootstrap process.

    Validation Process:
        1. Load YAML file from config_path
        2. Run contract validation via validate_runtime_config()
        3. Return validated configuration dictionary
        4. Raise ProtocolConfigurationError if validation fails

    Args:
        config_path: Path to the configuration YAML file to load.
        contract_path: Optional path to contract schema file. Reserved
            for future use when validation rules are loaded from contract.

    Returns:
        Validated configuration dictionary ready for Pydantic model creation.

    Raises:
        ProtocolConfigurationError: If the file cannot be loaded or
            validation fails. Error includes correlation_id for tracing
            and detailed validation_errors list in context.

    Example:
        >>> config_path = Path("contracts/runtime/runtime_config.yaml")
        >>> config = load_and_validate_config(config_path)
        >>> print(config["input_topic"])
        'requests'

        >>> # Invalid config raises error
        >>> bad_path = Path("invalid.yaml")
        >>> try:
        ...     load_and_validate_config(bad_path)
        ... except ProtocolConfigurationError as e:
        ...     print(f"Validation failed: {e}")
    """
    correlation_id = uuid4()
    context = ModelInfraErrorContext(
        transport_type=EnumInfraTransportType.RUNTIME,
        operation="validate_config",
        target_name=str(config_path),
        correlation_id=correlation_id,
    )

    # Load YAML file
    try:
        with config_path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError as e:
        raise ProtocolConfigurationError(
            f"Configuration file not found: {config_path}",
            context=context,
            config_path=str(config_path),
        ) from e
    except yaml.YAMLError as e:
        raise ProtocolConfigurationError(
            f"Failed to parse YAML from {config_path}: {e}",
            context=context,
            config_path=str(config_path),
            error_details=str(e),
        ) from e
    except UnicodeDecodeError as e:
        raise ProtocolConfigurationError(
            f"Configuration file contains binary or non-UTF-8 content: {config_path}",
            context=context,
            config_path=str(config_path),
            error_details=f"Encoding error at position {e.start}-{e.end}: {e.reason}",
        ) from e
    except OSError as e:
        raise ProtocolConfigurationError(
            f"Failed to read configuration file {config_path}: {e}",
            context=context,
            config_path=str(config_path),
            error_details=str(e),
        ) from e

    # Validate configuration
    errors = validate_runtime_config(config, contract_path)
    if errors:
        raise ProtocolConfigurationError(
            f"Configuration validation failed with {len(errors)} error(s): "
            f"{'; '.join(errors)}",
            context=context,
            config_path=str(config_path),
            validation_errors=errors,
            error_count=len(errors),
        )

    return config


__all__: list[str] = [
    "MAX_GRACE_PERIOD_SECONDS",
    "MIN_GRACE_PERIOD_SECONDS",
    "TOPIC_NAME_PATTERN",
    "VALID_EVENT_BUS_TYPES",
    "load_and_validate_config",
    "validate_runtime_config",
]
