# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Error Type Enumeration.

Defines the canonical error types for handler validation and lifecycle management.
These error types represent failures that can occur during handler registration,
validation, configuration, and execution within the ONEX infrastructure.

Error Categories:
    - Contract Errors: YAML parsing and schema validation failures
    - Validation Errors: Descriptor and constraint validation failures
    - Configuration Errors: Handler configuration and setup failures
    - Registration Errors: Handler registration and lifecycle failures
    - Type Errors: Handler type mismatches and compatibility issues
    - Security Errors: Security constraint violations
    - Architecture Errors: ONEX architecture pattern violations
    - Execution Errors: Runtime execution shape violations

These error types provide structured classification for handler-related failures,
enabling precise error reporting, debugging, and recovery strategies.

See Also:
    - EnumExecutionShapeViolation: Specific execution shape violations
    - EnumHandlerType: Defines the 4-node architecture handler types
    - ModelHandlerValidationError: Structured error reporting model
"""

from enum import Enum


class EnumHandlerErrorType(str, Enum):
    """Handler error types for validation and lifecycle management.

    These represent structured error classifications for failures that occur
    during handler validation, registration, configuration, and execution.
    Each error type indicates a specific category of failure.

    Contract Errors:
        CONTRACT_PARSE_ERROR: Contract YAML parsing failed.
            The handler's contract.yaml file could not be parsed due to
            invalid YAML syntax or structure.
        CONTRACT_VALIDATION_ERROR: Contract field validation failed.
            The parsed contract contains invalid field values or violates
            schema constraints defined in the ONEX contract specification.

    Discovery Errors:
        DISCOVERY_ERROR: Handler discovery failed due to infrastructure issues.
            The handler could not be discovered due to connection failures,
            timeouts, or other infrastructure problems (e.g., Consul unavailable,
            network errors). This is distinct from parse/validation errors which
            occur AFTER the contract is successfully retrieved.

    Validation Errors:
        DESCRIPTOR_VALIDATION_ERROR: Handler descriptor validation failed.
            The handler descriptor (metadata, annotations, signature) does not
            meet ONEX requirements for the declared handler type.
        SECURITY_VALIDATION_ERROR: Security constraint violated.
            The handler violates security policies such as introspection
            restrictions, input validation requirements, or secret handling.
        ARCHITECTURE_VALIDATION_ERROR: Architecture pattern violated.
            The handler violates ONEX architectural patterns such as layering,
            dependency injection, or node archetype responsibilities.

    Execution Errors:
        EXECUTION_SHAPE_VIOLATION: ONEX execution shape violated.
            The handler violates execution shape constraints defined by the
            4-node architecture (e.g., REDUCER returning events, direct publish).
            See EnumExecutionShapeViolation for specific violation types.

    Configuration Errors:
        CONFIGURATION_ERROR: Handler configuration invalid.
            The handler's configuration (environment variables, service endpoints,
            container dependencies) is invalid or incomplete.

    Registration Errors:
        REGISTRATION_ERROR: Handler registration failed.
            The handler could not be registered with the handler registry due to
            duplicate registration, missing dependencies, or lifecycle failures.

    Type Errors:
        TYPE_MISMATCH_ERROR: Handler type doesn't match expected.
            The handler's declared type (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR)
            does not match the contract specification or runtime behavior.
    """

    # Contract errors
    CONTRACT_PARSE_ERROR = "contract_parse_error"
    CONTRACT_VALIDATION_ERROR = "contract_validation_error"

    # Discovery errors
    DISCOVERY_ERROR = "discovery_error"

    # Validation errors
    DESCRIPTOR_VALIDATION_ERROR = "descriptor_validation_error"
    SECURITY_VALIDATION_ERROR = "security_validation_error"
    ARCHITECTURE_VALIDATION_ERROR = "architecture_validation_error"

    # Execution errors
    EXECUTION_SHAPE_VIOLATION = "execution_shape_violation"

    # Configuration errors
    CONFIGURATION_ERROR = "configuration_error"

    # Registration errors
    REGISTRATION_ERROR = "registration_error"

    # Type errors
    TYPE_MISMATCH_ERROR = "type_mismatch_error"


__all__ = ["EnumHandlerErrorType"]
