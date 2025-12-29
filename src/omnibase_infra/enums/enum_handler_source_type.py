# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Source Type Enumeration for Validation Error Identification.

Defines the canonical source types for handler validation errors in the ONEX
structured validation and error reporting system. Each source type identifies
where in the validation pipeline an error was detected, enabling precise
error attribution and targeted remediation.

Handler validation occurs across multiple stages:
    - CONTRACT: Error from contract.yaml parsing/validation.
                Issues with YAML structure, required fields, or schema violations.
    - DESCRIPTOR: Error from handler descriptor validation.
                  Issues with handler metadata, signatures, or interface compliance.
    - STATIC_ANALYSIS: Error from AST/static code analysis.
                       Issues detected by analyzing Python source without execution.
    - RUNTIME: Error detected during runtime execution.
               Issues that only manifest when the handler is invoked.
    - REGISTRATION: Error during handler registration phase.
                    Issues with handler registration process or registry state.
    - CONFIGURATION: Error in handler configuration loading.
                     Issues with configuration values, environment, or dependencies.

See Also:
    - EnumHandlerType: Defines handler types (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR)
    - EnumExecutionShapeViolation: Defines validation violation types
    - ModelHandlerValidationError: Error model that uses this enum for source attribution
"""

from enum import Enum


class EnumHandlerSourceType(str, Enum):
    """Handler validation error source types for structured error reporting.

    These represent the validation stages where handler errors can be detected.
    Each source type enables precise error attribution and provides context for
    debugging and remediation.

    Attributes:
        CONTRACT: Error from contract.yaml parsing or validation.
            Indicates issues with the contract definition itself, such as:
            - Invalid YAML syntax
            - Missing required fields
            - Schema validation failures
            - Invalid handler routing configuration
        DESCRIPTOR: Error from handler descriptor validation.
            Indicates issues with handler metadata or interface compliance:
            - Invalid handler signatures
            - Missing required methods
            - Protocol implementation issues
            - Type annotation problems
        STATIC_ANALYSIS: Error from AST or static code analysis.
            Indicates issues detected by analyzing source code without execution:
            - Forbidden API usage
            - Invalid import statements
            - Code structure violations
            - Purity constraint violations
        RUNTIME: Error detected during runtime execution.
            Indicates issues that only manifest when handler is invoked:
            - Unhandled exceptions
            - Type mismatches at runtime
            - Resource exhaustion
            - Integration failures
        REGISTRATION: Error during handler registration phase.
            Indicates issues with the registration process:
            - Duplicate handler registration
            - Registry state conflicts
            - Dependency resolution failures
            - Handler initialization errors
        CONFIGURATION: Error in handler configuration loading.
            Indicates issues with configuration or environment:
            - Missing environment variables
            - Invalid configuration values
            - Dependency injection failures
            - Service connection issues
    """

    CONTRACT = "contract"
    DESCRIPTOR = "descriptor"
    STATIC_ANALYSIS = "static_analysis"
    RUNTIME = "runtime"
    REGISTRATION = "registration"
    CONFIGURATION = "configuration"


__all__ = ["EnumHandlerSourceType"]
