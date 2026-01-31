# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handlers for NodeArchitectureValidatorCompute.

This module exports the architecture validation handler that implements
the validation logic for the declarative compute node.

Handler Architecture:
    - Handlers are stateless classes (no mutable state between calls)
    - Handlers receive all context via the request model
    - Handlers return structured result models
    - Time-based decisions use explicit timestamp injection

Related Tickets:
    - OMN-1138: Architecture Validator for omnibase_infra
    - OMN-1726: Refactor to declarative pattern with contract-driven handler

.. versionadded:: 0.9.0
    Created as part of OMN-1726 declarative refactoring.
"""

from omnibase_infra.nodes.architecture_validator.handlers.handler_architecture_validation import (
    HandlerArchitectureValidation,
)

__all__: list[str] = [
    "HandlerArchitectureValidation",
]
