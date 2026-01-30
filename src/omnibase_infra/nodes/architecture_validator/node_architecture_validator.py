# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Declarative COMPUTE_GENERIC node for architecture validation.

This module implements NodeArchitectureValidatorCompute, a declarative compute node
that validates nodes and handlers against architecture rules. All validation logic
is delegated to HandlerArchitectureValidation via the contract-driven handler pattern.

Design Pattern:
    NodeArchitectureValidatorCompute follows the ONEX declarative node pattern:
    - Extends NodeCompute from omnibase_core
    - All behavior defined in contract.yaml
    - Handler contains actual validation logic
    - Node is a lightweight shell with no custom methods

Thread Safety:
    The node delegates all operations to the handler, which is thread-safe
    when used with immutable rules.

Related:
    - OMN-1138: Architecture Validator for omnibase_infra
    - OMN-1726: Refactor to declarative pattern with contract-driven handler

Example:
    >>> from omnibase_core.models.container import ModelONEXContainer
    >>> from omnibase_infra.nodes.architecture_validator import (
    ...     NodeArchitectureValidatorCompute,
    ... )
    >>> from omnibase_infra.nodes.architecture_validator.handlers import (
    ...     HandlerArchitectureValidation,
    ... )
    >>>
    >>> # Create declarative node
    >>> container = ModelONEXContainer()
    >>> node = NodeArchitectureValidatorCompute(container)
    >>>
    >>> # Use handler for validation
    >>> handler = HandlerArchitectureValidation(rules=my_rules)
    >>> result = handler.handle(request)

.. versionadded:: 0.8.0
    Created as part of OMN-1138 Architecture Validator implementation.

.. versionchanged:: 0.9.0
    Refactored to declarative pattern as part of OMN-1726. All validation
    logic moved to HandlerArchitectureValidation.
"""

from __future__ import annotations

from omnibase_core.nodes import NodeCompute
from omnibase_infra.nodes.architecture_validator.models import (
    ModelArchitectureValidationRequest,
    ModelArchitectureValidationResult,
)

__all__ = ["NodeArchitectureValidatorCompute"]


class NodeArchitectureValidatorCompute(
    NodeCompute[ModelArchitectureValidationRequest, ModelArchitectureValidationResult]
):
    """Declarative COMPUTE_GENERIC node for architecture validation.

    This node follows the ONEX declarative pattern - all behavior is defined
    in contract.yaml and delegated to HandlerArchitectureValidation. The node
    itself is a lightweight shell with no custom methods.

    Validation Logic:
        All validation logic is implemented in HandlerArchitectureValidation.
        The handler validates nodes and handlers against architecture rules
        and returns ModelArchitectureValidationResult.

    Usage:
        For validation operations, use HandlerArchitectureValidation directly:

        >>> from omnibase_infra.nodes.architecture_validator.handlers import (
        ...     HandlerArchitectureValidation,
        ... )
        >>> handler = HandlerArchitectureValidation(rules=my_rules)
        >>> result = handler.handle(request)

    Can be called:
    - At startup (direct call, pre-runtime validation)
    - During runtime (via orchestrator for dynamic validation)
    - From CI/CD (standalone validation in test/build pipelines)

    .. versionadded:: 0.8.0

    .. versionchanged:: 0.9.0
        Refactored to declarative pattern. All validation logic moved to
        HandlerArchitectureValidation.
    """

    # Declarative node - all behavior defined in contract.yaml
