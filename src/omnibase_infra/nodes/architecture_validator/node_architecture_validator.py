# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""COMPUTE node for validating architecture compliance.

This module implements NodeArchitectureValidatorCompute, a pure transformation node
that validates nodes and handlers against architecture rules. The validator can be
invoked at startup (pre-runtime), during runtime (via orchestrator), or from CI/CD.

Design Pattern:
    NodeArchitectureValidatorCompute follows the COMPUTE node pattern from the
    ONEX 4-node architecture. As a COMPUTE node:
    - Pure transformation: input -> output, no side effects
    - Deterministic: same input always produces same output
    - Stateless validation: rules are injected, not stored
    - Thread-safe: can be invoked concurrently with different requests

Thread Safety:
    The validator is thread-safe when used with immutable rules. Each invocation
    of compute() is independent and does not modify shared state.

Related:
    - OMN-1138: Architecture Validator for omnibase_infra
    - OMN-1099: Validators implementing ProtocolArchitectureRule

Example:
    >>> from omnibase_core.models.container import ModelONEXContainer
    >>> from omnibase_infra.nodes.architecture_validator import (
    ...     NodeArchitectureValidatorCompute,
    ...     ModelArchitectureValidationRequest,
    ... )
    >>>
    >>> # Create validator with rules
    >>> container = ModelONEXContainer.minimal()
    >>> validator = NodeArchitectureValidatorCompute(
    ...     container=container,
    ...     rules=(no_handler_publishing_rule, no_workflow_in_reducer_rule),
    ... )
    >>>
    >>> # Validate nodes and handlers
    >>> request = ModelArchitectureValidationRequest(
    ...     nodes=(my_orchestrator, my_reducer),
    ...     handlers=(my_handler,),
    ... )
    >>> result = validator.compute(request)
    >>> if result.valid:
    ...     print("All architecture rules passed")
    ... else:
    ...     for v in result.violations:
    ...         print(f"Violation: {v.format_for_logging()}")

.. versionadded:: 0.8.0
    Created as part of OMN-1138 Architecture Validator implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes import NodeCompute

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

from omnibase_infra.nodes.architecture_validator.models import (
    ModelArchitectureValidationRequest,
    ModelArchitectureValidationResult,
    ModelArchitectureViolation,
    ModelRuleCheckResult,
)
from omnibase_infra.nodes.architecture_validator.protocols import (
    ProtocolArchitectureRule,
)

__all__ = ["NodeArchitectureValidatorCompute"]


class NodeArchitectureValidatorCompute(
    NodeCompute[ModelArchitectureValidationRequest, ModelArchitectureValidationResult]
):
    """COMPUTE node for validating architecture compliance.

    Validates nodes and handlers against architecture rules. This is a pure
    transformation node: input -> output, no side effects.

    Can be called:
    - At startup (direct call, pre-runtime validation)
    - During runtime (via orchestrator for dynamic validation)
    - From CI/CD (standalone validation in test/build pipelines)

    Attributes:
        _rules: Tuple of architecture rules to enforce during validation.

    Thread Safety:
        This node is thread-safe when:
        - Rules are stateless (recommended)
        - Request objects are not shared across threads
        - Each compute() call operates independently

    Example:
        >>> # Pre-runtime validation
        >>> validator = NodeArchitectureValidatorCompute(container, rules=rules)
        >>> result = validator.compute(request)
        >>> if not result:
        ...     raise RuntimeError(f"Validation failed: {result.violation_count} violations")

        >>> # CI/CD pipeline validation
        >>> result = validator.compute(ModelArchitectureValidationRequest(
        ...     nodes=discovered_nodes,
        ...     handlers=discovered_handlers,
        ...     fail_fast=True,  # Stop on first violation for fast feedback
        ... ))

    .. versionadded:: 0.8.0
    """

    def __init__(
        self,
        container: ModelONEXContainer,
        rules: tuple[ProtocolArchitectureRule, ...] = (),
    ) -> None:
        """Initialize validator with container and rules.

        Args:
            container: ONEX dependency injection container for node infrastructure.
            rules: Architecture rules to enforce. These should implement
                ProtocolArchitectureRule. Rules from OMN-1099 validators can be
                passed directly.

        Example:
            >>> from omnibase_core.models.container import ModelONEXContainer
            >>> from my_rules import NoHandlerPublishingRule, NoWorkflowInReducerRule
            >>>
            >>> container = ModelONEXContainer.minimal()
            >>> validator = NodeArchitectureValidatorCompute(
            ...     container=container,
            ...     rules=(NoHandlerPublishingRule(), NoWorkflowInReducerRule()),
            ... )

        .. versionadded:: 0.8.0
        """
        super().__init__(container)
        self._rules = rules

    def compute(
        self,
        request: ModelArchitectureValidationRequest,
    ) -> ModelArchitectureValidationResult:
        """Validate architecture compliance.

        Applies all registered rules to the nodes and handlers in the request.
        Returns a result containing any violations found and summary statistics.

        This is a pure transformation with no side effects:
        - Same input always produces same output
        - Does not modify request, rules, or external state
        - Safe to call concurrently from multiple threads

        Args:
            request: Validation request containing:
                - nodes: Nodes to validate
                - handlers: Handlers to validate
                - rule_ids: Optional filter for specific rules
                - fail_fast: Whether to stop on first violation
                - correlation_id: For distributed tracing

        Returns:
            ModelArchitectureValidationResult with:
            - violations: All violations found (empty if validation passed)
            - rules_checked: IDs of rules that were evaluated
            - nodes_checked: Count of nodes validated
            - handlers_checked: Count of handlers validated
            - correlation_id: Propagated from request

        Example:
            >>> # Check all rules
            >>> result = validator.compute(ModelArchitectureValidationRequest(
            ...     nodes=all_nodes,
            ...     handlers=all_handlers,
            ... ))
            >>> print(f"Checked {result.nodes_checked} nodes, {result.handlers_checked} handlers")
            >>> print(f"Found {result.violation_count} violations")

            >>> # Check specific rules only
            >>> result = validator.compute(ModelArchitectureValidationRequest(
            ...     nodes=all_nodes,
            ...     rule_ids=("NO_HANDLER_PUBLISHING", "NO_ANY_TYPES"),
            ...     fail_fast=True,
            ... ))

        .. versionadded:: 0.8.0
        """
        violations: list[ModelArchitectureViolation] = []
        rules_to_check = self._get_rules_to_check(request.rule_ids)

        # Validate nodes
        for node in request.nodes:
            for rule in rules_to_check:
                result = rule.check(node)
                if not result.passed:
                    violation = self._create_violation(rule, node, result)
                    violations.append(violation)
                    if request.fail_fast:
                        return self._build_result(violations, rules_to_check, request)

        # Validate handlers
        for handler in request.handlers:
            for rule in rules_to_check:
                result = rule.check(handler)
                if not result.passed:
                    violation = self._create_violation(rule, handler, result)
                    violations.append(violation)
                    if request.fail_fast:
                        return self._build_result(violations, rules_to_check, request)

        return self._build_result(violations, rules_to_check, request)

    def _get_rules_to_check(
        self,
        rule_ids: tuple[str, ...] | None,
    ) -> tuple[ProtocolArchitectureRule, ...]:
        """Get rules to check based on request filter.

        Args:
            rule_ids: Optional tuple of rule IDs to filter by.
                If None, all registered rules are returned.

        Returns:
            Tuple of rules to check. If rule_ids is provided, only
            rules with matching IDs are included.

        Example:
            >>> # No filter - returns all rules
            >>> rules = validator._get_rules_to_check(None)
            >>> len(rules) == len(validator._rules)
            True

            >>> # Filter by IDs
            >>> rules = validator._get_rules_to_check(("RULE_A", "RULE_B"))
            >>> all(r.rule_id in ("RULE_A", "RULE_B") for r in rules)
            True
        """
        if rule_ids is None:
            return self._rules
        return tuple(r for r in self._rules if r.rule_id in rule_ids)

    def _create_violation(
        self,
        rule: ProtocolArchitectureRule,
        target: object,
        result: ModelRuleCheckResult,
    ) -> ModelArchitectureViolation:
        """Create violation from rule check result.

        Args:
            rule: The rule that was violated.
            target: The node or handler that violated the rule.
            result: The check result with violation details.

        Returns:
            ModelArchitectureViolation with full context for debugging
            and remediation.

        Note:
            Uses getattr for target_name to handle both class types
            (with __name__) and instances (fallback to str()).
        """
        return ModelArchitectureViolation(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=rule.severity,
            target_type=type(target).__name__,
            target_name=getattr(target, "__name__", str(target)),
            message=result.message or rule.description,
            details=result.details,
        )

    def _build_result(
        self,
        violations: list[ModelArchitectureViolation],
        rules_checked: tuple[ProtocolArchitectureRule, ...],
        request: ModelArchitectureValidationRequest,
    ) -> ModelArchitectureValidationResult:
        """Build final validation result.

        Args:
            violations: All violations found during validation.
            rules_checked: Rules that were evaluated.
            request: Original validation request (for counts and correlation_id).

        Returns:
            ModelArchitectureValidationResult with complete validation summary.
        """
        return ModelArchitectureValidationResult(
            violations=tuple(violations),
            rules_checked=tuple(r.rule_id for r in rules_checked),
            nodes_checked=len(request.nodes),
            handlers_checked=len(request.handlers),
            correlation_id=request.correlation_id,
        )
