# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""COMPUTE_GENERIC node for validating architecture compliance.

This module implements NodeArchitectureValidatorCompute, a pure transformation node
that validates nodes and handlers against architecture rules. The validator can be
invoked at startup (pre-runtime), during runtime (via orchestrator), or from CI/CD.

Design Pattern:
    NodeArchitectureValidatorCompute follows the COMPUTE_GENERIC node pattern from the
    ONEX 4-node architecture. As a COMPUTE_GENERIC node:
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
    >>> container = ModelONEXContainer()
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
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)

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

# Supported rule IDs from contract_architecture_validator.yaml
# These are the only rules that this validator node is designed to handle.
# Any rule not in this set indicates a misconfiguration or version mismatch.
SUPPORTED_RULE_IDS: frozenset[str] = frozenset(
    {
        "NO_HANDLER_PUBLISHING",
        "PURE_REDUCERS",
        "NO_FSM_IN_ORCHESTRATORS",
        "NO_WORKFLOW_IN_REDUCERS",
        "NO_DIRECT_HANDLER_DISPATCH",
        "NO_LOCAL_ONLY_PATHS",
    }
)

__all__ = ["NodeArchitectureValidatorCompute", "SUPPORTED_RULE_IDS"]


class NodeArchitectureValidatorCompute(
    NodeCompute[ModelArchitectureValidationRequest, ModelArchitectureValidationResult]
):
    """COMPUTE_GENERIC node for validating architecture compliance.

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
        ...     from omnibase_infra.enums import EnumInfraTransportType
        ...     from omnibase_infra.errors import (
        ...         ModelInfraErrorContext,
        ...         ProtocolConfigurationError,
        ...     )
        ...     context = ModelInfraErrorContext.with_correlation(
        ...         transport_type=EnumInfraTransportType.RUNTIME,
        ...         operation="architecture_validation",
        ...     )
        ...     raise ProtocolConfigurationError(
        ...         f"Validation failed: {result.violation_count} violations",
        ...         context=context,
        ...         code="ARCHITECTURE_VALIDATION_FAILED",
        ...         violations=[v.to_structured_dict() for v in result.violations],
        ...     )

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
            >>> container = ModelONEXContainer()
            >>> validator = NodeArchitectureValidatorCompute(
            ...     container=container,
            ...     rules=(NoHandlerPublishingRule(), NoWorkflowInReducerRule()),
            ... )

        .. versionadded:: 0.8.0
        """
        super().__init__(container)
        self._validate_rules_against_contract(rules)
        self._rules = rules

    def _validate_rules_against_contract(
        self,
        rules: tuple[ProtocolArchitectureRule, ...],
    ) -> None:
        """Validate that all provided rules are in the contract's supported_rules.

        This validation ensures that only rules defined in the contract's
        supported_rules list can be used with this validator. Using unsupported
        rules would indicate a misconfiguration or version mismatch between
        the validator and the rules being passed.

        Args:
            rules: Tuple of architecture rules to validate against the contract.

        Raises:
            ProtocolConfigurationError: If any rule has a rule_id not in
                SUPPORTED_RULE_IDS. The error message includes the unsupported
                rule_id and the list of supported rules for debugging.
                Includes ModelInfraErrorContext with RUNTIME transport type.

        Example:
            >>> # This is called automatically in __init__
            >>> validator._validate_rules_against_contract((
            ...     NoHandlerPublishingRule(),  # Valid: in supported_rules
            ...     PureReducersRule(),         # Valid: in supported_rules
            ... ))
            >>> # No error raised

            >>> # Invalid rule raises ProtocolConfigurationError
            >>> validator._validate_rules_against_contract((
            ...     UnknownRule(),  # rule_id="UNKNOWN_RULE"
            ... ))
            ProtocolConfigurationError: Rule 'UNKNOWN_RULE' is not in contract...

        .. versionadded:: 0.8.0
        """
        for rule in rules:
            if rule.rule_id not in SUPPORTED_RULE_IDS:
                context = ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="validate_rules_against_contract",
                )
                raise ProtocolConfigurationError(
                    f"Rule '{rule.rule_id}' is not in contract supported_rules. "
                    f"Supported rules: {sorted(SUPPORTED_RULE_IDS)}",
                    context=context,
                    rule_id=rule.rule_id,
                    supported_rules=sorted(SUPPORTED_RULE_IDS),
                )

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
            ...     rule_ids=("NO_HANDLER_PUBLISHING", "PURE_REDUCERS"),
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
