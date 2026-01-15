# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol for architecture validation rules.

This module defines the ProtocolArchitectureRule protocol that all architecture
validation rules must implement. Rules check nodes, handlers, and other
architectural components for compliance with ONEX patterns and constraints.

Thread Safety:
    Rule implementations may be invoked concurrently from the validation engine.
    Rules should be stateless or use appropriate synchronization.

Related:
    - OMN-1138: Architecture Validator for omnibase_infra
    - OMN-1099: Implement validators against this protocol

Example:
    >>> from omnibase_infra.nodes.architecture_validator.protocols import (
    ...     ProtocolArchitectureRule,
    ... )
    >>> from omnibase_infra.enums import EnumValidationSeverity
    >>> from omnibase_infra.nodes.architecture_validator.models import (
    ...     ModelRuleCheckResult,
    ... )
    >>>
    >>> class NoHandlerPublishingRule:
    ...     '''Rule that validates handlers do not have direct event bus access.'''
    ...
    ...     @property
    ...     def rule_id(self) -> str:
    ...         return "NO_HANDLER_PUBLISHING"
    ...
    ...     @property
    ...     def name(self) -> str:
    ...         return "No Handler Publishing"
    ...
    ...     @property
    ...     def description(self) -> str:
    ...         return "Handlers must not have direct event bus access."
    ...
    ...     @property
    ...     def severity(self) -> EnumValidationSeverity:
    ...         return EnumValidationSeverity.ERROR
    ...
    ...     def check(self, target: object) -> ModelRuleCheckResult:
    ...         # Implementation validates the target
    ...         return ModelRuleCheckResult(passed=True, rule_id=self.rule_id)
    >>>
    >>> # Verify protocol compliance via duck typing
    >>> rule = NoHandlerPublishingRule()
    >>> assert hasattr(rule, 'rule_id')
    >>> assert hasattr(rule, 'check') and callable(rule.check)

See Also:
    omnibase_infra.nodes.architecture_validator.models.ModelRuleCheckResult:
        Result model returned by check method
    omnibase_infra.enums.EnumValidationSeverity:
        Severity levels for rule violations

.. versionadded:: 0.6.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from omnibase_infra.enums import EnumValidationSeverity
    from omnibase_infra.nodes.architecture_validator.models import ModelRuleCheckResult

__all__ = ["ProtocolArchitectureRule"]


@runtime_checkable
class ProtocolArchitectureRule(Protocol):
    """Contract for architecture validation rules.

    Validators implementing this protocol check nodes, handlers, and other
    architectural components for violations of ONEX patterns and constraints.

    Common rule categories:
    - **Handler Isolation**: Handlers must not have direct event bus access
    - **Node Archetype Compliance**: Nodes must follow their archetype constraints
    - **Workflow Logic Location**: Workflow logic must be in orchestrators only
    - **State Management**: Reducers must be the only source of state changes

    Protocol Verification:
        Per ONEX conventions, protocol compliance is verified via duck typing
        rather than isinstance checks. Verify required properties and methods:

        .. code-block:: python

            # Verify required properties and methods exist
            if (hasattr(rule, 'rule_id') and
                hasattr(rule, 'name') and
                hasattr(rule, 'severity') and
                hasattr(rule, 'check') and callable(rule.check)):
                validator.register_rule(rule)
            else:
                raise TypeError("Object does not implement ProtocolArchitectureRule")

    Attributes:
        rule_id: Unique identifier for this rule (e.g., 'NO_HANDLER_PUBLISHING').
        name: Human-readable name for the rule.
        description: Detailed description of what this rule checks.
        severity: Severity level if this rule is violated.

    Note:
        Method bodies in this Protocol use ``...`` (Ellipsis) rather than
        ``raise NotImplementedError()``. This is the standard Python convention
        for ``typing.Protocol`` classes per PEP 544.

    .. versionadded:: 0.6.0
    """

    @property
    def rule_id(self) -> str:
        """Return the unique identifier for this rule.

        The rule ID is used for:
        - Registration and lookup in the validator
        - Filtering and selection of rules to run
        - Error reporting and result aggregation
        - Configuration of rule-specific settings

        Convention: Use SCREAMING_SNAKE_CASE for rule IDs.

        Returns:
            str: Unique rule identifier (e.g., "NO_HANDLER_PUBLISHING")

        Example:
            >>> rule.rule_id
            'NO_HANDLER_PUBLISHING'
        """
        ...

    @property
    def name(self) -> str:
        """Return the human-readable name for this rule.

        The name is used for:
        - Display in validation reports
        - User-facing documentation
        - Log messages

        Convention: Use Title Case for rule names.

        Returns:
            str: Human-readable rule name (e.g., "No Handler Publishing")

        Example:
            >>> rule.name
            'No Handler Publishing'
        """
        ...

    @property
    def description(self) -> str:
        """Return a detailed description of what this rule checks.

        The description should explain:
        - What architectural constraint is being validated
        - Why the constraint exists
        - What constitutes a violation

        Returns:
            str: Detailed description of the rule

        Example:
            >>> rule.description
            'Handlers must not have direct event bus access. Only orchestrators may publish events.'
        """
        ...

    @property
    def severity(self) -> EnumValidationSeverity:
        """Return the severity level for violations of this rule.

        The severity determines how violations are handled:
        - ERROR: Fail validation, block runtime startup
        - WARNING: Log warning, allow startup
        - INFO: Informational only

        Returns:
            EnumValidationSeverity: The severity level for this rule

        Example:
            >>> rule.severity
            <EnumValidationSeverity.ERROR: 'error'>
            >>> rule.severity.blocks_startup()
            True
        """
        ...

    def check(self, target: object) -> ModelRuleCheckResult:
        """Check the target against this rule.

        Validates that the target (node, handler, or other architectural
        component) complies with the constraint defined by this rule.

        Typing Note:
            The target parameter uses ``object`` instead of ``Any`` to
            satisfy ONEX "no Any types" guideline. Implementations should
            validate the target type internally.

        Args:
            target: The node, handler, or other object to validate.
                   Implementations should validate the target type and
                   return a passing result if the target type is not
                   applicable to this rule.

        Returns:
            ModelRuleCheckResult with:
            - passed=True if the target complies with the rule
            - passed=False if a violation is detected
            - message describing the result (especially for violations)
            - details with additional debugging information

        Example:
            >>> result = rule.check(my_handler)
            >>> if result.is_violation():
            ...     print(f"Violation: {result.message}")
            ...     print(f"Details: {result.details}")
        """
        ...
