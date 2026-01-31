# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Enum for declarative node violation categories.

Defines the specific violation types detected by the declarative node validator.
Following the ONEX declarative pattern policy:
- Node classes MUST only extend base classes without custom logic
- Only __init__ with super().__init__(container) is allowed
- No custom methods, properties, or instance variables
"""

from enum import Enum


class EnumDeclarativeNodeViolation(str, Enum):
    """Violation types for declarative node pattern enforcement.

    These violation types correspond to the ONEX declarative node policy:
    Nodes must be contract-driven with no custom Python logic.

    Attributes:
        CUSTOM_METHOD: Node class contains custom method (not __init__).
        CUSTOM_PROPERTY: Node class contains property definition.
        INIT_CUSTOM_LOGIC: __init__ contains logic beyond super().__init__().
        INSTANCE_VARIABLE: __init__ creates custom instance variables.
        CLASS_VARIABLE: Node class defines class-level variables.
        SYNTAX_ERROR: File has Python syntax error, cannot validate.
        NO_NODE_CLASS: File named node.py but contains no node class.

    Policy Summary:
        - ALLOWED: ``class MyNode(NodeEffect): pass``
        - ALLOWED: ``def __init__(self, container): super().__init__(container)``
        - BLOCKED: ``def compute(self, data): ...``
        - BLOCKED: ``@property def my_prop(self): ...``
        - BLOCKED: ``self._custom_var = value`` in __init__
    """

    CUSTOM_METHOD = "custom_method"
    CUSTOM_PROPERTY = "custom_property"
    INIT_CUSTOM_LOGIC = "init_custom_logic"
    INSTANCE_VARIABLE = "instance_variable"
    CLASS_VARIABLE = "class_variable"
    SYNTAX_ERROR = "syntax_error"
    NO_NODE_CLASS = "no_node_class"

    @property
    def is_exemptable(self) -> bool:
        """Check if this violation type can be exempted.

        Some violations like SYNTAX_ERROR cannot be exempted because
        they indicate fundamental issues with the source file.

        Returns:
            True if the violation type can be exempted via decorator or comment.
        """
        return self not in {
            EnumDeclarativeNodeViolation.SYNTAX_ERROR,
            EnumDeclarativeNodeViolation.NO_NODE_CLASS,
        }

    @property
    def suggestion(self) -> str:
        """Get the suggested fix for this violation type.

        Returns:
            Human-readable suggestion for fixing the violation.
        """
        suggestions = {
            EnumDeclarativeNodeViolation.CUSTOM_METHOD: (
                "Move business logic to a Handler class. Node classes must be "
                "declarative - all behavior should be defined in contract.yaml "
                "and implemented by handlers."
            ),
            EnumDeclarativeNodeViolation.CUSTOM_PROPERTY: (
                "Remove property from node class. Node classes should not have "
                "custom properties - use container dependency injection instead."
            ),
            EnumDeclarativeNodeViolation.INIT_CUSTOM_LOGIC: (
                "Remove custom logic from __init__. The __init__ method should "
                "only call super().__init__(container). All initialization should "
                "be handled by the base class and contract.yaml."
            ),
            EnumDeclarativeNodeViolation.INSTANCE_VARIABLE: (
                "Remove instance variable assignment from __init__. Node classes "
                "should not store state - use container injection and handlers."
            ),
            EnumDeclarativeNodeViolation.CLASS_VARIABLE: (
                "Remove class variable from node class. Configuration should be "
                "in contract.yaml, not Python code."
            ),
            EnumDeclarativeNodeViolation.SYNTAX_ERROR: (
                "Fix the Python syntax error before validation can proceed."
            ),
            EnumDeclarativeNodeViolation.NO_NODE_CLASS: (
                "File is named node.py but does not contain a Node class. "
                "Either add a Node class or rename the file."
            ),
        }
        return suggestions.get(self, "Make the node class declarative.")


__all__ = ["EnumDeclarativeNodeViolation"]
