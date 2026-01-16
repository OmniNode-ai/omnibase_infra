# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Classification mixin for Any type detection.

This mixin provides methods for classifying AST nodes as type aliases,
Field() calls, type expressions, etc.
"""

from __future__ import annotations

import ast


class MixinAnyTypeClassification:
    """Mixin providing type classification methods for Any type detection.

    This mixin extracts classification logic from AnyTypeDetector to reduce
    method count while maintaining functionality.

    Methods:
        _is_likely_type_alias_name: Check if name suggests a type alias.
        _is_type_alias_annotation: Check if node is a type alias definition.
        _is_type_expression: Check if expression is a type expression.
        _is_field_call: Check if expression is a Pydantic Field() call.
    """

    def _is_likely_type_alias_name(self, name: str) -> bool:
        """Check if a name is likely a type alias based on naming conventions.

        Args:
            name: The variable name to check.

        Returns:
            True if the name suggests a type alias.
        """
        if not name:
            return False
        # Explicit type alias suffixes
        if name.endswith(("Type", "Types")):
            return True
        # PascalCase without underscores (like type names)
        if name[0].isupper() and "_" not in name and not name.isupper():
            return True
        return False

    def _is_type_alias_annotation(self, node: ast.AnnAssign) -> bool:
        """Check if an annotated assignment is a type alias.

        Type aliases are detected by:
        - Annotation is TypeAlias
        - Target name follows type alias conventions (PascalCase or ends with Type)

        Args:
            node: The annotated assignment node.

        Returns:
            True if this appears to be a type alias definition.
        """
        # Check for TypeAlias annotation
        if isinstance(node.annotation, ast.Name) and node.annotation.id == "TypeAlias":
            return True
        if (
            isinstance(node.annotation, ast.Attribute)
            and node.annotation.attr == "TypeAlias"
        ):
            return True

        # Check target name conventions
        if isinstance(node.target, ast.Name):
            name = node.target.id
            # Type aliases typically use PascalCase or end with "Type"
            # Exclude ALL_CAPS names (not name.isupper()) to match _is_likely_type_alias_name
            if name.endswith("Type") or (
                name[0].isupper() and "_" not in name and not name.isupper()
            ):
                # Only consider it a type alias if value is a type expression
                if node.value is not None:
                    return self._is_type_expression(node.value)

        return False

    def _is_type_expression(self, node: ast.expr) -> bool:
        """Check if an expression is likely a type expression.

        Args:
            node: The AST expression.

        Returns:
            True if the expression appears to be a type definition.
        """
        if isinstance(node, ast.Name):
            return True
        if isinstance(node, ast.Subscript):
            return True
        if isinstance(node, ast.BinOp):
            # Union syntax: X | Y
            return True
        if isinstance(node, ast.Attribute):
            return True
        return False

    def _is_field_call(self, node: ast.expr) -> bool:
        """Check if an expression is a Pydantic Field() call.

        Args:
            node: The AST expression.

        Returns:
            True if this is a Field() call.
        """
        if not isinstance(node, ast.Call):
            return False

        func = node.func
        if isinstance(func, ast.Name) and func.id == "Field":
            return True
        if isinstance(func, ast.Attribute) and func.attr == "Field":
            return True

        return False
