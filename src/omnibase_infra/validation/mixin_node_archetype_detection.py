# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Mixin for AST-based node archetype detection.

This module provides the MixinNodeArchetypeDetection mixin which extracts handler
detection logic from ExecutionShapeValidator. It detects ONEX node archetypes
(EFFECT, COMPUTE, REDUCER, ORCHESTRATOR) from Python AST structures.

Detection Methods:
    - Class name suffix: *EffectHandler, *ReducerHandler, etc.
    - Base class: class MyHandler(EffectHandler)
    - Decorator: @node_archetype(EnumNodeArchetype.EFFECT), @effect_handler
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from omnibase_infra.enums import EnumNodeArchetype

if TYPE_CHECKING:
    from omnibase_infra.validation.validator_execution_shape import (
        ModelDetectedNodeInfo,
    )

# Node archetype detection patterns
_NODE_ARCHETYPE_SUFFIX_MAP: dict[str, EnumNodeArchetype] = {
    "EffectHandler": EnumNodeArchetype.EFFECT,
    "ReducerHandler": EnumNodeArchetype.REDUCER,
    "OrchestratorHandler": EnumNodeArchetype.ORCHESTRATOR,
    "ComputeHandler": EnumNodeArchetype.COMPUTE,
    # Base class patterns
    "Effect": EnumNodeArchetype.EFFECT,
    "Reducer": EnumNodeArchetype.REDUCER,
    "Orchestrator": EnumNodeArchetype.ORCHESTRATOR,
    "Compute": EnumNodeArchetype.COMPUTE,
}


class MixinNodeArchetypeDetection:
    """Mixin providing AST-based node archetype detection capabilities.

    This mixin extracts handler detection logic from ExecutionShapeValidator
    to reduce method count while preserving functionality.

    Methods:
        _find_handlers: Find all handler classes and functions in an AST.
        _detect_node_archetype_from_class: Detect archetype from class definition.
        _detect_node_archetype_from_decorator: Detect archetype from decorators.
        _get_name_from_expr: Extract name string from AST expression.
    """

    def _find_handlers(
        self, tree: ast.AST, file_path: str
    ) -> list[ModelDetectedNodeInfo]:
        """Find all handler classes and functions in an AST.

        Detection methods:
            1. Class name suffix: *EffectHandler, *ReducerHandler, etc.
            2. Base class: class MyHandler(EffectHandler)
            3. Decorator: @node_archetype(EnumNodeArchetype.EFFECT)

        Args:
            tree: The parsed AST.
            file_path: Path to the source file.

        Returns:
            List of detected handlers with their archetype information.
        """
        # Import here to avoid circular import at module level
        from omnibase_infra.validation.validator_execution_shape import (
            ModelDetectedNodeInfo,
        )

        handlers: list[ModelDetectedNodeInfo] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                archetype = self._detect_node_archetype_from_class(node)
                if archetype is not None:
                    handlers.append(
                        ModelDetectedNodeInfo(
                            name=node.name,
                            node_archetype=archetype,
                            node=node,
                            line_number=node.lineno,
                            file_path=file_path,
                        )
                    )
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                archetype = self._detect_node_archetype_from_decorator(node)
                if archetype is not None:
                    handlers.append(
                        ModelDetectedNodeInfo(
                            name=node.name,
                            node_archetype=archetype,
                            node=node,
                            line_number=node.lineno,
                            file_path=file_path,
                        )
                    )

        return handlers

    def _detect_node_archetype_from_class(
        self,
        node: ast.ClassDef,
    ) -> EnumNodeArchetype | None:
        """Detect node archetype from class definition.

        Checks:
            1. Class name suffix (e.g., OrderEffectHandler)
            2. Base class names (e.g., class OrderHandler(EffectHandler))
            3. Decorator on class

        Args:
            node: The class definition AST node.

        Returns:
            The detected node archetype, or None if not a handler.
        """
        # Check class name suffix
        for suffix, archetype in _NODE_ARCHETYPE_SUFFIX_MAP.items():
            if node.name.endswith(suffix):
                return archetype

        # Check base classes
        for base in node.bases:
            base_name = self._get_name_from_expr(base)
            if base_name is not None:
                for pattern, archetype in _NODE_ARCHETYPE_SUFFIX_MAP.items():
                    if base_name.endswith(pattern):
                        return archetype

        # Check decorators
        return self._detect_node_archetype_from_decorator(node)

    def _detect_node_archetype_from_decorator(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> EnumNodeArchetype | None:
        """Detect node archetype from decorators.

        Looks for patterns like:
            @node_archetype(EnumNodeArchetype.EFFECT)
            @effect_handler
            @reducer_handler

        Args:
            node: The AST node with decorators.

        Returns:
            The detected node archetype, or None if not found.
        """
        for decorator in node.decorator_list:
            # Check @node_archetype(EnumNodeArchetype.X) pattern
            if isinstance(decorator, ast.Call):
                func = decorator.func
                func_name = self._get_name_from_expr(func)
                if func_name in ("node_archetype", "handler_type") and decorator.args:
                    arg = decorator.args[0]
                    arg_str = self._get_name_from_expr(arg)
                    if arg_str is not None:
                        # Handle EnumNodeArchetype.EFFECT or just EFFECT
                        # Use exact matching to avoid false positives from substring matches
                        # (e.g., "side_effect" should not match "effect")
                        arg_upper = arg_str.upper()
                        for archetype in EnumNodeArchetype:
                            # Match exact enum member name: "EFFECT", "COMPUTE", etc.
                            if arg_upper == archetype.name:
                                return archetype
                            # Match qualified name: "EnumNodeArchetype.EFFECT"
                            if arg_upper.endswith(f".{archetype.name}"):
                                return archetype
                            # Match quoted string value: "effect", "compute", etc.
                            if arg_upper == archetype.value.upper():
                                return archetype

            # Check @effect_handler, @reducer_handler patterns
            # Use suffix/prefix matching to reduce false positives from decorators
            # that happen to contain archetype keywords (e.g., @side_effect)
            decorator_name = self._get_name_from_expr(decorator)
            if decorator_name is not None:
                decorator_lower = decorator_name.lower()

                # Check for specific handler decorator patterns:
                # - Suffix: *_effect, *_reducer, *_orchestrator, *_compute
                # - Prefix: effect_*, reducer_*, orchestrator_*, compute_*
                # - Exact: effect, reducer, orchestrator, compute
                #
                # This avoids false positives like:
                # - @side_effect (not an effect handler decorator)
                # - @no_compute (not a compute handler decorator)

                # Effect handler patterns
                if (
                    decorator_lower.endswith("_effect")
                    or decorator_lower.startswith("effect_")
                    or decorator_lower == "effect"
                    or decorator_lower == "effect_handler"
                ):
                    return EnumNodeArchetype.EFFECT

                # Reducer handler patterns
                if (
                    decorator_lower.endswith("_reducer")
                    or decorator_lower.startswith("reducer_")
                    or decorator_lower == "reducer"
                    or decorator_lower == "reducer_handler"
                ):
                    return EnumNodeArchetype.REDUCER

                # Orchestrator handler patterns
                if (
                    decorator_lower.endswith("_orchestrator")
                    or decorator_lower.startswith("orchestrator_")
                    or decorator_lower == "orchestrator"
                    or decorator_lower == "orchestrator_handler"
                ):
                    return EnumNodeArchetype.ORCHESTRATOR

                # Compute handler patterns
                if (
                    decorator_lower.endswith("_compute")
                    or decorator_lower.startswith("compute_")
                    or decorator_lower == "compute"
                    or decorator_lower == "compute_handler"
                ):
                    return EnumNodeArchetype.COMPUTE

        return None

    def _get_name_from_expr(self, expr: ast.expr) -> str | None:
        """Extract a name string from an AST expression.

        Handles:
            - Name nodes: x -> "x"
            - Attribute nodes: a.b.c -> "a.b.c"

        Args:
            expr: The AST expression.

        Returns:
            The name as a string, or None if cannot be extracted.
        """
        if isinstance(expr, ast.Name):
            return expr.id
        if isinstance(expr, ast.Attribute):
            base = self._get_name_from_expr(expr.value)
            if base is not None:
                return f"{base}.{expr.attr}"
            return expr.attr
        return None
