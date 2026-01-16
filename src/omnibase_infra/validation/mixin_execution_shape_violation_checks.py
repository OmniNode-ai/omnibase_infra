# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Mixin for execution shape violation checking.

This module provides the MixinExecutionShapeViolationChecks mixin which extracts
violation detection logic from ExecutionShapeValidator. It validates ONEX handlers
against execution shape rules (return types, publish calls, system time access).

Violation Types:
    - REDUCER_RETURNS_EVENTS: Reducer handler returns Event types
    - ORCHESTRATOR_RETURNS_INTENTS: Orchestrator returns Intent types
    - ORCHESTRATOR_RETURNS_PROJECTIONS: Orchestrator returns Projection types
    - EFFECT_RETURNS_PROJECTIONS: Effect handler returns Projection types
    - HANDLER_DIRECT_PUBLISH: Any handler calls .publish() directly
    - REDUCER_ACCESSES_SYSTEM_TIME: Reducer calls time.time(), datetime.now(), etc.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from omnibase_infra.enums import (
    EnumExecutionShapeViolation,
    EnumMessageCategory,
    EnumNodeArchetype,
    EnumNodeOutputType,
    EnumValidationSeverity,
)
from omnibase_infra.models.validation.model_execution_shape_rule import (
    ModelExecutionShapeRule,
)
from omnibase_infra.models.validation.model_execution_shape_violation import (
    ModelExecutionShapeViolationResult,
)

if TYPE_CHECKING:
    from omnibase_infra.validation.validator_execution_shape import (
        ModelDetectedNodeInfo,
    )

# Forbidden direct publish method names
_FORBIDDEN_PUBLISH_METHODS: frozenset[str] = frozenset(
    {
        "publish",
        "send_event",
        "emit",
        "emit_event",
        "dispatch",
        "dispatch_event",
    }
)

# System time access patterns (non-deterministic for reducers)
_SYSTEM_TIME_PATTERNS: frozenset[str] = frozenset(
    {
        "time.time",
        "datetime.now",
        "datetime.utcnow",
        "datetime.datetime.now",
        "datetime.datetime.utcnow",
        # Django timezone support
        "timezone.now",
        "django.utils.timezone.now",
    }
)

# Module-level time function patterns
_TIME_FUNCTION_NAMES: frozenset[str] = frozenset(
    {
        "time",
        "now",
        "utcnow",
    }
)

# Mapping from EnumMessageCategory to EnumNodeOutputType for type normalization.
_MESSAGE_CATEGORY_TO_OUTPUT_TYPE: dict[EnumMessageCategory, EnumNodeOutputType] = {
    EnumMessageCategory.EVENT: EnumNodeOutputType.EVENT,
    EnumMessageCategory.COMMAND: EnumNodeOutputType.COMMAND,
    EnumMessageCategory.INTENT: EnumNodeOutputType.INTENT,
}


class MixinExecutionShapeViolationChecks:
    """Mixin providing execution shape violation checking capabilities.

    This mixin extracts violation checking logic from ExecutionShapeValidator
    to reduce method count while preserving functionality.

    Requires:
        - `_rules: dict[EnumNodeArchetype, ModelExecutionShapeRule]` attribute
        - `_get_name_from_expr(expr: ast.expr) -> str | None` method

    Methods:
        _validate_handler: Validate a single handler for violations.
        _check_return_type_violations: Check for forbidden return types.
        _is_return_type_allowed: Check if return type is allowed.
        _analyze_return_value: Analyze return statement for forbidden types.
        _detect_message_category: Detect message category from type name.
        _create_return_type_violation: Create return type violation result.
        _check_direct_publish_violations: Check for direct publish calls.
        _check_system_time_violations: Check for system time access.
    """

    # Type hint for _rules attribute (set by main class)
    _rules: dict[EnumNodeArchetype, ModelExecutionShapeRule]

    def _validate_handler(
        self,
        handler: ModelDetectedNodeInfo,
    ) -> list[ModelExecutionShapeViolationResult]:
        """Validate a single handler for execution shape violations.

        Args:
            handler: The handler information.

        Returns:
            List of violations found in this handler.
        """
        violations: list[ModelExecutionShapeViolationResult] = []
        rule = self._rules.get(handler.node_archetype)

        if rule is None:
            return violations

        # Get all methods if this is a class
        methods: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        if isinstance(handler.node, ast.ClassDef):
            for item in handler.node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    methods.append(item)
        else:
            methods = [handler.node]

        # Validate each method
        for method in methods:
            # Check return type violations
            return_violations = self._check_return_type_violations(
                method, handler, rule
            )
            violations.extend(return_violations)

            # Check direct publish violations
            publish_violations = self._check_direct_publish_violations(method, handler)
            violations.extend(publish_violations)

            # Check system time access violations (reducers only)
            if not rule.can_access_system_time:
                time_violations = self._check_system_time_violations(method, handler)
                violations.extend(time_violations)

        return violations

    def _check_return_type_violations(
        self,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
        handler: ModelDetectedNodeInfo,
        rule: ModelExecutionShapeRule,
    ) -> list[ModelExecutionShapeViolationResult]:
        """Check for forbidden return type violations.

        Analyzes:
            1. Return type annotations
            2. Actual return statements

        Args:
            method: The method AST node.
            handler: The handler information.
            rule: The execution shape rule to validate against.

        Returns:
            List of return type violations.
        """
        violations: list[ModelExecutionShapeViolationResult] = []

        # Check return type annotation
        if method.returns is not None:
            annotation_str = self._get_name_from_expr(method.returns)
            if annotation_str is not None:
                category = self._detect_message_category(annotation_str)
                if category is not None and not self._is_return_type_allowed(
                    category, handler.node_archetype, rule
                ):
                    violation = self._create_return_type_violation(
                        handler, method.lineno, category, annotation_str
                    )
                    if violation is not None:
                        violations.append(violation)

        # Check actual return statements
        for node in ast.walk(method):
            if isinstance(node, ast.Return) and node.value is not None:
                return_violations = self._analyze_return_value(node, handler, rule)
                violations.extend(return_violations)

        return violations

    def _is_return_type_allowed(
        self,
        category: EnumMessageCategory | EnumNodeOutputType,
        node_archetype: EnumNodeArchetype,
        rule: ModelExecutionShapeRule,
    ) -> bool:
        """Check if a return type is allowed for a handler.

        Handles both EnumMessageCategory (EVENT, COMMAND, INTENT) and
        EnumNodeOutputType (PROJECTION) appropriately by converting
        message categories to their corresponding node output types.

        Args:
            category: The detected message category or node output type.
            node_archetype: The node archetype to check against.
            rule: The execution shape rule (uses EnumNodeOutputType).

        Returns:
            True if the return type is allowed, False otherwise.
        """
        # Convert EnumMessageCategory to EnumNodeOutputType for validation
        # since the rules use EnumNodeOutputType consistently
        output_type: EnumNodeOutputType
        if isinstance(category, EnumNodeOutputType):
            output_type = category
        elif isinstance(category, EnumMessageCategory):
            # Use module-level mapping constant for type conversion
            mapped_type = _MESSAGE_CATEGORY_TO_OUTPUT_TYPE.get(category)
            if mapped_type is None:
                # Forward compatibility: Unknown/new message category - disallow
                # by default until explicitly added to the module-level mapping.
                return False
            output_type = mapped_type
        else:
            # Unknown category type - disallow by default
            return False

        return rule.is_return_type_allowed(output_type)

    def _analyze_return_value(
        self,
        return_node: ast.Return,
        handler: ModelDetectedNodeInfo,
        rule: ModelExecutionShapeRule,
    ) -> list[ModelExecutionShapeViolationResult]:
        """Analyze a return statement for forbidden types.

        Args:
            return_node: The return statement AST node.
            handler: The handler information.
            rule: The execution shape rule.

        Returns:
            List of violations from this return statement.
        """
        violations: list[ModelExecutionShapeViolationResult] = []

        if return_node.value is None:
            return violations

        # Check if returning a call like Event(...) or ModelEvent(...)
        if isinstance(return_node.value, ast.Call):
            func_name = self._get_name_from_expr(return_node.value.func)
            if func_name is not None:
                category = self._detect_message_category(func_name)
                if category is not None and not self._is_return_type_allowed(
                    category, handler.node_archetype, rule
                ):
                    violation = self._create_return_type_violation(
                        handler, return_node.lineno, category, func_name
                    )
                    if violation is not None:
                        violations.append(violation)

        # Check if returning a variable with a type-hinted name
        if isinstance(return_node.value, ast.Name):
            var_name = return_node.value.id
            category = self._detect_message_category(var_name)
            if category is not None and not self._is_return_type_allowed(
                category, handler.node_archetype, rule
            ):
                violation = self._create_return_type_violation(
                    handler, return_node.lineno, category, var_name
                )
                if violation is not None:
                    violations.append(violation)

        return violations

    def _detect_message_category(
        self, name: str
    ) -> EnumMessageCategory | EnumNodeOutputType | None:
        """Detect message category or node output type from a type or variable name.

        Uses a multi-phase detection strategy:
        1. First, check for exact suffix matches (most reliable, fewest false positives)
        2. Then, check for prefix patterns (Model* naming convention)
        3. Check for exact uppercase enum-style names
        4. Finally, lenient substring matching (catches non-standard names)

        Note:
            PROJECTION returns EnumNodeOutputType.PROJECTION because projections
            are node output types, not message categories for routing.

        Args:
            name: The name to analyze (type name, class name, or variable name).

        Returns:
            The detected message category or EnumNodeOutputType.PROJECTION, or None.
        """
        # Phase 1: Check suffix-based patterns (most reliable)
        suffix_patterns: list[tuple[str, EnumMessageCategory | EnumNodeOutputType]] = [
            # Event suffixes - ordered by length (longest first)
            ("EventMessage", EnumMessageCategory.EVENT),
            ("Event", EnumMessageCategory.EVENT),
            # Command suffixes - ordered by length (longest first)
            ("CommandMessage", EnumMessageCategory.COMMAND),
            ("Command", EnumMessageCategory.COMMAND),
            # Intent suffixes - ordered by length (longest first)
            ("IntentMessage", EnumMessageCategory.INTENT),
            ("Intent", EnumMessageCategory.INTENT),
            # Projection suffixes - use EnumNodeOutputType (not a message category)
            ("ProjectionMessage", EnumNodeOutputType.PROJECTION),
            ("Projection", EnumNodeOutputType.PROJECTION),
        ]

        for suffix, category in suffix_patterns:
            if name.endswith(suffix):
                return category

        # Phase 2: Check prefix patterns for Model* naming convention
        prefix_patterns: list[tuple[str, EnumMessageCategory | EnumNodeOutputType]] = [
            ("ModelEvent", EnumMessageCategory.EVENT),
            ("ModelCommand", EnumMessageCategory.COMMAND),
            ("ModelIntent", EnumMessageCategory.INTENT),
            ("ModelProjection", EnumNodeOutputType.PROJECTION),
        ]

        for prefix, category in prefix_patterns:
            if name.startswith(prefix):
                return category

        # Phase 3: Check for exact uppercase enum-style names
        uppercase_patterns: dict[str, EnumMessageCategory | EnumNodeOutputType] = {
            "EVENT": EnumMessageCategory.EVENT,
            "COMMAND": EnumMessageCategory.COMMAND,
            "INTENT": EnumMessageCategory.INTENT,
            "PROJECTION": EnumNodeOutputType.PROJECTION,
        }

        if name in uppercase_patterns:
            return uppercase_patterns[name]

        # Phase 4: Lenient substring matching for non-standard naming
        name_lower = name.lower()

        if name_lower.endswith("event") or "_event" in name_lower:
            return EnumMessageCategory.EVENT
        if name_lower.endswith("command") or "_command" in name_lower:
            return EnumMessageCategory.COMMAND
        if name_lower.endswith("intent") or "_intent" in name_lower:
            return EnumMessageCategory.INTENT
        if name_lower.endswith("projection") or "_projection" in name_lower:
            return EnumNodeOutputType.PROJECTION

        return None

    def _create_return_type_violation(
        self,
        handler: ModelDetectedNodeInfo,
        line_number: int,
        category: EnumMessageCategory | EnumNodeOutputType,
        type_name: str,
    ) -> ModelExecutionShapeViolationResult | None:
        """Create a return type violation result.

        Args:
            handler: The handler information.
            line_number: Line number of the violation.
            category: The forbidden message category or node output type.
            type_name: The actual type name in code.

        Returns:
            The violation result, or None if no matching violation type.
        """
        violation_type: EnumExecutionShapeViolation | None = None
        message = ""

        # Get the execution shape rule for context
        rule = self._rules.get(handler.node_archetype)
        allowed_types_str = (
            ", ".join(t.name for t in rule.allowed_return_types) if rule else "unknown"
        )

        # Helper to check if category matches EVENT (from either enum)
        def is_event(cat: EnumMessageCategory | EnumNodeOutputType) -> bool:
            return cat == EnumMessageCategory.EVENT or cat == EnumNodeOutputType.EVENT

        # Helper to check if category matches INTENT (from either enum)
        def is_intent(cat: EnumMessageCategory | EnumNodeOutputType) -> bool:
            return cat == EnumMessageCategory.INTENT or cat == EnumNodeOutputType.INTENT

        # Helper to check if category matches PROJECTION
        def is_projection(cat: EnumMessageCategory | EnumNodeOutputType) -> bool:
            return cat == EnumNodeOutputType.PROJECTION

        # Get the category name for consistent error messages
        category_name = category.name if hasattr(category, "name") else str(category)

        if handler.node_archetype == EnumNodeArchetype.REDUCER:
            if is_event(category):
                violation_type = EnumExecutionShapeViolation.REDUCER_RETURNS_EVENTS
                message = (
                    f"Execution shape violation: Node archetype 'REDUCER' cannot produce "
                    f"output type 'EVENT'. Handler '{handler.name}' returns type '{type_name}'. "
                    f"Expected output types: [{allowed_types_str}]. Found: {category_name}. "
                    f"Reducers must be pure state projectors; event emission is an EFFECT operation."
                )
        elif handler.node_archetype == EnumNodeArchetype.ORCHESTRATOR:
            if is_intent(category):
                violation_type = (
                    EnumExecutionShapeViolation.ORCHESTRATOR_RETURNS_INTENTS
                )
                message = (
                    f"Execution shape violation: Node archetype 'ORCHESTRATOR' cannot produce "
                    f"output type 'INTENT'. Handler '{handler.name}' returns type '{type_name}'. "
                    f"Expected output types: [{allowed_types_str}]. Found: {category_name}. "
                    f"Intents represent external user/system requests and should not be "
                    f"generated by orchestration logic."
                )
            elif is_projection(category):
                violation_type = (
                    EnumExecutionShapeViolation.ORCHESTRATOR_RETURNS_PROJECTIONS
                )
                message = (
                    f"Execution shape violation: Node archetype 'ORCHESTRATOR' cannot produce "
                    f"output type 'PROJECTION'. Handler '{handler.name}' returns type '{type_name}'. "
                    f"Expected output types: [{allowed_types_str}]. Found: {category_name}. "
                    f"PROJECTION (EnumNodeOutputType.PROJECTION) represents aggregated state "
                    f"and is only valid for REDUCER nodes. Orchestrators coordinate workflows "
                    f"and should emit COMMAND or EVENT types."
                )
        elif handler.node_archetype == EnumNodeArchetype.EFFECT:
            if is_projection(category):
                violation_type = EnumExecutionShapeViolation.EFFECT_RETURNS_PROJECTIONS
                message = (
                    f"Execution shape violation: Node archetype 'EFFECT' cannot produce "
                    f"output type 'PROJECTION'. Handler '{handler.name}' returns type '{type_name}'. "
                    f"Expected output types: [{allowed_types_str}]. Found: {category_name}. "
                    f"PROJECTION (EnumNodeOutputType.PROJECTION) represents aggregated state "
                    f"and is only valid for REDUCER nodes. Effect handlers interact with "
                    f"external systems and should emit EVENT or COMMAND types."
                )

        if violation_type is None:
            return None

        return ModelExecutionShapeViolationResult(
            violation_type=violation_type,
            node_archetype=handler.node_archetype,
            file_path=handler.file_path,
            line_number=line_number,
            message=message,
            severity=EnumValidationSeverity.ERROR,
        )

    def _check_direct_publish_violations(
        self,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
        handler: ModelDetectedNodeInfo,
    ) -> list[ModelExecutionShapeViolationResult]:
        """Check for direct publish method calls.

        Forbidden patterns:
            - self.publish(...)
            - event_bus.publish(...)
            - self.send_event(...)
            - self.emit(...)

        Args:
            method: The method AST node.
            handler: The handler information.

        Returns:
            List of direct publish violations.
        """
        violations: list[ModelExecutionShapeViolationResult] = []

        for node in ast.walk(method):
            if isinstance(node, ast.Call):
                # Check for method calls like x.publish()
                if isinstance(node.func, ast.Attribute):
                    method_name = node.func.attr
                    if method_name in _FORBIDDEN_PUBLISH_METHODS:
                        archetype_name = handler.node_archetype.name
                        violations.append(
                            ModelExecutionShapeViolationResult(
                                violation_type=EnumExecutionShapeViolation.HANDLER_DIRECT_PUBLISH,
                                node_archetype=handler.node_archetype,
                                file_path=handler.file_path,
                                line_number=node.lineno,
                                message=(
                                    f"Execution shape violation: Node archetype '{archetype_name}' "
                                    f"handler '{handler.name}' calls forbidden method '.{method_name}()' "
                                    f"directly at line {node.lineno}. Direct publish methods are "
                                    f"forbidden for all node archetypes. Allowed: Return message objects "
                                    f"and let the dispatcher handle routing. Forbidden methods: "
                                    f"{', '.join(sorted(_FORBIDDEN_PUBLISH_METHODS))}."
                                ),
                                severity=EnumValidationSeverity.ERROR,
                            )
                        )

        return violations

    def _check_system_time_violations(
        self,
        method: ast.FunctionDef | ast.AsyncFunctionDef,
        handler: ModelDetectedNodeInfo,
    ) -> list[ModelExecutionShapeViolationResult]:
        """Check for system time access in reducers.

        Forbidden patterns:
            - time.time()
            - datetime.now()
            - datetime.utcnow()
            - datetime.datetime.now()
            - datetime.datetime.utcnow()

        Args:
            method: The method AST node.
            handler: The handler information.

        Returns:
            List of system time access violations.
        """
        violations: list[ModelExecutionShapeViolationResult] = []

        for node in ast.walk(method):
            if isinstance(node, ast.Call):
                func_name = self._get_name_from_expr(node.func)
                if func_name is not None:
                    # Check against known system time patterns
                    if func_name in _SYSTEM_TIME_PATTERNS:
                        violations.append(
                            ModelExecutionShapeViolationResult(
                                violation_type=EnumExecutionShapeViolation.REDUCER_ACCESSES_SYSTEM_TIME,
                                node_archetype=handler.node_archetype,
                                file_path=handler.file_path,
                                line_number=node.lineno,
                                message=(
                                    f"Execution shape violation: Node archetype 'REDUCER' "
                                    f"cannot access system time. Handler '{handler.name}' calls "
                                    f"'{func_name}()' at line {node.lineno}. REDUCER nodes must "
                                    f"be deterministic for event replay consistency. Use timestamps "
                                    f"from the event payload instead of system time. "
                                    f"Forbidden patterns: {', '.join(sorted(_SYSTEM_TIME_PATTERNS))}."
                                ),
                                severity=EnumValidationSeverity.ERROR,
                            )
                        )
                    # Check for method calls on datetime/time modules
                    elif isinstance(node.func, ast.Attribute):
                        attr_name = node.func.attr
                        if attr_name in _TIME_FUNCTION_NAMES:
                            base_name = self._get_name_from_expr(node.func.value)
                            if base_name is not None and (
                                "datetime" in base_name or "time" in base_name
                            ):
                                full_name = f"{base_name}.{attr_name}"
                                violations.append(
                                    ModelExecutionShapeViolationResult(
                                        violation_type=EnumExecutionShapeViolation.REDUCER_ACCESSES_SYSTEM_TIME,
                                        node_archetype=handler.node_archetype,
                                        file_path=handler.file_path,
                                        line_number=node.lineno,
                                        message=(
                                            f"Execution shape violation: Node archetype 'REDUCER' "
                                            f"cannot access system time. Handler '{handler.name}' calls "
                                            f"'{full_name}()' at line {node.lineno}. REDUCER nodes must "
                                            f"be deterministic for event replay consistency. Use timestamps "
                                            f"from the event payload instead of system time."
                                        ),
                                        severity=EnumValidationSeverity.ERROR,
                                    )
                                )

        return violations

    # Abstract method declaration for type checking - implemented by MixinNodeArchetypeDetection
    def _get_name_from_expr(self, expr: ast.expr) -> str | None:
        """Extract a name string from an AST expression.

        Note:
            This method is provided by MixinNodeArchetypeDetection.
            Declared here for type checking purposes.
        """
        raise NotImplementedError("Provided by MixinNodeArchetypeDetection")
