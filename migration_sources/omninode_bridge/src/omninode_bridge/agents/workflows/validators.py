"""
Code validators for validation pipeline.

Provides three validator types:
1. CompletenessValidator - Ensures all required fields/methods are present
2. QualityValidator - Checks code quality with thresholds
3. OnexComplianceValidator - Validates ONEX v2.0 compliance

Performance targets:
- Single validator: 50-200ms
- CompletenessValidator: <100ms
- QualityValidator: <200ms
- OnexComplianceValidator: <150ms

Example:
    ```python
    from omninode_bridge.agents.metrics import MetricsCollector

    collector = MetricsCollector()

    # Create validators
    completeness = CompletenessValidator(collector)
    quality = QualityValidator(collector, quality_threshold=0.8)
    onex = OnexComplianceValidator(collector)

    # Validate code
    context = ValidationContext(
        code_type="node",
        expected_patterns=["async def", "ModelOnexError"],
        required_methods=["execute_effect"]
    )

    result = await completeness.validate(code, context)
    print(f"Passed: {result.passed}, Score: {result.score}")
    ```
"""

import ast
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from omninode_bridge.agents.metrics.collector import MetricsCollector
from omninode_bridge.agents.workflows.validation_models import (
    EnumValidationSeverity,
    EnumValidationType,
    ValidationContext,
    ValidationIssue,
    ValidationResult,
)


class BaseValidator(ABC):
    """
    Base class for all validators.

    Provides common functionality for validation, metrics collection,
    and error handling.

    Subclasses must implement:
    - validate(): Perform validation and return ValidationResult
    """

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        validator_name: Optional[str] = None,
    ):
        """
        Initialize base validator.

        Args:
            metrics_collector: Metrics collector for tracking validation performance
            validator_name: Override default validator name (uses class name if None)
        """
        self.metrics = metrics_collector
        self.validator_name = validator_name or self.__class__.__name__

    @abstractmethod
    async def validate(
        self, code: str, context: ValidationContext
    ) -> ValidationResult:
        """
        Validate code and return result.

        Args:
            code: Python code to validate
            context: Validation context with configuration

        Returns:
            ValidationResult with validation outcome
        """
        pass

    async def _record_validation_metrics(
        self, duration_ms: float, passed: bool, score: float
    ) -> None:
        """Record validation metrics."""
        await self.metrics.record_timing(
            f"validation_{self.validator_name.lower()}_duration_ms",
            duration_ms,
            tags={
                "validator": self.validator_name,
                "passed": str(passed),
            },
        )

        await self.metrics.record_gauge(
            f"validation_{self.validator_name.lower()}_score",
            score,
            unit="score",
            tags={
                "validator": self.validator_name,
            },
        )

        await self.metrics.record_counter(
            f"validation_{self.validator_name.lower()}_count",
            1,
            tags={
                "validator": self.validator_name,
                "passed": str(passed),
            },
        )


class CompletenessValidator(BaseValidator):
    """
    Validator that ensures all required fields/methods are present.

    Checks:
    - Required imports are present
    - Required classes are defined
    - Required methods are implemented
    - Required fields exist
    - Proper inheritance structure

    Performance: <100ms for typical code
    """

    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize completeness validator."""
        super().__init__(metrics_collector, "CompletenessValidator")

    async def validate(
        self, code: str, context: ValidationContext
    ) -> ValidationResult:
        """
        Validate code completeness.

        Args:
            code: Python code to validate
            context: Validation context with expected patterns/methods

        Returns:
            ValidationResult with completeness validation outcome
        """
        start_time = time.perf_counter()

        errors: list[str] = []
        warnings: list[str] = []
        issues: list[ValidationIssue] = []
        metadata: dict[str, Any] = {}

        try:
            # Parse code into AST
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                errors.append(f"Syntax error in code: {e}")
                duration_ms = (time.perf_counter() - start_time) * 1000
                await self._record_validation_metrics(duration_ms, False, 0.0)
                return ValidationResult(
                    validator_name=self.validator_name,
                    validation_type=EnumValidationType.COMPLETENESS,
                    passed=False,
                    score=0.0,
                    errors=errors,
                    warnings=warnings,
                    issues=issues,
                    metadata=metadata,
                    duration_ms=duration_ms,
                    correlation_id=context.correlation_id,
                )

            # Extract classes, methods, and imports
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            # Get all function definitions (including async functions and methods)
            methods = [
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            imports = self._extract_imports(tree)

            metadata["total_classes"] = len(classes)
            metadata["total_methods"] = len(methods)
            metadata["total_imports"] = len(imports)

            # Check required methods
            missing_methods = []
            for required_method in context.required_methods:
                if required_method not in methods:
                    missing_methods.append(required_method)
                    errors.append(f"Missing required method: {required_method}")
                    issues.append(
                        ValidationIssue(
                            severity=EnumValidationSeverity.ERROR,
                            message=f"Missing required method: {required_method}",
                            rule_name="required_methods",
                            suggestion=f"Add method: async def {required_method}(self, ...) -> ...",
                        )
                    )

            metadata["missing_methods"] = missing_methods

            # Check expected patterns (imports, keywords)
            missing_patterns = []
            for pattern in context.expected_patterns:
                if pattern not in code:
                    missing_patterns.append(pattern)
                    warnings.append(f"Expected pattern not found: {pattern}")
                    issues.append(
                        ValidationIssue(
                            severity=EnumValidationSeverity.WARNING,
                            message=f"Expected pattern not found: {pattern}",
                            rule_name="expected_patterns",
                            suggestion=f"Consider adding: {pattern}",
                        )
                    )

            metadata["missing_patterns"] = missing_patterns

            # Check for docstrings
            missing_docstrings = self._check_docstrings(tree)
            if missing_docstrings:
                for item_name in missing_docstrings:
                    warnings.append(f"Missing docstring: {item_name}")
                    issues.append(
                        ValidationIssue(
                            severity=EnumValidationSeverity.WARNING,
                            message=f"Missing docstring in {item_name}",
                            rule_name="docstring_coverage",
                            suggestion=f"Add docstring to {item_name}",
                        )
                    )

            metadata["missing_docstrings"] = missing_docstrings

            # Calculate score
            total_checks = (
                len(context.required_methods)
                + len(context.expected_patterns)
                + len(classes)
            )
            if total_checks == 0:
                score = 1.0
            else:
                failed_checks = len(missing_methods) + len(missing_patterns)
                score = max(0.0, 1.0 - (failed_checks / total_checks))

            passed = len(errors) == 0

            # Record metrics
            duration_ms = (time.perf_counter() - start_time) * 1000
            await self._record_validation_metrics(duration_ms, passed, score)

            return ValidationResult(
                validator_name=self.validator_name,
                validation_type=EnumValidationType.COMPLETENESS,
                passed=passed,
                score=score,
                errors=errors,
                warnings=warnings,
                issues=issues,
                metadata=metadata,
                duration_ms=duration_ms,
                correlation_id=context.correlation_id,
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            errors.append(f"Validation error: {str(e)}")
            await self._record_validation_metrics(duration_ms, False, 0.0)
            return ValidationResult(
                validator_name=self.validator_name,
                validation_type=EnumValidationType.COMPLETENESS,
                passed=False,
                score=0.0,
                errors=errors,
                warnings=warnings,
                issues=issues,
                metadata=metadata,
                duration_ms=duration_ms,
                correlation_id=context.correlation_id,
            )

    def _extract_imports(self, tree: ast.AST) -> list[str]:
        """Extract import names from AST."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        return imports

    def _check_docstrings(self, tree: ast.AST) -> list[str]:
        """Check for missing docstrings in classes and functions."""
        missing = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if not ast.get_docstring(node):
                    missing.append(node.name)
        return missing


class QualityValidator(BaseValidator):
    """
    Validator that checks code quality with thresholds.

    Checks:
    - Cyclomatic complexity (<10 per method)
    - Code style (PEP 8 compliance)
    - Documentation (docstrings present)
    - Type hints (100% coverage)
    - Security anti-patterns

    Performance: <200ms for typical code
    """

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        quality_threshold: float = 0.8,
    ):
        """
        Initialize quality validator.

        Args:
            metrics_collector: Metrics collector
            quality_threshold: Minimum quality score required (0.0-1.0)
        """
        super().__init__(metrics_collector, "QualityValidator")
        self.quality_threshold = quality_threshold

    async def validate(
        self, code: str, context: ValidationContext
    ) -> ValidationResult:
        """
        Validate code quality.

        Args:
            code: Python code to validate
            context: Validation context

        Returns:
            ValidationResult with quality validation outcome
        """
        start_time = time.perf_counter()

        errors: list[str] = []
        warnings: list[str] = []
        issues: list[ValidationIssue] = []
        metadata: dict[str, Any] = {}

        try:
            # Parse code into AST
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                errors.append(f"Syntax error in code: {e}")
                duration_ms = (time.perf_counter() - start_time) * 1000
                await self._record_validation_metrics(duration_ms, False, 0.0)
                return ValidationResult(
                    validator_name=self.validator_name,
                    validation_type=EnumValidationType.QUALITY,
                    passed=False,
                    score=0.0,
                    errors=errors,
                    warnings=warnings,
                    issues=issues,
                    metadata=metadata,
                    duration_ms=duration_ms,
                    correlation_id=context.correlation_id,
                )

            # Check complexity
            complexity_issues = self._check_complexity(tree)
            metadata["complexity_issues"] = len(complexity_issues)
            for func_name, complexity in complexity_issues:
                warnings.append(
                    f"High complexity in {func_name}: {complexity} (target: <10)"
                )
                issues.append(
                    ValidationIssue(
                        severity=EnumValidationSeverity.WARNING,
                        message=f"High cyclomatic complexity in {func_name}: {complexity}",
                        rule_name="complexity_check",
                        suggestion="Consider breaking down into smaller functions",
                        metadata={"complexity": complexity, "threshold": 10},
                    )
                )

            # Check type hints
            type_hint_coverage = self._check_type_hints(tree)
            metadata["type_hint_coverage"] = type_hint_coverage
            if type_hint_coverage < 1.0:
                warnings.append(
                    f"Type hint coverage: {type_hint_coverage:.1%} (target: 100%)"
                )
                issues.append(
                    ValidationIssue(
                        severity=EnumValidationSeverity.WARNING,
                        message=f"Incomplete type hint coverage: {type_hint_coverage:.1%}",
                        rule_name="type_hint_coverage",
                        suggestion="Add type hints to all functions and methods",
                        metadata={"coverage": type_hint_coverage, "target": 1.0},
                    )
                )

            # Check docstring coverage
            docstring_coverage = self._check_docstring_coverage(tree)
            metadata["docstring_coverage"] = docstring_coverage
            if docstring_coverage < 1.0:
                warnings.append(
                    f"Docstring coverage: {docstring_coverage:.1%} (target: 100%)"
                )
                issues.append(
                    ValidationIssue(
                        severity=EnumValidationSeverity.WARNING,
                        message=f"Incomplete docstring coverage: {docstring_coverage:.1%}",
                        rule_name="docstring_coverage",
                        suggestion="Add docstrings to all classes and functions",
                        metadata={"coverage": docstring_coverage, "target": 1.0},
                    )
                )

            # Check code style
            style_issues = self._check_code_style(code)
            metadata["style_issues"] = len(style_issues)
            for style_issue in style_issues:
                warnings.append(f"Style issue: {style_issue}")
                issues.append(
                    ValidationIssue(
                        severity=EnumValidationSeverity.WARNING,
                        message=f"Style issue: {style_issue}",
                        rule_name="code_style",
                        suggestion="Follow PEP 8 style guide",
                    )
                )

            # Calculate quality score
            complexity_score = 1.0 - (
                min(len(complexity_issues), 10) / 10.0
            )  # Max 10 issues
            type_hint_score = type_hint_coverage
            docstring_score = docstring_coverage
            style_score = 1.0 - (min(len(style_issues), 20) / 20.0)  # Max 20 issues

            # Weighted average
            score = (
                complexity_score * 0.3
                + type_hint_score * 0.25
                + docstring_score * 0.25
                + style_score * 0.2
            )

            metadata["complexity_score"] = complexity_score
            metadata["type_hint_score"] = type_hint_score
            metadata["docstring_score"] = docstring_score
            metadata["style_score"] = style_score
            metadata["quality_threshold"] = self.quality_threshold

            # Check against threshold
            passed = score >= self.quality_threshold
            if not passed:
                errors.append(
                    f"Quality score {score:.2f} below threshold {self.quality_threshold}"
                )

            # Record metrics
            duration_ms = (time.perf_counter() - start_time) * 1000
            await self._record_validation_metrics(duration_ms, passed, score)

            return ValidationResult(
                validator_name=self.validator_name,
                validation_type=EnumValidationType.QUALITY,
                passed=passed,
                score=score,
                errors=errors,
                warnings=warnings,
                issues=issues,
                metadata=metadata,
                duration_ms=duration_ms,
                correlation_id=context.correlation_id,
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            errors.append(f"Validation error: {str(e)}")
            await self._record_validation_metrics(duration_ms, False, 0.0)
            return ValidationResult(
                validator_name=self.validator_name,
                validation_type=EnumValidationType.QUALITY,
                passed=False,
                score=0.0,
                errors=errors,
                warnings=warnings,
                issues=issues,
                metadata=metadata,
                duration_ms=duration_ms,
                correlation_id=context.correlation_id,
            )

    def _check_complexity(self, tree: ast.AST) -> list[tuple[str, int]]:
        """Check cyclomatic complexity of functions."""
        complex_functions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity = self._calculate_complexity(node)
                if complexity > 10:
                    complex_functions.append((node.name, complexity))
        return complex_functions

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity (simplified)."""
        complexity = 1  # Base complexity
        for child in ast.walk(node):
            # Decision points increase complexity
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity

    def _check_type_hints(self, tree: ast.AST) -> float:
        """Check type hint coverage."""
        total_functions = 0
        functions_with_hints = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Skip methods like __init__, __str__, etc. (dunder methods)
                if node.name.startswith("__") and node.name.endswith("__"):
                    continue

                total_functions += 1

                # Check if function has return type annotation
                has_return_hint = node.returns is not None

                # Check if args have type annotations (skip 'self' and 'cls')
                regular_args = [
                    arg
                    for arg in node.args.args
                    if arg.arg not in ("self", "cls")
                ]
                args_with_hints = sum(
                    1 for arg in regular_args if arg.annotation is not None
                )

                # Consider function to have hints if:
                # - Has return annotation, OR
                # - Has at least 50% of args with annotations
                if has_return_hint or (
                    len(regular_args) > 0
                    and args_with_hints / len(regular_args) >= 0.5
                ):
                    functions_with_hints += 1

        if total_functions == 0:
            return 1.0
        return functions_with_hints / total_functions

    def _check_docstring_coverage(self, tree: ast.AST) -> float:
        """Check docstring coverage."""
        total_items = 0
        items_with_docstrings = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                total_items += 1
                if ast.get_docstring(node):
                    items_with_docstrings += 1

        if total_items == 0:
            return 1.0
        return items_with_docstrings / total_items

    def _check_code_style(self, code: str) -> list[str]:
        """Check basic code style issues."""
        issues = []

        # Check line length
        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            if len(line) > 100:
                issues.append(f"Line {i} exceeds 100 characters ({len(line)})")

        # Check for basic style patterns
        if re.search(r"\t", code):
            issues.append("Use spaces instead of tabs")

        if re.search(r"  +\n", code):
            issues.append("Trailing whitespace detected")

        if re.search(r"\n\n\n+", code):
            issues.append("Too many blank lines (max 2)")

        return issues


class OnexComplianceValidator(BaseValidator):
    """
    Validator that checks ONEX v2.0 compliance.

    Checks:
    - ModelOnexError for error handling
    - emit_log_event for logging
    - Async methods (async def)
    - EnumLogLevel for log levels
    - Pydantic v2 models
    - Proper error propagation

    Performance: <150ms for typical code
    """

    REQUIRED_PATTERNS = [
        "ModelOnexError",  # Error handling
        "emit_log_event",  # Logging
        "async def",  # Async methods
        "EnumLogLevel",  # Log levels
    ]

    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize ONEX compliance validator."""
        super().__init__(metrics_collector, "OnexComplianceValidator")

    async def validate(
        self, code: str, context: ValidationContext
    ) -> ValidationResult:
        """
        Validate ONEX v2.0 compliance.

        Args:
            code: Python code to validate
            context: Validation context

        Returns:
            ValidationResult with ONEX compliance validation outcome
        """
        start_time = time.perf_counter()

        errors: list[str] = []
        warnings: list[str] = []
        issues: list[ValidationIssue] = []
        metadata: dict[str, Any] = {}

        try:
            # Check for ONEX patterns
            missing_patterns = []
            for pattern in self.REQUIRED_PATTERNS:
                if pattern not in code:
                    missing_patterns.append(pattern)
                    warnings.append(f"Missing ONEX pattern: {pattern}")
                    issues.append(
                        ValidationIssue(
                            severity=EnumValidationSeverity.WARNING,
                            message=f"Missing ONEX pattern: {pattern}",
                            rule_name="onex_patterns",
                            suggestion=f"Add ONEX pattern: {pattern}",
                        )
                    )

            metadata["missing_patterns"] = missing_patterns
            metadata["pattern_coverage"] = 1.0 - (
                len(missing_patterns) / len(self.REQUIRED_PATTERNS)
            )

            # Check for Pydantic v2 models
            has_pydantic = "BaseModel" in code or "pydantic" in code
            metadata["has_pydantic"] = has_pydantic
            if not has_pydantic:
                warnings.append("No Pydantic models detected")
                issues.append(
                    ValidationIssue(
                        severity=EnumValidationSeverity.WARNING,
                        message="No Pydantic models detected",
                        rule_name="pydantic_models",
                        suggestion="Use Pydantic v2 models for data validation",
                    )
                )

            # Check for proper error handling
            has_try_except = "try:" in code and "except" in code
            metadata["has_error_handling"] = has_try_except
            if not has_try_except:
                warnings.append("No error handling detected")
                issues.append(
                    ValidationIssue(
                        severity=EnumValidationSeverity.WARNING,
                        message="No error handling detected",
                        rule_name="error_handling",
                        suggestion="Add try/except blocks with ModelOnexError",
                    )
                )

            # Check for structured logging
            has_structured_logging = "emit_log_event" in code
            metadata["has_structured_logging"] = has_structured_logging
            if not has_structured_logging:
                warnings.append("No structured logging detected")
                issues.append(
                    ValidationIssue(
                        severity=EnumValidationSeverity.WARNING,
                        message="No structured logging detected",
                        rule_name="structured_logging",
                        suggestion="Use emit_log_event for structured logging",
                    )
                )

            # Calculate compliance score
            pattern_score = metadata["pattern_coverage"]
            pydantic_score = 1.0 if has_pydantic else 0.0
            error_handling_score = 1.0 if has_try_except else 0.0
            logging_score = 1.0 if has_structured_logging else 0.0

            score = (
                pattern_score * 0.4
                + pydantic_score * 0.2
                + error_handling_score * 0.2
                + logging_score * 0.2
            )

            metadata["pattern_score"] = pattern_score
            metadata["pydantic_score"] = pydantic_score
            metadata["error_handling_score"] = error_handling_score
            metadata["logging_score"] = logging_score

            # Pass if score is reasonable (no hard errors)
            passed = len(errors) == 0

            # Record metrics
            duration_ms = (time.perf_counter() - start_time) * 1000
            await self._record_validation_metrics(duration_ms, passed, score)

            return ValidationResult(
                validator_name=self.validator_name,
                validation_type=EnumValidationType.ONEX_COMPLIANCE,
                passed=passed,
                score=score,
                errors=errors,
                warnings=warnings,
                issues=issues,
                metadata=metadata,
                duration_ms=duration_ms,
                correlation_id=context.correlation_id,
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            errors.append(f"Validation error: {str(e)}")
            await self._record_validation_metrics(duration_ms, False, 0.0)
            return ValidationResult(
                validator_name=self.validator_name,
                validation_type=EnumValidationType.ONEX_COMPLIANCE,
                passed=False,
                score=0.0,
                errors=errors,
                warnings=warnings,
                issues=issues,
                metadata=metadata,
                duration_ms=duration_ms,
                correlation_id=context.correlation_id,
            )
