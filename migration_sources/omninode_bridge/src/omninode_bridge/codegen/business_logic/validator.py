#!/usr/bin/env python3
"""
Code Validator for LLM-Generated Business Logic.

Validates generated code for:
1. Python syntax (AST parsing)
2. ONEX compliance (ModelOnexError, emit_log_event, type hints)
3. Type hint presence
4. Security issues (hardcoded secrets, SQL injection, dangerous patterns)
5. Code quality (complexity, length, best practices)

Used by BusinessLogicGenerator to ensure generated code meets quality standards.
"""

import ast
from typing import Optional

from pydantic import BaseModel, Field

from .validation_rules import (
    MAX_COMPLEXITY,
    MAX_FUNCTION_LENGTH,
    MIN_DOCSTRING_LENGTH,
    TYPE_HINT_EXCEPTIONS,
    check_dangerous_patterns,
    check_hardcoded_secrets,
    check_onex_compliance,
    estimate_complexity,
)


class ValidationResult(BaseModel):
    """
    Result of code validation.

    Provides comprehensive assessment of generated code quality.
    """

    # Overall result
    passed: bool = Field(..., description="Overall validation passed")
    quality_score: float = Field(
        ..., ge=0.0, le=1.0, description="Quality score (0.0-1.0)"
    )

    # Component validation results
    syntax_valid: bool = Field(..., description="Python syntax is valid")
    onex_compliant: bool = Field(..., description="Follows ONEX patterns")
    has_type_hints: bool = Field(..., description="Has proper type hints")
    security_clean: bool = Field(..., description="No security issues found")

    # Issues found
    issues: list[str] = Field(default_factory=list, description="All validation issues")
    syntax_errors: list[str] = Field(default_factory=list, description="Syntax errors")
    onex_issues: list[str] = Field(
        default_factory=list, description="ONEX compliance issues"
    )
    type_hint_issues: list[str] = Field(
        default_factory=list, description="Type hint issues"
    )
    security_issues: list[str] = Field(
        default_factory=list, description="Security issues"
    )
    quality_issues: list[str] = Field(
        default_factory=list, description="Code quality issues"
    )

    # Metadata
    complexity_score: int = Field(default=0, ge=0, description="Cyclomatic complexity")
    line_count: int = Field(default=0, ge=0, description="Number of lines")


class GenerationContext(BaseModel):
    """
    Context for code generation validation.

    Provides information about what is being generated.
    """

    node_type: str = Field(
        ..., description="Node type (effect/compute/reducer/orchestrator)"
    )
    method_name: str = Field(..., description="Method being generated")
    service_name: Optional[str] = Field(None, description="Service name")


class CodeValidator:
    """
    Validates LLM-generated code for quality and compliance.

    Performs multiple validation passes:
    1. Syntax validation (AST parsing)
    2. ONEX compliance checks
    3. Type hint validation
    4. Security checks
    5. Code quality checks
    """

    def __init__(self, strict: bool = True):
        """
        Initialize code validator.

        Args:
            strict: If True, validation fails on any issue. If False, only critical issues fail.
        """
        self.strict = strict

    async def validate(
        self, generated_code: str, context: GenerationContext
    ) -> ValidationResult:
        """
        Validate generated code.

        Args:
            generated_code: LLM-generated code to validate
            context: Generation context for validation

        Returns:
            ValidationResult with pass/fail status and issues

        Example:
            >>> validator = CodeValidator(strict=True)
            >>> result = await validator.validate(code, context)
            >>> if not result.passed:
            ...     print(f"Validation failed: {result.issues}")
        """
        all_issues = []

        # 1. Syntax validation
        syntax_valid, syntax_errors = self._validate_syntax(generated_code)
        all_issues.extend(syntax_errors)

        # 2. ONEX compliance
        onex_compliant, onex_issues = self._validate_onex_compliance(generated_code)
        all_issues.extend(onex_issues)

        # 3. Type hints
        type_hints_valid, type_issues = self._validate_type_hints(
            generated_code, context
        )
        all_issues.extend(type_issues)

        # 4. Security
        security_clean, security_issues = self._validate_security(generated_code)
        all_issues.extend(security_issues)

        # 5. Code quality
        quality_ok, quality_issues, complexity = self._validate_code_quality(
            generated_code
        )
        all_issues.extend(quality_issues)

        # Calculate quality score
        quality_score = self._calculate_quality_score(
            syntax_valid=syntax_valid,
            onex_compliant=onex_compliant,
            type_hints_valid=type_hints_valid,
            security_clean=security_clean,
            quality_ok=quality_ok,
            num_issues=len(all_issues),
        )

        # Determine pass/fail
        if self.strict:
            # Strict mode: fail on any issue
            passed = len(all_issues) == 0
        else:
            # Lenient mode: fail only on critical issues (syntax, security)
            passed = syntax_valid and security_clean

        # Count lines
        line_count = len(generated_code.split("\n"))

        return ValidationResult(
            passed=passed,
            quality_score=quality_score,
            syntax_valid=syntax_valid,
            onex_compliant=onex_compliant,
            has_type_hints=type_hints_valid,
            security_clean=security_clean,
            issues=all_issues,
            syntax_errors=syntax_errors,
            onex_issues=onex_issues,
            type_hint_issues=type_issues,
            security_issues=security_issues,
            quality_issues=quality_issues,
            complexity_score=complexity,
            line_count=line_count,
        )

    def _validate_syntax(self, code: str) -> tuple[bool, list[str]]:
        """
        Validate Python syntax using AST parsing.

        Args:
            code: Source code to validate

        Returns:
            Tuple of (is_valid, list of errors)
        """
        try:
            ast.parse(code)
            return True, []
        except SyntaxError as e:
            error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
            if e.text:
                error_msg += f"\n  {e.text.strip()}"
            return False, [error_msg]
        except Exception as e:
            return False, [f"Failed to parse code: {e!s}"]

    def _validate_onex_compliance(self, code: str) -> tuple[bool, list[str]]:
        """
        Check for ONEX patterns.

        Verifies:
        - ModelOnexError usage for exceptions
        - emit_log_event for logging
        - omnibase_core imports

        Args:
            code: Source code to validate

        Returns:
            Tuple of (is_compliant, list of issues)
        """
        return check_onex_compliance(code)

    def _validate_type_hints(
        self, code: str, context: GenerationContext
    ) -> tuple[bool, list[str]]:
        """
        Check for type hints on functions/methods.

        Args:
            code: Source code to validate
            context: Generation context

        Returns:
            Tuple of (has_type_hints, list of issues)
        """
        issues = []

        try:
            tree = ast.parse(code)

            # Find all function definitions
            functions = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            ]

            if not functions:
                # No functions found - might be just helper code
                return True, []

            for func in functions:
                func_name = func.name

                # Skip magic methods and private helpers
                if func_name in TYPE_HINT_EXCEPTIONS or func_name.startswith("_"):
                    continue

                # Check return type hint
                if func.returns is None:
                    issues.append(f"Function '{func_name}' missing return type hint")

                # Check argument type hints
                for arg in func.args.args:
                    if arg.arg == "self" or arg.arg == "cls":
                        continue
                    if arg.annotation is None:
                        issues.append(
                            f"Function '{func_name}' argument '{arg.arg}' missing type hint"
                        )

        except SyntaxError:
            # Syntax already checked in _validate_syntax
            pass
        except Exception as e:
            issues.append(f"Error checking type hints: {e!s}")

        return len(issues) == 0, issues

    def _validate_security(self, code: str) -> tuple[bool, list[str]]:
        """
        Check for security issues.

        Checks for:
        - Hardcoded secrets
        - SQL injection vulnerabilities
        - Dangerous patterns (eval, exec, pickle)

        Args:
            code: Source code to validate

        Returns:
            Tuple of (is_secure, list of issues)
        """
        issues = []

        # Check for hardcoded secrets
        secret_issues = check_hardcoded_secrets(code)
        issues.extend(secret_issues)

        # Check for dangerous patterns
        danger_issues = check_dangerous_patterns(code)
        issues.extend(danger_issues)

        return len(issues) == 0, issues

    def _validate_code_quality(self, code: str) -> tuple[bool, list[str], int]:
        """
        Check code quality metrics.

        Checks:
        - Function length
        - Cyclomatic complexity
        - Docstring presence

        Args:
            code: Source code to validate

        Returns:
            Tuple of (is_quality_ok, list of issues, complexity_score)
        """
        issues = []
        complexity = estimate_complexity(code)

        # Check complexity
        if complexity > MAX_COMPLEXITY:
            issues.append(
                f"High cyclomatic complexity ({complexity}), consider simplifying "
                f"(threshold: {MAX_COMPLEXITY})"
            )

        # Check function length
        lines = code.split("\n")
        if len(lines) > MAX_FUNCTION_LENGTH:
            issues.append(
                f"Function too long ({len(lines)} lines), consider breaking it up "
                f"(threshold: {MAX_FUNCTION_LENGTH})"
            )

        # Check for docstrings
        try:
            tree = ast.parse(code)
            functions = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            ]

            for func in functions:
                if func.name.startswith("_"):
                    continue  # Skip private functions

                docstring = ast.get_docstring(func)
                if not docstring:
                    issues.append(f"Function '{func.name}' missing docstring")
                elif len(docstring) < MIN_DOCSTRING_LENGTH:
                    issues.append(
                        f"Function '{func.name}' has short docstring "
                        f"({len(docstring)} chars, minimum: {MIN_DOCSTRING_LENGTH})"
                    )

        except SyntaxError:
            # Syntax already checked
            pass

        return len(issues) == 0, issues, complexity

    def _calculate_quality_score(
        self,
        syntax_valid: bool,
        onex_compliant: bool,
        type_hints_valid: bool,
        security_clean: bool,
        quality_ok: bool,
        num_issues: int,
    ) -> float:
        """
        Calculate overall quality score.

        Weighted scoring:
        - Syntax: 30% (critical)
        - Security: 25% (critical)
        - ONEX compliance: 20%
        - Type hints: 15%
        - Code quality: 10%

        Additional penalty for number of issues.

        Args:
            syntax_valid: Syntax is valid
            onex_compliant: ONEX patterns followed
            type_hints_valid: Type hints present
            security_clean: No security issues
            quality_ok: Code quality acceptable
            num_issues: Total number of issues

        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.0

        # Weighted component scores
        if syntax_valid:
            score += 0.30
        if security_clean:
            score += 0.25
        if onex_compliant:
            score += 0.20
        if type_hints_valid:
            score += 0.15
        if quality_ok:
            score += 0.10

        # Penalty for number of issues (max -0.2)
        issue_penalty = min(num_issues * 0.05, 0.2)
        score = max(0.0, score - issue_penalty)

        return round(score, 2)


__all__ = ["CodeValidator", "ValidationResult", "GenerationContext"]
