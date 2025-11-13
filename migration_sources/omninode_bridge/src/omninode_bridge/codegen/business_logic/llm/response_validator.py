#!/usr/bin/env python3
"""
Response Validator for LLM Code Generation.

Validates generated code for:
- Syntax correctness
- ONEX v2.0 compliance
- Pattern compliance (Phase 3 NEW)
- Template variant alignment (Phase 3 NEW)
- Security issues

Performance Target: <10ms per validation
"""

import ast
import logging
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from .response_parser import ModelParsedResponse

logger = logging.getLogger(__name__)


class ModelValidationResult(BaseModel):
    """Result of code validation with detailed findings."""

    passed: bool = Field(..., description="Overall validation passed")
    syntax_valid: bool = Field(..., description="Syntax check passed")
    onex_compliant: bool = Field(..., description="ONEX v2.0 compliance")
    pattern_compliant: bool = Field(..., description="Pattern compliance (Phase 3)")
    template_aligned: bool = Field(
        ..., description="Template variant alignment (Phase 3)"
    )
    security_clean: bool = Field(..., description="No security issues found")

    # Detailed findings
    issues: list[str] = Field(
        default_factory=list, description="Validation issues found"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Non-critical warnings"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Improvement recommendations"
    )

    # Compliance details
    has_type_hints: bool = Field(default=False, description="Type hints present")
    has_async: bool = Field(default=False, description="Uses async/await")
    has_error_handling: bool = Field(default=False, description="Has error handling")
    has_logging: bool = Field(default=False, description="Has emit_log_event calls")
    uses_model_onex_error: bool = Field(
        default=False, description="Uses ModelOnexError"
    )

    # Pattern compliance (Phase 3)
    patterns_found: list[str] = Field(
        default_factory=list, description="Patterns detected in code"
    )
    patterns_expected: list[str] = Field(
        default_factory=list, description="Patterns expected from context"
    )

    # Security findings
    security_issues: list[str] = Field(
        default_factory=list, description="Security issues found"
    )

    # Metrics
    validation_time_ms: float = Field(default=0.0, description="Validation duration")
    can_retry: bool = Field(
        default=False, description="Whether retry with adjusted prompt is recommended"
    )


class EnhancedResponseValidator:
    """
    Validate generated code for quality and compliance.

    Performs comprehensive validation:
    - Syntax validation (AST parsing)
    - ONEX v2.0 compliance checking
    - Pattern compliance (Phase 3)
    - Template variant alignment (Phase 3)
    - Security scanning

    Attributes:
        strict_mode: Whether to fail on warnings
        pattern_matching_enabled: Enable Phase 3 pattern validation
        security_scanning_enabled: Enable security scanning

    Example:
        >>> validator = EnhancedResponseValidator()
        >>> result = validator.validate_generated_method(
        ...     generated_code=code,
        ...     context=llm_context,
        ... )
        >>> if result.passed:
        ...     print("Validation passed!")
    """

    # ONEX v2.0 required patterns
    ONEX_PATTERNS = [
        "ModelOnexError",  # Error handling
        "emit_log_event",  # Logging
        "async def",  # Async methods
        "EnumLogLevel",  # Log levels
    ]

    # Security anti-patterns
    SECURITY_PATTERNS = [
        (r"password\s*=\s*['\"]", "Hardcoded password detected"),
        (r"api_key\s*=\s*['\"]", "Hardcoded API key detected"),
        (r"secret\s*=\s*['\"]", "Hardcoded secret detected"),
        (r"token\s*=\s*['\"](?![{])", "Hardcoded token detected"),
        (r"eval\s*\(", "Use of eval() is dangerous"),
        (r"exec\s*\(", "Use of exec() is dangerous"),
        (r"__import__\s*\(", "Dynamic imports are risky"),
        (r"os\.system\s*\(", "Use of os.system() is risky"),
        (r"subprocess\.call\s*\(", "Use of subprocess.call() without validation"),
    ]

    def __init__(
        self,
        strict_mode: bool = False,
        pattern_matching_enabled: bool = True,
        security_scanning_enabled: bool = True,
    ):
        """
        Initialize response validator.

        Args:
            strict_mode: Fail validation on warnings
            pattern_matching_enabled: Enable Phase 3 pattern validation
            security_scanning_enabled: Enable security scanning
        """
        self.strict_mode = strict_mode
        self.pattern_matching_enabled = pattern_matching_enabled
        self.security_scanning_enabled = security_scanning_enabled

        logger.debug(
            f"EnhancedResponseValidator initialized "
            f"(strict={strict_mode}, patterns={pattern_matching_enabled}, "
            f"security={security_scanning_enabled})"
        )

    def validate_generated_method(
        self,
        generated_code: str,
        context: Any,  # ModelLLMContext
        parsed_response: Optional[ModelParsedResponse] = None,
    ) -> ModelValidationResult:
        """
        Comprehensive validation of LLM output.

        Performs all validation checks and returns detailed results.

        Args:
            generated_code: Generated Python code
            context: LLM context used for generation
            parsed_response: Optional pre-parsed response

        Returns:
            ModelValidationResult with detailed findings

        Performance: <10ms target
        """
        import time

        start_time = time.perf_counter()

        issues = []
        warnings = []
        recommendations = []

        # 1. Syntax validation
        syntax_valid = self._validate_syntax(generated_code)
        if not syntax_valid:
            issues.append("Code has syntax errors - cannot parse with AST")

        # 2. ONEX compliance
        onex_checks = self._validate_onex_compliance(generated_code)
        onex_compliant = onex_checks["compliant"]

        if not onex_compliant:
            issues.extend(onex_checks["issues"])

        # 3. Pattern compliance (Phase 3)
        pattern_compliant = True
        patterns_found = []
        if self.pattern_matching_enabled and hasattr(context, "patterns_included"):
            pattern_result = self._validate_pattern_compliance(
                code=generated_code,
                context=context,
            )
            pattern_compliant = pattern_result["compliant"]
            patterns_found = pattern_result["patterns_found"]

            if not pattern_compliant:
                warnings.extend(pattern_result["warnings"])

        # 4. Template alignment (Phase 3)
        template_aligned = True
        if hasattr(context, "variant_selected"):
            template_result = self._validate_template_alignment(
                code=generated_code,
                variant=context.variant_selected,
            )
            template_aligned = template_result["aligned"]

            if not template_aligned:
                warnings.extend(template_result["warnings"])

        # 5. Security scanning
        security_issues = []
        if self.security_scanning_enabled:
            security_issues = self._scan_security_issues(generated_code)

        security_clean = len(security_issues) == 0

        if security_issues:
            issues.extend(security_issues)

        # 6. Generate recommendations
        recommendations = self._generate_recommendations(
            code=generated_code,
            onex_checks=onex_checks,
        )

        # Determine overall pass/fail
        passed = (
            syntax_valid
            and onex_compliant
            and security_clean
            and (not self.strict_mode or (pattern_compliant and template_aligned))
        )

        # Determine if retry is recommended
        can_retry = not syntax_valid or not onex_compliant

        validation_time_ms = (time.perf_counter() - start_time) * 1000

        logger.debug(
            f"Validation completed: passed={passed}, "
            f"issues={len(issues)}, warnings={len(warnings)} "
            f"({validation_time_ms:.1f}ms)"
        )

        return ModelValidationResult(
            passed=passed,
            syntax_valid=syntax_valid,
            onex_compliant=onex_compliant,
            pattern_compliant=pattern_compliant,
            template_aligned=template_aligned,
            security_clean=security_clean,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations,
            has_type_hints=onex_checks["has_type_hints"],
            has_async=onex_checks["has_async"],
            has_error_handling=onex_checks["has_error_handling"],
            has_logging=onex_checks["has_logging"],
            uses_model_onex_error=onex_checks["uses_model_onex_error"],
            patterns_found=patterns_found,
            patterns_expected=[],  # TODO: Extract from context
            security_issues=security_issues,
            validation_time_ms=validation_time_ms,
            can_retry=can_retry,
        )

    def _validate_syntax(
        self,
        code: str,
    ) -> bool:
        """
        Validate Python syntax with AST parsing.

        Args:
            code: Python code string

        Returns:
            True if syntax is valid
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
        except Exception:
            return False

    def _validate_onex_compliance(
        self,
        code: str,
    ) -> dict[str, Any]:
        """
        Validate ONEX v2.0 compliance.

        Checks for:
        - ModelOnexError usage
        - emit_log_event calls
        - Async/await
        - Type hints
        - Error handling (try/except)

        Args:
            code: Python code string

        Returns:
            Dictionary with compliance results
        """
        result = {
            "compliant": True,
            "issues": [],
            "has_type_hints": False,
            "has_async": False,
            "has_error_handling": False,
            "has_logging": False,
            "uses_model_onex_error": False,
        }

        # Check for ONEX patterns
        if "ModelOnexError" in code:
            result["uses_model_onex_error"] = True
        else:
            result["compliant"] = False
            result["issues"].append("Missing ModelOnexError for error handling")

        if "emit_log_event" in code:
            result["has_logging"] = True
        else:
            result["compliant"] = False
            result["issues"].append("Missing emit_log_event for logging")

        if "async def" in code or "await " in code:
            result["has_async"] = True
        else:
            result["compliant"] = False
            result["issues"].append("Code is not async (missing async/await)")

        if "try:" in code and "except" in code:
            result["has_error_handling"] = True
        else:
            result["compliant"] = False
            result["issues"].append("Missing try/except error handling")

        # Check for type hints (basic check)
        if ": " in code or " -> " in code:
            result["has_type_hints"] = True

        return result

    def _validate_pattern_compliance(
        self,
        code: str,
        context: Any,
    ) -> dict[str, Any]:
        """
        Validate that code uses recommended patterns (Phase 3).

        Args:
            code: Generated code
            context: LLM context with pattern information

        Returns:
            Dictionary with pattern compliance results
        """
        result = {
            "compliant": True,
            "warnings": [],
            "patterns_found": [],
        }

        # TODO: Implement actual pattern detection
        # For now, just check if any pattern keywords are present

        # Common patterns to look for
        pattern_keywords = {
            "connection_pooling": ["pool", "acquire", "connection"],
            "circuit_breaker": ["circuit", "breaker", "failure"],
            "retry": ["retry", "backoff", "attempt"],
            "timeout": ["timeout", "asyncio.timeout"],
            "transaction": ["transaction", "commit", "rollback"],
        }

        for pattern_name, keywords in pattern_keywords.items():
            if any(keyword in code.lower() for keyword in keywords):
                result["patterns_found"].append(pattern_name)

        # Pattern compliance is optional (warnings only, not errors)
        if context.patterns_included > 0 and len(result["patterns_found"]) == 0:
            result["warnings"].append(
                f"No patterns detected, but {context.patterns_included} patterns were provided"
            )

        return result

    def _validate_template_alignment(
        self,
        code: str,
        variant: str,
    ) -> dict[str, Any]:
        """
        Validate that code aligns with template variant (Phase 3).

        Args:
            code: Generated code
            variant: Template variant name

        Returns:
            Dictionary with template alignment results
        """
        result = {
            "aligned": True,
            "warnings": [],
        }

        # Variant-specific checks
        if "database" in variant.lower():
            if "pool" not in code.lower():
                result["warnings"].append(
                    "DATABASE variant selected but no connection pooling detected"
                )

        if "api" in variant.lower():
            if "http" not in code.lower() and "client" not in code.lower():
                result["warnings"].append(
                    "API variant selected but no HTTP client usage detected"
                )

        if "kafka" in variant.lower():
            if "kafka" not in code.lower() and "producer" not in code.lower():
                result["warnings"].append(
                    "KAFKA variant selected but no Kafka usage detected"
                )

        # Alignment issues are warnings, not errors
        return result

    def _scan_security_issues(
        self,
        code: str,
    ) -> list[str]:
        """
        Scan for security anti-patterns.

        Args:
            code: Python code string

        Returns:
            List of security issue descriptions
        """
        issues = []

        for pattern, message in self.SECURITY_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append(message)

        return issues

    def _generate_recommendations(
        self,
        code: str,
        onex_checks: dict[str, Any],
    ) -> list[str]:
        """
        Generate improvement recommendations.

        Args:
            code: Generated code
            onex_checks: ONEX compliance check results

        Returns:
            List of recommendation strings
        """
        recommendations = []

        if not onex_checks.get("has_type_hints"):
            recommendations.append("Add type hints to all variables and parameters")

        if "correlation_id" not in code:
            recommendations.append(
                "Include correlation_id in all log events for tracing"
            )

        if "time.perf_counter" not in code:
            recommendations.append(
                "Use time.perf_counter() for accurate latency measurement"
            )

        if "latency_ms" not in code:
            recommendations.append("Track and return latency_ms in response metadata")

        return recommendations


__all__ = ["EnhancedResponseValidator", "ModelValidationResult"]
