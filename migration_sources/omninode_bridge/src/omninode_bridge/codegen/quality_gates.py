#!/usr/bin/env python3
"""
Quality Gates Pipeline for Code Validation.

Provides multi-stage validation pipeline for generated code:
1. Syntax validation (AST parsing)
2. Type checking (mypy integration)
3. ONEX v2.0 compliance validation
4. Code injection validation (ensure stubs replaced)
5. Security scanning (basic checks)

ONEX v2.0 Compliance:
- Structured validation with detailed reporting
- Configurable validation levels (strict/permissive/development)
- Comprehensive logging and metrics
- Integration with existing CodeValidator

Example Usage:
    >>> pipeline = QualityGatePipeline(validation_level="strict")
    >>> result = await pipeline.validate(generated_code)
    >>> if not result.passed:
    ...     print(f"Validation failed: {result.failed_stages}")
    ...     for issue in result.all_issues:
    ...         print(f"  - {issue}")
"""

import ast
import logging
import re
import subprocess
import tempfile
import time
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from .business_logic.models import GenerationContext as ModelsGenerationContext
from .business_logic.validator import CodeValidator, GenerationContext
from .business_logic.validator import ValidationResult as ValidatorResult

logger = logging.getLogger(__name__)


class ValidationLevel(str, Enum):
    """
    Validation strictness level.

    Levels:
    - STRICT: All checks must pass, no warnings allowed
    - PERMISSIVE: Critical checks must pass, warnings allowed
    - DEVELOPMENT: Only syntax and critical security checks
    """

    STRICT = "strict"
    PERMISSIVE = "permissive"
    DEVELOPMENT = "development"


class ValidationStage(str, Enum):
    """
    Individual validation stage identifiers.

    Each stage represents a specific validation checkpoint.
    """

    SYNTAX = "syntax"
    TYPE_CHECKING = "type_checking"
    ONEX_COMPLIANCE = "onex_compliance"
    CODE_INJECTION = "code_injection"
    SECURITY = "security"


class StageResult(BaseModel):
    """
    Result of a single validation stage.

    Tracks pass/fail status, issues found, and performance metrics.
    """

    stage: ValidationStage = Field(..., description="Stage identifier")
    passed: bool = Field(..., description="Stage passed validation")
    issues: list[str] = Field(default_factory=list, description="Issues found in stage")
    warnings: list[str] = Field(
        default_factory=list, description="Non-critical warnings"
    )
    execution_time_ms: float = Field(
        default=0.0, ge=0.0, description="Stage execution time in milliseconds"
    )
    skipped: bool = Field(
        default=False, description="Stage was skipped due to validation level"
    )


class ValidationResult(BaseModel):
    """
    Complete validation result across all stages.

    Aggregates results from all quality gate stages with detailed reporting.
    """

    # Overall result
    passed: bool = Field(
        ..., description="Overall validation passed all enabled stages"
    )
    validation_level: ValidationLevel = Field(..., description="Validation level used")
    quality_score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall quality score (0.0-1.0)"
    )

    # Stage results
    stage_results: list[StageResult] = Field(
        default_factory=list, description="Results from each stage"
    )
    failed_stages: list[str] = Field(
        default_factory=list, description="Names of stages that failed"
    )
    passed_stages: list[str] = Field(
        default_factory=list, description="Names of stages that passed"
    )
    skipped_stages: list[str] = Field(
        default_factory=list, description="Names of stages that were skipped"
    )

    # Aggregated issues
    all_issues: list[str] = Field(
        default_factory=list, description="All issues across stages"
    )
    all_warnings: list[str] = Field(
        default_factory=list, description="All warnings across stages"
    )
    critical_issues: list[str] = Field(
        default_factory=list, description="Critical issues that must be fixed"
    )

    # Code validator result (detailed)
    validator_result: Optional[ValidatorResult] = Field(
        None, description="Detailed result from CodeValidator"
    )

    # Metrics
    total_execution_time_ms: float = Field(
        default=0.0, ge=0.0, description="Total pipeline execution time"
    )
    total_issues_count: int = Field(
        default=0, ge=0, description="Total number of issues"
    )
    total_warnings_count: int = Field(
        default=0, ge=0, description="Total number of warnings"
    )

    # Metadata
    timestamp: float = Field(
        default_factory=time.time, description="Validation timestamp"
    )


class QualityGatePipeline:
    """
    Multi-stage validation pipeline for generated code.

    Orchestrates comprehensive code validation through 5 stages:
    1. Syntax validation - AST parsing
    2. Type checking - mypy integration
    3. ONEX compliance - Framework patterns
    4. Code injection - Stub detection
    5. Security - Basic security checks

    Supports configurable validation levels:
    - STRICT: All stages required, no warnings
    - PERMISSIVE: Critical stages only, warnings allowed
    - DEVELOPMENT: Minimal validation for rapid iteration

    Example:
        >>> pipeline = QualityGatePipeline(validation_level="strict")
        >>> result = await pipeline.validate(
        ...     generated_code=llm_output,
        ...     context=GenerationContext(
        ...         node_type="effect",
        ...         service_name="postgres_crud"
        ...     )
        ... )
        >>> if result.passed:
        ...     print(f"Quality score: {result.quality_score:.2f}")
        ... else:
        ...     print(f"Failed stages: {result.failed_stages}")
    """

    def __init__(
        self,
        validation_level: str = "strict",
        enable_mypy: bool = True,
        mypy_config_path: Optional[Path] = None,
    ):
        """
        Initialize quality gate pipeline.

        Args:
            validation_level: Validation strictness (strict/permissive/development)
            enable_mypy: Enable mypy type checking stage
            mypy_config_path: Path to mypy configuration file (optional)

        Raises:
            ValueError: If validation_level is invalid
        """
        try:
            self.validation_level = ValidationLevel(validation_level)
        except ValueError as e:
            raise ValueError(
                f"Invalid validation_level '{validation_level}'. "
                f"Must be one of: {[v.value for v in ValidationLevel]}"
            ) from e

        self.enable_mypy = enable_mypy
        self.mypy_config_path = mypy_config_path

        # Initialize CodeValidator based on validation level
        # STRICT mode uses strict validation, others use lenient
        strict_mode = self.validation_level == ValidationLevel.STRICT
        self.code_validator = CodeValidator(strict=strict_mode)

        logger.info(
            f"QualityGatePipeline initialized "
            f"(level={validation_level}, mypy={enable_mypy}, strict={strict_mode})"
        )

    async def validate(
        self,
        generated_code: str,
        context: Optional[ModelsGenerationContext] = None,
    ) -> ValidationResult:
        """
        Validate generated code through all quality gate stages.

        Executes validation stages in sequence:
        1. Syntax validation (always runs)
        2. Type checking (if enabled)
        3. ONEX compliance (strict/permissive only)
        4. Code injection check (strict/permissive only)
        5. Security scan (always runs)

        Args:
            generated_code: Code to validate
            context: Optional generation context for validation

        Returns:
            ValidationResult with comprehensive validation report

        Example:
            >>> pipeline = QualityGatePipeline()
            >>> context = GenerationContext(
            ...     node_type="effect",
            ...     service_name="postgres_crud"
            ... )
            >>> result = await pipeline.validate(code, context)
            >>> print(f"Passed: {result.passed}, Score: {result.quality_score}")
        """
        start_time = time.time()
        stage_results: list[StageResult] = []

        logger.info(
            f"Starting quality gate validation "
            f"(level={self.validation_level.value}, code_length={len(generated_code)})"
        )

        # Stage 1: Syntax validation (always runs)
        syntax_result = await self._validate_syntax(generated_code)
        stage_results.append(syntax_result)

        # If syntax fails, stop here (can't continue without valid syntax)
        if not syntax_result.passed:
            logger.warning("Syntax validation failed, skipping remaining stages")
            return self._build_validation_result(
                stage_results=stage_results,
                total_time_ms=(time.time() - start_time) * 1000,
            )

        # Stage 2: Type checking (if enabled and not development mode)
        if self.enable_mypy and self.validation_level != ValidationLevel.DEVELOPMENT:
            type_result = await self._validate_types(generated_code)
            stage_results.append(type_result)
        else:
            logger.debug("Type checking skipped (disabled or development mode)")
            stage_results.append(
                StageResult(
                    stage=ValidationStage.TYPE_CHECKING,
                    passed=True,
                    skipped=True,
                )
            )

        # Stage 3: ONEX compliance (strict/permissive only)
        if self.validation_level != ValidationLevel.DEVELOPMENT:
            onex_result = await self._validate_onex_compliance(generated_code, context)
            stage_results.append(onex_result)
        else:
            logger.debug("ONEX compliance skipped (development mode)")
            stage_results.append(
                StageResult(
                    stage=ValidationStage.ONEX_COMPLIANCE,
                    passed=True,
                    skipped=True,
                )
            )

        # Stage 4: Code injection validation (strict/permissive only)
        if self.validation_level != ValidationLevel.DEVELOPMENT:
            injection_result = await self._validate_code_injection(generated_code)
            stage_results.append(injection_result)
        else:
            logger.debug("Code injection validation skipped (development mode)")
            stage_results.append(
                StageResult(
                    stage=ValidationStage.CODE_INJECTION,
                    passed=True,
                    skipped=True,
                )
            )

        # Stage 5: Security scan (always runs)
        security_result = await self._validate_security(generated_code)
        stage_results.append(security_result)

        # Build final result
        total_time_ms = (time.time() - start_time) * 1000
        result = self._build_validation_result(
            stage_results=stage_results,
            total_time_ms=total_time_ms,
        )

        logger.info(
            f"Quality gate validation complete "
            f"(passed={result.passed}, score={result.quality_score:.2f}, "
            f"time={total_time_ms:.1f}ms, issues={result.total_issues_count})"
        )

        return result

    async def _validate_syntax(self, code: str) -> StageResult:
        """
        Validate Python syntax using AST parsing.

        Catches all Python syntax errors:
        - IndentationError: Incorrect indentation
        - TabError: Inconsistent use of tabs/spaces
        - SyntaxError: General syntax errors

        Args:
            code: Source code to validate

        Returns:
            StageResult for syntax validation
        """
        start_time = time.time()
        issues = []

        try:
            ast.parse(code)
            logger.debug("Syntax validation passed")
        except IndentationError as e:
            error_msg = f"IndentationError at line {e.lineno}: {e.msg}"
            if e.text:
                error_msg += f"\n  {e.text.strip()}"
            issues.append(error_msg)
            logger.warning(f"Syntax validation failed: {error_msg}")
        except TabError as e:
            error_msg = f"TabError at line {e.lineno}: {e.msg}"
            if e.text:
                error_msg += f"\n  {e.text.strip()}"
            issues.append(error_msg)
            logger.warning(f"Syntax validation failed: {error_msg}")
        except SyntaxError as e:
            error_msg = f"SyntaxError at line {e.lineno}: {e.msg}"
            if e.text:
                error_msg += f"\n  {e.text.strip()}"
            issues.append(error_msg)
            logger.warning(f"Syntax validation failed: {error_msg}")
        except Exception as e:
            issues.append(f"Failed to parse code: {e!s}")
            logger.error(f"Syntax validation error: {e}")

        execution_time = (time.time() - start_time) * 1000

        return StageResult(
            stage=ValidationStage.SYNTAX,
            passed=len(issues) == 0,
            issues=issues,
            execution_time_ms=execution_time,
        )

    async def _validate_types(self, code: str) -> StageResult:
        """
        Validate type hints using mypy.

        Runs mypy in strict mode on the generated code.

        Args:
            code: Source code to validate

        Returns:
            StageResult for type checking
        """
        start_time = time.time()
        issues = []
        warnings = []

        # Write code to temporary file for mypy
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as tmp_file:
                tmp_file.write(code)
                tmp_path = tmp_file.name

            try:
                # Run mypy on temporary file
                cmd = ["mypy", "--strict", "--no-error-summary"]
                if self.mypy_config_path:
                    cmd.extend(["--config-file", str(self.mypy_config_path)])
                cmd.append(tmp_path)

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                # Parse mypy output
                if result.returncode != 0:
                    output_lines = result.stdout.strip().split("\n")
                    for line in output_lines:
                        if "error:" in line.lower():
                            # Extract error message (remove file path)
                            error_msg = line.split("error:", 1)[1].strip()
                            issues.append(f"Type error: {error_msg}")
                        elif "note:" in line.lower():
                            # Notes are warnings
                            note_msg = line.split("note:", 1)[1].strip()
                            warnings.append(f"Type note: {note_msg}")

                    logger.debug(
                        f"mypy found {len(issues)} errors and {len(warnings)} notes"
                    )
                else:
                    logger.debug("Type checking passed")

            finally:
                # Clean up temporary file
                Path(tmp_path).unlink(missing_ok=True)

        except subprocess.TimeoutExpired:
            issues.append("Type checking timed out (>30s)")
            logger.error("mypy timed out")
        except FileNotFoundError:
            warnings.append(
                "mypy not found - type checking skipped. Install with: pip install mypy"
            )
            logger.warning("mypy not installed, skipping type checking")
        except Exception as e:
            warnings.append(f"Type checking failed: {e!s}")
            logger.error(f"Type checking error: {e}")

        execution_time = (time.time() - start_time) * 1000

        # In permissive mode, warnings don't fail the stage
        passed = len(issues) == 0

        return StageResult(
            stage=ValidationStage.TYPE_CHECKING,
            passed=passed,
            issues=issues,
            warnings=warnings,
            execution_time_ms=execution_time,
        )

    async def _validate_onex_compliance(
        self, code: str, context: Optional[ModelsGenerationContext]
    ) -> StageResult:
        """
        Validate ONEX v2.0 compliance using CodeValidator.

        Checks for:
        - ModelOnexError usage
        - emit_log_event logging
        - omnibase_core imports
        - Type hints
        - Docstrings

        Args:
            code: Source code to validate
            context: Generation context (optional)

        Returns:
            StageResult for ONEX compliance
        """
        start_time = time.time()

        # Use CodeValidator for ONEX compliance
        # Convert ModelsGenerationContext to validator GenerationContext
        if context is None:
            # Create minimal context
            validator_context = GenerationContext(
                node_type="unknown",
                service_name="unknown",
                method_name="unknown",
            )
        else:
            validator_context = GenerationContext(
                node_type=context.node_type,
                method_name="execute_effect",  # Default method name
                service_name=context.service_name,
            )

        validator_result = await self.code_validator.validate(code, validator_context)

        # Extract ONEX-specific issues
        issues = []
        warnings = []

        # ONEX compliance issues are critical
        issues.extend(validator_result.onex_issues)

        # Type hints and quality issues are warnings in permissive mode
        if self.validation_level == ValidationLevel.STRICT:
            issues.extend(validator_result.type_hint_issues)
            issues.extend(validator_result.quality_issues)
        else:
            warnings.extend(validator_result.type_hint_issues)
            warnings.extend(validator_result.quality_issues)

        execution_time = (time.time() - start_time) * 1000

        logger.debug(
            f"ONEX compliance: {len(issues)} issues, {len(warnings)} warnings, "
            f"compliant={validator_result.onex_compliant}"
        )

        return StageResult(
            stage=ValidationStage.ONEX_COMPLIANCE,
            passed=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            execution_time_ms=execution_time,
        )

    async def _validate_code_injection(self, code: str) -> StageResult:
        """
        Validate that all stubs have been replaced with implementations.

        Checks for common stub patterns:
        - TODO comments
        - pass statements (in method bodies)
        - NotImplementedError
        - IMPLEMENTATION REQUIRED markers

        Args:
            code: Source code to validate

        Returns:
            StageResult for code injection validation
        """
        start_time = time.time()
        issues = []

        # Pattern 1: TODO comments
        todo_pattern = r"#\s*TODO[:\s]"
        todo_matches = re.finditer(todo_pattern, code, re.IGNORECASE)
        for match in todo_matches:
            line_num = code[: match.start()].count("\n") + 1
            issues.append(f"Line {line_num}: TODO comment found (stub not replaced)")

        # Pattern 2: IMPLEMENTATION REQUIRED markers
        impl_pattern = r"#\s*IMPLEMENTATION REQUIRED"
        impl_matches = re.finditer(impl_pattern, code, re.IGNORECASE)
        for match in impl_matches:
            line_num = code[: match.start()].count("\n") + 1
            issues.append(
                f"Line {line_num}: IMPLEMENTATION REQUIRED marker found (stub not replaced)"
            )

        # Pattern 3: NotImplementedError
        not_impl_pattern = r"raise\s+NotImplementedError"
        not_impl_matches = re.finditer(not_impl_pattern, code)
        for match in not_impl_matches:
            line_num = code[: match.start()].count("\n") + 1
            issues.append(
                f"Line {line_num}: NotImplementedError found (stub not replaced)"
            )

        # Pattern 4: Suspicious pass statements (in methods with minimal content)
        # This is a heuristic - detect methods that only contain 'pass'
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    # Skip __init__ and other dunder methods (they can be minimal)
                    if node.name.startswith("__") and node.name.endswith("__"):
                        continue

                    # Check if method body is just 'pass' or docstring + 'pass'
                    body = node.body
                    # Skip docstring if present
                    if (
                        body
                        and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                    ):
                        body = body[1:]

                    # If remaining body is just 'pass', flag it
                    if (
                        len(body) == 1
                        and isinstance(body[0], ast.Pass)
                        and not node.name.startswith("_")
                    ):
                        issues.append(
                            f"Method '{node.name}' contains only 'pass' statement (likely unimplemented stub)"
                        )

        except SyntaxError:
            # Syntax errors are caught in syntax stage, skip AST analysis
            pass

        execution_time = (time.time() - start_time) * 1000

        logger.debug(f"Code injection validation: {len(issues)} stubs found")

        return StageResult(
            stage=ValidationStage.CODE_INJECTION,
            passed=len(issues) == 0,
            issues=issues,
            execution_time_ms=execution_time,
        )

    async def _validate_security(self, code: str) -> StageResult:
        """
        Validate code for security issues using CodeValidator.

        Checks for:
        - Hardcoded secrets
        - SQL injection vulnerabilities
        - Dangerous patterns (eval, exec, pickle)

        Args:
            code: Source code to validate

        Returns:
            StageResult for security validation
        """
        start_time = time.time()

        # Use CodeValidator for security checks
        # Create minimal context
        context = GenerationContext(
            node_type="unknown",
            service_name="unknown",
            method_name="unknown",
        )

        validator_result = await self.code_validator.validate(code, context)

        # Security issues are always critical
        issues = validator_result.security_issues

        execution_time = (time.time() - start_time) * 1000

        logger.debug(
            f"Security validation: {len(issues)} issues, clean={validator_result.security_clean}"
        )

        return StageResult(
            stage=ValidationStage.SECURITY,
            passed=len(issues) == 0,
            issues=issues,
            execution_time_ms=execution_time,
        )

    def _build_validation_result(
        self,
        stage_results: list[StageResult],
        total_time_ms: float,
    ) -> ValidationResult:
        """
        Build comprehensive validation result from stage results.

        Args:
            stage_results: Results from all validation stages
            total_time_ms: Total execution time in milliseconds

        Returns:
            ValidationResult with aggregated data
        """
        # Aggregate results
        failed_stages = []
        passed_stages = []
        skipped_stages = []
        all_issues = []
        all_warnings = []
        critical_issues = []

        for result in stage_results:
            stage_name = result.stage.value

            if result.skipped:
                skipped_stages.append(stage_name)
            elif result.passed:
                passed_stages.append(stage_name)
            else:
                failed_stages.append(stage_name)

            # Aggregate issues and warnings
            all_issues.extend(result.issues)
            all_warnings.extend(result.warnings)

            # Mark syntax and security issues as critical
            if result.stage in (ValidationStage.SYNTAX, ValidationStage.SECURITY):
                critical_issues.extend(result.issues)

        # Determine overall pass/fail
        # In strict mode, any failed stage fails validation
        # In permissive/development mode, only critical failures fail validation
        if self.validation_level == ValidationLevel.STRICT:
            passed = len(failed_stages) == 0
        else:
            # Only fail if syntax or security failed
            critical_failed = any(
                stage in failed_stages for stage in ["syntax", "security"]
            )
            passed = not critical_failed

        # Calculate quality score
        quality_score = self._calculate_quality_score(stage_results)

        # Get validator result if available (from ONEX compliance stage)
        # Note: Removed asyncio.run() call as it conflicts with async context
        # The validator result is not critical for ValidationResult construction
        validator_result = None

        return ValidationResult(
            passed=passed,
            validation_level=self.validation_level,
            quality_score=quality_score,
            stage_results=stage_results,
            failed_stages=failed_stages,
            passed_stages=passed_stages,
            skipped_stages=skipped_stages,
            all_issues=all_issues,
            all_warnings=all_warnings,
            critical_issues=critical_issues,
            total_execution_time_ms=total_time_ms,
            total_issues_count=len(all_issues),
            total_warnings_count=len(all_warnings),
            validator_result=validator_result,
        )

    def _calculate_quality_score(self, stage_results: list[StageResult]) -> float:
        """
        Calculate overall quality score from stage results.

        Weighted scoring:
        - Syntax: 25% (critical)
        - Type checking: 20%
        - ONEX compliance: 25%
        - Code injection: 15%
        - Security: 15% (critical)

        Skipped stages count as passed for scoring purposes.

        Args:
            stage_results: Results from all stages

        Returns:
            Quality score between 0.0 and 1.0
        """
        weights = {
            ValidationStage.SYNTAX: 0.25,
            ValidationStage.TYPE_CHECKING: 0.20,
            ValidationStage.ONEX_COMPLIANCE: 0.25,
            ValidationStage.CODE_INJECTION: 0.15,
            ValidationStage.SECURITY: 0.15,
        }

        score = 0.0

        for result in stage_results:
            weight = weights.get(result.stage, 0.0)

            # Skipped stages count as passed (full weight)
            if result.skipped or result.passed:
                score += weight
            else:
                # Partial credit based on number of issues
                # More issues = lower partial credit
                if len(result.issues) == 0:
                    partial = 1.0
                elif len(result.issues) <= 2:
                    partial = 0.5
                elif len(result.issues) <= 5:
                    partial = 0.25
                else:
                    partial = 0.0

                score += weight * partial

        return round(score, 2)


__all__ = [
    "QualityGatePipeline",
    "ValidationResult",
    "ValidationLevel",
    "ValidationStage",
    "StageResult",
]
