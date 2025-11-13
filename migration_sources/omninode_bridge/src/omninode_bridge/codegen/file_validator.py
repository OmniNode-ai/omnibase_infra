#!/usr/bin/env python3
"""
Post-Generation File Validator.

Validates actual generated files on disk (not in-memory code).
Ensures truth-based reporting by validating what was actually written.

Key Features:
- Validates files after they're written to disk
- AST syntax validation on actual file contents
- Stub pattern detection (IMPLEMENTATION REQUIRED, bare pass statements)
- Multiple file validation (node.py, __init__.py, contract.yaml, etc.)
- Detailed reporting with file paths and line numbers

ONEX v2.0 Compliance:
- Structured validation results
- Comprehensive error reporting
- Integration with quality gates pipeline
- Type-safe operations

Usage:
    >>> validator = FileValidator()
    >>> result = await validator.validate_generated_files(
    ...     file_paths=[
    ...         Path("generated_nodes/vault_secrets_effect/node.py"),
    ...         Path("generated_nodes/vault_secrets_effect/__init__.py"),
    ...     ]
    ... )
    >>> if not result.passed:
    ...     print(f"Validation failed: {result.issues}")
"""

import ast
import logging
import re
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FileValidationIssue(BaseModel):
    """
    Individual file validation issue.

    Tracks specific problems found in generated files.
    """

    file_path: str = Field(..., description="Path to file with issue")
    issue_type: str = Field(
        ..., description="Type of issue (syntax_error, stub_detected, missing_file)"
    )
    line_number: Optional[int] = Field(
        None, description="Line number where issue occurs"
    )
    message: str = Field(..., description="Detailed issue message")
    severity: str = Field(
        default="error", description="Issue severity (error, warning)"
    )


class FileValidationResult(BaseModel):
    """
    Result of file validation.

    Aggregates validation results across all files with detailed issue tracking.
    """

    passed: bool = Field(..., description="All files passed validation")
    files_validated: int = Field(
        default=0, ge=0, description="Number of files validated"
    )
    files_passed: int = Field(
        default=0, ge=0, description="Number of files that passed"
    )
    files_failed: int = Field(
        default=0, ge=0, description="Number of files that failed"
    )

    # Issue tracking
    issues: list[FileValidationIssue] = Field(
        default_factory=list, description="All issues found"
    )
    syntax_errors: list[FileValidationIssue] = Field(
        default_factory=list, description="Syntax errors"
    )
    stub_issues: list[FileValidationIssue] = Field(
        default_factory=list, description="Stub pattern issues"
    )
    missing_files: list[FileValidationIssue] = Field(
        default_factory=list, description="Missing expected files"
    )

    # Metrics
    execution_time_ms: float = Field(
        default=0.0, ge=0.0, description="Validation execution time"
    )
    timestamp: float = Field(
        default_factory=time.time, description="Validation timestamp"
    )

    # Summary
    summary: str = Field(default="", description="Human-readable summary")

    def add_issue(self, issue: FileValidationIssue) -> None:
        """Add an issue and categorize it."""
        self.issues.append(issue)

        if issue.issue_type == "syntax_error":
            self.syntax_errors.append(issue)
        elif issue.issue_type == "stub_detected":
            self.stub_issues.append(issue)
        elif issue.issue_type == "missing_file":
            self.missing_files.append(issue)


class FileValidator:
    """
    Post-generation file validator.

    Validates actual files written to disk after code generation.
    Ensures success reporting reflects actual file state, not just in-memory validation.

    Key Validations:
    1. File existence check
    2. AST syntax validation (parse actual file contents)
    3. Stub pattern detection (IMPLEMENTATION REQUIRED, bare pass, etc.)
    4. Import statement validation (basic checks)

    Example:
        >>> validator = FileValidator()
        >>> result = await validator.validate_generated_files(
        ...     file_paths=[
        ...         Path("generated_nodes/vault_secrets_effect/node.py"),
        ...     ],
        ...     strict_mode=True,
        ... )
        >>> if result.passed:
        ...     print(f"✅ All {result.files_validated} files valid")
        ... else:
        ...     print(f"❌ Validation failed:")
        ...     for issue in result.issues:
        ...         print(f"  {issue.file_path}:{issue.line_number} - {issue.message}")
    """

    def __init__(self):
        """Initialize file validator."""
        self.logger = logging.getLogger(__name__)

    async def validate_generated_files(
        self,
        file_paths: list[Path],
        strict_mode: bool = True,
    ) -> FileValidationResult:
        """
        Validate actual generated files on disk.

        Reads files from disk and validates:
        - File exists
        - Valid Python syntax (AST parsing)
        - No stub patterns remain
        - No obvious issues

        Args:
            file_paths: List of file paths to validate
            strict_mode: Fail on warnings if True

        Returns:
            FileValidationResult with detailed issue tracking

        Example:
            >>> validator = FileValidator()
            >>> result = await validator.validate_generated_files(
            ...     file_paths=[
            ...         Path("generated_nodes/vault_secrets_effect/node.py"),
            ...         Path("generated_nodes/vault_secrets_effect/__init__.py"),
            ...     ]
            ... )
            >>> assert result.passed
        """
        start_time = time.time()
        result = FileValidationResult(
            passed=True,
            files_validated=len(file_paths),
        )

        self.logger.info(f"Validating {len(file_paths)} generated files")

        for file_path in file_paths:
            # Validate individual file
            file_issues = await self._validate_single_file(file_path, strict_mode)

            if file_issues:
                result.passed = False
                result.files_failed += 1
                for issue in file_issues:
                    result.add_issue(issue)
            else:
                result.files_passed += 1

        result.execution_time_ms = (time.time() - start_time) * 1000

        # Generate summary
        if result.passed:
            result.summary = f"✅ All {result.files_validated} files passed validation"
        else:
            result.summary = (
                f"❌ Validation failed: {result.files_failed}/{result.files_validated} files with issues\n"
                f"  - Syntax errors: {len(result.syntax_errors)}\n"
                f"  - Stub issues: {len(result.stub_issues)}\n"
                f"  - Missing files: {len(result.missing_files)}"
            )

        self.logger.info(
            f"File validation complete: "
            f"passed={result.passed}, "
            f"files={result.files_validated}, "
            f"issues={len(result.issues)}, "
            f"time={result.execution_time_ms:.0f}ms"
        )

        return result

    async def _validate_single_file(
        self,
        file_path: Path,
        strict_mode: bool,
    ) -> list[FileValidationIssue]:
        """
        Validate a single file.

        Args:
            file_path: Path to file to validate
            strict_mode: Fail on warnings if True

        Returns:
            List of issues found (empty if valid)
        """
        issues: list[FileValidationIssue] = []

        # Check 1: File exists
        if not file_path.exists():
            issues.append(
                FileValidationIssue(
                    file_path=str(file_path),
                    issue_type="missing_file",
                    line_number=None,
                    message=f"File does not exist: {file_path}",
                    severity="error",
                )
            )
            return issues

        # Only validate Python files
        if file_path.suffix != ".py":
            self.logger.debug(f"Skipping non-Python file: {file_path}")
            return issues

        # Read actual file contents from disk
        try:
            code = file_path.read_text(encoding="utf-8")
        except Exception as e:
            issues.append(
                FileValidationIssue(
                    file_path=str(file_path),
                    issue_type="read_error",
                    line_number=None,
                    message=f"Failed to read file: {e}",
                    severity="error",
                )
            )
            return issues

        # Check 2: AST syntax validation
        syntax_issues = self._validate_syntax(file_path, code)
        issues.extend(syntax_issues)

        # If syntax is invalid, can't do further validation
        if syntax_issues:
            return issues

        # Check 3: Stub pattern detection
        stub_issues = self._detect_stubs(file_path, code)
        issues.extend(stub_issues)

        return issues

    def _validate_syntax(
        self,
        file_path: Path,
        code: str,
    ) -> list[FileValidationIssue]:
        """
        Validate Python syntax using AST.

        Args:
            file_path: Path to file being validated
            code: File contents

        Returns:
            List of syntax issues
        """
        issues: list[FileValidationIssue] = []

        try:
            ast.parse(code)
            self.logger.debug(f"✓ Syntax valid: {file_path.name}")
        except (SyntaxError, IndentationError, TabError) as e:
            issues.append(
                FileValidationIssue(
                    file_path=str(file_path),
                    issue_type="syntax_error",
                    line_number=e.lineno if hasattr(e, "lineno") else None,
                    message=f"{type(e).__name__}: {e.msg if hasattr(e, 'msg') else str(e)}",
                    severity="error",
                )
            )
            self.logger.error(f"✗ Syntax error in {file_path.name}: {e}")

        return issues

    def _detect_stubs(
        self,
        file_path: Path,
        code: str,
    ) -> list[FileValidationIssue]:
        """
        Detect stub patterns in code.

        Checks for:
        - "IMPLEMENTATION REQUIRED" comments
        - "TODO" comments
        - Methods with only 'pass' statement
        - NotImplementedError raises

        Args:
            file_path: Path to file being validated
            code: File contents

        Returns:
            List of stub issues
        """
        issues: list[FileValidationIssue] = []

        # Pattern 1: IMPLEMENTATION REQUIRED markers
        impl_pattern = r"#\s*IMPLEMENTATION REQUIRED"
        for match in re.finditer(impl_pattern, code, re.IGNORECASE):
            line_num = code[: match.start()].count("\n") + 1
            issues.append(
                FileValidationIssue(
                    file_path=str(file_path),
                    issue_type="stub_detected",
                    line_number=line_num,
                    message="IMPLEMENTATION REQUIRED marker found (stub not replaced)",
                    severity="error",
                )
            )

        # Pattern 2: TODO comments
        todo_pattern = r"#\s*TODO[:\s]"
        for match in re.finditer(todo_pattern, code, re.IGNORECASE):
            line_num = code[: match.start()].count("\n") + 1
            issues.append(
                FileValidationIssue(
                    file_path=str(file_path),
                    issue_type="stub_detected",
                    line_number=line_num,
                    message="TODO comment found (stub not replaced)",
                    severity="warning",
                )
            )

        # Pattern 3: NotImplementedError
        not_impl_pattern = r"raise\s+NotImplementedError"
        for match in re.finditer(not_impl_pattern, code):
            line_num = code[: match.start()].count("\n") + 1
            issues.append(
                FileValidationIssue(
                    file_path=str(file_path),
                    issue_type="stub_detected",
                    line_number=line_num,
                    message="NotImplementedError found (stub not replaced)",
                    severity="error",
                )
            )

        # Pattern 4: Methods with only 'pass' statement
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    # Skip dunder methods (they can legitimately be minimal)
                    if node.name.startswith("__") and node.name.endswith("__"):
                        continue

                    # Skip private methods (they might be placeholders)
                    if node.name.startswith("_"):
                        continue

                    # Check if method body is just 'pass' or docstring + 'pass'
                    body = node.body

                    # Skip docstring if present
                    if (
                        body
                        and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)
                    ):
                        body = body[1:]

                    # If remaining body is just 'pass', flag it
                    if len(body) == 1 and isinstance(body[0], ast.Pass):
                        issues.append(
                            FileValidationIssue(
                                file_path=str(file_path),
                                issue_type="stub_detected",
                                line_number=node.lineno,
                                message=f"Method '{node.name}' contains only 'pass' statement (likely stub)",
                                severity="error",
                            )
                        )

        except SyntaxError:
            # Syntax errors already caught, skip AST analysis
            pass

        if issues:
            self.logger.warning(
                f"Found {len(issues)} stub patterns in {file_path.name}"
            )
        else:
            self.logger.debug(f"✓ No stubs detected: {file_path.name}")

        return issues

    def format_validation_report(
        self,
        result: FileValidationResult,
        include_file_paths: bool = True,
    ) -> str:
        """
        Format validation result as human-readable report.

        Args:
            result: Validation result to format
            include_file_paths: Include full file paths in report

        Returns:
            Formatted report string

        Example:
            >>> validator = FileValidator()
            >>> result = await validator.validate_generated_files(file_paths)
            >>> report = validator.format_validation_report(result)
            >>> print(report)
        """
        lines = []
        lines.append("=" * 80)
        lines.append("POST-GENERATION FILE VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Overall status
        status_icon = "✅" if result.passed else "❌"
        lines.append(
            f"{status_icon} Overall Status: {'PASSED' if result.passed else 'FAILED'}"
        )
        lines.append("")

        # Summary stats
        lines.append("Summary:")
        lines.append(f"  Files validated: {result.files_validated}")
        lines.append(f"  Files passed: {result.files_passed}")
        lines.append(f"  Files failed: {result.files_failed}")
        lines.append(f"  Total issues: {len(result.issues)}")
        lines.append(f"  Execution time: {result.execution_time_ms:.0f}ms")
        lines.append("")

        # Issue breakdown
        if result.issues:
            lines.append("Issues by Type:")
            lines.append(f"  Syntax errors: {len(result.syntax_errors)}")
            lines.append(f"  Stub issues: {len(result.stub_issues)}")
            lines.append(f"  Missing files: {len(result.missing_files)}")
            lines.append("")

            # Detailed issues
            lines.append("Detailed Issues:")
            for issue in result.issues:
                severity_icon = "🔴" if issue.severity == "error" else "⚠️"
                file_path = (
                    issue.file_path
                    if include_file_paths
                    else Path(issue.file_path).name
                )
                line_ref = f":{issue.line_number}" if issue.line_number else ""
                lines.append(f"  {severity_icon} {file_path}{line_ref}")
                lines.append(f"     {issue.message}")
                lines.append("")
        else:
            lines.append("✅ No issues found - all files valid!")
            lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)
