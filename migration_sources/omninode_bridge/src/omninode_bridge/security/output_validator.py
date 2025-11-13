#!/usr/bin/env python3
"""
Output Validation for Security Hardening.

Validates generated code for security issues using AST parsing and pattern matching.

ONEX v2.0 Compliance:
- AST-based code analysis
- Dangerous pattern detection
- Configurable security policies
- Non-blocking review recommendations
"""

import ast
import logging
import re
from dataclasses import dataclass, field
from typing import ClassVar, Optional

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class SecurityIssue:
    """Represents a security issue found in generated code."""

    severity: str  # "low", "medium", "high", "critical"
    message: str
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report for generated code."""

    is_safe: bool
    warnings: list[str] = field(default_factory=list)
    dangerous_patterns: list[SecurityIssue] = field(default_factory=list)
    needs_review: bool = False
    quality_score: float = 1.0  # 0.0-1.0, decreases with issues


class OutputValidator:
    """
    Validates generated code for security issues.

    Features:
    - AST-based Python code analysis
    - Dangerous import detection
    - Dangerous function call detection
    - Hardcoded credentials detection
    - File system access validation
    - Network operation detection
    """

    # Dangerous imports that require review
    DANGEROUS_IMPORTS: ClassVar[dict[str, str]] = {
        "os.system": "critical",
        "subprocess.call": "high",
        "subprocess.run": "medium",
        "subprocess.Popen": "high",
        "exec": "critical",
        "eval": "critical",
        "__import__": "high",
        "compile": "medium",
        "pickle": "medium",  # Can execute arbitrary code
        "shelve": "medium",  # Uses pickle internally
    }

    # Dangerous function calls
    DANGEROUS_FUNCTIONS: ClassVar[dict[str, str]] = {
        "eval": "critical",
        "exec": "critical",
        "compile": "medium",
        "__import__": "high",
        "open": "low",  # Needs context - may be legitimate
        "input": "low",  # May be legitimate for user interaction
    }

    # Sensitive file paths
    SENSITIVE_PATHS: ClassVar[list[str]] = [
        r"/etc/passwd",
        r"/etc/shadow",
        r"/var/www",
        r"~/.ssh",
        r"~/.aws",
        r"~/.config",
        r"/root/",
    ]

    # Credential patterns
    CREDENTIAL_PATTERNS: ClassVar[list[tuple[str, str]]] = [
        (r"\w*password\w*\s*=\s*['\"][^'\"]+['\"]", "Hardcoded password detected"),
        (r"\w*api[_-]?key\w*\s*=\s*['\"][^'\"]+['\"]", "Hardcoded API key detected"),
        (r"\w*secret\w*\s*=\s*['\"][^'\"]+['\"]", "Hardcoded secret detected"),
        (r"\w*token\w*\s*=\s*['\"][^'\"]+['\"]", "Hardcoded token detected"),
        (
            r"\w*aws[_-]?access[_-]?key[_-]?id\w*\s*=\s*['\"][^'\"]+['\"]",
            "Hardcoded AWS credentials detected",
        ),
    ]

    def __init__(self, strict_mode: bool = False):
        """
        Initialize output validator.

        Args:
            strict_mode: If True, reject on first critical issue
        """
        self.strict_mode = strict_mode

    def validate_generated_code(
        self, code: str, language: str = "python"
    ) -> ValidationReport:
        """
        Validate generated code for security issues.

        Args:
            code: Generated code to validate
            language: Programming language (currently only "python" supported)

        Returns:
            ValidationReport with safety assessment and issues
        """
        if language != "python":
            logger.warning(f"Validation not implemented for language: {language}")
            return ValidationReport(
                is_safe=True,
                warnings=[f"No validation available for {language}"],
                needs_review=True,
                quality_score=0.8,  # Lower score for unsupported languages
            )

        warnings: list[str] = []
        dangerous_patterns: list[SecurityIssue] = []

        try:
            # Parse AST
            tree = ast.parse(code)

            # Check imports
            import_issues = self._check_imports(tree)
            dangerous_patterns.extend(import_issues)

            # Check function calls
            call_issues = self._check_function_calls(tree)
            dangerous_patterns.extend(call_issues)

            # Check file operations
            file_issues = self._check_file_operations(tree)
            dangerous_patterns.extend(file_issues)

            # Check for hardcoded credentials (regex-based)
            cred_issues = self._check_credentials(code)
            dangerous_patterns.extend(cred_issues)

            # Check for sensitive paths
            path_issues = self._check_sensitive_paths(code)
            dangerous_patterns.extend(path_issues)

            # Calculate quality score
            quality_score = self._calculate_quality_score(dangerous_patterns)

            # Determine if needs review
            needs_review = any(
                issue.severity in ["high", "critical"] for issue in dangerous_patterns
            )

            # Determine if safe
            critical_issues = [
                issue for issue in dangerous_patterns if issue.severity == "critical"
            ]
            is_safe = len(critical_issues) == 0

            # Generate warnings
            if dangerous_patterns:
                warnings.append(f"Found {len(dangerous_patterns)} security issues")

            # In strict mode, fail on critical issues
            if self.strict_mode and critical_issues:
                is_safe = False

            return ValidationReport(
                is_safe=is_safe,
                warnings=warnings,
                dangerous_patterns=dangerous_patterns,
                needs_review=needs_review,
                quality_score=quality_score,
            )

        except SyntaxError as e:
            return ValidationReport(
                is_safe=False,
                warnings=[f"Syntax error: {e}"],
                dangerous_patterns=[
                    SecurityIssue(
                        severity="high",
                        message=f"Code has syntax errors: {e}",
                        line_number=e.lineno,
                    )
                ],
                needs_review=True,
                quality_score=0.0,
            )

    def _check_imports(self, tree: ast.AST) -> list[SecurityIssue]:
        """Check for dangerous imports."""
        issues: list[SecurityIssue] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for danger_import, severity in self.DANGEROUS_IMPORTS.items():
                        if danger_import in alias.name:
                            issues.append(
                                SecurityIssue(
                                    severity=severity,
                                    message=f"Dangerous import: {alias.name}",
                                    line_number=(
                                        node.lineno if hasattr(node, "lineno") else None
                                    ),
                                )
                            )

            elif isinstance(node, ast.ImportFrom):
                # Check module name
                if node.module:
                    for danger_import, severity in self.DANGEROUS_IMPORTS.items():
                        if danger_import in node.module or node.module in danger_import:
                            issues.append(
                                SecurityIssue(
                                    severity=severity,
                                    message=f"Dangerous import from: {node.module}",
                                    line_number=(
                                        node.lineno if hasattr(node, "lineno") else None
                                    ),
                                )
                            )

                # Check imported names (e.g., "from subprocess import call")
                for alias in node.names:
                    import_name = alias.name
                    # Construct full name if module exists
                    if node.module:
                        full_name = f"{node.module}.{import_name}"
                    else:
                        full_name = import_name

                    for danger_import, severity in self.DANGEROUS_IMPORTS.items():
                        if danger_import in full_name:
                            issues.append(
                                SecurityIssue(
                                    severity=severity,
                                    message=f"Dangerous import: {full_name}",
                                    line_number=(
                                        node.lineno if hasattr(node, "lineno") else None
                                    ),
                                )
                            )

        return issues

    def _check_function_calls(self, tree: ast.AST) -> list[SecurityIssue]:
        """Check for dangerous function calls."""
        issues: list[SecurityIssue] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = None
                full_name = None

                # Get function name
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    full_name = func_name
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                    # Get full qualified name (e.g., os.system, subprocess.call)
                    if isinstance(node.func.value, ast.Name):
                        full_name = f"{node.func.value.id}.{func_name}"
                    else:
                        full_name = func_name

                if func_name:
                    # Check against dangerous functions by simple name
                    for danger_func, severity in self.DANGEROUS_FUNCTIONS.items():
                        if func_name == danger_func:
                            issues.append(
                                SecurityIssue(
                                    severity=severity,
                                    message=f"Dangerous function call: {func_name}()",
                                    line_number=(
                                        node.lineno if hasattr(node, "lineno") else None
                                    ),
                                )
                            )

                # Check against dangerous imports by full name
                if full_name:
                    for danger_import, severity in self.DANGEROUS_IMPORTS.items():
                        if danger_import in full_name:
                            issues.append(
                                SecurityIssue(
                                    severity=severity,
                                    message=f"Dangerous function call: {full_name}()",
                                    line_number=(
                                        node.lineno if hasattr(node, "lineno") else None
                                    ),
                                )
                            )

        return issues

    def _check_file_operations(self, tree: ast.AST) -> list[SecurityIssue]:
        """Check for potentially dangerous file operations."""
        issues: list[SecurityIssue] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check for open() calls with write modes
                if isinstance(node.func, ast.Name) and node.func.id == "open":
                    if len(node.args) >= 2:
                        mode_arg = node.args[1]
                        if isinstance(mode_arg, ast.Constant):
                            mode = mode_arg.value
                            if isinstance(mode, str) and ("w" in mode or "a" in mode):
                                issues.append(
                                    SecurityIssue(
                                        severity="low",
                                        message=f"File write operation detected: open(..., '{mode}')",
                                        line_number=(
                                            node.lineno
                                            if hasattr(node, "lineno")
                                            else None
                                        ),
                                    )
                                )

        return issues

    def _check_credentials(self, code: str) -> list[SecurityIssue]:
        """Check for hardcoded credentials using regex patterns."""
        issues: list[SecurityIssue] = []

        for pattern, message in self.CREDENTIAL_PATTERNS:
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                # Calculate line number
                line_number = code[: match.start()].count("\n") + 1

                issues.append(
                    SecurityIssue(
                        severity="high",
                        message=message,
                        line_number=line_number,
                        code_snippet=match.group(0),
                    )
                )

        return issues

    def _check_sensitive_paths(self, code: str) -> list[SecurityIssue]:
        """Check for access to sensitive file system paths."""
        issues: list[SecurityIssue] = []

        for path_pattern in self.SENSITIVE_PATHS:
            if re.search(path_pattern, code):
                issues.append(
                    SecurityIssue(
                        severity="medium",
                        message=f"Access to sensitive path detected: {path_pattern}",
                    )
                )

        return issues

    def _calculate_quality_score(self, issues: list[SecurityIssue]) -> float:
        """
        Calculate quality score based on security issues.

        Score starts at 1.0 and decreases based on issue severity:
        - Critical: -0.3
        - High: -0.2
        - Medium: -0.1
        - Low: -0.05
        """
        score = 1.0

        severity_penalties = {
            "critical": 0.3,
            "high": 0.2,
            "medium": 0.1,
            "low": 0.05,
        }

        for issue in issues:
            penalty = severity_penalties.get(issue.severity, 0.0)
            score -= penalty

        # Clamp to 0.0-1.0
        return max(0.0, min(1.0, score))

    def generate_report(self, report: ValidationReport) -> str:
        """
        Generate human-readable security report.

        Args:
            report: ValidationReport to format

        Returns:
            Formatted report string
        """
        lines = ["=" * 60, "Security Validation Report", "=" * 60, ""]

        # Overall status
        status = "✅ SAFE" if report.is_safe else "⚠️  UNSAFE"
        lines.append(f"Status: {status}")
        lines.append(f"Quality Score: {report.quality_score:.2f}/1.0")
        lines.append(f"Needs Review: {'Yes' if report.needs_review else 'No'}")
        lines.append("")

        # Warnings
        if report.warnings:
            lines.append("Warnings:")
            for warning in report.warnings:
                lines.append(f"  - {warning}")
            lines.append("")

        # Security issues
        if report.dangerous_patterns:
            lines.append(f"Security Issues ({len(report.dangerous_patterns)}):")
            for issue in sorted(
                report.dangerous_patterns,
                key=lambda x: ["low", "medium", "high", "critical"].index(x.severity),
            ):
                severity_icon = {
                    "low": "[i]",
                    "medium": "[!]",
                    "high": "[X]",
                    "critical": "[!!]",
                }.get(issue.severity, "[?]")

                line_info = f" (line {issue.line_number})" if issue.line_number else ""
                lines.append(
                    f"  {severity_icon} [{issue.severity.upper()}] {issue.message}{line_info}"
                )

                if issue.code_snippet:
                    lines.append(f"    Code: {issue.code_snippet}")

            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)
