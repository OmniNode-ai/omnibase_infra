#!/usr/bin/env python3
"""
Validation rules for LLM-generated code.

Defines rules for:
- ONEX compliance patterns
- Security checks
- Type hint requirements
- Best practices
"""


# ONEX compliance patterns
ONEX_ERROR_PATTERNS = [
    ("ModelOnexError", "Should use ModelOnexError for error handling"),
    ("EnumCoreErrorCode", "Should use EnumCoreErrorCode for error codes"),
]

ONEX_LOGGING_PATTERNS = [
    ("emit_log_event", "Should use emit_log_event for structured logging"),
    ("EnumLogLevel", "Should use EnumLogLevel for log levels"),
]

ONEX_REQUIRED_IMPORTS = [
    "omnibase_core",
]

# Security patterns to check
SECURITY_KEYWORDS = [
    "password",
    "api_key",
    "secret",
    "token",
    "credentials",
    "auth",
    "private_key",
]

DANGEROUS_PATTERNS = [
    (r"eval\(", "Use of eval() is dangerous"),
    (r"exec\(", "Use of exec() is dangerous"),
    (r"__import__\(", "Dynamic imports should be avoided"),
    (r"\.execute\(['\"].*%", "Potential SQL injection (string formatting in SQL)"),
    (r"\.execute\(.*f['\"]", "Potential SQL injection (f-string in SQL)"),
    (r"pickle\.loads?\(", "Pickle can execute arbitrary code"),
]

# Type hint requirements
TYPE_HINT_EXCEPTIONS = [
    "__init__",
    "__str__",
    "__repr__",
    "__eq__",
    "__hash__",
]

# Best practices
MAX_FUNCTION_LENGTH = 50  # lines
MAX_COMPLEXITY = 10  # cyclomatic complexity estimate
MIN_DOCSTRING_LENGTH = 20  # characters


def check_hardcoded_secrets(code: str) -> list[str]:
    """
    Check for hardcoded secrets in code.

    Args:
        code: Source code to check

    Returns:
        List of issues found
    """
    issues = []
    lines = code.split("\n")

    for line_num, line in enumerate(lines, 1):
        # Skip comments
        if line.strip().startswith("#"):
            continue

        # Check for assignment with security keywords
        if "=" in line:
            lower_line = line.lower()
            for keyword in SECURITY_KEYWORDS:
                if keyword in lower_line:
                    # Check if it's a hardcoded string value
                    if ('= "' in line or "= '" in line) and not (
                        "os.getenv" in line
                        or "os.environ" in line
                        or "config." in line
                        or ".get(" in line
                    ):
                        issues.append(
                            f"Line {line_num}: Potential hardcoded {keyword} "
                            "(use environment variables or config)"
                        )
                        break

    return issues


def check_dangerous_patterns(code: str) -> list[str]:
    """
    Check for dangerous code patterns.

    Args:
        code: Source code to check

    Returns:
        List of issues found
    """
    import re

    issues = []

    for pattern, message in DANGEROUS_PATTERNS:
        if re.search(pattern, code):
            issues.append(f"Security: {message}")

    # Additional check: f-strings + execute() in same code (SQL injection)
    # This catches cases like: query = f"..."; execute(query)
    if (('f"' in code) or ("f'" in code)) and ("execute(" in code):
        # Check if it looks like SQL (common SQL keywords)
        sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "WHERE", "FROM"]
        if any(keyword in code.upper() for keyword in sql_keywords):
            if "Security: Potential SQL injection" not in " ".join(issues):
                issues.append(
                    "Security: Potential SQL injection (f-string with SQL execute)"
                )

    return issues


def check_onex_compliance(code: str) -> tuple[bool, list[str]]:
    """
    Check for ONEX compliance patterns.

    Args:
        code: Source code to check

    Returns:
        Tuple of (is_compliant, list of issues)
    """
    issues = []

    # Check for error handling patterns
    has_exception_handling = "raise Exception" in code or "raise " in code
    if has_exception_handling and "ModelOnexError" not in code:
        issues.append("Should use ModelOnexError instead of generic exceptions")

    # Check for logging patterns
    has_logging = "logger." in code or "logging." in code or "print(" in code
    if has_logging and "emit_log_event" not in code:
        issues.append(
            "Should use emit_log_event for structured logging instead of logger/print"
        )

    # Check for required imports
    if "omnibase_core" not in code and (has_exception_handling or has_logging):
        issues.append("Missing omnibase_core import for ONEX patterns")

    return len(issues) == 0, issues


def estimate_complexity(code: str) -> int:
    """
    Estimate cyclomatic complexity of code.

    Simple heuristic: count decision points (if, for, while, except, and, or)

    Args:
        code: Source code to check

    Returns:
        Estimated complexity score
    """
    complexity = 1  # Base complexity

    # Count decision points
    decision_keywords = ["if ", "elif ", "for ", "while ", "except ", " and ", " or "]

    for keyword in decision_keywords:
        complexity += code.count(keyword)

    return complexity


__all__ = [
    "ONEX_ERROR_PATTERNS",
    "ONEX_LOGGING_PATTERNS",
    "ONEX_REQUIRED_IMPORTS",
    "SECURITY_KEYWORDS",
    "DANGEROUS_PATTERNS",
    "TYPE_HINT_EXCEPTIONS",
    "MAX_FUNCTION_LENGTH",
    "MAX_COMPLEXITY",
    "MIN_DOCSTRING_LENGTH",
    "check_hardcoded_secrets",
    "check_dangerous_patterns",
    "check_onex_compliance",
    "estimate_complexity",
]
