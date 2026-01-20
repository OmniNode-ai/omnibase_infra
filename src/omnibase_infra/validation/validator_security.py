# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Security Validator for ONEX Infrastructure.

Contract-driven AST validator for detecting security concerns in Python code.
Part of OMN-1277: Refactor validators to be Handler and contract-driven.

Security Validation Scope:
    - Public methods with sensitive names (get_password, get_secret, etc.)
    - Method signatures containing sensitive parameter names
    - Admin/internal methods exposed without underscore prefix
    - Decrypt operations exposed publicly

Usage:
    >>> from pathlib import Path
    >>> from omnibase_infra.validation.validator_security import ValidatorSecurity
    >>>
    >>> validator = ValidatorSecurity()
    >>> result = validator.validate(Path("src/"))
    >>> if not result.is_valid:
    ...     for issue in result.issues:
    ...         print(f"{issue.file_path}:{issue.line_number}: {issue.message}")

CLI Usage:
    python -m omnibase_infra.validation.validator_security src/

See Also:
    - docs/patterns/security_patterns.md - Comprehensive security guide
    - ValidatorBase - Base class for contract-driven validators
"""

from __future__ import annotations

import ast
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from omnibase_core.models.common.model_validation_issue import ModelValidationIssue
from omnibase_core.models.contracts.subcontracts.model_validator_subcontract import (
    ModelValidatorSubcontract,
)
from omnibase_core.validation.validator_base import ValidatorBase

# Configure logger for this module
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MethodContext:
    """Context for method validation - groups related parameters."""

    method_name: str
    class_name: str
    file_path: Path
    line_number: int
    contract: ModelValidatorSubcontract


class ValidatorSecurity(ValidatorBase):
    """Contract-driven security validator for Python source files.

    This validator uses AST analysis to detect security concerns in Python code:
    - Public methods with sensitive names (get_password, get_secret, etc.)
    - Method signatures containing sensitive parameter names
    - Admin/internal methods exposed without underscore prefix
    - Decrypt operations exposed publicly

    The validator is contract-driven via security.validation.yaml, supporting:
    - Configurable rules with enable/disable per rule
    - Per-rule severity overrides
    - Suppression comments for intentional exceptions
    - Glob-based file targeting and exclusion

    Thread Safety:
        ValidatorSecurity instances are NOT thread-safe due to internal mutable
        state inherited from ValidatorBase. When using parallel execution
        (e.g., pytest-xdist), create separate validator instances per worker.

    Attributes:
        validator_id: Unique identifier for this validator ("security").

    Usage Example:
        >>> from pathlib import Path
        >>> from omnibase_infra.validation.validator_security import ValidatorSecurity
        >>> validator = ValidatorSecurity()
        >>> result = validator.validate(Path("src/"))
        >>> print(f"Valid: {result.is_valid}, Issues: {len(result.issues)}")

    CLI Usage:
        python -m omnibase_infra.validation.validator_security src/
    """

    # ONEX_EXCLUDE: string_id - human-readable validator identifier
    validator_id: ClassVar[str] = "security"

    def _validate_file(
        self,
        path: Path,
        contract: ModelValidatorSubcontract,
    ) -> tuple[ModelValidationIssue, ...]:
        """Validate a single Python file for security violations.

        Uses AST analysis to detect:
        - Sensitive method names in class definitions
        - Sensitive parameter names in method signatures

        Args:
            path: Path to the Python file to validate.
            contract: Validator contract with configuration.

        Returns:
            Tuple of ModelValidationIssue instances for violations found.
        """
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as e:
            # fallback-ok: log warning and skip file on read errors
            logger.warning("Cannot read file %s: %s", path, e)
            return ()

        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            # fallback-ok: log warning and skip file with syntax errors
            logger.warning(
                "Skipping file with syntax error: path=%s, line=%s, error=%s",
                path,
                e.lineno,
                e.msg,
            )
            return ()

        issues: list[ModelValidationIssue] = []

        # Visit all class definitions in the file
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_issues = self._check_class_methods(node, path, contract)
                issues.extend(class_issues)

        return tuple(issues)

    def _check_class_methods(
        self,
        class_node: ast.ClassDef,
        file_path: Path,
        contract: ModelValidatorSubcontract,
    ) -> list[ModelValidationIssue]:
        """Check class methods for security violations."""
        issues: list[ModelValidationIssue] = []

        for node in class_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_name = node.name

                # Skip private/protected methods (already safe)
                if method_name.startswith("_"):
                    continue

                # Skip dunder methods
                if method_name.startswith("__") and method_name.endswith("__"):
                    continue

                ctx = MethodContext(
                    method_name=method_name,
                    class_name=class_node.name,
                    file_path=file_path,
                    line_number=node.lineno,
                    contract=contract,
                )
                method_issues = self._check_method(node, ctx)
                issues.extend(method_issues)

        return issues

    def _check_method(
        self,
        method_node: ast.FunctionDef | ast.AsyncFunctionDef,
        ctx: MethodContext,
    ) -> list[ModelValidationIssue]:
        """Check a single method for security violations."""
        issues: list[ModelValidationIssue] = []

        # Check for sensitive method names
        method_issues = self._check_sensitive_method_name(ctx)
        issues.extend(method_issues)

        # Check for sensitive parameters in signature
        param_issues = self._check_sensitive_parameters(method_node, ctx)
        issues.extend(param_issues)

        return issues

    def _check_sensitive_method_name(
        self,
        ctx: MethodContext,
    ) -> list[ModelValidationIssue]:
        """Check if method name matches sensitive patterns."""
        issues: list[ModelValidationIssue] = []
        method_lower = ctx.method_name.lower()

        # Check admin/internal patterns (rule: admin_method_public)
        admin_patterns = [r"^admin_", r"^internal_"]
        for pattern in admin_patterns:
            if re.match(pattern, method_lower):
                enabled, severity = self._get_rule_config(
                    "admin_method_public", ctx.contract
                )
                if enabled:
                    issues.append(
                        ModelValidationIssue(
                            severity=severity,
                            message=f"Class '{ctx.class_name}' exposes admin/internal method '{ctx.method_name}'",
                            code="admin_method_public",
                            file_path=ctx.file_path,
                            line_number=ctx.line_number,
                            rule_name="admin_method_public",
                            suggestion=f"Prefix method with underscore: '_{ctx.method_name}' or move to separate admin module",
                            context={
                                "class_name": ctx.class_name,
                                "method_name": ctx.method_name,
                                "violation_type": "admin_method_public",
                            },
                        )
                    )
                return issues  # Don't double-report

        # Check decrypt patterns (rule: decrypt_method_public)
        if re.match(r"^decrypt_", method_lower):
            enabled, severity = self._get_rule_config(
                "decrypt_method_public", ctx.contract
            )
            if enabled:
                issues.append(
                    ModelValidationIssue(
                        severity=severity,
                        message=f"Class '{ctx.class_name}' exposes decrypt method '{ctx.method_name}'",
                        code="decrypt_method_public",
                        file_path=ctx.file_path,
                        line_number=ctx.line_number,
                        rule_name="decrypt_method_public",
                        suggestion=f"Prefix method with underscore: '_{ctx.method_name}' to exclude from public API",
                        context={
                            "class_name": ctx.class_name,
                            "method_name": ctx.method_name,
                            "violation_type": "decrypt_method_public",
                        },
                    )
                )
            return issues  # Don't double-report

        # Check other sensitive patterns (rule: sensitive_method_exposed)
        sensitive_patterns = [
            r"^get_password$",
            r"^get_secret$",
            r"^get_token$",
            r"^get_api_key$",
            r"^get_credential",
            r"^fetch_password$",
            r"^fetch_secret$",
            r"^fetch_token$",
            r"^validate_password$",
            r"^check_password$",
            r"^verify_password$",
        ]

        for pattern in sensitive_patterns:
            if re.match(pattern, method_lower):
                enabled, severity = self._get_rule_config(
                    "sensitive_method_exposed", ctx.contract
                )
                if enabled:
                    issues.append(
                        ModelValidationIssue(
                            severity=severity,
                            message=f"Class '{ctx.class_name}' exposes sensitive method '{ctx.method_name}'",
                            code="sensitive_method_exposed",
                            file_path=ctx.file_path,
                            line_number=ctx.line_number,
                            rule_name="sensitive_method_exposed",
                            suggestion=f"Prefix method with underscore: '_{ctx.method_name}' to exclude from introspection",
                            context={
                                "class_name": ctx.class_name,
                                "method_name": ctx.method_name,
                                "violation_type": "sensitive_method_exposed",
                            },
                        )
                    )
                break

        return issues

    def _check_sensitive_parameters(
        self,
        method_node: ast.FunctionDef | ast.AsyncFunctionDef,
        ctx: MethodContext,
    ) -> list[ModelValidationIssue]:
        """Check method signature for sensitive parameter names."""
        issues: list[ModelValidationIssue] = []

        # Get rule configuration
        enabled, severity = self._get_rule_config(
            "credential_in_signature", ctx.contract
        )
        if not enabled:
            return issues

        # Sensitive parameter names to check
        sensitive_params = {
            "password",
            "secret",
            "token",
            "api_key",
            "apikey",
            "access_key",
            "private_key",
            "credential",
            "auth_token",
            "bearer_token",
            "decrypt_key",
            "encryption_key",
        }

        # Extract parameter names from AST
        found_sensitive: list[str] = []
        for arg in method_node.args.args:
            arg_name_lower = arg.arg.lower()
            if arg_name_lower in sensitive_params:
                found_sensitive.append(arg.arg)

        # Also check keyword-only args
        for arg in method_node.args.kwonlyargs:
            arg_name_lower = arg.arg.lower()
            if arg_name_lower in sensitive_params:
                found_sensitive.append(arg.arg)

        if found_sensitive:
            issues.append(
                ModelValidationIssue(
                    severity=severity,
                    message=f"Method '{ctx.class_name}.{ctx.method_name}' has sensitive parameters: {', '.join(found_sensitive)}",
                    code="credential_in_signature",
                    file_path=ctx.file_path,
                    line_number=ctx.line_number,
                    rule_name="credential_in_signature",
                    suggestion=f"Use generic parameter names (e.g., 'data' instead of '{found_sensitive[0]}') or make method private",
                    context={
                        "class_name": ctx.class_name,
                        "method_name": ctx.method_name,
                        "sensitive_parameters": ", ".join(found_sensitive),
                        "violation_type": "credential_in_signature",
                    },
                )
            )

        return issues


# CLI entry point
if __name__ == "__main__":
    sys.exit(ValidatorSecurity.main())


__all__ = ["ValidatorSecurity"]
