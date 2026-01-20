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
from dataclasses import dataclass, field
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
class CompiledPatterns:
    """Pre-compiled regex patterns for security validation.

    Patterns are compiled once per contract/file validation, not per method.
    This significantly improves performance for large codebases by avoiding
    repeated pattern extraction from contract rules and regex compilation.

    Attributes:
        admin_patterns: Compiled patterns for admin_method_public rule.
        decrypt_patterns: Compiled patterns for decrypt_method_public rule.
        sensitive_patterns: Compiled patterns for sensitive_method_exposed rule.
        sensitive_params: Lowercase sensitive parameter names (exact match, no regex).
    """

    admin_patterns: tuple[re.Pattern[str], ...] = field(default_factory=tuple)
    decrypt_patterns: tuple[re.Pattern[str], ...] = field(default_factory=tuple)
    sensitive_patterns: tuple[re.Pattern[str], ...] = field(default_factory=tuple)
    sensitive_params: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class MethodContext:
    """Context for method validation - groups related parameters."""

    method_name: str
    class_name: str
    file_path: Path
    line_number: int
    contract: ModelValidatorSubcontract
    compiled_patterns: CompiledPatterns


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

    def _get_rule_patterns(
        self,
        rule_id: str,
        param_key: str,
        contract: ModelValidatorSubcontract,
    ) -> list[str]:
        """Extract patterns from a rule's parameters in the contract.

        Searches the contract's rules for the specified rule_id and extracts
        the list of patterns from the specified parameter key.

        Args:
            rule_id: The rule identifier to look up (e.g., "sensitive_method_exposed").
            param_key: The parameter key containing the patterns (e.g., "patterns").
            contract: Validator contract with rule configurations.

        Returns:
            List of pattern strings from the rule's parameters. Returns an empty
            list if the rule is not found, has no parameters, or the param_key
            is missing/invalid (fail-safe behavior).
        """
        for rule in contract.rules:
            if rule.rule_id == rule_id:
                if rule.parameters is None:
                    logger.debug(
                        "Rule %s has no parameters, returning empty list",
                        rule_id,
                    )
                    return []
                patterns = rule.parameters.get(param_key)
                if patterns is None:
                    logger.debug(
                        "Rule %s missing param_key %s, returning empty list",
                        rule_id,
                        param_key,
                    )
                    return []
                if isinstance(patterns, list):
                    # Ensure all items are strings
                    return [str(p) for p in patterns]
                logger.warning(
                    "Rule %s param_key %s is not a list: %s",
                    rule_id,
                    param_key,
                    type(patterns).__name__,
                )
                return []
        logger.debug("Rule %s not found in contract, returning empty list", rule_id)
        return []

    def _compile_patterns(
        self,
        contract: ModelValidatorSubcontract,
    ) -> CompiledPatterns:
        """Compile and cache all regex patterns from contract rules.

        This method extracts patterns from the contract once per file validation,
        compiles them into re.Pattern objects, and returns a frozen dataclass.
        This avoids repeated pattern extraction and compilation for every method.

        Args:
            contract: Validator contract with rule configurations.

        Returns:
            CompiledPatterns instance with pre-compiled patterns.
        """

        def compile_pattern_list(patterns: list[str]) -> tuple[re.Pattern[str], ...]:
            """Compile a list of pattern strings into regex Pattern objects."""
            compiled: list[re.Pattern[str]] = []
            for pattern in patterns:
                try:
                    compiled.append(re.compile(pattern))
                except re.error as e:
                    logger.warning(
                        "Invalid regex pattern '%s': %s - skipping",
                        pattern,
                        e,
                    )
            return tuple(compiled)

        # Extract and compile patterns for each rule
        admin_patterns = compile_pattern_list(
            self._get_rule_patterns("admin_method_public", "patterns", contract)
        )
        decrypt_patterns = compile_pattern_list(
            self._get_rule_patterns("decrypt_method_public", "patterns", contract)
        )
        sensitive_patterns = compile_pattern_list(
            self._get_rule_patterns("sensitive_method_exposed", "patterns", contract)
        )

        # Extract sensitive parameter names (exact match, no regex compilation needed)
        sensitive_params_list = self._get_rule_patterns(
            "credential_in_signature", "sensitive_params", contract
        )
        sensitive_params = frozenset(p.lower() for p in sensitive_params_list)

        return CompiledPatterns(
            admin_patterns=admin_patterns,
            decrypt_patterns=decrypt_patterns,
            sensitive_patterns=sensitive_patterns,
            sensitive_params=sensitive_params,
        )

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

        # Compile patterns once for the entire file validation
        # This avoids repeated pattern extraction and regex compilation per method
        compiled_patterns = self._compile_patterns(contract)

        # Visit all class definitions in the file
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_issues = self._check_class_methods(
                    node, path, contract, compiled_patterns
                )
                issues.extend(class_issues)

        return tuple(issues)

    def _check_class_methods(
        self,
        class_node: ast.ClassDef,
        file_path: Path,
        contract: ModelValidatorSubcontract,
        compiled_patterns: CompiledPatterns,
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
                    compiled_patterns=compiled_patterns,
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
        """Check if method name matches sensitive patterns.

        Uses pre-compiled patterns from ctx.compiled_patterns for efficiency.
        Patterns are originally read from contract.rules[].parameters.
        """
        issues: list[ModelValidationIssue] = []
        method_lower = ctx.method_name.lower()

        # Check admin/internal patterns (rule: admin_method_public)
        # Uses pre-compiled patterns from ctx.compiled_patterns
        for pattern in ctx.compiled_patterns.admin_patterns:
            if pattern.match(method_lower):
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
        # Uses pre-compiled patterns from ctx.compiled_patterns
        for pattern in ctx.compiled_patterns.decrypt_patterns:
            if pattern.match(method_lower):
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
        # Uses pre-compiled patterns from ctx.compiled_patterns
        for pattern in ctx.compiled_patterns.sensitive_patterns:
            if pattern.match(method_lower):
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
        """Check method signature for sensitive parameter names.

        Uses pre-compiled sensitive_params frozenset from ctx.compiled_patterns.
        Patterns are originally read from contract.rules[].parameters.
        """
        issues: list[ModelValidationIssue] = []

        # Get rule configuration
        enabled, severity = self._get_rule_config(
            "credential_in_signature", ctx.contract
        )
        if not enabled:
            return issues

        # Use pre-compiled sensitive params (already lowercase frozenset)
        sensitive_params = ctx.compiled_patterns.sensitive_params

        # If no patterns defined in contract, return early (fail-safe)
        if not sensitive_params:
            return issues

        # Extract parameter names from AST
        found_sensitive: list[str] = []
        for arg in method_node.args.args:
            arg_name_lower = arg.arg.lower()
            if arg_name_lower in sensitive_params:
                found_sensitive.append(arg.arg)

        # Also check keyword-only args (parameters after * in signature)
        for arg in method_node.args.kwonlyargs:
            arg_name_lower = arg.arg.lower()
            if arg_name_lower in sensitive_params:
                found_sensitive.append(arg.arg)

        # Also check positional-only args (Python 3.8+, parameters before / in signature)
        # Example: def authenticate(username, /, password): ...
        # Here 'username' is positional-only and would be in posonlyargs
        for arg in method_node.args.posonlyargs:
            arg_name_lower = arg.arg.lower()
            if arg_name_lower in sensitive_params:
                found_sensitive.append(arg.arg)

        # Check vararg (*args parameter)
        # Example: def process(*passwords): ... would expose sensitive vararg name
        if method_node.args.vararg:
            vararg_name_lower = method_node.args.vararg.arg.lower()
            if vararg_name_lower in sensitive_params:
                found_sensitive.append(method_node.args.vararg.arg)

        # Check kwarg (**kwargs parameter)
        # Example: def process(**secrets): ... would expose sensitive kwarg name
        if method_node.args.kwarg:
            kwarg_name_lower = method_node.args.kwarg.arg.lower()
            if kwarg_name_lower in sensitive_params:
                found_sensitive.append(method_node.args.kwarg.arg)

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
