#!/usr/bin/env python3
"""
NodeCodegenCodeValidatorEffect - Validate generated code for compliance and quality.

ONEX v2.0 Compliance:
- Suffix-based naming: NodeCodegenCodeValidatorEffect
- Extends NodeEffect from omnibase_core
- Uses ModelOnexError for error handling
- Structured logging with correlation tracking
"""

import ast
import os
import re
import time
from pathlib import Path
from typing import Any, ClassVar

# ONEX Core Imports
from omnibase_core import EnumCoreErrorCode, ModelOnexError
from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.core import ModelContainer
from omnibase_core.nodes.node_effect import NodeEffect

# Node-specific imports
from .models import (
    EnumValidationRule,
    ModelCodeValidationResult,
    ModelValidationError,
    ModelValidationWarning,
)

# Aliases
OnexError = ModelOnexError
CoreErrorCode = EnumCoreErrorCode


class NodeCodegenCodeValidatorEffect(NodeEffect):
    """
    Code Validator Effect for validating generated code quality and compliance.

    Validates generated code for syntax correctness, ONEX v2.0 compliance,
    type hints, security issues, and other quality metrics.

    Responsibilities:
    - Validate Python syntax (AST parsing)
    - Check ONEX v2.0 compliance (ModelOnexError usage, emit_log_event, etc.)
    - Verify type hints on methods and parameters
    - Scan for security issues (hardcoded secrets, SQL injection patterns)
    - Check naming conventions and import patterns
    - Return structured validation results with errors and warnings

    ONEX v2.0 Compliance:
    - Suffix-based naming: NodeCodegenCodeValidatorEffect
    - Extends NodeEffect from omnibase_core
    - Uses ModelOnexError for error handling
    - Structured logging with correlation tracking

    Performance Targets:
    - Validation time: <500ms for typical node file
    - Throughput: >50 files/second
    - Memory: <100MB per file

    Example Usage:
        ```python
        container = ModelContainer(...)
        validator = NodeCodegenCodeValidatorEffect(container)

        contract = ModelContractEffect(
            correlation_id=uuid4(),
            input_state={
                "generated_code": "def foo(): pass",
                "validation_rules": [EnumValidationRule.SYNTAX, EnumValidationRule.ONEX_COMPLIANCE]
            }
        )

        result = await validator.execute_effect(contract)
        if result.is_valid:
            print("Code is valid!")
        else:
            print(f"Found {len(result.validation_errors)} errors")
        ```
    """

    # Security patterns to check
    SECURITY_PATTERNS: ClassVar[dict[str, str]] = {
        "hardcoded_password": r"password\s*=\s*['\"][\w\d]+['\"]",
        "hardcoded_api_key": r"api[_-]?key\s*=\s*['\"][\w\d\-]+['\"]",
        "sql_injection": r"(execute|cursor\.execute)\(['\"].*%s.*['\"].*\)",
        "command_injection": r"(os\.system|subprocess\.call|exec|eval)\(",
    }

    # ONEX required patterns
    ONEX_PATTERNS: ClassVar[dict[str, str]] = {
        "modelonexerror_import": r"from omnibase_core import.*ModelOnexError",
        "emit_log_event_import": r"from omnibase_core\.logging\.structured import.*emit_log_event",
        "modelonexerror_usage": r"raise ModelOnexError\(",
    }

    def __init__(self, container: ModelContainer) -> None:
        """
        Initialize code validator with dependency injection container.

        Args:
            container: ONEX container for dependency injection

        Raises:
            ModelOnexError: If container is invalid or initialization fails
        """
        super().__init__(container)

        # Configuration - defensive pattern
        try:
            if hasattr(container.config, "get") and callable(container.config.get):
                self.strict_mode = container.config.get("validation_strict_mode", False)
                # Consul configuration
                self.consul_host = os.getenv("CONSUL_HOST", "omninode-bridge-consul")
                self.consul_port = int(os.getenv("CONSUL_PORT", "28500"))
                self.consul_enable_registration = True
                self.enable_security_checks = container.config.get(
                    "enable_security_checks", True
                )
            else:
                self.strict_mode = False
                self.enable_security_checks = True
        except Exception:
            self.strict_mode = False
            self.enable_security_checks = True

        # Metrics tracking
        self._total_validations = 0
        self._total_errors_found = 0
        self._total_warnings_found = 0
        self._total_duration_ms = 0.0
        self._failed_validations = 0

        emit_log_event(
            LogLevel.INFO,
            "NodeCodegenCodeValidatorEffect initialized successfully",
            {
                "node_id": self.node_id,
                "strict_mode": self.strict_mode,
                "security_checks_enabled": self.enable_security_checks,
            },
        )

    async def execute_effect(
        self, contract: ModelContractEffect
    ) -> ModelCodeValidationResult:
        """
        Execute code validation.

        Args:
            contract: Effect contract with input_data containing:
                - generated_code (str): Python source code to validate
                - validation_rules (list[str], optional): Rules to check
                - strict_mode (bool, optional): Fail on warnings if True
                - file_path (str, optional): Path to file (for metadata)

        Returns:
            ModelCodeValidationResult with validation errors, warnings, and metadata

        Raises:
            OnexError: If validation fails or invalid input
        """
        start_time = time.perf_counter()
        correlation_id = contract.correlation_id

        emit_log_event(
            LogLevel.INFO,
            "Starting code validation",
            {
                "node_id": str(self.node_id),
                "correlation_id": str(correlation_id),
            },
        )

        try:
            # Parse input from contract
            input_data = contract.input_state or {}
            generated_code = input_data.get("generated_code")

            if not generated_code:
                raise OnexError(
                    error_code=CoreErrorCode.VALIDATION_ERROR,
                    message="Missing required field: generated_code",
                    details={"correlation_id": str(correlation_id)},
                )

            validation_rules = input_data.get(
                "validation_rules", [EnumValidationRule.ALL]
            )
            strict_mode = input_data.get("strict_mode", self.strict_mode)
            file_path = input_data.get("file_path", "<unknown>")

            # Convert string rules to enum
            rules = self._parse_validation_rules(validation_rules)

            # Initialize results
            errors: list[ModelValidationError] = []
            warnings: list[ModelValidationWarning] = []
            syntax_valid = True
            onex_compliant = True
            security_issues = 0

            # Run validations
            if EnumValidationRule.ALL in rules or EnumValidationRule.SYNTAX in rules:
                syntax_errors = self._validate_syntax(generated_code)
                errors.extend(syntax_errors)
                syntax_valid = len(syntax_errors) == 0

            if (
                EnumValidationRule.ALL in rules
                or EnumValidationRule.ONEX_COMPLIANCE in rules
            ):
                onex_errors, onex_warnings = self._validate_onex_compliance(
                    generated_code
                )
                errors.extend(onex_errors)
                warnings.extend(onex_warnings)
                onex_compliant = len(onex_errors) == 0

            if (
                EnumValidationRule.ALL in rules
                or EnumValidationRule.TYPE_HINTS in rules
            ):
                type_warnings = self._validate_type_hints(generated_code)
                warnings.extend(type_warnings)

            if EnumValidationRule.ALL in rules or EnumValidationRule.SECURITY in rules:
                if self.enable_security_checks:
                    security_errors = self._validate_security(generated_code)
                    errors.extend(security_errors)
                    security_issues = len(security_errors)

            if EnumValidationRule.ALL in rules or EnumValidationRule.IMPORTS in rules:
                import_warnings = self._validate_imports(generated_code)
                warnings.extend(import_warnings)

            # Determine if valid
            is_valid = len(errors) == 0
            if strict_mode:
                is_valid = is_valid and len(warnings) == 0

            # Calculate metrics
            file_lines = len(generated_code.splitlines())
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Update metrics
            self._total_validations += 1
            self._total_errors_found += len(errors)
            self._total_warnings_found += len(warnings)
            self._total_duration_ms += duration_ms

            emit_log_event(
                LogLevel.INFO,
                "Code validation completed",
                {
                    "node_id": str(self.node_id),
                    "correlation_id": str(correlation_id),
                    "is_valid": is_valid,
                    "errors_found": len(errors),
                    "warnings_found": len(warnings),
                    "duration_ms": round(duration_ms, 2),
                },
            )

            return ModelCodeValidationResult(
                is_valid=is_valid,
                validation_errors=errors,
                validation_warnings=warnings,
                validation_time_ms=duration_ms,
                rules_checked=[r.value for r in rules],
                file_path=file_path,
                file_lines=file_lines,
                syntax_valid=syntax_valid,
                onex_compliant=onex_compliant,
                security_issues_found=security_issues,
            )

        except OnexError:
            self._failed_validations += 1
            raise

        except Exception as e:
            self._failed_validations += 1

            emit_log_event(
                LogLevel.ERROR,
                f"Code validation failed: {e}",
                {
                    "node_id": str(self.node_id),
                    "correlation_id": str(correlation_id),
                    "error": str(e),
                },
            )

            raise OnexError(
                message=f"Code validation failed: {e}",
                error_code=CoreErrorCode.OPERATION_FAILED,
                node_id=str(self.node_id),
                correlation_id=str(correlation_id),
                error=str(e),
            ) from e

    def _parse_validation_rules(
        self, rules: list[str | EnumValidationRule]
    ) -> list[EnumValidationRule]:
        """Parse validation rules from strings or enums."""
        parsed_rules = []
        for rule in rules:
            if isinstance(rule, EnumValidationRule):
                parsed_rules.append(rule)
            elif isinstance(rule, str):
                try:
                    parsed_rules.append(EnumValidationRule(rule))
                except ValueError:
                    # Skip invalid rules
                    continue
        return parsed_rules

    def _validate_syntax(self, code: str) -> list[ModelValidationError]:
        """Validate Python syntax using AST parsing."""
        import textwrap

        errors = []
        try:
            # Dedent code to handle code snippets with leading indentation
            # This is important for validating code that will be injected into methods
            dedented_code = textwrap.dedent(code)
            ast.parse(dedented_code)
        except SyntaxError as e:
            errors.append(
                ModelValidationError(
                    rule="syntax",
                    message=f"Syntax error: {e.msg}",
                    line_number=e.lineno,
                    column=e.offset,
                    code_snippet=e.text.strip() if e.text else None,
                    severity="error",
                    suggested_fix="Fix syntax error before proceeding",
                )
            )
        return errors

    def _validate_onex_compliance(
        self, code: str
    ) -> tuple[list[ModelValidationError], list[ModelValidationWarning]]:
        """Validate ONEX v2.0 compliance patterns."""
        errors = []
        warnings = []

        # Check for ModelOnexError import
        if not re.search(self.ONEX_PATTERNS["modelonexerror_import"], code):
            warnings.append(
                ModelValidationWarning(
                    rule="onex_compliance",
                    message="Missing ModelOnexError import from omnibase_core",
                    severity="warning",
                    suggested_fix="Add: from omnibase_core import ModelOnexError",
                )
            )

        # Check for emit_log_event import
        if not re.search(self.ONEX_PATTERNS["emit_log_event_import"], code):
            warnings.append(
                ModelValidationWarning(
                    rule="onex_compliance",
                    message="Missing emit_log_event import from omnibase_core.logging.structured",
                    severity="warning",
                    suggested_fix="Add: from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event",
                )
            )

        # Check for proper error handling (at least one ModelOnexError raise)
        if "def " in code and not re.search(
            self.ONEX_PATTERNS["modelonexerror_usage"], code
        ):
            warnings.append(
                ModelValidationWarning(
                    rule="onex_compliance",
                    message="No ModelOnexError usage found - should use for error handling",
                    severity="warning",
                    suggested_fix="Wrap exceptions with ModelOnexError",
                )
            )

        return errors, warnings

    def _validate_type_hints(self, code: str) -> list[ModelValidationWarning]:
        """Validate type hints on methods and parameters."""
        warnings = []

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Check parameters for type hints
                    for arg in node.args.args:
                        if arg.annotation is None and arg.arg != "self":
                            warnings.append(
                                ModelValidationWarning(
                                    rule="type_hints",
                                    message=f"Missing type hint on parameter '{arg.arg}' in function '{node.name}'",
                                    line_number=node.lineno,
                                    severity="warning",
                                    suggested_fix=f"Add type hint to parameter '{arg.arg}'",
                                )
                            )

                    # Check return type hint
                    if node.returns is None and node.name != "__init__":
                        warnings.append(
                            ModelValidationWarning(
                                rule="type_hints",
                                message=f"Missing return type hint on function '{node.name}'",
                                line_number=node.lineno,
                                severity="warning",
                                suggested_fix=f"Add return type hint to function '{node.name}'",
                            )
                        )
        except SyntaxError:
            # Skip type hint validation if syntax is invalid
            pass

        return warnings

    def _validate_security(self, code: str) -> list[ModelValidationError]:
        """Validate for common security issues."""
        errors = []

        for issue_type, pattern in self.SECURITY_PATTERNS.items():
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                line_number = code[: match.start()].count("\n") + 1
                errors.append(
                    ModelValidationError(
                        rule="security",
                        message=f"Security issue: {issue_type.replace('_', ' ')}",
                        line_number=line_number,
                        code_snippet=match.group(0),
                        severity="error",
                        suggested_fix=f"Remove {issue_type.replace('_', ' ')} - use environment variables or secure storage",
                    )
                )

        return errors

    def _validate_imports(self, code: str) -> list[ModelValidationWarning]:
        """Validate import statements."""
        warnings = []

        # Check for relative imports (should use absolute)
        if re.search(r"from \. import", code):
            warnings.append(
                ModelValidationWarning(
                    rule="imports",
                    message="Relative imports found - prefer absolute imports",
                    severity="info",
                    suggested_fix="Use absolute imports: from omninode_bridge.x import y",
                )
            )

        return warnings

    def get_metrics(self) -> dict[str, Any]:
        """
        Get validation metrics for monitoring.

        Returns:
            Dictionary with metrics
        """
        avg_duration_ms = (
            self._total_duration_ms / self._total_validations
            if self._total_validations > 0
            else 0
        )

        success_rate = (
            (self._total_validations - self._failed_validations)
            / self._total_validations
            if self._total_validations > 0
            else 1.0
        )

        return {
            "total_validations": self._total_validations,
            "total_errors_found": self._total_errors_found,
            "total_warnings_found": self._total_warnings_found,
            "failed_validations": self._failed_validations,
            "success_rate": round(success_rate, 4),
            "avg_duration_ms": round(avg_duration_ms, 2),
            "avg_errors_per_file": (
                self._total_errors_found / self._total_validations
                if self._total_validations > 0
                else 0
            ),
        }

    async def startup(self) -> None:
        """Node startup lifecycle hook."""
        emit_log_event(
            LogLevel.INFO,
            "Node startup initiated",
            {"node_name": self.__class__.__name__},
        )

    async def shutdown(self) -> None:
        """Node shutdown lifecycle hook."""
        self._deregister_from_consul()
        emit_log_event(
            LogLevel.INFO,
            "Node shutdown completed",
            {"node_name": self.__class__.__name__},
        )

    def _register_with_consul_sync(self) -> None:
        """Register node with Consul for service discovery (synchronous)."""
        try:
            import consul

            consul_client = consul.Consul(host=self.consul_host, port=self.consul_port)
            service_id = (
                f"omninode-bridge-{self.__class__.__name__.lower()}-{self.node_id}"
            )
            service_port = 8065  # Default port
            service_host = "localhost"

            consul_client.agent.service.register(
                name=f"omninode-bridge-{self.__class__.__name__.lower()}",
                service_id=service_id,
                address=service_host,
                port=service_port,
                tags=["onex", "codegen", "effect"],
                http=f"http://{service_host}:{service_port}/health",
                interval="30s",
                timeout="5s",
            )

            self._consul_service_id = service_id

            emit_log_event(
                LogLevel.INFO,
                "Registered with Consul successfully",
                {"node_id": self.node_id, "service_id": service_id},
            )

        except ImportError:
            emit_log_event(
                LogLevel.WARNING,
                "python-consul not installed - Consul registration skipped",
                {"node_id": self.node_id},
            )
        except Exception as e:
            emit_log_event(
                LogLevel.ERROR,
                "Failed to register with Consul",
                {"node_id": self.node_id, "error": str(e)},
            )

    def _deregister_from_consul(self) -> None:
        """Deregister node from Consul on shutdown (synchronous)."""
        try:
            if not hasattr(self, "_consul_service_id"):
                return

            import consul

            consul_client = consul.Consul(host=self.consul_host, port=self.consul_port)
            consul_client.agent.service.deregister(self._consul_service_id)

            emit_log_event(
                LogLevel.INFO,
                "Deregistered from Consul successfully",
                {"node_id": self.node_id, "service_id": self._consul_service_id},
            )

        except ImportError:
            pass
        except Exception as e:
            emit_log_event(
                LogLevel.WARNING,
                "Failed to deregister from Consul",
                {"node_id": self.node_id, "error": str(e)},
            )


def main() -> int:
    """
    Entry point for node execution.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        from omnibase_core.infrastructure.node_base import NodeBase

        CONTRACT_FILENAME = "contract.yaml"
        node_base = NodeBase(Path(__file__).parent / CONTRACT_FILENAME)
        return 0
    except Exception as e:
        emit_log_event(
            LogLevel.ERROR,
            f"NodeCodegenCodeValidatorEffect execution failed: {e!s}",
            {"error": str(e), "error_type": type(e).__name__},
        )
        return 1


if __name__ == "__main__":
    exit(main())
