#!/usr/bin/env python3
"""
NodeCodegenCodeInjectorEffect - Inject validated code back into node files.

ONEX v2.0 Compliance:
- Suffix-based naming: NodeCodegenCodeInjectorEffect
- Extends NodeEffect from omnibase_core
- Uses ModelOnexError for error handling
- Structured logging with correlation tracking
"""

import ast
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

# ONEX Core Imports
from omnibase_core import EnumCoreErrorCode, ModelOnexError
from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.core import ModelContainer
from omnibase_core.nodes.node_effect import NodeEffect

# Node-specific imports
from .models import (
    ModelCodeInjectionRequest,
    ModelCodeInjectionResult,
    ModelInjectionError,
)

# Aliases
OnexError = ModelOnexError
CoreErrorCode = EnumCoreErrorCode


class NodeCodegenCodeInjectorEffect(NodeEffect):
    """
    Code Injector Effect for injecting validated code back into node files.

    Replaces stub method implementations with actual validated code while
    preserving method signatures, docstrings, decorators, and indentation.

    Responsibilities:
    - Locate methods by name and line number using AST
    - Replace stub implementations with validated code
    - Preserve method signatures and docstrings
    - Maintain correct indentation and formatting
    - Handle async methods and decorators
    - Track injection metrics and errors

    ONEX v2.0 Compliance:
    - Suffix-based naming: NodeCodegenCodeInjectorEffect
    - Extends NodeEffect from omnibase_core
    - Uses ModelOnexError for error handling
    - Structured logging with correlation tracking

    Performance Targets:
    - Injection time: <200ms for typical node file
    - Throughput: >100 injections/second
    - Memory: <50MB per file

    Example Usage:
        ```python
        container = ModelContainer(...)
        injector = NodeCodegenCodeInjectorEffect(container)

        contract = ModelContractEffect(
            correlation_id=uuid4(),
            input_state={
                "source_code": "def foo(): pass",
                "injection_requests": [
                    {
                        "method_name": "foo",
                        "line_number": 1,
                        "generated_code": "    return 42"
                    }
                ]
            }
        )

        result = await injector.execute_effect(contract)
        if result.success:
            print(f"Modified {result.injections_performed} methods")
        ```
    """

    def __init__(self, container: ModelContainer) -> None:
        """
        Initialize code injector with dependency injection container.

        Args:
            container: ONEX container for dependency injection

        Raises:
            ModelOnexError: If container is invalid or initialization fails
        """
        super().__init__(container)

        # Configuration - defensive pattern
        try:
            if hasattr(container.config, "get") and callable(container.config.get):
                self.preserve_formatting = container.config.get(
                    "preserve_code_formatting", True
                )
                self.strict_line_matching = container.config.get(
                    "strict_line_matching", True
                )
                # Consul configuration for service discovery
                self.consul_host: str = container.config.get(
                    "consul_host", os.getenv("CONSUL_HOST", "omninode-bridge-consul")
                )
                self.consul_port: int = container.config.get(
                    "consul_port", int(os.getenv("CONSUL_PORT", "28500"))
                )
                self.consul_enable_registration: bool = container.config.get(
                    "consul_enable_registration", True
                )
            else:
                self.preserve_formatting = True
                self.strict_line_matching = True
                self.consul_host = os.getenv("CONSUL_HOST", "omninode-bridge-consul")
                self.consul_port = int(os.getenv("CONSUL_PORT", "28500"))
                self.consul_enable_registration = True
        except Exception:
            self.preserve_formatting = True
            self.strict_line_matching = True
            self.consul_host = os.getenv("CONSUL_HOST", "omninode-bridge-consul")
            self.consul_port = int(os.getenv("CONSUL_PORT", "28500"))
            self.consul_enable_registration = True

        # Metrics tracking
        self._total_injections = 0
        self._successful_injections = 0
        self._failed_injections = 0
        self._total_duration_ms = 0.0

        emit_log_event(
            LogLevel.INFO,
            "NodeCodegenCodeInjectorEffect initialized successfully",
            {
                "node_id": self.node_id,
                "preserve_formatting": self.preserve_formatting,
                "strict_line_matching": self.strict_line_matching,
            },
        )

        # Health check mode detection
        try:
            health_check_mode = (
                container.config.get("health_check_mode", False)
                if hasattr(container.config, "get")
                else False
            )
        except Exception:
            health_check_mode = False

        # Consul registration (skip in health check mode)
        if not health_check_mode and self.consul_enable_registration:
            self._register_with_consul_sync()

    async def execute_effect(
        self, contract: ModelContractEffect
    ) -> ModelCodeInjectionResult:
        """
        Execute code injection.

        Args:
            contract: Effect contract with input_state containing:
                - source_code (str): Original Python source code
                - injection_requests (list[dict]): List of injection requests
                - preserve_comments (bool, optional): Keep comments (default: True)

        Returns:
            ModelCodeInjectionResult with modified source and metrics

        Raises:
            OnexError: If injection fails or invalid input
        """
        start_time = time.perf_counter()
        correlation_id = contract.correlation_id

        emit_log_event(
            LogLevel.INFO,
            "Starting code injection",
            {
                "node_id": str(self.node_id),
                "correlation_id": str(correlation_id),
            },
        )

        try:
            # Parse input from contract
            input_data = contract.input_state or {}
            source_code = input_data.get("source_code")

            if not source_code:
                raise OnexError(
                    error_code=CoreErrorCode.VALIDATION_ERROR,
                    message="Missing required field: source_code",
                    details={"correlation_id": str(correlation_id)},
                )

            injection_requests_data = input_data.get("injection_requests", [])
            if not injection_requests_data:
                raise OnexError(
                    error_code=CoreErrorCode.VALIDATION_ERROR,
                    message="Missing required field: injection_requests",
                    details={"correlation_id": str(correlation_id)},
                )

            # Parse injection requests
            injection_requests = [
                ModelCodeInjectionRequest(**req) for req in injection_requests_data
            ]

            # Store original metrics
            file_lines_before = len(source_code.splitlines())

            # Perform injections
            modified_source, injection_errors, methods_modified = self._inject_code(
                source_code, injection_requests
            )

            # Calculate metrics
            file_lines_after = len(modified_source.splitlines())
            injections_performed = len(methods_modified)
            success = len(injection_errors) == 0
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Update metrics
            self._total_injections += len(injection_requests)
            self._successful_injections += injections_performed
            self._failed_injections += len(injection_errors)
            self._total_duration_ms += duration_ms

            emit_log_event(
                LogLevel.INFO,
                "Code injection completed",
                {
                    "node_id": str(self.node_id),
                    "correlation_id": str(correlation_id),
                    "success": success,
                    "injections_performed": injections_performed,
                    "errors": len(injection_errors),
                    "duration_ms": round(duration_ms, 2),
                },
            )

            return ModelCodeInjectionResult(
                success=success,
                modified_source=modified_source,
                injections_performed=injections_performed,
                injection_errors=injection_errors,
                injection_time_ms=duration_ms,
                file_lines_before=file_lines_before,
                file_lines_after=file_lines_after,
                methods_modified=methods_modified,
            )

        except OnexError:
            self._failed_injections += 1
            raise

        except Exception as e:
            self._failed_injections += 1

            emit_log_event(
                LogLevel.ERROR,
                f"Code injection failed: {e}",
                {
                    "node_id": str(self.node_id),
                    "correlation_id": str(correlation_id),
                    "error": str(e),
                },
            )

            raise OnexError(
                message=f"Code injection failed: {e}",
                error_code=CoreErrorCode.OPERATION_FAILED,
                node_id=str(self.node_id),
                correlation_id=str(correlation_id),
                error=str(e),
            ) from e

    def _inject_code(
        self, source_code: str, injection_requests: list[ModelCodeInjectionRequest]
    ) -> tuple[str, list[ModelInjectionError], list[str]]:
        """
        Inject code into source file.

        Args:
            source_code: Original Python source code
            injection_requests: List of injection requests

        Returns:
            Tuple of (modified_source, errors, methods_modified)
        """
        errors = []
        methods_modified = []

        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            errors.append(
                ModelInjectionError(
                    method_name="<unknown>",
                    error_type="syntax_error",
                    message=f"Failed to parse source code: {e.msg}",
                    line_number=e.lineno,
                    suggested_fix="Fix syntax errors in source code before injection",
                )
            )
            return source_code, errors, methods_modified

        # Build line-indexed source
        source_lines = source_code.splitlines(keepends=True)

        # Sort requests by line number (descending) to avoid offset issues
        sorted_requests = sorted(
            injection_requests, key=lambda r: r.line_number, reverse=True
        )

        # Process each injection
        for request in sorted_requests:
            # Find the method in AST
            method_node = self._find_method_at_line(
                tree, request.method_name, request.line_number
            )

            if not method_node:
                errors.append(
                    ModelInjectionError(
                        method_name=request.method_name,
                        error_type="method_not_found",
                        message=f"Method '{request.method_name}' not found at line {request.line_number}",
                        line_number=request.line_number,
                        suggested_fix="Verify method name and line number are correct",
                    )
                )
                continue

            # Perform injection
            try:
                source_lines = self._replace_method_body(
                    source_lines, method_node, request
                )
                methods_modified.append(request.method_name)
            except Exception as e:
                errors.append(
                    ModelInjectionError(
                        method_name=request.method_name,
                        error_type="injection_failed",
                        message=f"Failed to inject code: {e!s}",
                        line_number=request.line_number,
                        suggested_fix="Check code formatting and indentation",
                    )
                )

        # Reconstruct source
        modified_source = "".join(source_lines)

        return modified_source, errors, methods_modified

    def _find_method_at_line(
        self, tree: ast.AST, method_name: str, line_number: int
    ) -> Optional[ast.FunctionDef | ast.AsyncFunctionDef]:
        """
        Find method node at specific line number.

        Args:
            tree: AST tree
            method_name: Method name to find
            line_number: Expected line number (can be decorator line or def line)

        Returns:
            Method node if found, None otherwise
        """
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name != method_name:
                    continue

                # Include decorator line numbers as candidate lines
                candidate_lines = {node.lineno}
                for decorator in getattr(node, "decorator_list", []):
                    decorator_lineno = getattr(decorator, "lineno", None)
                    if decorator_lineno is not None:
                        candidate_lines.add(decorator_lineno)

                if self.strict_line_matching:
                    # Exact match required for any candidate line
                    if line_number in candidate_lines:
                        return node
                else:
                    # Allow tolerance (+/- 2 lines) for any candidate line
                    if any(abs(candidate - line_number) <= 2 for candidate in candidate_lines):
                        return node
        return None

    def _replace_method_body(
        self,
        source_lines: list[str],
        method_node: ast.FunctionDef | ast.AsyncFunctionDef,
        request: ModelCodeInjectionRequest,
    ) -> list[str]:
        """
        Replace method body with new code.

        Args:
            source_lines: Source code split into lines
            method_node: AST node for the method
            request: Injection request

        Returns:
            Modified source lines
        """
        # Determine method extent (including decorators)
        start_line = method_node.lineno - 1  # 0-indexed

        # Find decorator start if any
        if hasattr(method_node, "decorator_list") and method_node.decorator_list:
            decorator_start = min(d.lineno for d in method_node.decorator_list) - 1
            start_line = decorator_start

        # Find method end (find next def/class or end of indent)
        end_line = self._find_method_end(source_lines, method_node)

        # Extract signature and docstring
        signature_line = method_node.lineno - 1
        signature = source_lines[signature_line].rstrip()

        # Detect indentation
        indent = self._detect_indentation(signature)

        # Build new method
        new_method_lines = []

        # Add decorators if present
        if start_line < signature_line:
            for i in range(start_line, signature_line):
                new_method_lines.append(source_lines[i])

        # Add signature
        if request.preserve_signature:
            new_method_lines.append(signature + "\n")
        else:
            # Use signature from AST
            new_method_lines.append(signature + "\n")

        # Add docstring if present and preserve is True
        docstring = ast.get_docstring(method_node)
        if docstring and request.preserve_docstring:
            # Preserve original docstring formatting
            docstring_start = method_node.body[0].lineno - 1
            docstring_end = method_node.body[0].end_lineno
            for i in range(docstring_start, docstring_end):
                new_method_lines.append(source_lines[i])

        # Add new code (ensure proper indentation)
        new_code_lines = request.generated_code.splitlines()
        for line in new_code_lines:
            if line.strip():  # Skip empty lines
                # Ensure proper indentation
                if not line.startswith(indent + " "):
                    # Add base indentation + one level
                    new_method_lines.append(indent + "    " + line.lstrip() + "\n")
                else:
                    new_method_lines.append(line + "\n")
            else:
                new_method_lines.append("\n")

        # Replace in source
        result_lines = (
            source_lines[:start_line] + new_method_lines + source_lines[end_line + 1 :]
        )

        return result_lines

    def _find_method_end(
        self,
        source_lines: list[str],
        method_node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> int:
        """
        Find the last line of a method.

        Args:
            source_lines: Source code lines
            method_node: AST method node

        Returns:
            0-indexed line number of method end
        """
        if hasattr(method_node, "end_lineno") and method_node.end_lineno:
            return method_node.end_lineno - 1

        # Fallback: find next line with same or less indentation
        start_line = method_node.lineno - 1
        signature = source_lines[start_line]
        base_indent = len(signature) - len(signature.lstrip())

        for i in range(start_line + 1, len(source_lines)):
            line = source_lines[i]
            if line.strip():  # Non-empty line
                line_indent = len(line) - len(line.lstrip())
                if line_indent <= base_indent:
                    return i - 1

        return len(source_lines) - 1

    def _detect_indentation(self, line: str) -> str:
        """
        Detect indentation from a line.

        Args:
            line: Source line

        Returns:
            Indentation string (spaces or tabs)
        """
        match = re.match(r"^(\s*)", line)
        if match:
            return match.group(1)
        return ""

    def get_metrics(self) -> dict[str, Any]:
        """
        Get injection metrics for monitoring.

        Returns:
            Dictionary with metrics
        """
        avg_duration_ms = (
            self._total_duration_ms / self._total_injections
            if self._total_injections > 0
            else 0
        )

        success_rate = (
            self._successful_injections / self._total_injections
            if self._total_injections > 0
            else 1.0
        )

        return {
            "total_injections": self._total_injections,
            "successful_injections": self._successful_injections,
            "failed_injections": self._failed_injections,
            "success_rate": round(success_rate, 4),
            "avg_duration_ms": round(avg_duration_ms, 2),
        }

    async def startup(self) -> None:
        """Node startup lifecycle hook."""
        emit_log_event(
            LogLevel.INFO,
            "Node startup initiated",
            {"node_name": "NodeCodegenCodeInjectorEffect"},
        )

    async def shutdown(self) -> None:
        """Node shutdown lifecycle hook."""
        self._deregister_from_consul()
        emit_log_event(
            LogLevel.INFO,
            "Node shutdown completed",
            {"node_name": "NodeCodegenCodeInjectorEffect"},
        )

    def _register_with_consul_sync(self) -> None:
        """Register node with Consul for service discovery (synchronous)."""
        try:
            import consul

            consul_client = consul.Consul(host=self.consul_host, port=self.consul_port)
            service_id = f"omninode-bridge-codegen-code-injector-{self.node_id}"
            service_port = 8064  # Default port for code injector
            service_host = "localhost"

            consul_client.agent.service.register(
                name="omninode-bridge-codegen-code-injector",
                service_id=service_id,
                address=service_host,
                port=service_port,
                tags=["onex", "codegen", "code-injector", "effect"],
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
            f"NodeCodegenCodeInjectorEffect execution failed: {e!s}",
            {"error": str(e), "error_type": type(e).__name__},
        )
        return 1


if __name__ == "__main__":
    exit(main())
