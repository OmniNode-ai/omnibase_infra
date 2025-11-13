#!/usr/bin/env python3
"""
NodeCodegenStubExtractorEffect - Extract method stubs from generated node files.

ONEX v2.0 Compliance:
- Suffix-based naming: NodeCodegenStubExtractorEffect
- Extends NodeEffect from omnibase_core
- Uses ModelOnexError for error handling
- Structured logging with correlation tracking
"""

import ast
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
from .models import ModelMethodStub, ModelStubExtractionResult

# Aliases
OnexError = ModelOnexError
CoreErrorCode = EnumCoreErrorCode


class NodeCodegenStubExtractorEffect(NodeEffect):
    """
    Stub Extractor Effect for extracting method stubs from generated code.

    Extracts methods marked as stubs (requiring implementation) from generated
    node files using AST parsing.

    Responsibilities:
    - Parse Python source code using AST
    - Identify stub methods by markers (# IMPLEMENTATION REQUIRED, pass # Stub)
    - Extract method metadata (name, signature, docstring, line number)
    - Return structured stub information

    ONEX v2.0 Compliance:
    - Suffix-based naming: NodeCodegenStubExtractorEffect
    - Extends NodeEffect from omnibase_core
    - Uses ModelOnexError for error handling
    - Structured logging with correlation tracking

    Performance Targets:
    - Extraction time: <100ms for typical node file
    - Throughput: >100 files/second
    - Memory: <50MB per file

    Example Usage:
        ```python
        container = ModelContainer(...)
        extractor = NodeCodegenStubExtractorEffect(container)

        contract = ModelContractEffect(
            correlation_id=uuid4(),
            input_data={
                "node_file_content": "def foo(): # IMPLEMENTATION REQUIRED\\n    pass",
                "extraction_patterns": ["# IMPLEMENTATION REQUIRED"]
            }
        )

        result = await extractor.execute_effect(contract)
        print(f"Found {result.total_stubs_found} stubs")
        ```
    """

    def __init__(self, container: ModelContainer) -> None:
        """
        Initialize stub extractor with dependency injection container.

        Args:
            container: ONEX container for dependency injection

        Raises:
            ModelOnexError: If container is invalid or initialization fails
        """
        super().__init__(container)

        # Configuration - defensive pattern
        try:
            if hasattr(container.config, "get") and callable(container.config.get):
                self.default_patterns = container.config.get(
                    "stub_extraction_patterns",
                    ["# IMPLEMENTATION REQUIRED", "pass  # Stub"],
                )
            else:
                self.default_patterns = ["# IMPLEMENTATION REQUIRED", "pass  # Stub"]
        except Exception:
            self.default_patterns = ["# IMPLEMENTATION REQUIRED", "pass  # Stub"]

        # Metrics tracking
        self._total_extractions = 0
        self._total_stubs_found = 0
        self._total_duration_ms = 0.0
        self._failed_extractions = 0

        emit_log_event(
            LogLevel.INFO,
            "NodeCodegenStubExtractorEffect initialized successfully",
            {"node_id": self.node_id, "default_patterns": self.default_patterns},
        )

    async def execute_effect(
        self, contract: ModelContractEffect
    ) -> ModelStubExtractionResult:
        """
        Execute stub extraction from node file.

        Args:
            contract: Effect contract with input_data containing:
                - node_file_content (str): Python source code to analyze
                - extraction_patterns (list[str], optional): Stub markers to look for
                - file_path (str, optional): Path to file (for metadata)

        Returns:
            ModelStubExtractionResult with extracted stubs and metadata

        Raises:
            OnexError: If extraction fails or invalid input
        """
        start_time = time.perf_counter()
        correlation_id = contract.correlation_id

        emit_log_event(
            LogLevel.INFO,
            "Starting stub extraction",
            {
                "node_id": str(self.node_id),
                "correlation_id": str(correlation_id),
            },
        )

        try:
            # Parse input from contract
            input_data = contract.input_data or {}
            node_file_content = input_data.get("node_file_content")

            if not node_file_content:
                raise OnexError(
                    error_code=CoreErrorCode.VALIDATION_ERROR,
                    message="Missing required field: node_file_content",
                    details={"correlation_id": str(correlation_id)},
                )

            extraction_patterns = (
                input_data.get("extraction_patterns") or self.default_patterns
            )
            file_path = input_data.get("file_path", "<unknown>")

            # Parse AST
            try:
                tree = ast.parse(node_file_content)
            except SyntaxError as e:
                raise OnexError(
                    error_code=CoreErrorCode.VALIDATION_ERROR,
                    message=f"Syntax error in node file: {e!s}",
                    details={
                        "correlation_id": str(correlation_id),
                        "line": e.lineno,
                        "offset": e.offset,
                    },
                    cause=e,
                )

            # Extract stubs
            stubs = self._extract_stubs(tree, node_file_content, extraction_patterns)

            # Calculate file metrics
            # splitlines() doesn't count trailing empty line when string ends with \n
            file_lines = len(node_file_content.splitlines())
            if node_file_content and node_file_content.endswith('\n'):
                file_lines += 1

            duration_ms = (time.perf_counter() - start_time) * 1000

            # Update metrics
            self._total_extractions += 1
            self._total_stubs_found += len(stubs)
            self._total_duration_ms += duration_ms

            emit_log_event(
                LogLevel.INFO,
                "Stub extraction completed",
                {
                    "node_id": str(self.node_id),
                    "correlation_id": str(correlation_id),
                    "stubs_found": len(stubs),
                    "duration_ms": round(duration_ms, 2),
                },
            )

            return ModelStubExtractionResult(
                success=True,
                stubs=stubs,
                total_stubs_found=len(stubs),
                extraction_time_ms=duration_ms,
                file_path=file_path,
                file_lines=file_lines,
                extraction_patterns_used=extraction_patterns,
            )

        except OnexError:
            self._failed_extractions += 1
            raise

        except Exception as e:
            self._failed_extractions += 1

            emit_log_event(
                LogLevel.ERROR,
                f"Stub extraction failed: {e}",
                {
                    "node_id": str(self.node_id),
                    "correlation_id": str(correlation_id),
                    "error": str(e),
                },
            )

            raise OnexError(
                message=f"Stub extraction failed: {e}",
                error_code=CoreErrorCode.OPERATION_FAILED,
                node_id=str(self.node_id),
                correlation_id=str(correlation_id),
                error=str(e),
            ) from e

    def _extract_stubs(
        self, tree: ast.AST, source_code: str, patterns: list[str]
    ) -> list[ModelMethodStub]:
        """
        Extract stub methods from AST.

        Args:
            tree: Parsed AST tree
            source_code: Original source code
            patterns: Stub marker patterns to look for

        Returns:
            List of ModelMethodStub objects
        """
        stubs = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if self._is_stub_method(node, source_code, patterns):
                    stub = ModelMethodStub(
                        name=node.name,
                        signature=self._extract_signature(node),
                        docstring=ast.get_docstring(node),
                        line_number=node.lineno,
                        context=self._extract_context(node, tree),
                        stub_marker=self._find_stub_marker(node, source_code, patterns),
                    )
                    stubs.append(stub)

        return stubs

    def _is_stub_method(
        self, node: ast.FunctionDef, source_code: str, patterns: list[str]
    ) -> bool:
        """Check if method is a stub (requires implementation)."""
        source_lines = source_code.splitlines()

        # Check function body for stub markers
        if node.lineno <= len(source_lines):
            # Get function body lines
            end_line = (
                node.end_lineno if hasattr(node, "end_lineno") else node.lineno + 10
            )
            func_lines = source_lines[
                node.lineno - 1 : min(end_line, len(source_lines))
            ]
            func_body = "\n".join(func_lines)

            # Check for any stub pattern
            return any(pattern in func_body for pattern in patterns)

        return False

    def _extract_signature(self, node: ast.FunctionDef) -> str:
        """Extract full method signature."""
        # Build parameter list
        params = []

        for arg in node.args.args:
            param = arg.arg
            if arg.annotation:
                param += f": {ast.unparse(arg.annotation)}"
            params.append(param)

        params_str = ", ".join(params)

        # Add return type if available
        return_type = ""
        if node.returns:
            return_type = f" -> {ast.unparse(node.returns)}"

        # Build full signature
        async_prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        return f"{async_prefix}def {node.name}({params_str}){return_type}"

    def _extract_context(self, node: ast.FunctionDef, tree: ast.AST) -> Optional[str]:
        """Extract surrounding context (e.g., class name)."""
        # Find parent class if any
        for parent in ast.walk(tree):
            if isinstance(parent, ast.ClassDef):
                if node in ast.walk(parent):
                    return f"class {parent.name}"
        return None

    def _find_stub_marker(
        self, node: ast.FunctionDef, source_code: str, patterns: list[str]
    ) -> str:
        """Find which stub marker was used."""
        source_lines = source_code.splitlines()
        end_line = node.end_lineno if hasattr(node, "end_lineno") else node.lineno + 10
        func_lines = source_lines[node.lineno - 1 : min(end_line, len(source_lines))]
        func_body = "\n".join(func_lines)

        for pattern in patterns:
            if pattern in func_body:
                return pattern

        return patterns[0]  # Default to first pattern

    def get_metrics(self) -> dict[str, Any]:
        """
        Get extraction metrics for monitoring.

        Returns:
            Dictionary with metrics
        """
        avg_duration_ms = (
            self._total_duration_ms / self._total_extractions
            if self._total_extractions > 0
            else 0
        )

        success_rate = (
            (self._total_extractions - self._failed_extractions)
            / self._total_extractions
            if self._total_extractions > 0
            else 1.0
        )

        return {
            "total_extractions": self._total_extractions,
            "total_stubs_found": self._total_stubs_found,
            "failed_extractions": self._failed_extractions,
            "success_rate": round(success_rate, 4),
            "avg_duration_ms": round(avg_duration_ms, 2),
            "avg_stubs_per_file": (
                self._total_stubs_found / self._total_extractions
                if self._total_extractions > 0
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
            f"NodeCodegenStubExtractorEffect execution failed: {e!s}",
            {"error": str(e), "error_type": type(e).__name__},
        )
        return 1


if __name__ == "__main__":
    exit(main())
