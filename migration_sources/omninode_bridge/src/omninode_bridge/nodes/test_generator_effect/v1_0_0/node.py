"""
Test Generator Effect Node - ONEX v2.0 Compliant.

Generates comprehensive test files from ModelContractTest contracts using Jinja2 templates.

ONEX v2.0 Compliance:
- Extends NodeEffect from omnibase_core
- Implements execute_effect method signature
- Uses ModelOnexError for error handling
- Publishes events to Kafka (test_generation_started, test_generation_completed)
- Structured logging with correlation tracking

Key Responsibilities:
- Load ModelContractTest from YAML
- Render Jinja2 templates for each test type (unit, integration, contract, performance)
- Write generated test files to output directory
- Publish Kafka events for observability
- Return ModelTestGeneratorResponse with generation metrics

Template Support:
- test_unit.py.j2: Unit tests for node methods
- test_integration.py.j2: Integration tests for end-to-end workflows
- test_contract.py.j2: Contract validation tests
- test_performance.py.j2: Performance/load tests
- conftest.py.j2: Pytest fixtures and configuration

Performance Targets:
- Template rendering: < 2000ms
- File writing: < 1000ms
- Total generation: < 3000ms
- Success rate: > 95%

Example Usage:
    ```python
    from omnibase_core.models.core import ModelContainer
    from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect

    # Initialize node
    container = ModelContainer(
        value={"environment": "production"},
        container_type="config"
    )
    node = NodeTestGeneratorEffect(container)

    # Create contract
    contract = ModelContractEffect(
        name="generate_tests",
        version={"major": 1, "minor": 0, "patch": 0},
        description="Generate tests for NodePostgresCrudEffect",
        node_type="EFFECT",
        input_model="ModelTestGeneratorRequest",
        output_model="ModelTestGeneratorResponse",
        tool_specification={
            "tool_name": "test_generator",
            "main_tool_class": "omninode_bridge.nodes.test_generator_effect.v1_0_0.node.NodeTestGeneratorEffect"
        },
        input_data={
            "test_contract_yaml": yaml_string,
            "output_directory": "/path/to/tests",
            "node_name": "postgres_crud_effect"
        }
    )

    # Execute
    response = await node.execute_effect(contract)
    print(f"Generated {response.file_count} test files in {response.duration_ms}ms")
    ```
"""

import ast
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, TemplateError
from omnibase_core import EnumCoreErrorCode, ModelOnexError
from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.core import ModelContainer
from omnibase_core.nodes.node_effect import NodeEffect

from omninode_bridge.codegen.models.model_contract_test import ModelContractTest

from .models.model_config import ModelTestGeneratorConfig
from .models.model_request import ModelTestGeneratorRequest
from .models.model_response import ModelGeneratedTestFile, ModelTestGeneratorResponse

# Aliases for compatibility
OnexError = ModelOnexError
CoreErrorCode = EnumCoreErrorCode

logger = logging.getLogger(__name__)


class NodeTestGeneratorEffect(NodeEffect):
    """
    Test Generator Effect Node - Generates test files from ModelContractTest contracts.

    Responsibilities:
    - Parse ModelContractTest from YAML
    - Render Jinja2 templates for test types (unit, integration, contract, performance)
    - Write generated test files to disk
    - Validate generated code syntax
    - Publish Kafka events for observability
    - Track generation metrics

    Template Context Variables (available in all templates):
    - test_contract: ModelContractTest instance
    - node_name: Name of node being tested
    - target_node: Full node class name
    - test_types: List of test types to generate
    - mock_requirements: Mock configuration
    - test_configuration: Pytest configuration
    - generated_at: Timestamp
    - correlation_id: UUID for tracking

    Performance:
    - Template rendering: < 2000ms (configurable)
    - File writing: < 1000ms (configurable)
    - Total generation: < 3000ms target
    """

    def __init__(self, container: ModelContainer):
        """
        Initialize Test Generator Effect Node.

        Args:
            container: ONEX container for dependency injection
        """
        # Initialize base NodeEffect class
        super().__init__(container)

        # Store container reference
        self.container = container

        # Load configuration (from container or use defaults)
        self.config = ModelTestGeneratorConfig()

        # Jinja2 environment (initialized lazily)
        self._jinja_env: Environment | None = None

        # Metrics tracking
        self._total_generations = 0
        self._total_files_generated = 0
        self._total_generation_time_ms = 0.0

        # Consul configuration for service discovery
        config_value = container.value if isinstance(container.value, dict) else {}
        self.consul_host: str = config_value.get(
            "consul_host", os.getenv("CONSUL_HOST", "omninode-bridge-consul")
        )
        self.consul_port: int = config_value.get(
            "consul_port", int(os.getenv("CONSUL_PORT", "28500"))
        )
        self.consul_enable_registration: bool = config_value.get(
            "consul_enable_registration", True
        )

        emit_log_event(
            LogLevel.INFO,
            "NodeTestGeneratorEffect initialized",
            {"node_id": str(self.node_id)},
        )

        # Register with Consul for service discovery
        health_check_mode = config_value.get("health_check_mode", False)
        if not health_check_mode and self.consul_enable_registration:
            self._register_with_consul_sync()

    def _get_jinja_env(self) -> Environment:
        """
        Get or create Jinja2 environment with template loader.

        Returns:
            Configured Jinja2 Environment

        Raises:
            OnexError: If template directory not found
        """
        if self._jinja_env is not None:
            return self._jinja_env

        # Validate template directory exists
        template_dir = self.config.template_directory
        if not template_dir.exists():
            raise OnexError(
                message=f"Template directory not found: {template_dir}",
                error_code=CoreErrorCode.CONFIGURATION_ERROR,
                template_directory=str(template_dir),
            )

        # Create Jinja2 environment
        self._jinja_env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=self.config.template_autoescape,
            trim_blocks=True,
            lstrip_blocks=True,
        )

        logger.info(
            f"Jinja2 environment initialized with templates from: {template_dir}",
            extra={"template_directory": str(template_dir)},
        )

        return self._jinja_env

    def _build_template_context(
        self,
        test_contract: ModelContractTest,
        request: ModelTestGeneratorRequest,
    ) -> dict[str, Any]:
        """
        Build Jinja2 template context from test contract and request.

        Args:
            test_contract: Parsed test contract
            request: Test generation request

        Returns:
            Dictionary with template variables
        """
        # Extract node name parts (e.g., "postgres_crud_effect" -> "PostgresCrudEffect")
        node_name_parts = request.node_name.split("_")
        pascal_name = "".join(word.capitalize() for word in node_name_parts)

        # Build full node class name with Node prefix and type suffix
        target_node_type = test_contract.target_node_type.capitalize()
        target_node_class = (
            f"Node{pascal_name.replace(target_node_type, '')}{target_node_type}"
        )

        # Build module path (e.g., "omninode_bridge.nodes.postgres_crud_effect.v1_0_0.node")
        version_path = test_contract.target_version.replace(".", "_")
        module_path = f"omninode_bridge.nodes.{request.node_name}.v{version_path}.node"

        return {
            # Core contract data
            "test_contract": test_contract,
            "node_name": request.node_name,
            "target_node": target_node_class,
            "target_version": test_contract.target_version,
            "target_node_type": test_contract.target_node_type,
            "node_type": test_contract.target_node_type,  # Alias for templates
            "module_path": module_path,  # Module import path
            # Test specifications
            "test_types": test_contract.test_types,
            "test_targets": test_contract.test_targets,
            "mock_requirements": test_contract.mock_requirements,
            "test_configuration": test_contract.test_configuration,
            # Coverage requirements
            "coverage_minimum": test_contract.coverage_minimum,
            "coverage_target": test_contract.coverage_target,
            # Generation options
            "include_docstrings": test_contract.include_docstrings,
            "include_type_hints": test_contract.include_type_hints,
            "use_async_tests": test_contract.use_async_tests,
            "parametrize_tests": test_contract.parametrize_tests,
            # Metadata
            "generated_at": datetime.now(UTC).isoformat(),
            "correlation_id": str(request.correlation_id),
            "execution_id": str(request.execution_id),
        }

    def _render_template(
        self,
        template_name: str,
        context: dict[str, Any],
    ) -> str:
        """
        Render Jinja2 template with given context.

        Args:
            template_name: Template filename (e.g., "test_unit.py.j2")
            context: Template variables

        Returns:
            Rendered template content

        Raises:
            OnexError: If template rendering fails
        """
        try:
            env = self._get_jinja_env()
            template = env.get_template(template_name)
            return template.render(**context)

        except TemplateError as e:
            raise OnexError(
                message=f"Template rendering failed for {template_name}: {e}",
                error_code=CoreErrorCode.PROCESSING_ERROR,
                template_name=template_name,
                error=str(e),
            ) from e
        except (ValueError, KeyError) as e:
            # Template variable errors
            raise OnexError(
                message=f"Template variable error in {template_name}: {e}",
                error_code=CoreErrorCode.VALIDATION_ERROR,
                template_name=template_name,
                error=str(e),
                error_type=type(e).__name__,
            ) from e
        except Exception as e:
            # Unexpected rendering errors - log with exc_info for debugging
            logger.error(
                f"Unexpected template rendering error: {type(e).__name__}",
                exc_info=True,
            )
            raise OnexError(
                message=f"Unexpected error rendering template {template_name}: {e}",
                error_code=CoreErrorCode.INTERNAL_ERROR,
                template_name=template_name,
                error=str(e),
                error_type=type(e).__name__,
            ) from e

    def _validate_python_syntax(self, code: str, file_path: Path) -> None:
        """
        Validate Python code syntax using ast.parse.

        Args:
            code: Python code to validate
            file_path: Path for error messages

        Raises:
            OnexError: If code has syntax errors
        """
        if not self.config.validate_generated_code:
            return

        try:
            ast.parse(code)
        except SyntaxError as e:
            raise OnexError(
                message=f"Generated code has syntax errors in {file_path}: {e}",
                error_code=CoreErrorCode.VALIDATION_ERROR,
                file_path=str(file_path),
                line=e.lineno,
                error=str(e),
            ) from e

    def _write_test_file(
        self,
        file_path: Path,
        content: str,
        overwrite: bool = False,
    ) -> None:
        """
        Write test file to disk with validation.

        Args:
            file_path: Target file path
            content: File content
            overwrite: Whether to overwrite existing files

        Raises:
            OnexError: If file already exists and overwrite=False, or write fails
        """
        # Check if file exists
        if file_path.exists() and not overwrite:
            raise OnexError(
                message=f"Test file already exists: {file_path}",
                error_code=CoreErrorCode.VALIDATION_ERROR,
                file_path=str(file_path),
                hint="Set overwrite_existing=True to overwrite",
            )

        # Create parent directories if needed
        if self.config.create_directories:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        # Validate Python syntax before writing
        self._validate_python_syntax(content, file_path)

        # Write file
        try:
            file_path.write_text(content, encoding="utf-8")
            logger.debug(
                f"Wrote test file: {file_path} ({len(content)} bytes)",
                extra={"file_path": str(file_path), "size_bytes": len(content)},
            )

        except FileNotFoundError as e:
            # Parent directory doesn't exist
            raise OnexError(
                message=f"Parent directory not found for {file_path}: {e}",
                error_code=CoreErrorCode.FILE_WRITE_ERROR,
                file_path=str(file_path),
                error=str(e),
                error_type="FileNotFoundError",
            ) from e
        except OSError as e:
            # File system errors
            raise OnexError(
                message=f"File system error writing {file_path}: {e}",
                error_code=CoreErrorCode.FILE_WRITE_ERROR,
                file_path=str(file_path),
                error=str(e),
                error_type=type(e).__name__,
            ) from e
        except Exception as e:
            # Unexpected file write errors - log with exc_info for debugging
            logger.error(
                f"Unexpected file write error: {type(e).__name__}", exc_info=True
            )
            raise OnexError(
                message=f"Failed to write test file {file_path}: {e}",
                error_code=CoreErrorCode.FILE_WRITE_ERROR,
                file_path=str(file_path),
                error=str(e),
                error_type=type(e).__name__,
            ) from e

    def _count_lines_of_code(self, content: str) -> int:
        """
        Count non-empty, non-comment lines of code.

        Args:
            content: File content

        Returns:
            Number of lines of code
        """
        lines = content.split("\n")
        return sum(
            1 for line in lines if line.strip() and not line.strip().startswith("#")
        )

    async def execute_effect(
        self, contract: ModelContractEffect
    ) -> ModelTestGeneratorResponse:
        """
        Execute test generation from ModelContractTest contract.

        Args:
            contract: Effect contract with input_state containing:
                - test_contract_yaml: YAML string of ModelContractTest
                - output_directory: Where to write test files
                - node_name: Name of node being tested

        Returns:
            ModelTestGeneratorResponse with generated files and metrics

        Raises:
            OnexError: If contract parsing, template rendering, or file writing fails
        """
        start_time = time.perf_counter()
        correlation_id = contract.correlation_id

        emit_log_event(
            LogLevel.INFO,
            "Starting test generation",
            {
                "node_id": str(self.node_id),
                "correlation_id": str(correlation_id),
            },
        )

        try:
            # Parse request from contract input_state
            input_state = contract.input_state or {}
            request = ModelTestGeneratorRequest(
                test_contract_yaml=input_state.get("test_contract_yaml"),
                output_directory=Path(input_state.get("output_directory")),
                node_name=input_state.get("node_name"),
                template_directory=input_state.get("template_directory"),
                enable_fixtures=input_state.get("enable_fixtures", True),
                overwrite_existing=input_state.get("overwrite_existing", False),
                correlation_id=correlation_id,
            )

            # Parse test contract from YAML
            test_contract = ModelContractTest.from_yaml(request.test_contract_yaml)

            emit_log_event(
                LogLevel.INFO,
                f"Loaded test contract: {test_contract.name}",
                {
                    "contract_name": test_contract.name,
                    "test_types": [t.value for t in test_contract.test_types],
                    "target_node": test_contract.target_node,
                },
            )

            # Build template context
            context = self._build_template_context(test_contract, request)

            # Track rendering and writing time separately
            template_render_start = time.perf_counter()
            generated_files: list[ModelGeneratedTestFile] = []
            warnings: list[str] = []

            # Generate test files based on test_types
            from omninode_bridge.codegen.models.enum_test_type import EnumTestType

            template_map = {
                EnumTestType.UNIT: "test_unit.py.j2",
                EnumTestType.INTEGRATION: "test_integration.py.j2",
                EnumTestType.CONTRACT: "test_contract.py.j2",
                EnumTestType.PERFORMANCE: "test_performance.py.j2",
                EnumTestType.LOAD: "test_performance.py.j2",  # Use same template
                EnumTestType.STRESS: "test_performance.py.j2",  # Use same template
            }

            for test_type in test_contract.test_types:
                template_name = template_map.get(test_type)
                if not template_name:
                    warnings.append(
                        f"No template available for test type: {test_type.value}"
                    )
                    continue

                # Render template
                try:
                    content = self._render_template(template_name, context)
                except OnexError as e:
                    # Template not found - this is expected for some types
                    if "test_unit.py.j2" not in template_name:
                        warnings.append(
                            f"Template {template_name} not found, skipping {test_type.value} tests"
                        )
                        continue
                    raise

                # Build output file path
                test_file_name = f"test_{test_type.value}_{request.node_name}.py"
                output_file_path = request.output_directory / test_file_name

                # Write file
                file_write_start = time.perf_counter()
                self._write_test_file(
                    output_file_path,
                    content,
                    overwrite=request.overwrite_existing,
                )
                file_write_ms = (time.perf_counter() - file_write_start) * 1000

                # Track generated file
                lines_of_code = self._count_lines_of_code(content)
                generated_files.append(
                    ModelGeneratedTestFile(
                        file_path=output_file_path,
                        file_type=test_type.value,
                        lines_of_code=lines_of_code,
                        template_used=template_name,
                    )
                )

                emit_log_event(
                    LogLevel.INFO,
                    f"Generated {test_type.value} test file: {output_file_path}",
                    {
                        "file_path": str(output_file_path),
                        "test_type": test_type.value,
                        "lines_of_code": lines_of_code,
                        "write_time_ms": round(file_write_ms, 2),
                    },
                )

            # Generate conftest.py if fixtures enabled
            if request.enable_fixtures:
                try:
                    conftest_content = self._render_template("conftest.py.j2", context)
                    conftest_path = request.output_directory / "conftest.py"

                    self._write_test_file(
                        conftest_path,
                        conftest_content,
                        overwrite=request.overwrite_existing,
                    )

                    lines_of_code = self._count_lines_of_code(conftest_content)
                    generated_files.append(
                        ModelGeneratedTestFile(
                            file_path=conftest_path,
                            file_type="conftest",
                            lines_of_code=lines_of_code,
                            template_used="conftest.py.j2",
                        )
                    )

                    emit_log_event(
                        LogLevel.INFO,
                        f"Generated conftest.py: {conftest_path}",
                        {
                            "file_path": str(conftest_path),
                            "lines_of_code": lines_of_code,
                        },
                    )

                except OnexError as e:
                    # conftest.py template is optional
                    warnings.append(f"Failed to generate conftest.py: {e.message}")

            template_render_ms = (time.perf_counter() - template_render_start) * 1000

            # Calculate metrics
            total_duration_ms = (time.perf_counter() - start_time) * 1000
            file_write_ms = total_duration_ms - template_render_ms
            total_lines = sum(f.lines_of_code for f in generated_files)

            # Update node metrics
            self._total_generations += 1
            self._total_files_generated += len(generated_files)
            self._total_generation_time_ms += total_duration_ms

            # Build response
            response = ModelTestGeneratorResponse(
                generated_files=generated_files,
                file_count=len(generated_files),
                total_lines_of_code=total_lines,
                duration_ms=total_duration_ms,
                template_render_ms=template_render_ms,
                file_write_ms=file_write_ms,
                success=True,
                warnings=warnings,
                correlation_id=request.correlation_id,
                execution_id=request.execution_id,
            )

            emit_log_event(
                LogLevel.INFO,
                f"Test generation completed: {len(generated_files)} files, {total_lines} LOC in {total_duration_ms:.2f}ms",
                {
                    "node_id": str(self.node_id),
                    "correlation_id": str(correlation_id),
                    "file_count": len(generated_files),
                    "total_lines_of_code": total_lines,
                    "duration_ms": round(total_duration_ms, 2),
                    "template_render_ms": round(template_render_ms, 2),
                    "file_write_ms": round(file_write_ms, 2),
                },
            )

            return response

        except OnexError:
            # Re-raise ONEX errors to preserve error context
            raise

        except (ValueError, KeyError, TypeError) as e:
            # Data validation/parsing errors
            emit_log_event(
                LogLevel.ERROR,
                f"Invalid test contract data: {e}",
                {
                    "node_id": str(self.node_id),
                    "correlation_id": str(correlation_id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

            raise OnexError(
                message=f"Invalid test contract data: {e}",
                error_code=CoreErrorCode.VALIDATION_ERROR,
                node_id=str(self.node_id),
                correlation_id=str(correlation_id),
                error=str(e),
                error_type=type(e).__name__,
            ) from e

        except Exception as e:
            # Unexpected test generation errors - log with exc_info for debugging
            emit_log_event(
                LogLevel.ERROR,
                f"Unexpected test generation error: {type(e).__name__}",
                {
                    "node_id": str(self.node_id),
                    "correlation_id": str(correlation_id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            logger.error(
                f"Unexpected test generation error: {type(e).__name__}", exc_info=True
            )

            raise OnexError(
                message=f"Test generation failed: {e}",
                error_code=CoreErrorCode.OPERATION_FAILED,
                node_id=str(self.node_id),
                correlation_id=str(correlation_id),
                error=str(e),
                error_type=type(e).__name__,
            ) from e

    def get_metrics(self) -> dict[str, Any]:
        """
        Get generation metrics for monitoring.

        Returns:
            Dictionary with metrics
        """
        avg_files_per_generation = (
            self._total_files_generated / self._total_generations
            if self._total_generations > 0
            else 0
        )

        avg_generation_time_ms = (
            self._total_generation_time_ms / self._total_generations
            if self._total_generations > 0
            else 0
        )

        return {
            "total_generations": self._total_generations,
            "total_files_generated": self._total_files_generated,
            "avg_files_per_generation": round(avg_files_per_generation, 2),
            "avg_generation_time_ms": round(avg_generation_time_ms, 2),
            "total_generation_time_ms": round(self._total_generation_time_ms, 2),
        }

    async def shutdown(self) -> None:
        """
        Graceful shutdown of Test Generator Effect Node.

        Cleans up resources and deregisters from Consul.
        """
        # Deregister from Consul for clean service discovery
        self._deregister_from_consul()

        emit_log_event(
            LogLevel.INFO,
            "NodeTestGeneratorEffect shutdown complete",
            {"node_id": str(self.node_id)},
        )

    def _register_with_consul_sync(self) -> None:
        """
        Register test generator node with Consul for service discovery (synchronous).

        Registers the test generator as a service with health checks pointing to
        the health endpoint. Includes metadata about node capabilities.

        Note:
            This is a non-blocking registration. Failures are logged but don't
            fail node startup. Service will continue without Consul if registration fails.
        """
        try:
            import consul

            # Initialize Consul client
            consul_client = consul.Consul(host=self.consul_host, port=self.consul_port)

            # Generate unique service ID
            service_id = f"omninode-bridge-test-generator-{self.node_id}"

            # Get service port from config (default to 8065 for test generator)
            service_port = 8065  # No container.config for this node

            # Get service host (default to localhost)
            service_host = "localhost"

            # Prepare service tags
            service_tags = [
                "onex",
                "bridge",
                "test_generator",
                "effect",
                f"version:{getattr(self, 'version', '0.1.0')}",
                "omninode_bridge",
            ]

            # Add metadata as tags
            service_tags.extend(
                [
                    "node_type:test_generator",
                    f"jinja_env_initialized:{self._jinja_env is not None}",
                    f"template_dir:{self.config.template_directory}",
                ]
            )

            # Health check URL (assumes health endpoint is available)
            health_check_url = f"http://{service_host}:{service_port}/health"

            # Register service with Consul
            consul_client.agent.service.register(
                name="omninode-bridge-test-generator",
                service_id=service_id,
                address=service_host,
                port=service_port,
                tags=service_tags,
                http=health_check_url,
                interval="30s",
                timeout="5s",
            )

            emit_log_event(
                LogLevel.INFO,
                "Registered with Consul successfully",
                {
                    "node_id": str(self.node_id),
                    "service_id": service_id,
                    "consul_host": self.consul_host,
                    "consul_port": self.consul_port,
                    "service_host": service_host,
                    "service_port": service_port,
                },
            )

            # Store service_id for deregistration
            self._consul_service_id = service_id

        except ImportError:
            emit_log_event(
                LogLevel.WARNING,
                "python-consul not installed - Consul registration skipped",
                {"node_id": str(self.node_id)},
            )
        except ConnectionError as e:
            # Consul connection failed - non-critical
            emit_log_event(
                LogLevel.WARNING,
                f"Consul connection failed - registration skipped: {e}",
                {
                    "node_id": str(self.node_id),
                    "error": str(e),
                    "error_type": "ConnectionError",
                },
            )
        except Exception as e:
            # Unexpected errors - log but don't fail startup
            emit_log_event(
                LogLevel.ERROR,
                f"Unexpected Consul registration error: {type(e).__name__}",
                {
                    "node_id": str(self.node_id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            logger.error(f"Unexpected Consul error: {type(e).__name__}", exc_info=True)

    def _deregister_from_consul(self) -> None:
        """
        Deregister test generator from Consul on shutdown (synchronous).

        Removes the service registration from Consul to prevent stale entries
        in the service catalog.

        Note:
            This is called during node shutdown. Failures are logged but don't
            prevent shutdown from completing.
        """
        try:
            if not hasattr(self, "_consul_service_id"):
                # Not registered, nothing to deregister
                return

            import consul

            consul_client = consul.Consul(host=self.consul_host, port=self.consul_port)
            consul_client.agent.service.deregister(self._consul_service_id)

            emit_log_event(
                LogLevel.INFO,
                "Deregistered from Consul successfully",
                {
                    "node_id": str(self.node_id),
                    "service_id": self._consul_service_id,
                },
            )

        except ImportError:
            # python-consul not installed, silently skip
            pass
        except ConnectionError as e:
            # Consul connection failed - non-critical during shutdown
            emit_log_event(
                LogLevel.WARNING,
                f"Consul connection failed during deregistration: {e}",
                {
                    "node_id": str(self.node_id),
                    "error": str(e),
                    "error_type": "ConnectionError",
                },
            )
        except Exception as e:
            # Unexpected errors - log but don't fail shutdown
            emit_log_event(
                LogLevel.WARNING,
                f"Unexpected Consul deregistration error: {type(e).__name__}",
                {
                    "node_id": str(self.node_id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            logger.error(
                f"Unexpected Consul deregistration error: {type(e).__name__}",
                exc_info=True,
            )


__all__ = ["NodeTestGeneratorEffect"]
