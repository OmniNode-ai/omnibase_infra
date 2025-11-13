#!/usr/bin/env python3
"""
NodeCodegenOrchestrator - ONEX Node Code Generation Coordinator.

Orchestrates the 8-stage code generation pipeline using LlamaIndex Workflows
with Kafka event publishing and RAG intelligence integration.

ONEX v2.0 Compliance:
- Suffix-based naming: NodeCodegenOrchestrator
- Extends NodeOrchestrator from omnibase_core
- LlamaIndex Workflows integration
- Event-driven coordination with Kafka
- 95% code reuse from NodeBridgeOrchestrator pattern
"""

import os
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from omnibase_core import EnumCoreErrorCode, ModelOnexError
from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event
from omnibase_core.models.contracts.model_contract_effect import (
    ModelContractEffect,
)
from omnibase_core.models.contracts.model_contract_orchestrator import (
    ModelContractOrchestrator,
)
from omnibase_core.models.core import ModelContainer
from omnibase_core.nodes.node_orchestrator import NodeOrchestrator

from omninode_bridge.events.codegen import (
    KAFKA_TOPICS,
    ModelEventNodeGenerationCompleted,
    ModelEventNodeGenerationFailed,
)
from omninode_bridge.nodes.codegen_orchestrator.v1_0_0.typed_results import (
    GenerationResult,
)
from omninode_bridge.nodes.codegen_orchestrator.v1_0_0.workflow import (
    CodeGenerationWorkflow,
)
from omninode_bridge.protocols import KafkaClientProtocol
from omninode_bridge.services.kafka_client import KafkaClient


class NodeCodegenOrchestrator(NodeOrchestrator):
    """
    Code Generation Orchestrator for ONEX nodes.

    Coordinates the 8-stage generation pipeline:
    1. Prompt parsing (5s)
    2. Intelligence gathering (3s) - optional RAG query
    3. Contract building (2s)
    4. Code generation (10-15s)
    5. Event bus integration (2s)
    6. Validation (5s)
    7. Refinement (3s)
    8. File writing (3s)

    Total Target: 53 seconds

    Event Publishing:
    - NODE_GENERATION_STARTED: Workflow begins
    - NODE_GENERATION_STAGE_COMPLETED: Each stage completes (8x)
    - NODE_GENERATION_COMPLETED: Successful generation
    - NODE_GENERATION_FAILED: Generation failure

    ONEX Pattern Compliance:
    - Extends NodeOrchestrator from omnibase_core
    - Uses ModelContainer for dependency injection
    - Integrates LlamaIndex Workflows for stage coordination
    - Publishes events to Kafka for real-time progress tracking
    """

    def __init__(self, container: ModelContainer) -> None:
        """
        Initialize Code Generation Orchestrator.

        Args:
            container: ONEX container for dependency injection

        Raises:
            ModelOnexError: If container is invalid or initialization fails
        """
        super().__init__(container)

        # Configuration - defensive pattern for dependency_injector
        try:
            if hasattr(container.config, "get") and callable(container.config.get):
                self.kafka_broker_url: str = container.config.get(
                    "kafka_broker_url",
                    os.getenv(
                        "KAFKA_BOOTSTRAP_SERVERS", "omninode-bridge-redpanda:9092"
                    ),
                )
                self.omniarchon_url: str = container.config.get(
                    "omniarchon_url", "http://omniarchon:8060"
                )
                self.default_output_dir: str = container.config.get(
                    "default_output_dir", "./generated_nodes"
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
                # Fallback to defaults
                self.kafka_broker_url = os.getenv(
                    "KAFKA_BOOTSTRAP_SERVERS", "omninode-bridge-redpanda:9092"
                )
                self.omniarchon_url = "http://omniarchon:8060"
                self.default_output_dir = "./generated_nodes"
                self.consul_host = os.getenv("CONSUL_HOST", "omninode-bridge-consul")
                self.consul_port = int(os.getenv("CONSUL_PORT", "28500"))
                self.consul_enable_registration = True
        except Exception:
            # Fallback to defaults if any error
            self.kafka_broker_url = os.getenv(
                "KAFKA_BOOTSTRAP_SERVERS", "omninode-bridge-redpanda:9092"
            )
            self.omniarchon_url = "http://omniarchon:8060"
            self.default_output_dir = "./generated_nodes"
            self.consul_host = os.getenv("CONSUL_HOST", "omninode-bridge-consul")
            self.consul_port = int(os.getenv("CONSUL_PORT", "28500"))
            self.consul_enable_registration = True

        # Get or create KafkaClient from container
        try:
            health_check_mode = (
                container.config.get("health_check_mode", False)
                if hasattr(container.config, "get")
                else False
            )
        except Exception:
            health_check_mode = False

        # Get KafkaClient from container using protocol type
        try:
            self.kafka_client = container.get_service(KafkaClientProtocol)
        except Exception:
            # Service not available in container
            self.kafka_client = None

        if self.kafka_client is None and not health_check_mode:
            try:
                self.kafka_client = KafkaClient(
                    bootstrap_servers=self.kafka_broker_url,
                    enable_dead_letter_queue=True,
                    max_retry_attempts=3,
                    timeout_seconds=30,
                )
                # Register service if container supports it
                if hasattr(container, "register_service"):
                    container.register_service(KafkaClientProtocol, self.kafka_client)
            except ImportError:
                emit_log_event(
                    LogLevel.WARNING,
                    "KafkaClient not available - events will be logged only",
                    {"node_id": self.node_id},
                )
                self.kafka_client = None

        # Workflow state tracking
        self.active_workflows: dict[str, CodeGenerationWorkflow] = {}

        emit_log_event(
            LogLevel.INFO,
            "NodeCodegenOrchestrator initialized successfully",
            {
                "node_id": self.node_id,
                "kafka_enabled": self.kafka_client is not None,
                "omniarchon_url": self.omniarchon_url,
            },
        )

        # Register with Consul for service discovery (skip in health check mode)
        health_check_mode = (
            container.config.get("health_check_mode", False)
            if hasattr(container.config, "get")
            else False
        )
        if not health_check_mode and self.consul_enable_registration:
            self._register_with_consul_sync()

    async def execute_orchestration(
        self, contract: ModelContractOrchestrator
    ) -> GenerationResult:
        """
        Execute code generation orchestration using LlamaIndex Workflows.

        Args:
            contract: Orchestrator contract with generation parameters

        Returns:
            GenerationResult with strongly-typed generation results

        Raises:
            ModelOnexError: If generation fails
        """
        correlation_id = contract.correlation_id
        workflow_id = uuid4()

        emit_log_event(
            LogLevel.INFO,
            "Starting code generation workflow",
            {
                "node_id": self.node_id,
                "correlation_id": str(correlation_id),
                "workflow_id": str(workflow_id),
            },
        )

        try:
            # Extract generation parameters from contract
            if not hasattr(contract, "input_data") or contract.input_data is None:
                raise ModelOnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message="Contract missing input_data for code generation",
                    details={
                        "node_id": self.node_id,
                        "correlation_id": str(correlation_id),
                    },
                )

            input_data = contract.input_data
            prompt = input_data.get("prompt")
            output_directory = input_data.get(
                "output_directory", self.default_output_dir
            )
            node_type_hint = input_data.get("node_type_hint")
            interactive_mode = input_data.get("interactive_mode", False)
            enable_intelligence = input_data.get("enable_intelligence", True)
            enable_quorum = input_data.get("enable_quorum", False)

            if not prompt:
                raise ModelOnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message="Missing required 'prompt' parameter",
                    details={"node_id": self.node_id},
                )

            # Initialize workflow
            workflow = CodeGenerationWorkflow(
                kafka_client=self.kafka_client,
                enable_intelligence=enable_intelligence,
                enable_quorum=enable_quorum,
                timeout=300.0,  # 5 minute timeout
                verbose=True,
            )
            self.active_workflows[str(workflow_id)] = workflow

            # Execute workflow with LlamaIndex
            result = await workflow.run(
                prompt=prompt,
                output_directory=output_directory,
                node_type_hint=node_type_hint,
                interactive_mode=interactive_mode,
                correlation_id=correlation_id,
            )

            # Publish completion event
            await self._publish_completed_event(correlation_id, workflow_id, result)

            emit_log_event(
                LogLevel.INFO,
                "Code generation workflow completed successfully",
                {
                    "node_id": self.node_id,
                    "workflow_id": str(workflow_id),
                    "duration_seconds": result.get("total_duration_seconds", 0),
                    "files_generated": len(result.get("generated_files", [])),
                },
            )

            # Cleanup
            self.active_workflows.pop(str(workflow_id), None)

            return result

        except Exception as e:
            # Publish failure event
            await self._publish_failed_event(correlation_id, workflow_id, e)

            # Cleanup
            self.active_workflows.pop(str(workflow_id), None)

            # Re-raise or wrap exception
            if isinstance(e, ModelOnexError):
                raise

            raise ModelOnexError(
                error_code=EnumCoreErrorCode.OPERATION_FAILED,
                message=f"Code generation workflow failed: {e!s}",
                details={
                    "node_id": self.node_id,
                    "correlation_id": str(correlation_id),
                    "workflow_id": str(workflow_id),
                },
                cause=e,
            )

    async def _publish_completed_event(
        self, correlation_id: UUID, workflow_id: UUID, result: GenerationResult
    ) -> None:
        """Publish workflow completed event to Kafka with OnexEnvelopeV1 wrapping."""
        if not self.kafka_client or not self.kafka_client.is_connected:
            emit_log_event(
                LogLevel.DEBUG,
                "Kafka client not available - skipping event publish",
                {"workflow_id": str(workflow_id)},
            )
            return

        event = ModelEventNodeGenerationCompleted(
            correlation_id=correlation_id,
            workflow_id=workflow_id,
            total_duration_seconds=result.get("total_duration_seconds", 0),
            generated_files=result.get("generated_files", []),
            node_type=result.get("node_type", "unknown"),
            service_name=result.get("service_name", "unknown"),
            quality_score=result.get("quality_score", 0.0),
            test_coverage=result.get("test_coverage"),
            complexity_score=result.get("complexity_score"),
            patterns_applied=result.get("patterns_applied", []),
            intelligence_sources=result.get("intelligence_sources", []),
            primary_model=result.get("primary_model", "gemini-2.5-flash"),
            total_tokens=result.get("total_tokens", 0),
            total_cost_usd=result.get("total_cost_usd", 0.0),
            contract_yaml=result.get("contract_yaml", ""),
            node_module=result.get("node_module", ""),
            models=result.get("models", []),
            enums=result.get("enums", []),
            tests=result.get("tests", []),
        )

        try:
            # Publish with OnexEnvelopeV1 wrapping for standardized event format
            success = await self.kafka_client.publish_with_envelope(
                event_type="NODE_GENERATION_COMPLETED",
                source_node_id=str(self.node_id),
                payload=event.model_dump(),
                topic=KAFKA_TOPICS["NODE_GENERATION_COMPLETED"],
                correlation_id=correlation_id,
                metadata={
                    "event_category": "code_generation",
                    "node_type": "orchestrator",
                    "workflow_id": str(workflow_id),
                },
            )

            if success:
                emit_log_event(
                    LogLevel.DEBUG,
                    "Published NODE_GENERATION_COMPLETED event (OnexEnvelopeV1)",
                    {
                        "node_id": self.node_id,
                        "workflow_id": str(workflow_id),
                        "correlation_id": str(correlation_id),
                        "envelope_wrapped": True,
                    },
                )
        except Exception as e:
            emit_log_event(
                LogLevel.WARNING,
                f"Failed to publish completed event: {e}",
                {"workflow_id": str(workflow_id), "error": str(e)},
            )

    async def _publish_failed_event(
        self, correlation_id: UUID, workflow_id: UUID, error: Exception
    ) -> None:
        """Publish workflow failed event to Kafka with OnexEnvelopeV1 wrapping."""
        if not self.kafka_client or not self.kafka_client.is_connected:
            emit_log_event(
                LogLevel.DEBUG,
                "Kafka client not available - skipping event publish",
                {"workflow_id": str(workflow_id)},
            )
            return

        event = ModelEventNodeGenerationFailed(
            correlation_id=correlation_id,
            workflow_id=workflow_id,
            failed_stage="unknown",
            partial_duration_seconds=0.0,
            error_code="GENERATION_FAILED",
            error_message=str(error),
            error_context={},
            stack_trace=None,
            partial_files=[],
        )

        try:
            # Publish with OnexEnvelopeV1 wrapping for standardized event format
            success = await self.kafka_client.publish_with_envelope(
                event_type="NODE_GENERATION_FAILED",
                source_node_id=str(self.node_id),
                payload=event.model_dump(),
                topic=KAFKA_TOPICS["NODE_GENERATION_FAILED"],
                correlation_id=correlation_id,
                metadata={
                    "event_category": "code_generation",
                    "node_type": "orchestrator",
                    "workflow_id": str(workflow_id),
                    "error_code": "GENERATION_FAILED",
                },
            )

            if success:
                emit_log_event(
                    LogLevel.DEBUG,
                    "Published NODE_GENERATION_FAILED event (OnexEnvelopeV1)",
                    {
                        "node_id": self.node_id,
                        "workflow_id": str(workflow_id),
                        "correlation_id": str(correlation_id),
                        "envelope_wrapped": True,
                    },
                )
        except Exception as e:
            emit_log_event(
                LogLevel.WARNING,
                f"Failed to publish failed event: {e}",
                {"workflow_id": str(workflow_id), "error": str(e)},
            )

    async def startup(self) -> None:
        """Node startup lifecycle hook."""
        emit_log_event(
            LogLevel.INFO,
            "NodeCodegenOrchestrator starting up",
            {"node_id": self.node_id},
        )

        # Initialize container services
        if hasattr(self.container, "initialize"):
            try:
                await self.container.initialize()
            except Exception as e:
                emit_log_event(
                    LogLevel.WARNING,
                    f"Container initialization failed: {e}",
                    {"node_id": self.node_id},
                )

        # Connect Kafka if available
        if self.kafka_client:
            try:
                await self.kafka_client.connect()
                emit_log_event(
                    LogLevel.INFO,
                    "Kafka client connected",
                    {"node_id": self.node_id},
                )
            except Exception as e:
                emit_log_event(
                    LogLevel.WARNING,
                    f"Kafka connection failed: {e}",
                    {"node_id": self.node_id},
                )

    async def shutdown(self) -> None:
        """Node shutdown lifecycle hook."""
        emit_log_event(
            LogLevel.INFO,
            "NodeCodegenOrchestrator shutting down",
            {"node_id": self.node_id},
        )

        # Cancel active workflows
        for workflow_id, workflow in list(self.active_workflows.items()):
            emit_log_event(
                LogLevel.WARNING,
                f"Cancelling active workflow: {workflow_id}",
                {"node_id": self.node_id},
            )

        # Cleanup container services
        if hasattr(self.container, "cleanup"):
            try:
                await self.container.cleanup()
            except Exception as e:
                emit_log_event(
                    LogLevel.WARNING,
                    f"Container cleanup failed: {e}",
                    {"node_id": self.node_id},
                )

        # Deregister from Consul for clean service discovery
        self._deregister_from_consul()

    def _register_with_consul_sync(self) -> None:
        """
        Register codegen orchestrator node with Consul for service discovery (synchronous).

        Registers the codegen orchestrator as a service with health checks pointing to
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
            service_id = f"omninode-bridge-codegen-orchestrator-{self.node_id}"

            # Get service port from config (default to 8062 for codegen orchestrator)
            service_port = int(
                self.container.config.get("service_port", 8062)
                if hasattr(self.container.config, "get")
                else 8062
            )

            # Get service host from config (default to localhost)
            service_host = (
                self.container.config.get("service_host", "localhost")
                if hasattr(self.container.config, "get")
                else "localhost"
            )

            # Prepare service tags
            service_tags = [
                "onex",
                "bridge",
                "codegen-orchestrator",
                f"version:{getattr(self, 'version', '0.1.0')}",
                "omninode_bridge",
            ]

            # Prepare service metadata (encoded in tags for MVP compatibility)
            service_tags.extend(
                [
                    "node_type:codegen-orchestrator",
                    f"kafka_enabled:{self.kafka_client is not None}",
                ]
            )

            # Health check URL (assumes health endpoint is available)
            health_check_url = f"http://{service_host}:{service_port}/health"

            # Register service with Consul
            consul_client.agent.service.register(
                name="omninode-bridge-codegen-orchestrator",
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
                    "node_id": self.node_id,
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
                {"node_id": self.node_id},
            )
        except Exception as e:
            emit_log_event(
                LogLevel.ERROR,
                "Failed to register with Consul",
                {
                    "node_id": self.node_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

    def _deregister_from_consul(self) -> None:
        """
        Deregister codegen orchestrator from Consul on shutdown (synchronous).

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
                    "node_id": self.node_id,
                    "service_id": self._consul_service_id,
                },
            )

        except ImportError:
            # python-consul not installed, silently skip
            pass
        except Exception as e:
            emit_log_event(
                LogLevel.WARNING,
                "Failed to deregister from Consul",
                {
                    "node_id": self.node_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

    async def execute_intents(self, intents: list[dict]) -> dict[str, any]:
        """
        Execute intents from reducer nodes.

        This method routes intents to appropriate Effect nodes for execution.
        Used to handle persistence, storage, and event publishing intents
        from pure reducer nodes.

        Args:
            intents: List of intent dictionaries with:
                - intent_type: Type of intent (PERSIST_METRICS, STORE_ARTIFACT, etc.)
                - target: Target Effect node (store_effect, event_publisher, etc.)
                - payload: Intent-specific data
                - priority: Execution priority (higher = earlier)

        Returns:
            Dictionary with execution results:
                - executed_intents: Number of successfully executed intents
                - failed_intents: Number of failed intents
                - execution_time_ms: Total execution time
                - results: List of individual intent execution results

        Example:
            ```python
            intents = [
                {
                    "intent_type": "STORE_ARTIFACT",
                    "target": "store_effect",
                    "payload": {"file_path": "/path/to/file.py", "content": "..."},
                    "priority": 1
                }
            ]
            results = await orchestrator.execute_intents(intents)
            ```
        """
        import time

        start_time = time.perf_counter()
        executed = 0
        failed = 0
        results = []

        # Sort intents by priority (higher priority first)
        sorted_intents = sorted(
            intents, key=lambda i: i.get("priority", 0), reverse=True
        )

        emit_log_event(
            LogLevel.INFO,
            f"Executing {len(sorted_intents)} intents from reducer",
            {"node_id": self.node_id, "intent_count": len(sorted_intents)},
        )

        for intent in sorted_intents:
            try:
                intent_type = intent.get("intent_type")
                target = intent.get("target")
                payload = intent.get("payload", {})

                emit_log_event(
                    LogLevel.DEBUG,
                    f"Executing intent: {intent_type} -> {target}",
                    {"intent_type": intent_type, "target": target},
                )

                # Route to appropriate Effect node
                if target == "store_effect":
                    result = await self._execute_store_intent(payload)
                elif target == "event_publisher":
                    result = await self._execute_event_intent(payload)
                else:
                    emit_log_event(
                        LogLevel.WARNING,
                        f"Unknown intent target: {target}",
                        {"target": target, "intent_type": intent_type},
                    )
                    result = {"success": False, "error": f"Unknown target: {target}"}
                    failed += 1
                    continue

                executed += 1
                results.append(
                    {"intent_type": intent_type, "target": target, "result": result}
                )

            except Exception as e:
                failed += 1
                error_msg = f"Intent execution failed: {e}"
                emit_log_event(
                    LogLevel.ERROR,
                    error_msg,
                    {
                        "node_id": self.node_id,
                        "intent": intent,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
                results.append(
                    {
                        "intent_type": intent.get("intent_type"),
                        "target": intent.get("target"),
                        "result": {"success": False, "error": str(e)},
                    }
                )

        duration_ms = (time.perf_counter() - start_time) * 1000

        emit_log_event(
            LogLevel.INFO,
            f"Intent execution complete: {executed} succeeded, {failed} failed",
            {
                "node_id": self.node_id,
                "executed": executed,
                "failed": failed,
                "duration_ms": round(duration_ms, 2),
            },
        )

        return {
            "executed_intents": executed,
            "failed_intents": failed,
            "execution_time_ms": duration_ms,
            "results": results,
        }

    async def _execute_store_intent(self, payload: dict) -> dict:
        """
        Execute storage intent by routing to NodeCodegenStoreEffect.

        Args:
            payload: Storage intent payload with file_path, content, etc.

        Returns:
            Storage execution result
        """
        from omninode_bridge.nodes.codegen_store_effect.v1_0_0.node import (
            NodeCodegenStoreEffect,
        )

        try:
            # Create or get store effect node from container
            store_effect = NodeCodegenStoreEffect(self.container)

            # Build contract for store effect
            contract = ModelContractEffect(
                correlation_id=uuid4(),
                input_state={
                    "storage_requests": [payload],
                    "base_directory": payload.get(
                        "base_directory", self.default_output_dir
                    ),
                },
            )

            # Execute storage
            result = await store_effect.execute_effect(contract)

            return {
                "success": result.success,
                "artifacts_stored": result.artifacts_stored,
                "stored_files": result.stored_files,
                "storage_time_ms": result.storage_time_ms,
            }

        except Exception as e:
            emit_log_event(
                LogLevel.ERROR,
                f"Store intent execution failed: {e}",
                {"error": str(e), "payload": payload},
            )
            return {"success": False, "error": str(e)}

    async def _execute_event_intent(self, payload: dict) -> dict:
        """
        Execute event publishing intent via Kafka.

        Args:
            payload: Event intent payload with topic, key, event data

        Returns:
            Event publishing result
        """
        try:
            if not self.kafka_client:
                return {
                    "success": False,
                    "error": "Kafka client not available",
                }

            topic = payload.get("topic")
            key = payload.get("key")
            event = payload.get("event")

            if not topic or not event:
                return {
                    "success": False,
                    "error": "Missing required fields: topic, event",
                }

            # Publish event via Kafka client
            await self.kafka_client.publish_with_envelope(
                topic=topic, key=key, event=event
            )

            return {
                "success": True,
                "topic": topic,
                "key": key,
            }

        except Exception as e:
            emit_log_event(
                LogLevel.ERROR,
                f"Event intent execution failed: {e}",
                {"error": str(e), "payload": payload},
            )
            return {"success": False, "error": str(e)}

    def get_metrics(self) -> dict[str, Any]:
        """Get orchestrator metrics."""
        return {
            "active_workflows": len(getattr(self, "active_workflows", {})),
            "kafka_connected": bool(
                getattr(self, "kafka_client", None) and self.kafka_client.is_connected
            ),
            "orchestrator_type": "codegen",
        }


def main() -> int:
    """Entry point for node execution."""
    try:
        from omnibase_core.infrastructure.node_base import NodeBase

        CONTRACT_FILENAME = "contract.yaml"
        node_base = NodeBase(Path(__file__).parent / CONTRACT_FILENAME)
        return 0
    except Exception as e:
        emit_log_event(
            LogLevel.ERROR,
            f"NodeCodegenOrchestrator execution failed: {e!s}",
            {"error": str(e)},
        )
        return 1


if __name__ == "__main__":
    exit(main())
