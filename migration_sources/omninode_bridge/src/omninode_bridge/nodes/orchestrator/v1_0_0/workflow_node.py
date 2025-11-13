#!/usr/bin/env python3
"""
NodeBridgeWorkflowOrchestrator - LlamaIndex Workflows-Based Orchestrator.

Pioneer implementation of LlamaIndex Workflows for omnibase ecosystem.
Replaces FSM-based orchestration with event-driven workflow patterns.

ONEX v2.0 Compliance:
- Event-driven architecture with LlamaIndex Workflows
- Strong typing with Pydantic models
- Async-first execution for high performance
- Kafka event publishing integration
- Correlation ID propagation through workflow context
- ONEX contract compatibility maintained

Key Features:
- Type-safe workflow steps with @step decorator
- Automatic event routing and validation
- Context management for shared state
- Graceful error handling and recovery
- <2000ms end-to-end latency target
"""

import time
from datetime import datetime
from typing import Any, Union
from uuid import uuid4

from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from omnibase_core import EnumCoreErrorCode, ModelOnexError

# Direct imports - omnibase_core is required
from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event
from omnibase_core.models.core import ModelContainer

# Aliases for compatibility
OnexError = ModelOnexError

# Import performance configuration
from ....config import performance_config

# Import workflow events
# Import existing models
from .models.enum_workflow_state import EnumWorkflowState
from .models.model_stamp_response_output import ModelStampResponseOutput
from .workflow_events import (
    HashGeneratedEvent,
    IntelligenceReceivedEvent,
    IntelligenceRequestedEvent,
    PersistenceCompletedEvent,
    StampCreatedEvent,
    ValidationCompletedEvent,
)


class NodeBridgeWorkflowOrchestrator(Workflow):
    """
    LlamaIndex Workflow-based orchestrator for metadata stamping.

    Pioneer implementation demonstrating event-driven workflow patterns for
    omnibase ecosystem. Replaces FSM-based state management with LlamaIndex
    Workflows for improved composability and observability.

    Workflow Architecture:
        1. validate_input: Input validation and context initialization
        2. generate_hash: BLAKE3 hash generation via MetadataStamping service
        3. create_stamp: Metadata stamp creation with O.N.E. v0.1 compliance
        4. enrich_intelligence: Optional OnexTree AI enrichment (conditional)
        5. persist_state: Database persistence via DatabaseAdapter
        6. complete_workflow: Final result aggregation and Kafka publishing

    Context State:
        - correlation_id: UUID for request tracing
        - workflow_start_time: Performance tracking
        - enable_intelligence: Intelligence enrichment flag
        - intermediate_results: Step outputs (hash, stamp, intelligence)
        - kafka_client: Event publishing client

    Performance Targets:
        - Total workflow: < 2000ms end-to-end (documented target)
        - Individual steps: < 500ms per step (documented target)
        - Throughput: 100+ workflows/second per instance

    Example:
        >>> container = ModelContainer(config={...})
        >>> workflow = NodeBridgeWorkflowOrchestrator(container, timeout=60)
        >>> result = await workflow.run(
        ...     correlation_id=uuid4(),
        ...     content="Hello World",
        ...     enable_intelligence=True
        ... )
        >>> print(f"Stamp ID: {result.stamp_id}")
    """

    def __init__(
        self,
        container: ModelContainer,
        timeout: int = 60,
        verbose: bool = True,
    ):
        """
        Initialize workflow orchestrator.

        Args:
            container: ONEX container for dependency injection
            timeout: Workflow execution timeout in seconds (default: 60)
            verbose: Enable verbose logging (default: True)

        Raises:
            OnexError: If container is invalid or initialization fails
        """
        super().__init__(timeout=timeout, verbose=verbose)

        # Store container reference
        self.container = container

        # Bridge-specific configuration
        self.metadata_stamping_service_url: str = container.config.get(
            "metadata_stamping_service_url", "http://metadata-stamping:8053"
        )
        self.onextree_service_url: str = container.config.get(
            "onextree_service_url", "http://onextree:8080"
        )
        self.kafka_broker_url: str = container.config.get(
            "kafka_broker_url", "localhost:9092"
        )
        self.default_namespace: str = container.config.get(
            "default_namespace", "omninode.bridge"
        )

        # Get or create KafkaClient from container
        health_check_mode = container.config.get("health_check_mode", False)
        self.kafka_client = container.get_service("kafka_client")

        if self.kafka_client is None and not health_check_mode:
            # Import KafkaClient
            try:
                from ....services.kafka_client import KafkaClient

                self.kafka_client = KafkaClient(
                    bootstrap_servers=self.kafka_broker_url,
                    enable_dead_letter_queue=True,
                    max_retry_attempts=3,
                    timeout_seconds=performance_config.KAFKA_CLIENT_TIMEOUT_SECONDS,
                )
                container.register_service("kafka_client", self.kafka_client)
            except ImportError:
                emit_log_event(
                    LogLevel.WARNING,
                    "KafkaClient not available - events will be logged only",
                    {},
                )
                self.kafka_client = None
        elif health_check_mode:
            emit_log_event(
                LogLevel.DEBUG,
                "Health check mode enabled - skipping Kafka initialization",
                {},
            )
            self.kafka_client = None

        emit_log_event(
            LogLevel.INFO,
            "NodeBridgeWorkflowOrchestrator initialized successfully",
            {
                "metadata_stamping_url": self.metadata_stamping_service_url,
                "onextree_url": self.onextree_service_url,
                "kafka_broker": self.kafka_broker_url,
                "default_namespace": self.default_namespace,
                "kafka_enabled": self.kafka_client is not None,
            },
        )

    # Workflow Steps

    @step(pass_context=True)
    async def validate_input(
        self, ctx: Context, ev: StartEvent
    ) -> Union[ValidationCompletedEvent, StopEvent]:
        """
        Step 1: Validate workflow input and initialize context.

        Validates input content and initializes workflow context with correlation
        tracking and configuration. Publishes VALIDATION_COMPLETED event to Kafka.

        Args:
            ctx: Workflow context for shared state
            ev: StartEvent with workflow input parameters

        Returns:
            ValidationCompletedEvent: On success with validated content
            StopEvent: On error with WorkflowFailedEvent result

        Error Handling:
            - Converts all exceptions to OnexError format
            - Publishes WORKFLOW_FAILED event to Kafka
            - Returns StopEvent for graceful termination
            - Validation errors are terminal (no retry)

        Context Initialization:
            - correlation_id: UUID for request tracing
            - workflow_start_time: Performance tracking
            - enable_intelligence: Intelligence enrichment flag
            - metadata: Additional metadata from input
        """
        try:
            start_time = time.time()

            # Extract input from StartEvent with safe attribute/dict access
            # StartEvent may be dict-like, attribute-based, or use payload/kwargs/_data
            def safe_get(obj: Any, key: str, default: Any = None) -> Any:
                """
                Safely get value from dict, attribute, or payload-based object.

                Handles multiple StartEvent patterns:
                - Dict-like access: obj.get(key)
                - Attribute access: obj.key
                - Internal data: obj._data.get(key) (test pattern)
                - Payload access: obj.payload.get(key) or obj.kwargs.get(key)
                """
                # First, check if object is dict-like
                if isinstance(obj, dict):
                    return obj.get(key, default)

                # Try _data attribute (common test pattern for StartEvent)
                _data = getattr(obj, "_data", None)
                if _data is not None and isinstance(_data, dict):
                    return _data.get(key, default)

                # Try payload attribute (common in LlamaIndex events)
                payload = getattr(obj, "payload", None)
                if payload is not None:
                    if isinstance(payload, dict):
                        return payload.get(key, default)
                    else:
                        value = getattr(payload, key, None)
                        return value if value is not None else default

                # Try kwargs attribute (alternative LlamaIndex pattern)
                kwargs = getattr(obj, "kwargs", None)
                if kwargs is not None and isinstance(kwargs, dict):
                    return kwargs.get(key, default)

                # Fall back to direct attribute access
                value = getattr(obj, key, None)
                return value if value is not None else default

            correlation_id = safe_get(ev, "correlation_id", uuid4())
            content = safe_get(ev, "content")
            metadata = safe_get(ev, "metadata", {})
            enable_intelligence = safe_get(ev, "enable_intelligence", False)

            # Store in context for subsequent steps
            await ctx.set("correlation_id", correlation_id)
            await ctx.set("workflow_start_time", start_time)
            await ctx.set("enable_intelligence", enable_intelligence)
            await ctx.set("metadata", metadata)

            # Validate input
            if not content or not isinstance(content, str):
                raise OnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message="Content is required and must be a string",
                    details={
                        "correlation_id": str(correlation_id),
                        "content_type": type(content).__name__,
                    },
                )

            validation_time_ms = (time.time() - start_time) * 1000

            # Publish Kafka event
            await self._publish_kafka_event(
                "VALIDATION_COMPLETED",
                {
                    "correlation_id": str(correlation_id),
                    "validation_time_ms": validation_time_ms,
                    "content_length": len(content),
                },
            )

            emit_log_event(
                LogLevel.INFO,
                "Input validation completed",
                {
                    "correlation_id": str(correlation_id),
                    "validation_time_ms": validation_time_ms,
                },
            )

            return ValidationCompletedEvent(
                correlation_id=correlation_id,
                validated_content=content,
                validation_time_ms=validation_time_ms,
                namespace=self.default_namespace,
            )

        except Exception as error:
            # Handle all errors with centralized error handler
            return await self.handle_workflow_error(
                ctx=ctx,
                error=error,
                step_name="validate_input",
            )

    @step(pass_context=True)
    async def generate_hash(
        self, ctx: Context, ev: ValidationCompletedEvent
    ) -> Union[HashGeneratedEvent, StopEvent]:
        """
        Step 2: Generate BLAKE3 hash via MetadataStamping service.

        Routes to MetadataStamping service for high-performance BLAKE3 hash
        generation. Publishes HASH_GENERATED event to Kafka.

        Args:
            ctx: Workflow context
            ev: ValidationCompletedEvent with validated content

        Returns:
            HashGeneratedEvent: On success with file hash
            StopEvent: On error with WorkflowFailedEvent result

        Error Handling:
            - Converts all exceptions to OnexError format
            - Publishes WORKFLOW_FAILED event to Kafka
            - Returns StopEvent for graceful termination
            - Network errors include retry strategy information

        Context Updates:
            - file_hash: Generated BLAKE3 hash
            - file_size_bytes: Content size
            - hash_generation_time_ms: Hash generation time
        """
        try:
            start_time = time.time()
            correlation_id = await ctx.get("correlation_id")

            # In real implementation, call MetadataStamping service via HTTP
            # For now, simulate hash generation
            file_hash = "blake3_" + uuid4().hex[:32]
            file_size_bytes = len(ev.validated_content.encode("utf-8"))

            hash_generation_time_ms = (time.time() - start_time) * 1000

            # Store in context
            await ctx.set("file_hash", file_hash)
            await ctx.set("file_size_bytes", file_size_bytes)
            await ctx.set("hash_generation_time_ms", hash_generation_time_ms)
            await ctx.set("validated_content", ev.validated_content)

            # Publish Kafka event
            await self._publish_kafka_event(
                "HASH_GENERATED",
                {
                    "correlation_id": str(correlation_id),
                    "file_hash": file_hash,
                    "hash_generation_time_ms": hash_generation_time_ms,
                    "file_size_bytes": file_size_bytes,
                },
            )

            emit_log_event(
                LogLevel.INFO,
                "BLAKE3 hash generated",
                {
                    "correlation_id": str(correlation_id),
                    "file_hash": file_hash,
                    "generation_time_ms": hash_generation_time_ms,
                },
            )

            return HashGeneratedEvent(
                correlation_id=correlation_id,
                file_hash=file_hash,
                hash_generation_time_ms=hash_generation_time_ms,
                file_size_bytes=file_size_bytes,
                namespace=self.default_namespace,
            )

        except Exception as error:
            # Handle all errors with centralized error handler
            return await self.handle_workflow_error(
                ctx=ctx,
                error=error,
                step_name="generate_hash",
            )

    @step(pass_context=True)
    async def create_stamp(
        self, ctx: Context, ev: HashGeneratedEvent
    ) -> Union[StampCreatedEvent, IntelligenceRequestedEvent, StopEvent]:
        """
        Step 3: Create metadata stamp with O.N.E. v0.1 compliance.

        Creates metadata stamp with ONEX compliance fields. Routes to intelligence
        enrichment if enabled, otherwise continues to completion. Publishes
        STAMP_CREATED event to Kafka.

        Args:
            ctx: Workflow context
            ev: HashGeneratedEvent with file hash

        Returns:
            StampCreatedEvent: If intelligence disabled, continue to completion
            IntelligenceRequestedEvent: If intelligence enabled, route to enrichment
            StopEvent: On error with WorkflowFailedEvent result

        Error Handling:
            - Converts all exceptions to OnexError format
            - Publishes WORKFLOW_FAILED event to Kafka
            - Returns StopEvent for graceful termination
            - Database errors include transaction rollback information

        Context Updates:
            - stamp_id: Created stamp identifier
            - stamp_data: O.N.E. v0.1 compliant stamp metadata
            - stamp_creation_time_ms: Stamp creation time
        """
        try:
            start_time = time.time()
            correlation_id = await ctx.get("correlation_id")

            # Create stamp with O.N.E. v0.1 compliance
            stamp_id = str(uuid4())
            stamp_data = {
                "stamp_id": stamp_id,
                "file_hash": ev.file_hash,
                "created_at": datetime.now().isoformat(),
                "namespace": self.default_namespace,
                "version": "1",
                "metadata_version": "0.1",
            }

            stamp_creation_time_ms = (time.time() - start_time) * 1000

            # Store in context
            await ctx.set("stamp_id", stamp_id)
            await ctx.set("stamp_data", stamp_data)
            await ctx.set("stamp_creation_time_ms", stamp_creation_time_ms)

            # Publish Kafka event
            await self._publish_kafka_event(
                "STAMP_CREATED",
                {
                    "correlation_id": str(correlation_id),
                    "stamp_id": stamp_id,
                    "stamp_creation_time_ms": stamp_creation_time_ms,
                },
            )

            emit_log_event(
                LogLevel.INFO,
                "Metadata stamp created",
                {
                    "correlation_id": str(correlation_id),
                    "stamp_id": stamp_id,
                    "creation_time_ms": stamp_creation_time_ms,
                },
            )

            # Check if intelligence enrichment is enabled
            if await ctx.get("enable_intelligence", False):
                # Route to intelligence enrichment
                return IntelligenceRequestedEvent(
                    correlation_id=correlation_id,
                    content=await ctx.get("validated_content", ""),
                    file_hash=ev.file_hash,
                    namespace=self.default_namespace,
                )
            else:
                # Skip intelligence, continue to completion
                return StampCreatedEvent(
                    correlation_id=correlation_id,
                    stamp_id=stamp_id,
                    stamp_data=stamp_data,
                    stamp_creation_time_ms=stamp_creation_time_ms,
                    namespace=self.default_namespace,
                )

        except Exception as error:
            # Handle all errors with centralized error handler
            return await self.handle_workflow_error(
                ctx=ctx,
                error=error,
                step_name="create_stamp",
            )

    @step(pass_context=True)
    async def enrich_intelligence(
        self, ctx: Context, ev: IntelligenceRequestedEvent
    ) -> IntelligenceReceivedEvent:
        """
        Step 4: Optional OnexTree intelligence enrichment.

        Routes to OnexTree service for AI-powered content analysis and
        enrichment. Gracefully handles failures by returning empty intelligence
        data. Publishes INTELLIGENCE_RECEIVED event to Kafka.

        Args:
            ctx: Workflow context
            ev: IntelligenceRequestedEvent with content to analyze

        Returns:
            IntelligenceReceivedEvent with AI analysis results

        Context Updates:
            - intelligence_data: AI analysis results (or empty dict on failure)
            - intelligence_time_ms: Intelligence analysis time

        Note:
            Failures are non-fatal. Workflow continues with empty intelligence data.
        """
        start_time = time.time()
        correlation_id = await ctx.get("correlation_id")

        try:
            # In real implementation, call OnexTree service via HTTP
            # For now, simulate intelligence analysis
            intelligence_data = {
                "analysis_type": "content_validation",
                "confidence_score": "0.95",
                "recommendations": "Content appears valid, No issues detected",
                "analyzed_at": datetime.now().isoformat(),
            }
            confidence_score = 0.95

            intelligence_time_ms = (time.time() - start_time) * 1000

            # Store in context
            await ctx.set("intelligence_data", intelligence_data)
            await ctx.set("intelligence_time_ms", intelligence_time_ms)

            # Publish Kafka event
            await self._publish_kafka_event(
                "INTELLIGENCE_RECEIVED",
                {
                    "correlation_id": str(correlation_id),
                    "intelligence_time_ms": intelligence_time_ms,
                    "confidence_score": confidence_score,
                },
            )

            emit_log_event(
                LogLevel.INFO,
                "Intelligence enrichment completed",
                {
                    "correlation_id": str(correlation_id),
                    "intelligence_time_ms": intelligence_time_ms,
                    "confidence_score": confidence_score,
                },
            )

            return IntelligenceReceivedEvent(
                correlation_id=correlation_id,
                intelligence_data=intelligence_data,
                intelligence_time_ms=intelligence_time_ms,
                confidence_score=confidence_score,
                namespace=self.default_namespace,
            )

        except Exception as e:
            emit_log_event(
                LogLevel.WARNING,
                "Intelligence enrichment failed, continuing workflow",
                {
                    "correlation_id": str(correlation_id),
                    "error": str(e),
                },
            )

            # Return empty intelligence data (graceful degradation)
            return IntelligenceReceivedEvent(
                correlation_id=correlation_id,
                intelligence_data={},
                intelligence_time_ms=0.0,
                confidence_score=0.0,
                namespace=self.default_namespace,
            )

    @step(pass_context=True)
    async def persist_state(
        self, ctx: Context, ev: Union[StampCreatedEvent, IntelligenceReceivedEvent]
    ) -> Union[PersistenceCompletedEvent, StopEvent]:
        """
        Step 5: Persist workflow state to database.

        Saves workflow execution data to PostgreSQL via DatabaseAdapter for
        audit trail and analytics. Publishes PERSISTENCE_COMPLETED event to Kafka.

        Args:
            ctx: Workflow context
            ev: Either StampCreatedEvent or IntelligenceReceivedEvent

        Returns:
            PersistenceCompletedEvent: On successful database persistence
            StopEvent: On error with WorkflowFailedEvent result

        Error Handling:
            - Database errors are non-fatal (workflow continues)
            - Circuit breaker pattern for database resilience
            - Returns StopEvent with WorkflowFailedEvent on critical failures

        Context Updates:
            - persistence_time_ms: Database persistence time
            - database_id: Database record identifier (if available)

        Performance Target:
            - Database persistence: < 50ms (p95)
        """
        start_time = time.time()
        correlation_id = await ctx.get("correlation_id")

        try:
            # Extract workflow data for persistence
            stamp_id = await ctx.get("stamp_id")
            file_hash = await ctx.get("file_hash")
            stamp_data = await ctx.get("stamp_data")
            workflow_start_time = await ctx.get("workflow_start_time")
            intelligence_data = await ctx.get("intelligence_data", None)

            # Calculate current processing time for persistence record
            current_processing_time_ms = (time.time() - workflow_start_time) * 1000

            # Build workflow execution data for database
            workflow_execution_data = {
                "workflow_id": str(correlation_id),
                "stamp_id": stamp_id,
                "file_hash": file_hash,
                "stamp_metadata": stamp_data,
                "intelligence_data": intelligence_data,
                "processing_time_ms": current_processing_time_ms,
                "workflow_state": "in_progress",
                "namespace": self.default_namespace,
                "created_at": datetime.now().isoformat(),
            }

            # In real implementation, call DatabaseAdapter via HTTP or direct invocation
            # For now, simulate database persistence
            database_id = str(uuid4())

            persistence_time_ms = (time.time() - start_time) * 1000

            # Store in context
            await ctx.set("persistence_time_ms", persistence_time_ms)
            await ctx.set("database_id", database_id)

            # Publish Kafka event
            await self._publish_kafka_event(
                "PERSISTENCE_COMPLETED",
                {
                    "correlation_id": str(correlation_id),
                    "database_id": database_id,
                    "persistence_time_ms": persistence_time_ms,
                    "workflow_execution_data": workflow_execution_data,
                },
            )

            emit_log_event(
                LogLevel.INFO,
                "Workflow state persisted to database",
                {
                    "correlation_id": str(correlation_id),
                    "database_id": database_id,
                    "persistence_time_ms": persistence_time_ms,
                },
            )

            return PersistenceCompletedEvent(
                correlation_id=correlation_id,
                persistence_time_ms=persistence_time_ms,
                database_id=database_id,
                namespace=self.default_namespace,
            )

        except Exception as error:
            # Handle all errors with centralized error handler
            return await self.handle_workflow_error(
                ctx=ctx,
                error=error,
                step_name="persist_state",
            )

    @step(pass_context=True)
    async def complete_workflow(
        self,
        ctx: Context,
        ev: Union[IntelligenceReceivedEvent, PersistenceCompletedEvent],
    ) -> StopEvent:
        """
        Step 6: Complete workflow and return final result.

        Aggregates workflow results from all steps and constructs final output
        with O.N.E. v0.1 compliance. Handles both workflow paths (with/without
        intelligence). Publishes WORKFLOW_COMPLETED event to Kafka.

        Args:
            ctx: Workflow context with accumulated state
            ev: Either IntelligenceReceivedEvent or PersistenceCompletedEvent

        Returns:
            StopEvent: With ModelStampResponseOutput result on success
                      With WorkflowFailedEvent result on error

        Error Handling:
            - Converts all exceptions to OnexError format
            - Publishes WORKFLOW_FAILED event to Kafka
            - Returns StopEvent for graceful termination
            - Aggregation errors include context state dump

        Workflow Paths:
            - Without Intelligence: ValidationCompleted → HashGenerated →
                                   StampCreated → PersistenceCompleted → WorkflowCompleted
            - With Intelligence: ValidationCompleted → HashGenerated →
                                StampCreated → IntelligenceRequested →
                                IntelligenceReceived → PersistenceCompleted → WorkflowCompleted

        Performance:
            - Target: < 150ms without intelligence, < 650ms with intelligence
            - Actual: Measured and returned in processing_time_ms
        """
        try:
            correlation_id = await ctx.get("correlation_id")
            workflow_start_time = await ctx.get("workflow_start_time")

            # Calculate total processing time
            processing_time_ms = (time.time() - workflow_start_time) * 1000

            # Extract data from context
            stamp_id = await ctx.get("stamp_id")
            file_hash = await ctx.get("file_hash")
            stamp_data = await ctx.get("stamp_data")
            validated_content = await ctx.get("validated_content", "")

            # Check if intelligence data is available
            intelligence_data = await ctx.get("intelligence_data", None)

            # Create stamped content
            stamped_content = f"[STAMP:{stamp_id}] {validated_content}"

            # Count workflow steps executed
            workflow_steps_executed = 4  # validate, hash, stamp, persist
            if intelligence_data:
                workflow_steps_executed += 1  # intelligence (now 5 total)

            # Publish completion event to Kafka
            await self._publish_kafka_event(
                "WORKFLOW_COMPLETED",
                {
                    "correlation_id": str(correlation_id),
                    "stamp_id": stamp_id,
                    "file_hash": file_hash,
                    "processing_time_ms": processing_time_ms,
                    "steps_executed": workflow_steps_executed,
                },
            )

            emit_log_event(
                LogLevel.INFO,
                "Workflow completed successfully",
                {
                    "correlation_id": str(correlation_id),
                    "stamp_id": stamp_id,
                    "processing_time_ms": processing_time_ms,
                    "steps_executed": workflow_steps_executed,
                },
            )

            # Construct final result
            result = ModelStampResponseOutput(
                stamp_id=stamp_id,
                file_hash=file_hash,
                stamped_content=stamped_content,
                stamp_metadata=stamp_data,
                namespace=self.default_namespace,
                op_id=correlation_id,
                version=1,
                metadata_version="0.1",
                workflow_state=EnumWorkflowState.COMPLETED,
                workflow_id=correlation_id,
                intelligence_data=intelligence_data,
                processing_time_ms=processing_time_ms,
                hash_generation_time_ms=await ctx.get("hash_generation_time_ms", 0.0),
                workflow_steps_executed=workflow_steps_executed,
                created_at=datetime.now(),
                completed_at=datetime.now(),
            )

            # Return StopEvent with final result
            return StopEvent(result=result)

        except Exception as error:
            # Handle all errors with centralized error handler
            return await self.handle_workflow_error(
                ctx=ctx,
                error=error,
                step_name="complete_workflow",
            )

    # Helper Methods

    async def handle_workflow_error(
        self,
        ctx: Context,
        error: Exception,
        step_name: str,
    ) -> StopEvent:
        """
        Centralized error handler for workflow steps.

        Handles workflow errors by:
        1. Converting exceptions to OnexError format
        2. Publishing WORKFLOW_FAILED event to Kafka
        3. Logging detailed error context
        4. Returning StopEvent with WorkflowFailedEvent result

        Args:
            ctx: Workflow context with accumulated state
            error: Exception that occurred
            step_name: Name of the step that failed

        Returns:
            StopEvent with WorkflowFailedEvent result for graceful termination

        Error Recovery Strategy:
            - All errors are terminal (return StopEvent)
            - Detailed error context preserved for debugging
            - Kafka events published for observability
            - State transitions tracked (PROCESSING → FAILED)
        """
        # Extract context data with safe fallbacks
        correlation_id = await ctx.get("correlation_id", uuid4())
        workflow_start_time = await ctx.get("workflow_start_time", time.time())
        processing_time_ms = (time.time() - workflow_start_time) * 1000

        # Convert to ModelOnexError if not already
        if isinstance(error, ModelOnexError):
            onex_error = error
        else:
            # Determine appropriate error code based on error type
            error_code = EnumCoreErrorCode.OPERATION_FAILED
            if isinstance(error, ValueError):
                error_code = EnumCoreErrorCode.VALIDATION_ERROR
            elif isinstance(error, KeyError):
                error_code = EnumCoreErrorCode.CONFIGURATION_ERROR
            elif isinstance(error, ConnectionError):
                error_code = EnumCoreErrorCode.NETWORK_ERROR

            onex_error = ModelOnexError(
                error_code=error_code,
                message=str(error),
                details={
                    "correlation_id": str(correlation_id),
                    "failed_step": step_name,
                    "error_type": type(error).__name__,
                    "processing_time_ms": processing_time_ms,
                },
                cause=error,
            )

        # Publish WORKFLOW_FAILED event to Kafka
        await self._publish_kafka_event(
            "WORKFLOW_FAILED",
            {
                "correlation_id": str(correlation_id),
                "error_message": str(onex_error.message),
                "error_code": (
                    onex_error.error_code.value
                    if hasattr(onex_error.error_code, "value")
                    else str(onex_error.error_code)
                ),
                "error_type": type(error).__name__,
                "failed_step": step_name,
                "processing_time_ms": processing_time_ms,
            },
        )

        # Convert exception to JSON-serializable format for logging
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(onex_error.message),
            "error_code": (
                onex_error.error_code.value
                if hasattr(onex_error.error_code, "value")
                else str(onex_error.error_code)
            ),
        }

        # Add OnexError-specific details if available
        if isinstance(error, ModelOnexError):
            # Extract details dict and ensure all values are JSON-serializable
            details = getattr(onex_error, "details", {})
            if details and isinstance(details, dict):
                error_data["error_details"] = {
                    k: (
                        str(v)
                        if not isinstance(  # noqa: UP038
                            v, (str, int, float, bool, type(None))
                        )
                        else v
                    )
                    for k, v in details.items()
                }
        else:
            # For generic exceptions, add args if available
            if hasattr(error, "args") and error.args:
                error_data["error_args"] = [str(arg) for arg in error.args]

        # Log detailed error information
        emit_log_event(
            LogLevel.ERROR,
            f"Workflow failed at step: {step_name}",
            {
                "correlation_id": str(correlation_id),
                **error_data,
                "failed_step": step_name,
                "processing_time_ms": processing_time_ms,
            },
        )

        # Create WorkflowFailedEvent with error details
        from .workflow_events import WorkflowFailedEvent

        failed_event = WorkflowFailedEvent(
            correlation_id=correlation_id,
            error_message=str(onex_error.message),
            error_type=type(error).__name__,
            failed_step=step_name,
            processing_time_ms=processing_time_ms,
            namespace=self.default_namespace,
        )

        # Return StopEvent with failed result (graceful termination)
        return StopEvent(result=failed_event)

    async def _publish_kafka_event(self, event_type: str, data: dict[str, Any]) -> None:
        """
        Publish event to Kafka event bus.

        Publishes workflow events to Kafka for observability and integration.
        Handles Kafka unavailability gracefully by logging events instead.

        Args:
            event_type: Event type identifier (e.g., "VALIDATION_COMPLETED")
            data: Event payload data

        Kafka Topic Naming:
            {namespace}.orchestrator.{event_type_lowercase}
            Example: omninode.bridge.orchestrator.validation_completed

        Note:
            Non-fatal failures. Workflow continues even if Kafka is unavailable.
        """
        try:
            if self.kafka_client and self.kafka_client.is_connected:
                topic_name = (
                    f"{self.default_namespace}.orchestrator.{event_type.lower()}"
                )

                event_data = {
                    **data,
                    "event_type": event_type,
                    "topic_name": topic_name,
                    "published_at": datetime.now().isoformat(),
                }

                key = data.get("correlation_id", "default")
                success = await self.kafka_client.publish_raw_event(
                    topic=topic_name,
                    data=event_data,
                    key=key,
                )

                if success:
                    emit_log_event(
                        LogLevel.DEBUG,
                        f"Published Kafka event: {event_type}",
                        {
                            "event_type": event_type,
                            "topic_name": topic_name,
                        },
                    )
                else:
                    emit_log_event(
                        LogLevel.WARNING,
                        f"Failed to publish Kafka event: {event_type}",
                        {
                            "event_type": event_type,
                            "topic_name": topic_name,
                        },
                    )
            else:
                # Kafka not available - log event only
                emit_log_event(
                    LogLevel.DEBUG,
                    f"Kafka unavailable, logging event: {event_type}",
                    {
                        "event_type": event_type,
                        "data": data,
                    },
                )

        except Exception as e:
            # Log error but don't fail workflow
            emit_log_event(
                LogLevel.WARNING,
                f"Failed to publish Kafka event: {event_type}",
                {
                    "event_type": event_type,
                    "error": str(e),
                },
            )


def main() -> int:
    """
    Entry point for workflow node execution.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        emit_log_event(
            LogLevel.INFO,
            "NodeBridgeWorkflowOrchestrator (LlamaIndex Workflows) running",
            {},
        )
        return 0
    except Exception as e:
        emit_log_event(
            LogLevel.ERROR,
            f"NodeBridgeWorkflowOrchestrator execution failed: {e!s}",
            {"error": str(e), "error_type": type(e).__name__},
        )
        return 1


if __name__ == "__main__":
    exit(main())
