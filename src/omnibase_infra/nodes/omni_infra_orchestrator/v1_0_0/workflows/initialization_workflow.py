"""Initialization Sequence Workflow - LlamaIndex Implementation.

Orchestrates initialization of all infrastructure adapters in sequence.
Declared in contract: initialization_workflow
"""

from typing import Any

from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)


class InitializePostgresEvent(Event):
    """Event for postgres initialization."""
    postgres_initialized: bool


class InitializeKafkaEvent(Event):
    """Event for kafka initialization."""
    kafka_initialized: bool


class InitializeConsulEvent(Event):
    """Event for consul initialization."""
    consul_initialized: bool


class VerifyAllAdaptersEvent(Event):
    """Event for verifying all adapters."""
    all_adapters_verified: bool
    health_status: dict[str, Any]


class InitializationSequenceWorkflow(Workflow):
    """
    LlamaIndex workflow for orchestrating infrastructure initialization.

    Steps (from contract):
    1. initialize_postgres - Initialize PostgreSQL adapter first
    2. initialize_kafka - Initialize Kafka after postgres
    3. initialize_consul - Initialize Consul after kafka
    4. verify_all_adapters - Run health checks on all adapters
    """

    @step
    async def initialize_postgres(
        self, ctx: Context, ev: StartEvent
    ) -> InitializePostgresEvent:
        """
        Step 1: Initialize PostgreSQL adapter.

        PostgreSQL must be initialized first as it's required by
        the reducer for state storage.
        """
        orchestrator = await ctx.get("orchestrator")

        try:
            await orchestrator._postgres_adapter.initialize()
            postgres_initialized = True
        except Exception as e:
            postgres_initialized = False
            await ctx.set("postgres_error", str(e))

        return InitializePostgresEvent(postgres_initialized=postgres_initialized)

    @step
    async def initialize_kafka(
        self, ctx: Context, ev: InitializePostgresEvent
    ) -> InitializeKafkaEvent:
        """
        Step 2: Initialize Kafka adapter.

        Kafka is initialized after postgres for event publishing.
        """
        if not ev.postgres_initialized:
            # Skip if postgres failed
            return InitializeKafkaEvent(kafka_initialized=False)

        orchestrator = await ctx.get("orchestrator")

        try:
            await orchestrator._kafka_adapter.initialize()
            kafka_initialized = True
        except Exception as e:
            kafka_initialized = False
            await ctx.set("kafka_error", str(e))

        return InitializeKafkaEvent(kafka_initialized=kafka_initialized)

    @step
    async def initialize_consul(
        self, ctx: Context, ev: InitializeKafkaEvent
    ) -> InitializeConsulEvent:
        """
        Step 3: Initialize Consul adapter.

        Consul is initialized last for service discovery.
        """
        if not ev.kafka_initialized:
            # Skip if kafka failed
            return InitializeConsulEvent(consul_initialized=False)

        orchestrator = await ctx.get("orchestrator")

        try:
            await orchestrator._consul_adapter.initialize()
            consul_initialized = True
        except Exception as e:
            consul_initialized = False
            await ctx.set("consul_error", str(e))

        return InitializeConsulEvent(consul_initialized=consul_initialized)

    @step
    async def verify_all_adapters(
        self, ctx: Context, ev: InitializeConsulEvent
    ) -> StopEvent:
        """
        Step 4: Verify all adapters via health check workflow.

        Runs comprehensive health checks on all initialized adapters
        to ensure they're operational.
        """
        orchestrator = await ctx.get("orchestrator")

        # Run health check workflow to verify
        from .health_check_workflow import HealthCheckOrchestrationWorkflow

        health_workflow = HealthCheckOrchestrationWorkflow()

        # Execute health check workflow
        try:
            health_result = await health_workflow.run(orchestrator=orchestrator)
            all_verified = health_result.get("overall_status") == "healthy"
        except Exception as e:
            all_verified = False
            health_result = {"error": str(e)}

        result = {
            "postgres_initialized": await ctx.get("postgres_initialized", False),
            "kafka_initialized": await ctx.get("kafka_initialized", False),
            "consul_initialized": await ctx.get("consul_initialized", False),
            "all_adapters_verified": all_verified,
            "health_status": health_result,
            "timestamp": orchestrator._get_timestamp(),
        }

        return StopEvent(result=result)
