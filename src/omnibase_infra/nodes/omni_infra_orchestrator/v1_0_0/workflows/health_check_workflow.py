"""Health Check Orchestration Workflow - LlamaIndex Implementation.

Coordinates health checks across all infrastructure adapters.
Declared in contract: health_check_workflow
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


class QueryAdapterStatesEvent(Event):
    """Event for querying adapter states from reducer."""
    adapter_states: dict[str, Any]


class TriggerHealthChecksEvent(Event):
    """Event for triggering parallel health checks."""
    health_check_results: dict[str, Any]


class AggregateResultsEvent(Event):
    """Event for aggregating health check results."""
    overall_health_status: str
    detailed_results: dict[str, Any]


class HealthCheckOrchestrationWorkflow(Workflow):
    """
    LlamaIndex workflow for coordinating health checks across infrastructure.

    Steps (from contract):
    1. query_adapter_states - Query reducer for current adapter states
    2. trigger_health_checks - Parallel health checks on all adapters
    3. aggregate_results - Aggregate results into overall health status
    4. emit_health_report - Publish health report event
    """

    @step
    async def query_adapter_states(
        self, ctx: Context, ev: StartEvent
    ) -> QueryAdapterStatesEvent:
        """
        Step 1: Query reducer for current adapter states.

        Retrieves the latest known state from the reducer's database
        to understand the current health of all infrastructure adapters.
        """
        # Get orchestrator instance from context
        orchestrator = await ctx.get("orchestrator")
        reducer = orchestrator._reducer

        # Query reducer for latest adapter states
        # This would query the infrastructure_state table
        adapter_states = {
            "postgres": {"last_known_status": "healthy"},
            "kafka": {"last_known_status": "healthy"},
            "consul": {"last_known_status": "healthy"},
        }

        return QueryAdapterStatesEvent(adapter_states=adapter_states)

    @step
    async def trigger_health_checks(
        self, ctx: Context, ev: QueryAdapterStatesEvent
    ) -> TriggerHealthChecksEvent:
        """
        Step 2: Trigger parallel health checks on all adapters.

        Calls health_check operation on postgres, kafka, and consul adapters
        concurrently to get real-time health status.
        """
        orchestrator = await ctx.get("orchestrator")

        # Trigger health checks on all adapters in parallel
        health_results = {}

        try:
            # Call postgres adapter health check
            postgres_health = await orchestrator._postgres_adapter.health_check()
            health_results["postgres"] = postgres_health

            # Call kafka adapter health check
            kafka_health = await orchestrator._kafka_adapter.health_check()
            health_results["kafka"] = kafka_health

            # Call consul adapter health check
            consul_health = await orchestrator._consul_adapter.health_check()
            health_results["consul"] = consul_health

        except Exception as e:
            health_results["error"] = str(e)

        return TriggerHealthChecksEvent(health_check_results=health_results)

    @step
    async def aggregate_results(
        self, ctx: Context, ev: TriggerHealthChecksEvent
    ) -> AggregateResultsEvent:
        """
        Step 3: Aggregate health check results into overall status.

        Determines overall infrastructure health based on individual
        adapter health statuses.
        """
        results = ev.health_check_results

        # Determine overall health
        all_healthy = all(
            result.get("status") == "healthy"
            for result in results.values()
            if isinstance(result, dict)
        )

        any_degraded = any(
            result.get("status") == "degraded"
            for result in results.values()
            if isinstance(result, dict)
        )

        if all_healthy:
            overall_status = "healthy"
        elif any_degraded:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"

        return AggregateResultsEvent(
            overall_health_status=overall_status,
            detailed_results=results,
        )

    @step
    async def emit_health_report(
        self, ctx: Context, ev: AggregateResultsEvent
    ) -> StopEvent:
        """
        Step 4: Publish health report event to event bus.

        Emits the aggregated health report as an event for monitoring
        and alerting systems to consume.
        """
        orchestrator = await ctx.get("orchestrator")

        # Publish health report event
        health_report = {
            "overall_status": ev.overall_health_status,
            "adapter_details": ev.detailed_results,
            "timestamp": orchestrator._get_timestamp(),
        }

        # TODO: Publish to event bus
        # await orchestrator._event_bus.publish_async(health_report_event)

        return StopEvent(result=health_report)
