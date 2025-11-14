"""Intent Processing Workflow - LlamaIndex Implementation.

Routes intents from reducer to appropriate workflows.
Declared in contract: intent_processing_workflow
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


class ParseIntentEvent(Event):
    """Event for parsed intent."""
    intent_details: dict[str, Any]


class RouteIntentEvent(Event):
    """Event for routed intent."""
    workflow_triggered: bool
    workflow_name: str
    workflow_result: dict[str, Any]


class IntentProcessingWorkflow(Workflow):
    """
    LlamaIndex workflow for processing intents from reducer.

    Steps (from contract):
    1. parse_intent - Extract intent type and data
    2. route_intent - Route to appropriate workflow based on intent type
    """

    @step
    async def parse_intent(
        self, ctx: Context, ev: StartEvent
    ) -> ParseIntentEvent:
        """
        Step 1: Parse intent from reducer.

        Extracts intent type, source adapter, and related data
        from the intent payload.
        """
        intent_type = ev.get("intent_type")
        intent_data = ev.get("intent_data", {})

        intent_details = {
            "intent_type": intent_type,
            "source_adapter": intent_data.get("source_adapter"),
            "state_id": intent_data.get("state_id"),
            "correlation_id": intent_data.get("correlation_id"),
            "timestamp": intent_data.get("timestamp"),
        }

        return ParseIntentEvent(intent_details=intent_details)

    @step
    async def route_intent(
        self, ctx: Context, ev: ParseIntentEvent
    ) -> StopEvent:
        """
        Step 2: Route intent to appropriate workflow.

        Uses intent routing table from contract to determine which
        workflow should handle this intent.
        """
        orchestrator = await ctx.get("orchestrator")
        intent_type = ev.intent_details["intent_type"]

        # Intent routing table (from contract)
        routing_table = {
            "infrastructure_health_degraded": "health_check_workflow",
            "infrastructure_health_critical": "health_check_workflow",
            "circuit_breaker_opened": "failover_workflow",
            "failover_required": "failover_workflow",
            "connection_pool_exhausted": "health_check_workflow",
            "recovery_initiated": "failover_workflow",
        }

        workflow_name = routing_table.get(intent_type)

        if not workflow_name:
            return StopEvent(
                result={
                    "workflow_triggered": False,
                    "error": f"No workflow mapped for intent: {intent_type}",
                }
            )

        # Execute appropriate workflow
        workflow_result = {}
        workflow_triggered = False

        try:
            if workflow_name == "health_check_workflow":
                from .health_check_workflow import HealthCheckOrchestrationWorkflow

                workflow = HealthCheckOrchestrationWorkflow()
                workflow_result = await workflow.run(orchestrator=orchestrator)
                workflow_triggered = True

            elif workflow_name == "failover_workflow":
                from .failover_workflow import FailoverCoordinationWorkflow

                workflow = FailoverCoordinationWorkflow()
                workflow_result = await workflow.run(
                    orchestrator=orchestrator,
                    intent_data=ev.intent_details,
                )
                workflow_triggered = True

        except Exception as e:
            workflow_result = {"error": str(e)}
            workflow_triggered = False

        return StopEvent(
            result={
                "workflow_triggered": workflow_triggered,
                "workflow_name": workflow_name,
                "workflow_result": workflow_result,
                "intent_type": intent_type,
            }
        )
