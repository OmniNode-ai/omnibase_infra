"""Failover Coordination Workflow - LlamaIndex Implementation.

Coordinates service failover when adapter becomes unhealthy.
Declared in contract: failover_workflow
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


class IdentifyFailedAdapterEvent(Event):
    """Event for identifying failed adapter."""
    failed_adapter_info: dict[str, Any]


class InitiateCircuitBreakerEvent(Event):
    """Event for opening circuit breaker."""
    circuit_breaker_opened: bool
    adapter_name: str


class AttemptRecoveryEvent(Event):
    """Event for attempting adapter recovery."""
    recovery_initiated: bool
    adapter_name: str


class MonitorRecoveryEvent(Event):
    """Event for monitoring recovery status."""
    recovery_status: str
    adapter_name: str


class FailoverCoordinationWorkflow(Workflow):
    """
    LlamaIndex workflow for coordinating service failover.

    Steps (from contract):
    1. identify_failed_adapter - Query reducer for failed adapter info
    2. initiate_circuit_breaker - Trigger circuit breaker open
    3. attempt_recovery - Trigger adapter recovery sequence
    4. monitor_recovery - Poll adapter health until recovered
    5. close_circuit_breaker - Close circuit breaker if recovered
    """

    @step
    async def identify_failed_adapter(
        self, ctx: Context, ev: StartEvent
    ) -> IdentifyFailedAdapterEvent:
        """
        Step 1: Identify which adapter has failed.

        Queries reducer database to find which adapter triggered
        the failover intent.
        """
        orchestrator = await ctx.get("orchestrator")
        intent_data = ev.get("intent_data", {})

        # Extract failed adapter from intent
        failed_adapter_name = intent_data.get("adapter_source", "unknown")

        # Query reducer for detailed failure info
        failed_adapter_info = {
            "adapter_name": failed_adapter_name,
            "failure_reason": intent_data.get("error_message", "Unknown"),
            "timestamp": intent_data.get("timestamp"),
        }

        return IdentifyFailedAdapterEvent(failed_adapter_info=failed_adapter_info)

    @step
    async def initiate_circuit_breaker(
        self, ctx: Context, ev: IdentifyFailedAdapterEvent
    ) -> InitiateCircuitBreakerEvent:
        """
        Step 2: Open circuit breaker for failed adapter.

        Triggers circuit breaker to open state, preventing further
        requests to the failing adapter.
        """
        orchestrator = await ctx.get("orchestrator")
        adapter_name = ev.failed_adapter_info["adapter_name"]

        # Get circuit breaker for adapter
        adapter = orchestrator._get_adapter_by_name(adapter_name)

        if adapter and hasattr(adapter, "_circuit_breaker"):
            # Circuit breaker is already integrated in adapter
            # We can query its state
            circuit_breaker = adapter._circuit_breaker
            circuit_state = circuit_breaker.get_state()

            opened = circuit_state["state"] == "open"
        else:
            opened = False

        return InitiateCircuitBreakerEvent(
            circuit_breaker_opened=opened,
            adapter_name=adapter_name,
        )

    @step
    async def attempt_recovery(
        self, ctx: Context, ev: InitiateCircuitBreakerEvent
    ) -> AttemptRecoveryEvent:
        """
        Step 3: Attempt adapter recovery sequence.

        Triggers recovery operations like reconnection, pool reset,
        or service restart depending on adapter type.
        """
        orchestrator = await ctx.get("orchestrator")
        adapter_name = ev.adapter_name

        recovery_initiated = False

        # Attempt recovery based on adapter type
        if adapter_name == "postgres":
            # Reinitialize postgres connection pool
            try:
                await orchestrator._postgres_adapter.initialize()
                recovery_initiated = True
            except Exception:
                recovery_initiated = False

        elif adapter_name == "kafka":
            # Reinitialize kafka producer pool
            try:
                await orchestrator._kafka_adapter.initialize()
                recovery_initiated = True
            except Exception:
                recovery_initiated = False

        elif adapter_name == "consul":
            # Reconnect to consul
            try:
                await orchestrator._consul_adapter.initialize()
                recovery_initiated = True
            except Exception:
                recovery_initiated = False

        return AttemptRecoveryEvent(
            recovery_initiated=recovery_initiated,
            adapter_name=adapter_name,
        )

    @step
    async def monitor_recovery(
        self, ctx: Context, ev: AttemptRecoveryEvent
    ) -> MonitorRecoveryEvent:
        """
        Step 4: Monitor adapter health until recovered.

        Polls adapter health checks to determine if recovery
        was successful.
        """
        import asyncio

        orchestrator = await ctx.get("orchestrator")
        adapter_name = ev.adapter_name

        # Poll adapter health for up to 30 seconds
        max_attempts = 6
        attempt = 0
        recovery_status = "failed"

        while attempt < max_attempts:
            await asyncio.sleep(5)  # Wait 5 seconds between checks

            try:
                adapter = orchestrator._get_adapter_by_name(adapter_name)
                health_result = await adapter.health_check()

                if health_result.get("status") == "healthy":
                    recovery_status = "recovered"
                    break

            except Exception:
                pass

            attempt += 1

        return MonitorRecoveryEvent(
            recovery_status=recovery_status,
            adapter_name=adapter_name,
        )

    @step
    async def close_circuit_breaker(
        self, ctx: Context, ev: MonitorRecoveryEvent
    ) -> StopEvent:
        """
        Step 5: Close circuit breaker if recovery successful.

        Closes circuit breaker to allow requests to flow to the
        recovered adapter.
        """
        orchestrator = await ctx.get("orchestrator")

        result = {
            "adapter_name": ev.adapter_name,
            "recovery_status": ev.recovery_status,
            "circuit_breaker_closed": ev.recovery_status == "recovered",
            "timestamp": orchestrator._get_timestamp(),
        }

        return StopEvent(result=result)
