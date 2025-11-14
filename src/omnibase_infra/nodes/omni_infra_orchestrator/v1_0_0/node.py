"""NodeOmniInfraOrchestrator - Infrastructure Workflow Orchestration with LlamaIndex.

This orchestrator coordinates infrastructure operations using LlamaIndex workflows.
All workflows are declared in the contract and executed through LlamaIndex engine.

Communication Pattern:
Reducer → Intents → Orchestrator → LlamaIndex Workflows → Adapters
"""

import time
from uuid import UUID

from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError
from omnibase_core.core.node_orchestrator_service import NodeOrchestratorService
from omnibase_core.core.onex_container import ModelONEXContainer

from .models.model_omni_infra_orchestrator_input import ModelOmniInfraOrchestratorInput
from .models.model_omni_infra_orchestrator_output import ModelOmniInfraOrchestratorOutput
from .workflows.health_check_workflow import HealthCheckOrchestrationWorkflow
from .workflows.failover_workflow import FailoverCoordinationWorkflow
from .workflows.initialization_workflow import InitializationSequenceWorkflow
from .workflows.intent_processing_workflow import IntentProcessingWorkflow


class NodeOmniInfraOrchestrator(NodeOrchestratorService):
    """
    Infrastructure Workflow Orchestrator using LlamaIndex.

    Responsibilities:
    1. Consume intents from reducer
    2. Route intents to appropriate LlamaIndex workflows
    3. Execute workflow steps as declared in contract
    4. Coordinate infrastructure adapters
    5. Monitor workflow execution and results

    Workflows (all LlamaIndex-based):
    - health_check_workflow: Coordinate health checks
    - failover_workflow: Coordinate failover and recovery
    - initialization_workflow: Orchestrate startup sequence
    - intent_processing_workflow: Route intents to workflows
    """

    def __init__(self, container: ModelONEXContainer):
        """Initialize orchestrator with LlamaIndex workflows."""
        super().__init__(container)
        self.node_type = "orchestrator"
        self.domain = "infrastructure"

        # Get event bus for intent consumption
        self._event_bus = self.container.get_service("ProtocolEventBus")
        if self._event_bus is None:
            raise OnexError(
                code=CoreErrorCode.DEPENDENCY_RESOLUTION_ERROR,
                message="ProtocolEventBus required for intent consumption",
            )

        # Get reducer for state queries
        self._reducer = self.container.get_service("omni_infra_reducer")

        # Get infrastructure adapters
        self._postgres_adapter = self.container.get_service("postgres_adapter")
        self._kafka_adapter = self.container.get_service("kafka_adapter")
        self._consul_adapter = self.container.get_service("consul_adapter")

        # Initialize LlamaIndex workflows (declared in contract)
        self._workflows = {
            "health_check_workflow": HealthCheckOrchestrationWorkflow(),
            "failover_workflow": FailoverCoordinationWorkflow(),
            "initialization_workflow": InitializationSequenceWorkflow(),
            "intent_processing_workflow": IntentProcessingWorkflow(),
        }

    async def orchestrate(
        self, input_data: ModelOmniInfraOrchestratorInput
    ) -> ModelOmniInfraOrchestratorOutput:
        """
        Main orchestration method - routes to appropriate workflow.

        Process:
        1. Determine operation type (process_intent | trigger_workflow | query_status)
        2. Route to appropriate LlamaIndex workflow
        3. Execute workflow steps as declared in contract
        4. Return workflow execution results
        """
        start_time = time.perf_counter()

        try:
            if input_data.operation_type == "process_intent":
                result = await self._process_intent(input_data)

            elif input_data.operation_type == "trigger_workflow":
                result = await self._trigger_workflow(input_data)

            elif input_data.operation_type == "query_workflow_status":
                result = await self._query_workflow_status(input_data)

            else:
                raise OnexError(
                    code=CoreErrorCode.VALIDATION_ERROR,
                    message=f"Unsupported operation type: {input_data.operation_type}",
                )

            execution_time_ms = (time.perf_counter() - start_time) * 1000

            return ModelOmniInfraOrchestratorOutput(
                workflow_executed=result["workflow_executed"],
                workflow_name=result["workflow_name"],
                workflow_result=result.get("workflow_result", {}),
                correlation_id=input_data.correlation_id,
                timestamp=time.time(),
            )

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000

            raise OnexError(
                code=CoreErrorCode.PROCESSING_ERROR,
                message=f"Orchestration failed: {str(e)}",
                details={
                    "operation_type": input_data.operation_type,
                    "execution_time_ms": execution_time_ms,
                },
            ) from e

    async def _process_intent(self, input_data: ModelOmniInfraOrchestratorInput) -> dict:
        """
        Process intent from reducer using intent_processing_workflow.

        The workflow routes the intent to the appropriate handler workflow
        based on the intent routing table declared in contract.
        """
        workflow = self._workflows["intent_processing_workflow"]

        # Execute LlamaIndex workflow
        result = await workflow.run(
            orchestrator=self,
            intent_type=input_data.intent_type,
            intent_data=input_data.workflow_params,
        )

        return {
            "workflow_executed": True,
            "workflow_name": "intent_processing_workflow",
            "workflow_result": result,
        }

    async def _trigger_workflow(self, input_data: ModelOmniInfraOrchestratorInput) -> dict:
        """
        Directly trigger a specific workflow by name.

        Allows direct invocation of workflows without going through
        intent processing (useful for testing and manual triggers).
        """
        workflow_name = input_data.workflow_name

        if workflow_name not in self._workflows:
            return {
                "workflow_executed": False,
                "workflow_name": workflow_name,
                "error": f"Unknown workflow: {workflow_name}",
            }

        workflow = self._workflows[workflow_name]

        # Execute LlamaIndex workflow with params
        result = await workflow.run(
            orchestrator=self,
            **input_data.workflow_params,
        )

        return {
            "workflow_executed": True,
            "workflow_name": workflow_name,
            "workflow_result": result,
        }

    async def _query_workflow_status(self, input_data: ModelOmniInfraOrchestratorInput) -> dict:
        """
        Query status of running or completed workflows.

        Returns workflow execution history and current states.
        """
        # TODO: Implement workflow status tracking
        return {
            "workflow_executed": False,
            "workflow_name": "status_query",
            "workflow_result": {
                "message": "Workflow status tracking not yet implemented",
            },
        }

    def _get_adapter_by_name(self, adapter_name: str):
        """Get adapter instance by name."""
        adapters = {
            "postgres": self._postgres_adapter,
            "kafka": self._kafka_adapter,
            "consul": self._consul_adapter,
        }
        return adapters.get(adapter_name)

    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        return time.time()

    async def initialize(self) -> None:
        """
        Initialize orchestrator and all infrastructure adapters.

        Uses initialization_workflow to coordinate startup sequence.
        """
        workflow = self._workflows["initialization_workflow"]

        # Execute initialization workflow
        result = await workflow.run(orchestrator=self)

        if not result.get("all_adapters_verified"):
            raise OnexError(
                code=CoreErrorCode.INITIALIZATION_FAILED,
                message="Infrastructure initialization failed",
                details=result,
            )

    async def cleanup(self) -> None:
        """Cleanup orchestrator resources."""
        # Cleanup workflows if needed
        pass

    async def health_check(self) -> dict:
        """
        Execute health check workflow.

        Returns comprehensive health status of all infrastructure.
        """
        workflow = self._workflows["health_check_workflow"]

        result = await workflow.run(orchestrator=self)

        return result
