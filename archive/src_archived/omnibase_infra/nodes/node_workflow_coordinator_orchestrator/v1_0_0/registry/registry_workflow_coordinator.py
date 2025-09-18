"""Registry for workflow coordinator orchestrator dependency injection."""

from typing import Any, Protocol

from omnibase_core.core.onex_container import ModelONEXContainer


class ProtocolWorkflowCoordinatorRegistry(Protocol):
    """Protocol for workflow coordinator registry dependency injection."""

    def register_dependencies(self, container: ModelONEXContainer) -> None:
        """Register workflow coordinator dependencies in the ONEX container.

        Args:
            container: ONEX container for dependency injection
        """
        ...

    def get_configuration(self) -> dict[str, Any]:
        """Get workflow coordinator configuration.

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        ...


class RegistryWorkflowCoordinator:
    """Registry for workflow coordinator orchestrator dependencies."""

    def __init__(self, container: ModelONEXContainer):
        """Initialize the workflow coordinator registry.

        Args:
            container: ONEX container for dependency injection
        """
        self.container = container
        self._config = self._load_default_config()

    def register_dependencies(self, container: ModelONEXContainer) -> None:
        """Register workflow coordinator dependencies in the ONEX container.

        Args:
            container: ONEX container for dependency injection
        """
        # Register workflow coordination protocols and services
        # This follows the protocol-based dependency injection pattern

    def get_configuration(self) -> dict[str, Any]:
        """Get workflow coordinator configuration.

        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return self._config

    def _load_default_config(self) -> dict[str, Any]:
        """Load default configuration for workflow coordinator.

        Returns:
            Dict[str, Any]: Default configuration
        """
        return {
            "max_concurrent_workflows": 10,
            "default_workflow_timeout_seconds": 300,
            "agent_coordination_timeout_seconds": 60,
            "background_task_queue_size": 100,
            "progress_update_interval_seconds": 5,
            "metrics_collection_enabled": True,
            "health_check_interval_seconds": 30,
            "retry_attempts": 3,
            "circuit_breaker_enabled": True,
            "performance_monitoring_enabled": True,
        }
