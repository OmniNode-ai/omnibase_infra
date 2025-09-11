"""PostgreSQL Adapter Registry for dependency injection."""

from typing import Any, Dict, Optional

from omnibase_core.core.spi_service_registry import SPIServiceRegistry

from omnibase_infra.infrastructure.postgres_connection_manager import (
    PostgresConnectionManager,
    ConnectionConfig,
)


class PostgresAdapterRegistry(SPIServiceRegistry):
    """Registry for PostgreSQL adapter dependencies."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize registry with configuration."""
        super().__init__()
        self.config = config or {}
        self._connection_manager: Optional[PostgresConnectionManager] = None

    def get_connection_manager(self) -> PostgresConnectionManager:
        """Get PostgreSQL connection manager instance."""
        if self._connection_manager is None:
            # Create connection config from registry configuration
            if "postgres" in self.config:
                connection_config = ConnectionConfig(**self.config["postgres"])
            else:
                # Use environment-based configuration as fallback
                connection_config = ConnectionConfig.from_environment()
                
            self._connection_manager = PostgresConnectionManager(connection_config)
        
        return self._connection_manager

    def register_dependencies(self) -> None:
        """Register PostgreSQL adapter dependencies."""
        # Register connection manager
        self.register_singleton("postgres_connection_manager", self.get_connection_manager)
        
        # Register configuration
        self.register_value("postgres_config", self.config.get("postgres", {}))

    async def cleanup(self) -> None:
        """Cleanup registry resources."""
        if self._connection_manager:
            try:
                await self._connection_manager.close()
            except Exception:
                # Log error but don't raise during cleanup
                pass
            finally:
                self._connection_manager = None
                
        await super().cleanup()