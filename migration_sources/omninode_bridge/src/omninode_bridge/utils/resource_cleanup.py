"""Enhanced resource cleanup and context management utilities.

Addresses resource leak issues and improves context management for:
- Database connections
- Kafka clients
- Async tasks and coroutines
- File handles and network connections
- Memory cleanup
"""

import asyncio
import atexit
import logging
import threading
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ResourceInfo:
    """Information about a managed resource."""

    resource_id: str
    resource_type: str
    created_at: datetime
    last_accessed: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    cleanup_callback: Optional[Callable[[], Any]] = None


class ResourceManager:
    """Centralized resource management with automatic cleanup."""

    def __init__(self, cleanup_interval_seconds: float = 60.0):
        self.cleanup_interval = cleanup_interval_seconds
        self._resources: dict[str, ResourceInfo] = {}
        self._cleanup_tasks: set[asyncio.Task] = set()
        self._cleanup_timer: threading.Timer | None = None
        self._lock = asyncio.Lock()

        # Register cleanup on program exit
        atexit.register(self._emergency_cleanup)

    async def register_resource(
        self,
        resource: Any,
        resource_type: str,
        cleanup_callback: Optional[Callable[[], Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Register a resource for managed cleanup.

        Args:
            resource: Resource to manage
            resource_type: Type classification for the resource
            cleanup_callback: Optional cleanup function
            metadata: Additional resource metadata

        Returns:
            Resource ID for tracking
        """
        resource_id = f"{resource_type}_{id(resource)}_{datetime.now(UTC).timestamp()}"

        async with self._lock:
            self._resources[resource_id] = ResourceInfo(
                resource_id=resource_id,
                resource_type=resource_type,
                created_at=datetime.now(UTC),
                last_accessed=datetime.now(UTC),
                metadata=metadata or {},
                cleanup_callback=cleanup_callback,
            )

        logger.debug(f"Registered resource {resource_id} of type {resource_type}")
        return resource_id

    async def update_resource_access(self, resource_id: str) -> None:
        """Update the last accessed time for a resource."""
        async with self._lock:
            if resource_id in self._resources:
                self._resources[resource_id].last_accessed = datetime.now(UTC)

    async def cleanup_resource(self, resource_id: str) -> bool:
        """Manually cleanup a specific resource.

        Args:
            resource_id: ID of resource to cleanup

        Returns:
            True if cleaned up successfully, False otherwise
        """
        async with self._lock:
            resource_info = self._resources.get(resource_id)
            if not resource_info:
                return False

            try:
                if resource_info.cleanup_callback:
                    if asyncio.iscoroutinefunction(resource_info.cleanup_callback):
                        await resource_info.cleanup_callback()
                    else:
                        resource_info.cleanup_callback()

                del self._resources[resource_id]
                logger.debug(f"Cleaned up resource {resource_id}")
                return True

            except Exception as e:
                logger.error(f"Error cleaning up resource {resource_id}: {e}")
                return False

    async def cleanup_stale_resources(
        self,
        max_age_seconds: float = 3600.0,
    ) -> dict[str, Any]:
        """Clean up resources that haven't been accessed recently.

        Args:
            max_age_seconds: Maximum age before cleanup

        Returns:
            Cleanup statistics
        """
        cutoff_time = datetime.now(UTC) - timedelta(seconds=max_age_seconds)
        stale_resources = []

        async with self._lock:
            for resource_id, resource_info in self._resources.items():
                if resource_info.last_accessed < cutoff_time:
                    stale_resources.append(resource_id)

        cleanup_stats = {
            "total_stale": len(stale_resources),
            "successfully_cleaned": 0,
            "failed_cleanup": 0,
            "cleanup_errors": [],
        }

        for resource_id in stale_resources:
            try:
                if await self.cleanup_resource(resource_id):
                    cleanup_stats["successfully_cleaned"] += 1
                else:
                    cleanup_stats["failed_cleanup"] += 1
            except Exception as e:
                cleanup_stats["failed_cleanup"] += 1
                cleanup_stats["cleanup_errors"].append(str(e))

        logger.info(f"Stale resource cleanup: {cleanup_stats}")
        return cleanup_stats

    def get_resource_stats(self) -> dict[str, Any]:
        """Get current resource statistics."""
        resource_types = {}
        total_resources = len(self._resources)

        for resource_info in self._resources.values():
            resource_type = resource_info.resource_type
            if resource_type not in resource_types:
                resource_types[resource_type] = 0
            resource_types[resource_type] += 1

        return {
            "total_resources": total_resources,
            "by_type": resource_types,
            "active_cleanup_tasks": len(self._cleanup_tasks),
        }

    def _emergency_cleanup(self) -> None:
        """Emergency cleanup called at program exit."""
        logger.info("Performing emergency resource cleanup")

        cleanup_count = 0
        for resource_info in list(self._resources.values()):
            try:
                if resource_info.cleanup_callback and not asyncio.iscoroutinefunction(
                    resource_info.cleanup_callback,
                ):
                    resource_info.cleanup_callback()
                    cleanup_count += 1
            except Exception as e:
                logger.error(f"Error in emergency cleanup: {e}")

        logger.info(f"Emergency cleanup completed: {cleanup_count} resources cleaned")


# Global resource manager instance
resource_manager = ResourceManager()


@asynccontextmanager
async def managed_database_connection(postgres_client):
    """Context manager for database connections with automatic cleanup."""
    connection = None
    resource_id = None

    try:
        if not postgres_client.pool:
            raise RuntimeError("Database not connected")

        connection = await postgres_client.pool.acquire()

        # Register for cleanup
        resource_id = await resource_manager.register_resource(
            connection,
            "database_connection",
            cleanup_callback=lambda: postgres_client.pool.release(connection),
            metadata={"acquired_at": datetime.now(UTC).isoformat()},
        )

        yield connection

    except Exception as e:
        logger.error(f"Error in managed database connection: {e}")
        raise
    finally:
        # Ensure connection is released
        try:
            if connection and postgres_client.pool:
                await postgres_client.pool.release(connection)

            if resource_id:
                await resource_manager.cleanup_resource(resource_id)

        except Exception as e:
            logger.error(f"Error releasing database connection: {e}")


@asynccontextmanager
async def managed_kafka_producer(kafka_client):
    """Context manager for Kafka producer with automatic cleanup."""
    resource_id = None

    try:
        if not kafka_client.is_connected:
            await kafka_client.connect()

        # Register for cleanup
        resource_id = await resource_manager.register_resource(
            kafka_client,
            "kafka_producer",
            cleanup_callback=kafka_client.disconnect,
            metadata={"connected_at": datetime.now(UTC).isoformat()},
        )

        yield kafka_client

    except Exception as e:
        logger.error(f"Error in managed Kafka producer: {e}")
        raise
    finally:
        if resource_id:
            await resource_manager.cleanup_resource(resource_id)


@asynccontextmanager
async def managed_async_tasks(
    *tasks: asyncio.Task,
) -> AsyncGenerator[list[asyncio.Task], None]:
    """Context manager for async tasks with automatic cleanup."""
    resource_ids = []

    try:
        # Register all tasks for cleanup
        for task in tasks:
            if not task.done():
                resource_id = await resource_manager.register_resource(
                    task,
                    "async_task",
                    cleanup_callback=lambda t=task: (
                        t.cancel() if not t.done() else None
                    ),
                    metadata={"created_at": datetime.now(UTC).isoformat()},
                )
                resource_ids.append(resource_id)

        yield list(tasks)

    except Exception as e:
        logger.error(f"Error in managed async tasks: {e}")
        raise
    finally:
        # Cancel unfinished tasks and cleanup
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Cleanup registered resources
        for resource_id in resource_ids:
            await resource_manager.cleanup_resource(resource_id)


class ConnectionPool:
    """Enhanced connection pool with automatic resource management."""

    def __init__(self, max_connections: int = 10, connection_timeout: float = 30.0):
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self._connections: asyncio.Queue = asyncio.Queue(maxsize=max_connections)
        self._active_connections: set[Any] = set()
        self._connection_factory: Optional[Callable[[], Awaitable[Any]]] = None
        self._cleanup_callback: Optional[Callable[[Any], Any]] = None

    def set_connection_factory(
        self,
        factory: Callable[[], Awaitable[Any]],
        cleanup_callback: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        """Set the connection factory and cleanup callback."""
        self._connection_factory = factory
        self._cleanup_callback = cleanup_callback

    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool with automatic return."""
        if not self._connection_factory:
            raise RuntimeError("Connection factory not set")

        connection = None
        try:
            # Try to get existing connection or create new one
            try:
                connection = self._connections.get_nowait()
            except asyncio.QueueEmpty:
                if len(self._active_connections) < self.max_connections:
                    connection = await self._connection_factory()
                else:
                    # Wait for a connection to be available
                    connection = await asyncio.wait_for(
                        self._connections.get(),
                        timeout=self.connection_timeout,
                    )

            self._active_connections.add(connection)
            yield connection

        except TimeoutError:
            raise RuntimeError("Connection pool timeout - no connections available")
        except Exception as e:
            logger.error(f"Error getting connection from pool: {e}")
            raise
        finally:
            # Return connection to pool
            if connection:
                self._active_connections.discard(connection)
                try:
                    self._connections.put_nowait(connection)
                except asyncio.QueueFull:
                    # Pool is full, cleanup this connection
                    if self._cleanup_callback:
                        try:
                            if asyncio.iscoroutinefunction(self._cleanup_callback):
                                await self._cleanup_callback(connection)
                            else:
                                self._cleanup_callback(connection)
                        except Exception as e:
                            logger.error(f"Error cleaning up excess connection: {e}")

    async def close_all_connections(self):
        """Close all connections in the pool."""
        cleanup_count = 0

        # Cleanup active connections
        for connection in list(self._active_connections):
            try:
                if self._cleanup_callback:
                    if asyncio.iscoroutinefunction(self._cleanup_callback):
                        await self._cleanup_callback(connection)
                    else:
                        self._cleanup_callback(connection)
                cleanup_count += 1
            except Exception as e:
                logger.error(f"Error cleaning up active connection: {e}")

        self._active_connections.clear()

        # Cleanup pooled connections
        while not self._connections.empty():
            try:
                connection = self._connections.get_nowait()
                if self._cleanup_callback:
                    if asyncio.iscoroutinefunction(self._cleanup_callback):
                        await self._cleanup_callback(connection)
                    else:
                        self._cleanup_callback(connection)
                cleanup_count += 1
            except Exception as e:
                logger.error(f"Error cleaning up pooled connection: {e}")

        logger.info(f"Closed {cleanup_count} connections from pool")


class CircuitBreaker:
    """Circuit breaker with resource management for preventing cascade failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self._failure_count = 0
        self._last_failure_time = None
        self._state = "closed"  # closed, open, half-open

    @asynccontextmanager
    async def protected_call(self):
        """Context manager for circuit breaker protected calls."""
        if self._state == "open":
            if (
                datetime.now(UTC).timestamp() - self._last_failure_time
                < self.recovery_timeout
            ):
                raise RuntimeError("Circuit breaker is open")
            else:
                self._state = "half-open"

        try:
            yield

            # Success - reset failure count
            if self._state == "half-open":
                self._state = "closed"
            self._failure_count = 0

        except self.expected_exception:
            self._failure_count += 1
            self._last_failure_time = datetime.now(UTC).timestamp()

            if self._failure_count >= self.failure_threshold:
                self._state = "open"
                logger.warning(
                    f"Circuit breaker opened after {self._failure_count} failures",
                )

            raise

    def get_state(self) -> dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "state": self._state,
            "failure_count": self._failure_count,
            "last_failure_time": self._last_failure_time,
            "recovery_timeout": self.recovery_timeout,
        }


@contextmanager
def null_safe_context(data: dict[str, Any], required_keys: list[str]):
    """Context manager for null-safe dictionary operations.

    Args:
        data: Dictionary to validate
        required_keys: Keys that must be present

    Raises:
        ValueError: If required keys are missing
    """
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")

    try:
        yield data
    except KeyError as e:
        logger.error(f"KeyError in null-safe context: {e}")
        raise ValueError(f"Key access error: {e}")


# Performance monitoring for resource cleanup
async def monitor_resource_cleanup():
    """Background task to monitor and report resource cleanup statistics."""
    while True:
        try:
            stats = resource_manager.get_resource_stats()
            logger.info(f"Resource manager stats: {stats}")

            # Perform periodic cleanup
            cleanup_stats = await resource_manager.cleanup_stale_resources()
            if cleanup_stats["total_stale"] > 0:
                logger.info(f"Periodic cleanup completed: {cleanup_stats}")

            await asyncio.sleep(60)  # Check every minute

        except Exception as e:
            logger.error(f"Error in resource cleanup monitor: {e}")
            await asyncio.sleep(10)  # Brief pause before retry
