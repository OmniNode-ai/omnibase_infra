"""Graceful degradation service for external service failures."""

import asyncio
import logging
import time
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Status of an external service."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class DegradationMode(Enum):
    """Mode of degradation when service is unavailable."""

    FAIL_FAST = "fail_fast"  # Fail immediately when service is down
    GRACEFUL = "graceful"  # Use fallback functionality
    CACHED = "cached"  # Use cached responses
    OFFLINE = "offline"  # Work without external service
    CIRCUIT_BREAKER = "circuit_breaker"  # Use circuit breaker pattern


@dataclass
class ServiceConfig:
    """Configuration for an external service."""

    name: str
    health_check_url: Optional[str] = None
    health_check_timeout: float = 5.0
    health_check_interval: float = 30.0
    degradation_mode: DegradationMode = DegradationMode.GRACEFUL
    circuit_breaker_threshold: int = 5  # Failures before opening circuit
    circuit_breaker_timeout: float = 60.0  # Time before trying to close circuit
    cache_ttl: float = 300.0  # Cache TTL in seconds
    max_retries: int = 3
    retry_delay: float = 1.0
    fallback_enabled: bool = True


@dataclass
class ServiceHealth:
    """Health status of an external service."""

    name: str
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_check: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    response_time: Optional[float] = None
    error_message: Optional[str] = None
    circuit_breaker_open: bool = False
    circuit_breaker_until: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return self.status == ServiceStatus.HEALTHY

    @property
    def is_available(self) -> bool:
        """Check if service is available (healthy or degraded)."""
        return self.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]

    @property
    def should_use_circuit_breaker(self) -> bool:
        """Check if circuit breaker should be used."""
        return (
            self.circuit_breaker_open
            and self.circuit_breaker_until
            and datetime.now(UTC) < self.circuit_breaker_until
        )


class GracefulDegradationService:
    """Service for managing graceful degradation of external dependencies."""

    def __init__(self):
        self.services: dict[str, ServiceConfig] = {}
        self.health_status: dict[str, ServiceHealth] = {}
        self.health_check_tasks: dict[str, asyncio.Task] = {}
        self.cache: dict[str, dict[str, Any]] = {}
        self.fallback_handlers: dict[str, Callable] = {}
        self.shutdown_event = asyncio.Event()
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "degraded_requests": 0,
            "cached_responses": 0,
            "fallback_responses": 0,
        }

    def register_service(
        self,
        name: str,
        config: ServiceConfig,
        health_check_func: Optional[Callable[[], asyncio.Task]] = None,
        fallback_func: Optional[Callable] = None,
    ) -> None:
        """
        Register an external service for monitoring and degradation.

        Args:
            name: Service name
            config: Service configuration
            health_check_func: Custom health check function
            fallback_func: Fallback function when service is unavailable
        """
        self.services[name] = config
        self.health_status[name] = ServiceHealth(name=name)

        if fallback_func:
            self.fallback_handlers[name] = fallback_func

        # Start health check task if not already running
        # Only create tasks if we have an active event loop and we're not in test mode
        if name not in self.health_check_tasks:
            try:
                # Check if we're in test mode
                import os

                is_test_mode = (
                    os.getenv("ENVIRONMENT") == "test"
                    or os.getenv("PYTEST_CURRENT_TEST") is not None
                    or "pytest" in os.getenv("_", "")
                )

                # Check if there's an active event loop
                try:
                    loop = asyncio.get_running_loop()
                    loop_available = True
                except RuntimeError:
                    loop_available = False

                # Only create tasks if not in test mode and loop is available
                if not is_test_mode and loop_available:
                    if health_check_func:
                        self.health_check_tasks[name] = asyncio.create_task(
                            self._health_check_loop(name, health_check_func)
                        )
                    elif config.health_check_url:
                        self.health_check_tasks[name] = asyncio.create_task(
                            self._health_check_loop(name, self._default_health_check)
                        )
                else:
                    # In test mode or no event loop, just register without health checks
                    logger.debug(
                        f"Skipping health check task creation for '{name}' (test mode: {is_test_mode}, loop available: {loop_available})"
                    )

            except Exception as e:
                logger.warning(f"Could not create health check task for '{name}': {e}")

        logger.info(f"Registered service '{name}' for graceful degradation monitoring")

    async def _default_health_check(self, service_name: str) -> bool:
        """Default health check using HTTP request."""
        config = self.services[service_name]
        if not config.health_check_url:
            return True

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    config.health_check_url,
                    timeout=aiohttp.ClientTimeout(total=config.health_check_timeout),
                ) as response:
                    return response.status in [200, 201, 202, 204]
        except ImportError:
            logger.warning("aiohttp not available for health checks, assuming healthy")
            return True
        except Exception as e:
            logger.warning(f"Health check failed for {service_name}: {e}")
            return False

    async def _health_check_loop(
        self, service_name: str, health_check_func: Callable
    ) -> None:
        """Run periodic health checks for a service."""
        config = self.services[service_name]
        health = self.health_status[service_name]

        while not self.shutdown_event.is_set():
            try:
                start_time = time.time()

                # Skip health check if circuit breaker is open and not expired
                if health.should_use_circuit_breaker:
                    await asyncio.sleep(config.health_check_interval)
                    continue

                is_healthy = await health_check_func(service_name)
                response_time = time.time() - start_time

                health.last_check = datetime.now(UTC)
                health.response_time = response_time

                if is_healthy:
                    # Service is healthy
                    health.last_success = datetime.now(UTC)
                    health.consecutive_failures = 0
                    health.circuit_breaker_open = False
                    health.circuit_breaker_until = None
                    health.error_message = None

                    if health.status != ServiceStatus.HEALTHY:
                        logger.info(
                            f"Service '{service_name}' recovered to healthy state"
                        )
                        health.status = ServiceStatus.HEALTHY
                else:
                    # Service is unhealthy
                    health.last_failure = datetime.now(UTC)
                    health.consecutive_failures += 1

                    # Update status based on failure count
                    if health.consecutive_failures >= config.circuit_breaker_threshold:
                        health.status = ServiceStatus.UNAVAILABLE
                        health.circuit_breaker_open = True
                        health.circuit_breaker_until = datetime.now(UTC) + timedelta(
                            seconds=config.circuit_breaker_timeout
                        )
                        logger.error(
                            f"Service '{service_name}' circuit breaker opened after "
                            f"{health.consecutive_failures} consecutive failures"
                        )
                    elif health.consecutive_failures > 1:
                        health.status = ServiceStatus.DEGRADED
                        logger.warning(f"Service '{service_name}' is degraded")
                    else:
                        health.status = ServiceStatus.DEGRADED

            except Exception as e:
                logger.error(f"Error in health check loop for {service_name}: {e}")
                health = self.health_status[service_name]
                health.consecutive_failures += 1
                health.last_failure = datetime.now(UTC)
                health.error_message = str(e)

            await asyncio.sleep(config.health_check_interval)

    @asynccontextmanager
    async def with_service(
        self,
        service_name: str,
        operation: str,
        cache_key: Optional[str] = None,
    ) -> AbstractAsyncContextManager[dict[str, Any]]:
        """
        Context manager for executing operations with graceful degradation.

        Args:
            service_name: Name of the service
            operation: Operation being performed
            cache_key: Key for caching results

        Yields:
            Context dict with service status and helper methods
        """
        config = self.services.get(service_name)
        if not config:
            raise ValueError(f"Service '{service_name}' not registered")

        health = self.health_status[service_name]
        self.stats["total_requests"] += 1

        context = {
            "service_name": service_name,
            "operation": operation,
            "config": config,
            "health": health,
            "should_use_service": False,
            "should_use_cache": False,
            "should_use_fallback": False,
            "cache_key": cache_key,
            "response": None,
            "error": None,
            "degraded": False,
        }

        # Determine execution strategy
        if health.is_healthy or config.degradation_mode == DegradationMode.FAIL_FAST:
            context["should_use_service"] = True
        elif health.should_use_circuit_breaker:
            if config.degradation_mode == DegradationMode.CACHED and cache_key:
                context["should_use_cache"] = True
            elif config.degradation_mode in [
                DegradationMode.GRACEFUL,
                DegradationMode.OFFLINE,
            ]:
                context["should_use_fallback"] = True
        elif health.status == ServiceStatus.DEGRADED:
            if config.degradation_mode == DegradationMode.CACHED and cache_key:
                context["should_use_cache"] = True
            else:
                context["should_use_service"] = True
                context["degraded"] = True

        try:
            yield context
        except Exception as e:
            context["error"] = e
            health.consecutive_failures += 1
            health.last_failure = datetime.now(UTC)
            health.error_message = str(e)
            self.stats["failed_requests"] += 1
            raise
        else:
            if context.get("response") is not None:
                self.stats["successful_requests"] += 1
                # Cache successful responses
                if cache_key and config.cache_ttl > 0:
                    self._cache_response(
                        service_name, cache_key, context["response"], config.cache_ttl
                    )
            elif context["degraded"]:
                self.stats["degraded_requests"] += 1

    def get_cached_response(self, service_name: str, cache_key: str) -> Optional[Any]:
        """Get cached response for a service operation."""
        if service_name not in self.cache or cache_key not in self.cache[service_name]:
            return None

        cached_data = self.cache[service_name][cache_key]
        if time.time() > cached_data["expires_at"]:
            # Cache expired
            del self.cache[service_name][cache_key]
            return None

        self.stats["cached_responses"] += 1
        logger.info(f"Using cached response for {service_name}:{cache_key}")
        return cached_data["data"]

    def _cache_response(
        self, service_name: str, cache_key: str, response: Any, ttl: float
    ) -> None:
        """Cache a successful response."""
        if service_name not in self.cache:
            self.cache[service_name] = {}

        self.cache[service_name][cache_key] = {
            "data": response,
            "cached_at": time.time(),
            "expires_at": time.time() + ttl,
        }

    async def execute_with_fallback(
        self,
        service_name: str,
        operation: str,
        primary_func: Callable,
        fallback_func: Optional[Callable] = None,
        cache_key: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """
        Execute an operation with automatic fallback handling.

        Args:
            service_name: Name of the service
            operation: Operation description
            primary_func: Primary function to execute
            fallback_func: Fallback function if primary fails
            cache_key: Key for caching results
            **kwargs: Arguments to pass to functions

        Returns:
            Result of primary or fallback function
        """
        async with self.with_service(service_name, operation, cache_key) as ctx:
            # Check if we should use cached response
            if ctx["should_use_cache"] and cache_key:
                cached = self.get_cached_response(service_name, cache_key)
                if cached is not None:
                    ctx["response"] = cached
                    return cached

            # Try primary service
            if ctx["should_use_service"]:
                try:
                    if asyncio.iscoroutinefunction(primary_func):
                        result = await primary_func(**kwargs)
                    else:
                        result = primary_func(**kwargs)
                    ctx["response"] = result
                    return result
                except Exception as e:
                    logger.warning(
                        f"Primary function failed for {service_name}.{operation}: {e}"
                    )
                    if ctx["config"].degradation_mode == DegradationMode.FAIL_FAST:
                        raise

            # Use fallback if available
            if ctx["should_use_fallback"] or ctx["degraded"]:
                fallback = fallback_func or self.fallback_handlers.get(service_name)
                if fallback:
                    try:
                        logger.info(f"Using fallback for {service_name}.{operation}")
                        if asyncio.iscoroutinefunction(fallback):
                            result = await fallback(**kwargs)
                        else:
                            result = fallback(**kwargs)
                        ctx["response"] = result
                        self.stats["fallback_responses"] += 1
                        return result
                    except Exception as e:
                        logger.error(
                            f"Fallback also failed for {service_name}.{operation}: {e}"
                        )
                        raise

            # No fallback available
            raise RuntimeError(
                f"Service {service_name} unavailable and no fallback configured"
            )

    def get_service_health(self, service_name: str) -> Optional[ServiceHealth]:
        """Get health status for a service."""
        return self.health_status.get(service_name)

    def get_all_health_status(self) -> dict[str, ServiceHealth]:
        """Get health status for all registered services."""
        return self.health_status.copy()

    def get_stats(self) -> dict[str, Any]:
        """Get degradation service statistics."""
        return {
            **self.stats,
            "registered_services": len(self.services),
            "healthy_services": len(
                [h for h in self.health_status.values() if h.is_healthy]
            ),
            "degraded_services": len(
                [
                    h
                    for h in self.health_status.values()
                    if h.status == ServiceStatus.DEGRADED
                ]
            ),
            "unavailable_services": len(
                [
                    h
                    for h in self.health_status.values()
                    if h.status == ServiceStatus.UNAVAILABLE
                ]
            ),
        }

    async def shutdown(self) -> None:
        """Shutdown the degradation service."""
        logger.info("Shutting down graceful degradation service")
        self.shutdown_event.set()

        # Cancel all health check tasks
        for task in self.health_check_tasks.values():
            task.cancel()

        # Wait for tasks to complete
        if self.health_check_tasks:
            await asyncio.gather(
                *self.health_check_tasks.values(), return_exceptions=True
            )

        self.health_check_tasks.clear()
        logger.info("Graceful degradation service shutdown complete")


# Global instance
degradation_service = GracefulDegradationService()


# Convenience decorator for adding graceful degradation to functions
def with_graceful_degradation(
    service_name: str,
    operation: str,
    fallback_func: Optional[Callable] = None,
    cache_key_func: Optional[Callable] = None,
):
    """
    Decorator to add graceful degradation to a function.

    Args:
        service_name: Name of the service
        operation: Operation description
        fallback_func: Fallback function
        cache_key_func: Function to generate cache key from arguments
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache_key = None
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)

            return await degradation_service.execute_with_fallback(
                service_name=service_name,
                operation=operation,
                primary_func=func,
                fallback_func=fallback_func,
                cache_key=cache_key,
                **kwargs,
            )

        return wrapper

    return decorator


# AI Service specific implementations
async def ai_model_fallback(**kwargs) -> dict[str, Any]:
    """Fallback for AI model services when unavailable."""
    return {
        "response": "AI service temporarily unavailable. Please try again later.",
        "confidence": 0.0,
        "fallback": True,
        "generated": False,
    }


async def smart_responder_fallback(**kwargs) -> dict[str, Any]:
    """Fallback for Smart Responder service."""
    return {
        "response": "Smart responder service is currently unavailable. Using basic response.",
        "tier": "fallback",
        "model": "none",
        "fallback": True,
    }


def register_common_services():
    """Register common services with the degradation service."""
    # AI Model Services
    degradation_service.register_service(
        "ai_model",
        ServiceConfig(
            name="ai_model",
            degradation_mode=DegradationMode.GRACEFUL,
            circuit_breaker_threshold=3,
            circuit_breaker_timeout=120.0,
            fallback_enabled=True,
        ),
        fallback_func=ai_model_fallback,
    )

    # Smart Responder Chain
    degradation_service.register_service(
        "smart_responder",
        ServiceConfig(
            name="smart_responder",
            health_check_url="http://localhost:8001/health",
            degradation_mode=DegradationMode.CACHED,
            cache_ttl=600.0,  # 10 minutes
            circuit_breaker_threshold=5,
            circuit_breaker_timeout=300.0,
            fallback_enabled=True,
        ),
        fallback_func=smart_responder_fallback,
    )

    # External API Services
    degradation_service.register_service(
        "external_api",
        ServiceConfig(
            name="external_api",
            degradation_mode=DegradationMode.CACHED,
            cache_ttl=1800.0,  # 30 minutes
            circuit_breaker_threshold=10,
            circuit_breaker_timeout=600.0,
            max_retries=5,
        ),
    )

    logger.info("Common services registered for graceful degradation")
