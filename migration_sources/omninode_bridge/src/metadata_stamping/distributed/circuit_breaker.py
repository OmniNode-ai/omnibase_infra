"""
Distributed circuit breaker implementation for MetadataStampingService.

Provides resilience patterns for distributed systems with automatic failure detection,
recovery mechanisms, and distributed state coordination across service instances.
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Optional

from redis import asyncio as aioredis

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""

    name: str
    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_calls: int = 3
    success_threshold: int = 3
    timeout: float = 30.0
    expected_exception: type = Exception
    slow_call_threshold: float = 1.0
    slow_call_rate_threshold: float = 0.5
    minimum_throughput: int = 10


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring."""

    name: str
    state: CircuitState
    failure_count: int
    success_count: int
    total_calls: int
    last_failure_time: Optional[float]
    state_change_time: float
    slow_calls: int
    avg_response_time: float
    error_rate: float
    throughput: float


class CircuitBreakerStateManager:
    """Manages circuit breaker state in distributed environments."""

    def __init__(self, redis_client: aioredis.Redis, key_prefix: str = "cb:"):
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.local_cache: dict[str, dict] = {}
        self.cache_ttl = 5  # Local cache TTL in seconds

    async def get_state(self, circuit_name: str) -> dict[str, Any]:
        """Get circuit breaker state from distributed store."""
        # Check local cache first
        cache_key = f"{circuit_name}_state"
        if cache_key in self.local_cache:
            cached_data = self.local_cache[cache_key]
            if time.time() - cached_data["cached_at"] < self.cache_ttl:
                return cached_data["data"]

        # Get from Redis
        redis_key = f"{self.key_prefix}{circuit_name}"
        try:
            state_data = await self.redis.get(redis_key)
            if state_data:
                state = json.loads(state_data)
            else:
                # Initialize default state
                state = {
                    "state": CircuitState.CLOSED.value,
                    "failure_count": 0,
                    "success_count": 0,
                    "total_calls": 0,
                    "last_failure_time": None,
                    "state_change_time": time.time(),
                    "slow_calls": 0,
                    "response_times": [],
                    "half_open_calls": 0,
                }
                await self._set_state(circuit_name, state)

            # Update local cache
            self.local_cache[cache_key] = {"data": state, "cached_at": time.time()}

            return state

        except Exception as e:
            logger.error(f"Error getting circuit breaker state for {circuit_name}: {e}")
            # Return default state on error
            return {
                "state": CircuitState.CLOSED.value,
                "failure_count": 0,
                "success_count": 0,
                "total_calls": 0,
                "last_failure_time": None,
                "state_change_time": time.time(),
                "slow_calls": 0,
                "response_times": [],
                "half_open_calls": 0,
            }

    async def _set_state(self, circuit_name: str, state: dict[str, Any]) -> None:
        """Set circuit breaker state in distributed store."""
        redis_key = f"{self.key_prefix}{circuit_name}"
        try:
            await self.redis.setex(
                redis_key, 300, json.dumps(state, default=str)  # 5 minutes TTL
            )

            # Update local cache
            cache_key = f"{circuit_name}_state"
            self.local_cache[cache_key] = {"data": state, "cached_at": time.time()}

        except Exception as e:
            logger.error(f"Error setting circuit breaker state for {circuit_name}: {e}")

    async def record_success(
        self, circuit_name: str, response_time: float, config: CircuitBreakerConfig
    ) -> None:
        """Record a successful operation."""
        state = await self.get_state(circuit_name)

        # Ensure success_count key exists
        if "success_count" not in state:
            state["success_count"] = 0
        if "total_calls" not in state:
            state["total_calls"] = 0
        if "slow_calls" not in state:
            state["slow_calls"] = 0

        state["success_count"] += 1
        state["total_calls"] += 1

        # Track half-open calls
        if state["state"] == CircuitState.HALF_OPEN.value:
            state["half_open_calls"] = state.get("half_open_calls", 0) + 1

        # Track response times (keep only recent ones)
        response_times = state.get("response_times", [])
        response_times.append(response_time)
        if len(response_times) > 100:  # Keep only last 100
            response_times = response_times[-100:]
        state["response_times"] = response_times

        # Check if call was slow (use circuit-specific threshold)
        if "slow_calls" not in state:
            state["slow_calls"] = 0
        if response_time > config.slow_call_threshold:
            state["slow_calls"] += 1

        await self._set_state(circuit_name, state)

    async def record_failure(
        self, circuit_name: str, exception: Exception, config: CircuitBreakerConfig
    ) -> None:
        """Record a failed operation."""
        state = await self.get_state(circuit_name)

        # Initialize missing keys for defensive programming
        if "failure_count" not in state:
            state["failure_count"] = 0
        if "total_calls" not in state:
            state["total_calls"] = 0
        if "slow_calls" not in state:
            state["slow_calls"] = 0

        state["failure_count"] += 1
        state["total_calls"] += 1
        state["last_failure_time"] = time.time()

        # Track half-open calls
        if state["state"] == CircuitState.HALF_OPEN.value:
            state["half_open_calls"] = state.get("half_open_calls", 0) + 1

        await self._set_state(circuit_name, state)

    async def transition_state(
        self, circuit_name: str, new_state: CircuitState
    ) -> None:
        """Transition circuit breaker to a new state."""
        state = await self.get_state(circuit_name)

        if state["state"] != new_state.value:
            logger.info(
                f"Circuit breaker {circuit_name} transitioning from {state['state']} to {new_state.value}"
            )

            state["state"] = new_state.value
            state["state_change_time"] = time.time()

            # Reset counters on state transitions
            if new_state == CircuitState.HALF_OPEN:
                state["success_count"] = 0
                state["failure_count"] = 0
                state["half_open_calls"] = 0
            elif new_state == CircuitState.CLOSED:
                state["failure_count"] = 0
                state["slow_calls"] = 0
                state["half_open_calls"] = 0

            await self._set_state(circuit_name, state)

    async def should_allow_request(
        self, circuit_name: str, config: CircuitBreakerConfig
    ) -> bool:
        """Check if a request should be allowed based on current state."""
        state = await self.get_state(circuit_name)
        current_state = CircuitState(state["state"])
        current_time = time.time()

        if current_state == CircuitState.CLOSED:
            return True
        elif current_state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if (current_time - state["state_change_time"]) >= config.recovery_timeout:
                await self.transition_state(circuit_name, CircuitState.HALF_OPEN)
                return True
            return False
        elif current_state == CircuitState.HALF_OPEN:
            # Allow limited requests to test recovery
            return state.get("half_open_calls", 0) < config.half_open_max_calls
        else:
            return False

    async def check_state_transitions(
        self, circuit_name: str, config: CircuitBreakerConfig
    ) -> None:
        """Check and perform necessary state transitions."""
        state = await self.get_state(circuit_name)
        current_state = CircuitState(state["state"])

        if current_state == CircuitState.CLOSED:
            # Check if we should open the circuit
            if self._should_open_circuit(state, config):
                await self.transition_state(circuit_name, CircuitState.OPEN)

        elif current_state == CircuitState.HALF_OPEN:
            # Check if we should close or reopen the circuit
            failure_count = state.get("failure_count", 0)
            success_count = state.get("success_count", 0)

            if failure_count > 0:
                # Had failures, go back to open
                await self.transition_state(circuit_name, CircuitState.OPEN)
            elif success_count >= config.success_threshold:
                # Enough successes, close the circuit
                await self.transition_state(circuit_name, CircuitState.CLOSED)

    def _should_open_circuit(
        self, state: dict[str, Any], config: CircuitBreakerConfig
    ) -> bool:
        """Determine if circuit should be opened based on failure rate."""
        # Defensive initialization of required keys
        failure_count = state.get("failure_count", 0)
        total_calls = state.get("total_calls", 0)
        slow_calls = state.get("slow_calls", 0)

        if total_calls < config.minimum_throughput:
            return False

        # Check failure rate
        if total_calls > 0:
            failure_rate = failure_count / total_calls
            if failure_rate >= 0.5:  # 50% failure rate
                return True

            # Check slow call rate
            slow_call_rate = slow_calls / total_calls
            if slow_call_rate >= config.slow_call_rate_threshold:
                return True

        # Check absolute failure count
        return failure_count >= config.failure_threshold

    async def get_metrics(self, circuit_name: str) -> CircuitBreakerMetrics:
        """Get metrics for a circuit breaker."""
        state = await self.get_state(circuit_name)

        # Calculate metrics
        response_times = state.get("response_times", [])
        avg_response_time = (
            sum(response_times) / len(response_times) if response_times else 0.0
        )
        error_rate = (
            state["failure_count"] / state["total_calls"]
            if state["total_calls"] > 0
            else 0.0
        )

        # Calculate throughput (calls per second in last window)
        window_start = time.time() - 60  # Last minute
        throughput = state["total_calls"]  # Simplified

        return CircuitBreakerMetrics(
            name=circuit_name,
            state=CircuitState(state["state"]),
            failure_count=state["failure_count"],
            success_count=state["success_count"],
            total_calls=state["total_calls"],
            last_failure_time=state["last_failure_time"],
            state_change_time=state["state_change_time"],
            slow_calls=state["slow_calls"],
            avg_response_time=avg_response_time,
            error_rate=error_rate,
            throughput=throughput,
        )


class DistributedCircuitBreaker:
    """Distributed circuit breaker for resilient service calls."""

    def __init__(
        self, config: CircuitBreakerConfig, state_manager: CircuitBreakerStateManager
    ):
        self.config = config
        self.state_manager = state_manager
        self.name = config.name

    @asynccontextmanager
    async def call(self):
        """Execute a call with circuit breaker protection."""
        # Check if request should be allowed
        if not await self.state_manager.should_allow_request(self.name, self.config):
            raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is open")

        start_time = time.time()
        success = False

        try:
            yield
            success = True

        except self.config.expected_exception as e:
            await self.state_manager.record_failure(self.name, e, self.config)
            raise

        except TimeoutError as e:
            await self.state_manager.record_failure(self.name, e, self.config)
            raise

        finally:
            if success:
                response_time = time.time() - start_time
                await self.state_manager.record_success(
                    self.name, response_time, self.config
                )

            # Check for state transitions
            await self.state_manager.check_state_transitions(self.name, self.config)

    async def get_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        state_data = await self.state_manager.get_state(self.name)
        return CircuitState(state_data["state"])

    async def get_metrics(self) -> CircuitBreakerMetrics:
        """Get circuit breaker metrics."""
        return await self.state_manager.get_metrics(self.name)

    async def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        await self.state_manager.transition_state(self.name, CircuitState.CLOSED)

    async def force_open(self) -> None:
        """Force circuit breaker to open state."""
        await self.state_manager.transition_state(self.name, CircuitState.OPEN)


class CircuitBreakerOpenError(Exception):
    """Error raised when circuit breaker is open."""

    pass


class CircuitBreakerManager:
    """Manages multiple circuit breakers for different services."""

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or "redis://localhost:6379"
        self.redis_client: Optional[aioredis.Redis] = None
        self.state_manager: Optional[CircuitBreakerStateManager] = None
        self.circuit_breakers: dict[str, DistributedCircuitBreaker] = {}
        self.default_configs: dict[str, CircuitBreakerConfig] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the circuit breaker manager."""
        if self._initialized:
            return

        try:
            # Initialize Redis connection
            self.redis_client = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                health_check_interval=30,
            )

            # Test connection
            await self.redis_client.ping()

            # Initialize state manager
            self.state_manager = CircuitBreakerStateManager(self.redis_client)

            # Load default configurations
            self._load_default_configs()

            self._initialized = True
            logger.info("Circuit breaker manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize circuit breaker manager: {e}")
            raise

    def _load_default_configs(self) -> None:
        """Load default circuit breaker configurations."""
        # Database operations
        self.default_configs["database"] = CircuitBreakerConfig(
            name="database",
            failure_threshold=5,
            recovery_timeout=60,
            half_open_max_calls=3,
            timeout=30.0,
            slow_call_threshold=1.0,
        )

        # External API calls
        self.default_configs["external_api"] = CircuitBreakerConfig(
            name="external_api",
            failure_threshold=3,
            recovery_timeout=30,
            half_open_max_calls=2,
            timeout=10.0,
            slow_call_threshold=2.0,
        )

        # Redis operations
        self.default_configs["redis"] = CircuitBreakerConfig(
            name="redis",
            failure_threshold=3,
            recovery_timeout=15,
            half_open_max_calls=2,
            timeout=5.0,
            slow_call_threshold=0.5,
        )

        # Hash generation (internal)
        self.default_configs["hash_generation"] = CircuitBreakerConfig(
            name="hash_generation",
            failure_threshold=10,
            recovery_timeout=10,
            half_open_max_calls=5,
            timeout=5.0,
            slow_call_threshold=0.002,  # 2ms threshold
        )

    def get_circuit_breaker(
        self, name: str, config: CircuitBreakerConfig = None
    ) -> DistributedCircuitBreaker:
        """Get or create a circuit breaker."""
        if not self._initialized:
            raise RuntimeError("Circuit breaker manager not initialized")

        if name not in self.circuit_breakers:
            # Use provided config or default
            if config is None:
                config = self.default_configs.get(name)
                if config is None:
                    # Create default config
                    config = CircuitBreakerConfig(name=name)

            circuit_breaker = DistributedCircuitBreaker(config, self.state_manager)
            self.circuit_breakers[name] = circuit_breaker

        return self.circuit_breakers[name]

    async def get_all_metrics(self) -> dict[str, CircuitBreakerMetrics]:
        """Get metrics for all circuit breakers."""
        metrics = {}
        for name, cb in self.circuit_breakers.items():
            metrics[name] = await cb.get_metrics()
        return metrics

    async def health_check(self) -> dict[str, Any]:
        """Perform health check of the circuit breaker system."""
        if not self._initialized:
            return {"status": "not_initialized"}

        try:
            # Test Redis connectivity
            await self.redis_client.ping()

            # Get metrics for all circuit breakers
            metrics = await self.get_all_metrics()

            open_circuits = [
                name
                for name, metric in metrics.items()
                if metric.state == CircuitState.OPEN
            ]

            return {
                "status": "healthy",
                "total_circuits": len(self.circuit_breakers),
                "open_circuits": len(open_circuits),
                "open_circuit_names": open_circuits,
                "redis_connected": True,
            }

        except Exception as e:
            logger.error(f"Circuit breaker health check failed: {e}")
            return {"status": "unhealthy", "error": str(e), "redis_connected": False}

    async def close(self) -> None:
        """Close the circuit breaker manager."""
        if self.redis_client:
            await self.redis_client.close()
        self._initialized = False
        logger.info("Circuit breaker manager closed")


# Decorator for easy circuit breaker usage
def circuit_breaker(name: str, config: CircuitBreakerConfig = None):
    """Decorator to apply circuit breaker to a function."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            manager = await get_circuit_breaker_manager()
            cb = manager.get_circuit_breaker(name, config)

            async with cb.call():
                return await func(*args, **kwargs)

        return wrapper

    return decorator


# Global circuit breaker manager
_circuit_breaker_manager: Optional[CircuitBreakerManager] = None


async def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get the global circuit breaker manager."""
    global _circuit_breaker_manager

    if _circuit_breaker_manager is None:
        redis_url = os.getenv("REDIS_CLUSTER_NODES", "redis://localhost:6379")
        # Use first node if cluster nodes provided
        if "," in redis_url:
            redis_url = f"redis://{redis_url.split(',')[0]}"

        _circuit_breaker_manager = CircuitBreakerManager(redis_url)
        await _circuit_breaker_manager.initialize()

    return _circuit_breaker_manager


async def close_circuit_breaker_manager() -> None:
    """Close the global circuit breaker manager."""
    global _circuit_breaker_manager

    if _circuit_breaker_manager is not None:
        await _circuit_breaker_manager.close()
        _circuit_breaker_manager = None
