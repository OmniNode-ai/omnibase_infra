"""Performance Optimizations for OmniNode Bridge Container Restart Loop Fixes.

This module implements targeted performance optimizations identified during the
container restart loop fix analysis. Focus areas include database connection
pooling, async patterns, memory management, and startup performance.
"""

import asyncio
import gc
import logging
import os
import time
import weakref
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, ClassVar, Optional, TypeVar
from uuid import uuid4

import psutil

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class PerformanceMetrics:
    """Container for performance metrics tracking."""

    startup_time_ms: float = 0.0
    db_connection_time_ms: float = 0.0
    kafka_connection_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    active_connections: int = 0
    connection_pool_utilization: float = 0.0
    avg_query_time_ms: float = 0.0
    failed_connections: int = 0
    gc_collections: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for logging/monitoring."""
        return {
            "startup_time_ms": self.startup_time_ms,
            "db_connection_time_ms": self.db_connection_time_ms,
            "kafka_connection_time_ms": self.kafka_connection_time_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "active_connections": self.active_connections,
            "connection_pool_utilization": self.connection_pool_utilization,
            "avg_query_time_ms": self.avg_query_time_ms,
            "failed_connections": self.failed_connections,
            "gc_collections": self.gc_collections,
        }


class OptimizedConfigLoader:
    """Optimized configuration loading with caching and lazy evaluation."""

    _instance: ClassVar[Optional["OptimizedConfigLoader"]] = None
    _cache: ClassVar[dict[str, Any]] = {}
    _loaded_modules: ClassVar[set[str]] = set()

    def __new__(cls) -> "OptimizedConfigLoader":
        """Singleton pattern for config loader."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_cached_env(
        cls, key: str, default: Any = None, cast_type: type = str
    ) -> Any:
        """Get environment variable with caching and type casting."""
        cache_key = f"env_{key}_{cast_type.__name__}"

        if cache_key not in cls._cache:
            raw_value = os.getenv(key, default)
            if raw_value is None:
                cls._cache[cache_key] = default
            else:
                try:
                    if cast_type is bool:
                        cls._cache[cache_key] = raw_value.lower() in (
                            "true",
                            "1",
                            "yes",
                            "on",
                        )
                    elif cast_type is int:
                        cls._cache[cache_key] = int(raw_value)
                    elif cast_type is float:
                        cls._cache[cache_key] = float(raw_value)
                    else:
                        cls._cache[cache_key] = cast_type(raw_value)
                except (ValueError, TypeError):
                    logger.warning(
                        f"Failed to cast env var {key}={raw_value} to {cast_type}"
                    )
                    cls._cache[cache_key] = default

        return cls._cache[cache_key]

    @classmethod
    def lazy_import_config(cls, module_name: str) -> Any:
        """Lazy import configuration modules to reduce startup time."""
        if module_name in cls._loaded_modules:
            return cls._cache.get(f"module_{module_name}")

        try:
            if module_name == "DatabaseConfig":
                from omninode_bridge.config.environment_config import DatabaseConfig

                config = DatabaseConfig()
                cls._cache[f"module_{module_name}"] = config
            elif module_name == "KafkaConfig":
                from omninode_bridge.config.environment_config import KafkaConfig

                config = KafkaConfig()
                cls._cache[f"module_{module_name}"] = config
            # Add other config modules as needed

            cls._loaded_modules.add(module_name)
            return cls._cache.get(f"module_{module_name}")

        except ImportError as e:
            logger.error(f"Failed to import config module {module_name}: {e}")
            return None


class ConnectionPoolOptimizer:
    """Advanced connection pool optimization for PostgreSQL."""

    def __init__(self, postgres_client):
        self.postgres_client = postgres_client
        self.metrics = PerformanceMetrics()
        self._connection_refs: weakref.WeakSet = weakref.WeakSet()
        self._last_optimization = 0
        self.optimization_interval = 300  # 5 minutes

    async def optimize_pool_parameters(self) -> dict[str, Any]:
        """Dynamically optimize connection pool parameters based on workload."""
        current_time = time.time()
        if current_time - self._last_optimization < self.optimization_interval:
            return {"status": "skipped", "reason": "optimization_interval_not_met"}

        try:
            # Get current pool metrics
            pool_metrics = await self.postgres_client.get_pool_metrics()

            # Analyze workload patterns
            utilization = pool_metrics.utilization_percent
            current_size = pool_metrics.current_size
            max_size = pool_metrics.max_size

            recommendations = {}

            # High utilization optimization
            if utilization > 80:
                recommended_max = min(max_size * 1.2, 50)  # Cap at 50 connections
                recommendations["increase_max_size"] = {
                    "current": max_size,
                    "recommended": int(recommended_max),
                    "reason": f"High utilization: {utilization:.1f}%",
                }

            # Low utilization optimization
            elif utilization < 30 and current_size > 5:
                recommended_max = max(max_size * 0.8, 10)  # Minimum 10 connections
                recommendations["decrease_max_size"] = {
                    "current": max_size,
                    "recommended": int(recommended_max),
                    "reason": f"Low utilization: {utilization:.1f}%",
                }

            # Memory optimization
            memory_usage = self._get_memory_usage()
            if memory_usage > 500:  # MB
                recommendations["memory_optimization"] = {
                    "current_memory_mb": memory_usage,
                    "action": "Consider reducing connection_max_age_seconds",
                    "suggested_value": 1800,  # 30 minutes
                }

            self._last_optimization = current_time

            logger.info(
                "Connection pool optimization analysis completed",
                extra={
                    "recommendations": recommendations,
                    "current_metrics": pool_metrics.__dict__,
                },
            )

            return {"status": "completed", "recommendations": recommendations}

        except Exception as e:
            logger.error(f"Pool optimization failed: {e}")
            return {"status": "failed", "error": str(e)}

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except (OSError, AttributeError):
            # Memory usage retrieval failed due to OS errors or missing psutil attributes
            return 0.0

    async def monitor_connection_health(self) -> dict[str, Any]:
        """Monitor connection health and detect potential issues."""
        health_report = {
            "timestamp": time.time(),
            "status": "healthy",
            "issues": [],
            "metrics": {},
        }

        try:
            if not self.postgres_client.pool:
                health_report["status"] = "unhealthy"
                health_report["issues"].append("No connection pool available")
                return health_report

            # Test connection acquisition speed
            acquisition_start = time.time()
            try:
                async with asyncio.timeout(5.0):  # 5 second timeout
                    async with self.postgres_client.pool.acquire() as conn:
                        await conn.fetchval("SELECT 1")
                acquisition_time = (time.time() - acquisition_start) * 1000
                health_report["metrics"]["connection_acquisition_ms"] = acquisition_time

                if acquisition_time > 1000:  # > 1 second
                    health_report["issues"].append(
                        f"Slow connection acquisition: {acquisition_time:.1f}ms"
                    )

            except TimeoutError:
                health_report["status"] = "unhealthy"
                health_report["issues"].append("Connection acquisition timeout")

            # Check pool statistics
            try:
                pool_size = self.postgres_client.pool.get_size()
                max_size = self.postgres_client.pool.get_max_size()
                utilization = (pool_size / max_size) * 100 if max_size > 0 else 0

                health_report["metrics"]["pool_utilization_percent"] = utilization

                if utilization > 90:
                    health_report["issues"].append(
                        f"High pool utilization: {utilization:.1f}%"
                    )

            except Exception as e:
                health_report["issues"].append(f"Failed to get pool stats: {e}")

            # Memory usage check
            memory_mb = self._get_memory_usage()
            health_report["metrics"]["memory_usage_mb"] = memory_mb

            if memory_mb > 1000:  # > 1GB
                health_report["issues"].append(f"High memory usage: {memory_mb:.1f}MB")

            if health_report["issues"]:
                health_report["status"] = "degraded"

        except Exception as e:
            health_report["status"] = "unhealthy"
            health_report["issues"].append(f"Health check failed: {e}")

        return health_report


class AsyncPatternOptimizer:
    """Optimizations for async/await patterns and concurrency."""

    @staticmethod
    def optimized_semaphore(max_concurrent: int = 10):
        """Decorator for limiting concurrent async operations."""
        semaphore = asyncio.Semaphore(max_concurrent)

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                async with semaphore:
                    return await func(*args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    @asynccontextmanager
    async def timeout_context(timeout_seconds: float):
        """Context manager for operation timeouts with cleanup."""
        try:
            async with asyncio.timeout(timeout_seconds):
                yield
        except TimeoutError:
            logger.warning(f"Operation timed out after {timeout_seconds}s")
            raise

    @staticmethod
    async def gather_with_error_handling(
        *coroutines, return_exceptions: bool = True, max_concurrent: int = 5
    ) -> list[Any]:
        """Optimized gather with concurrency limiting and error handling."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_coro(coro):
            async with semaphore:
                return await coro

        limited_coros = [limited_coro(coro) for coro in coroutines]
        return await asyncio.gather(*limited_coros, return_exceptions=return_exceptions)


class MemoryOptimizer:
    """Memory usage optimization and leak detection."""

    def __init__(self):
        self.gc_thresholds = gc.get_threshold()
        self.baseline_objects = {}
        self._setup_optimized_gc()

    def _setup_optimized_gc(self):
        """Configure garbage collection for optimal performance."""
        # Increase generation 0 threshold to reduce GC frequency
        # This is beneficial for applications with many temporary objects
        gc.set_threshold(1000, 15, 15)  # Default is usually (700, 10, 10)

        # Enable automatic garbage collection
        gc.enable()

        logger.info("Optimized garbage collection configuration applied")

    def take_memory_snapshot(self, label: str = None) -> dict[str, Any]:
        """Take a memory snapshot for leak detection."""
        snapshot = {
            "timestamp": time.time(),
            "label": label or f"snapshot_{uuid4().hex[:8]}",
            "memory_info": {},
            "gc_stats": {},
            "object_counts": {},
        }

        try:
            # Process memory info
            process = psutil.Process()
            memory_info = process.memory_info()
            snapshot["memory_info"] = {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent(),
            }

            # Garbage collection stats
            snapshot["gc_stats"] = {f"gen_{i}": gc.get_count()[i] for i in range(3)}

            # Object counts by type
            object_counts = {}
            for obj in gc.get_objects():
                obj_type = type(obj).__name__
                object_counts[obj_type] = object_counts.get(obj_type, 0) + 1

            # Store top 10 object types
            top_objects = dict(
                sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            )
            snapshot["object_counts"] = top_objects

        except Exception as e:
            logger.error(f"Failed to take memory snapshot: {e}")
            snapshot["error"] = str(e)

        return snapshot

    async def force_cleanup(self) -> dict[str, Any]:
        """Force garbage collection and cleanup."""
        start_time = time.time()

        # Run garbage collection for all generations
        collected = {}
        for generation in range(3):
            collected[f"gen_{generation}"] = gc.collect(generation)

        cleanup_time = (time.time() - start_time) * 1000

        result = {
            "cleanup_time_ms": cleanup_time,
            "objects_collected": collected,
            "total_collected": sum(collected.values()),
        }

        logger.info("Forced cleanup completed", extra=result)
        return result


class StartupOptimizer:
    """Optimize application startup performance."""

    def __init__(self):
        self.startup_metrics = PerformanceMetrics()
        self.startup_start_time = time.time()

    async def parallel_initialization(
        self, init_tasks: list[Callable]
    ) -> dict[str, Any]:
        """Run initialization tasks in parallel where possible."""
        start_time = time.time()

        try:
            # Group tasks by dependency requirements
            independent_tasks = []
            dependent_tasks = []

            for task in init_tasks:
                # Simple heuristic: if task name contains 'postgres' or 'kafka',
                # it's likely dependent on network/external services
                task_name = getattr(task, "__name__", str(task))
                if any(
                    keyword in task_name.lower()
                    for keyword in ["postgres", "kafka", "redis", "http"]
                ):
                    dependent_tasks.append(task)
                else:
                    independent_tasks.append(task)

            results = {}

            # Run independent tasks in parallel
            if independent_tasks:
                logger.info(
                    f"Running {len(independent_tasks)} independent initialization tasks in parallel"
                )
                independent_results = await asyncio.gather(
                    *[task() for task in independent_tasks], return_exceptions=True
                )

                for i, result in enumerate(independent_results):
                    task_name = getattr(independent_tasks[i], "__name__", f"task_{i}")
                    results[task_name] = result

            # Run dependent tasks sequentially with some parallelism
            if dependent_tasks:
                logger.info(
                    f"Running {len(dependent_tasks)} dependent initialization tasks"
                )
                for task in dependent_tasks:
                    task_name = getattr(task, "__name__", str(task))
                    try:
                        results[task_name] = await task()
                    except Exception as e:
                        logger.error(f"Dependent task {task_name} failed: {e}")
                        results[task_name] = e

            total_time = (time.time() - start_time) * 1000

            return {
                "status": "completed",
                "total_time_ms": total_time,
                "independent_tasks": len(independent_tasks),
                "dependent_tasks": len(dependent_tasks),
                "results": results,
            }

        except Exception as e:
            logger.error(f"Parallel initialization failed: {e}")
            return {"status": "failed", "error": str(e)}

    def record_startup_milestone(self, milestone: str):
        """Record startup milestone for performance tracking."""
        current_time = time.time()
        elapsed = (current_time - self.startup_start_time) * 1000

        logger.info(f"Startup milestone: {milestone} at {elapsed:.1f}ms")

        # Store milestone in metrics
        if not hasattr(self.startup_metrics, "milestones"):
            self.startup_metrics.milestones = {}
        self.startup_metrics.milestones[milestone] = elapsed


# Performance monitoring decorators
def monitor_performance(operation_name: str):
    """Decorator to monitor async operation performance."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            success = False

            try:
                result = await func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                logger.error(
                    f"Performance monitored operation {operation_name} failed: {e}"
                )
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                logger.info(
                    f"Operation {operation_name} completed",
                    extra={
                        "operation": operation_name,
                        "duration_ms": duration_ms,
                        "success": success,
                    },
                )

        return wrapper

    return decorator


def cache_result(ttl_seconds: int = 300):
    """Simple result caching decorator for async functions."""
    cache = {}
    cache_times = {}

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function name and args
            cache_key = (
                f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            )
            current_time = time.time()

            # Check if cached result is still valid
            if (
                cache_key in cache
                and cache_key in cache_times
                and current_time - cache_times[cache_key] < ttl_seconds
            ):
                return cache[cache_key]

            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache[cache_key] = result
            cache_times[cache_key] = current_time

            # Clean old cache entries (simple cleanup)
            if len(cache) > 100:  # Limit cache size
                oldest_key = min(cache_times.items(), key=lambda x: x[1])[0]
                del cache[oldest_key]
                del cache_times[oldest_key]

            return result

        return wrapper

    return decorator


# Utility functions for performance optimization
async def optimize_postgres_client(postgres_client) -> dict[str, Any]:
    """Apply comprehensive PostgreSQL client optimizations."""
    optimizer = ConnectionPoolOptimizer(postgres_client)

    results = {}

    # Pool parameter optimization
    results["pool_optimization"] = await optimizer.optimize_pool_parameters()

    # Health monitoring
    results["health_check"] = await optimizer.monitor_connection_health()

    return results


def get_performance_recommendations(metrics: PerformanceMetrics) -> list[str]:
    """Generate performance optimization recommendations based on metrics."""
    recommendations = []

    if metrics.startup_time_ms > 5000:  # > 5 seconds
        recommendations.append(
            "Consider lazy loading of non-critical modules during startup"
        )

    if metrics.connection_pool_utilization > 80:
        recommendations.append("Increase database connection pool size")

    if metrics.avg_query_time_ms > 100:  # > 100ms
        recommendations.append("Review database queries for optimization opportunities")

    if metrics.memory_usage_mb > 500:  # > 500MB
        recommendations.append(
            "Monitor for memory leaks and consider memory optimization"
        )

    if metrics.failed_connections > 0:
        recommendations.append(
            "Investigate connection failures and implement retry mechanisms"
        )

    return recommendations


# Export main optimization components
__all__ = [
    "PerformanceMetrics",
    "OptimizedConfigLoader",
    "ConnectionPoolOptimizer",
    "AsyncPatternOptimizer",
    "MemoryOptimizer",
    "StartupOptimizer",
    "monitor_performance",
    "cache_result",
    "optimize_postgres_client",
    "get_performance_recommendations",
]
