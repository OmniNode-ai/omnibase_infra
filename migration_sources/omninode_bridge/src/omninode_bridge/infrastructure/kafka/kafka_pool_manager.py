"""Kafka connection pool manager for high-performance event publishing.

This module provides a connection pool manager for AIOKafkaProducer instances
to optimize throughput and reduce connection overhead in high-concurrency scenarios.

Performance Targets:
- Support 100+ concurrent publishing operations
- <5ms overhead for producer acquisition/release
- Automatic health checks and reconnection
- Graceful degradation on connection failures

Key Features:
- Configurable pool size (default: 10 producers)
- Automatic producer recycling on errors
- Connection health monitoring
- Circuit breaker integration
- Metrics collection (pool utilization, wait times)

Environment Configuration:
- KAFKA_POOL_SIZE: Number of producers in pool (default: 10)
- KAFKA_POOL_MAX_WAIT_MS: Max wait time for producer (default: 5000)
- KAFKA_POOL_HEALTH_CHECK_INTERVAL: Health check interval in seconds (default: 60)
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError

from ...config.batch_sizes import get_batch_manager
from ...security.audit_logger import AuditEventType, AuditSeverity, get_audit_logger
from ...utils.circuit_breaker_config import KAFKA_CIRCUIT_BREAKER

logger = logging.getLogger(__name__)


@dataclass
class PoolMetrics:
    """Metrics for connection pool performance."""

    total_acquisitions: int = 0
    total_releases: int = 0
    total_wait_time_ms: float = 0.0
    peak_utilization: int = 0
    current_utilization: int = 0
    total_errors: int = 0
    total_reconnections: int = 0
    pool_created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def get_average_wait_time_ms(self) -> float:
        """Calculate average wait time for producer acquisition."""
        if self.total_acquisitions == 0:
            return 0.0
        return self.total_wait_time_ms / self.total_acquisitions

    def get_utilization_percentage(self, pool_size: int) -> float:
        """Calculate current pool utilization percentage."""
        if pool_size == 0:
            return 0.0
        return (self.current_utilization / pool_size) * 100.0


@dataclass
class ProducerWrapper:
    """Wrapper for producer instance with health tracking."""

    producer: AIOKafkaProducer
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_used_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    error_count: int = 0
    is_healthy: bool = True

    def mark_used(self) -> None:
        """Mark producer as recently used."""
        self.last_used_at = datetime.now(UTC)

    def mark_error(self) -> None:
        """Mark producer as having an error."""
        self.error_count += 1
        if self.error_count >= 3:
            self.is_healthy = False


class KafkaConnectionPool:
    """Manages a pool of AIOKafkaProducer connections for high-performance publishing.

    This pool manager provides:
    - Multiple producer instances for concurrent operations
    - Automatic producer recycling on errors
    - Health checks and reconnection logic
    - Performance metrics collection

    Example:
        ```python
        pool = KafkaConnectionPool(
            bootstrap_servers="localhost:29092",
            pool_size=10,
            max_wait_ms=5000
        )
        await pool.initialize()

        # Use producer from pool
        async with pool.acquire() as producer:
            await producer.send("my-topic", value={"data": "value"})

        # Get metrics
        metrics = pool.get_metrics()
        print(f"Pool utilization: {metrics.current_utilization}/{pool.pool_size}")
        ```
    """

    def __init__(
        self,
        bootstrap_servers: str | None = None,
        pool_size: int | None = None,
        max_wait_ms: int | None = None,
        health_check_interval: int | None = None,
        # Producer configuration
        compression_type: str | None = None,
        batch_size: int | None = None,
        linger_ms: int | None = None,
        buffer_memory: int | None = None,
        max_request_size: int | None = None,
        enable_idempotence: bool = True,
        acks: str = "all",
    ):
        """Initialize Kafka connection pool.

        Args:
            bootstrap_servers: Kafka bootstrap servers
            pool_size: Number of producers in pool (default: 10)
            max_wait_ms: Max wait time for producer acquisition (default: 5000)
            health_check_interval: Health check interval in seconds (default: 60)
            compression_type: Producer compression type (gzip, lz4, snappy, zstd)
            batch_size: Producer batch size
            linger_ms: Producer linger time
            buffer_memory: Producer buffer memory
            max_request_size: Max request size
            enable_idempotence: Enable idempotent producer
            acks: Producer acknowledgment level (all, 1, 0)
        """
        # Configuration
        # Default to remote infrastructure (resolves to 192.168.86.200:9092 via /etc/hosts)
        self.bootstrap_servers = bootstrap_servers or os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", "omninode-bridge-redpanda:9092"
        )
        self.pool_size = pool_size or int(os.getenv("KAFKA_POOL_SIZE", "10"))
        self.max_wait_ms = max_wait_ms or int(
            os.getenv("KAFKA_POOL_MAX_WAIT_MS", "5000")
        )
        self.health_check_interval = health_check_interval or int(
            os.getenv("KAFKA_POOL_HEALTH_CHECK_INTERVAL", "60")
        )

        # Producer configuration
        batch_manager = get_batch_manager()
        environment = os.getenv("ENVIRONMENT", "development").lower()

        # Environment-based defaults
        if environment == "production":
            default_compression = "lz4"
            default_linger_ms = 10
            default_buffer_memory = 67108864  # 64MB
            default_max_request_size = 1048576  # 1MB
        elif environment == "staging":
            default_compression = "gzip"
            default_linger_ms = 5
            default_buffer_memory = 33554432  # 32MB
            default_max_request_size = 524288  # 512KB
        else:  # development
            default_compression = "gzip"  # Use compression in dev for testing
            default_linger_ms = 10
            default_buffer_memory = 16777216  # 16MB
            default_max_request_size = 262144  # 256KB

        self.producer_config = {
            "bootstrap_servers": self.bootstrap_servers,
            "acks": acks,
            "enable_idempotence": enable_idempotence,
        }

        # Add optional configurations
        if compression_type or default_compression:
            self.producer_config["compression_type"] = (
                compression_type or default_compression
            )

        # Note: batch_size and linger_ms not supported in aiokafka 0.11.0
        # When upgrading, uncomment these:
        # self.producer_config["batch_size"] = batch_size or batch_manager.kafka_producer_batch_size
        # self.producer_config["linger_ms"] = linger_ms or default_linger_ms
        # self.producer_config["buffer_memory"] = buffer_memory or default_buffer_memory
        # self.producer_config["max_request_size"] = max_request_size or default_max_request_size

        # Pool management
        self._pool: list[ProducerWrapper] = []
        self._available: asyncio.Queue = asyncio.Queue(maxsize=self.pool_size)
        self._lock = asyncio.Lock()
        self._initialized = False
        self._health_check_task: asyncio.Task | None = None

        # Metrics
        self.metrics = PoolMetrics()

        # Audit logger
        self.audit_logger = get_audit_logger("kafka_pool", "1.0.0")

    @KAFKA_CIRCUIT_BREAKER()
    async def initialize(self) -> None:
        """Initialize connection pool with producers."""
        if self._initialized:
            logger.warning("Pool already initialized")
            return

        async with self._lock:
            if self._initialized:
                return

            logger.info(
                f"Initializing Kafka connection pool with {self.pool_size} producers"
            )

            # Create and start producers
            for i in range(self.pool_size):
                try:
                    producer = await self._create_producer()
                    wrapper = ProducerWrapper(producer=producer)
                    self._pool.append(wrapper)
                    await self._available.put(wrapper)

                    logger.debug(f"Created producer {i+1}/{self.pool_size}")

                except Exception as e:
                    logger.error(f"Failed to create producer {i+1}: {e}")
                    # Continue creating remaining producers
                    continue

            if not self._pool:
                error_msg = "Failed to create any producers in pool"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            self._initialized = True

            # Start health check task
            self._health_check_task = asyncio.create_task(self._health_check_loop())

            # Audit log
            self.audit_logger.log_event(
                event_type=AuditEventType.SERVICE_STARTUP,
                severity=AuditSeverity.LOW,
                request=None,
                additional_data={
                    "service": "kafka_pool",
                    "pool_size": len(self._pool),
                    "target_pool_size": self.pool_size,
                    "bootstrap_servers": self.bootstrap_servers,
                },
                message=f"Kafka connection pool initialized with {len(self._pool)} producers",
            )

            logger.info(
                f"Kafka connection pool initialized with {len(self._pool)} producers"
            )

    async def _create_producer(self) -> AIOKafkaProducer:
        """Create and start a new producer instance."""
        import json

        producer = AIOKafkaProducer(
            **self.producer_config,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            key_serializer=lambda v: str(v).encode("utf-8") if v is not None else None,
        )
        await producer.start()
        return producer

    @asynccontextmanager
    async def acquire(self, timeout_ms: int | None = None):
        """Acquire a producer from the pool.

        Args:
            timeout_ms: Max wait time in milliseconds (default: pool max_wait_ms)

        Yields:
            AIOKafkaProducer: Producer instance from pool

        Raises:
            TimeoutError: If no producer available within timeout
            RuntimeError: If pool not initialized
        """
        if not self._initialized:
            raise RuntimeError("Pool not initialized. Call initialize() first.")

        wait_timeout = (timeout_ms or self.max_wait_ms) / 1000.0
        start_time = time.perf_counter()
        wrapper = None  # Initialize to prevent UnboundLocalError in finally block

        try:
            # Wait for available producer
            wrapper = await asyncio.wait_for(
                self._available.get(), timeout=wait_timeout
            )

            # Track metrics
            wait_time_ms = (time.perf_counter() - start_time) * 1000
            self.metrics.total_acquisitions += 1
            self.metrics.total_wait_time_ms += wait_time_ms
            self.metrics.current_utilization += 1
            self.metrics.peak_utilization = max(
                self.metrics.peak_utilization, self.metrics.current_utilization
            )

            if wait_time_ms > 100:
                logger.warning(
                    f"Producer acquisition took {wait_time_ms:.2f}ms (target: <5ms)"
                )

            # Check if producer is healthy
            if not wrapper.is_healthy:
                logger.warning("Acquired unhealthy producer, attempting recreation")
                try:
                    await wrapper.producer.stop()
                except (TimeoutError, KafkaError) as e:
                    logger.warning(f"Failed to stop unhealthy producer: {e}")
                except Exception as e:
                    logger.error(
                        f"Unexpected error stopping producer: {e}",
                        exc_info=True,
                    )

                wrapper.producer = await self._create_producer()
                wrapper.is_healthy = True
                wrapper.error_count = 0
                self.metrics.total_reconnections += 1

            wrapper.mark_used()

            # Yield producer to caller
            yield wrapper.producer

        except TimeoutError:
            logger.error(
                f"Timeout acquiring producer after {wait_timeout*1000:.0f}ms. "
                f"Pool utilization: {self.metrics.current_utilization}/{self.pool_size}"
            )
            self.metrics.total_errors += 1
            raise TimeoutError(
                f"No producer available within {wait_timeout*1000:.0f}ms"
            )

        except Exception as e:
            logger.error(f"Error acquiring producer: {e}")
            self.metrics.total_errors += 1
            raise

        finally:
            # Return producer to pool (only if successfully acquired)
            if wrapper is not None:
                try:
                    await self._available.put(wrapper)
                    self.metrics.total_releases += 1
                    self.metrics.current_utilization -= 1
                except Exception as e:
                    logger.error(f"Error returning producer to pool: {e}")

    async def _health_check_loop(self) -> None:
        """Background task to check producer health."""
        while self._initialized:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_producer_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

    async def _check_producer_health(self) -> None:
        """Check health of all producers in pool."""
        unhealthy_count = 0

        for wrapper in self._pool:
            try:
                # Check if producer is still connected
                if hasattr(wrapper.producer, "_closed") and wrapper.producer._closed:
                    wrapper.is_healthy = False
                    unhealthy_count += 1
                    logger.warning("Producer is closed, marking as unhealthy")

            except Exception as e:
                logger.error(f"Error checking producer health: {e}")
                wrapper.mark_error()

        if unhealthy_count > 0:
            logger.warning(
                f"Health check found {unhealthy_count}/{len(self._pool)} unhealthy producers"
            )

    async def shutdown(self) -> None:
        """Shutdown connection pool and close all producers."""
        if not self._initialized:
            return

        logger.info("Shutting down Kafka connection pool")

        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Close all producers
        close_tasks = []
        for wrapper in self._pool:
            if not hasattr(wrapper.producer, "_closed") or not wrapper.producer._closed:
                close_tasks.append(wrapper.producer.stop())

        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

        self._pool.clear()
        self._initialized = False

        # Audit log
        self.audit_logger.log_event(
            event_type=AuditEventType.SERVICE_SHUTDOWN,
            severity=AuditSeverity.LOW,
            request=None,
            additional_data={
                "service": "kafka_pool",
                "total_acquisitions": self.metrics.total_acquisitions,
                "total_releases": self.metrics.total_releases,
                "average_wait_time_ms": self.metrics.get_average_wait_time_ms(),
                "peak_utilization": self.metrics.peak_utilization,
            },
            message="Kafka connection pool shutdown",
        )

        logger.info("Kafka connection pool shutdown complete")

    def get_metrics(self) -> dict[str, Any]:
        """Get pool performance metrics.

        Returns:
            Dictionary with pool metrics:
            - pool_size: Total producers in pool
            - current_utilization: Currently in-use producers
            - peak_utilization: Peak concurrent usage
            - total_acquisitions: Total producer acquisitions
            - total_releases: Total producer releases
            - average_wait_time_ms: Average wait time for acquisition
            - total_errors: Total acquisition errors
            - total_reconnections: Total producer reconnections
            - uptime_seconds: Pool uptime
        """
        uptime = (datetime.now(UTC) - self.metrics.pool_created_at).total_seconds()

        return {
            "pool_size": self.pool_size,
            "current_utilization": self.metrics.current_utilization,
            "utilization_percentage": self.metrics.get_utilization_percentage(
                self.pool_size
            ),
            "peak_utilization": self.metrics.peak_utilization,
            "total_acquisitions": self.metrics.total_acquisitions,
            "total_releases": self.metrics.total_releases,
            "average_wait_time_ms": round(self.metrics.get_average_wait_time_ms(), 2),
            "total_errors": self.metrics.total_errors,
            "total_reconnections": self.metrics.total_reconnections,
            "uptime_seconds": round(uptime, 2),
            "is_initialized": self._initialized,
        }

    @property
    def is_initialized(self) -> bool:
        """Check if pool is initialized."""
        return self._initialized

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()
