"""Infrastructure-specific health checks for OmniNode Bridge services."""

import asyncio
import time
from typing import Any, Optional

import structlog
from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaConnectionError, KafkaError, KafkaTimeoutError

from ..health.health_checker import HealthStatus

logger = structlog.get_logger(__name__)


async def check_postgresql_health(
    postgres_client: Any,
    timeout_seconds: float = 5.0,
) -> tuple[HealthStatus, str, dict[str, Any]]:
    """Check PostgreSQL database health.

    Args:
        postgres_client: PostgreSQL client instance
        timeout_seconds: Timeout for the health check

    Returns:
        Tuple of (status, message, details)
    """
    start_time = time.time()

    try:
        if not postgres_client:
            return (
                HealthStatus.UNHEALTHY,
                "PostgreSQL client not initialized",
                {"client_available": False},
            )

        # Simple query to test database connectivity and responsiveness
        query = "SELECT 1 as health_check, NOW() as current_time"

        # Execute query with timeout
        result = await asyncio.wait_for(
            postgres_client.fetch_one(query), timeout=timeout_seconds
        )

        duration_ms = (time.time() - start_time) * 1000

        if result:
            return (
                HealthStatus.HEALTHY,
                f"PostgreSQL connection healthy (response time: {duration_ms:.1f}ms)",
                {
                    "response_time_ms": duration_ms,
                    "current_time": str(result.get("current_time")),
                    "connection_active": True,
                },
            )
        else:
            return (
                HealthStatus.UNHEALTHY,
                "PostgreSQL query returned no result",
                {"response_time_ms": duration_ms, "connection_active": False},
            )

    except TimeoutError:
        duration_ms = (time.time() - start_time) * 1000
        return (
            HealthStatus.UNHEALTHY,
            f"PostgreSQL health check timed out after {timeout_seconds}s",
            {"timeout_seconds": timeout_seconds, "duration_ms": duration_ms},
        )

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        error_type = type(e).__name__

        # Different error types indicate different health states
        if "connection" in str(e).lower() or "connect" in error_type.lower():
            status = HealthStatus.UNHEALTHY
            message = f"PostgreSQL connection failed: {e!s}"
        else:
            status = HealthStatus.DEGRADED
            message = f"PostgreSQL query error: {e!s}"

        return (
            status,
            message,
            {
                "error": str(e),
                "error_type": error_type,
                "duration_ms": duration_ms,
            },
        )


async def check_kafka_health(
    kafka_bootstrap_servers: str,
    timeout_seconds: float = 10.0,
    test_topic: Optional[str] = None,
) -> tuple[HealthStatus, str, dict[str, Any]]:
    """Check Kafka cluster health.

    Args:
        kafka_bootstrap_servers: Kafka bootstrap servers
        timeout_seconds: Timeout for the health check
        test_topic: Optional topic to test (will create temporary consumer)

    Returns:
        Tuple of (status, message, details)
    """
    start_time = time.time()
    producer = None
    consumer = None

    try:
        # Test producer connectivity
        producer = AIOKafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            request_timeout_ms=int(timeout_seconds * 1000),
            metadata_max_age_ms=30000,  # 30 seconds
        )

        await asyncio.wait_for(producer.start(), timeout=timeout_seconds)

        # Get cluster metadata
        cluster_metadata = producer.client.cluster

        duration_ms = (time.time() - start_time) * 1000

        # Get broker list (brokers() is a method in aiokafka 0.11.0)
        brokers = cluster_metadata.brokers()

        details = {
            "bootstrap_servers": kafka_bootstrap_servers,
            "response_time_ms": duration_ms,
            "broker_count": len(brokers),
            # Note: controller info not available via ClusterMetadata API in aiokafka 0.11.0
            "controller_id": None,
        }

        # If test topic is provided, try to get topic metadata
        if test_topic:
            try:
                topic_metadata = cluster_metadata.topics.get(test_topic)
                if topic_metadata:
                    details.update(
                        {
                            "test_topic": test_topic,
                            "topic_partition_count": len(topic_metadata.partitions),
                            "topic_available": True,
                        }
                    )
                else:
                    details.update(
                        {
                            "test_topic": test_topic,
                            "topic_available": False,
                        }
                    )
            except (KeyError, AttributeError, ValueError) as topic_error:
                # Expected errors when topic metadata unavailable
                logger.debug(
                    "Could not check test topic metadata",
                    topic=test_topic,
                    error=str(topic_error),
                )
                details.update(
                    {
                        "test_topic": test_topic,
                        "topic_check_error": str(topic_error),
                    }
                )

        # Determine health status based on broker availability
        broker_count = len(brokers)
        if broker_count == 0:
            return (HealthStatus.UNHEALTHY, "No Kafka brokers available", details)
        elif broker_count == 1:
            return (
                HealthStatus.DEGRADED,
                f"Kafka cluster healthy with {broker_count} broker (single point of failure)",
                details,
            )
        else:
            return (
                HealthStatus.HEALTHY,
                f"Kafka cluster healthy with {broker_count} brokers",
                details,
            )

    except TimeoutError:
        duration_ms = (time.time() - start_time) * 1000
        return (
            HealthStatus.UNHEALTHY,
            f"Kafka health check timed out after {timeout_seconds}s",
            {
                "timeout_seconds": timeout_seconds,
                "duration_ms": duration_ms,
                "bootstrap_servers": kafka_bootstrap_servers,
            },
        )

    except (KafkaConnectionError, KafkaTimeoutError) as e:
        duration_ms = (time.time() - start_time) * 1000
        return (
            HealthStatus.UNHEALTHY,
            f"Kafka connection failed: {e!s}",
            {
                "error": str(e),
                "error_type": type(e).__name__,
                "duration_ms": duration_ms,
                "bootstrap_servers": kafka_bootstrap_servers,
            },
        )

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return (
            HealthStatus.UNHEALTHY,
            f"Kafka health check failed: {e!s}",
            {
                "error": str(e),
                "error_type": type(e).__name__,
                "duration_ms": duration_ms,
                "bootstrap_servers": kafka_bootstrap_servers,
            },
        )

    finally:
        # Clean up connections
        if producer:
            try:
                await producer.stop()
            except (TimeoutError, KafkaError, KafkaTimeoutError) as cleanup_error:
                # Expected Kafka errors during cleanup
                logger.warning(
                    "Kafka producer cleanup failed", error=str(cleanup_error)
                )
            except Exception as cleanup_error:
                # Unexpected errors
                logger.error(
                    "Unexpected error cleaning up Kafka producer",
                    error=str(cleanup_error),
                    exc_info=True,
                )

        if consumer:
            try:
                await consumer.stop()
            except (TimeoutError, KafkaError, KafkaTimeoutError) as cleanup_error:
                # Expected Kafka errors during cleanup
                logger.warning(
                    "Kafka consumer cleanup failed", error=str(cleanup_error)
                )
            except Exception as cleanup_error:
                # Unexpected errors
                logger.error(
                    "Unexpected error cleaning up Kafka consumer",
                    error=str(cleanup_error),
                    exc_info=True,
                )


async def check_external_service_health(
    service_name: str,
    service_url: str,
    timeout_seconds: float = 5.0,
    expected_status_code: int = 200,
    health_endpoint: str = "/health",
) -> tuple[HealthStatus, str, dict[str, Any]]:
    """Check external service health via HTTP.

    Args:
        service_name: Name of the service for logging
        service_url: Base URL of the service
        timeout_seconds: Timeout for the health check
        expected_status_code: Expected HTTP status code
        health_endpoint: Health check endpoint path

    Returns:
        Tuple of (status, message, details)
    """
    import aiohttp

    start_time = time.time()
    full_url = f"{service_url.rstrip('/')}{health_endpoint}"

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout_seconds)
        ) as session:
            async with session.get(full_url) as response:
                duration_ms = (time.time() - start_time) * 1000
                response_text = await response.text()

                details = {
                    "service_name": service_name,
                    "service_url": service_url,
                    "health_endpoint": health_endpoint,
                    "full_url": full_url,
                    "response_time_ms": duration_ms,
                    "status_code": response.status,
                    "expected_status_code": expected_status_code,
                }

                if response.status == expected_status_code:
                    # Try to parse JSON response if available
                    try:
                        response_data = await response.json()
                        details["response_data"] = response_data
                    except (
                        ValueError,
                        TypeError,
                        aiohttp.ContentTypeError,
                    ) as json_error:
                        # JSON parsing failed - use text response
                        logger.debug(
                            f"Could not parse JSON response from {service_name}: {json_error}"
                        )
                        details["response_text"] = response_text[
                            :500
                        ]  # Truncate long responses

                    return (
                        HealthStatus.HEALTHY,
                        f"{service_name} service healthy (response time: {duration_ms:.1f}ms)",
                        details,
                    )
                elif 500 <= response.status < 600:
                    return (
                        HealthStatus.UNHEALTHY,
                        f"{service_name} service error (HTTP {response.status})",
                        details,
                    )
                else:
                    return (
                        HealthStatus.DEGRADED,
                        f"{service_name} service returned unexpected status (HTTP {response.status})",
                        details,
                    )

    except TimeoutError:
        duration_ms = (time.time() - start_time) * 1000
        return (
            HealthStatus.UNHEALTHY,
            f"{service_name} service health check timed out after {timeout_seconds}s",
            {
                "service_name": service_name,
                "timeout_seconds": timeout_seconds,
                "duration_ms": duration_ms,
                "full_url": full_url,
            },
        )

    except aiohttp.ClientConnectorError as e:
        duration_ms = (time.time() - start_time) * 1000
        return (
            HealthStatus.UNHEALTHY,
            f"{service_name} service connection failed: {e!s}",
            {
                "service_name": service_name,
                "error": str(e),
                "error_type": "connection_error",
                "duration_ms": duration_ms,
                "full_url": full_url,
            },
        )

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return (
            HealthStatus.UNHEALTHY,
            f"{service_name} service health check failed: {e!s}",
            {
                "service_name": service_name,
                "error": str(e),
                "error_type": type(e).__name__,
                "duration_ms": duration_ms,
                "full_url": full_url,
            },
        )


async def check_redis_health(
    redis_client: Any,
    timeout_seconds: float = 3.0,
) -> tuple[HealthStatus, str, dict[str, Any]]:
    """Check Redis health.

    Args:
        redis_client: Redis client instance
        timeout_seconds: Timeout for the health check

    Returns:
        Tuple of (status, message, details)
    """
    start_time = time.time()

    try:
        if not redis_client:
            return (
                HealthStatus.UNHEALTHY,
                "Redis client not initialized",
                {"client_available": False},
            )

        # Simple ping test
        result = await asyncio.wait_for(redis_client.ping(), timeout=timeout_seconds)

        duration_ms = (time.time() - start_time) * 1000

        if result:
            # Get additional info
            try:
                info = await redis_client.info()
                details = {
                    "response_time_ms": duration_ms,
                    "ping_successful": True,
                    "redis_version": info.get("redis_version"),
                    "connected_clients": info.get("connected_clients"),
                    "used_memory": info.get("used_memory"),
                    "used_memory_human": info.get("used_memory_human"),
                }
            except (TimeoutError, AttributeError, KeyError, RuntimeError) as info_error:
                # Expected errors when retrieving Redis info
                logger.debug(f"Could not retrieve Redis info: {info_error}")
                details = {
                    "response_time_ms": duration_ms,
                    "ping_successful": True,
                }

            return (
                HealthStatus.HEALTHY,
                f"Redis connection healthy (response time: {duration_ms:.1f}ms)",
                details,
            )
        else:
            return (
                HealthStatus.UNHEALTHY,
                "Redis ping returned False",
                {"response_time_ms": duration_ms, "ping_successful": False},
            )

    except TimeoutError:
        duration_ms = (time.time() - start_time) * 1000
        return (
            HealthStatus.UNHEALTHY,
            f"Redis health check timed out after {timeout_seconds}s",
            {"timeout_seconds": timeout_seconds, "duration_ms": duration_ms},
        )

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        return (
            HealthStatus.UNHEALTHY,
            f"Redis health check failed: {e!s}",
            {
                "error": str(e),
                "error_type": type(e).__name__,
                "duration_ms": duration_ms,
            },
        )
