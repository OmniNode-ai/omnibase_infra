"""Base service client with common patterns for resilience and observability.

Provides:
    - Circuit breaker integration
    - Exponential backoff retry logic
    - Timeout handling
    - Structured logging
    - Metrics collection
    - Correlation ID propagation
    - Health check interface
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional
from uuid import UUID

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError

logger = logging.getLogger(__name__)


class ClientError(Exception):
    """Base exception for client errors."""

    pass


class ServiceUnavailableError(ClientError):
    """Raised when service is unavailable."""

    pass


class BaseServiceClient(ABC):
    """Abstract base class for service clients.

    Provides common functionality:
        - HTTP client management
        - Circuit breaker protection
        - Retry logic with exponential backoff
        - Timeout handling
        - Correlation ID propagation
        - Health check interface
        - Metrics collection
    """

    def __init__(
        self,
        base_url: str,
        service_name: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        app: Optional[Any] = None,
    ):
        """Initialize base service client.

        Args:
            base_url: Base URL for the service (e.g., http://localhost:8053)
            service_name: Name of the service for logging/metrics
            timeout: Default timeout for requests in seconds
            max_retries: Maximum number of retry attempts
            circuit_breaker_config: Optional circuit breaker configuration
            app: Optional ASGI app for testing (bypasses network calls)
        """
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.service_name = service_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.app = app

        # HTTP client with connection pooling
        self._http_client: Optional[httpx.AsyncClient] = None

        # Circuit breaker for resilience
        cb_config = circuit_breaker_config or CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60,
            timeout=timeout,
        )
        self.circuit_breaker = CircuitBreaker(
            name=f"{service_name}_circuit_breaker",
            config=cb_config,
        )

        # Metrics
        self._request_count = 0
        self._error_count = 0
        self._total_response_time = 0.0

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def initialize(self) -> None:
        """Initialize client resources (HTTP client, connections, etc.)."""
        if self._http_client is None:
            # Use ASGI transport if app provided (for testing), otherwise HTTP
            if self.app is not None:
                self._http_client = httpx.AsyncClient(
                    app=self.app,
                    base_url="http://testserver",
                    timeout=httpx.Timeout(self.timeout),
                )
                logger.info(
                    f"Initialized {self.service_name} client with ASGI transport",
                    extra={"mode": "test"},
                )
            else:
                self._http_client = httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=httpx.Timeout(self.timeout),
                    limits=httpx.Limits(
                        max_connections=100,
                        max_keepalive_connections=20,
                    ),
                )
                logger.info(
                    f"Initialized {self.service_name} client",
                    extra={"base_url": self.base_url},
                )

    async def close(self) -> None:
        """Close client resources."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
            logger.info(f"Closed {self.service_name} client")

    def _get_correlation_headers(self, correlation_id: Optional[UUID] = None) -> dict:
        """Get headers with correlation ID.

        Args:
            correlation_id: Optional correlation ID for request tracing

        Returns:
            Dictionary with correlation headers
        """
        headers = {
            "User-Agent": f"omninode-bridge/{self.service_name}",
        }

        if correlation_id is not None:
            headers["X-Correlation-ID"] = str(correlation_id)

        return headers

    async def _make_request_with_retry(
        self,
        method: str,
        endpoint: str,
        correlation_id: Optional[UUID] = None,
        allow_statuses: Optional[tuple[int, ...]] = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make HTTP request with retry logic and circuit breaker.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., /health)
            correlation_id: Optional correlation ID for tracing
            allow_statuses: Optional tuple of status codes that bypass raise_for_status
            **kwargs: Additional arguments for httpx request

        Returns:
            httpx.Response object

        Raises:
            ServiceUnavailableError: If service is unavailable
            CircuitBreakerError: If circuit breaker is open
            ClientError: For other client errors
        """
        if self._http_client is None:
            await self.initialize()

        # Add correlation headers
        headers = kwargs.pop("headers", {})
        headers.update(self._get_correlation_headers(correlation_id))

        url = f"{self.base_url}{endpoint}"

        # Retry configuration with exponential backoff
        retry_policy = AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
            reraise=True,
        )

        try:
            # Execute with circuit breaker protection
            async def _do_request():
                async for attempt in retry_policy:
                    with attempt:
                        response = await self._http_client.request(
                            method=method,
                            url=url,
                            headers=headers,
                            **kwargs,
                        )
                        # Skip raise_for_status for whitelisted status codes
                        if allow_statuses and response.status_code in allow_statuses:
                            return response
                        response.raise_for_status()
                        return response

            result = await self.circuit_breaker.call(_do_request)
            self._request_count += 1
            return result

        except CircuitBreakerError as e:
            self._error_count += 1
            logger.error(
                f"{self.service_name} circuit breaker is open",
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "endpoint": endpoint,
                },
            )
            raise ServiceUnavailableError(
                f"{self.service_name} is currently unavailable"
            ) from e

        except httpx.TimeoutException as e:
            self._error_count += 1
            logger.error(
                f"{self.service_name} request timeout",
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "endpoint": endpoint,
                    "timeout": self.timeout,
                },
            )
            raise ServiceUnavailableError(
                f"{self.service_name} request timed out"
            ) from e

        except httpx.HTTPStatusError as e:
            self._error_count += 1
            logger.error(
                f"{self.service_name} HTTP error",
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "endpoint": endpoint,
                    "status_code": e.response.status_code,
                },
            )
            raise ClientError(
                f"{self.service_name} returned error: {e.response.status_code}"
            ) from e

        except Exception as e:
            self._error_count += 1
            logger.exception(
                f"{self.service_name} unexpected error",
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "endpoint": endpoint,
                },
            )
            raise ClientError(f"{self.service_name} error: {e!s}") from e

    async def health_check(self) -> dict[str, Any]:
        """Check service health.

        Returns:
            Health status dictionary

        Raises:
            ServiceUnavailableError: If health check fails
        """
        try:
            response = await self._make_request_with_retry(
                method="GET",
                endpoint="/health",
            )
            return response.json()

        except Exception as e:
            logger.error(f"{self.service_name} health check failed: {e}")
            raise ServiceUnavailableError(
                f"{self.service_name} health check failed"
            ) from e

    def get_metrics(self) -> dict[str, Any]:
        """Get client metrics.

        Returns:
            Dictionary with client metrics including circuit breaker state
        """
        cb_metrics = self.circuit_breaker.get_metrics()

        return {
            "service_name": self.service_name,
            "base_url": self.base_url,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "error_rate": (
                self._error_count / self._request_count
                if self._request_count > 0
                else 0
            ),
            "circuit_breaker": cb_metrics.to_dict(),
        }

    @abstractmethod
    async def _validate_connection(self) -> bool:
        """Validate client connection to service.

        Subclasses should implement service-specific validation.

        Returns:
            True if connection is valid, False otherwise
        """
        pass
