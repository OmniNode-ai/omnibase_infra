# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""HTTP REST Adapter - MVP implementation using httpx async client.

Supports GET and POST operations with 30-second fixed timeout.
PUT, DELETE, PATCH deferred to Beta. Retry logic and rate limiting deferred to Beta.
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Union
from uuid import UUID, uuid4

import httpx
from omnibase_core.enums.enum_handler_type import EnumHandlerType

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
    ModelInfraErrorContext,
    ProtocolConfigurationError,
    RuntimeHostError,
)

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS: float = 30.0
_DEFAULT_MAX_REQUEST_SIZE: int = 10 * 1024 * 1024  # 10 MB
_DEFAULT_MAX_RESPONSE_SIZE: int = 50 * 1024 * 1024  # 50 MB
_SUPPORTED_OPERATIONS: frozenset[str] = frozenset({"http.get", "http.post"})
# Streaming chunk size for responses without Content-Length header
_STREAMING_CHUNK_SIZE: int = 8192  # 8 KB chunks

# Size category thresholds for sanitized logging
_SIZE_THRESHOLD_KB: int = 1024  # 1 KB
_SIZE_THRESHOLD_MB: int = 1024 * 1024  # 1 MB
_SIZE_THRESHOLD_10MB: int = 10 * 1024 * 1024  # 10 MB


def _categorize_size(size: int) -> str:
    """Categorize byte size into security-safe categories.

    This prevents exact payload sizes from being exposed in error messages
    and logs, which could help attackers probe size limits.

    Args:
        size: Size in bytes

    Returns:
        Size category: "small", "medium", "large", or "very_large"
    """
    if size < _SIZE_THRESHOLD_KB:
        return "small"
    elif size < _SIZE_THRESHOLD_MB:
        return "medium"
    elif size < _SIZE_THRESHOLD_10MB:
        return "large"
    else:
        return "very_large"


class HttpRestAdapter:
    """HTTP REST protocol adapter using httpx async client (MVP: GET, POST only).

    Security Features:
        - Configurable request/response size limits to prevent DoS attacks
        - Pre-read Content-Length validation to prevent memory exhaustion
        - Streaming body validation for chunked transfer encoding
    """

    def __init__(self) -> None:
        """Initialize HttpRestAdapter in uninitialized state."""
        self._client: Optional[httpx.AsyncClient] = None
        self._timeout: float = _DEFAULT_TIMEOUT_SECONDS
        self._max_request_size: int = _DEFAULT_MAX_REQUEST_SIZE
        self._max_response_size: int = _DEFAULT_MAX_RESPONSE_SIZE
        self._initialized: bool = False

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return EnumHandlerType.HTTP."""
        return EnumHandlerType.HTTP

    async def initialize(self, config: dict[str, object]) -> None:
        """Initialize HTTP client with configurable timeout and size limits.

        Args:
            config: Configuration dict containing:
                - max_request_size: Optional max request body size in bytes (default: 10 MB)
                - max_response_size: Optional max response body size in bytes (default: 50 MB)
                - correlation_id: Optional UUID or string for error tracing

        Raises:
            ProtocolConfigurationError: If client initialization fails.

        Security:
            Size limits provide DoS (Denial of Service) protection by preventing:
            - Memory exhaustion from oversized request bodies
            - Memory exhaustion from malicious/misconfigured server responses
            - Resource starvation attacks via large payload processing

            Content-Length headers are validated BEFORE reading response bodies,
            and streaming validation protects against chunked transfer encoding attacks.

        Note:
            Size limit violations raise InfraUnavailableError with sanitized size
            categories (small/medium/large/very_large) - exact sizes are not exposed
            in error messages to prevent attackers from probing limits.
        """
        try:
            self._timeout = _DEFAULT_TIMEOUT_SECONDS

            # Extract configurable size limits
            max_request_raw = config.get("max_request_size")
            if max_request_raw is not None:
                if isinstance(max_request_raw, int) and max_request_raw > 0:
                    self._max_request_size = max_request_raw
                else:
                    logger.warning(
                        "Invalid max_request_size config value ignored, using default",
                        extra={
                            "provided_value": max_request_raw,
                            "default_value": self._max_request_size,
                        },
                    )

            max_response_raw = config.get("max_response_size")
            if max_response_raw is not None:
                if isinstance(max_response_raw, int) and max_response_raw > 0:
                    self._max_response_size = max_response_raw
                else:
                    logger.warning(
                        "Invalid max_response_size config value ignored, using default",
                        extra={
                            "provided_value": max_response_raw,
                            "default_value": self._max_response_size,
                        },
                    )

            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                follow_redirects=True,
            )
            self._initialized = True
            logger.info(
                "HttpRestAdapter initialized",
                extra={
                    "timeout_seconds": self._timeout,
                    "max_request_size": self._max_request_size,
                    "max_response_size": self._max_response_size,
                },
            )
        except Exception as e:
            # Extract correlation_id from config if provided, otherwise generate new
            raw_correlation_id = config.get("correlation_id")
            if isinstance(raw_correlation_id, UUID):
                error_correlation_id = raw_correlation_id
            elif isinstance(raw_correlation_id, str):
                try:
                    error_correlation_id = UUID(raw_correlation_id)
                except ValueError:
                    error_correlation_id = uuid4()
            else:
                error_correlation_id = uuid4()

            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.HTTP,
                operation="initialize",
                target_name="http_rest_adapter",
                correlation_id=error_correlation_id,
            )
            raise ProtocolConfigurationError(
                "Failed to initialize HTTP adapter", context=ctx
            ) from e

    async def shutdown(self) -> None:
        """Close HTTP client and release resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._initialized = False
        logger.info("HttpRestAdapter shutdown complete")

    async def execute(self, envelope: dict[str, object]) -> dict[str, object]:
        """Execute HTTP operation (http.get or http.post) from envelope."""
        correlation_id = self._extract_correlation_id(envelope)

        if not self._initialized or self._client is None:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.HTTP,
                operation="execute",
                target_name="http_rest_adapter",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "HttpRestAdapter not initialized. Call initialize() first.", context=ctx
            )

        operation = envelope.get("operation")
        if not isinstance(operation, str):
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.HTTP,
                operation="execute",
                target_name="http_rest_adapter",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                "Missing or invalid 'operation' in envelope", context=ctx
            )

        if operation not in _SUPPORTED_OPERATIONS:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.HTTP,
                operation=operation,
                target_name="http_rest_adapter",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                f"Operation '{operation}' not supported in MVP. Available: {', '.join(sorted(_SUPPORTED_OPERATIONS))}",
                context=ctx,
            )

        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.HTTP,
                operation=operation,
                target_name="http_rest_adapter",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                "Missing or invalid 'payload' in envelope", context=ctx
            )

        url = payload.get("url")
        if not isinstance(url, str) or not url:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.HTTP,
                operation=operation,
                target_name="http_rest_adapter",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                "Missing or invalid 'url' in payload", context=ctx
            )

        headers = self._extract_headers(payload, operation, url, correlation_id)

        if operation == "http.get":
            return await self._execute_request(
                "GET", url, headers, None, correlation_id, None
            )
        else:  # http.post
            body = payload.get("body")
            # Validate request body size and get pre-serialized bytes for dict bodies.
            # This avoids double serialization - dict bodies are serialized once here
            # and the cached bytes are passed to _execute_request().
            pre_serialized = self._validate_request_size(body, correlation_id)
            return await self._execute_request(
                "POST", url, headers, body, correlation_id, pre_serialized
            )

    def _extract_correlation_id(self, envelope: dict[str, object]) -> UUID:
        """Extract or generate correlation ID from envelope."""
        raw = envelope.get("correlation_id")
        if isinstance(raw, UUID):
            return raw
        if isinstance(raw, str):
            try:
                return UUID(raw)
            except ValueError:
                pass
        return uuid4()

    def _extract_headers(
        self, payload: dict[str, object], operation: str, url: str, correlation_id: UUID
    ) -> dict[str, str]:
        """Extract and validate headers from payload."""
        headers_raw = payload.get("headers")
        if headers_raw is None:
            return {}
        if isinstance(headers_raw, dict):
            return {str(k): str(v) for k, v in headers_raw.items()}
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.HTTP,
            operation=operation,
            target_name=url,
            correlation_id=correlation_id,
        )
        raise ProtocolConfigurationError(
            "Invalid 'headers' in payload - must be a dict", context=ctx
        )

    def _validate_request_size(
        self, body: object, correlation_id: UUID
    ) -> Optional[bytes]:
        """Validate request body size and cache serialized bytes for dict bodies.

        For dict bodies, this method serializes the body once and returns the
        serialized bytes. This avoids double serialization - the returned bytes
        can be passed directly to _execute_request() instead of re-serializing.

        Args:
            body: Request body (str, dict, bytes, or None)
            correlation_id: Correlation ID for error context

        Returns:
            For dict bodies: The pre-serialized JSON bytes (cached for reuse)
            For other body types: None (no caching needed)

        Raises:
            InfraUnavailableError: If body size exceeds max_request_size limit.
        """
        if body is None:
            return None

        size: int = 0
        serialized_bytes: Optional[bytes] = None

        if isinstance(body, str):
            size = len(body.encode("utf-8"))
        elif isinstance(body, dict):
            # DESIGN TRADEOFF: Double-Serialization Avoidance
            #
            # We serialize dict bodies once here during validation and cache the bytes.
            # The cached bytes are then passed to _execute_request() via the return value,
            # avoiding re-serialization when building the HTTP request.
            #
            # Tradeoff considerations:
            # - Memory: Serialized bytes are held in memory during validation and request
            #   execution. For large payloads near the size limit, this adds ~10MB overhead.
            # - Performance: Single serialization is faster than serializing twice (once
            #   for size check, once for request body).
            # - Alternative: We could serialize twice (once here for size, once in execute)
            #   which would use less peak memory but double the CPU cost for serialization.
            #
            # Current approach prioritizes CPU efficiency over peak memory usage, which is
            # appropriate since we enforce max_request_size limits anyway.
            try:
                serialized_bytes = json.dumps(body).encode("utf-8")
                size = len(serialized_bytes)
            except (TypeError, ValueError):
                # If we can't serialize, skip validation and let execute() handle it
                return None
        elif isinstance(body, bytes):
            size = len(body)
        else:
            # Unknown type - skip validation, let execute() handle serialization
            return None

        if size > self._max_request_size:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.HTTP,
                operation="validate_request_size",
                target_name="http_adapter",
                correlation_id=correlation_id,
            )
            raise InfraUnavailableError(
                f"Request body size ({_categorize_size(size)}) exceeds configured limit",
                context=ctx,
            )

        logger.debug(
            "Request size validated",
            extra={
                "request_size_category": _categorize_size(size),
                "limit": self._max_request_size,
                "correlation_id": str(correlation_id),
            },
        )

        # Return cached serialized bytes for dict bodies, None for other types
        return serialized_bytes

    def _validate_content_length_header(
        self, response: httpx.Response, url: str, correlation_id: UUID
    ) -> None:
        """Validate Content-Length header BEFORE reading response body.

        This prevents memory exhaustion by rejecting large responses before
        they are loaded into memory. This is critical for security as it
        prevents denial-of-service attacks via large response payloads.

        Args:
            response: The httpx Response object (body not yet read)
            url: Target URL for error context
            correlation_id: Correlation ID for error context

        Raises:
            InfraUnavailableError: If Content-Length exceeds max_response_size limit.
        """
        content_length_header = response.headers.get("content-length")
        if content_length_header is None:
            # No Content-Length header - will use streaming validation
            return

        try:
            content_length = int(content_length_header)
        except ValueError:
            # Invalid Content-Length header - log warning and proceed with body-based validation
            logger.warning(
                "Invalid Content-Length header value",
                extra={
                    "content_length_header": content_length_header,
                    "url": url,
                    "correlation_id": str(correlation_id),
                },
            )
            return

        if content_length < 0:
            # Negative Content-Length is invalid per HTTP spec - log warning and proceed
            # with body-based validation (streaming protection still applies)
            logger.warning(
                "Negative Content-Length header value ignored",
                extra={
                    "content_length_header": content_length_header,
                    "url": url,
                    "correlation_id": str(correlation_id),
                },
            )
            return

        if content_length > self._max_response_size:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.HTTP,
                operation="validate_content_length",
                target_name=url,
                correlation_id=correlation_id,
            )
            raise InfraUnavailableError(
                f"Response Content-Length ({_categorize_size(content_length)}) exceeds configured limit",
                context=ctx,
            )

        logger.debug(
            "Content-Length header validated",
            extra={
                "content_length_category": _categorize_size(content_length),
                "limit": self._max_response_size,
                "url": url,
                "correlation_id": str(correlation_id),
            },
        )

    async def _read_response_body_with_limit(
        self, response: httpx.Response, url: str, correlation_id: UUID
    ) -> bytes:
        """Read response body with streaming size limit enforcement.

        For responses without Content-Length header (e.g., chunked transfer encoding),
        this method reads the body in chunks and tracks the total size, stopping
        and raising an error if the limit is exceeded.

        Security Note: This method provides DoS protection for chunked transfer
        encoding responses where Content-Length is not available for pre-read
        validation. By enforcing limits during streaming, we prevent memory
        exhaustion attacks from maliciously large chunked responses.

        Args:
            response: The httpx Response object
            url: Target URL for error context
            correlation_id: Correlation ID for error context

        Returns:
            The complete response body as bytes

        Raises:
            InfraUnavailableError: If body size exceeds max_response_size during streaming.
        """
        chunks: list[bytes] = []
        total_size: int = 0

        async for chunk in response.aiter_bytes(chunk_size=_STREAMING_CHUNK_SIZE):
            total_size += len(chunk)
            if total_size > self._max_response_size:
                logger.warning(
                    "Response body exceeded size limit during streaming read",
                    extra={
                        "size_category": _categorize_size(total_size),
                        "limit": self._max_response_size,
                        "url": url,
                        "correlation_id": str(correlation_id),
                    },
                )
                ctx = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.HTTP,
                    operation="read_response_body",
                    target_name=url,
                    correlation_id=correlation_id,
                )
                raise InfraUnavailableError(
                    f"Response body size ({_categorize_size(total_size)}) exceeds configured limit during streaming read",
                    context=ctx,
                )
            chunks.append(chunk)

        return b"".join(chunks)

    async def _execute_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: object,
        correlation_id: UUID,
        pre_serialized: Optional[bytes] = None,
    ) -> dict[str, object]:
        """Execute HTTP request with pre-read response size validation.

        Uses httpx streaming to validate Content-Length header BEFORE reading
        the response body into memory, preventing memory exhaustion attacks.

        Args:
            method: HTTP method (GET, POST)
            url: Target URL
            headers: Request headers
            body: Request body (used only if pre_serialized is None)
            correlation_id: Correlation ID for tracing
            pre_serialized: Pre-serialized JSON bytes for dict bodies (from
                _validate_request_size). When provided, this is used directly
                instead of re-serializing the body, avoiding double serialization.
        """
        if self._client is None:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.HTTP,
                operation=f"http.{method.lower()}",
                target_name=url,
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                "HttpRestAdapter not initialized - call initialize() first", context=ctx
            )

        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.HTTP,
            operation=f"http.{method.lower()}",
            target_name=url,
            correlation_id=correlation_id,
        )

        # Prepare request content for POST
        request_content: Optional[Union[bytes, str]] = None
        request_json: Optional[dict[str, object]] = None
        request_headers = dict(headers)  # Copy to avoid mutating caller's headers

        if method == "POST" and body is not None:
            if pre_serialized is not None:
                # Use pre-serialized bytes from _validate_request_size to avoid
                # double serialization. Set Content-Type header since we're using
                # content= instead of json= parameter.
                request_content = pre_serialized
                if "content-type" not in {k.lower() for k in request_headers}:
                    request_headers["Content-Type"] = "application/json"
            elif isinstance(body, dict):
                # Fallback for dict bodies without pre-serialized content
                # (shouldn't happen in normal flow, but handles edge cases)
                request_json = body
            elif isinstance(body, str):
                request_content = body
            else:
                try:
                    request_content = json.dumps(body)
                except TypeError as e:
                    raise ProtocolConfigurationError(
                        f"Body is not JSON-serializable: {type(body).__name__}",
                        context=ctx,
                    ) from e

        try:
            # Use streaming request to get response headers before reading body
            # This allows us to check Content-Length before loading body into memory
            async with self._client.stream(
                method,
                url,
                headers=request_headers,
                content=request_content,
                json=request_json,
            ) as response:
                # CRITICAL: Validate Content-Length header BEFORE reading body
                # This prevents memory exhaustion from large responses
                self._validate_content_length_header(response, url, correlation_id)

                # Read body with streaming size limit enforcement
                # For responses without Content-Length, this stops early if limit exceeded
                response_body_bytes = await self._read_response_body_with_limit(
                    response, url, correlation_id
                )

                return self._build_response_from_bytes(
                    response, response_body_bytes, correlation_id
                )

        except httpx.TimeoutException as e:
            raise InfraTimeoutError(
                f"HTTP {method} request timed out after {self._timeout}s",
                context=ctx,
                timeout_seconds=self._timeout,
            ) from e
        except httpx.ConnectError as e:
            raise InfraConnectionError(
                f"Failed to connect to {url}", context=ctx
            ) from e
        except httpx.HTTPError as e:
            raise InfraConnectionError(
                f"HTTP error during {method} request: {type(e).__name__}", context=ctx
            ) from e

    def _build_response_from_bytes(
        self,
        response: httpx.Response,
        body_bytes: bytes,
        correlation_id: UUID,
    ) -> dict[str, object]:
        """Build response envelope from httpx Response and pre-read body bytes.

        This method is used with streaming responses where the body has already
        been read with size limit enforcement.

        Args:
            response: The httpx Response object (headers already available)
            body_bytes: The pre-read response body bytes
            correlation_id: Correlation ID for tracing

        Returns:
            Response envelope dict with status, payload, and correlation_id
        """
        content_type = response.headers.get("content-type", "")
        body: object

        # TODO(Beta): When rate limiting is implemented, extract and log rate limit
        # response headers: x-ratelimit-remaining, x-ratelimit-limit, x-ratelimit-reset
        # These headers will be added to the debug log metadata below for observability.

        logger.debug(
            "Response body received",
            extra={
                "body_size": len(body_bytes),
                "size_utilization_pct": round(
                    (len(body_bytes) / self._max_response_size) * 100, 2
                ),
                "content_type": content_type,
                "status_code": response.status_code,
                "correlation_id": str(correlation_id),
                # TODO(Beta): Add rate limit metadata here when rate limiting is implemented:
                # "ratelimit_remaining": response.headers.get("x-ratelimit-remaining"),
                # "ratelimit_limit": response.headers.get("x-ratelimit-limit"),
                # "ratelimit_reset": response.headers.get("x-ratelimit-reset"),
            },
        )

        # Decode bytes to string first
        try:
            body_text = body_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try latin-1 as fallback
            body_text = body_bytes.decode("latin-1")

        # Try to parse as JSON if content type indicates JSON
        if "application/json" in content_type:
            try:
                body = json.loads(body_text)
            except json.JSONDecodeError:
                body = body_text
        else:
            body = body_text

        return {
            "status": "success",
            "payload": {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": body,
            },
            "correlation_id": correlation_id,
        }

    async def health_check(self) -> dict[str, object]:
        """Return adapter health status."""
        return {
            "healthy": self._initialized and self._client is not None,
            "initialized": self._initialized,
            "adapter_type": self.handler_type.value,
            "timeout_seconds": self._timeout,
            "max_request_size": self._max_request_size,
            "max_response_size": self._max_response_size,
        }

    def describe(self) -> dict[str, object]:
        """Return adapter metadata and capabilities."""
        return {
            "adapter_type": self.handler_type.value,
            "supported_operations": sorted(_SUPPORTED_OPERATIONS),
            "timeout_seconds": self._timeout,
            "max_request_size": self._max_request_size,
            "max_response_size": self._max_response_size,
            "initialized": self._initialized,
            "version": "0.1.0-mvp",
        }


__all__: list[str] = ["HttpRestAdapter"]
