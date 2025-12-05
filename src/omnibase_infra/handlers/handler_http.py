# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""HTTP REST Adapter - MVP implementation using httpx async client.

Supports GET and POST operations with 30-second fixed timeout.
PUT, DELETE, PATCH deferred to Beta. Retry logic and rate limiting deferred to Beta.
"""

from __future__ import annotations

import json
import logging
from typing import Optional
from uuid import UUID, uuid4

import httpx
from omnibase_core.enums.enum_handler_type import EnumHandlerType

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    ModelInfraErrorContext,
    RuntimeHostError,
)

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS: float = 30.0
_DEFAULT_MAX_REQUEST_SIZE: int = 10 * 1024 * 1024  # 10 MB
_DEFAULT_MAX_RESPONSE_SIZE: int = 50 * 1024 * 1024  # 50 MB
_SUPPORTED_OPERATIONS: frozenset[str] = frozenset({"http.get", "http.post"})


class HttpRestAdapter:
    """HTTP REST protocol adapter using httpx async client (MVP: GET, POST only)."""

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
        """
        try:
            self._timeout = _DEFAULT_TIMEOUT_SECONDS

            # Extract configurable size limits
            max_request_raw = config.get("max_request_size")
            if isinstance(max_request_raw, int) and max_request_raw > 0:
                self._max_request_size = max_request_raw

            max_response_raw = config.get("max_response_size")
            if isinstance(max_response_raw, int) and max_response_raw > 0:
                self._max_response_size = max_response_raw

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
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.HTTP,
                operation="initialize",
                target_name="http_rest_adapter",
            )
            raise RuntimeHostError(
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
            raise RuntimeHostError(
                "Missing or invalid 'operation' in envelope", context=ctx
            )

        if operation not in _SUPPORTED_OPERATIONS:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.HTTP,
                operation=operation,
                target_name="http_rest_adapter",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
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
            raise RuntimeHostError(
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
            raise RuntimeHostError("Missing or invalid 'url' in payload", context=ctx)

        headers = self._extract_headers(payload, operation, url, correlation_id)

        if operation == "http.get":
            return await self._execute_request(
                "GET", url, headers, None, correlation_id
            )
        else:  # http.post
            body = payload.get("body")
            # Validate request body size before sending
            self._validate_request_size(body, correlation_id)
            return await self._execute_request(
                "POST", url, headers, body, correlation_id
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
        raise RuntimeHostError(
            "Invalid 'headers' in payload - must be a dict", context=ctx
        )

    def _validate_request_size(self, body: object, correlation_id: UUID) -> None:
        """Validate request body size against configured limit.

        Args:
            body: Request body (str, dict, bytes, or None)
            correlation_id: Correlation ID for error context

        Raises:
            RuntimeHostError: If body size exceeds max_request_size limit.
        """
        if body is None:
            return

        size: int = 0
        if isinstance(body, str):
            size = len(body.encode("utf-8"))
        elif isinstance(body, dict):
            try:
                size = len(json.dumps(body).encode("utf-8"))
            except (TypeError, ValueError):
                # If we can't serialize, skip validation and let execute() handle it
                return
        elif isinstance(body, bytes):
            size = len(body)
        else:
            # Unknown type - skip validation, let execute() handle serialization
            return

        if size > self._max_request_size:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.HTTP,
                operation="validate_request_size",
                target_name="http_adapter",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                f"Request body size ({size} bytes) exceeds limit ({self._max_request_size} bytes)",
                context=ctx,
            )

    def _validate_response_size(self, body: object, correlation_id: UUID) -> None:
        """Validate response body size against configured limit.

        Args:
            body: Response body (str, dict, bytes, or None)
            correlation_id: Correlation ID for error context

        Raises:
            RuntimeHostError: If body size exceeds max_response_size limit.
        """
        if body is None:
            return

        size: int = 0
        if isinstance(body, str):
            size = len(body.encode("utf-8"))
        elif isinstance(body, dict):
            try:
                size = len(json.dumps(body).encode("utf-8"))
            except (TypeError, ValueError):
                # If we can't serialize, skip validation
                return
        elif isinstance(body, bytes):
            size = len(body)
        else:
            # Unknown type - skip validation
            return

        if size > self._max_response_size:
            logger.warning(
                "Response body size exceeds limit",
                extra={
                    "size": size,
                    "limit": self._max_response_size,
                    "correlation_id": str(correlation_id),
                },
            )
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.HTTP,
                operation="validate_response_size",
                target_name="http_adapter",
                correlation_id=correlation_id,
            )
            raise RuntimeHostError(
                f"Response body size ({size} bytes) exceeds limit ({self._max_response_size} bytes)",
                context=ctx,
            )

    async def _execute_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: object,
        correlation_id: UUID,
    ) -> dict[str, object]:
        """Execute HTTP request and handle errors."""
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

        try:
            if method == "GET":
                response = await self._client.get(url, headers=headers)
            elif body is None:
                response = await self._client.post(url, headers=headers)
            elif isinstance(body, dict):
                response = await self._client.post(url, headers=headers, json=body)
            elif isinstance(body, str):
                response = await self._client.post(url, headers=headers, content=body)
            else:
                try:
                    serialized_body = json.dumps(body)
                except TypeError as e:
                    raise RuntimeHostError(
                        f"Body is not JSON-serializable: {type(body).__name__}",
                        context=ctx,
                    ) from e
                response = await self._client.post(
                    url, headers=headers, content=serialized_body
                )

            return self._build_response(response, correlation_id)

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
        except httpx.HTTPStatusError as e:
            return self._build_response(e.response, correlation_id)
        except httpx.HTTPError as e:
            raise InfraConnectionError(
                f"HTTP error during {method} request: {type(e).__name__}", context=ctx
            ) from e

    def _build_response(
        self, response: httpx.Response, correlation_id: UUID
    ) -> dict[str, object]:
        """Build response envelope from httpx Response."""
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                body: object = response.json()
            except json.JSONDecodeError:
                body = response.text
        else:
            body = response.text

        # Validate response body size after receiving
        self._validate_response_size(body, correlation_id)

        return {
            "status": "success",
            "payload": {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": body,
            },
            "correlation_id": str(correlation_id),
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
