# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""LLM HTTP Transport Mixin for infrastructure components.

This module provides a reusable mixin class that encapsulates HTTP transport
logic for communicating with LLM providers (OpenAI-compatible APIs, vLLM,
local inference servers, etc.).

Features:
    - Self-contained retry loop with exponential backoff
    - Circuit breaker integration via MixinAsyncCircuitBreaker
    - HTTP status code to typed exception mapping
    - Retry-After header parsing for 429 rate limit responses
    - Lazy httpx.AsyncClient management (create or inject)
    - Content-type validation for JSON responses (case-insensitive)
    - Correlation ID propagation on all errors
    - CIDR allowlist validation for local LLM endpoints
    - HMAC request signing for trust boundary enforcement

Security:
    - Response bodies are sanitized via ``sanitize_error_string()`` before
      inclusion in error context or exception messages, preventing accidental
      leakage of secrets or PII through error propagation paths.
    - CIDR allowlist (default: ``192.168.86.0/24``, configurable via
      ``LLM_ENDPOINT_CIDR_ALLOWLIST``) restricts outbound LLM calls to the
      local network trust boundary. Requests to IPs outside the configured
      ranges are rejected before any HTTP call is made (fail-closed).
    - HMAC-SHA256 request signing using the ``LOCAL_LLM_SHARED_SECRET``
      environment variable adds an ``x-omn-node-signature`` header to all
      outbound requests. If the secret is not configured, requests are
      rejected (fail-closed).

Design Rationale:
    This mixin extracts common HTTP transport patterns from LLM-calling nodes
    into a reusable component. It builds on MixinAsyncCircuitBreaker and
    MixinRetryExecution to provide a high-level ``_execute_llm_http_call``
    method that handles the full lifecycle of an HTTP POST to an LLM endpoint.

Error Mapping:
    HTTP status codes are mapped to typed infrastructure exceptions:
    - 401/403 -> InfraAuthenticationError (no retry, no CB failure)
    - 404     -> ProtocolConfigurationError (assumed misconfiguration)
    - 429     -> InfraRateLimitedError (retry with Retry-After)
    - 400/422 -> InfraRequestRejectedError (provider rejection)
    - 500-504 -> InfraUnavailableError (retry, CB failure)
    - Other   -> InfraUnavailableError (retry, CB failure)

See Also:
    - MixinAsyncCircuitBreaker for circuit breaker state management
    - MixinRetryExecution for retry pattern abstractions
    - docs/patterns/error_recovery_patterns.md for retry documentation

.. versionadded:: 0.7.0
    Part of OMN-2104 LLM HTTP transport.

.. versionchanged:: 0.8.0
    Added CIDR allowlist and HMAC signing (OMN-2250).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json as json_module
import logging
import math
import os
import socket
from ipaddress import IPv4Address, IPv4Network, ip_address
from json import JSONDecodeError
from typing import TYPE_CHECKING, ClassVar, cast
from urllib.parse import urlparse
from uuid import UUID

import httpx

from omnibase_core.types import JsonType
from omnibase_infra.enums import EnumInfraTransportType, EnumRetryErrorCategory
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraProtocolError,
    InfraRateLimitedError,
    InfraRequestRejectedError,
    InfraTimeoutError,
    InfraUnavailableError,
    ModelInfraErrorContext,
    ModelTimeoutErrorContext,
    ProtocolConfigurationError,
    RuntimeHostError,
)
from omnibase_infra.mixins.mixin_async_circuit_breaker import MixinAsyncCircuitBreaker
from omnibase_infra.mixins.mixin_retry_execution import MixinRetryExecution
from omnibase_infra.models.model_retry_error_classification import (
    ModelRetryErrorClassification,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_string

if TYPE_CHECKING:
    from omnibase_infra.handlers.models.model_retry_state import ModelRetryState

logger = logging.getLogger(__name__)

_DEFAULT_CIDR = "192.168.86.0/24"


def _parse_cidr_allowlist() -> tuple[IPv4Network, ...]:
    """Parse CIDR allowlist from the ``LLM_ENDPOINT_CIDR_ALLOWLIST`` env var.

    When the env var is not set, logs a WARNING and falls back to the
    default ``192.168.86.0/24``.  Parses each comma-separated value as
    an ``IPv4Network``. Malformed entries are logged at WARNING level and
    skipped. If **all** entries are malformed (or the env var is empty
    after parsing), falls back to the default and logs a warning.

    Returns:
        Tuple of parsed ``IPv4Network`` objects, never empty.
    """
    raw = os.environ.get("LLM_ENDPOINT_CIDR_ALLOWLIST")
    if raw is None:
        logger.warning(
            "LLM_ENDPOINT_CIDR_ALLOWLIST not set — using default %s",
            _DEFAULT_CIDR,
        )
        raw = _DEFAULT_CIDR
    parsed: list[IPv4Network] = []
    for entry in raw.split(","):
        cidr = entry.strip()
        if not cidr:
            continue
        try:
            parsed.append(IPv4Network(cidr, strict=False))
        except ValueError:
            logger.warning(
                "Skipping malformed CIDR in LLM_ENDPOINT_CIDR_ALLOWLIST: %r",
                cidr,
            )
    if not parsed:
        logger.warning(
            "All entries in LLM_ENDPOINT_CIDR_ALLOWLIST were malformed or "
            "empty; falling back to default %s",
            _DEFAULT_CIDR,
        )
        parsed.append(IPv4Network(_DEFAULT_CIDR))
    return tuple(parsed)


class MixinLlmHttpTransport(MixinAsyncCircuitBreaker, MixinRetryExecution):
    """HTTP transport mixin for LLM provider communication.

    Provides ``_execute_llm_http_call`` for making resilient HTTP POST requests
    to LLM endpoints with retry logic, circuit breaker protection, and typed
    error handling.

    This mixin implements the abstract methods required by MixinRetryExecution
    (``_classify_error``, ``_get_transport_type``, ``_get_target_name``) with
    HTTP/LLM-specific error classification.

    Required Initialization:
        Call ``_init_llm_http_transport()`` during subclass ``__init__``.

    Example:
        ```python
        class NodeLlmInferenceEffect(NodeEffect, MixinLlmHttpTransport):
            def __init__(self, container: ModelONEXContainer) -> None:
                super().__init__(container)
                self._init_llm_http_transport(
                    target_name="vllm-coder",
                    max_timeout_seconds=120.0,
                )

            async def execute_effect(self, input_data):
                result = await self._execute_llm_http_call(
                    url="http://192.168.86.201:8000/v1/chat/completions",
                    payload={"messages": [...]},
                    correlation_id=input_data.correlation_id,
                )
                return result
        ```
    """

    # ── Class-level constants ────────────────────────────────────────────

    #: CIDR networks defining the local LLM trust boundary.
    #: Only endpoints within these networks are permitted.
    #: Configurable via the ``LLM_ENDPOINT_CIDR_ALLOWLIST`` environment variable
    #: (comma-separated CIDR ranges). Defaults to ``192.168.86.0/24``.
    #:
    #: .. important:: Configuration reload asymmetry
    #:
    #:    This value is **parsed once at module import time** by
    #:    ``_parse_cidr_allowlist()`` and stored as an immutable class variable.
    #:    Changes to the ``LLM_ENDPOINT_CIDR_ALLOWLIST`` environment variable
    #:    after the module has been imported have **no effect** until the
    #:    process is restarted.
    #:
    #:    This differs intentionally from ``LOCAL_LLM_SHARED_SECRET``, which
    #:    is read from ``os.environ`` on **every call** to
    #:    ``_compute_hmac_signature()`` so that the secret can be rotated at
    #:    runtime (e.g., by a sidecar, operator, or orchestrator updating the
    #:    environment) without requiring a process restart.
    #:
    #:    **Rationale**: CIDR allowlist changes are rare infrastructure-level
    #:    modifications (adding or removing a network segment) that typically
    #:    accompany a deployment or topology change -- situations where a
    #:    process restart is already expected. In contrast, secret rotation
    #:    is a routine security operation that should complete without service
    #:    interruption; requiring a restart for secret rotation would create
    #:    unnecessary downtime and discourage frequent rotation.
    # Parsed at import time; use _reload_cidr_allowlist() to refresh after env changes.
    LOCAL_LLM_CIDRS: ClassVar[tuple[IPv4Network, ...]] = _parse_cidr_allowlist()

    @classmethod
    def _reload_cidr_allowlist(cls) -> None:
        """Re-parse ``LLM_ENDPOINT_CIDR_ALLOWLIST`` and update ``LOCAL_LLM_CIDRS``.

        Primarily for testing: after modifying the ``LLM_ENDPOINT_CIDR_ALLOWLIST``
        environment variable at runtime, call this method to refresh the cached
        CIDR allowlist without restarting the process.

        Example::

            os.environ["LLM_ENDPOINT_CIDR_ALLOWLIST"] = "10.0.0.0/8"
            MixinLlmHttpTransport._reload_cidr_allowlist()
            # LOCAL_LLM_CIDRS now contains IPv4Network('10.0.0.0/8')
        """
        cls.LOCAL_LLM_CIDRS = _parse_cidr_allowlist()

    #: Environment variable name for the HMAC shared secret.
    LOCAL_LLM_SECRET_ENV: ClassVar[str] = "LOCAL_LLM_SHARED_SECRET"

    #: HTTP header name for the HMAC signature.
    HMAC_HEADER: ClassVar[str] = "x-omn-node-signature"

    # Type hints for instance attributes set by _init_llm_http_transport
    _llm_target_name: str
    _max_timeout_seconds: float
    _max_retry_after_seconds: float
    _http_client: httpx.AsyncClient | None
    _owns_http_client: bool
    _http_client_lock: asyncio.Lock

    def _init_llm_http_transport(
        self,
        target_name: str,
        max_timeout_seconds: float = 120.0,
        max_retry_after_seconds: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize LLM HTTP transport with circuit breaker and client management.

        Circuit breaker uses defaults (threshold=5, reset_timeout=60.0).
        For custom CB configuration, call ``_init_circuit_breaker()`` or
        ``_init_circuit_breaker_from_config()`` after this method.

        Args:
            target_name: Identifier for the LLM target (e.g., "vllm-coder").
                Used in error context and logging.
            max_timeout_seconds: Maximum allowed timeout for any single request.
                Per-call timeouts are clamped to this value. Default: 120.0.
            max_retry_after_seconds: Maximum Retry-After delay to honor from
                429 responses. Values above this cap are clamped. Default: 30.0.
            http_client: Optional pre-configured httpx.AsyncClient. If None,
                a client is created lazily on first use. When provided, the
                caller retains ownership and must close it.
        """
        # Initialize circuit breaker from parent mixin
        self._init_circuit_breaker(
            threshold=5,
            reset_timeout=60.0,
            service_name=target_name,
            transport_type=EnumInfraTransportType.HTTP,
            half_open_successes=1,
        )
        # MixinAsyncCircuitBreaker._init_circuit_breaker does not set this flag;
        # it is required by MixinRetryExecution to gate circuit breaker helper calls.
        self._circuit_breaker_initialized = True

        # Satisfy MixinRetryExecution type hint
        self._executor = None

        # Transport configuration
        self._llm_target_name = target_name
        self._max_timeout_seconds = max_timeout_seconds
        self._max_retry_after_seconds = max_retry_after_seconds

        # HTTP client management
        self._http_client_lock = asyncio.Lock()
        if http_client is not None:
            self._http_client = http_client
            self._owns_http_client = False
        else:
            self._http_client = None
            self._owns_http_client = True

    # ── MixinRetryExecution abstract method implementations ──────────────

    def _classify_error(
        self, error: Exception, operation: str
    ) -> ModelRetryErrorClassification:
        """Classify an exception for retry handling.

        Maps both raw httpx exceptions and typed infrastructure exceptions
        to retry classifications.

        Args:
            error: The exception to classify.
            operation: The operation name for context.

        Returns:
            ModelRetryErrorClassification with retry decision and error details.
        """
        if isinstance(error, httpx.ConnectError):
            return ModelRetryErrorClassification(
                category=EnumRetryErrorCategory.CONNECTION,
                should_retry=True,
                record_circuit_failure=True,
                error_message=f"Connection error during {operation}: {error}",
            )

        if isinstance(error, httpx.TimeoutException):
            return ModelRetryErrorClassification(
                category=EnumRetryErrorCategory.TIMEOUT,
                should_retry=True,
                record_circuit_failure=True,
                error_message=f"Timeout during {operation}: {error}",
            )

        if isinstance(error, InfraAuthenticationError):
            return ModelRetryErrorClassification(
                category=EnumRetryErrorCategory.AUTHENTICATION,
                should_retry=False,
                record_circuit_failure=False,
                error_message=f"Authentication failed during {operation}",
            )

        if isinstance(error, InfraRateLimitedError):
            return ModelRetryErrorClassification(
                category=EnumRetryErrorCategory.UNKNOWN,
                should_retry=True,
                record_circuit_failure=False,
                error_message=f"Rate limited during {operation}",
            )

        if isinstance(error, InfraRequestRejectedError):
            return ModelRetryErrorClassification(
                category=EnumRetryErrorCategory.NOT_FOUND,
                should_retry=False,
                record_circuit_failure=False,
                error_message=f"Request rejected during {operation}: {error}",
            )

        if isinstance(error, InfraProtocolError):
            return ModelRetryErrorClassification(
                category=EnumRetryErrorCategory.CONNECTION,
                should_retry=True,
                record_circuit_failure=True,
                error_message=f"Protocol error during {operation}: {error}",
            )

        if isinstance(error, ProtocolConfigurationError):
            return ModelRetryErrorClassification(
                category=EnumRetryErrorCategory.NOT_FOUND,
                should_retry=False,
                record_circuit_failure=False,
                error_message=f"Configuration error during {operation}: {error}",
            )

        if isinstance(error, InfraUnavailableError):
            return ModelRetryErrorClassification(
                category=EnumRetryErrorCategory.CONNECTION,
                should_retry=True,
                record_circuit_failure=True,
                error_message=f"Service unavailable during {operation}: {error}",
            )

        # Default: unknown errors are retriable with CB failure
        return ModelRetryErrorClassification(
            category=EnumRetryErrorCategory.UNKNOWN,
            should_retry=True,
            record_circuit_failure=True,
            error_message=f"Unexpected error during {operation}: {type(error).__name__}: {error}",
        )

    def _get_transport_type(self) -> EnumInfraTransportType:
        """Return the transport type for error context.

        Returns:
            EnumInfraTransportType.HTTP for LLM HTTP transport.
        """
        return EnumInfraTransportType.HTTP

    def _get_target_name(self) -> str:
        """Return the target name for error context.

        Returns:
            The configured LLM target name.
        """
        return self._llm_target_name

    # ── Endpoint trust boundary ─────────────────────────────────────────

    async def _validate_endpoint_allowlist(
        self,
        url: str,
        correlation_id: UUID,
    ) -> None:
        """Validate that the URL target IP is within the local LLM CIDR allowlist.

        Resolves the hostname to an IPv4 address and checks membership in
        ``LOCAL_LLM_CIDRS`` (default: ``192.168.86.0/24``, configurable via
        the ``LLM_ENDPOINT_CIDR_ALLOWLIST`` environment variable). This is a
        fail-closed check: if the hostname cannot be resolved or the IP is
        outside all configured allowlist ranges, the request is rejected before
        any HTTP call is made.

        DNS resolution uses ``asyncio.get_running_loop().getaddrinfo()`` to
        avoid blocking the event loop on synchronous ``socket.getaddrinfo()``.

        Known Limitations:
            There is a time-of-check-to-time-of-use (TOCTOU) gap between the
            DNS resolution performed here and the independent DNS resolution
            performed by httpx when the actual HTTP request is made. An attacker
            with control over DNS responses could return a permitted IP during
            the allowlist check and a different (malicious) IP when httpx
            resolves the same hostname moments later.

            This is acceptable for the current local-network trust boundary
            (``192.168.86.0/24``) where DNS is resolved by a trusted local
            resolver and endpoints are on a private LAN segment not exposed to
            the public internet. If the trust boundary is extended to untrusted
            networks, this method should be replaced with an approach that pins
            the resolved IP and passes it directly to the HTTP client (e.g.,
            via httpx transport-level address binding).

        Args:
            url: The full URL of the LLM endpoint.
            correlation_id: Correlation ID for error context.

        Raises:
            InfraAuthenticationError: If the resolved IP is outside the
                allowlist or the hostname cannot be resolved.
        """
        parsed = urlparse(url)
        hostname = parsed.hostname
        if hostname is None:
            ctx = self._build_error_context(f"allowlist_check:{url}", correlation_id)
            raise InfraAuthenticationError(
                f"Cannot extract hostname from URL for allowlist validation: {url}",
                context=ctx,
            )

        # Resolve hostname to IP address
        try:
            resolved_ip = ip_address(hostname)
        except ValueError:
            # hostname is not an IP literal - resolve via async DNS
            try:
                loop = asyncio.get_running_loop()
                resolved = await loop.getaddrinfo(
                    hostname, None, family=socket.AF_INET, type=socket.SOCK_STREAM
                )
                if not resolved:
                    ctx = self._build_error_context(
                        f"allowlist_check:{url}", correlation_id
                    )
                    raise InfraAuthenticationError(
                        f"DNS resolution returned no IPv4 results for {hostname} "
                        "(IPv6-only hosts are not supported)",
                        context=ctx,
                    )
                resolved_ip = ip_address(resolved[0][4][0])
            except socket.gaierror as exc:
                ctx = self._build_error_context(
                    f"allowlist_check:{url}", correlation_id
                )
                raise InfraAuthenticationError(
                    f"Cannot resolve hostname {hostname} for allowlist validation",
                    context=ctx,
                ) from exc

        if not isinstance(resolved_ip, IPv4Address):
            ctx = self._build_error_context(f"allowlist_check:{url}", correlation_id)
            raise InfraAuthenticationError(
                f"IPv6 addresses are not supported by the LLM endpoint allowlist: "
                f"{resolved_ip}",
                context=ctx,
            )

        if not any(resolved_ip in cidr for cidr in self.LOCAL_LLM_CIDRS):
            ctx = self._build_error_context(f"allowlist_check:{url}", correlation_id)
            allowlist_str = ", ".join(str(c) for c in self.LOCAL_LLM_CIDRS)
            raise InfraAuthenticationError(
                f"Endpoint IP {resolved_ip} is outside the local LLM allowlist "
                f"({allowlist_str})",
                context=ctx,
            )

        logger.debug(
            "Endpoint passed CIDR allowlist check",
            extra={
                "resolved_ip": str(resolved_ip),
                "allowlist": ", ".join(str(c) for c in self.LOCAL_LLM_CIDRS),
                "hostname": hostname,
                "correlation_id": str(correlation_id),
                "target": self._llm_target_name,
            },
        )

    def _compute_hmac_signature(
        self,
        payload: dict[str, JsonType],
        correlation_id: UUID,
    ) -> str:
        """Compute HMAC-SHA256 signature for the request payload.

        Uses the ``LOCAL_LLM_SHARED_SECRET`` environment variable as the
        signing key. This is a fail-closed check: if the secret is not
        configured, the request is rejected.

        The secret is read from ``os.environ`` on every call rather than
        being cached at init time. This is intentional: it allows the shared
        secret to be rotated at runtime (e.g., via a sidecar or operator
        updating the environment) without requiring a process restart.

        The signature is computed over the canonical JSON serialization of
        the payload (sorted keys, no extra whitespace) to ensure deterministic
        signing regardless of dict ordering.

        Known Limitations:
            The HMAC signature does not include a timestamp or nonce, so it
            provides no replay protection. A captured request can be replayed
            verbatim as long as the shared secret remains unchanged. Additionally,
            the signature is computed once before the retry loop in
            ``_execute_llm_http_call``, meaning all retry attempts for a given
            call share the same signature value.

            This is acceptable for the current local-network trust boundary
            (``192.168.86.0/24``) where traffic stays on a private LAN segment
            not exposed to the public internet. The HMAC serves as a proof-of-
            origin to prevent accidental cross-service calls, not as a defense
            against active network attackers. If the trust boundary is extended
            to untrusted networks, the signing scheme should be upgraded to
            include a timestamp and/or nonce to mitigate replay attacks.

        Args:
            payload: JSON-serializable request payload.
            correlation_id: Correlation ID for error context.

        Returns:
            Hex-encoded HMAC-SHA256 signature string.

        Raises:
            ProtocolConfigurationError: If ``LOCAL_LLM_SHARED_SECRET`` is
                not set or is empty.
        """
        secret = os.environ.get(self.LOCAL_LLM_SECRET_ENV, "")
        if not secret:
            ctx = self._build_error_context("hmac_signing", correlation_id)
            raise ProtocolConfigurationError(
                f"Environment variable {self.LOCAL_LLM_SECRET_ENV} is not set. "
                "HMAC signing requires a shared secret (fail-closed).",
                context=ctx,
            )

        # Canonical JSON: sorted keys, compact encoding
        canonical = json_module.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

        signature = hmac.new(
            secret.encode("utf-8"),
            canonical,
            hashlib.sha256,
        ).hexdigest()

        logger.debug(
            "Computed HMAC signature for LLM request",
            extra={
                "correlation_id": str(correlation_id),
                "target": self._llm_target_name,
                "payload_bytes": len(canonical),
            },
        )

        return signature

    # ── Core HTTP execution ──────────────────────────────────────────────

    async def _execute_llm_http_call(
        self,
        url: str,
        payload: dict[str, JsonType],
        correlation_id: UUID,
        max_retries: int = 3,
        timeout_seconds: float = 30.0,
    ) -> dict[str, JsonType]:
        """Execute an HTTP POST to an LLM endpoint with retry and circuit breaker.

        This is the primary method for making LLM API calls. It handles:
        - Circuit breaker checking before each attempt
        - HTTP POST with JSON payload
        - Status code to typed exception mapping
        - Retry-After header honoring for 429 responses
        - Content-type validation for JSON responses
        - Exponential backoff between retries
        - Circuit breaker state updates on success/failure

        Args:
            url: The full URL of the LLM endpoint.
            payload: JSON-serializable request payload.
            correlation_id: Correlation ID for distributed tracing.
            max_retries: Maximum number of retry attempts (default: 3).
                Total attempts = 1 + max_retries.
            timeout_seconds: Per-request timeout in seconds (default: 30.0).
                Clamped to [0.1, self._max_timeout_seconds].

        Returns:
            Parsed JSON response as a dictionary.

        Raises:
            InfraAuthenticationError: On 401/403 responses.
            InfraRateLimitedError: On 429 when retries exhausted.
            InfraRequestRejectedError: On 400/422 responses.
            ProtocolConfigurationError: On 404 responses.
            InfraProtocolError: On non-JSON 2xx responses.
            InfraConnectionError: On connection failures after retries.
            InfraTimeoutError: On timeout after retries.
            InfraUnavailableError: On 5xx or circuit breaker open.

        Note:
            Response bodies are sanitized via ``sanitize_error_string()``
            before being attached to error context, ensuring that sensitive
            data from LLM provider responses is never leaked through
            exception messages or logging.

        .. versionchanged:: 0.8.0
            Added CIDR allowlist and HMAC signing pre-checks (OMN-2250).
        """
        # Runtime import to avoid circular dependency:
        # mixins/__init__ -> mixin_llm_http_transport -> handlers.models -> handlers/__init__
        # -> handler_consul -> mixins/__init__ (cycle)
        from omnibase_infra.handlers.models.model_retry_state import ModelRetryState

        # ── Pre-flight security checks (fail-closed) ────────────────
        # These run BEFORE any HTTP call to enforce the local trust boundary.
        await self._validate_endpoint_allowlist(url, correlation_id)
        # Intentionally computed once before the retry loop; see Known Limitations in _compute_hmac_signature.
        hmac_signature = self._compute_hmac_signature(payload, correlation_id)

        operation = f"llm_http_call:{url}"
        total_attempts = 1 + max_retries
        effective_timeout = max(min(timeout_seconds, self._max_timeout_seconds), 0.1)
        retry_state = ModelRetryState(max_attempts=total_attempts)

        client = await self._get_http_client()

        while retry_state.is_retriable():
            try:
                # Check circuit breaker
                await self._check_circuit_if_enabled(operation, correlation_id)

                # Make HTTP POST with HMAC signature header
                # headers dict intentionally overrides any default HMAC_HEADER key on the client.
                response = await client.post(
                    url,
                    json=payload,
                    headers={self.HMAC_HEADER: hmac_signature},
                    timeout=effective_timeout,
                )

                # Handle non-2xx responses
                if response.status_code < 200 or response.status_code >= 300:
                    error = self._map_http_status_to_error(response, correlation_id)

                    # Special 429 handling: use Retry-After as delay
                    if response.status_code == 429:
                        retry_after = self._parse_retry_after(response)
                        next_state = retry_state.next_attempt(
                            error_message=f"Rate limited (429), retry after {retry_after}s",
                        )
                        if next_state.is_retriable():
                            logger.debug(
                                "Rate limited, waiting before retry",
                                extra={
                                    "retry_after_seconds": retry_after,
                                    "attempt": next_state.attempt,
                                    "max_attempts": next_state.max_attempts,
                                    "correlation_id": str(correlation_id),
                                    "target": self._llm_target_name,
                                },
                            )
                            await asyncio.sleep(retry_after)
                            retry_state = next_state
                            continue
                        raise error

                    classification = self._classify_error(error, operation)
                    if classification.record_circuit_failure:
                        await self._record_circuit_failure_if_enabled(
                            operation, correlation_id
                        )
                    next_state = retry_state.next_attempt(
                        error_message=classification.error_message,
                    )
                    if classification.should_retry and next_state.is_retriable():
                        retry_state = next_state
                        logger.debug(
                            "Retrying after HTTP error",
                            extra={
                                "status_code": response.status_code,
                                "attempt": retry_state.attempt,
                                "max_attempts": retry_state.max_attempts,
                                "delay_seconds": retry_state.delay_seconds,
                                "correlation_id": str(correlation_id),
                                "target": self._llm_target_name,
                            },
                        )
                        await asyncio.sleep(retry_state.delay_seconds)
                        continue
                    raise error

                # 2xx response: validate content-type
                # Only reject when a non-JSON content-type is explicitly present.
                # Missing/empty content-type falls through to JSON parsing which
                # will raise InfraProtocolError on its own if the body is invalid.
                content_type = response.headers.get("content-type", "")
                if content_type and "json" not in content_type.lower():
                    await self._record_circuit_failure_if_enabled(
                        operation, correlation_id
                    )
                    body_snippet = (
                        sanitize_error_string(response.text) if response.text else ""
                    )
                    ctx = self._build_error_context(operation, correlation_id)
                    raise InfraProtocolError(
                        f"Expected JSON response from {self._llm_target_name}, "
                        f"got content-type: {content_type}",
                        context=ctx,
                        status_code=response.status_code,
                        content_type=content_type,
                        response_body=body_snippet,
                    )

                # Parse JSON
                try:
                    data = cast("dict[str, JsonType]", response.json())
                except (JSONDecodeError, ValueError) as exc:
                    await self._record_circuit_failure_if_enabled(
                        operation, correlation_id
                    )
                    body_snippet = (
                        sanitize_error_string(response.text) if response.text else ""
                    )
                    ctx = self._build_error_context(operation, correlation_id)
                    raise InfraProtocolError(
                        f"Failed to parse JSON response from {self._llm_target_name}: {exc}",
                        context=ctx,
                        status_code=response.status_code,
                        content_type=content_type,
                        response_body=body_snippet,
                    ) from exc

                # Success - reset circuit breaker
                await self._reset_circuit_if_enabled()
                return data

            except httpx.ConnectError as exc:
                classification = self._classify_error(exc, operation)
                if classification.record_circuit_failure:
                    await self._record_circuit_failure_if_enabled(
                        operation, correlation_id
                    )
                next_state = retry_state.next_attempt(
                    error_message=classification.error_message,
                )
                if next_state.is_retriable():
                    retry_state = next_state
                    logger.debug(
                        "Retrying after connection error",
                        extra={
                            "attempt": retry_state.attempt,
                            "max_attempts": retry_state.max_attempts,
                            "delay_seconds": retry_state.delay_seconds,
                            "correlation_id": str(correlation_id),
                            "target": self._llm_target_name,
                        },
                    )
                    await asyncio.sleep(retry_state.delay_seconds)
                    continue
                ctx = self._build_error_context(operation, correlation_id)
                raise InfraConnectionError(
                    f"Connection to {self._llm_target_name} failed after "
                    f"{retry_state.attempt + 1} attempts: {exc}",
                    context=ctx,
                ) from exc

            except httpx.TimeoutException as exc:
                classification = self._classify_error(exc, operation)
                if classification.record_circuit_failure:
                    await self._record_circuit_failure_if_enabled(
                        operation, correlation_id
                    )
                next_state = retry_state.next_attempt(
                    error_message=classification.error_message,
                )
                if next_state.is_retriable():
                    retry_state = next_state
                    logger.debug(
                        "Retrying after timeout",
                        extra={
                            "attempt": retry_state.attempt,
                            "max_attempts": retry_state.max_attempts,
                            "delay_seconds": retry_state.delay_seconds,
                            "timeout_seconds": effective_timeout,
                            "correlation_id": str(correlation_id),
                            "target": self._llm_target_name,
                        },
                    )
                    await asyncio.sleep(retry_state.delay_seconds)
                    continue
                timeout_ctx = ModelTimeoutErrorContext(
                    transport_type=EnumInfraTransportType.HTTP,
                    operation=operation,
                    target_name=self._llm_target_name,
                    correlation_id=correlation_id,
                    timeout_seconds=effective_timeout,
                )
                raise InfraTimeoutError(
                    f"Request to {self._llm_target_name} timed out after "
                    f"{effective_timeout}s ({retry_state.attempt + 1} attempts)",
                    context=timeout_ctx,
                ) from exc

            except (
                InfraRateLimitedError,
                InfraRequestRejectedError,
                InfraAuthenticationError,
                ProtocolConfigurationError,
                InfraProtocolError,
            ):
                raise  # Already typed, don't wrap

            except InfraUnavailableError:
                raise  # Already handled (e.g., circuit breaker open)

            except Exception as exc:
                # Unexpected error - classify and handle
                classification = self._classify_error(exc, operation)
                if classification.record_circuit_failure:
                    await self._record_circuit_failure_if_enabled(
                        operation, correlation_id
                    )
                next_state = retry_state.next_attempt(
                    error_message=classification.error_message,
                )
                if classification.should_retry and next_state.is_retriable():
                    retry_state = next_state
                    logger.debug(
                        "Retrying after unexpected error",
                        extra={
                            "error_type": type(exc).__name__,
                            "attempt": retry_state.attempt,
                            "max_attempts": retry_state.max_attempts,
                            "delay_seconds": retry_state.delay_seconds,
                            "correlation_id": str(correlation_id),
                            "target": self._llm_target_name,
                        },
                    )
                    await asyncio.sleep(retry_state.delay_seconds)
                    continue
                ctx = self._build_error_context(operation, correlation_id)
                raise InfraConnectionError(
                    f"Unexpected error calling {self._llm_target_name}: "
                    f"{type(exc).__name__}: {exc}",
                    context=ctx,
                ) from exc

        # Loop exited without return or raise - all retries exhausted
        ctx = self._build_error_context(operation, correlation_id)
        raise InfraUnavailableError(
            f"All retry attempts exhausted for {self._llm_target_name} "
            f"({total_attempts} attempts)",
            context=ctx,
        )

    # ── HTTP status to error mapping ─────────────────────────────────────

    def _map_http_status_to_error(
        self,
        response: httpx.Response,
        correlation_id: UUID,
    ) -> RuntimeHostError:
        """Map an HTTP response status code to a typed infrastructure exception.

        Response body snippets included in exceptions are sanitized via
        ``sanitize_error_string()`` to prevent leakage of secrets or PII.

        Args:
            response: The httpx.Response with a non-2xx status code.
            correlation_id: Correlation ID for error context.

        Returns:
            A typed infrastructure exception instance (not raised). The caller
            is responsible for raising or classifying the returned exception.
        """
        ctx = self._build_error_context(f"llm_http_call:{response.url}", correlation_id)
        body_snippet = sanitize_error_string(response.text) if response.text else ""
        status = response.status_code

        if status in (401, 403):
            return InfraAuthenticationError(
                f"Authentication failed ({status}) from {self._llm_target_name}",
                context=ctx,
                status_code=status,
                response_body=body_snippet,
            )

        if status == 429:
            retry_after = self._parse_retry_after(response)
            return InfraRateLimitedError(
                f"Rate limited (429) by {self._llm_target_name}",
                context=ctx,
                retry_after_seconds=retry_after,
            )

        if status == 404:
            return ProtocolConfigurationError(
                f"Endpoint not found (404) at {self._llm_target_name} - "
                "assumed misconfiguration",
                context=ctx,
                status_code=status,
                response_body=body_snippet,
            )

        if status in (400, 422):
            return InfraRequestRejectedError(
                f"Request rejected ({status}) by {self._llm_target_name}",
                context=ctx,
                status_code=status,
                response_body=body_snippet,
            )

        if status in (500, 502, 503, 504):
            return InfraUnavailableError(
                f"Server error ({status}) from {self._llm_target_name}",
                context=ctx,
                status_code=status,
                response_body=body_snippet,
            )

        # Other non-2xx: treat as unavailable
        return InfraUnavailableError(
            f"Unexpected HTTP {status} from {self._llm_target_name}",
            context=ctx,
            status_code=status,
            response_body=body_snippet,
        )

    # ── Retry-After parsing ──────────────────────────────────────────────

    def _parse_retry_after(self, response: httpx.Response) -> float:
        """Parse the Retry-After header from an HTTP response.

        Supports delta-seconds format (integer or float). HTTP-date format
        is not supported and falls back to default backoff.

        Args:
            response: The httpx.Response (typically 429).

        Returns:
            Seconds to wait before retrying, clamped to
            [0.0, self._max_retry_after_seconds]. Returns 1.0 if header
            is absent or unparseable.
        """
        retry_after_raw = response.headers.get("retry-after")
        if retry_after_raw is None:
            return 1.0

        try:
            retry_after = float(retry_after_raw)
        except (ValueError, OverflowError):
            logger.debug(
                "Could not parse Retry-After header, using default backoff",
                extra={
                    "retry_after_raw": retry_after_raw,
                    "target": self._llm_target_name,
                },
            )
            return 1.0

        # Guard against NaN/Inf which would cause asyncio.sleep() to raise
        # ValueError.  float() happily parses 'nan', 'inf', and '-inf'.
        if not math.isfinite(retry_after):
            logger.debug(
                "Non-finite Retry-After value (%s), using default backoff",
                retry_after,
                extra={
                    "retry_after_raw": retry_after_raw,
                    "target": self._llm_target_name,
                },
            )
            return 1.0

        return max(0.0, min(retry_after, self._max_retry_after_seconds))

    # ── HTTP client management ───────────────────────────────────────────

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Return the HTTP client, creating a lazy singleton if needed.

        If an external client was injected via ``_init_llm_http_transport``,
        that client is returned directly. Otherwise, a new client is created
        with reasonable defaults for LLM API communication.

        Uses double-checked locking to prevent concurrent coroutines from
        creating duplicate clients (one would leak).

        Returns:
            An httpx.AsyncClient ready for use.
        """
        if self._http_client is not None:
            return self._http_client

        async with self._http_client_lock:
            # Double-check after acquiring lock to avoid creating a second client
            if self._http_client is not None:
                return self._http_client

            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                ),
            )
            return self._http_client

    async def _close_http_client(self) -> None:
        """Close the HTTP client if this mixin owns it.

        Only closes the client if it was created internally (not injected).
        After closing, the client reference is set to None so a new one
        can be created lazily if needed.

        Thread-safety is ensured by acquiring ``_http_client_lock``
        (an ``asyncio.Lock``) to prevent races with the lazy creation
        in ``_get_http_client``.
        """
        async with self._http_client_lock:
            if self._owns_http_client and self._http_client is not None:
                await self._http_client.aclose()
                self._http_client = None


__all__: list[str] = [
    "MixinLlmHttpTransport",
]
