# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry Effect Node for dual registration to Consul and PostgreSQL.

This effect node bridges the message bus to external infrastructure services,
implementing the 2-way registration pattern for ONEX nodes.

Features:
    - Dual registration (Consul + PostgreSQL) with parallel execution
    - Circuit breaker protection via MixinAsyncCircuitBreaker
    - Graceful degradation (partial success when one backend fails)
    - Correlation ID propagation for distributed tracing
    - UPSERT pattern for idempotent re-registration

Operations:
    - register: Register node with Consul and PostgreSQL
    - deregister: Remove node from both backends
    - discover: Query registered nodes with filters
    - request_introspection: Publish introspection request to event bus

Terminology - register vs resolve:
    This module deals with two distinct concepts that use similar terminology:

    1. **External Service Registration** (this node's purpose):
       - Operations: register, deregister, discover
       - Target: Consul (service discovery) and PostgreSQL (persistent registry)
       - Purpose: Allow distributed services to find each other at runtime
       - Example: `await node.execute(ModelRegistryRequest(operation="register", ...))`

    2. **DI Container Resolution** (used internally for dependencies):
       - Method: `container.service_registry.resolve_service(ProtocolType)`
       - Target: ModelONEXContainer's internal service registry
       - Purpose: Wire up internal dependencies at startup/construction time
       - Example: `handler = await container.service_registry.resolve_service(
           ProtocolEnvelopeExecutor, name="consul")`

    These are orthogonal concepts - this node uses DI resolution internally
    to obtain its handler dependencies, then provides external registration
    services to callers.

Handler Interface (duck typing):
    consul_handler and db_handler must implement:
        async def execute(self, envelope: EnvelopeDict) -> ResultDict

Event Bus Interface (duck typing):
    event_bus must implement:
        async def publish(self, topic: str, key: bytes, value: bytes) -> None
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, cast
from uuid import UUID

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

from omnibase_core.models.node_metadata import ModelNodeCapabilitiesInfo

from omnibase_infra.enums import EnumInfraTransportType

# Type alias for registry operation status (must match ModelRegistryResponse.status)
RegistryStatus = Literal["success", "partial", "failed"]

from omnibase_infra.errors import (
    InfraUnavailableError,
    ModelInfraErrorContext,
    RuntimeHostError,
    ServiceResolutionError,
)
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.nodes.node_registry_effect.v1_0_0.models import (
    EnumEnvironment,
    ModelConsulOperationResult,
    ModelNodeIntrospectionPayload,
    ModelNodeRegistration,
    ModelNodeRegistrationMetadata,
    ModelNodeRegistryEffectConfig,
    ModelPostgresOperationResult,
    ModelRegistryRequest,
    ModelRegistryResponse,
)
from omnibase_infra.nodes.node_registry_effect.v1_0_0.protocols import (
    EnvelopeDict,
    JsonValue,
    ProtocolEnvelopeExecutor,
    ProtocolEventBus,
    ResultDict,
)

logger = logging.getLogger(__name__)

# =============================================================================
# SECURITY CONSTANTS
# =============================================================================

# Whitelist of allowed filter keys for SQL query building (SQL injection prevention)
# These correspond to actual columns in the node_registrations table that support
# direct equality filtering. JSONB columns (capabilities, endpoints, metadata) are
# excluded as they require specialized query operators.
#
# SECURITY: This whitelist prevents SQL injection by ensuring only known-safe
# column names can be interpolated into SQL queries. Filter VALUES are always
# parameterized (never interpolated). Invalid filter keys cause the request to
# be rejected with an error, not silently ignored.
ALLOWED_FILTER_KEYS: frozenset[str] = frozenset(
    {
        "node_id",  # Primary key, VARCHAR(255)
        "node_type",  # Node classification (effect, compute, reducer, orchestrator)
        "node_version",  # Semantic version string
        "health_endpoint",  # Health check URL (nullable)
    }
)

# Maximum length for string inputs to prevent DoS via oversized inputs
# These limits align with typical database column sizes and prevent memory exhaustion
MAX_NODE_ID_LENGTH: int = 255  # VARCHAR(255) in PostgreSQL
MAX_NODE_VERSION_LENGTH: int = 50  # Semantic versions rarely exceed 20 chars
MAX_FILTER_VALUE_LENGTH: int = 1024  # Reasonable max for filter values
MAX_HEALTH_ENDPOINT_LENGTH: int = 2048  # URLs can be long but should be bounded
MAX_ENDPOINT_KEY_LENGTH: int = 64  # Endpoint names (e.g., "health", "api")
MAX_ENDPOINT_VALUE_LENGTH: int = 2048  # URLs for endpoints

# Character validation pattern for node_id and other identifiers
# Allows: alphanumeric, hyphens, underscores, periods (for domain-style naming)
# Prevents: SQL injection characters, null bytes, control characters, Unicode exploits
# Pattern explanation: ^[a-zA-Z0-9][a-zA-Z0-9._-]*$
#   - Must start with alphanumeric
#   - Can contain alphanumeric, periods, underscores, hyphens
#   - No spaces, quotes, semicolons, or special SQL characters
NODE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")

# Semantic version pattern (relaxed to allow common formats)
# Allows: X.Y.Z, X.Y.Z-alpha, X.Y.Z-beta.1, X.Y.Z+build
VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+([-.+][a-zA-Z0-9._-]*)?$")

# URL pattern for health endpoints (basic validation, not comprehensive)
# Allows: http:// and https:// URLs with standard characters
# Supports: hostnames, IP addresses (IPv4 and IPv6), ports, paths with query strings
# Examples:
#   - http://localhost:8080/health
#   - https://api.example.com/v1/status
#   - http://192.168.1.1:3000/healthz?verbose=true
#   - https://my-service.internal/health-check
#   - http://my_service.local:8080/health (underscores allowed)
#   - http://[::1]:8080/health (IPv6 loopback)
#   - http://[2001:db8::1]:8080/health (IPv6 with port)
#
# Pattern design:
#   - Protocol: http:// or https://
#   - Host: Either IPv6 in brackets, or standard hostname/IPv4
#   - IPv6: [hex:colon:notation] - allows compressed form (::)
#   - Hostname: alphanumeric start, can contain hyphens, underscores, dots
#   - Port: Optional, 1-5 digits (allows invalid ports > 65535, but that's fine for validation)
#   - Path: RFC 3986 compliant characters including query string and fragment
URL_PATTERN = re.compile(
    r"^https?://"  # Protocol (required)
    r"(?:"
    # IPv6 address in brackets: [2001:db8::1] or [::1]
    r"\[[a-fA-F0-9:]+\]"
    r"|"
    # Hostname or IPv4: must start with alphanumeric
    # Can contain: alphanumeric, hyphens, underscores, dots
    # Examples: localhost, my-service.internal, my_service.local, 192.168.1.1
    r"[a-zA-Z0-9][a-zA-Z0-9._-]*"
    r")"
    r"(:[0-9]{1,5})?"  # Optional port (1-5 digits)
    r"(/[-a-zA-Z0-9._~:/?#\[\]@!$&'()*+,;=%]*)?"  # Optional path with query string
    r"$"
)

# =============================================================================
# PERFORMANCE CONSTANTS
# =============================================================================

# Default performance threshold for slow operation warnings (milliseconds)
# Used as fallback when config not provided
DEFAULT_SLOW_OPERATION_THRESHOLD_MS: float = 1000.0

# =============================================================================
# ERROR SANITIZATION PATTERNS
# =============================================================================

# Patterns for redacting sensitive information from error messages.
# Order matters: more specific patterns should come before generic ones.
# This list is used by _sanitize_error() to prevent credential leakage.
#
# Pattern format: (regex_pattern, replacement_string)
#
# JWT Pattern Design:
#   Real JWTs have three base64url-encoded segments: header.payload.signature
#   - Header: MUST contain "alg" claim (RFC 7515), base64url encodes to contain "hbGci"
#     Common headers: {"alg":"HS256"}, {"alg":"RS256","typ":"JWT"}
#   - Payload: Contains claims, typically starts with {"sub":, {"iss":, etc.
#   - Signature: Varies by algorithm (HS256=43 chars, RS256=342 chars, ES256=86 chars)
#
#   The tightened pattern requires:
#   - Header starting with 'eyJ' (base64 for '{"') followed by 'hbGci' (base64 for '"alg"')
#     or reasonable header content (15+ chars after eyJ to cover minimal {"alg":"HS256"})
#   - Payload starting with 'eyJ' with 15+ additional chars (minimal claims)
#   - Signature with 40+ chars (covers HS256=43, ES256=86, RS256=342)
#
#   False positive reduction:
#   - Increased minimum lengths for all segments
#   - Header pattern more specific (requires eyJ followed by substantial content)
#   - Signature minimum increased to 40 chars (real signatures are rarely shorter)
#   - Word boundary check ensures we match complete tokens, not substrings
SENSITIVE_PATTERNS: tuple[tuple[str, str], ...] = (
    # Private keys and certificates (multiline patterns)
    (
        r"-----BEGIN\s+[A-Z\s]+KEY-----[\s\S]*?-----END\s+[A-Z\s]+KEY-----",
        "[REDACTED_PRIVATE_KEY]",
    ),
    (
        r"-----BEGIN\s+CERTIFICATE-----[\s\S]*?-----END\s+CERTIFICATE-----",
        "[REDACTED_CERTIFICATE]",
    ),
    # SSH keys
    (r"ssh-rsa\s+[A-Za-z0-9+/=]+", "[REDACTED_SSH_KEY]"),
    (r"ssh-ed25519\s+[A-Za-z0-9+/=]+", "[REDACTED_SSH_KEY]"),
    # JWT tokens (header.payload.signature) - tightened to reduce false positives
    # Header: eyJ + 15+ chars (covers minimal {"alg":"HS256"} = 18 chars base64)
    # Payload: eyJ + 15+ chars (covers minimal claims like {"sub":"x"} = 12+ chars)
    # Signature: 40+ chars (HS256=43, ES256=86, RS256=342 - none are < 40)
    # Word boundary \b ensures we match complete tokens
    (
        r"\beyJ[A-Za-z0-9_-]{15,}\.eyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{40,}\b",
        "[REDACTED_JWT]",
    ),
    # Password variants with various delimiters
    (r"password[=:\"'\s]+\S+", "password=[REDACTED]"),
    (r"passwd[=:\"'\s]+\S+", "passwd=[REDACTED]"),
    (r"pwd[=:\"'\s]+\S+", "pwd=[REDACTED]"),
    # Token variants
    (r"access_token[=:\"'\s]+\S+", "access_token=[REDACTED]"),
    (r"refresh_token[=:\"'\s]+\S+", "refresh_token=[REDACTED]"),
    (r"token[=:\"'\s]+\S+", "token=[REDACTED]"),
    # API key variants
    (r"x-api-key[=:\"'\s]+\S+", "x-api-key=[REDACTED]"),
    (r"api[-_]?key[=:\"'\s]+\S+", "api_key=[REDACTED]"),
    (r"apikey[=:\"'\s]+\S+", "apikey=[REDACTED]"),
    # Secret variants
    (r"client[-_]?secret[=:\"'\s]+\S+", "client_secret=[REDACTED]"),
    (r"secret[-_]?key[=:\"'\s]+\S+", "secret_key=[REDACTED]"),
    (r"secret[=:\"'\s]+\S+", "secret=[REDACTED]"),
    # Credentials and auth
    (r"credential[s]?[=:\"'\s]+\S+", "credentials=[REDACTED]"),
    (r"auth[-_]?token[=:\"'\s]+\S+", "auth_token=[REDACTED]"),
    (r"authorization[=:\"'\s]+\S+", "authorization=[REDACTED]"),
    (r"auth[=:\"'\s]+\S+", "auth=[REDACTED]"),
    # Bearer tokens
    (r"bearer\s+[A-Za-z0-9._-]+", "bearer [REDACTED]"),
    # Connection string credentials (user:pass@host)
    (r"://[^/:]+:[^/@]+@", "://[REDACTED]:[REDACTED]@"),
    # AWS access keys (AKIA followed by 16 alphanumeric chars)
    (r"AKIA[A-Z0-9]{16}", "[REDACTED_AWS_KEY]"),
    # AWS secret keys
    (
        r"aws[-_]?secret[-_]?access[-_]?key[=:\"'\s]+\S+",
        "aws_secret=[REDACTED]",
    ),
    # GCP/Azure style service account JSON keys
    (r'"private_key"\s*:\s*"[^"]+"', '"private_key": "[REDACTED]"'),
    (r'"client_secret"\s*:\s*"[^"]+"', '"client_secret": "[REDACTED]"'),
    # Database connection strings with passwords
    (r"(postgres|mysql|mongodb|redis)://[^:]+:[^@]+@", r"\1://[REDACTED]@"),
    # Generic long alphanumeric tokens (40+ chars, likely tokens/keys)
    # This is intentionally last and broad to catch anything missed
    (r"[A-Za-z0-9+/]{40,}={0,2}", "[REDACTED_TOKEN]"),
)


class NodeRegistryEffect(MixinAsyncCircuitBreaker):
    """Registry Effect Node for dual registration to Consul and PostgreSQL.

    This node implements the EFFECT node type, performing I/O operations
    to external services (Consul, PostgreSQL, Kafka) for node registration.

    Thread Safety:
        Uses MixinAsyncCircuitBreaker for circuit breaker state management.
        All circuit breaker operations require holding _circuit_breaker_lock.

    Error Handling:
        - InfraConnectionError: Backend connection failures
        - InfraUnavailableError: Circuit breaker open
        - RuntimeHostError: Configuration/validation errors

    Duck Typing Interfaces:
        consul_handler: Must have async execute(envelope: dict) -> dict method
        db_handler: Must have async execute(envelope: dict) -> dict method
        event_bus: Must have async publish(topic: str, key: bytes, value: bytes) -> None method

    Connection Pooling:
        This node delegates database operations to db_handler and does not manage
        connections directly. The db_handler implementation (e.g., PostgreSQL adapter)
        is responsible for:
        - Connection pool management to prevent connection exhaustion
        - Connection health checks and automatic reconnection
        - Proper connection release after each operation

        For production deployments, ensure the PostgreSQL adapter uses asyncpg
        with connection pooling (recommended pool size: 10-20 connections).

    Dependency Injection:
        Takes a ModelONEXContainer and resolves dependencies internally.

        **Prerequisites**:
        Before constructing, the following must be registered in the container:
        - ProtocolEnvelopeExecutor with name="consul" (Consul handler)
        - ProtocolEnvelopeExecutor with name="postgres" (PostgreSQL handler)
        - ProtocolEventBus (optional, for request_introspection operation)

        **Usage** (recommended - use factory method):
        ```python
        container = ModelONEXContainer()
        await wire_infrastructure_services(container)
        # Register handlers first (consul, postgres) with container
        node = await NodeRegistryEffect.create(container)
        # Node is ready to use - create() handles dependency resolution
        ```

        **Usage** (alternative - manual initialization):
        ```python
        container = ModelONEXContainer()
        await wire_infrastructure_services(container)
        node = NodeRegistryEffect(container)  # __init__ is NOT awaitable
        await node._resolve_dependencies()     # Must call before use
        ```

    Key Operations:
        - **register**: Dual registration to Consul AND PostgreSQL in parallel.
          Use when a node starts up and needs to announce itself to external
          service registries. This is EXTERNAL SERVICE REGISTRATION, not DI.
        - **discover**: Query registered nodes from PostgreSQL.
          Use for service discovery to find other running nodes.
        - **deregister**: Remove from both Consul and PostgreSQL.
          Use during graceful shutdown.

    Important - register vs resolve distinction:
        - **register** (this node's operation): Writes node metadata to external
          service registries (Consul, PostgreSQL) so other services can discover it.
          This is runtime service registration for distributed systems.
        - **resolve** (DI container method): Retrieves service instances from the
          ModelONEXContainer's service_registry. This is compile-time/startup
          dependency injection for internal service wiring.

        Do NOT confuse these two concepts:
        - `await node.execute(ModelRegistryRequest(operation="register", ...))` - external registration
        - `await container.service_registry.resolve_service(SomeProtocol)` - DI resolution
    """

    def __init__(
        self,
        container: ModelONEXContainer,
        config: ModelNodeRegistryEffectConfig | None = None,
    ) -> None:
        """Initialize Registry Effect Node with dependencies from container.

        Args:
            container: ONEX container with registered handler services.
                Must have ProtocolEnvelopeExecutor with name="consul" and
                name="postgres" registered.
            config: Configuration for circuit breaker and resilience settings.
                Uses defaults if not provided.

        Note:
            The __init__ method is synchronous (NOT awaitable). However, this
            node requires async initialization for dependency resolution. The
            actual dependency resolution happens in _resolve_dependencies()
            which must be awaited after construction.

            Recommended: Use the async create() factory method which handles
            both construction and initialization:

            ```python
            node = await NodeRegistryEffect.create(container, config)
            ```

            Alternative: Manual two-phase initialization:

            ```python
            node = NodeRegistryEffect(container, config)  # Sync construction
            await node.initialize()  # Async initialization
            ```
        """
        self._container = container
        self._config = config or ModelNodeRegistryEffectConfig()
        self._consul_handler: ProtocolEnvelopeExecutor | None = None
        self._db_handler: ProtocolEnvelopeExecutor | None = None
        self._event_bus: ProtocolEventBus | None = None
        self._initialized = False
        self._dependencies_resolved = False

        # Store slow operation threshold from config (configurable per environment)
        self._slow_operation_threshold_ms: float = (
            self._config.slow_operation_threshold_ms
        )

        # Initialize circuit breaker
        self._init_circuit_breaker(
            threshold=self._config.circuit_breaker_threshold,
            reset_timeout=self._config.circuit_breaker_reset_timeout,
            service_name="node_registry_effect",
            transport_type=EnumInfraTransportType.RUNTIME,
        )

    async def _resolve_dependencies(self) -> None:
        """Resolve handler dependencies from container.

        Called automatically by initialize() if not already resolved.

        This method uses specific exception handling to distinguish between:
        - Service not registered (KeyError, LookupError, ServiceResolutionError): Expected
          when a service simply hasn't been registered in the container.
        - Configuration/validation errors (ValueError, TypeError): Unexpected errors
          that indicate a problem with the service configuration.
        - Other exceptions: Unexpected errors that may indicate infrastructure issues.

        Raises:
            RuntimeError: If required handlers are not registered or resolution fails.
        """
        if self._dependencies_resolved:
            return

        # Resolve required consul handler
        self._consul_handler = await self._resolve_required_handler(
            protocol_type=ProtocolEnvelopeExecutor,
            handler_name="consul",
            handler_description="Consul handler",
        )

        # Resolve required postgres handler
        self._db_handler = await self._resolve_required_handler(
            protocol_type=ProtocolEnvelopeExecutor,
            handler_name="postgres",
            handler_description="PostgreSQL handler",
        )

        # Resolve optional event bus (not all deployments need it)
        self._event_bus = await self._resolve_optional_service(
            service_type=ProtocolEventBus,
            service_name=None,  # No name qualifier for event bus
            service_description="ProtocolEventBus",
            fallback_message=("request_introspection operation will not be available"),
        )

        self._dependencies_resolved = True

    async def _resolve_required_handler(
        self,
        protocol_type: type,
        handler_name: str,
        handler_description: str,
    ) -> ProtocolEnvelopeExecutor:
        """Resolve a required handler from the container.

        Args:
            protocol_type: The protocol type to resolve (e.g., ProtocolEnvelopeExecutor).
            handler_name: The name qualifier for the handler (e.g., "consul", "postgres").
            handler_description: Human-readable description for error messages.

        Returns:
            The resolved handler instance.

        Raises:
            RuntimeError: If the handler is not registered or resolution fails.
        """
        try:
            handler = await self._container.service_registry.resolve_service(
                protocol_type, name=handler_name
            )
            logger.debug(
                f"Successfully resolved {handler_description}",
                extra={
                    "handler_name": handler_name,
                    "protocol_type": protocol_type.__name__,
                },
            )
            return cast(ProtocolEnvelopeExecutor, handler)

        except (KeyError, LookupError, ServiceResolutionError) as e:
            # Service not registered - this is the expected error type when
            # a service simply hasn't been registered in the container
            # Sanitize error message to prevent credential leakage
            sanitized_msg = self._sanitize_error(e)
            logger.debug(
                f"Handler '{handler_name}' not registered in container",
                extra={
                    "handler_name": handler_name,
                    "handler_description": handler_description,
                    "error_type": type(e).__name__,
                    "error_message": sanitized_msg,
                },
            )
            raise RuntimeError(
                f"Failed to resolve {handler_description} (name='{handler_name}') "
                f"from container. "
                f"Ensure {protocol_type.__name__} with name='{handler_name}' is registered. "
                f"Error type: {type(e).__name__}"
            ) from e

        except (ValueError, TypeError) as e:
            # Configuration or type mismatch errors - these indicate a problem
            # with the service configuration rather than missing registration
            # Sanitize error message to prevent credential leakage
            sanitized_msg = self._sanitize_error(e)
            logger.warning(
                f"Configuration error resolving {handler_description}",
                extra={
                    "handler_name": handler_name,
                    "handler_description": handler_description,
                    "error_type": type(e).__name__,
                    "error_message": sanitized_msg,
                },
            )
            raise RuntimeError(
                f"Configuration error resolving {handler_description} "
                f"(name='{handler_name}'): {sanitized_msg}"
            ) from e

        except Exception as e:
            # Unexpected error - log at warning level for investigation
            # Sanitize error message to prevent credential leakage
            sanitized_msg = self._sanitize_error(e)
            logger.warning(
                f"Unexpected error resolving {handler_description}",
                extra={
                    "handler_name": handler_name,
                    "handler_description": handler_description,
                    "error_type": type(e).__name__,
                    "error_message": sanitized_msg,
                },
            )
            raise RuntimeError(
                f"Unexpected error resolving {handler_description} "
                f"(name='{handler_name}'): {sanitized_msg}"
            ) from e

    async def _resolve_optional_service(
        self,
        service_type: type,
        service_name: str | None,
        service_description: str,
        fallback_message: str,
    ) -> ProtocolEventBus | None:
        """Resolve an optional service from the container.

        Unlike required handlers, optional services return None instead of raising
        an error when not registered. This allows the node to operate with reduced
        functionality when certain services are not available.

        Args:
            service_type: The service type to resolve.
            service_name: Optional name qualifier for the service.
            service_description: Human-readable description for logging.
            fallback_message: Message describing what functionality is unavailable.

        Returns:
            The resolved service instance, or None if not registered.
        """
        try:
            if service_name:
                service = await self._container.service_registry.resolve_service(
                    service_type, name=service_name
                )
            else:
                service = await self._container.service_registry.resolve_service(
                    service_type
                )
            logger.debug(
                f"Successfully resolved {service_description}",
                extra={
                    "service_name": service_name,
                    "service_type": service_type.__name__,
                },
            )
            return cast(ProtocolEventBus, service)

        except (KeyError, LookupError, ServiceResolutionError) as e:
            # Service not registered - this is expected for optional services
            # Sanitize error message to prevent credential leakage
            sanitized_msg = self._sanitize_error(e)
            logger.debug(
                f"{service_description} not registered in container; {fallback_message}",
                extra={
                    "service_name": service_name,
                    "service_description": service_description,
                    "error_type": type(e).__name__,
                    "error_message": sanitized_msg,
                },
            )
            return None

        except (ValueError, TypeError) as e:
            # Configuration error for optional service - log warning but don't fail
            # Sanitize error message to prevent credential leakage
            sanitized_msg = self._sanitize_error(e)
            logger.warning(
                f"Configuration error resolving optional {service_description}; "
                f"{fallback_message}",
                extra={
                    "service_name": service_name,
                    "service_description": service_description,
                    "error_type": type(e).__name__,
                    "error_message": sanitized_msg,
                },
            )
            return None

        except Exception as e:
            # Unexpected error for optional service - log warning but don't fail
            # This allows graceful degradation when there are infrastructure issues
            # Sanitize error message to prevent credential leakage
            sanitized_msg = self._sanitize_error(e)
            logger.warning(
                f"Unexpected error resolving optional {service_description}; "
                f"{fallback_message}",
                extra={
                    "service_name": service_name,
                    "service_description": service_description,
                    "error_type": type(e).__name__,
                    "error_message": sanitized_msg,
                },
            )
            return None

    def _ensure_dependencies(self) -> None:
        """Ensure dependencies are resolved before operations.

        Raises:
            RuntimeError: If dependencies haven't been resolved yet.
        """
        if not self._dependencies_resolved:
            raise RuntimeError(
                "Dependencies not resolved. Call _resolve_dependencies() or use "
                "NodeRegistryEffect.create() factory method."
            )
        if self._consul_handler is None or self._db_handler is None:
            raise RuntimeError(
                "Required handlers (consul_handler, db_handler) are None. "
                "Dependency resolution may have failed."
            )

    @property
    def consul_handler(self) -> ProtocolEnvelopeExecutor:
        """Get consul handler, ensuring it's resolved."""
        self._ensure_dependencies()
        assert self._consul_handler is not None  # for mypy
        return self._consul_handler

    @property
    def db_handler(self) -> ProtocolEnvelopeExecutor:
        """Get db handler, ensuring it's resolved."""
        self._ensure_dependencies()
        assert self._db_handler is not None  # for mypy
        return self._db_handler

    @classmethod
    async def create(
        cls,
        container: ModelONEXContainer,
        config: ModelNodeRegistryEffectConfig | None = None,
    ) -> NodeRegistryEffect:
        """Factory method to create and initialize NodeRegistryEffect.

        This is the recommended way to create NodeRegistryEffect instances.
        The returned node is fully initialized and ready for use.

        Args:
            container: ONEX container with registered services.
            config: Optional configuration override.

        Returns:
            Fully initialized NodeRegistryEffect ready for use.

        Raises:
            RuntimeError: If required handlers are not registered in the container.

        Example:
            ```python
            node = await NodeRegistryEffect.create(container)
            result = await node.execute(request)
            ```
        """
        node = cls(container, config)
        await node.initialize()  # initialize() calls _resolve_dependencies() if needed
        return node

    def _sanitize_error(self, exception: BaseException) -> str:
        """Sanitize exception message for safe logging and response.

        Removes potential sensitive information and includes exception type.
        Uses SENSITIVE_PATTERNS module constant for redaction patterns.

        See SENSITIVE_PATTERNS constant for full list of covered credential types
        and pattern design documentation.

        Args:
            exception: The exception to sanitize

        Returns:
            Sanitized error string in format: "{ExceptionType}: {sanitized_message}"
        """
        message = str(exception)
        sanitized = self._redact_sensitive_patterns(message)
        sanitized = self._truncate_message(sanitized)
        return f"{type(exception).__name__}: {sanitized}"

    def _redact_sensitive_patterns(self, message: str) -> str:
        """Apply all sensitive pattern redactions to a message.

        Uses SENSITIVE_PATTERNS module constant which contains patterns ordered
        from most specific to most generic for proper redaction precedence.

        Args:
            message: The message to redact sensitive information from

        Returns:
            Message with all sensitive patterns redacted
        """
        for pattern, replacement in SENSITIVE_PATTERNS:
            message = re.sub(
                pattern, replacement, message, flags=re.IGNORECASE | re.DOTALL
            )
        return message

    def _truncate_message(self, message: str, max_length: int = 500) -> str:
        """Truncate message if too long to prevent DoS via large error messages.

        Args:
            message: The message to truncate
            max_length: Maximum allowed length (default: 500)

        Returns:
            Original message if within limit, otherwise truncated with "..."
        """
        if len(message) > max_length:
            return message[: max_length - 3] + "..."
        return message

    # =========================================================================
    # Row Parsing Helpers
    # =========================================================================
    # These methods are extracted from _row_to_node_registration for:
    # - Improved testability (can be tested independently)
    # - Reuse across different row-parsing contexts
    # - Better separation of concerns

    def _parse_json_field(
        self,
        value: JsonValue,
        field_name: str,
        correlation_id: UUID | None = None,
    ) -> dict[str, JsonValue]:
        """Parse a JSON field from database row to a dictionary.

        Handles both pre-parsed dict values (from JSONB columns) and
        string values that need JSON deserialization.

        Args:
            value: The raw value from database (dict or JSON string)
            field_name: Name of the field for logging context
            correlation_id: Optional correlation ID for distributed tracing

        Returns:
            Parsed dictionary, or empty dict if parsing fails

        Note:
            Logs warnings for parse failures but does not raise exceptions.
            This supports graceful degradation when database contains
            malformed data.
        """
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return cast(dict[str, JsonValue], parsed)
                logger.warning(
                    f"JSON parse result not a dict for {field_name}",
                    extra={
                        "correlation_id": (
                            str(correlation_id) if correlation_id else None
                        ),
                        "field_name": field_name,
                    },
                )
            except json.JSONDecodeError as e:
                logger.warning(
                    f"JSON parse failed for {field_name}: {type(e).__name__}",
                    extra={
                        "correlation_id": (
                            str(correlation_id) if correlation_id else None
                        ),
                        "field_name": field_name,
                        "error_type": type(e).__name__,
                    },
                )
        return {}

    def _parse_datetime_field(
        self,
        value: JsonValue,
        field_name: str,
        correlation_id: UUID | None = None,
    ) -> datetime:
        """Parse a datetime field from database row.

        Handles datetime objects (from TIMESTAMP columns) and ISO-8601
        string representations.

        Args:
            value: The raw value from database (datetime or ISO string)
            field_name: Name of the field for logging context
            correlation_id: Optional correlation ID for distributed tracing

        Returns:
            Parsed datetime, or current UTC time as fallback

        Note:
            Falls back to current time when value cannot be parsed.
            This supports graceful degradation while ensuring the model
            always has a valid datetime.
        """
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            # Handle both 'Z' suffix and explicit timezone offset
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        # Fallback to current time when no valid datetime provided
        logger.warning(
            f"Using datetime fallback for {field_name}",
            extra={
                "correlation_id": str(correlation_id) if correlation_id else None,
                "field_name": field_name,
            },
        )
        return datetime.now(UTC)

    def _parse_optional_string_field(
        self,
        value: JsonValue,
    ) -> str | None:
        """Parse an optional string field from database row.

        Args:
            value: The raw value from database

        Returns:
            The string value if valid and non-empty, None otherwise
        """
        if isinstance(value, str) and value:
            return value
        return None

    def _validate_node_id(
        self,
        node_id: str,
        correlation_id: UUID,
        operation: str,
    ) -> None:
        """Validate node_id for security constraints.

        Security Checks:
            1. Non-empty string
            2. Maximum length (prevents DoS via oversized inputs)
            3. Character whitelist (prevents injection attacks)
            4. No null bytes or control characters

        Args:
            node_id: The node identifier to validate
            correlation_id: Correlation ID for logging
            operation: Operation name for error context

        Raises:
            RuntimeHostError: If validation fails with sanitized error details
        """
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation=operation,
            target_name="node_registry_effect",
            correlation_id=correlation_id,
        )

        # Check for empty/whitespace-only
        if not node_id or not node_id.strip():
            raise RuntimeHostError(
                "node_id cannot be empty or whitespace-only",
                context=context,
            )

        # Check length
        if len(node_id) > MAX_NODE_ID_LENGTH:
            raise RuntimeHostError(
                f"node_id exceeds maximum length of {MAX_NODE_ID_LENGTH} characters "
                f"(received: {len(node_id)} characters)",
                context=context,
            )

        # Check for null bytes (critical security check)
        if "\x00" in node_id:
            logger.warning(
                "Null byte detected in node_id - potential attack attempt",
                extra={
                    "event_type": "security_null_byte_detected",
                    "correlation_id": str(correlation_id),
                    "operation": operation,
                    "field": "node_id",
                },
            )
            raise RuntimeHostError(
                "node_id contains invalid characters (null bytes not allowed)",
                context=context,
            )

        # Check character pattern
        if not NODE_ID_PATTERN.match(node_id):
            # Sanitize for logging (replace non-allowed chars)
            sanitized = re.sub(r"[^a-zA-Z0-9._-]", "_", node_id[:50])
            logger.warning(
                "Invalid node_id format rejected",
                extra={
                    "event_type": "security_invalid_node_id",
                    "correlation_id": str(correlation_id),
                    "operation": operation,
                    "sanitized_node_id": sanitized,
                },
            )
            raise RuntimeHostError(
                "node_id contains invalid characters. "
                "Must start with alphanumeric and contain only alphanumeric, "
                "periods, underscores, and hyphens",
                context=context,
            )

    def _validate_filter_values(
        self,
        filters: dict[str, str],
        correlation_id: UUID,
    ) -> None:
        """Validate filter values for security constraints.

        Security Checks:
            1. Maximum value length (prevents DoS via oversized inputs)
            2. No null bytes (prevents null byte injection)
            3. Values are properly parameterized (not interpolated into SQL)

        Note: Filter KEYS are validated separately in _validate_filter_keys().
        This method only validates the VALUES to ensure they are safe for
        parameterized queries.

        Args:
            filters: Dictionary of filter key-value pairs
            correlation_id: Correlation ID for logging

        Raises:
            RuntimeHostError: If any filter value fails validation
        """
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="discover",
            target_name="node_registry_effect",
            correlation_id=correlation_id,
        )

        for key, value in filters.items():
            # Check for null bytes
            if "\x00" in value:
                sanitized_key = re.sub(r"[^a-zA-Z0-9_-]", "_", key[:30])
                logger.warning(
                    "Null byte detected in filter value - potential attack attempt",
                    extra={
                        "event_type": "security_null_byte_in_filter",
                        "correlation_id": str(correlation_id),
                        "filter_key": sanitized_key,
                    },
                )
                raise RuntimeHostError(
                    f"Filter value for '{sanitized_key}' contains invalid characters",
                    context=context,
                )

            # Check value length
            if len(value) > MAX_FILTER_VALUE_LENGTH:
                sanitized_key = re.sub(r"[^a-zA-Z0-9_-]", "_", key[:30])
                raise RuntimeHostError(
                    f"Filter value for '{sanitized_key}' exceeds maximum length "
                    f"of {MAX_FILTER_VALUE_LENGTH} characters",
                    context=context,
                )

    def _validate_introspection_payload(
        self,
        introspection: ModelNodeIntrospectionPayload,
        correlation_id: UUID,
    ) -> None:
        """Validate introspection payload fields for security constraints.

        Security Checks:
            1. node_id: Format, length, and character validation
            2. node_version: Format validation (semver pattern)
            3. health_endpoint: URL format and length validation
            4. endpoints: Key/value length and format validation

        Args:
            introspection: The introspection payload to validate
            correlation_id: Correlation ID for logging

        Raises:
            RuntimeHostError: If any field fails validation
        """
        # Validate node_id (reuse existing validation)
        self._validate_node_id(introspection.node_id, correlation_id, "register")

        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="register",
            target_name="node_registry_effect",
            correlation_id=correlation_id,
        )

        # Validate node_version length and format
        if len(introspection.node_version) > MAX_NODE_VERSION_LENGTH:
            raise RuntimeHostError(
                f"node_version exceeds maximum length of {MAX_NODE_VERSION_LENGTH} "
                f"characters (received: {len(introspection.node_version)})",
                context=context,
            )

        if not VERSION_PATTERN.match(introspection.node_version):
            raise RuntimeHostError(
                f"node_version '{introspection.node_version}' does not match "
                "semantic version format (X.Y.Z with optional suffix)",
                context=context,
            )

        # Validate health_endpoint if provided
        if introspection.health_endpoint:
            if len(introspection.health_endpoint) > MAX_HEALTH_ENDPOINT_LENGTH:
                raise RuntimeHostError(
                    "health_endpoint exceeds maximum length of "
                    f"{MAX_HEALTH_ENDPOINT_LENGTH} characters",
                    context=context,
                )

            if not URL_PATTERN.match(introspection.health_endpoint):
                raise RuntimeHostError(
                    "health_endpoint must be a valid HTTP/HTTPS URL",
                    context=context,
                )

        # Validate endpoints dictionary
        for endpoint_key, endpoint_value in introspection.endpoints.items():
            if len(endpoint_key) > MAX_ENDPOINT_KEY_LENGTH:
                raise RuntimeHostError(
                    f"Endpoint key '{endpoint_key[:20]}...' exceeds maximum "
                    f"length of {MAX_ENDPOINT_KEY_LENGTH} characters",
                    context=context,
                )

            if len(endpoint_value) > MAX_ENDPOINT_VALUE_LENGTH:
                raise RuntimeHostError(
                    f"Endpoint value for '{endpoint_key}' exceeds maximum "
                    f"length of {MAX_ENDPOINT_VALUE_LENGTH} characters",
                    context=context,
                )

    def _json_default_serializer(
        self, obj: object
    ) -> str | list[str] | dict[str, JsonValue]:
        """Default serializer for non-standard JSON types.

        This method is used as the `default` parameter for json.dumps() to handle
        types that are not natively JSON serializable.

        Supported types:
            - datetime: Converted to ISO 8601 format string
            - UUID: Converted to string representation
            - Enum: Converted to its value
            - bytes: Decoded as UTF-8, with fallback to repr()
            - set/frozenset: Converted to sorted list
            - Pydantic BaseModel: Converted via model_dump()
            - Other: Falls back to str() representation

        Args:
            obj: The object to serialize

        Returns:
            A JSON-serializable representation of the object

        Note:
            This method should handle serialization gracefully without raising
            exceptions. If an object cannot be converted, it returns a string
            representation with a warning indicator.
        """
        from enum import Enum

        from pydantic import BaseModel

        # datetime types
        if isinstance(obj, datetime):
            return obj.isoformat()

        # UUID
        if isinstance(obj, UUID):
            return str(obj)

        # Enum values - convert to string value
        if isinstance(obj, Enum):
            return str(obj.value)

        # bytes
        if isinstance(obj, bytes):
            try:
                return obj.decode("utf-8")
            except UnicodeDecodeError:
                return repr(obj)

        # set/frozenset - convert to sorted list for consistent output
        if isinstance(obj, (set, frozenset)):
            try:
                return sorted(str(item) for item in obj)
            except TypeError:
                # Items not comparable, just convert to list
                return [str(item) for item in obj]

        # Pydantic models
        if isinstance(obj, BaseModel):
            try:
                return cast(dict[str, JsonValue], obj.model_dump(mode="json"))
            except (TypeError, ValueError, AttributeError, RecursionError):
                # If model_dump fails (type error, circular reference, etc.),
                # fall back to string representation
                return str(obj)

        # Fallback: string representation with warning indicator
        return f"<non-serializable: {type(obj).__name__}>"

    def _safe_model_dump(
        self,
        model: object,
        correlation_id: UUID | None = None,
        field_name: str = "unknown",
    ) -> dict[str, JsonValue]:
        """Safely dump a Pydantic model to a dictionary.

        This method wraps Pydantic's model_dump() with proper error handling,
        returning an empty dict on failure rather than raising an exception.

        Args:
            model: The Pydantic model to dump (or any object with model_dump method)
            correlation_id: Optional correlation ID for logging
            field_name: Name of the field being serialized for logging

        Returns:
            Dictionary representation of the model, or empty dict on failure
        """
        try:
            # Check if object has model_dump method (Pydantic v2)
            if hasattr(model, "model_dump"):
                result = model.model_dump(mode="json")
                if isinstance(result, dict):
                    return cast(dict[str, JsonValue], result)
                logger.warning(
                    f"model_dump() returned non-dict for {field_name}",
                    extra={
                        "correlation_id": str(correlation_id)
                        if correlation_id
                        else None,
                        "field_name": field_name,
                        "result_type": type(result).__name__,
                    },
                )
                return {}

            # Fallback for dict-like objects
            if hasattr(model, "__dict__"):
                return cast(dict[str, JsonValue], dict(model.__dict__))

            logger.warning(
                f"Object has no model_dump or __dict__ for {field_name}",
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "field_name": field_name,
                    "object_type": type(model).__name__,
                },
            )
            return {}

        except (TypeError, ValueError, AttributeError, RecursionError) as e:
            logger.warning(
                f"Model serialization failed for {field_name}: {type(e).__name__}",
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "field_name": field_name,
                    "error_type": type(e).__name__,
                    "model_type": type(model).__name__,
                },
            )
            return {}

    def _log_operation_performance(
        self,
        operation: str,
        processing_time_ms: float,
        success: bool,
        correlation_id: UUID | None = None,
        node_id: str | None = None,
        record_count: int | None = None,
        status: str | None = None,
    ) -> None:
        """Log structured performance metrics for registry operations.

        Args:
            operation: The operation type (register, deregister, discover, heartbeat)
            processing_time_ms: Time taken to complete the operation in milliseconds
            success: Whether the operation succeeded
            correlation_id: Optional correlation ID for distributed tracing
            node_id: Optional node ID for node-specific operations
            record_count: Optional count of records (for discover operations)
            status: Optional status string (success, partial, failed)
        """
        # Build structured log extra dict
        log_extra: dict[str, str | float | int | bool | None] = {
            "event_type": "registry_operation_complete",
            "operation": operation,
            "processing_time_ms": round(processing_time_ms, 3),
            "correlation_id": str(correlation_id) if correlation_id else None,
            "success": success,
        }

        # Add optional fields only if provided
        if node_id is not None:
            log_extra["node_id"] = node_id
        if record_count is not None:
            log_extra["record_count"] = record_count
        if status is not None:
            log_extra["status"] = status

        # Log the operation completion
        logger.info(
            f"Registry operation completed: {operation}",
            extra=log_extra,
        )

        # Log slow operation warning if threshold exceeded
        if processing_time_ms > self._slow_operation_threshold_ms:
            slow_extra: dict[str, str | float | int | bool | None] = {
                "event_type": "registry_operation_slow",
                "operation": operation,
                "processing_time_ms": round(processing_time_ms, 3),
                "threshold_ms": self._slow_operation_threshold_ms,
                "correlation_id": str(correlation_id) if correlation_id else None,
            }
            if node_id is not None:
                slow_extra["node_id"] = node_id

            logger.warning(
                f"Slow registry operation detected: {operation} took "
                f"{processing_time_ms:.1f}ms (threshold: {self._slow_operation_threshold_ms}ms)",
                extra=slow_extra,
            )

    def _safe_json_dumps(
        self,
        data: JsonValue | dict[str, str],
        correlation_id: UUID | None = None,
        field_name: str = "unknown",
        fallback: str = "{}",
    ) -> str:
        """Safely serialize data to JSON with error handling.

        This method provides robust JSON serialization with:
        - Custom default serializer for non-standard types (datetime, UUID, Enum, etc.)
        - RecursionError handling for deeply nested or circular structures
        - Detailed logging with correlation_id and data type for debugging
        - Fallback value support for graceful degradation

        Args:
            data: The data to serialize (any JSON-serializable value)
            correlation_id: Optional correlation ID for logging
            field_name: Name of the field being serialized for logging
            fallback: Value to return on serialization failure (default: "{}")

        Returns:
            JSON string, or fallback value on serialization failure

        See Also:
            _json_default_serializer: Custom serializer for non-standard types
            _safe_json_dumps_strict: Version that reports errors to caller
        """
        try:
            return json.dumps(data, default=self._json_default_serializer)
        except (TypeError, ValueError, RecursionError) as e:
            logger.warning(
                f"JSON serialization failed for {field_name}: {type(e).__name__}",
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "field_name": field_name,
                    "error_type": type(e).__name__,
                    "data_type": type(data).__name__,
                },
            )
            return fallback

    def _safe_json_dumps_strict(
        self,
        data: JsonValue | dict[str, str],
        correlation_id: UUID | None = None,
        field_name: str = "unknown",
    ) -> tuple[str, str | None]:
        """Safely serialize data to JSON with strict error reporting.

        Unlike _safe_json_dumps, this method returns both the result and any error
        that occurred, allowing callers to decide how to handle serialization failures.

        This method is used for critical serialization paths (like event publishing)
        where the caller needs to know if serialization failed and handle it explicitly.

        Features:
        - Custom default serializer for non-standard types (datetime, UUID, Enum, etc.)
        - RecursionError handling for deeply nested or circular structures
        - Returns error message to caller instead of silently falling back
        - Detailed logging with correlation_id and data type for debugging

        Args:
            data: The data to serialize
            correlation_id: Optional correlation ID for logging
            field_name: Name of the field being serialized for logging

        Returns:
            Tuple of (json_string, error_message). If successful, error_message is None.
            If failed, json_string is "{}" and error_message describes the failure.

        See Also:
            _json_default_serializer: Custom serializer for non-standard types
            _safe_json_dumps: Version that returns fallback on error
        """
        try:
            return json.dumps(data, default=self._json_default_serializer), None
        except (TypeError, ValueError, RecursionError) as e:
            error_msg = (
                f"JSON serialization failed for {field_name}: {type(e).__name__}"
            )
            logger.warning(
                error_msg,
                extra={
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "field_name": field_name,
                    "error_type": type(e).__name__,
                    "data_type": type(data).__name__,
                },
            )
            return "{}", error_msg

    async def initialize(self) -> None:
        """Initialize the effect node and verify backend connectivity.

        This method ensures all dependencies are resolved before marking the node
        as initialized. It is idempotent - calling multiple times is safe and
        will not re-resolve dependencies or re-log initialization messages.

        Behavior:
            - If already initialized, returns immediately (no-op for idempotency)
            - If dependencies have not been resolved, calls _resolve_dependencies()
            - Sets _initialized flag to True
            - Logs initialization status including event bus availability

        Idempotency Guarantees:
            - Multiple calls to initialize() are safe and efficient
            - Dependencies are resolved only once (tracked by _dependencies_resolved)
            - Initialization logging occurs only on first successful initialization
            - No side effects on subsequent calls

        Raises:
            RuntimeError: If dependency resolution fails (missing required handlers).
                Required handlers: consul (ProtocolEnvelopeExecutor),
                postgres (ProtocolEnvelopeExecutor).
                Optional: ProtocolEventBus (for request_introspection operation).

        Example:
            ```python
            node = NodeRegistryEffect(container, config)
            await node.initialize()  # Resolves dependencies and initializes
            await node.initialize()  # Safe to call again (no-op, returns immediately)
            ```

        Note:
            The recommended way to create and initialize a node is via the
            create() factory method, which handles both construction and
            initialization in a single async call.
        """
        # Idempotency: skip if already initialized
        if self._initialized:
            logger.debug(
                "NodeRegistryEffect.initialize() called but already initialized, skipping",
                extra={"event_type": "node_initialize_idempotent_skip"},
            )
            return

        logger.debug(
            "NodeRegistryEffect initialization starting",
            extra={"event_type": "node_initialize_start"},
        )

        # Ensure dependencies are resolved before initializing
        # This makes the node resilient to incorrect usage patterns where
        # users forget to call _resolve_dependencies() before initialize()
        if not self._dependencies_resolved:
            logger.debug(
                "Dependencies not yet resolved, calling _resolve_dependencies()",
                extra={"event_type": "node_initialize_resolve_deps"},
            )
            await self._resolve_dependencies()

        self._initialized = True

        # Log initialization status including optional service availability
        event_bus_status = "available" if self._event_bus is not None else "unavailable"
        logger.info(
            "NodeRegistryEffect initialized successfully",
            extra={
                "event_type": "node_initialize_complete",
                "consul_handler": "available",
                "db_handler": "available",
                "event_bus": event_bus_status,
                "circuit_breaker_threshold": self._config.circuit_breaker_threshold,
                "circuit_breaker_reset_timeout": self._config.circuit_breaker_reset_timeout,
            },
        )

        # Log warning if event bus is unavailable (graceful degradation)
        if self._event_bus is None:
            logger.warning(
                "NodeRegistryEffect initialized without event bus - "
                "request_introspection operation will not be available",
                extra={
                    "event_type": "node_initialize_degraded",
                    "missing_service": "ProtocolEventBus",
                    "unavailable_operations": ["request_introspection"],
                },
            )

    async def shutdown(self) -> None:
        """Shutdown the effect node and cleanup resources.

        This method performs graceful shutdown:
        - Resets the circuit breaker state
        - Marks the node as uninitialized

        After shutdown, the node cannot be used for operations until
        initialize() is called again.

        This method is idempotent and safe to call multiple times.
        """
        async with self._circuit_breaker_lock:
            await self._reset_circuit_breaker()
        self._initialized = False
        logger.info("NodeRegistryEffect shutdown")

    async def execute(self, request: ModelRegistryRequest) -> ModelRegistryResponse:
        """Execute registry operation from request.

        This method routes to the appropriate operation handler based on
        request.operation: register, deregister, discover, or request_introspection.

        Circuit Breaker Behavior:
            This method is protected by MixinAsyncCircuitBreaker. The circuit breaker
            monitors consecutive failures and transitions through three states:

            - CLOSED (normal): Requests proceed normally. Failures increment the
              counter. Success resets the counter.
            - OPEN (blocking): After threshold consecutive failures, the circuit
              opens. All requests fail fast with InfraUnavailableError until
              reset_timeout expires. No backend calls are made.
            - HALF_OPEN (testing): After reset_timeout, the next request is allowed
              through as a test. Success closes the circuit; failure reopens it.

            Configuration (via ModelNodeRegistryEffectConfig):
            - circuit_breaker_threshold: Failures before opening (default: 5)
            - circuit_breaker_reset_timeout: Seconds until auto-reset (default: 60.0)

            Success Criteria:
            - Full success (status="success"): Both backends succeeded, resets circuit
            - Partial success (status="partial"): One backend succeeded, resets circuit
            - Full failure (status="failed"): Both backends failed, does NOT reset circuit

        Retry Guidance for Callers:
            - DO NOT retry when InfraUnavailableError is raised (circuit is open).
              The circuit breaker is protecting degraded backends; retrying wastes
              resources and delays recovery.
            - Check error context for retry_after_seconds which indicates when the
              circuit will attempt auto-reset.
            - For transient failures (status="failed" but circuit not open), implement
              exponential backoff at the caller level (e.g., 1s, 2s, 4s delays).
            - Partial success (status="partial") indicates one backend is healthy;
              the unhealthy backend may recover on subsequent calls.

        Error Recovery Strategies:
            - InfraUnavailableError: Wait for retry_after_seconds, then retry once.
              If still failing, escalate or use fallback data source.
            - RuntimeHostError: Do not retry; fix the request (missing fields,
              invalid operation, not initialized).
            - status="partial": Log warning, consider the operation successful for
              the healthy backend. The failed backend will be retried on next call.
            - status="failed": Implement exponential backoff, consider circuit
              breaker may open after threshold failures.

        Args:
            request: Registry request containing:
                - operation: One of "register", "deregister", "discover",
                  "request_introspection"
                - correlation_id: UUID for distributed tracing
                - introspection_event: Required for "register" operation
                - node_id: Required for "deregister" operation
                - filters: Optional dict for "discover" operation

        Returns:
            ModelRegistryResponse with:
                - operation: Echo of requested operation
                - success: True if at least one backend succeeded
                - status: "success" | "partial" | "failed"
                - consul_result: Result from Consul backend (if applicable)
                - postgres_result: Result from PostgreSQL backend (if applicable)
                - nodes: List of discovered nodes (for "discover" operation)
                - processing_time_ms: Operation duration in milliseconds
                - correlation_id: Echo of request correlation_id
                - error: Sanitized error message (if status="failed")

        Raises:
            RuntimeHostError: If not initialized, invalid request, or missing
                required fields for the operation.
            InfraUnavailableError: If circuit breaker is open. Check
                error.model.context for retry_after_seconds.

        Example:
            ```python
            try:
                response = await registry_effect.execute(request)
                if response.status == "partial":
                    logger.warning("Partial success", extra={...})
            except InfraUnavailableError as e:
                # Circuit is open - do not retry immediately
                retry_after = e.model.context.get("retry_after_seconds", 60)
                logger.exception(f"Service unavailable, retry after {retry_after}s")
            ```

        See Also:
            - MixinAsyncCircuitBreaker: Circuit breaker implementation
            - ModelNodeRegistryEffectConfig: Configuration options
            - docs/patterns/circuit_breaker_implementation.md: Detailed patterns
        """
        if not self._initialized:
            raise RuntimeHostError(
                "NodeRegistryEffect not initialized",
                context=ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="execute",
                    target_name="node_registry_effect",
                    correlation_id=request.correlation_id,
                ),
            )

        # Check circuit breaker
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(
                operation=request.operation,
                correlation_id=request.correlation_id,
            )

        start_time = time.perf_counter()

        try:
            if request.operation == "register":
                return await self._register_node(request, start_time)
            elif request.operation == "deregister":
                return await self._deregister_node(request, start_time)
            elif request.operation == "discover":
                return await self._discover_nodes(request, start_time)
            elif request.operation == "request_introspection":
                return await self._request_introspection(request, start_time)
            else:
                raise RuntimeHostError(
                    f"Unknown operation: {request.operation}",
                    context=ModelInfraErrorContext(
                        transport_type=EnumInfraTransportType.RUNTIME,
                        operation=request.operation,
                        target_name="node_registry_effect",
                        correlation_id=request.correlation_id,
                    ),
                )
        except (RuntimeHostError, InfraUnavailableError):
            # Re-raise our own errors without circuit breaker recording
            raise
        except (OSError, ConnectionError, TimeoutError) as e:
            # Network-level errors - record circuit breaker failure and wrap
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation=request.operation,
                    correlation_id=request.correlation_id,
                )
            logger.warning(
                f"Network error during {request.operation}: {type(e).__name__}",
                extra={
                    "operation": request.operation,
                    "correlation_id": str(request.correlation_id),
                    "error_type": type(e).__name__,
                },
            )
            raise RuntimeHostError(
                f"Network error during {request.operation} operation",
                context=ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation=request.operation,
                    target_name="node_registry_effect",
                    correlation_id=request.correlation_id,
                ),
            ) from e
        except (ValueError, TypeError, KeyError) as e:
            # Data/validation errors - record circuit breaker failure and wrap
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation=request.operation,
                    correlation_id=request.correlation_id,
                )
            sanitized_msg = self._sanitize_error(e)
            logger.warning(
                f"Data error during {request.operation}: {type(e).__name__}",
                extra={
                    "operation": request.operation,
                    "correlation_id": str(request.correlation_id),
                    "error_type": type(e).__name__,
                    "error_message": sanitized_msg,
                },
            )
            raise RuntimeHostError(
                f"Data error during {request.operation} operation: {sanitized_msg}",
                context=ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation=request.operation,
                    target_name="node_registry_effect",
                    correlation_id=request.correlation_id,
                ),
            ) from e
        except Exception as e:
            # Catch-all for unexpected errors - log at error level and wrap
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation=request.operation,
                    correlation_id=request.correlation_id,
                )
            sanitized_msg = self._sanitize_error(e)
            logger.exception(
                f"Unexpected error during {request.operation}: {type(e).__name__}",
                extra={
                    "operation": request.operation,
                    "correlation_id": str(request.correlation_id),
                    "error_type": type(e).__name__,
                    "error_message": sanitized_msg,
                },
            )
            raise RuntimeHostError(
                f"Unexpected error during {request.operation} operation: {sanitized_msg}",
                context=ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation=request.operation,
                    target_name="node_registry_effect",
                    correlation_id=request.correlation_id,
                ),
            ) from e

    async def _register_node(
        self,
        request: ModelRegistryRequest,
        start_time: float,
    ) -> ModelRegistryResponse:
        """Register node with Consul and PostgreSQL in parallel."""
        if request.introspection_event is None:
            raise RuntimeHostError(
                "introspection_event required for register operation",
                context=ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="register",
                    target_name="node_registry_effect",
                    correlation_id=request.correlation_id,
                ),
            )

        introspection = request.introspection_event

        # Security: Validate all input fields before proceeding
        self._validate_introspection_payload(introspection, request.correlation_id)

        # Execute dual registration in parallel
        consul_task = asyncio.create_task(
            self._register_consul(introspection, request.correlation_id)
        )
        postgres_task = asyncio.create_task(
            self._register_postgres(introspection, request.correlation_id)
        )

        consul_result, postgres_result = await asyncio.gather(
            consul_task,
            postgres_task,
            return_exceptions=True,
        )

        # Process results
        consul_op_result = self._process_consul_result(consul_result)
        postgres_op_result = self._process_postgres_result(postgres_result)

        # Determine overall status
        both_success = consul_op_result.success and postgres_op_result.success
        any_success = consul_op_result.success or postgres_op_result.success

        if both_success:
            status: RegistryStatus = "success"
            success = True
        elif any_success:
            status = "partial"
            success = True  # Partial success is still success
        else:
            status = "failed"
            success = False

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        # Reset circuit breaker on success
        if success:
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

        # Log structured performance metrics
        self._log_operation_performance(
            operation="register",
            processing_time_ms=processing_time_ms,
            success=success,
            correlation_id=request.correlation_id,
            node_id=introspection.node_id,
            status=status,
        )

        return ModelRegistryResponse(
            operation="register",
            success=success,
            status=status,
            consul_result=consul_op_result,
            postgres_result=postgres_op_result,
            processing_time_ms=processing_time_ms,
            correlation_id=request.correlation_id,
        )

    async def _register_consul(
        self,
        introspection: ModelNodeIntrospectionPayload,
        correlation_id: UUID,
    ) -> ModelConsulOperationResult:
        """Register node with Consul."""
        try:
            # Build Consul registration payload
            consul_payload: dict[str, JsonValue] = {
                "name": introspection.node_id,
                "service_id": introspection.node_id,
                "tags": [introspection.node_type, f"v{introspection.node_version}"],
            }

            # Add health check if endpoint provided
            if introspection.health_endpoint:
                consul_payload["check"] = {
                    "http": introspection.health_endpoint,
                    "interval": "30s",
                    "timeout": "10s",
                }

            # Execute via consul handler (Protocol-typed)
            result = await self.consul_handler.execute(
                {
                    "operation": "consul.register",
                    "payload": consul_payload,
                    "correlation_id": correlation_id,
                }
            )

            return ModelConsulOperationResult(
                success=result.get("status") == "success",
                service_id=introspection.node_id,
            )
        except (OSError, ConnectionError, TimeoutError) as e:
            # Network-level errors (connection refused, timeout, DNS failure)
            logger.warning(
                f"Consul registration failed (network error): {type(e).__name__}",
                extra={
                    "node_id": introspection.node_id,
                    "correlation_id": str(correlation_id),
                    "error_type": type(e).__name__,
                },
            )
            return ModelConsulOperationResult(
                success=False,
                error=self._sanitize_error(e),
            )
        except (ValueError, TypeError, KeyError) as e:
            # Data/protocol errors (malformed response, missing keys)
            logger.warning(
                f"Consul registration failed (data error): {type(e).__name__}",
                extra={
                    "node_id": introspection.node_id,
                    "correlation_id": str(correlation_id),
                    "error_type": type(e).__name__,
                },
            )
            return ModelConsulOperationResult(
                success=False,
                error=self._sanitize_error(e),
            )
        except RuntimeHostError:
            # Re-raise our infrastructure errors (already well-typed)
            raise
        except Exception as e:
            # Catch-all for unexpected errors - log at error level for investigation
            logger.exception(
                f"Consul registration failed (unexpected error): {type(e).__name__}",
                extra={
                    "node_id": introspection.node_id,
                    "correlation_id": str(correlation_id),
                    "error_type": type(e).__name__,
                },
            )
            return ModelConsulOperationResult(
                success=False,
                error=self._sanitize_error(e),
            )

    async def _register_postgres(
        self,
        introspection: ModelNodeIntrospectionPayload,
        correlation_id: UUID,
    ) -> ModelPostgresOperationResult:
        """Register node in PostgreSQL with UPSERT."""
        try:
            # Build UPSERT query for node_registrations table
            upsert_sql = """
                INSERT INTO node_registrations (
                    node_id, node_type, node_version, capabilities,
                    endpoints, metadata, health_endpoint, registered_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), NOW())
                ON CONFLICT (node_id) DO UPDATE SET
                    node_type = EXCLUDED.node_type,
                    node_version = EXCLUDED.node_version,
                    capabilities = EXCLUDED.capabilities,
                    endpoints = EXCLUDED.endpoints,
                    metadata = EXCLUDED.metadata,
                    health_endpoint = EXCLUDED.health_endpoint,
                    updated_at = NOW()
            """

            # Execute via db handler (Protocol-typed)
            # Use _safe_model_dump to handle Pydantic serialization errors
            # before passing to _safe_json_dumps for JSON encoding.
            # This provides two layers of error handling:
            # 1. _safe_model_dump catches Pydantic model_dump() failures
            # 2. _safe_json_dumps catches json.dumps() failures
            result = await self.db_handler.execute(
                {
                    "operation": "db.execute",
                    "payload": {
                        "sql": upsert_sql,
                        "params": [
                            introspection.node_id,
                            introspection.node_type,
                            introspection.node_version,
                            self._safe_json_dumps(
                                self._safe_model_dump(
                                    introspection.capabilities,
                                    correlation_id,
                                    "capabilities",
                                ),
                                correlation_id,
                                "capabilities",
                            ),
                            self._safe_json_dumps(
                                introspection.endpoints, correlation_id, "endpoints"
                            ),
                            self._safe_json_dumps(
                                self._safe_model_dump(
                                    introspection.runtime_metadata,
                                    correlation_id,
                                    "runtime_metadata",
                                ),
                                correlation_id,
                                "runtime_metadata",
                            ),
                            introspection.health_endpoint,
                        ],
                    },
                    "correlation_id": correlation_id,
                }
            )

            payload = result.get("payload", {})
            rows_affected = 1
            if isinstance(payload, dict):
                raw_rows = payload.get("rows_affected", 1)
                if isinstance(raw_rows, int):
                    rows_affected = raw_rows

            return ModelPostgresOperationResult(
                success=result.get("status") == "success",
                rows_affected=rows_affected,
            )
        except (OSError, ConnectionError, TimeoutError) as e:
            # Network-level errors (connection refused, timeout, DNS failure)
            logger.warning(
                f"PostgreSQL registration failed (network error): {type(e).__name__}",
                extra={
                    "node_id": introspection.node_id,
                    "correlation_id": str(correlation_id),
                    "error_type": type(e).__name__,
                },
            )
            return ModelPostgresOperationResult(
                success=False,
                error=self._sanitize_error(e),
            )
        except (ValueError, TypeError, KeyError) as e:
            # Data/protocol errors (malformed response, missing keys)
            logger.warning(
                f"PostgreSQL registration failed (data error): {type(e).__name__}",
                extra={
                    "node_id": introspection.node_id,
                    "correlation_id": str(correlation_id),
                    "error_type": type(e).__name__,
                },
            )
            return ModelPostgresOperationResult(
                success=False,
                error=self._sanitize_error(e),
            )
        except RuntimeHostError:
            # Re-raise our infrastructure errors (already well-typed)
            raise
        except Exception as e:
            # Catch-all for unexpected errors - log at error level for investigation
            logger.exception(
                f"PostgreSQL registration failed (unexpected error): {type(e).__name__}",
                extra={
                    "node_id": introspection.node_id,
                    "correlation_id": str(correlation_id),
                    "error_type": type(e).__name__,
                },
            )
            return ModelPostgresOperationResult(
                success=False,
                error=self._sanitize_error(e),
            )

    async def _deregister_node(
        self,
        request: ModelRegistryRequest,
        start_time: float,
    ) -> ModelRegistryResponse:
        """Deregister node from Consul and PostgreSQL in parallel."""
        if not request.node_id:
            raise RuntimeHostError(
                "node_id required for deregister operation",
                context=ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="deregister",
                    target_name="node_registry_effect",
                    correlation_id=request.correlation_id,
                ),
            )

        # Security: Validate node_id format before proceeding
        self._validate_node_id(request.node_id, request.correlation_id, "deregister")

        # Execute dual deregistration in parallel
        consul_task = asyncio.create_task(
            self._deregister_consul(request.node_id, request.correlation_id)
        )
        postgres_task = asyncio.create_task(
            self._deregister_postgres(request.node_id, request.correlation_id)
        )

        consul_result, postgres_result = await asyncio.gather(
            consul_task,
            postgres_task,
            return_exceptions=True,
        )

        consul_op_result = self._process_consul_result(consul_result)
        postgres_op_result = self._process_postgres_result(postgres_result)

        both_success = consul_op_result.success and postgres_op_result.success
        any_success = consul_op_result.success or postgres_op_result.success

        if both_success:
            status: RegistryStatus = "success"
            success = True
        elif any_success:
            status = "partial"
            success = True
        else:
            status = "failed"
            success = False

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        if success:
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

        # Log structured performance metrics
        self._log_operation_performance(
            operation="deregister",
            processing_time_ms=processing_time_ms,
            success=success,
            correlation_id=request.correlation_id,
            node_id=request.node_id,
            status=status,
        )

        return ModelRegistryResponse(
            operation="deregister",
            success=success,
            status=status,
            consul_result=consul_op_result,
            postgres_result=postgres_op_result,
            processing_time_ms=processing_time_ms,
            correlation_id=request.correlation_id,
        )

    async def _deregister_consul(
        self,
        node_id: str,
        correlation_id: UUID,
    ) -> ModelConsulOperationResult:
        """Deregister node from Consul."""
        try:
            # Execute via consul handler (Protocol-typed)
            result = await self.consul_handler.execute(
                {
                    "operation": "consul.deregister",
                    "payload": {"service_id": node_id},
                    "correlation_id": correlation_id,
                }
            )
            return ModelConsulOperationResult(
                success=result.get("status") == "success",
                service_id=node_id,
            )
        except (OSError, ConnectionError, TimeoutError) as e:
            # Network-level errors (connection refused, timeout, DNS failure)
            logger.warning(
                f"Consul deregistration failed (network error): {type(e).__name__}",
                extra={
                    "node_id": node_id,
                    "correlation_id": str(correlation_id),
                    "error_type": type(e).__name__,
                },
            )
            return ModelConsulOperationResult(
                success=False,
                error=self._sanitize_error(e),
            )
        except (ValueError, TypeError, KeyError) as e:
            # Data/protocol errors (malformed response, missing keys)
            logger.warning(
                f"Consul deregistration failed (data error): {type(e).__name__}",
                extra={
                    "node_id": node_id,
                    "correlation_id": str(correlation_id),
                    "error_type": type(e).__name__,
                },
            )
            return ModelConsulOperationResult(
                success=False,
                error=self._sanitize_error(e),
            )
        except RuntimeHostError:
            # Re-raise our infrastructure errors (already well-typed)
            raise
        except Exception as e:
            # Catch-all for unexpected errors - log at error level for investigation
            logger.exception(
                f"Consul deregistration failed (unexpected error): {type(e).__name__}",
                extra={
                    "node_id": node_id,
                    "correlation_id": str(correlation_id),
                    "error_type": type(e).__name__,
                },
            )
            return ModelConsulOperationResult(
                success=False,
                error=self._sanitize_error(e),
            )

    async def _deregister_postgres(
        self,
        node_id: str,
        correlation_id: UUID,
    ) -> ModelPostgresOperationResult:
        """Delete node from PostgreSQL."""
        try:
            # Execute via db handler (Protocol-typed)
            result = await self.db_handler.execute(
                {
                    "operation": "db.execute",
                    "payload": {
                        "sql": "DELETE FROM node_registrations WHERE node_id = $1",
                        "params": [node_id],
                    },
                    "correlation_id": correlation_id,
                }
            )
            payload = result.get("payload", {})
            rows_affected = 0
            if isinstance(payload, dict):
                raw_rows = payload.get("rows_affected", 0)
                if isinstance(raw_rows, int):
                    rows_affected = raw_rows

            return ModelPostgresOperationResult(
                success=result.get("status") == "success",
                rows_affected=rows_affected,
            )
        except (OSError, ConnectionError, TimeoutError) as e:
            # Network-level errors (connection refused, timeout, DNS failure)
            logger.warning(
                f"PostgreSQL deregistration failed (network error): {type(e).__name__}",
                extra={
                    "node_id": node_id,
                    "correlation_id": str(correlation_id),
                    "error_type": type(e).__name__,
                },
            )
            return ModelPostgresOperationResult(
                success=False,
                error=self._sanitize_error(e),
            )
        except (ValueError, TypeError, KeyError) as e:
            # Data/protocol errors (malformed response, missing keys)
            logger.warning(
                f"PostgreSQL deregistration failed (data error): {type(e).__name__}",
                extra={
                    "node_id": node_id,
                    "correlation_id": str(correlation_id),
                    "error_type": type(e).__name__,
                },
            )
            return ModelPostgresOperationResult(
                success=False,
                error=self._sanitize_error(e),
            )
        except RuntimeHostError:
            # Re-raise our infrastructure errors (already well-typed)
            raise
        except Exception as e:
            # Catch-all for unexpected errors - log at error level for investigation
            logger.exception(
                f"PostgreSQL deregistration failed (unexpected error): {type(e).__name__}",
                extra={
                    "node_id": node_id,
                    "correlation_id": str(correlation_id),
                    "error_type": type(e).__name__,
                },
            )
            return ModelPostgresOperationResult(
                success=False,
                error=self._sanitize_error(e),
            )

    def _validate_filter_keys(
        self,
        filters: dict[str, str],
        correlation_id: UUID,
    ) -> list[str]:
        """Validate filter keys against the whitelist.

        SECURITY: This method prevents SQL injection by ensuring only known-safe
        column names can be used in SQL queries. Invalid keys are rejected, not
        silently ignored, to prevent attackers from probing for vulnerabilities.

        Args:
            filters: Dictionary of filter key-value pairs from the request.
            correlation_id: Correlation ID for security logging.

        Returns:
            List of invalid filter keys (empty if all keys are valid).

        Security Note:
            Invalid filter keys are logged at WARNING level for security monitoring.
            The log includes a sanitized version of the key (truncated, special chars
            removed) to prevent log injection attacks while still enabling detection
            of SQL injection attempts.
        """
        invalid_keys: list[str] = []

        for key in filters:
            if key not in ALLOWED_FILTER_KEYS:
                invalid_keys.append(key)
                # Log security event with sanitized key to prevent log injection
                # Truncate and remove special characters for safe logging
                sanitized_key = re.sub(r"[^a-zA-Z0-9_\-]", "_", key[:50])
                logger.warning(
                    "Invalid filter key rejected in discover operation",
                    extra={
                        "event_type": "security_filter_key_rejected",
                        "correlation_id": str(correlation_id),
                        "sanitized_key": sanitized_key,
                        "key_length": len(key),
                        "allowed_keys": list(ALLOWED_FILTER_KEYS),
                    },
                )

        return invalid_keys

    async def _discover_nodes(
        self,
        request: ModelRegistryRequest,
        start_time: float,
    ) -> ModelRegistryResponse:
        """Query registered nodes from PostgreSQL with optional filters.

        Security:
            Filter keys are validated against ALLOWED_FILTER_KEYS whitelist.
            Invalid filter keys cause the request to be rejected with a
            RuntimeHostError, preventing SQL injection attempts. Filter values
            are always parameterized (never interpolated into SQL).
        """
        # Validate filter keys BEFORE building SQL query (SQL injection prevention)
        if request.filters:
            invalid_keys = self._validate_filter_keys(
                request.filters, request.correlation_id
            )
            if invalid_keys:
                # Reject request with invalid filter keys - do not silently ignore
                # Sanitize invalid keys for error message (no SQL structure leaked)
                sanitized_invalid = [
                    re.sub(r"[^a-zA-Z0-9_\-]", "_", k[:30]) for k in invalid_keys[:5]
                ]
                raise RuntimeHostError(
                    f"Invalid filter keys: {sanitized_invalid}. "
                    f"Allowed keys: {sorted(ALLOWED_FILTER_KEYS)}",
                    context=ModelInfraErrorContext(
                        transport_type=EnumInfraTransportType.DATABASE,
                        operation="discover",
                        target_name="node_registry_effect",
                        correlation_id=request.correlation_id,
                    ),
                )

            # Security: Validate filter values for length and null bytes
            self._validate_filter_values(request.filters, request.correlation_id)

        try:
            # Build query with validated filters
            sql = "SELECT * FROM node_registrations"
            params: list[str] = []

            if request.filters:
                conditions = []
                param_idx = 1
                for key, value in request.filters.items():
                    # Keys already validated above - safe to interpolate column names
                    conditions.append(f"{key} = ${param_idx}")
                    params.append(value)
                    param_idx += 1
                if conditions:
                    sql += " WHERE " + " AND ".join(conditions)

            # Execute via db handler (Protocol-typed)
            result = await self.db_handler.execute(
                {
                    "operation": "db.query",
                    "payload": {"sql": sql, "params": params},
                    "correlation_id": request.correlation_id,
                }
            )

            # Parse results into ModelNodeRegistration
            payload = result.get("payload", {})
            rows: list[dict[str, JsonValue]] = []
            if isinstance(payload, dict):
                raw_rows = payload.get("rows", [])
                if isinstance(raw_rows, list):
                    rows = cast(list[dict[str, JsonValue]], raw_rows)
            nodes = [
                self._row_to_node_registration(row, request.correlation_id)
                for row in rows
            ]

            processing_time_ms = (time.perf_counter() - start_time) * 1000

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            # Log structured performance metrics for successful discovery
            self._log_operation_performance(
                operation="discover",
                processing_time_ms=processing_time_ms,
                success=True,
                correlation_id=request.correlation_id,
                record_count=len(nodes),
                status="success",
            )

            return ModelRegistryResponse(
                operation="discover",
                success=True,
                status="success",
                nodes=nodes,
                processing_time_ms=processing_time_ms,
                correlation_id=request.correlation_id,
            )
        except (OSError, ConnectionError, TimeoutError) as e:
            # Network-level errors (connection refused, timeout, DNS failure)
            # Record circuit breaker failure
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("discover", request.correlation_id)

            logger.warning(
                f"Node discovery failed (network error): {type(e).__name__}",
                extra={
                    "filters": request.filters,
                    "correlation_id": str(request.correlation_id),
                    "error_type": type(e).__name__,
                },
            )
            processing_time_ms = (time.perf_counter() - start_time) * 1000

            # Log structured performance metrics for network failure
            self._log_operation_performance(
                operation="discover",
                processing_time_ms=processing_time_ms,
                success=False,
                correlation_id=request.correlation_id,
                record_count=0,
                status="failed",
            )

            return ModelRegistryResponse(
                operation="discover",
                success=False,
                status="failed",
                error=self._sanitize_error(e),
                processing_time_ms=processing_time_ms,
                correlation_id=request.correlation_id,
            )
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as e:
            # Data/protocol errors (malformed response, missing keys, invalid JSON)
            # Record circuit breaker failure
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("discover", request.correlation_id)

            logger.warning(
                f"Node discovery failed (data error): {type(e).__name__}",
                extra={
                    "filters": request.filters,
                    "correlation_id": str(request.correlation_id),
                    "error_type": type(e).__name__,
                },
            )
            processing_time_ms = (time.perf_counter() - start_time) * 1000

            # Log structured performance metrics for data error
            self._log_operation_performance(
                operation="discover",
                processing_time_ms=processing_time_ms,
                success=False,
                correlation_id=request.correlation_id,
                record_count=0,
                status="failed",
            )

            return ModelRegistryResponse(
                operation="discover",
                success=False,
                status="failed",
                error=self._sanitize_error(e),
                processing_time_ms=processing_time_ms,
                correlation_id=request.correlation_id,
            )
        except RuntimeHostError:
            # Re-raise our infrastructure errors (already well-typed)
            raise
        except Exception as e:
            # Catch-all for unexpected errors - log at error level for investigation
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("discover", request.correlation_id)

            logger.exception(
                f"Node discovery failed (unexpected error): {type(e).__name__}",
                extra={
                    "filters": request.filters,
                    "correlation_id": str(request.correlation_id),
                    "error_type": type(e).__name__,
                },
            )
            processing_time_ms = (time.perf_counter() - start_time) * 1000

            # Log structured performance metrics for failed discovery
            self._log_operation_performance(
                operation="discover",
                processing_time_ms=processing_time_ms,
                success=False,
                correlation_id=request.correlation_id,
                record_count=0,
                status="failed",
            )

            return ModelRegistryResponse(
                operation="discover",
                success=False,
                status="failed",
                error=self._sanitize_error(e),
                processing_time_ms=processing_time_ms,
                correlation_id=request.correlation_id,
            )

    async def _request_introspection(
        self,
        request: ModelRegistryRequest,
        start_time: float,
    ) -> ModelRegistryResponse:
        """Publish introspection request to event bus."""
        if self._event_bus is None:
            raise RuntimeHostError(
                "Event bus not configured for request_introspection",
                context=ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="request_introspection",
                    target_name="node_registry_effect",
                    correlation_id=request.correlation_id,
                ),
            )

        try:
            # Publish REQUEST_INTROSPECTION event (Protocol-typed)
            event_payload = {
                "event_type": "REGISTRY_REQUEST_INTROSPECTION",
                "correlation_id": str(request.correlation_id),
            }

            # Use strict serialization - don't publish malformed events
            json_payload, serialization_error = self._safe_json_dumps_strict(
                event_payload, request.correlation_id, "introspection_event"
            )
            if serialization_error:
                processing_time_ms = (time.perf_counter() - start_time) * 1000

                # Log structured performance metrics for serialization failure
                self._log_operation_performance(
                    operation="request_introspection",
                    processing_time_ms=processing_time_ms,
                    success=False,
                    correlation_id=request.correlation_id,
                    status="failed",
                )

                return ModelRegistryResponse(
                    operation="request_introspection",
                    success=False,
                    status="failed",
                    error=serialization_error,
                    processing_time_ms=processing_time_ms,
                    correlation_id=request.correlation_id,
                )

            await self._event_bus.publish(
                topic="onex.evt.registry-request-introspection.v1",
                key=b"registry",
                value=json_payload.encode("utf-8"),
            )

            processing_time_ms = (time.perf_counter() - start_time) * 1000

            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            # Log structured performance metrics for successful introspection request
            self._log_operation_performance(
                operation="request_introspection",
                processing_time_ms=processing_time_ms,
                success=True,
                correlation_id=request.correlation_id,
                status="success",
            )

            return ModelRegistryResponse(
                operation="request_introspection",
                success=True,
                status="success",
                processing_time_ms=processing_time_ms,
                correlation_id=request.correlation_id,
            )
        except (OSError, ConnectionError, TimeoutError) as e:
            # Network-level errors (connection refused, timeout, DNS failure)
            logger.warning(
                f"Introspection request failed (network error): {type(e).__name__}",
                extra={
                    "correlation_id": str(request.correlation_id),
                    "error_type": type(e).__name__,
                },
            )
            processing_time_ms = (time.perf_counter() - start_time) * 1000

            # Log structured performance metrics for failed introspection request
            self._log_operation_performance(
                operation="request_introspection",
                processing_time_ms=processing_time_ms,
                success=False,
                correlation_id=request.correlation_id,
                status="failed",
            )

            return ModelRegistryResponse(
                operation="request_introspection",
                success=False,
                status="failed",
                error=self._sanitize_error(e),
                processing_time_ms=processing_time_ms,
                correlation_id=request.correlation_id,
            )
        except (ValueError, TypeError, KeyError) as e:
            # Data/protocol errors (malformed response, missing keys)
            logger.warning(
                f"Introspection request failed (data error): {type(e).__name__}",
                extra={
                    "correlation_id": str(request.correlation_id),
                    "error_type": type(e).__name__,
                },
            )
            processing_time_ms = (time.perf_counter() - start_time) * 1000

            # Log structured performance metrics for failed introspection request
            self._log_operation_performance(
                operation="request_introspection",
                processing_time_ms=processing_time_ms,
                success=False,
                correlation_id=request.correlation_id,
                status="failed",
            )

            return ModelRegistryResponse(
                operation="request_introspection",
                success=False,
                status="failed",
                error=self._sanitize_error(e),
                processing_time_ms=processing_time_ms,
                correlation_id=request.correlation_id,
            )
        except RuntimeHostError:
            # Re-raise our infrastructure errors (already well-typed)
            raise
        except Exception as e:
            # Catch-all for unexpected errors - log at error level for investigation
            logger.exception(
                f"Introspection request failed (unexpected error): {type(e).__name__}",
                extra={
                    "correlation_id": str(request.correlation_id),
                    "error_type": type(e).__name__,
                },
            )
            processing_time_ms = (time.perf_counter() - start_time) * 1000

            # Log structured performance metrics for failed introspection request
            self._log_operation_performance(
                operation="request_introspection",
                processing_time_ms=processing_time_ms,
                success=False,
                correlation_id=request.correlation_id,
                status="failed",
            )

            return ModelRegistryResponse(
                operation="request_introspection",
                success=False,
                status="failed",
                error=self._sanitize_error(e),
                processing_time_ms=processing_time_ms,
                correlation_id=request.correlation_id,
            )

    def _process_consul_result(
        self,
        result: ModelConsulOperationResult | BaseException,
    ) -> ModelConsulOperationResult:
        """Process consul task result, handling exceptions."""
        if isinstance(result, BaseException):
            return ModelConsulOperationResult(
                success=False,
                error=self._sanitize_error(result),
            )
        return result

    def _process_postgres_result(
        self,
        result: ModelPostgresOperationResult | BaseException,
    ) -> ModelPostgresOperationResult:
        """Process postgres task result, handling exceptions."""
        if isinstance(result, BaseException):
            return ModelPostgresOperationResult(
                success=False,
                error=self._sanitize_error(result),
            )
        return result

    def _validate_metadata_field_type(
        self,
        value: JsonValue,
        expected_type: type,
        field_name: str,
        parent_field: str,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Validate metadata field type and log warning if unexpected.

        This method ensures no silent data loss occurs when parsing database
        rows. When a field has an unexpected type, a warning is logged with
        full context for debugging and auditing.

        Args:
            value: The value to validate
            expected_type: The expected Python type (list, dict, str, etc.)
            field_name: Name of the specific field (e.g., "tags", "labels")
            parent_field: Name of the parent field (e.g., "metadata", "capabilities")
            correlation_id: Optional correlation ID for logging

        Returns:
            True if the value is of the expected type, False otherwise
        """
        if isinstance(value, expected_type):
            return True

        # Log detailed warning about unexpected type
        actual_type = type(value).__name__ if value is not None else "None"
        logger.warning(
            f"Unexpected type for {parent_field}.{field_name}: "
            f"expected {expected_type.__name__}, got {actual_type}. "
            f"Using default value to prevent data corruption.",
            extra={
                "event_type": "metadata_type_validation_warning",
                "correlation_id": str(correlation_id) if correlation_id else None,
                "field_name": field_name,
                "parent_field": parent_field,
                "expected_type": expected_type.__name__,
                "actual_type": actual_type,
                "actual_value_repr": repr(value)[:100] if value is not None else "None",
            },
        )
        return False

    def _row_to_node_registration(
        self,
        row: dict[str, JsonValue],
        correlation_id: UUID | None = None,
    ) -> ModelNodeRegistration:
        """Convert database row to ModelNodeRegistration.

        Uses extracted helper methods for parsing:
        - _parse_json_field: Parses JSONB/JSON string columns
        - _parse_datetime_field: Parses TIMESTAMP/ISO string columns
        - _parse_optional_string_field: Parses nullable string columns

        Args:
            row: Database row dictionary with JSON-serializable values
            correlation_id: Optional correlation ID for logging

        Note:
            This method validates all metadata field types and logs warnings
            when unexpected types are encountered. This prevents silent data
            loss while maintaining backward compatibility with graceful degradation.
        """
        # Handle health_endpoint which can be str or None
        health_endpoint = self._parse_optional_string_field(row.get("health_endpoint"))

        # Handle last_heartbeat which can be datetime, str, or None
        last_heartbeat_raw = row.get("last_heartbeat")
        last_heartbeat: datetime | None = None
        if last_heartbeat_raw is not None:
            last_heartbeat = self._parse_datetime_field(
                last_heartbeat_raw, "last_heartbeat", correlation_id
            )

        # Parse timestamps - use current time if missing (shouldn't happen in valid data)
        registered_at_raw = row.get("registered_at")
        registered_at = (
            self._parse_datetime_field(
                registered_at_raw, "registered_at", correlation_id
            )
            if registered_at_raw is not None
            else datetime.now(UTC)
        )

        updated_at_raw = row.get("updated_at")
        updated_at = (
            self._parse_datetime_field(updated_at_raw, "updated_at", correlation_id)
            if updated_at_raw is not None
            else datetime.now(UTC)
        )

        # Convert endpoints dict to proper type (values must be strings)
        raw_endpoints = self._parse_json_field(
            row.get("endpoints", {}), "endpoints", correlation_id
        )
        endpoints: dict[str, str] = {
            str(k): str(v) for k, v in raw_endpoints.items() if isinstance(v, str)
        }

        # Parse capabilities from database and convert to ModelNodeCapabilitiesInfo
        # Validate field types and log warnings for unexpected values
        raw_capabilities = self._parse_json_field(
            row.get("capabilities", {}), "capabilities", correlation_id
        )

        raw_capabilities_list = raw_capabilities.get("capabilities", [])
        capabilities_list: list[str] = (
            cast(list[str], raw_capabilities_list)
            if self._validate_metadata_field_type(
                raw_capabilities_list,
                list,
                "capabilities",
                "capabilities",
                correlation_id,
            )
            else []
        )

        raw_supported_operations = raw_capabilities.get("supported_operations", [])
        supported_operations_list: list[str] = (
            cast(list[str], raw_supported_operations)
            if self._validate_metadata_field_type(
                raw_supported_operations,
                list,
                "supported_operations",
                "capabilities",
                correlation_id,
            )
            else []
        )

        capabilities = ModelNodeCapabilitiesInfo(
            capabilities=capabilities_list,
            supported_operations=supported_operations_list,
        )

        # Parse runtime_metadata from database and convert to ModelNodeRegistrationMetadata
        # Database column is 'metadata', but model field is 'runtime_metadata'
        # Validate field types and log warnings for unexpected values to prevent silent data loss
        raw_metadata = self._parse_json_field(
            row.get("metadata", {}), "metadata", correlation_id
        )

        # Validate environment field
        env_str = raw_metadata.get("environment", "testing")
        if not self._validate_metadata_field_type(
            env_str, str, "environment", "metadata", correlation_id
        ):
            env_str = "testing"
        try:
            environment = (
                EnumEnvironment(env_str) if env_str else EnumEnvironment.TESTING
            )
        except ValueError:
            logger.warning(
                f"Invalid environment value '{env_str}', using TESTING as default",
                extra={
                    "event_type": "metadata_type_validation_warning",
                    "correlation_id": str(correlation_id) if correlation_id else None,
                    "field_name": "environment",
                    "parent_field": "metadata",
                    "invalid_value": str(env_str)[:50],
                },
            )
            environment = EnumEnvironment.TESTING

        # Validate tags field (must be list)
        raw_tags = raw_metadata.get("tags", [])
        tags_list: list[str] = (
            cast(list[str], raw_tags)
            if self._validate_metadata_field_type(
                raw_tags, list, "tags", "metadata", correlation_id
            )
            else []
        )

        # Validate labels field (must be dict)
        raw_labels = raw_metadata.get("labels", {})
        labels_dict: dict[str, str] = (
            cast(dict[str, str], raw_labels)
            if self._validate_metadata_field_type(
                raw_labels, dict, "labels", "metadata", correlation_id
            )
            else {}
        )

        # Validate release_channel field (must be str or None)
        raw_release_channel = raw_metadata.get("release_channel")
        release_channel: str | None = None
        if raw_release_channel is not None:
            if self._validate_metadata_field_type(
                raw_release_channel, str, "release_channel", "metadata", correlation_id
            ):
                release_channel = cast(str, raw_release_channel)

        # Validate region field (must be str or None)
        raw_region = raw_metadata.get("region")
        region: str | None = None
        if raw_region is not None:
            if self._validate_metadata_field_type(
                raw_region, str, "region", "metadata", correlation_id
            ):
                region = cast(str, raw_region)

        runtime_metadata = ModelNodeRegistrationMetadata(
            environment=environment,
            tags=tags_list,
            labels=labels_dict,
            release_channel=release_channel,
            region=region,
        )

        return ModelNodeRegistration(
            node_id=str(row.get("node_id", "")),
            node_type=str(row.get("node_type", "")),
            node_version=str(row.get("node_version", "1.0.0")),
            capabilities=capabilities,
            endpoints=endpoints,
            runtime_metadata=runtime_metadata,
            health_endpoint=health_endpoint,
            last_heartbeat=last_heartbeat,
            registered_at=registered_at,
            updated_at=updated_at,
        )


__all__ = ["NodeRegistryEffect"]
