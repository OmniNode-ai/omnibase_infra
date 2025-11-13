#!/usr/bin/env python3
"""
NodeBridgeRegistry - Node Discovery and Dual Registration System (Production Ready).

Listens for introspection events from nodes and performs dual registration
in both Consul (service discovery) and PostgreSQL (tool registry database).

ONEX v2.0 Compliance:
- Suffix-based naming: NodeBridgeRegistry
- Import from omnibase_core infrastructure (with stubs for bridge environment)
- ModelContainer for dependency injection
- Strong typing (no Any types except where needed for flexibility)
- FSM-driven lifecycle management

Production Readiness Fixes v1.0:
- Memory leak prevention with TTL cache for offset tracking
- Race condition prevention with cleanup locks
- Circuit breaker pattern for error recovery
- Atomic registration with proper transaction handling
- Security-aware logging with sensitive data masking
- Configurable magic numbers via environment variables
- Performance-optimized FSM state discovery caching
"""

import asyncio
import logging
import os
import sys
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar, Optional, TypedDict
from uuid import uuid4

if TYPE_CHECKING:
    from ....services.kafka_client import KafkaClient
    from ....services.metadata_stamping.registry.consul_client import (
        RegistryConsulClient,
    )
    from ....services.node_registration_repository import NodeRegistrationRepository
    from ....services.postgres_client import PostgresClient

# Import with fallback to stubs when omnibase_core is not available
try:
    from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
    from omnibase_core.errors.error_codes import EnumCoreErrorCode
    from omnibase_core.errors.model_onex_error import ModelOnexError
    from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
    from omnibase_core.nodes.node_effect import NodeEffect
except ImportError:
    # Fallback to stubs when omnibase_core is not available (testing/demo mode)
    from ._stubs import (
        EnumCoreErrorCode,
        LogLevel,
        ModelContractEffect,
        ModelONEXContainer,
        ModelOnexError,
        NodeEffect,
        emit_log_event,
    )

# Aliases for compatibility
OnexError = ModelOnexError

# Import health check mixin
# Import production-ready utilities
from ....config.config_loader import get_registry_config
from ....config.registry_config import get_registry_config as get_registry_config_legacy
from ....utils.circuit_breaker import create_circuit_breaker
from ....utils.secure_logging import get_secure_logger, sanitize_log_data
from ....utils.ttl_cache import create_ttl_cache
from ...mixins.health_mixin import HealthCheckMixin, HealthStatus

# Import introspection mixin
from ...mixins.introspection_mixin import IntrospectionMixin

# Import event models
from ...orchestrator.v1_0_0.models.model_node_introspection_event import (
    ModelNodeIntrospectionEvent,
)
from ...orchestrator.v1_0_0.models.model_registry_request_event import (
    EnumIntrospectionReason,
    ModelRegistryRequestEvent,
)

# Import envelope model
from .models.model_onex_envelope_v1 import ModelOnexEnvelopeV1

# Standard logger for system-level events
logger = logging.getLogger(__name__)


# Type definitions for better type safety and metrics tracking
class RegistrationMetrics(TypedDict):
    """Metrics for node registration operations."""

    total_registrations: int
    successful_registrations: int
    failed_registrations: int
    consul_registrations: int
    postgres_registrations: int
    current_nodes_count: int
    registered_nodes_count: int  # Alias for current_nodes_count for test compatibility
    last_cleanup_time: Optional[float]
    cleanup_operations_count: int
    memory_usage_mb: float


class MemoryGrowthMetrics(TypedDict):
    """Metrics for tracking memory growth and cleanup effectiveness."""

    nodes_before_cleanup: int
    nodes_after_cleanup: int
    nodes_removed: int
    cleanup_time_ms: float
    memory_freed_mb: float
    total_memory_usage_mb: float
    cleanup_operations_total: int
    last_cleanup_timestamp: float
    average_node_size_bytes: float
    offset_cleanup_operations: int  # Track offset cleanup operations
    total_offsets_cleaned: int  # Track total offsets cleaned


class KafkaMessageOffsets(TypedDict):
    """Kafka message offset information."""

    partition: int
    offset: int
    topic: str


class NodeBridgeRegistry(NodeEffect, HealthCheckMixin, IntrospectionMixin):
    """
    Production-ready Bridge Registry for node discovery and dual registration.

    Production Readiness Features:
    ===============================
    1. **Memory Leak Prevention**: TTL cache for offset tracking with automatic cleanup
    2. **Race Condition Prevention**: Cleanup locks for safe background task management
    3. **Circuit Breaker Pattern**: Robust error recovery with exponential backoff
    4. **Atomic Registration**: Transactional registration preventing partial state
    5. **Security-Aware Logging**: Sensitive data masking and production-safe logs
    6. **Configurable Values**: Environment-based configuration for all magic numbers
    7. **Performance Monitoring**: Real-time memory usage and performance metrics
    8. **Self-Registration (MVP)**: Registry publishes own introspection and registers itself

    Security Best Practices:
    ========================
    - **Password Management**: ALWAYS use POSTGRES_PASSWORD environment variable for database passwords
    - **Secrets Manager Integration**: For production, integrate with secrets management service
    - **Container Config**: Only use container config for passwords in local development/testing
    - **Warning System**: Automatic warnings when passwords are configured insecurely
    - **No Logging**: Passwords are never logged (sanitize_log_data ensures this)

    ⚠️ **MVP Security Warning**: Self-registration currently lacks trust model (SPIFFE).
       Phase 1b will add cryptographic verification. See docs/planning/POST_MVP_PRODUCTION_ENHANCEMENTS.md

    Responsibilities:
    1. Listen for NODE_INTROSPECTION events from Kafka
    2. Deserialize OnexEnvelopeV1 and extract introspection data
    3. Perform dual registration:
       - Consul: Register node for service discovery
       - PostgreSQL: Store tool registration for orchestration
    4. On startup: Register self, then request all nodes to re-broadcast introspection
    5. Monitor health of registration services

    Kafka Topics:
    - Consumes: node-introspection.v1
    - Produces: registry-request-introspection.v1

    Dependencies:
    - KafkaClient: Event streaming
    - RegistryConsulClient: Service discovery registration
    - NodeRegistrationRepository: Database persistence
    - PostgresClient: Database connection

    Thread Safety:
    ==============
    This class is designed to be thread-safe with:
    - Async locks for critical sections
    - Cleanup locks to prevent race conditions
    - Thread-safe TTL cache for offset tracking
    - Circuit breaker with internal locking

    Usage Pattern:
        registry = NodeBridgeRegistry(container, environment="production")
        await registry.on_startup()  # Starts background tasks
        # ... use registry ...
        await registry.on_shutdown() # REQUIRED: Cleans up background tasks
    """

    # Security and validation constants
    MAX_REGISTRY_ID_LENGTH: ClassVar[int] = 128
    MAX_JSONB_SIZE_BYTES: ClassVar[int] = (
        1_048_576  # 1 MB limit for JSONB fields (DoS protection)
    )
    ALLOWED_REGISTRY_ID_CHARS: ClassVar[set[str]] = set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"  # pragma: allowlist secret
    )

    def __init__(
        self, container: ModelONEXContainer, environment: str = "development"
    ) -> None:
        """
        Initialize Bridge Registry with dependency injection container.

        Args:
            container: ONEX container for dependency injection
            environment: Environment name (development, test, staging, production)

        Raises:
            OnexError: If container is invalid or initialization fails
        """
        super().__init__(container)

        # Environment and configuration
        self.environment = environment.lower()

        # Load configuration from ConfigLoader (YAML + env overrides)
        # This replaces hardcoded localhost defaults with proper config cascade
        try:
            loaded_config = get_registry_config(self.environment)
            self.config = get_registry_config_legacy(
                environment
            )  # Legacy registry-specific config
        except ImportError as e:
            # ConfigLoader not available
            logger.warning(f"ConfigLoader not available: {e}")
            self.config = get_registry_config_legacy(environment)
            loaded_config = None
        except FileNotFoundError as e:
            # Configuration file not found
            logger.warning(f"Configuration file not found: {e}")
            self.config = get_registry_config_legacy(environment)
            loaded_config = None
        except OSError as e:
            # File system errors
            logger.warning(f"File system error loading config: {e}")
            self.config = get_registry_config_legacy(environment)
            loaded_config = None
        except (ValueError, KeyError) as e:
            # Validation/parsing errors
            logger.warning(f"Configuration validation failed: {e}")
            self.config = get_registry_config_legacy(environment)
            loaded_config = None
        except Exception as e:
            # Unexpected - log as ERROR for visibility
            logger.error(
                f"Unexpected config loading error: {type(e).__name__}", exc_info=True
            )
            self.config = get_registry_config_legacy(environment)
            loaded_config = None

        # Extract container config/value (for runtime overrides)
        container_config = container.value if hasattr(container, "value") else {}

        # Registry identity (must be initialized before secure_logger)
        raw_registry_id = container_config.get(
            "registry_id", f"registry-{uuid4().hex[:8]}"
        )
        # Validate and sanitize registry_id to prevent injection attacks
        self.registry_id: str = self._validate_registry_id(raw_registry_id)

        # Initialize secure logger with registry_id
        self.secure_logger = get_secure_logger(
            f"{__name__}.{self.registry_id}", environment
        )

        # Registry-specific configuration from ConfigLoader cascade (Vault → Env → YAML)
        # Container config still takes precedence for runtime overrides
        if loaded_config:
            # Use ConfigLoader with proper cascade (Vault → Env → YAML)
            self.kafka_broker_url: str = container_config.get(
                "kafka_broker_url", loaded_config.kafka.bootstrap_servers
            )
            self.consul_host: str = container_config.get(
                "consul_host", loaded_config.consul.host
            )
            self.consul_port: int = int(
                container_config.get("consul_port", loaded_config.consul.port)
            )
            self.postgres_host: str = container_config.get(
                "postgres_host", loaded_config.database.host
            )
            self.postgres_port: int = int(
                container_config.get("postgres_port", loaded_config.database.port)
            )
            self.postgres_db: str = container_config.get(
                "postgres_db", loaded_config.database.database
            )
            self.postgres_user: str = container_config.get(
                "postgres_user", loaded_config.database.user
            )
        else:
            # Fallback to environment variables (backward compatibility)
            self.kafka_broker_url = container_config.get(
                "kafka_broker_url", os.getenv("KAFKA_BROKER_URL", "localhost:29092")
            )
            self.consul_host = container_config.get(
                "consul_host", os.getenv("CONSUL_HOST", "localhost")
            )
            self.consul_port = int(
                container_config.get("consul_port", os.getenv("CONSUL_PORT", "8500"))
            )
            self.postgres_host = container_config.get(
                "postgres_host", os.getenv("POSTGRES_HOST", "localhost")
            )
            self.postgres_port = int(
                container_config.get(
                    "postgres_port", os.getenv("POSTGRES_PORT", "5432")
                )
            )
            self.postgres_db = container_config.get(
                "postgres_db", os.getenv("POSTGRES_DB", "omninode_bridge")
            )
            self.postgres_user = container_config.get(
                "postgres_user", os.getenv("POSTGRES_USER", "postgres")
            )
        # Security: Prefer environment variable over container config for passwords
        # IMPORTANT: Use POSTGRES_PASSWORD environment variable for production deployments
        # Container config should only be used for local development/testing
        env_password = os.getenv("POSTGRES_PASSWORD")
        config_password = container_config.get("postgres_password")

        if env_password:
            # Environment variable takes precedence (most secure)
            self.postgres_password: Optional[str] = env_password
        elif config_password:
            # Container config fallback (warn about security)
            self.postgres_password = config_password
            # Security warning: Password in container config is less secure
            logger.warning(
                "PostgreSQL password configured via container config instead of environment variable. "
                "For production deployments, set POSTGRES_PASSWORD environment variable instead. "
                "Container config passwords should only be used for local development/testing."
            )
        else:
            # No password configured
            self.postgres_password = None

        # Security validation: Ensure password is provided during initialization
        if self.postgres_password is None:
            raise ModelOnexError(
                code=EnumCoreErrorCode.CONFIGURATION_ERROR,
                message="PostgreSQL password is required but not configured. "
                "RECOMMENDED: Set POSTGRES_PASSWORD environment variable for secure password management. "
                "Alternative (development only): Provide 'postgres_password' in container configuration. "
                "For production deployments, consider using a secrets manager service.",
                context={
                    # Security: Do not expose database connection details in error context
                    "error_type": "security_missing_password",
                    "node_id": self.node_id,
                    "config_sources": "environment_variable_recommended_or_container_config_dev_only",
                    "security_note": "Use POSTGRES_PASSWORD environment variable or secrets manager",
                },
            )

        # Kafka environment prefix (configurable for dev/staging/prod)
        env_prefix: str = container_config.get(
            "kafka_environment", os.getenv("KAFKA_ENVIRONMENT", "dev")
        )

        # Kafka topics (using standardized ONEX event format)
        self.introspection_topic = (
            f"{env_prefix}.omninode_bridge.onex.evt.node-introspection.v1"
        )
        self.request_topic = (
            f"{env_prefix}.omninode_bridge.onex.evt.registry-request-introspection.v1"
        )

        # Consumer group for registry
        self.consumer_group = "bridge-registry-group"

        # Registration tracking with TTL-based cleanup (using config values)
        self.registered_nodes: dict[str, ModelNodeIntrospectionEvent] = {}
        self.node_last_seen: dict[str, datetime] = {}
        self.node_ttl_hours = self.config.node_ttl_hours
        self.cleanup_interval_hours = self.config.cleanup_interval_hours

        self.registration_metrics: RegistrationMetrics = {
            "total_registrations": 0,
            "successful_registrations": 0,
            "failed_registrations": 0,
            "consul_registrations": 0,
            "postgres_registrations": 0,
            "current_nodes_count": 0,
            "registered_nodes_count": 0,  # Alias for current_nodes_count for test compatibility
            "last_cleanup_time": None,
            "cleanup_operations_count": 0,
            "memory_usage_mb": 0.0,
        }

        # Memory growth tracking for TTL cleanup operations
        self.memory_metrics: MemoryGrowthMetrics = {
            "nodes_before_cleanup": 0,
            "nodes_after_cleanup": 0,
            "nodes_removed": 0,
            "cleanup_time_ms": 0.0,
            "memory_freed_mb": 0.0,
            "total_memory_usage_mb": 0.0,
            "cleanup_operations_total": 0,
            "last_cleanup_timestamp": 0.0,
            "average_node_size_bytes": 0.0,
            "offset_cleanup_operations": 0,
            "total_offsets_cleaned": 0,
        }

        # === PRODUCTION READINESS FIXES ===

        # 1. Memory Leak Prevention: Replace set with TTL cache
        self._offset_cache = create_ttl_cache(
            name=f"{self.registry_id}-offsets",
            environment=environment,
            max_size=self.config.max_tracked_offsets,
            ttl_seconds=self.config.offset_cache_ttl_seconds,
            cleanup_interval_seconds=self.config.offset_cleanup_interval_seconds,
        )

        # 2. Race Condition Prevention: Add cleanup locks and flags
        self._cleanup_lock = asyncio.Lock()
        self._shutdown_lock = asyncio.Lock()
        self._cleanup_in_progress = False  # Prevent concurrent cleanup operations

        # 3. Circuit Breaker for error recovery
        self._registration_circuit_breaker = create_circuit_breaker(
            name=f"{self.registry_id}-registration",
            environment=environment,
        )
        self._kafka_circuit_breaker = create_circuit_breaker(
            name=f"{self.registry_id}-kafka",
            environment=environment,
        )

        # Running state and background task management
        # ⚠️ These tasks require explicit cleanup via stop_consuming() or on_shutdown()
        self._running = False
        self._consumer_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._memory_monitor_task: Optional[asyncio.Task] = None

        # Message processing state for idempotency and offset management
        # Use dict with timestamps for true LRU cleanup (mapping offset_key -> timestamp)
        self._processed_message_offsets: dict[str, float] = (
            {}
        )  # Track processed messages to prevent duplicates with timestamps
        # Use configurable max tracked offsets instead of hardcoded value
        self._max_tracked_offsets = self.config.max_tracked_offsets

        # Get services from container (lazy initialization)
        self.kafka_client: Optional[KafkaClient] = None
        self.consul_client: Optional[RegistryConsulClient] = None
        self.postgres_client: Optional[PostgresClient] = None
        self.node_repository: Optional[NodeRegistrationRepository] = None

        # Store container for later async initialization
        self._container_for_init = container

        # Check if in health check mode
        health_check_mode = container_config.get("health_check_mode", False)
        self._health_check_mode = health_check_mode

        # Skip service initialization in __init__ - will be done in async startup()
        # This fixes Issue #2: Async Initialization Issue
        if health_check_mode:
            self.secure_logger.debug(
                "Health check mode enabled - services will not be initialized",
                registry_id=self.registry_id,
            )
        else:
            self.secure_logger.debug(
                "Service initialization deferred to async startup() method",
                registry_id=self.registry_id,
            )

        # Initialize health check system
        self.initialize_health_checks()

        # Initialize introspection system for self-registration (MVP Phase 1a)
        # Security Warning: This is MVP implementation without trust model (SPIFFE)
        # Phase 1b will add cryptographic verification
        self.initialize_introspection()

        # Log initialization success with security
        self.secure_logger.info(
            "NodeBridgeRegistry initialized successfully",
            registry_id=self.registry_id,
            node_id=self.node_id,
            environment=self.environment,
            kafka_broker=self.kafka_broker_url,
            consul=f"{self.consul_host}:{self.consul_port}",
            introspection_topic=self.introspection_topic,
            request_topic=self.request_topic,
            offset_cache_enabled=self.config.offset_tracking_enabled,
            circuit_breaker_enabled=self.config.circuit_breaker_enabled,
            atomic_registration_enabled=self.config.atomic_registration_enabled,
        )

        emit_log_event(
            LogLevel.INFO,
            "NodeBridgeRegistry initialized successfully",
            sanitize_log_data(
                {
                    "node_id": self.node_id,
                    "registry_id": self.registry_id,
                    "environment": self.environment,
                    "kafka_broker": self.kafka_broker_url,
                    "consul": f"{self.consul_host}:{self.consul_port}",
                    "introspection_topic": self.introspection_topic,
                    "request_topic": self.request_topic,
                },
                environment,
            ),
        )

    def _run_async_sync(self, coro):
        """
        Run an async coroutine synchronously.

        Args:
            coro: Async coroutine to run

        Returns:
            Result of the coroutine
        """
        try:
            import asyncio

            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop, we need to run in a separate thread
                import concurrent.futures

                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(coro)
                    finally:
                        new_loop.close()

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    return future.result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError as e:
            # Event loop already running or closed - use asyncio.run as fallback
            import asyncio

            return asyncio.run(coro)
        except Exception as e:
            # Unexpected event loop errors - log and retry with asyncio.run
            logger.error(
                f"Unexpected event loop error: {type(e).__name__}", exc_info=True
            )
            import asyncio

            return asyncio.run(coro)

    @staticmethod
    def _validate_registry_id(registry_id: str) -> str:
        """
        Validate and sanitize registry_id to prevent injection attacks.

        Args:
            registry_id: Registry ID to validate

        Returns:
            Validated and sanitized registry_id

        Raises:
            OnexError: If registry_id is invalid or contains malicious characters
        """
        if not registry_id:
            raise ModelOnexError(
                code=EnumCoreErrorCode.VALIDATION_ERROR,
                message="registry_id cannot be empty",
                context={"registry_id": registry_id},
            )

        if len(registry_id) > NodeBridgeRegistry.MAX_REGISTRY_ID_LENGTH:
            raise ModelOnexError(
                code=EnumCoreErrorCode.VALIDATION_ERROR,
                message=f"registry_id exceeds maximum length of {NodeBridgeRegistry.MAX_REGISTRY_ID_LENGTH}",
                context={
                    "registry_id_length": len(registry_id),
                    "max_length": NodeBridgeRegistry.MAX_REGISTRY_ID_LENGTH,
                },
            )

        # Check for invalid characters
        invalid_chars = set(registry_id) - NodeBridgeRegistry.ALLOWED_REGISTRY_ID_CHARS
        if invalid_chars:
            raise ModelOnexError(
                code=EnumCoreErrorCode.VALIDATION_ERROR,
                message="registry_id contains invalid characters",
                context={
                    "invalid_chars": "".join(sorted(invalid_chars)),
                    "allowed_chars": "alphanumeric, hyphen, underscore",
                },
            )

        return registry_id

    @staticmethod
    def _validate_jsonb_size(data: dict, field_name: str = "data") -> None:
        """
        Validate JSONB field size to prevent DoS attacks.

        Args:
            data: Dictionary to validate
            field_name: Name of field for error messages

        Raises:
            OnexError: If data exceeds maximum size
        """
        import json

        try:
            # Serialize to JSON to check size
            json_str = json.dumps(data)
            size_bytes = len(json_str.encode("utf-8"))

            if size_bytes > NodeBridgeRegistry.MAX_JSONB_SIZE_BYTES:
                raise ModelOnexError(
                    code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message=f"{field_name} exceeds maximum JSONB size",
                    context={
                        "field_name": field_name,
                        "size_bytes": size_bytes,
                        "max_size_bytes": NodeBridgeRegistry.MAX_JSONB_SIZE_BYTES,
                        "size_mb": round(size_bytes / (1024 * 1024), 2),
                        "max_size_mb": round(
                            NodeBridgeRegistry.MAX_JSONB_SIZE_BYTES / (1024 * 1024), 2
                        ),
                    },
                )
        except (TypeError, ValueError) as e:
            raise ModelOnexError(
                code=EnumCoreErrorCode.VALIDATION_ERROR,
                message=f"Failed to serialize {field_name} to JSON",
                context={"field_name": field_name, "error": str(e)},
            )

    async def _create_postgres_client_with_retry(self) -> Any:
        """
        Create PostgreSQL client with exponential backoff retry logic.

        This method implements retry logic with exponential backoff to handle
        temporary PostgreSQL unavailability during startup.

        Returns:
            Initialized PostgresClient instance

        Raises:
            OnexError: If all retry attempts fail
        """
        import asyncio

        max_retries = self.config.database_max_retries
        base_delay_seconds = self.config.database_retry_base_delay_seconds
        max_delay_seconds = self.config.database_retry_max_delay_seconds

        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                from ....services.postgres_client import PostgresClient

                # Password is validated in __init__, so this assert is safe
                assert (
                    self.postgres_password is not None
                ), "Password should be validated in __init__"

                client = PostgresClient(
                    host=self.postgres_host,
                    port=self.postgres_port,
                    database=self.postgres_db,
                    user=self.postgres_user,
                    password=self.postgres_password,
                    max_size=self.config.connection_pool_size,
                )

                # Test the connection
                await client.connect()

                self.secure_logger.info(
                    "PostgreSQL client created and connected successfully",
                    attempt=attempt + 1,
                    host=self.postgres_host,
                    port=self.postgres_port,
                    database=self.postgres_db,
                )

                return client

            except (ConnectionError, TimeoutError, asyncio.TimeoutError) as e:
                # PostgreSQL connection failed or timeout - retriable errors
                last_error = e

                if attempt < max_retries:
                    # Calculate exponential backoff delay
                    delay_seconds = min(
                        base_delay_seconds * (2**attempt), max_delay_seconds
                    )
            except Exception as e:
                # Unexpected database errors - log with exc_info for debugging
                logger.error(
                    f"Unexpected database connection error: {type(e).__name__}",
                    exc_info=True,
                )
                last_error = e

                if attempt < max_retries:
                    # Calculate exponential backoff delay
                    delay_seconds = min(
                        base_delay_seconds * (2**attempt), max_delay_seconds
                    )

                    self.secure_logger.warning(
                        "PostgreSQL connection attempt failed, retrying...",
                        attempt=attempt + 1,
                        max_attempts=max_retries + 1,
                        delay_seconds=delay_seconds,
                        error=str(e),
                        error_type=type(e).__name__,
                    )

                    await asyncio.sleep(delay_seconds)
                else:
                    self.secure_logger.error(
                        "PostgreSQL connection failed after all retry attempts",
                        total_attempts=max_retries + 1,
                        final_error=str(e),
                        error_type=type(e).__name__,
                    )

        # All retry attempts failed
        error_context = sanitize_log_data(
            {
                "postgres_host": self.postgres_host,
                "postgres_port": self.postgres_port,
                "postgres_db": self.postgres_db,
                "postgres_user": self.postgres_user,
                "total_attempts": max_retries + 1,
                "final_error": str(last_error),
                "error_type": type(last_error).__name__ if last_error else "unknown",
                "node_id": self.node_id,
            },
            self.environment,
        )

        raise ModelOnexError(
            error_code=EnumCoreErrorCode.DATABASE_CONNECTION_ERROR,
            message=f"Failed to connect to PostgreSQL after {max_retries + 1} attempts: {last_error}",
            context=error_context,
        )

    def _initialize_services(self, container: ModelONEXContainer) -> None:
        """
        Initialize required services from container or create new instances.

        DEPRECATED: This method is kept for backward compatibility but should not be used.
        Use _initialize_services_async() instead for proper async initialization.

        Args:
            container: ONEX container for dependency injection
        """
        logger.warning(
            "DEPRECATED: _initialize_services() is deprecated and will be removed in v2.0.0. "
            "Use _initialize_services_async() instead for proper async initialization."
        )
        # For backward compatibility, run async version synchronously
        self._run_async_sync(self._initialize_services_async(container))

    async def _initialize_services_async(self, container: ModelONEXContainer) -> None:
        """
        Initialize required services from container or create new instances asynchronously.

        This is the proper async initialization method that should be called from on_startup().
        Fixes Issue #2: Async Initialization Issue.

        Args:
            container: ONEX container for dependency injection
        """
        # Get or create KafkaClient
        self.kafka_client = container.get_service("kafka_client")
        if self.kafka_client is None:
            try:
                from ....services.kafka_client import KafkaClient

                self.kafka_client = KafkaClient(
                    bootstrap_servers=self.kafka_broker_url,
                    enable_dead_letter_queue=True,
                    max_retry_attempts=self.config.max_retry_attempts,
                    timeout_seconds=self.config.kafka_consumer_timeout_ms // 1000,
                )

                # Fix: Connect KafkaClient immediately after creation
                # This sets self.kafka_client._connected = True, allowing consume_messages() to work
                try:
                    assert (
                        self.kafka_client is not None
                    ), "KafkaClient should be initialized"
                    await self.kafka_client.connect()

                    # CRITICAL: Check if connection actually succeeded
                    # connect() uses graceful degradation and may not raise exceptions
                    if not self.kafka_client._connected:
                        raise ConnectionError(
                            "KafkaClient.connect() completed but connection status is False"
                        )

                    self.secure_logger.info(
                        "KafkaClient created and connected successfully",
                        bootstrap_servers=self.kafka_broker_url,
                        timeout_seconds=self.config.kafka_consumer_timeout_ms // 1000,
                    )
                except ConnectionError as e:
                    # Kafka connection failed - non-critical
                    self.secure_logger.warning(
                        "Kafka connection failed during initialization",
                        error=str(e),
                        error_type="ConnectionError",
                        bootstrap_servers=self.kafka_broker_url,
                    )
                except (TimeoutError, asyncio.TimeoutError) as e:
                    # Kafka timeout - non-critical
                    self.secure_logger.warning(
                        "Kafka connection timeout during initialization",
                        error=str(e),
                        error_type="TimeoutError",
                        bootstrap_servers=self.kafka_broker_url,
                    )
                except Exception as e:
                    # Unexpected Kafka errors - log with exc_info
                    self.secure_logger.error(
                        f"Unexpected Kafka initialization error: {type(e).__name__}",
                        error=str(e),
                        error_type=type(e).__name__,
                        bootstrap_servers=self.kafka_broker_url,
                    )
                    logger.error(
                        f"Unexpected Kafka error: {type(e).__name__}", exc_info=True
                    )

                    # Environment-aware error handling
                    if self.environment == "production":
                        # Fail fast in production
                        raise ModelOnexError(
                            code=EnumCoreErrorCode.STARTUP_ERROR,
                            message=f"Failed to connect KafkaClient in production environment: {e}",
                            context={
                                "bootstrap_servers": self.kafka_broker_url,
                                "error": str(e),
                                "error_type": type(e).__name__,
                            },
                        )
                    else:
                        # Warn and continue in dev/test
                        logger.warning(
                            f"Kafka connection failed in {self.environment} environment - continuing without Kafka"
                        )
                        self.kafka_client = None

                # Register only if connection succeeded
                if self.kafka_client:
                    container.register_service("kafka_client", self.kafka_client)
                    logger.info("KafkaClient registered with container")
            except ImportError:
                logger.warning(
                    "KafkaClient not available - registry will not consume events"
                )
                self.kafka_client = None
        else:
            # Fix: If kafka_client was provided from container, verify it's connected
            if self.kafka_client and not self.kafka_client._connected:
                self.secure_logger.warning(
                    "Existing KafkaClient is not connected - attempting connection",
                    bootstrap_servers=self.kafka_broker_url,
                )
                try:
                    await self.kafka_client.connect()

                    # CRITICAL: Check if connection actually succeeded
                    if not self.kafka_client._connected:
                        raise ConnectionError(
                            "KafkaClient.connect() completed but connection status is False"
                        )

                except ConnectionError as e:
                    # Kafka connection failed - non-critical for existing client
                    self.secure_logger.warning(
                        "Failed to connect existing KafkaClient (connection error)",
                        error=str(e),
                        error_type="ConnectionError",
                    )
                except (TimeoutError, asyncio.TimeoutError) as e:
                    # Kafka timeout - non-critical for existing client
                    self.secure_logger.warning(
                        "Failed to connect existing KafkaClient (timeout)",
                        error=str(e),
                        error_type="TimeoutError",
                    )
                except Exception as e:
                    # Unexpected errors - log with exc_info
                    self.secure_logger.error(
                        f"Unexpected error connecting existing KafkaClient: {type(e).__name__}",
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    logger.error(
                        f"Unexpected Kafka connection error: {type(e).__name__}",
                        exc_info=True,
                    )
                    if self.environment == "production":
                        raise ModelOnexError(
                            code=EnumCoreErrorCode.STARTUP_ERROR,
                            message=f"Failed to connect existing KafkaClient: {e}",
                            context={"error": str(e), "error_type": type(e).__name__},
                        )
                    else:
                        # In dev/test, set to None if connection failed
                        self.kafka_client = None

        # Get or create RegistryConsulClient
        self.consul_client = container.get_service("consul_client")
        if self.consul_client is None:
            try:
                from ....services.metadata_stamping.registry.consul_client import (
                    RegistryConsulClient,
                )

                self.consul_client = RegistryConsulClient(
                    consul_host=self.consul_host,
                    consul_port=self.consul_port,
                )
                container.register_service("consul_client", self.consul_client)
                logger.info("Created new RegistryConsulClient instance")
            except ImportError:
                logger.warning(
                    "RegistryConsulClient not available - service discovery disabled"
                )
                self.consul_client = None

        # Get or create PostgresClient and NodeRegistrationRepository
        self.postgres_client = container.get_service("postgres_client")
        if self.postgres_client is None:
            try:
                # Security: Ensure password is explicitly provided for database connections
                # This should never happen as __init__ validates password, but check anyway
                if self.postgres_password is None:
                    error_context = sanitize_log_data(
                        {
                            "postgres_host": self.postgres_host,
                            "postgres_port": self.postgres_port,
                            "postgres_db": self.postgres_db,
                            "postgres_user": self.postgres_user,
                            "error_type": "security_missing_password",
                            "node_id": self.node_id,
                            "registry_id": self.registry_id,
                            "security_note": "Use POSTGRES_PASSWORD environment variable",
                        },
                        self.environment,
                    )

                    self.secure_logger.error(
                        "PostgreSQL password is required but not configured. "
                        "Set POSTGRES_PASSWORD environment variable.",
                        **error_context,
                    )

                    raise ModelOnexError(
                        code=EnumCoreErrorCode.CONFIGURATION_ERROR,
                        message="PostgreSQL password is required but not configured. "
                        "RECOMMENDED: Set POSTGRES_PASSWORD environment variable for secure password management.",
                        context=error_context,
                    )

                # Now use proper async call instead of _run_async_sync
                self.postgres_client = await self._create_postgres_client_with_retry()
                container.register_service("postgres_client", self.postgres_client)
                logger.info(
                    "Created new PostgresClient instance with retry logic (async)"
                )

                # Create node repository
                from ....services.node_registration_repository import (
                    NodeRegistrationRepository,
                )

                self.node_repository = NodeRegistrationRepository(self.postgres_client)
                container.register_service("node_repository", self.node_repository)
                logger.info("Created new NodeRegistrationRepository instance")
            except ImportError:
                logger.warning(
                    "PostgresClient not available - database registration disabled"
                )
                self.postgres_client = None
                self.node_repository = None
        else:
            # Get node repository if postgres client exists
            self.node_repository = container.get_service("node_repository")
            if self.node_repository is None:
                try:
                    from ....services.node_registration_repository import (
                        NodeRegistrationRepository,
                    )

                    self.node_repository = NodeRegistrationRepository(
                        self.postgres_client
                    )
                    container.register_service("node_repository", self.node_repository)
                    logger.info("Created new NodeRegistrationRepository instance")
                except ImportError:
                    logger.warning(
                        "NodeRegistrationRepository not available - database registration disabled"
                    )
                    self.node_repository = None

    async def execute_effect(self, contract: ModelContractEffect) -> Any:
        """
        Execute registry effect operation.

        This is the main entry point for ONEX effect execution.
        For the registry, this primarily manages the lifecycle.

        Args:
            contract: Effect contract with operation configuration

        Returns:
            Effect execution result

        Raises:
            OnexError: If effect execution fails
        """
        operation = contract.input_data.get("operation", "register")

        if operation == "start":
            return await self.on_startup()
        elif operation == "stop":
            return await self.on_shutdown()
        elif operation == "health_check":
            return await self.health_check()
        elif operation == "register_node":
            introspection_data = contract.input_data.get("introspection")
            if introspection_data:
                return await self.dual_register(introspection_data)
            else:
                raise ModelOnexError(
                    code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message="Missing introspection data for node registration",
                    context={"operation": operation},
                )
        else:
            raise ModelOnexError(
                code=EnumCoreErrorCode.VALIDATION_ERROR,
                message=f"Unknown operation: {operation}",
                context={"operation": operation},
            )

    async def get_capabilities(self) -> dict[str, Any]:
        """
        Get registry-specific capabilities.

        Overrides IntrospectionMixin.get_capabilities() to provide
        registry-specific information including registration metrics,
        service integration status, and registry operations.

        Returns:
            Dictionary with registry capabilities and status
        """
        # Call parent for base capabilities
        capabilities = await super().get_capabilities()

        # Override node_type to effect (registry is an effect node with specialized role)
        capabilities["node_type"] = "effect"
        capabilities["node_role"] = "registry"

        # Registry-specific operations
        capabilities["registry_operations"] = [
            "dual_register",
            "introspection_consume",
            "health_monitoring",
            "ttl_cleanup",
        ]

        # Registration metrics
        capabilities["registration_metrics"] = {
            "registered_nodes_count": self.registration_metrics.get(
                "current_nodes_count", 0
            ),
            "total_registrations": self.registration_metrics.get(
                "total_registrations", 0
            ),
            "successful_registrations": self.registration_metrics.get(
                "successful_registrations", 0
            ),
        }

        # Service integration status
        capabilities["service_integration"] = {
            "consul_enabled": self.consul_client is not None,
            "postgres_enabled": self.postgres_client is not None,
            "kafka_enabled": self.kafka_client is not None,
        }

        return capabilities

    async def on_startup(self) -> dict[str, Any]:
        """
        Start the registry service.

        This method:
        1. Initializes async services (KafkaClient, PostgresClient) - Fix for Issue #2
        2. Starts consuming NODE_INTROSPECTION events from Kafka
        3. Starts periodic cleanup task for expired nodes
        4. Starts memory monitoring task
        5. Requests all nodes to re-broadcast introspection

        Returns:
            Startup status dictionary

        Raises:
            OnexError: If startup fails
        """
        try:
            self.secure_logger.info("Starting NodeBridgeRegistry service")

            # Initialize health status (set to UNKNOWN during startup)
            self.health_status = HealthStatus.UNKNOWN
            self.last_health_check = datetime.now(UTC)

            # Initialize services asynchronously (Issue #2 fix)
            if not self._health_check_mode:
                await self._initialize_services_async(self._container_for_init)

            # Start background tasks
            await self.start_consuming()

            # Start memory monitoring
            if self.config.memory_monitoring_interval_seconds > 0:
                self._memory_monitor_task = asyncio.create_task(
                    self._memory_monitor_loop()
                )
                logger.debug("Started memory monitoring task")

            # Register self first (MVP Phase 1a)
            self_registration_result = await self._register_self()

            # Request all nodes to re-broadcast introspection
            await self._request_introspection_rebroadcast()

            self.health_status = HealthStatus.HEALTHY
            self.last_health_check = datetime.now(UTC)

            startup_info = {
                "status": "started",
                "registry_id": self.registry_id,
                "environment": self.environment,
                "introspection_topic": self.introspection_topic,
                "request_topic": self.request_topic,
                "consumer_group": self.consumer_group,
                "config_summary": self.config.get_summary(),
                "self_registration": self_registration_result,
            }

            self.secure_logger.info(
                "NodeBridgeRegistry started successfully", **startup_info
            )

            emit_log_event(
                LogLevel.INFO,
                "NodeBridgeRegistry started successfully",
                sanitize_log_data(startup_info, self.environment),
            )

            return startup_info

        except ModelOnexError:
            # Re-raise OnexError to preserve error context
            self.health_status = HealthStatus.UNHEALTHY
            self.last_health_check = datetime.now(UTC)
            raise
        except (ConnectionError, TimeoutError, asyncio.TimeoutError) as e:
            # Connection/timeout errors during startup - retriable
            self.health_status = HealthStatus.UNHEALTHY
            self.last_health_check = datetime.now(UTC)

            error_context = sanitize_log_data(
                {
                    "status": "startup_failed",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                self.environment,
            )
            self.secure_logger.error(
                "NodeBridgeRegistry startup failed (connection/timeout)",
                **error_context,
            )

            raise ModelOnexError(
                code=EnumCoreErrorCode.STARTUP_ERROR,
                message=f"Registry startup failed: {type(e).__name__}: {e}",
                context=error_context,
            )
        except Exception as e:
            # Unexpected startup errors - log with exc_info for debugging
            self.health_status = HealthStatus.UNHEALTHY
            self.last_health_check = datetime.now(UTC)
            logger.error(f"Unexpected startup error: {type(e).__name__}", exc_info=True)

            error_context = sanitize_log_data(
                {
                    "registry_id": self.registry_id,
                    "environment": self.environment,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                self.environment,
            )

            self.secure_logger.error(
                "Failed to start NodeBridgeRegistry", **error_context
            )

            raise ModelOnexError(
                code=EnumCoreErrorCode.OPERATION_FAILED,
                message=f"Failed to start NodeBridgeRegistry: {e}",
                context=error_context,
            )

    async def on_shutdown(self) -> dict[str, Any]:
        """
        Shutdown the registry service gracefully.

        This method:
        1. Stops consuming events from Kafka
        2. Stops all background tasks (cleanup, memory monitoring)
        3. Closes all client connections
        4. Cleans up resources

        Returns:
            Shutdown status dictionary

        Raises:
            OnexError: If shutdown fails
        """
        async with self._shutdown_lock:
            try:
                self.secure_logger.info("Shutting down NodeBridgeRegistry service")

                # Update health status (set to UNKNOWN during shutdown)
                self.health_status = HealthStatus.UNKNOWN
                self.last_health_check = datetime.now(UTC)

                # Stop consuming and cleanup tasks
                await self.stop_consuming()

                # Deregister self from Consul and PostgreSQL
                try:
                    self.secure_logger.info(
                        "Deregistering registry from service discovery"
                    )

                    # Publish shutdown introspection
                    await self.publish_introspection(reason="shutdown")

                    # Remove from Consul
                    if self.consul_client:
                        try:
                            await self._rollback_consul_registration(str(self.node_id))
                        except ConnectionError as e:
                            # Consul connection failed - non-critical during shutdown
                            logger.warning(
                                f"Consul connection failed during deregistration: {e}"
                            )
                        except Exception as e:
                            # Unexpected errors - log but don't fail shutdown
                            logger.warning(
                                f"Unexpected Consul deregistration error: {type(e).__name__}: {e}"
                            )
                            logger.error(
                                f"Consul deregistration error: {type(e).__name__}",
                                exc_info=True,
                            )

                    # Remove from PostgreSQL
                    if self.node_repository:
                        try:
                            await self._rollback_postgres_registration(
                                str(self.node_id)
                            )
                        except ConnectionError as e:
                            # PostgreSQL connection failed - non-critical during shutdown
                            logger.warning(
                                f"PostgreSQL connection failed during deregistration: {e}"
                            )
                        except Exception as e:
                            # Unexpected errors - log but don't fail shutdown
                            logger.warning(
                                f"Unexpected PostgreSQL deregistration error: {type(e).__name__}: {e}"
                            )
                            logger.error(
                                f"PostgreSQL deregistration error: {type(e).__name__}",
                                exc_info=True,
                            )

                except (ConnectionError, TimeoutError, asyncio.TimeoutError) as e:
                    # Connection/timeout errors - log but don't fail shutdown
                    logger.warning(
                        f"Self-deregistration connection/timeout error: {type(e).__name__}: {e}"
                    )
                except Exception as e:
                    # Unexpected self-deregistration errors - log with exc_info
                    logger.error(
                        f"Unexpected self-deregistration error: {type(e).__name__}",
                        exc_info=True,
                    )
                    logger.error(f"Self-deregistration failed: {e}")

                # Stop memory monitoring task
                if self._memory_monitor_task and not self._memory_monitor_task.done():
                    self._memory_monitor_task.cancel()
                    try:
                        await self._memory_monitor_task
                    except asyncio.CancelledError:
                        pass
                    self._memory_monitor_task = None
                    logger.debug("Memory monitoring task cleaned up")

                # Stop TTL cache
                await self._offset_cache.stop()

                # Update final metrics
                nodes_count = len(self.registered_nodes)
                self.registration_metrics["current_nodes_count"] = nodes_count
                self.registration_metrics["registered_nodes_count"] = nodes_count
                self.registration_metrics["memory_usage_mb"] = (
                    self._calculate_memory_usage()
                )

                # Update health status (set to UNHEALTHY after shutdown)
                self.health_status = HealthStatus.UNHEALTHY
                self.last_health_check = datetime.now(UTC)

                shutdown_info = {
                    "status": "stopped",
                    "registry_id": self.registry_id,
                    "environment": self.environment,
                    "final_metrics": self.registration_metrics,
                    "offset_cache_metrics": self._offset_cache.get_metrics().__dict__,
                }

                self.secure_logger.info(
                    "NodeBridgeRegistry shutdown successfully", **shutdown_info
                )

                emit_log_event(
                    LogLevel.INFO,
                    "NodeBridgeRegistry shutdown successfully",
                    sanitize_log_data(shutdown_info, self.environment),
                )

                return shutdown_info

            except Exception as e:
                error_context = sanitize_log_data(
                    {
                        "registry_id": self.registry_id,
                        "environment": self.environment,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    self.environment,
                )

                self.secure_logger.error(
                    "Error during NodeBridgeRegistry shutdown", **error_context
                )

                raise ModelOnexError(
                    code=EnumCoreErrorCode.OPERATION_FAILED,
                    message=f"Error during NodeBridgeRegistry shutdown: {e}",
                    context=error_context,
                )

    async def _register_self(self) -> dict[str, Any]:
        """
        Register registry node itself (MVP Phase 1a).

        ⚠️ MVP Security Warning: This is self-registration without trust model (SPIFFE).
           Phase 1b will add cryptographic verification.

        Returns:
            Registration result with status
        """
        try:
            self.secure_logger.warning(
                "MVP self-registration: No cryptographic verification (SPIFFE in Phase 1b)",
                registry_id=self.registry_id,
                security_phase="mvp_1a",
            )

            # Publish introspection event
            await self.publish_introspection(reason="startup")

            # Create introspection data for self-registration
            capabilities = await self.get_capabilities()

            self_introspection = ModelNodeIntrospectionEvent(
                node_id=str(self.node_id),
                node_type="effect",
                node_role="registry",
                capabilities=capabilities,
                endpoints={
                    "health": f"http://{self.consul_host}:{self.consul_port}/health",
                    "metrics": f"http://{self.consul_host}:{self.consul_port}/metrics",
                },
                metadata={
                    "registry_id": self.registry_id,
                    "environment": self.environment,
                },
            )

            # Register self in dual registration system
            result = await self.dual_register(self_introspection)

            self.secure_logger.info(
                "Registry self-registration completed",
                status=result.get("status"),
                consul_registered=result.get("consul_registered", False),
                postgres_registered=result.get("postgres_registered", False),
            )

            return result

        except Exception as e:
            # Graceful degradation - log error but don't fail startup
            self.secure_logger.error(
                "Registry self-registration failed - continuing with degraded functionality",
                error=str(e),
                error_type=type(e).__name__,
            )

            return {
                "status": "failed",
                "error": str(e),
                "degraded_mode": True,
            }

    async def start_consuming(self) -> None:
        """
        Start consuming NODE_INTROSPECTION events from Kafka.

        Starts background task for continuous event consumption.
        This method manages the lifecycle of the consumer task.
        """
        if self._running:
            logger.warning("NodeBridgeRegistry is already running")
            return

        if not self.kafka_client:
            # Issue #6 fix: Add comprehensive context for debugging
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.CONFIGURATION_ERROR,
                message="KafkaClient is not available - cannot start consuming",
                context={
                    "registry_id": self.registry_id,
                    "kafka_broker_url": self.kafka_broker_url,
                    "consumer_group": self.consumer_group,
                    "error_type": "kafka_client_unavailable",
                },
            )

        # Fix: Verify client is connected before starting consumer
        # This prevents the "Kafka client not connected" error in consume_messages()
        if not self.kafka_client._connected:
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.CONFIGURATION_ERROR,
                message="KafkaClient is not connected - cannot start consuming",
                context={
                    "registry_id": self.registry_id,
                    "kafka_broker_url": self.kafka_broker_url,
                    "consumer_group": self.consumer_group,
                    "error_type": "kafka_client_not_connected",
                    "resolution": "Call 'await kafka_client.connect()' before starting consumer",
                    "connected_status": self.kafka_client._connected,
                },
            )

        try:
            self._running = True

            # Start consumer task (circuit breaker removed - it's a long-running background task)
            # Individual operations within the consumer have their own timeouts
            self._consumer_task = asyncio.create_task(
                self._consume_introspection_events()
            )

            # Start periodic cleanup task
            cleanup_interval_seconds = self.cleanup_interval_hours * 3600
            self._cleanup_task = asyncio.create_task(
                self._periodic_cleanup_loop(cleanup_interval_seconds)
            )

            logger.info("Started consuming introspection events")
            self.secure_logger.info(
                "Started consuming introspection events",
                consumer_task_id=id(self._consumer_task),
                cleanup_task_id=id(self._cleanup_task),
                cleanup_interval_hours=self.cleanup_interval_hours,
            )

            emit_log_event(
                LogLevel.INFO,
                "Started consuming introspection events",
                sanitize_log_data(
                    {
                        "node_id": self.node_id,
                        "registry_id": self.registry_id,
                        "topic": self.introspection_topic,
                        "consumer_group": self.consumer_group,
                    },
                    self.environment,
                ),
            )

        except Exception as e:
            self._running = False

            # Cleanup tasks to prevent task leaks
            if hasattr(self, "_consumer_task") and self._consumer_task is not None:
                if not self._consumer_task.done():
                    self._consumer_task.cancel()
                    try:
                        await self._consumer_task
                    except asyncio.CancelledError:
                        pass
                    except Exception as cleanup_error:
                        logger.debug(
                            f"Error cleaning up consumer task: {cleanup_error}"
                        )
                self._consumer_task = None

            if hasattr(self, "_cleanup_task") and self._cleanup_task is not None:
                if not self._cleanup_task.done():
                    self._cleanup_task.cancel()
                    try:
                        await self._cleanup_task
                    except asyncio.CancelledError:
                        pass
                    except Exception as cleanup_error:
                        logger.debug(f"Error cleaning up cleanup task: {cleanup_error}")
                self._cleanup_task = None

            # Issue #6 fix: Add comprehensive Kafka context for debugging
            error_context = sanitize_log_data(
                {
                    "registry_id": self.registry_id,
                    "kafka_broker_url": self.kafka_broker_url,
                    "consumer_group": self.consumer_group,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                self.environment,
            )

            self.secure_logger.error(
                "Failed to start consuming introspection events", **error_context
            )

            raise ModelOnexError(
                code=EnumCoreErrorCode.STARTUP_ERROR,
                message=f"Failed to start consuming introspection events: {e}",
                context=error_context,
            )

    async def stop_consuming(self) -> None:
        """
        Stop consuming NODE_INTROSPECTION events from Kafka.

        Gracefully cancels all background tasks created by this class.
        This method MUST be called to clean up background tasks and prevent resource leaks.

        Tasks cleaned up:
        - _consumer_task: Kafka message consumer task
        - _cleanup_task: Periodic TTL cleanup task

        Race Condition Prevention:
        Uses _cleanup_lock to ensure safe task cancellation.
        """
        if not self._running:
            return

        # Prevent race conditions with cleanup lock
        async with self._cleanup_lock:
            self._running = False

            # Clean up consumer task
            if self._consumer_task:
                self._consumer_task.cancel()
                try:
                    await self._consumer_task
                except asyncio.CancelledError:
                    pass  # Expected during cancellation
                except Exception as e:
                    logger.warning(f"Error in consumer task cleanup: {e}")
                finally:
                    self._consumer_task = None
                    logger.debug("Consumer task cleaned up")

            # Clean up periodic cleanup task
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass  # Expected during cancellation
                except Exception as e:
                    logger.warning(f"Error in cleanup task cleanup: {e}")
                finally:
                    self._cleanup_task = None
                    logger.debug("Cleanup task cleaned up")

            self.secure_logger.info("Stopped consuming introspection events")

            emit_log_event(
                LogLevel.INFO,
                "Stopped consuming introspection events",
                sanitize_log_data(
                    {"node_id": self.node_id, "registry_id": self.registry_id},
                    self.environment,
                ),
            )

    async def _consume_introspection_events(self) -> None:
        """
        Background task to consume NODE_INTROSPECTION events from Kafka.

        Continuously polls for messages, processes them, and commits offsets.
        Includes circuit breaker protection and comprehensive error handling.

        Error Recovery:
        - Uses circuit breaker for Kafka operations
        - Implements exponential backoff retry logic
        - Provides detailed error context for debugging
        """
        logger.info("Starting introspection event consumer")

        try:
            while self._running:
                try:
                    # Ensure kafka_client is available
                    assert self.kafka_client is not None, "KafkaClient not initialized"

                    # Poll for messages with timeout
                    messages = await asyncio.wait_for(
                        self.kafka_client.consume_messages(
                            topic=self.introspection_topic,
                            group_id=self.consumer_group,
                            max_messages=self.config.kafka_max_poll_records,
                            timeout_ms=self.config.kafka_consumer_timeout_ms,
                        ),
                        timeout=self.config.kafka_consumer_timeout_ms / 1000 + 5,
                    )

                    if not messages:
                        continue

                    logger.debug(f"Received {len(messages)} introspection events")
                    processed_messages = []
                    failed_messages = []

                    # Process messages with error handling
                    for message in messages:
                        try:
                            result = await self._process_introspection_message(message)
                            if result.get("success", False):
                                processed_messages.append(message)
                            else:
                                failed_messages.append(message)
                                logger.warning(
                                    f"Failed to process message: {result.get('error', 'Unknown error')}"
                                )
                        except Exception as e:
                            failed_messages.append(message)
                            logger.error(
                                f"Error processing introspection message: {e}",
                                exc_info=True,
                            )

                    # Atomic offset commit with circuit breaker protection
                    if processed_messages:
                        try:
                            # Circuit breaker wraps the async call
                            _ = await self._kafka_circuit_breaker.call(
                                self._commit_message_offsets_atomic,
                                processed_messages,
                                [True]
                                * len(processed_messages),  # All processed successfully
                            )
                            logger.info(
                                f"Committed offsets for {len(processed_messages)} successfully processed messages"
                            )
                        except Exception as e:
                            error_context = sanitize_log_data(
                                {
                                    "node_id": self.node_id,
                                    "error": str(e),
                                    "processed_count": len(processed_messages),
                                    "failed_count": len(failed_messages),
                                    "total_count": len(messages),
                                },
                                self.environment,
                            )

                            self.secure_logger.error(
                                "Failed to commit message offsets atomically",
                                **error_context,
                            )

                            emit_log_event(
                                LogLevel.ERROR,
                                "Critical: Failed to commit message offsets atomically",
                                error_context,
                            )

                            # Circuit breaker will handle retry logic
                            # If circuit is open, this will raise CircuitBreakerError
                            raise

                    # Log failed message count for monitoring
                    if failed_messages:
                        self.secure_logger.warning(
                            "Failed to process introspection messages",
                            failed_count=len(failed_messages),
                            processed_count=len(processed_messages),
                            total_count=len(messages),
                        )

                        emit_log_event(
                            LogLevel.WARNING,
                            f"Failed to process {len(failed_messages)} introspection messages",
                            sanitize_log_data(
                                {
                                    "node_id": self.node_id,
                                    "failed_count": len(failed_messages),
                                    "processed_count": len(processed_messages),
                                    "total_count": len(messages),
                                },
                                self.environment,
                            ),
                        )

                except TimeoutError:
                    # Normal timeout - continue polling
                    continue
                except asyncio.CancelledError:
                    logger.info("Introspection event consumer cancelled")
                    break
                except Exception as e:
                    # Log error and continue if possible
                    logger.error(
                        f"Error in introspection event consumer: {e}", exc_info=True
                    )

                    # Check if we should continue or break
                    if self._running:
                        # Back off before retrying
                        await asyncio.sleep(
                            min(30, self.config.retry_backoff_base_seconds)
                        )
                        continue
                    else:
                        break

        except asyncio.CancelledError:
            logger.info("Introspection event consumer cancelled")
        except Exception as e:
            logger.error(
                f"Fatal error in introspection event consumer: {e}", exc_info=True
            )

            # Update health status
            self.health_status = HealthStatus.UNHEALTHY
            self.last_health_check = datetime.now(UTC)

            error_context = sanitize_log_data(
                {
                    "registry_id": self.registry_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                self.environment,
            )

            self.secure_logger.error(
                "Fatal error in introspection event consumer", **error_context
            )

            raise
        finally:
            logger.info("Introspection event consumer stopped")

    async def _process_introspection_message(self, message: Any) -> dict[str, Any]:
        """
        Process a single introspection message.

        Args:
            message: Kafka message containing introspection event

        Returns:
            Processing result dictionary

        Raises:
            OnexError: If message processing fails
        """
        try:
            # Check for duplicate processing using TTL cache
            message_id = self._get_message_id(message)
            if self._offset_cache.get(message_id) is not None:
                logger.debug(f"Skipping duplicate message: {message_id}")
                return {
                    "success": True,
                    "is_duplicate": True,
                    "message": "Duplicate message",
                }

            # Deserialize envelope
            envelope = ModelOnexEnvelopeV1.from_bytes(message.value)

            # Extract introspection event
            introspection_event = ModelNodeIntrospectionEvent.model_validate(
                envelope.payload
            )

            # Process registration with circuit breaker protection
            registration_result = await self._registration_circuit_breaker.call(
                self.dual_register, introspection_event
            )

            # Mark message as processed in TTL cache
            self._offset_cache.put(
                message_id, True, ttl_seconds=self.config.offset_cache_ttl_seconds
            )

            logger.debug(
                f"Processed introspection event for node: {introspection_event.node_id}"
            )

            return {
                "success": True,
                "node_id": introspection_event.node_id,
                "registration_result": registration_result,
                "message": "Processing successful",
            }

        except Exception as e:
            logger.error(f"Error processing introspection message: {e}", exc_info=True)

            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "message": "Processing failed",
            }

    def _get_message_id(self, message: Any) -> str:
        """
        Generate unique message ID for deduplication.

        Args:
            message: Kafka message

        Returns:
            Unique message identifier
        """
        # Use topic, partition, and offset for unique identification
        return f"{message.topic}:{message.partition}:{message.offset}"

    async def _commit_message_offsets_atomic(
        self, messages: list[Any], processing_results: list[bool]
    ) -> None:
        """
        Commit message offsets atomically with transaction semantics.

        Args:
            messages: List of processed messages
            processing_results: List of processing success flags

        Raises:
            OnexError: If offset commit fails
        """
        if not messages:
            return

        try:
            # Filter successfully processed messages
            successful_messages = [
                msg
                for msg, success in zip(messages, processing_results, strict=False)
                if success
            ]

            if not successful_messages:
                logger.debug("No successful messages to commit")
                return

            # Ensure kafka_client is available
            assert self.kafka_client is not None, "KafkaClient not initialized"

            # Commit offsets for successful messages
            await self.kafka_client.commit_offsets(successful_messages)  # type: ignore[attr-defined]

            logger.debug(f"Committed offsets for {len(successful_messages)} messages")

        except Exception as e:
            logger.error(f"Failed to commit message offsets: {e}", exc_info=True)

            raise ModelOnexError(
                code=EnumCoreErrorCode.EXTERNAL_SERVICE_ERROR,
                message=f"Failed to commit message offsets: {e}",
                context={
                    "messages_count": len(messages),
                    "successful_count": (
                        len(successful_messages)
                        if "successful_messages" in locals()
                        else 0
                    ),
                    "error": str(e),
                },
            )

    async def dual_register(
        self, introspection: ModelNodeIntrospectionEvent
    ) -> dict[str, Any]:
        """
        Perform dual registration: Consul + PostgreSQL with atomic transaction handling.

        PRODUCTION FIX: Implements proper transaction handling to prevent partial registration.
        Either both Consul and PostgreSQL registration succeed, or both fail with compensation.

        Args:
            introspection: Node introspection event data

        Returns:
            Registration result dictionary

        Raises:
            OnexError: If registration fails
        """
        start_time = time.time()
        node_id = introspection.node_id

        self.registration_metrics["total_registrations"] += 1

        # Initialize result tracking
        consul_success = False
        postgres_success = False
        consul_result = None
        postgres_result = None

        try:
            if self.config.atomic_registration_enabled:
                # Atomic registration: both must succeed or both fail
                consul_result, postgres_result = await self._atomic_dual_register(
                    introspection
                )
                consul_success = consul_result.get("success", False)
                postgres_success = postgres_result.get("success", False)
            else:
                # Legacy non-atomic registration (development mode)
                consul_result = await self._register_with_consul(introspection)
                postgres_result = await self._register_with_postgres(introspection)
                consul_success = consul_result.get("success", False)
                postgres_success = postgres_result.get("success", False)

            # Update metrics
            if consul_success:
                self.registration_metrics["consul_registrations"] += 1
            if postgres_success:
                self.registration_metrics["postgres_registrations"] += 1

            # Track registration with TTL
            if consul_success or postgres_success:
                self.registered_nodes[node_id] = introspection
                self.node_last_seen[node_id] = datetime.now(UTC)
                self.registration_metrics["successful_registrations"] += 1
            else:
                self.registration_metrics["failed_registrations"] += 1

            # Update current nodes count and memory usage
            nodes_count = len(self.registered_nodes)
            self.registration_metrics["current_nodes_count"] = nodes_count
            self.registration_metrics["registered_nodes_count"] = nodes_count
            self.registration_metrics["memory_usage_mb"] = (
                self._calculate_memory_usage()
            )

            registration_time_ms = (time.time() - start_time) * 1000

            result = {
                "status": (
                    "success"
                    if (consul_success and postgres_success)
                    else "partial" if (consul_success or postgres_success) else "error"
                ),
                "registered_node_id": node_id,
                "consul_registered": consul_success,
                "postgres_registered": postgres_success,
                "registration_time_ms": registration_time_ms,
                "atomic_mode": self.config.atomic_registration_enabled,
                "consul_result": consul_result,
                "postgres_result": postgres_result,
            }

            self.secure_logger.info(
                "Dual registration completed",
                registered_node_id=node_id,
                consul_success=consul_success,
                postgres_success=postgres_success,
                registration_time_ms=registration_time_ms,
                atomic_mode=self.config.atomic_registration_enabled,
                status=result["status"],
            )

            emit_log_event(
                LogLevel.INFO,
                "Dual registration completed",
                sanitize_log_data(
                    {
                        "node_id": self.node_id,
                        "registered_node_id": node_id,
                        "consul_success": consul_success,
                        "postgres_success": postgres_success,
                        "registration_time_ms": registration_time_ms,
                        "atomic_mode": self.config.atomic_registration_enabled,
                        "status": result["status"],
                    },
                    self.environment,
                ),
            )

            return result

        except Exception as e:
            self.registration_metrics["failed_registrations"] += 1
            registration_time_ms = (time.time() - start_time) * 1000

            error_context = sanitize_log_data(
                {
                    "node_id": self.node_id,
                    "registered_node_id": node_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "registration_time_ms": registration_time_ms,
                    "atomic_mode": self.config.atomic_registration_enabled,
                },
                self.environment,
            )

            self.secure_logger.error("Dual registration failed", **error_context)

            emit_log_event(
                LogLevel.ERROR,
                "Dual registration failed",
                error_context,
            )

            raise ModelOnexError(
                code=EnumCoreErrorCode.OPERATION_FAILED,
                message=f"Dual registration failed: {e}",
                context=error_context,
            )

    async def _atomic_dual_register(
        self, introspection: ModelNodeIntrospectionEvent
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Perform atomic dual registration with transaction semantics.

        Implements Saga pattern with compensation for rollbacks.

        Args:
            introspection: Node introspection event data

        Returns:
            Tuple of (consul_result, postgres_result)

        Raises:
            OnexError: If atomic registration fails
        """
        node_id = introspection.node_id
        consul_result = {"success": False, "node_id": node_id}
        postgres_result = {"success": False, "node_id": node_id}

        try:
            # Phase 1: Register with Consul
            if self.consul_client:
                consul_result = await self._register_with_consul(introspection)
                if not consul_result.get("success", False):
                    raise ModelOnexError(
                        code=EnumCoreErrorCode.CONSUL_ERROR,
                        message=f"Consul registration failed: {consul_result.get('error', 'Unknown error')}",
                        context={"node_id": node_id, "consul_result": consul_result},
                    )

            # Phase 2: Register with PostgreSQL
            if self.node_repository:
                postgres_result = await self._register_with_postgres(introspection)
                if not postgres_result.get("success", False):
                    # Rollback Consul registration
                    if self.consul_client and consul_result.get("success", False):
                        await self._rollback_consul_registration(node_id)
                        consul_result["success"] = False
                        consul_result["rollback"] = True

                    raise ModelOnexError(
                        code=EnumCoreErrorCode.POSTGRES_ERROR,
                        message=f"PostgreSQL registration failed: {postgres_result.get('error', 'Unknown error')}",
                        context={
                            "node_id": node_id,
                            "postgres_result": postgres_result,
                        },
                    )

            return consul_result, postgres_result

        except Exception as e:
            # Compensation: Rollback any successful registrations
            if consul_result.get("success", False) and self.consul_client:
                await self._rollback_consul_registration(node_id)
                consul_result["success"] = False
                consul_result["rollback"] = True

            if postgres_result.get("success", False) and self.node_repository:
                await self._rollback_postgres_registration(node_id)
                postgres_result["success"] = False
                postgres_result["rollback"] = True

            raise

    async def _rollback_consul_registration(self, node_id: str) -> None:
        """
        Rollback Consul registration.

        Args:
            node_id: Node ID to deregister
        """
        try:
            if self.consul_client:
                await self.consul_client.deregister_node(node_id)
                logger.info(f"Rolled back Consul registration for node: {node_id}")
        except Exception as e:
            logger.error(
                f"Failed to rollback Consul registration for node {node_id}: {e}"
            )

    async def _rollback_postgres_registration(self, node_id: str) -> None:
        """
        Rollback PostgreSQL registration.

        Args:
            node_id: Node ID to remove from database
        """
        try:
            if self.node_repository:
                await self.node_repository.delete_node_registration(node_id)
                logger.info(f"Rolled back PostgreSQL registration for node: {node_id}")
        except Exception as e:
            logger.error(
                f"Failed to rollback PostgreSQL registration for node {node_id}: {e}"
            )

    async def _register_with_consul(
        self, introspection: ModelNodeIntrospectionEvent
    ) -> dict[str, Any]:
        """
        Register node with Consul service discovery.

        Args:
            introspection: Node introspection event data

        Returns:
            Registration result dictionary
        """
        if not self.consul_client:
            return {
                "success": False,
                "error": "Consul client not available",
                "node_id": introspection.node_id,
            }

        try:
            # Extract port from endpoints if available
            service_port = None
            service_host = "localhost"  # Default host
            if introspection.endpoints:
                # Try to extract host and port from any endpoint URL
                for endpoint_url in introspection.endpoints.values():
                    if isinstance(endpoint_url, str) and ":" in endpoint_url:
                        try:
                            # Parse URL to extract host and port (e.g., "http://localhost:8053/health")
                            parts = endpoint_url.split(":")
                            if len(parts) >= 3:  # Has protocol and port
                                # Extract host
                                host_part = parts[1].lstrip("/")
                                if host_part:
                                    service_host = host_part
                                # Extract port
                                port_part = parts[2].split("/")[0]
                                service_port = int(port_part)
                                break
                        except (ValueError, IndexError):
                            continue

            # Create settings object for consul registration
            # Use a simple dict-based settings object
            from types import SimpleNamespace

            settings = SimpleNamespace(
                service_name=introspection.node_id,  # Use node_id as unique service identifier
                service_host=service_host,
                service_port=service_port or 8080,  # Default port
                service_tags=[introspection.node_type, "omninode_bridge"],
                service_meta={
                    "node_type": introspection.node_type,
                    "capabilities": ",".join(introspection.capabilities or []),
                },
            )

            # Actually call the consul client to register
            await self.consul_client.register_service(settings)

            return {
                "success": True,
                "node_id": introspection.node_id,
                "service_name": introspection.node_type,
                "service_port": service_port,
                "registration_time": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            logger.error(
                f"Consul registration failed for node {introspection.node_id}: {e}"
            )
            return {
                "success": False,
                "error": str(e),
                "node_id": introspection.node_id,
            }

    async def _register_with_postgres(
        self, introspection: ModelNodeIntrospectionEvent
    ) -> dict[str, Any]:
        """
        Register node with PostgreSQL database.

        Args:
            introspection: Node introspection event data

        Returns:
            Registration result dictionary
        """
        if not self.node_repository:
            return {
                "success": False,
                "error": "Node repository not available",
                "node_id": introspection.node_id,
            }

        try:
            # Create registration object for PostgreSQL
            from types import SimpleNamespace

            registration_create = SimpleNamespace(
                node_id=introspection.node_id,
                node_type=introspection.node_type,
                capabilities=introspection.capabilities or {},
                endpoints=introspection.endpoints or {},
                metadata=introspection.metadata or {},
                health_endpoint=(
                    introspection.endpoints.get("health")
                    if introspection.endpoints
                    else None
                ),
            )

            # Actually call the node repository to create registration
            # SimpleNamespace is used for duck typing compatibility
            await self.node_repository.create_registration(registration_create)  # type: ignore[arg-type]

            return {
                "success": True,
                "node_id": introspection.node_id,
                "node_type": introspection.node_type,
                "capabilities": introspection.capabilities,
                "endpoints": introspection.endpoints,
                "metadata": introspection.metadata,
                "registration_time": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            logger.error(
                f"PostgreSQL registration failed for node {introspection.node_id}: {e}"
            )
            return {
                "success": False,
                "error": str(e),
                "node_id": introspection.node_id,
            }

    async def _periodic_cleanup_loop(self, interval_seconds: float) -> None:
        """
        Background task for periodic cleanup of expired nodes.

        Args:
            interval_seconds: Cleanup interval in seconds
        """
        logger.info(f"Starting periodic cleanup loop with {interval_seconds}s interval")

        try:
            while self._running:
                try:
                    await asyncio.sleep(interval_seconds)
                    if self._running:
                        await self._cleanup_expired_nodes()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in periodic cleanup: {e}")
                    # Continue running despite cleanup errors

        except asyncio.CancelledError:
            logger.info("Periodic cleanup loop cancelled")
        except Exception as e:
            logger.error(f"Fatal error in periodic cleanup loop: {e}")
        finally:
            logger.info("Periodic cleanup loop stopped")

    async def _cleanup_expired_nodes(self) -> None:
        """
        Clean up expired nodes based on TTL.

        Race Condition Prevention:
        Uses _cleanup_lock to ensure safe cleanup operations.
        """
        async with self._cleanup_lock:
            try:
                start_time = time.time()
                current_time = datetime.now(UTC)
                ttl_threshold = current_time.timestamp() - (self.node_ttl_hours * 3600)

                # Find expired nodes
                expired_nodes = [
                    node_id
                    for node_id, last_seen in self.node_last_seen.items()
                    if last_seen.timestamp() < ttl_threshold
                ]

                if not expired_nodes:
                    return

                # Update metrics
                self.memory_metrics["nodes_before_cleanup"] = len(self.registered_nodes)
                memory_before = self._calculate_memory_usage()

                # Remove expired nodes with additional safety checks
                nodes_removed = 0
                for node_id in expired_nodes:
                    try:
                        if node_id in self.registered_nodes:
                            del self.registered_nodes[node_id]
                            nodes_removed += 1
                        if node_id in self.node_last_seen:
                            del self.node_last_seen[node_id]
                    except Exception as e:
                        logger.warning(f"Error cleaning up node {node_id}: {e}")
                        # Continue with other nodes even if one fails
                        continue

                # Verify cleanup was successful
                actual_nodes_removed = nodes_removed
                if actual_nodes_removed != len(expired_nodes):
                    logger.warning(
                        f"Cleanup mismatch: expected to remove {len(expired_nodes)} nodes, "
                        f"actually removed {actual_nodes_removed} nodes"
                    )

                # Update metrics
                self.memory_metrics["nodes_after_cleanup"] = len(self.registered_nodes)
                self.memory_metrics["nodes_removed"] = len(expired_nodes)
                self.memory_metrics["cleanup_time_ms"] = (
                    time.time() - start_time
                ) * 1000
                self.memory_metrics["total_memory_usage_mb"] = (
                    self._calculate_memory_usage()
                )
                self.memory_metrics["memory_freed_mb"] = (
                    memory_before - self.memory_metrics["total_memory_usage_mb"]
                )
                self.memory_metrics["cleanup_operations_total"] += 1
                self.memory_metrics["last_cleanup_timestamp"] = time.time()

                # Update registration metrics
                nodes_count = len(self.registered_nodes)
                self.registration_metrics["current_nodes_count"] = nodes_count
                self.registration_metrics["registered_nodes_count"] = nodes_count
                self.registration_metrics["memory_usage_mb"] = self.memory_metrics[
                    "total_memory_usage_mb"
                ]
                self.registration_metrics["cleanup_operations_count"] += 1
                self.registration_metrics["last_cleanup_time"] = time.time()

                self.secure_logger.info(
                    "Cleaned up expired nodes",
                    expired_nodes_count=len(expired_nodes),
                    current_nodes_count=len(self.registered_nodes),
                    cleanup_time_ms=self.memory_metrics["cleanup_time_ms"],
                    memory_freed_mb=self.memory_metrics["memory_freed_mb"],
                )

                emit_log_event(
                    LogLevel.INFO,
                    f"Cleaned up {len(expired_nodes)} expired nodes",
                    sanitize_log_data(
                        {
                            "node_id": self.node_id,
                            "expired_nodes_count": len(expired_nodes),
                            "current_nodes_count": len(self.registered_nodes),
                            "cleanup_time_ms": self.memory_metrics["cleanup_time_ms"],
                            "memory_freed_mb": self.memory_metrics["memory_freed_mb"],
                        },
                        self.environment,
                    ),
                )

            except Exception as e:
                logger.error(f"Error during expired nodes cleanup: {e}", exc_info=True)

                error_context = sanitize_log_data(
                    {
                        "registry_id": self.registry_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    self.environment,
                )

                self.secure_logger.error(
                    "Error during expired nodes cleanup", **error_context
                )

    async def _memory_monitor_loop(self) -> None:
        """
        Background task for memory monitoring and alerts.
        """
        logger.info("Starting memory monitoring loop")

        try:
            while self._running:
                try:
                    await asyncio.sleep(self.config.memory_monitoring_interval_seconds)
                    if self._running:
                        await self._check_memory_usage()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in memory monitoring: {e}")

        except asyncio.CancelledError:
            logger.info("Memory monitoring loop cancelled")
        finally:
            logger.info("Memory monitoring loop stopped")

    async def _check_memory_usage(self) -> None:
        """
        Check memory usage and log warnings if needed.
        """
        try:
            memory_usage_mb = self._calculate_memory_usage()
            offset_cache_metrics = self._offset_cache.get_metrics()

            # Check thresholds
            if memory_usage_mb >= self.config.memory_critical_threshold_mb:
                self.secure_logger.error(
                    "Memory usage CRITICAL",
                    memory_usage_mb=memory_usage_mb,
                    threshold_mb=self.config.memory_critical_threshold_mb,
                    registered_nodes_count=len(self.registered_nodes),
                    offset_cache_size=self._offset_cache.size(),
                    offset_cache_memory_mb=offset_cache_metrics.memory_usage_mb,
                )

                # Trigger immediate cleanup
                await self._cleanup_expired_nodes()
                self._offset_cache.cleanup_expired()

            elif memory_usage_mb >= self.config.memory_warning_threshold_mb:
                self.secure_logger.warning(
                    "Memory usage WARNING",
                    memory_usage_mb=memory_usage_mb,
                    threshold_mb=self.config.memory_warning_threshold_mb,
                    registered_nodes_count=len(self.registered_nodes),
                    offset_cache_size=self._offset_cache.size(),
                    offset_cache_memory_mb=offset_cache_metrics.memory_usage_mb,
                )

        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")

    def _calculate_memory_usage(self) -> float:
        """
        Calculate estimated memory usage in MB.

        Returns:
            Estimated memory usage in megabytes
        """
        try:

            # Rough estimation based on data structures
            memory_usage_bytes = 0

            # Memory for registered nodes (accurate measurement)
            for node_id, introspection in self.registered_nodes.items():
                memory_usage_bytes += sys.getsizeof(node_id)
                memory_usage_bytes += sys.getsizeof(introspection)

            # Memory for node_last_seen timestamps
            memory_usage_bytes += sum(
                sys.getsizeof(node_id) + sys.getsizeof(timestamp)
                for node_id, timestamp in self.node_last_seen.items()
            )

            # Offset cache memory
            offset_cache_metrics = self._offset_cache.get_metrics()
            memory_usage_bytes += offset_cache_metrics.memory_usage_bytes

            # Convert to MB
            memory_usage_mb = memory_usage_bytes / (1024 * 1024)
            return round(memory_usage_mb, 2)

        except (AttributeError, TypeError, ArithmeticError):
            # Memory calculation failed due to missing attributes, type errors, or arithmetic issues
            return 0.0

    async def _request_introspection_rebroadcast(self) -> None:
        """
        Request all nodes to re-broadcast their introspection events.
        """
        try:
            # Create registry request event
            request_event = ModelRegistryRequestEvent(
                registry_id=self.registry_id,
                reason=EnumIntrospectionReason.STARTUP_REBROADCAST,
                request_timestamp=datetime.now(UTC),
            )

            # Generate correlation ID for event tracking
            correlation_id = uuid4()

            # Publish to Kafka with OnexEnvelopeV1 wrapping (automatic)
            if self.kafka_client:
                success = await self.kafka_client.publish_with_envelope(
                    event_type="registry-request-introspection",
                    source_node_id=str(self.registry_id),
                    payload=request_event.to_dict(),
                    topic=self.request_topic,
                    correlation_id=correlation_id,
                    metadata={
                        "event_category": "registry_management",
                        "node_type": "effect",
                        "registry_id": str(self.registry_id),
                    },
                )

                if success:
                    self.secure_logger.info(
                        "Requested introspection rebroadcast (OnexEnvelopeV1)",
                        request_topic=self.request_topic,
                        correlation_id=str(correlation_id),
                        envelope_wrapped=True,
                    )

                    emit_log_event(
                        LogLevel.INFO,
                        "Requested introspection rebroadcast from all nodes",
                        sanitize_log_data(
                            {
                                "node_id": self.node_id,
                                "registry_id": self.registry_id,
                                "request_topic": self.request_topic,
                                "correlation_id": str(correlation_id),
                                "envelope_wrapped": True,
                            },
                            self.environment,
                        ),
                    )
                else:
                    self.secure_logger.warning(
                        "Failed to publish introspection rebroadcast request",
                        request_topic=self.request_topic,
                        correlation_id=str(correlation_id),
                    )

        except Exception as e:
            logger.error(f"Failed to request introspection rebroadcast: {e}")

            error_context = sanitize_log_data(
                {
                    "registry_id": self.registry_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                self.environment,
            )

            self.secure_logger.error(
                "Failed to request introspection rebroadcast", **error_context
            )

    async def health_check(self) -> dict[str, Any]:
        """
        Perform comprehensive health check.

        Returns:
            Health check result dictionary
        """
        try:
            health_status: dict[str, Any] = {
                "status": "healthy",
                "registry_id": self.registry_id,
                "environment": self.environment,
                "timestamp": datetime.now(UTC).isoformat(),
                "checks": {},
                "metrics": {},
            }

            # Check background tasks
            health_status["checks"]["background_tasks"] = {
                "consumer_task_running": self._consumer_task is not None
                and not self._consumer_task.done(),
                "cleanup_task_running": self._cleanup_task is not None
                and not self._cleanup_task.done(),
                "memory_monitor_running": self._memory_monitor_task is not None
                and not self._memory_monitor_task.done(),
            }

            # Check service connectivity
            health_status["checks"]["services"] = {
                "kafka_client_available": self.kafka_client is not None,
                "consul_client_available": self.consul_client is not None,
                "postgres_client_available": self.postgres_client is not None,
                "node_repository_available": self.node_repository is not None,
            }

            # Check circuit breakers
            health_status["checks"]["circuit_breakers"] = {
                "registration_circuit": self._registration_circuit_breaker.state.value,
                "kafka_circuit": self._kafka_circuit_breaker.state.value,
            }

            # Include metrics
            health_status["metrics"]["registration"] = self.registration_metrics
            health_status["metrics"]["memory"] = self.memory_metrics
            health_status["metrics"]["offset_cache"] = self._offset_cache.get_status()

            # Include configuration summary
            health_status["config"] = self.config.get_summary()

            # Determine overall status
            if (
                not health_status["checks"]["background_tasks"]["consumer_task_running"]
                or self._registration_circuit_breaker.state.value == "open"
                or self._kafka_circuit_breaker.state.value == "open"
            ):
                health_status["status"] = "unhealthy"

            return health_status

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "registry_id": self.registry_id,
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    def get_registration_metrics(self) -> dict[str, Any]:
        """
        Get registration-specific metrics.

        Returns:
            Registration metrics dictionary
        """
        # TypedDict is compatible with dict[str, Any]
        return dict(self.registration_metrics)

    def get_metrics(self) -> dict[str, Any]:
        """
        Get comprehensive metrics for monitoring.

        Returns:
            Metrics dictionary
        """
        return {
            "registry_id": self.registry_id,
            "environment": self.environment,
            "timestamp": datetime.now(UTC).isoformat(),
            "registration_metrics": self.registration_metrics,
            "memory_metrics": self.memory_metrics,
            "offset_cache_metrics": self._offset_cache.get_metrics().__dict__,
            "circuit_breaker_metrics": {
                "registration": self._registration_circuit_breaker.get_status(),
                "kafka": self._kafka_circuit_breaker.get_status(),
            },
            "background_tasks": {
                "consumer_task_running": self._consumer_task is not None
                and not self._consumer_task.done(),
                "cleanup_task_running": self._cleanup_task is not None
                and not self._cleanup_task.done(),
                "memory_monitor_running": self._memory_monitor_task is not None
                and not self._memory_monitor_task.done(),
            },
            "configuration": self.config.get_summary(),
        }

    async def is_offset_processed(self, offset_key: str) -> bool:
        """
        Check if a message offset has been processed with race condition prevention.

        This method provides thread-safe checking of processed offsets using
        both the TTL cache (primary) and the legacy set (fallback).

        Args:
            offset_key: Unique identifier for the message offset

        Returns:
            True if the offset has been processed, False otherwise
        """
        # First check TTL cache (thread-safe by design)
        cached_result = self._offset_cache.get(offset_key)
        if cached_result is not None:
            return bool(cached_result)

        # Fallback to legacy dict with lock protection
        async with self._cleanup_lock:
            return offset_key in self._processed_message_offsets

    async def _add_processed_offset(self, offset_key: str) -> None:
        """
        Add a processed message offset to tracking with race condition prevention.

        This method ensures atomic operations when adding offsets to both the
        TTL cache and the legacy set for backward compatibility.

        Args:
            offset_key: Unique identifier for the processed message offset
        """
        # Add to TTL cache (thread-safe by design)
        self._offset_cache.put(
            offset_key, True, ttl_seconds=self.config.offset_cache_ttl_seconds
        )

        # Also add to legacy dict with timestamp for true LRU cleanup
        async with self._cleanup_lock:
            self._processed_message_offsets[offset_key] = time.time()

            # Trigger cleanup if we're approaching the limit (with race condition prevention)
            if len(self._processed_message_offsets) > self._max_tracked_offsets * 0.9:
                # Only schedule cleanup if one is not already in progress
                if not self._cleanup_in_progress:
                    # Set flag immediately while holding lock to prevent race condition
                    # The cleanup task will reset this flag when it completes
                    self._cleanup_in_progress = True
                    logger.debug(
                        f"Approaching offset limit ({len(self._processed_message_offsets)}/{self._max_tracked_offsets}), "
                        "scheduling cleanup"
                    )
                    # Schedule cleanup asynchronously to avoid blocking
                    asyncio.create_task(self._cleanup_processed_offsets())
                else:
                    logger.debug(
                        f"Cleanup already in progress, skipping duplicate cleanup request "
                        f"({len(self._processed_message_offsets)}/{self._max_tracked_offsets})"
                    )

    async def _cleanup_processed_offsets(self) -> None:
        """
        Clean up processed message offsets to prevent memory leaks.

        When the offset tracking set grows too large, remove older entries
        to maintain memory efficiency while preserving duplicate protection.

        Race Condition Prevention:
        - Uses _cleanup_lock to ensure atomic operations on the offsets set
        - Uses _cleanup_in_progress flag to prevent concurrent cleanup tasks
        - Always resets flag in finally block for robust cleanup
        """
        if not hasattr(self, "_processed_message_offsets"):
            return

        # Use cleanup lock to prevent race conditions during offset operations
        async with self._cleanup_lock:
            # Set flag to prevent concurrent cleanup operations
            if self._cleanup_in_progress:
                logger.debug("Cleanup already in progress, skipping duplicate cleanup")
                return

            try:
                self._cleanup_in_progress = True
                start_time = time.time()
                initial_size = len(self._processed_message_offsets)

                if initial_size <= self._max_tracked_offsets:
                    logger.debug("No offset cleanup needed - within limits")
                    return

                # Calculate how many offsets to remove (remove 20% when over limit)
                target_size = int(self._max_tracked_offsets * 0.8)
                offsets_to_remove = initial_size - target_size

                if offsets_to_remove > 0:
                    # Sort by timestamp to find the oldest entries (true LRU behavior)
                    offset_items = sorted(
                        self._processed_message_offsets.items(),
                        key=lambda x: x[1],  # Sort by timestamp (value)
                    )

                    # Remove the oldest offsets
                    offsets_to_remove_list = [
                        offset_key for offset_key, _ in offset_items[:offsets_to_remove]
                    ]

                    for offset_key in offsets_to_remove_list:
                        del self._processed_message_offsets[offset_key]

                    cleanup_time_ms = (time.time() - start_time) * 1000
                    final_size = len(self._processed_message_offsets)

                    logger.info(
                        f"Cleaned up {len(offsets_to_remove_list)} processed message offsets "
                        f"in {cleanup_time_ms:.2f}ms. "
                        f"Size: {initial_size} → {final_size} (target: {target_size})"
                    )

                    # Track cleanup in memory metrics
                    self.memory_metrics["offset_cleanup_operations"] = (
                        self.memory_metrics.get("offset_cleanup_operations", 0) + 1
                    )
                    self.memory_metrics["total_offsets_cleaned"] = (
                        self.memory_metrics.get("total_offsets_cleaned", 0)
                        + len(offsets_to_remove_list)
                    )
                else:
                    logger.debug("No offsets to remove despite exceeding limit")

            finally:
                # Always reset flag to allow future cleanups
                self._cleanup_in_progress = False


# Factory function for backward compatibility
def create_node_bridge_registry(
    container: ModelONEXContainer, environment: str = "development"
) -> NodeBridgeRegistry:
    """
    Create NodeBridgeRegistry instance with environment configuration.

    Args:
        container: ONEX container for dependency injection
        environment: Environment name

    Returns:
        Configured NodeBridgeRegistry instance
    """
    return NodeBridgeRegistry(container, environment=environment)
