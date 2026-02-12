# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration domain plugin for kernel-level initialization.

This module provides the PluginRegistration class, which implements
ProtocolDomainPlugin for the Registration domain. It encapsulates all
Registration-specific initialization code that was previously embedded
in kernel.py.

The plugin handles:
    - PostgreSQL pool creation for registration projections
    - Projector discovery and loading from contracts
    - Schema initialization for registration projection table
    - Consul handler initialization (optional)
    - Handler wiring (HandlerNodeIntrospected, HandlerRuntimeTick, etc.)
    - Dispatcher creation and introspection event consumer startup

Design Pattern:
    The plugin pattern enables the kernel to remain generic while allowing
    domain-specific initialization to be encapsulated in domain modules.
    This follows the dependency inversion principle - the kernel depends
    on the abstract ProtocolDomainPlugin protocol, not this concrete class.

Configuration:
    The plugin activates based on environment variables:
    - OMNIBASE_INFRA_DB_URL: Required for plugin activation (PostgreSQL DSN)
    - CONSUL_HOST: Optional, enables Consul dual-registration
    - CONSUL_PORT: Optional (default: 8500)

Example Usage:
    ```python
    from omnibase_infra.nodes.node_registration_orchestrator.plugin import (
        PluginRegistration,
    )
    from omnibase_infra.runtime.protocol_domain_plugin import (
        ModelDomainPluginConfig,
        RegistryDomainPlugin,
    )

    # Register plugin
    registry = RegistryDomainPlugin()
    registry.register(PluginRegistration())

    # During kernel bootstrap
    config = ModelDomainPluginConfig(container=container, event_bus=event_bus, ...)
    plugin = registry.get("registration")

    if plugin and plugin.should_activate(config):
        await plugin.initialize(config)
        await plugin.wire_handlers(config)
        await plugin.start_consumers(config)
    ```

Related:
    - OMN-1346: Registration Code Extraction
    - OMN-888: Registration Orchestrator
    - OMN-892: 2-way Registration E2E Integration Test
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from omnibase_infra.runtime.models.model_postgres_pool_config import (
    ModelPostgresPoolConfig,
)

if TYPE_CHECKING:
    from uuid import UUID

    import asyncpg

    from omnibase_infra.handlers import HandlerConsul
    from omnibase_infra.projectors.snapshot_publisher_registration import (
        SnapshotPublisherRegistration,
    )
    from omnibase_infra.runtime.event_bus_subcontract_wiring import (
        EventBusSubcontractWiring,
    )
    from omnibase_infra.runtime.projector_shell import ProjectorShell
    from omnibase_infra.runtime.service_intent_executor import IntentExecutor

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    ContainerWiringError,
    DbOwnershipMismatchError,
    DbOwnershipMissingError,
)
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.runtime.protocol_domain_plugin import (
    ModelDomainPluginConfig,
    ModelDomainPluginResult,
    ProtocolDomainPlugin,
)
from omnibase_infra.runtime.util_db_ownership import validate_db_ownership
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

logger = logging.getLogger(__name__)

# =============================================================================
# Projector Discovery Configuration
# =============================================================================


# Default path for projector contract files, calculated using importlib.resources
# for robustness across different deployment scenarios (standard installs, frozen
# executables, various packaging tools).
#
# This path can be overridden via ONEX_PROJECTOR_CONTRACTS_DIR environment variable.
#
# Package structure assumption:
#   omnibase_infra/
#     projectors/
#       contracts/
#         registration_projector.yaml
#
# The default resolves to: <package_root>/projectors/contracts
def _get_default_projector_contracts_dir() -> Path:
    """Calculate default projector contracts directory from package root.

    Uses importlib.resources for robust resource path resolution across different
    deployment scenarios (standard pip installs, frozen executables, editable
    installs, and various packaging tools).

    Note:
        Falls back to __file__-based resolution if importlib.resources path
        is not a concrete filesystem path (e.g., in zip imports).

    Returns:
        Path to the projectors/contracts directory within omnibase_infra package.
    """
    from importlib.resources import files

    # Use importlib.resources for robust path resolution
    resource_path = files("omnibase_infra").joinpath("projectors", "contracts")

    # Convert to Path - handles both Traversable and actual Path objects
    # Note: For zip imports, this may need special handling, but standard
    # installs and editable installs will work correctly
    try:
        # Try to get a concrete filesystem path
        return Path(str(resource_path))
    except (TypeError, ValueError):
        # Fallback for edge cases where path conversion fails
        import omnibase_infra

        package_root = Path(omnibase_infra.__file__).parent
        return package_root / "projectors" / "contracts"


PROJECTOR_CONTRACTS_DEFAULT_DIR = _get_default_projector_contracts_dir()


class PluginRegistration:
    """Registration domain plugin for kernel initialization.

    This plugin encapsulates all Registration-specific initialization that was
    previously in kernel.py. It implements ProtocolDomainPlugin to provide
    lifecycle hooks for the kernel bootstrap sequence.

    Resources Created:
        - PostgreSQL connection pool (asyncpg.Pool)
        - ProjectorShell for registration projections
        - HandlerConsul for dual-registration (optional)
        - Introspection event consumer

    Thread Safety:
        This class is NOT thread-safe. The kernel calls plugin methods
        sequentially during bootstrap. Resource access during runtime
        should be via container-resolved handlers.

    Attributes:
        _pool: PostgreSQL connection pool (created in initialize())
        _projector: ProjectorShell for projections (created in initialize())
        _consul_handler: HandlerConsul for dual-registration (optional)
        _wiring: EventBusSubcontractWiring for dispatch engine consumers
    """

    def __init__(self) -> None:
        """Initialize the plugin with empty state."""
        self._pool: asyncpg.Pool | None = None
        self._projector: ProjectorShell | None = None
        self._consul_handler: HandlerConsul | None = None
        self._snapshot_publisher: SnapshotPublisherRegistration | None = None
        self._wiring: EventBusSubcontractWiring | None = None
        self._shutdown_in_progress: bool = False
        self._handler_wiring_succeeded: bool = False

    @property
    def plugin_id(self) -> str:
        """Return unique identifier for this plugin."""
        return "registration"

    @property
    def display_name(self) -> str:
        """Return human-readable name for this plugin."""
        return "Registration"

    @property
    def postgres_pool(self) -> asyncpg.Pool | None:
        """Return the PostgreSQL pool (for external access)."""
        return self._pool

    @property
    def projector(self) -> ProjectorShell | None:
        """Return the projector (for external access)."""
        return self._projector

    @property
    def consul_handler(self) -> HandlerConsul | None:
        """Return the Consul handler (for external access)."""
        return self._consul_handler

    @property
    def snapshot_publisher(self) -> SnapshotPublisherRegistration | None:
        """Return the snapshot publisher (for external access)."""
        return self._snapshot_publisher

    def should_activate(self, config: ModelDomainPluginConfig) -> bool:
        """Check if Registration should activate based on environment.

        Returns True if OMNIBASE_INFRA_DB_URL is set, indicating PostgreSQL
        is configured for registration support.

        Args:
            config: Plugin configuration (not used for this check).

        Returns:
            True if OMNIBASE_INFRA_DB_URL environment variable is set.
        """
        db_url = (os.getenv("OMNIBASE_INFRA_DB_URL") or "").strip()
        if not db_url:
            logger.debug(
                "Registration plugin inactive: OMNIBASE_INFRA_DB_URL not set "
                "(correlation_id=%s)",
                config.correlation_id,
            )
            return False
        return True

    async def initialize(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """Initialize Registration resources.

        Creates:
        - PostgreSQL connection pool
        - ProjectorShell from contract discovery
        - Registration projection schema
        - HandlerConsul (if CONSUL_HOST is set)

        Args:
            config: Plugin configuration with container and correlation_id.

        Returns:
            Result with resources_created list on success.
        """
        import asyncpg

        start_time = time.time()
        resources_created: list[str] = []
        correlation_id = config.correlation_id

        try:
            # 1. Create PostgreSQL pool from OMNIBASE_INFRA_DB_URL
            postgres_dsn = (os.getenv("OMNIBASE_INFRA_DB_URL") or "").strip()

            # Shared error context for all DSN validation failures
            pool_error_context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="create_postgres_pool",
            )

            if not postgres_dsn:
                raise ContainerWiringError(
                    "OMNIBASE_INFRA_DB_URL is required but not set",
                    context=pool_error_context,
                )

            # Validate DSN scheme and database name before pool creation
            try:
                ModelPostgresPoolConfig.validate_dsn(postgres_dsn)
            except ValueError as exc:
                raise ContainerWiringError(
                    str(exc),
                    context=pool_error_context,
                ) from exc

            self._pool = await asyncpg.create_pool(
                postgres_dsn,
                min_size=2,
                max_size=10,
            )
            # Validate pool creation succeeded - asyncpg.create_pool() can return None
            # in edge cases (e.g., connection issues during pool warmup)
            if self._pool is None:
                context = ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.DATABASE,
                    operation="create_postgres_pool",
                )
                raise ContainerWiringError(
                    "PostgreSQL pool creation returned None - connection may have failed",
                    context=context,
                )
            resources_created.append("postgres_pool")
            logger.info(
                "PostgreSQL pool created (correlation_id=%s)",
                correlation_id,
                extra={"dsn_var": "OMNIBASE_INFRA_DB_URL"},
            )

            # 1.5 Validate DB ownership (OMN-2085)
            # Hard gate: prevents operating on a database owned by another
            # service after the DB-per-repo split.
            await validate_db_ownership(
                pool=self._pool,
                expected_owner="omnibase_infra",
                correlation_id=correlation_id,
            )

            # 2. Load projectors from contracts via ProjectorPluginLoader
            await self._load_projector(config)
            if self._projector is not None:
                resources_created.append("projector")

            # 3. Initialize schema
            await self._initialize_schema(config)
            resources_created.append("registration_schema")

            # 4. Initialize Consul handler (optional)
            await self._initialize_consul_handler(config)
            if self._consul_handler is not None:
                resources_created.append("consul_handler")

            # 5. Initialize SnapshotPublisher (optional, requires Kafka)
            await self._initialize_snapshot_publisher(config)
            if self._snapshot_publisher is not None:
                resources_created.append("snapshot_publisher")

            duration = time.time() - start_time
            # Use constructor directly for results with resources_created
            return ModelDomainPluginResult(
                plugin_id=self.plugin_id,
                success=True,
                message="Registration plugin initialized",
                resources_created=resources_created,
                duration_seconds=duration,
            )

        except (DbOwnershipMismatchError, DbOwnershipMissingError):
            # Hard gate: must propagate to kill the kernel (OMN-2085).
            # DB ownership errors indicate the service connected to a database
            # owned by a different service after the DB-per-repo split.
            # Swallowing these would defeat the entire safety mechanism.
            await self._cleanup_on_failure(config)
            raise

        except Exception as e:
            duration = time.time() - start_time
            logger.exception(
                "Failed to initialize Registration plugin (correlation_id=%s)",
                correlation_id,
                extra={"error_type": type(e).__name__},
            )
            # Clean up any resources created before failure
            await self._cleanup_on_failure(config)
            return ModelDomainPluginResult.failed(
                plugin_id=self.plugin_id,
                error_message=sanitize_error_message(e),
                duration_seconds=duration,
            )

    async def _load_projector(self, config: ModelDomainPluginConfig) -> None:
        """Load projector from contracts via ProjectorPluginLoader."""
        from omnibase_infra.runtime.models.model_projector_plugin_loader_config import (
            ModelProjectorPluginLoaderConfig,
        )
        from omnibase_infra.runtime.projector_plugin_loader import (
            ProjectorPluginLoader,
        )
        from omnibase_infra.runtime.projector_shell import ProjectorShell

        correlation_id = config.correlation_id

        # Configurable projector contracts directory (supports different deployment layouts)
        # Environment variable allows overriding the default path when package structure differs
        # Uses PROJECTOR_CONTRACTS_DEFAULT_DIR constant which is calculated from package root
        # for robustness against internal directory restructuring
        projector_contracts_dir = Path(
            os.getenv(
                "ONEX_PROJECTOR_CONTRACTS_DIR",
                str(PROJECTOR_CONTRACTS_DEFAULT_DIR),
            )
        )

        if not projector_contracts_dir.exists():
            logger.debug(
                "Projector contracts directory not found (correlation_id=%s)",
                correlation_id,
                extra={"contracts_dir": str(projector_contracts_dir)},
            )
            return

        projector_loader = ProjectorPluginLoader(
            config=ModelProjectorPluginLoaderConfig(graceful_mode=True),
            container=config.container,
            pool=self._pool,
        )

        try:
            discovered_projectors = await projector_loader.load_from_directory(
                projector_contracts_dir
            )
            if discovered_projectors:
                logger.info(
                    "Discovered %d projector(s) from contracts (correlation_id=%s)",
                    len(discovered_projectors),
                    correlation_id,
                    extra={
                        "discovered_count": len(discovered_projectors),
                        "projector_ids": [
                            getattr(p, "projector_id", "unknown")
                            for p in discovered_projectors
                        ],
                    },
                )

                # Extract registration projector
                registration_projector_id = "registration-projector"
                for discovered in discovered_projectors:
                    if (
                        getattr(discovered, "projector_id", None)
                        == registration_projector_id
                    ):
                        if isinstance(discovered, ProjectorShell):
                            self._projector = discovered
                            logger.info(
                                "Using contract-loaded ProjectorShell for registration "
                                "(correlation_id=%s)",
                                correlation_id,
                                extra={
                                    "projector_id": registration_projector_id,
                                    "aggregate_type": self._projector.aggregate_type,
                                },
                            )
                        break

                if self._projector is None:
                    logger.warning(
                        "Registration projector not found in contracts "
                        "(correlation_id=%s)",
                        correlation_id,
                        extra={
                            "expected_projector_id": registration_projector_id,
                            "discovered_count": len(discovered_projectors),
                        },
                    )
            else:
                logger.warning(
                    "No projector contracts found (correlation_id=%s)",
                    correlation_id,
                    extra={"contracts_dir": str(projector_contracts_dir)},
                )

        except Exception as discovery_error:
            # Log warning but continue - projector discovery is best-effort
            logger.warning(
                "Projector contract discovery failed: %s (correlation_id=%s)",
                sanitize_error_message(discovery_error),
                correlation_id,
                extra={
                    "error_type": type(discovery_error).__name__,
                    "contracts_dir": str(projector_contracts_dir),
                },
            )

    async def _initialize_schema(self, config: ModelDomainPluginConfig) -> None:
        """Initialize registration projection schema."""
        correlation_id = config.correlation_id

        schema_file = (
            Path(__file__).parent.parent.parent
            / "schemas"
            / "schema_registration_projection.sql"
        )

        if not schema_file.exists():
            logger.warning(
                "Schema file not found: %s (correlation_id=%s)",
                schema_file,
                correlation_id,
            )
            return

        if self._pool is None:
            logger.warning(
                "Cannot initialize schema: pool is None (correlation_id=%s)",
                correlation_id,
            )
            return

        try:
            schema_sql = schema_file.read_text()
            async with self._pool.acquire() as conn:
                await conn.execute(schema_sql)
            logger.info(
                "Registration projection schema initialized (correlation_id=%s)",
                correlation_id,
            )
        except Exception as schema_error:
            # Import asyncpg exceptions at runtime to check for duplicate object errors
            # PostgreSQL error codes: 42P07 = duplicate_table, 42710 = duplicate_object
            import asyncpg.exceptions

            # Catch both DuplicateTableError (42P07) and DuplicateObjectError (42710)
            # These are sibling classes covering tables and other schema objects (indexes, etc.)
            #
            # Note: isinstance is used here for exception type checking, which is standard
            # Python practice and an accepted exception to the "duck typing, never isinstance"
            # rule from CLAUDE.md. Exception handling inherently requires type discrimination
            # since exceptions don't implement protocols for error categorization.
            duplicate_errors = (
                asyncpg.exceptions.DuplicateTableError,
                asyncpg.exceptions.DuplicateObjectError,
            )
            if isinstance(schema_error, duplicate_errors):
                # Expected for idempotent schema initialization - log at DEBUG
                logger.debug(
                    "Schema already initialized (idempotent, correlation_id=%s)",
                    correlation_id,
                    extra={"error_type": type(schema_error).__name__},
                )
            else:
                # Unexpected error - log at WARNING
                logger.warning(
                    "Schema initialization encountered error: %s (correlation_id=%s)",
                    sanitize_error_message(schema_error),
                    correlation_id,
                    extra={"error_type": type(schema_error).__name__},
                )

    async def _initialize_consul_handler(self, config: ModelDomainPluginConfig) -> None:
        """Initialize Consul handler if configured."""
        correlation_id = config.correlation_id

        consul_host = os.getenv("CONSUL_HOST")
        if not consul_host:
            logger.debug(
                "CONSUL_HOST not set, Consul registration disabled (correlation_id=%s)",
                correlation_id,
            )
            return

        _MIN_PORT = 1
        _MAX_PORT = 65535
        try:
            consul_port = int(os.getenv("CONSUL_PORT", "8500"))
            if not (_MIN_PORT <= consul_port <= _MAX_PORT):
                logger.warning(
                    "CONSUL_PORT=%d out of range (%d-%d), falling back to 8500 "
                    "(correlation_id=%s)",
                    consul_port,
                    _MIN_PORT,
                    _MAX_PORT,
                    correlation_id,
                )
                consul_port = 8500
        except (ValueError, TypeError):
            logger.warning(
                "Invalid CONSUL_PORT value %r, falling back to 8500 "
                "(correlation_id=%s)",
                os.getenv("CONSUL_PORT"),
                correlation_id,
            )
            consul_port = 8500

        try:
            # Deferred import: Only load HandlerConsul when Consul is configured
            from omnibase_infra.handlers import HandlerConsul

            self._consul_handler = HandlerConsul(config.container)
            await self._consul_handler.initialize(
                {"host": consul_host, "port": consul_port}
            )
            logger.info(
                "HandlerConsul initialized for dual registration (correlation_id=%s)",
                correlation_id,
                extra={"consul_host": consul_host, "consul_port": consul_port},
            )
        except Exception as consul_error:
            # Log warning but continue without Consul (PostgreSQL is source of truth)
            logger.warning(
                "Failed to initialize HandlerConsul, proceeding without Consul: %s "
                "(correlation_id=%s)",
                sanitize_error_message(consul_error),
                correlation_id,
                extra={"error_type": type(consul_error).__name__},
            )
            self._consul_handler = None

    async def _initialize_snapshot_publisher(
        self, config: ModelDomainPluginConfig
    ) -> None:
        """Initialize SnapshotPublisher if Kafka is available.

        Creates a SnapshotPublisherRegistration for publishing compacted
        snapshots to Kafka. This is best-effort - if creation or start
        fails, the system continues without snapshot publishing.

        Args:
            config: Plugin configuration with kafka_bootstrap_servers.
        """
        correlation_id = config.correlation_id
        kafka_bootstrap_servers = config.kafka_bootstrap_servers

        if not kafka_bootstrap_servers:
            logger.debug(
                "kafka_bootstrap_servers not set, snapshot publishing disabled "
                "(correlation_id=%s)",
                correlation_id,
            )
            return

        try:
            from aiokafka import AIOKafkaProducer

            from omnibase_infra.models.projection import ModelSnapshotTopicConfig
            from omnibase_infra.projectors.snapshot_publisher_registration import (
                SnapshotPublisherRegistration,
            )

            snapshot_config = ModelSnapshotTopicConfig.default()
            snapshot_producer = AIOKafkaProducer(
                bootstrap_servers=kafka_bootstrap_servers,
            )
            self._snapshot_publisher = SnapshotPublisherRegistration(
                snapshot_producer,
                snapshot_config,
                bootstrap_servers=kafka_bootstrap_servers,
            )
            try:
                await self._snapshot_publisher.start()
            except Exception:
                # Clean up the raw AIOKafkaProducer to prevent resource leak
                try:
                    await snapshot_producer.stop()
                except Exception as producer_cleanup_error:
                    logger.warning(
                        "Failed to stop AIOKafkaProducer during cleanup: %s "
                        "(correlation_id=%s)",
                        sanitize_error_message(producer_cleanup_error),
                        correlation_id,
                        extra={
                            "error_type": type(producer_cleanup_error).__name__,
                        },
                    )
                raise
            logger.info(
                "SnapshotPublisherRegistration started for topic %s "
                "(correlation_id=%s)",
                snapshot_config.topic,
                correlation_id,
                extra={
                    "topic": snapshot_config.topic,
                    "bootstrap_servers": kafka_bootstrap_servers,
                },
            )
        except Exception as snap_pub_error:
            logger.warning(
                "Failed to start SnapshotPublisherRegistration, "
                "continuing without snapshot publishing: %s (correlation_id=%s)",
                sanitize_error_message(snap_pub_error),
                correlation_id,
                extra={
                    "error_type": type(snap_pub_error).__name__,
                },
            )
            self._snapshot_publisher = None

    async def _cleanup_on_failure(self, config: ModelDomainPluginConfig) -> None:
        """Clean up resources if initialization fails."""
        correlation_id = config.correlation_id

        if self._snapshot_publisher is not None:
            try:
                await self._snapshot_publisher.stop()
            except Exception as cleanup_error:
                logger.warning(
                    "Cleanup failed for snapshot publisher stop: %s (correlation_id=%s)",
                    sanitize_error_message(cleanup_error),
                    correlation_id,
                )
            self._snapshot_publisher = None

        if self._pool is not None:
            try:
                await self._pool.close()
            except Exception as cleanup_error:
                logger.warning(
                    "Cleanup failed for PostgreSQL pool close: %s (correlation_id=%s)",
                    sanitize_error_message(cleanup_error),
                    correlation_id,
                )
            self._pool = None

        self._projector = None
        self._consul_handler = None

    async def wire_handlers(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """Register Registration handlers with the container.

        Calls wire_registration_handlers from the wiring module to register:
        - ProjectionReaderRegistration
        - HandlerNodeIntrospected
        - HandlerRuntimeTick
        - HandlerNodeRegistrationAcked

        Args:
            config: Plugin configuration with container.

        Returns:
            Result with services_registered list on success.
        """
        from omnibase_infra.nodes.node_registration_orchestrator.wiring import (
            wire_registration_handlers,
        )

        start_time = time.time()
        correlation_id = config.correlation_id

        if self._pool is None:
            return ModelDomainPluginResult.failed(
                plugin_id=self.plugin_id,
                error_message="Cannot wire handlers: PostgreSQL pool not initialized",
            )

        try:
            registration_summary = await wire_registration_handlers(
                config.container,
                self._pool,
                projector=self._projector,
                consul_handler=self._consul_handler,
                snapshot_publisher=self._snapshot_publisher,
                correlation_id=correlation_id,
            )
            duration = time.time() - start_time

            logger.info(
                "Registration handlers wired (correlation_id=%s)",
                correlation_id,
                extra={"services": registration_summary["services"]},
            )

            # Mark handler wiring as successful so start_consumers() knows
            # it is safe to start event consumers.
            self._handler_wiring_succeeded = True

            # WiringResult TypedDict provides precise typing - direct key access is safe
            return ModelDomainPluginResult(
                plugin_id=self.plugin_id,
                success=True,
                message="Registration handlers wired",
                services_registered=registration_summary["services"],
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.exception(
                "Failed to wire Registration handlers (correlation_id=%s)",
                correlation_id,
            )
            return ModelDomainPluginResult.failed(
                plugin_id=self.plugin_id,
                error_message=sanitize_error_message(e),
                duration_seconds=duration,
            )

    async def wire_dispatchers(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """Wire registration dispatchers into the MessageDispatchEngine.

        Calls wire_registration_dispatchers() to register 4 dispatchers + 4 routes
        with the dispatch engine. The engine is frozen by the kernel after all
        plugins have completed wire_dispatchers().

        Args:
            config: Plugin configuration with container and dispatch_engine.

        Returns:
            Result indicating success/failure.
        """
        start_time = time.time()
        correlation_id = config.correlation_id

        # Check if service_registry is available
        if config.container.service_registry is None:
            logger.warning(
                "DEGRADED_MODE: ServiceRegistry not available, skipping "
                "dispatcher wiring (correlation_id=%s)",
                correlation_id,
            )
            return ModelDomainPluginResult.skipped(
                plugin_id=self.plugin_id,
                reason="ServiceRegistry not available",
            )

        if config.dispatch_engine is None:
            logger.warning(
                "DEGRADED_MODE: dispatch_engine not available, skipping "
                "dispatcher wiring (correlation_id=%s)",
                correlation_id,
            )
            return ModelDomainPluginResult.skipped(
                plugin_id=self.plugin_id,
                reason="dispatch_engine not available",
            )

        try:
            from omnibase_infra.nodes.node_registration_orchestrator.wiring import (
                wire_registration_dispatchers,
            )

            dispatch_summary = await wire_registration_dispatchers(
                container=config.container,
                engine=config.dispatch_engine,
                correlation_id=correlation_id,
            )

            duration = time.time() - start_time
            logger.info(
                "Registration dispatchers wired into engine (correlation_id=%s)",
                correlation_id,
                extra={
                    "dispatchers": dispatch_summary.get("dispatchers", []),
                    "routes": dispatch_summary.get("routes", []),
                },
            )

            return ModelDomainPluginResult(
                plugin_id=self.plugin_id,
                success=True,
                message="Registration dispatchers wired into engine",
                resources_created=list(dispatch_summary.get("dispatchers", [])),
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.exception(
                "Failed to wire registration dispatchers (correlation_id=%s)",
                correlation_id,
            )
            return ModelDomainPluginResult.failed(
                plugin_id=self.plugin_id,
                error_message=sanitize_error_message(e),
                duration_seconds=duration,
            )

    async def start_consumers(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """Start event consumers via EventBusSubcontractWiring.

        Reads subscribe_topics from the registration orchestrator contract
        and wires Kafka subscriptions through EventBusSubcontractWiring.
        Messages are deserialized and dispatched through the frozen
        MessageDispatchEngine. Output events are published by the
        DispatchResultApplier.

        Requires config.node_identity and config.dispatch_engine to be set.

        Args:
            config: Plugin configuration with event_bus, node_identity,
                and dispatch_engine.

        Returns:
            Result with unsubscribe_callbacks for cleanup.
        """
        start_time = time.time()
        correlation_id = config.correlation_id

        # Guard: do not start consumers if handler wiring failed. Starting
        # consumers without wired handlers would route messages to an empty
        # dispatch engine, causing silent message loss or RuntimeHostError.
        if not self._handler_wiring_succeeded:
            logger.warning(
                "Skipping consumer startup: handler wiring did not succeed "
                "for plugin '%s' (correlation_id=%s)",
                self.plugin_id,
                correlation_id,
            )
            return ModelDomainPluginResult.skipped(
                plugin_id=self.plugin_id,
                reason="Handler wiring did not succeed — consumers not started",
            )

        if config.dispatch_engine is None:
            return ModelDomainPluginResult.skipped(
                plugin_id=self.plugin_id,
                reason="dispatch_engine not available",
            )

        # Check for subscribe capability via runtime-checkable protocol.
        # ProtocolEventBusSubscriber is @runtime_checkable, so isinstance works
        # without coupling to concrete InMemoryEventBus / KafkaEventBus.
        from omnibase_core.protocols.event_bus.protocol_event_bus_subscriber import (
            ProtocolEventBusSubscriber,
        )

        if not isinstance(config.event_bus, ProtocolEventBusSubscriber):
            return ModelDomainPluginResult.skipped(
                plugin_id=self.plugin_id,
                reason="Event bus does not support subscribe",
            )

        if config.node_identity is None:
            return ModelDomainPluginResult.skipped(
                plugin_id=self.plugin_id,
                reason="node_identity not set (required for consumer subscription)",
            )

        try:
            from omnibase_core.enums import EnumInjectionScope
            from omnibase_infra.runtime.event_bus_subcontract_wiring import (
                EventBusSubcontractWiring,
                load_event_bus_subcontract,
            )
            from omnibase_infra.runtime.service_dispatch_result_applier import (
                DispatchResultApplier,
            )
            from omnibase_infra.runtime.service_intent_executor import (
                IntentExecutor,
            )

            # Load event_bus subcontract from registration orchestrator contract
            contract_path = Path(__file__).parent / "contract.yaml"
            subcontract = load_event_bus_subcontract(contract_path, logger=logger)

            if subcontract is None:
                return ModelDomainPluginResult.skipped(
                    plugin_id=self.plugin_id,
                    reason=f"No event_bus subcontract in {contract_path}",
                )

            # Create intent executor for effect layer delegation (Phase C)
            # and register in the DI container for consistent service resolution.
            intent_executor = IntentExecutor(container=config.container)

            # Wire intent effect adapters from contract-driven routing table.
            # Effect adapters are registered with the intent_executor and also
            # placed into the DI container for discoverability.
            await self._wire_intent_effects(
                intent_executor=intent_executor,
                contract_path=contract_path,
                correlation_id=correlation_id,
                config=config,
            )

            # Register IntentExecutor in the DI container so downstream services
            # can resolve it via container.service_registry.resolve_service()
            # rather than receiving direct references. This follows the
            # container-based DI pattern required by coding guidelines.
            if config.container.service_registry is not None:
                await config.container.service_registry.register_instance(
                    interface=IntentExecutor,
                    instance=intent_executor,
                    scope=EnumInjectionScope.GLOBAL,
                    metadata={
                        "description": "Intent executor for registration domain",
                        "plugin_id": self.plugin_id,
                    },
                )
                logger.debug(
                    "Registered IntentExecutor in container (correlation_id=%s)",
                    correlation_id,
                )

            # Create dispatch result applier for output event publishing + intent delegation
            # ProtocolEventBusSubscriber satisfies ProtocolEventBusLike structurally
            # (both define publish_envelope) but mypy can't infer this across
            # unrelated protocol hierarchies.
            result_applier = DispatchResultApplier(
                event_bus=config.event_bus,  # type: ignore[arg-type]
                output_topic=config.output_topic,
                intent_executor=intent_executor,
            )

            # Register DispatchResultApplier in the DI container for the same
            # container-based DI consistency.
            if config.container.service_registry is not None:
                await config.container.service_registry.register_instance(
                    interface=DispatchResultApplier,
                    instance=result_applier,
                    scope=EnumInjectionScope.GLOBAL,
                    metadata={
                        "description": "Dispatch result applier for registration domain",
                        "plugin_id": self.plugin_id,
                    },
                )
                logger.debug(
                    "Registered DispatchResultApplier in container (correlation_id=%s)",
                    correlation_id,
                )

            # Create EventBusSubcontractWiring with dispatch engine and result applier
            self._wiring = EventBusSubcontractWiring(
                event_bus=config.event_bus,
                dispatch_engine=config.dispatch_engine,
                environment=config.node_identity.env,
                node_name=config.node_identity.node_name,
                service=config.node_identity.service,
                version=config.node_identity.version,
                result_applier=result_applier,
            )

            # Register EventBusSubcontractWiring in the DI container for
            # consistent service resolution and discoverability.
            if config.container.service_registry is not None:
                await config.container.service_registry.register_instance(
                    interface=EventBusSubcontractWiring,
                    instance=self._wiring,
                    scope=EnumInjectionScope.GLOBAL,
                    metadata={
                        "description": "Event bus subcontract wiring for registration domain",
                        "plugin_id": self.plugin_id,
                    },
                )
                logger.debug(
                    "Registered EventBusSubcontractWiring in container "
                    "(correlation_id=%s)",
                    correlation_id,
                )

            # Wire subscriptions from contract-declared topics
            await self._wiring.wire_subscriptions(
                subcontract=subcontract,
                node_name="registration-orchestrator",
            )

            logger.info(
                "Registration consumers started via EventBusSubcontractWiring "
                "(correlation_id=%s)",
                correlation_id,
                extra={
                    "subscribe_topics": subcontract.subscribe_topics,
                    "topic_count": len(subcontract.subscribe_topics)
                    if subcontract.subscribe_topics
                    else 0,
                },
            )

            duration = time.time() - start_time

            # Cleanup callback wraps the wiring cleanup
            async def _cleanup_wiring() -> None:
                if self._wiring is not None:
                    await self._wiring.cleanup()

            return ModelDomainPluginResult(
                plugin_id=self.plugin_id,
                success=True,
                message="Registration consumers started via EventBusSubcontractWiring",
                duration_seconds=duration,
                unsubscribe_callbacks=[_cleanup_wiring],
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.exception(
                "Failed to start registration consumers (correlation_id=%s)",
                correlation_id,
            )
            return ModelDomainPluginResult.failed(
                plugin_id=self.plugin_id,
                error_message=sanitize_error_message(e),
                duration_seconds=duration,
            )

    async def _wire_intent_effects(
        self,
        intent_executor: IntentExecutor,
        contract_path: Path,
        correlation_id: UUID | None,
        config: ModelDomainPluginConfig | None = None,
    ) -> None:
        """Wire intent effect adapters from contract-driven routing table.

        Reads the intent_routing_table from the contract and registers
        appropriate effect adapters with the IntentExecutor. Effect adapters
        are created only for intent types where the required infrastructure
        resources (projector, consul_handler) are available.

        When a config with a service_registry is available, effect adapters
        are also registered in the DI container for discoverability and
        consistent service resolution.

        Args:
            intent_executor: IntentExecutor to register handlers with.
            contract_path: Path to the contract YAML with intent_routing_table.
            correlation_id: Correlation ID for logging.
            config: Optional plugin config for DI container registration.
        """
        from omnibase_infra.runtime.service_intent_routing_loader import (
            load_intent_routing_table,
        )

        routing_table = load_intent_routing_table(contract_path, logger_override=logger)
        if not routing_table:
            logger.debug(
                "No intent routing table found, IntentExecutor has no effect "
                "handlers (correlation_id=%s)",
                correlation_id,
            )
            return

        # IntentExecutor is a known type with register_handler() — direct call,
        # no getattr duck-typing needed.

        # Build set of wirable intent_types based on available infrastructure.
        # This avoids duplicating the protocol-match conditional in both the
        # registration loop and the post-registration validation.
        #
        # Each intent_type maps to its specific resource gate. This allows
        # different postgres.* intents to require different resources
        # (e.g. upsert needs projector, update needs pool directly).
        _protocol_resources = {
            "postgres.upsert_registration": self._projector is not None,
            "postgres.update_registration": self._pool is not None,
            "consul.register": self._consul_handler is not None,
            "consul.deregister": self._consul_handler is not None,
        }
        wirable_intent_types: set[str] = set()
        for it in routing_table:
            if _protocol_resources.get(it, False):
                wirable_intent_types.add(it)

        registered_count = 0

        for intent_type in routing_table:
            if intent_type not in wirable_intent_types:
                logger.debug(
                    "Skipping intent_type=%s (no matching adapter or resource "
                    "unavailable) (correlation_id=%s)",
                    intent_type,
                    correlation_id,
                )
                continue

            if (
                intent_type == "postgres.upsert_registration"
                and self._projector is not None
            ):
                from omnibase_infra.runtime.intent_effects import (
                    IntentEffectPostgresUpsert,
                )

                pg_upsert_effect = IntentEffectPostgresUpsert(projector=self._projector)
                intent_executor.register_handler(intent_type, pg_upsert_effect)
                await self._register_effect_in_container(
                    config, IntentEffectPostgresUpsert, pg_upsert_effect, correlation_id
                )
                registered_count += 1
                logger.debug(
                    "Registered IntentEffectPostgresUpsert for intent_type=%s "
                    "(correlation_id=%s)",
                    intent_type,
                    correlation_id,
                )

            elif (
                intent_type == "postgres.update_registration" and self._pool is not None
            ):
                from omnibase_infra.runtime.intent_effects import (
                    IntentEffectPostgresUpdate,
                )

                pg_update_effect = IntentEffectPostgresUpdate(pool=self._pool)
                intent_executor.register_handler(intent_type, pg_update_effect)
                await self._register_effect_in_container(
                    config, IntentEffectPostgresUpdate, pg_update_effect, correlation_id
                )
                registered_count += 1
                logger.debug(
                    "Registered IntentEffectPostgresUpdate for intent_type=%s "
                    "(correlation_id=%s)",
                    intent_type,
                    correlation_id,
                )

            elif intent_type == "consul.register" and self._consul_handler is not None:
                from omnibase_infra.runtime.intent_effects import (
                    IntentEffectConsulRegister,
                )

                consul_effect = IntentEffectConsulRegister(
                    consul_handler=self._consul_handler,
                )
                intent_executor.register_handler(intent_type, consul_effect)
                await self._register_effect_in_container(
                    config, IntentEffectConsulRegister, consul_effect, correlation_id
                )
                registered_count += 1
                logger.debug(
                    "Registered IntentEffectConsulRegister for intent_type=%s "
                    "(correlation_id=%s)",
                    intent_type,
                    correlation_id,
                )

            elif (
                intent_type == "consul.deregister" and self._consul_handler is not None
            ):
                from omnibase_infra.runtime.intent_effects import (
                    IntentEffectConsulDeregister,
                )

                deregister_effect = IntentEffectConsulDeregister(
                    consul_handler=self._consul_handler,
                )
                intent_executor.register_handler(intent_type, deregister_effect)
                await self._register_effect_in_container(
                    config,
                    IntentEffectConsulDeregister,
                    deregister_effect,
                    correlation_id,
                )
                registered_count += 1
                logger.debug(
                    "Registered IntentEffectConsulDeregister for intent_type=%s "
                    "(correlation_id=%s)",
                    intent_type,
                    correlation_id,
                )

        # Startup validation: warn about intent_types declared in the routing
        # table that have no registered handler.  These will cause RuntimeHostError
        # at runtime when DispatchResultApplier calls IntentExecutor.execute().
        # We log at WARNING (not ERROR) so the system still starts, but operators
        # can quickly identify missing infrastructure.
        unwired_intents = [it for it in routing_table if it not in wirable_intent_types]

        if unwired_intents:
            logger.warning(
                "Intent routing table declares %d intent type(s) with no "
                "registered effect handler: %s. Intents of these types will "
                "raise RuntimeHostError at runtime. Check that the required "
                "infrastructure (projector, consul_handler) is available. "
                "(correlation_id=%s)",
                len(unwired_intents),
                unwired_intents,
                correlation_id,
                extra={
                    "unwired_intent_types": unwired_intents,
                    "routing_table_size": len(routing_table),
                    "registered_count": registered_count,
                },
            )

        logger.info(
            "Wired %d intent effect adapter(s) from contract routing table "
            "(correlation_id=%s)",
            registered_count,
            correlation_id,
            extra={
                "routing_table_size": len(routing_table),
                "registered_count": registered_count,
            },
        )

    @staticmethod
    async def _register_effect_in_container(
        config: ModelDomainPluginConfig | None,
        interface: type,
        instance: object,
        correlation_id: UUID | None,
    ) -> None:
        """Register an intent effect adapter in the DI container.

        Best-effort registration: if the service_registry is not available,
        the effect is still registered with the IntentExecutor (the primary
        routing mechanism) and a debug message is logged.

        Args:
            config: Plugin config providing the container. None is tolerated
                for backwards compatibility with callers that do not pass config.
            interface: The class/type to use as the DI interface key.
            instance: The effect adapter instance to register.
            correlation_id: Correlation ID for logging.
        """
        if config is None or config.container.service_registry is None:
            logger.debug(
                "Skipping DI registration for %s (no service_registry) "
                "(correlation_id=%s)",
                interface.__name__,
                correlation_id,
            )
            return

        from omnibase_core.enums import EnumInjectionScope

        await config.container.service_registry.register_instance(
            interface=interface,
            instance=instance,
            scope=EnumInjectionScope.GLOBAL,
            metadata={
                "description": f"Intent effect adapter: {interface.__name__}",
                "plugin_id": "registration",
            },
        )
        logger.debug(
            "Registered %s in DI container (correlation_id=%s)",
            interface.__name__,
            correlation_id,
        )

    async def shutdown(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """Clean up Registration resources.

        Closes the PostgreSQL pool. Other resources (handlers, dispatchers)
        are managed by the container.

        Thread Safety:
            Guards against concurrent shutdown calls via _shutdown_in_progress flag.
            While the kernel's LIFO shutdown prevents double-shutdown at the
            orchestration level, this guard protects against direct concurrent
            calls to the plugin's shutdown method.

        Args:
            config: Plugin configuration.

        Returns:
            Result indicating cleanup success/failure.
        """
        # Guard against concurrent shutdown calls
        if self._shutdown_in_progress:
            return ModelDomainPluginResult.skipped(
                plugin_id=self.plugin_id,
                reason="Shutdown already in progress",
            )
        self._shutdown_in_progress = True

        try:
            return await self._do_shutdown(config)
        finally:
            self._shutdown_in_progress = False

    async def _do_shutdown(
        self,
        config: ModelDomainPluginConfig,
    ) -> ModelDomainPluginResult:
        """Internal shutdown implementation.

        Shutdown order: snapshot_publisher -> pool -> clear references.
        Snapshot publisher is stopped first because it depends on Kafka,
        which may be shutting down independently.

        Args:
            config: Plugin configuration.

        Returns:
            Result indicating cleanup success/failure.
        """
        start_time = time.time()
        correlation_id = config.correlation_id
        errors: list[str] = []

        # Stop snapshot publisher first (depends on external Kafka)
        if self._snapshot_publisher is not None:
            try:
                await self._snapshot_publisher.stop()
                logger.debug(
                    "Snapshot publisher stopped (correlation_id=%s)",
                    correlation_id,
                )
            except Exception as snap_stop_error:
                error_msg = sanitize_error_message(snap_stop_error)
                errors.append(f"snapshot_publisher: {error_msg}")
                logger.warning(
                    "Failed to stop snapshot publisher: %s (correlation_id=%s)",
                    error_msg,
                    correlation_id,
                )
            self._snapshot_publisher = None

        if self._pool is not None:
            try:
                await self._pool.close()
                logger.debug(
                    "PostgreSQL pool closed (correlation_id=%s)",
                    correlation_id,
                )
            except Exception as pool_close_error:
                error_msg = sanitize_error_message(pool_close_error)
                errors.append(f"pool_close: {error_msg}")
                logger.warning(
                    "Failed to close PostgreSQL pool: %s (correlation_id=%s)",
                    error_msg,
                    correlation_id,
                )
            self._pool = None

        self._projector = None
        self._consul_handler = None
        self._wiring = None

        duration = time.time() - start_time

        if errors:
            return ModelDomainPluginResult.failed(
                plugin_id=self.plugin_id,
                error_message="; ".join(errors),
                duration_seconds=duration,
            )

        return ModelDomainPluginResult.succeeded(
            plugin_id=self.plugin_id,
            message="Registration resources cleaned up",
            duration_seconds=duration,
        )

    def get_status_line(self) -> str:
        """Get status line for kernel banner.

        Returns:
            Status string indicating enabled state and backends.
        """
        if self._pool is None:
            return "disabled"

        parts = ["PostgreSQL"]
        if self._consul_handler is not None:
            parts.append("Consul")
        if self._snapshot_publisher is not None:
            parts.append("Snapshots")
        return f"enabled ({' + '.join(parts)})"


# Verify protocol compliance at type-check time (mypy/pyright).
# No runtime instantiation needed — avoids side-effects at import time.
if TYPE_CHECKING:
    _: ProtocolDomainPlugin = PluginRegistration()

__all__: list[str] = [
    "PROJECTOR_CONTRACTS_DEFAULT_DIR",
    "PluginRegistration",
]
