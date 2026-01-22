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
    - POSTGRES_HOST: Required for plugin activation
    - POSTGRES_PORT: Optional (default: 5432)
    - POSTGRES_USER: Optional (default: postgres)
    - POSTGRES_PASSWORD: Required when POSTGRES_HOST is set
    - POSTGRES_DATABASE: Optional (default: omninode_bridge)
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

if TYPE_CHECKING:
    import asyncpg

    from omnibase_infra.handlers import HandlerConsul
    from omnibase_infra.nodes.node_registration_orchestrator.dispatchers import (
        DispatcherNodeIntrospected,
    )
    from omnibase_infra.runtime.projector_shell import ProjectorShell

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ContainerWiringError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.runtime.protocol_domain_plugin import (
    ModelDomainPluginConfig,
    ModelDomainPluginResult,
    ProtocolDomainPlugin,
)
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
        _introspection_dispatcher: Dispatcher for introspection events
    """

    def __init__(self) -> None:
        """Initialize the plugin with empty state."""
        self._pool: asyncpg.Pool | None = None
        self._projector: ProjectorShell | None = None
        self._consul_handler: HandlerConsul | None = None
        self._introspection_dispatcher: DispatcherNodeIntrospected | None = None
        self._shutdown_in_progress: bool = False

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

    def should_activate(self, config: ModelDomainPluginConfig) -> bool:
        """Check if Registration should activate based on environment.

        Returns True if POSTGRES_HOST is set, indicating PostgreSQL
        is configured for registration support.

        Args:
            config: Plugin configuration (not used for this check).

        Returns:
            True if POSTGRES_HOST environment variable is set.
        """
        postgres_host = os.getenv("POSTGRES_HOST")
        if not postgres_host:
            logger.debug(
                "Registration plugin inactive: POSTGRES_HOST not set "
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
            # 1. Create PostgreSQL pool
            postgres_host = os.getenv("POSTGRES_HOST")
            postgres_dsn = (
                f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}:"
                f"{os.getenv('POSTGRES_PASSWORD', '')}@"
                f"{postgres_host}:"
                f"{os.getenv('POSTGRES_PORT', '5432')}/"
                f"{os.getenv('POSTGRES_DATABASE', 'omninode_bridge')}"
            )

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
                extra={
                    "host": postgres_host,
                    "port": os.getenv("POSTGRES_PORT", "5432"),
                    "database": os.getenv("POSTGRES_DATABASE", "omninode_bridge"),
                },
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

            duration = time.time() - start_time
            # Use constructor directly for results with resources_created
            return ModelDomainPluginResult(
                plugin_id=self.plugin_id,
                success=True,
                message="Registration plugin initialized",
                resources_created=resources_created,
                duration_seconds=duration,
            )

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

        consul_port = int(os.getenv("CONSUL_PORT", "8500"))

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

    async def _cleanup_on_failure(self, config: ModelDomainPluginConfig) -> None:
        """Clean up resources if initialization fails."""
        correlation_id = config.correlation_id

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
                correlation_id=correlation_id,
            )
            duration = time.time() - start_time

            logger.info(
                "Registration handlers wired (correlation_id=%s)",
                correlation_id,
                extra={"services": registration_summary["services"]},
            )

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
        """Create introspection dispatcher (no dispatch engine wiring).

        Creates the introspection dispatcher from container-resolved handlers.
        Note: This plugin does not register with MessageDispatchEngine directly;
        introspection events are consumed via dedicated Kafka consumer.

        Args:
            config: Plugin configuration with container.

        Returns:
            Result indicating success/failure.
        """
        start_time = time.time()
        correlation_id = config.correlation_id

        # Check if service_registry is available
        if config.container.service_registry is None:
            logger.warning(
                "DEGRADED_MODE: ServiceRegistry not available, skipping introspection "
                "dispatcher creation (correlation_id=%s)",
                correlation_id,
            )
            return ModelDomainPluginResult.skipped(
                plugin_id=self.plugin_id,
                reason="ServiceRegistry not available",
            )

        try:
            # Deferred import: HandlerNodeIntrospected depends on registration infra
            from omnibase_infra.nodes.node_registration_orchestrator.dispatchers import (
                DispatcherNodeIntrospected,
            )
            from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
                HandlerNodeIntrospected,
            )

            logger.debug(
                "Resolving HandlerNodeIntrospected from container (correlation_id=%s)",
                correlation_id,
            )

            handler_introspected: HandlerNodeIntrospected = (
                await config.container.service_registry.resolve_service(
                    HandlerNodeIntrospected
                )
            )

            self._introspection_dispatcher = DispatcherNodeIntrospected(
                handler_introspected
            )

            duration = time.time() - start_time
            logger.info(
                "Introspection dispatcher created (correlation_id=%s)",
                correlation_id,
                extra={
                    "dispatcher_class": self._introspection_dispatcher.__class__.__name__,
                    "handler_class": handler_introspected.__class__.__name__,
                },
            )

            # Use constructor directly for results with resources_created
            return ModelDomainPluginResult(
                plugin_id=self.plugin_id,
                success=True,
                message="Introspection dispatcher created",
                resources_created=["introspection_dispatcher"],
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.exception(
                "Failed to create introspection dispatcher (correlation_id=%s)",
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
        """Start introspection event consumer.

        Subscribes to the input topic to route introspection events to
        HandlerNodeIntrospected via IntrospectionEventRouter.

        Note: Only starts consumer for KafkaEventBus, not InMemoryEventBus.

        Args:
            config: Plugin configuration with event_bus.

        Returns:
            Result with unsubscribe_callbacks for cleanup.
        """
        from omnibase_infra.nodes.node_registration_orchestrator.introspection_event_router import (
            IntrospectionEventRouter,
        )

        start_time = time.time()
        correlation_id = config.correlation_id

        if self._introspection_dispatcher is None:
            return ModelDomainPluginResult.skipped(
                plugin_id=self.plugin_id,
                reason="Introspection dispatcher not available",
            )

        # Duck typing: check for subscribe capability rather than concrete type
        # Per CLAUDE.md: "Protocol Resolution - Duck typing through protocols, never isinstance"
        if not hasattr(config.event_bus, "subscribe"):
            return ModelDomainPluginResult.skipped(
                plugin_id=self.plugin_id,
                reason="Event bus does not support subscribe",
            )

        try:
            # Create event router with container-based DI pattern
            introspection_event_router = IntrospectionEventRouter(
                container=config.container,
                dispatcher=self._introspection_dispatcher,
                event_bus=config.event_bus,
                output_topic=config.output_topic,
            )

            # Subscribe to input topic
            logger.info(
                "Subscribing to introspection events on Kafka (correlation_id=%s)",
                correlation_id,
                extra={
                    "topic": config.input_topic,
                    "consumer_group": f"{config.consumer_group}-introspection",
                },
            )

            introspection_unsubscribe = await config.event_bus.subscribe(
                topic=config.input_topic,
                group_id=f"{config.consumer_group}-introspection",
                on_message=introspection_event_router.handle_message,
            )

            duration = time.time() - start_time
            logger.info(
                "Introspection event consumer started (correlation_id=%s)",
                correlation_id,
                extra={
                    "subscribe_duration_seconds": duration,
                },
            )

            # Use constructor directly since we need unsubscribe_callbacks
            return ModelDomainPluginResult(
                plugin_id=self.plugin_id,
                success=True,
                message="Introspection event consumer started",
                duration_seconds=duration,
                unsubscribe_callbacks=[introspection_unsubscribe],
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.exception(
                "Failed to start introspection consumer (correlation_id=%s)",
                correlation_id,
            )
            return ModelDomainPluginResult.failed(
                plugin_id=self.plugin_id,
                error_message=sanitize_error_message(e),
                duration_seconds=duration,
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

        Args:
            config: Plugin configuration.

        Returns:
            Result indicating cleanup success/failure.
        """
        start_time = time.time()
        correlation_id = config.correlation_id
        errors: list[str] = []

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
        self._introspection_dispatcher = None

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

        if self._consul_handler is not None:
            return "enabled (PostgreSQL + Consul)"
        return "enabled (PostgreSQL only)"


# Verify protocol compliance at module load time
_: ProtocolDomainPlugin = PluginRegistration()

__all__: list[str] = [
    "PROJECTOR_CONTRACTS_DEFAULT_DIR",
    "PluginRegistration",
]
