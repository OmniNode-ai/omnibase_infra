# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Kernel - Minimal bootstrap for contract-driven runtime.

This is the kernel entrypoint for the ONEX runtime. It provides a contract-driven
bootstrap that wires configuration into the existing RuntimeHostProcess.

The kernel is responsible for:
    1. Loading runtime configuration from contracts or environment
    2. Creating and starting the event bus (EventBusInmemory or EventBusKafka)
    3. Building the dependency container (event_bus, config)
    4. Instantiating RuntimeHostProcess with contract-driven configuration
    5. Starting the HTTP health server for Docker/K8s probes
    6. Setting up graceful shutdown signal handlers
    7. Running the runtime until shutdown is requested

Event Bus Selection:
    The kernel supports two event bus implementations:
    - EventBusInmemory: For local development and testing (default)
    - EventBusKafka: For production use with Kafka/Redpanda

    Selection is determined by:
    - KAFKA_BOOTSTRAP_SERVERS environment variable (if set, uses Kafka)
    - config.event_bus.type field in runtime_config.yaml

Usage:
    # Run with default contracts directory (./contracts)
    python -m omnibase_infra.runtime.service_kernel

    # Run with custom contracts directory
    ONEX_CONTRACTS_DIR=/path/to/contracts python -m omnibase_infra.runtime.service_kernel

    # Or via the installed entrypoint
    onex-runtime

Environment Variables:
    ONEX_CONTRACTS_DIR: Path to contracts directory (default: ./contracts)
    ONEX_HTTP_PORT: Port for health check HTTP server (default: 8085)
    ONEX_LOG_LEVEL: Logging level (default: INFO)
    ONEX_ENVIRONMENT: Runtime environment name (default: local)

Note:
    This kernel uses the existing RuntimeHostProcess as the core runtime engine.
    A future refactor may integrate NodeOrchestrator as the primary execution
    engine, but for MVP this lean kernel provides contract-driven bootstrap
    with minimal risk and maximum reuse of tested code.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from collections.abc import Awaitable, Callable
from importlib.metadata import version as get_package_version
from pathlib import Path
from typing import cast
from uuid import UUID

import asyncpg
import yaml
from pydantic import ValidationError

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.enums import EnumConsumerGroupPurpose, EnumInfraTransportType
from omnibase_infra.errors import (
    ModelInfraErrorContext,
    ProtocolConfigurationError,
    RuntimeHostError,
    ServiceResolutionError,
)
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.nodes.node_registration_orchestrator.dispatchers import (
    DispatcherNodeIntrospected,
)
from omnibase_infra.nodes.node_registration_orchestrator.introspection_event_router import (
    IntrospectionEventRouter,
)
from omnibase_infra.runtime.handler_registry import RegistryProtocolBinding
from omnibase_infra.runtime.models import (
    ModelProjectorPluginLoaderConfig,
    ModelRuntimeConfig,
)
from omnibase_infra.runtime.projector_plugin_loader import (
    ProjectorPluginLoader,
    ProjectorShell,
    ProtocolEventProjector,
)
from omnibase_infra.runtime.service_runtime_host_process import RuntimeHostProcess
from omnibase_infra.runtime.util_container_wiring import (
    wire_infrastructure_services,
    wire_registration_handlers,
)

# Circular Import Note (OMN-529):
# ---------------------------------
# ServiceHealth and DEFAULT_HTTP_PORT are imported inside bootstrap() rather than
# at module level to avoid a circular import. The import chain is:
#
#   1. omnibase_infra/runtime/__init__.py imports kernel_bootstrap from kernel.py
#   2. If kernel.py imported ServiceHealth at module level, it would load service_health.py
#   3. service_health.py imports ModelHealthCheckResponse from runtime.models
#   4. This triggers initialization of omnibase_infra.runtime package (step 1)
#   5. Runtime package tries to import kernel.py which is still initializing -> circular!
#
# The lazy import in bootstrap() is acceptable because:
#   - ServiceHealth is only instantiated at runtime, not at import time
#   - Type checking uses forward references (no import needed)
#   - No import-time side effects are bypassed
#   - The omnibase_infra.services.__init__.py already excludes ServiceHealth exports
#     to prevent accidental circular imports from other modules
#
# See also: omnibase_infra/services/__init__.py "ServiceHealth Import Guide" section
from omnibase_infra.runtime.util_validation import validate_runtime_config
from omnibase_infra.utils.correlation import generate_correlation_id
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

logger = logging.getLogger(__name__)

# Kernel version - read from installed package metadata to avoid version drift
# between code and pyproject.toml. Falls back to "unknown" if package is not
# installed (e.g., during development without editable install).
try:
    KERNEL_VERSION = get_package_version("omnibase_infra")
except Exception:
    KERNEL_VERSION = "unknown"

# Default configuration
DEFAULT_CONTRACTS_DIR = "./contracts"
DEFAULT_RUNTIME_CONFIG = "runtime/runtime_config.yaml"

# Environment variable name for contracts directory
ENV_CONTRACTS_DIR = "ONEX_CONTRACTS_DIR"
DEFAULT_INPUT_TOPIC = "requests"
DEFAULT_OUTPUT_TOPIC = "responses"
DEFAULT_GROUP_ID = "onex-runtime"

# Port validation constants
MIN_PORT = 1
MAX_PORT = 65535


def _get_contracts_dir() -> Path:
    """Get contracts directory from environment.

    Reads the ONEX_CONTRACTS_DIR environment variable. If not set,
    returns the default contracts directory.

    Returns:
        Path to the contracts directory.
    """
    onex_value = os.environ.get(ENV_CONTRACTS_DIR)
    if onex_value:
        return Path(onex_value)

    return Path(DEFAULT_CONTRACTS_DIR)


def load_runtime_config(
    contracts_dir: Path,
    correlation_id: UUID | None = None,
) -> ModelRuntimeConfig:
    """Load runtime configuration from contract file or return defaults.

    Attempts to load runtime_config.yaml from the contracts directory.
    If the file doesn't exist, returns sensible defaults to allow
    the runtime to start without requiring a config file.

    Configuration Loading Process:
        1. Check for runtime_config.yaml in contracts directory
        2. If found, parse YAML and validate against ModelRuntimeConfig schema
        3. If not found, construct config from environment variables and defaults
        4. Return fully validated configuration model

    Configuration Precedence:
        - File-based config is returned as-is when present (no environment overrides)
        - Environment variables are only used when no config file exists
        - Defaults are used when neither file nor environment variables are set
        - Note: Environment overrides (e.g., ONEX_ENVIRONMENT) are applied by the
          caller (bootstrap), not by this function

    Args:
        contracts_dir: Path to the contracts directory containing runtime_config.yaml.
            Example: Path("./contracts") or Path("/app/contracts")
        correlation_id: Optional correlation ID for distributed tracing. If not
            provided, a new one will be generated. Passing a correlation_id from
            the caller (e.g., bootstrap) ensures consistent tracing across the
            initialization sequence.

    Returns:
        ModelRuntimeConfig: Fully validated configuration model with runtime settings.
            Contains event bus configuration, topic names, consumer group, shutdown
            behavior, and logging configuration.

    Raises:
        ProtocolConfigurationError: If config file exists but cannot be parsed,
            fails validation, or cannot be read due to filesystem errors. Error
            includes correlation_id for tracing and detailed context for debugging.

    Example:
        >>> contracts_dir = Path("./contracts")
        >>> config = load_runtime_config(contracts_dir)
        >>> print(config.input_topic)
        requests
        >>> print(config.event_bus.type)
        inmemory

    Example Error:
        >>> # If runtime_config.yaml has invalid YAML syntax
        >>> load_runtime_config(Path("./invalid"))
        ProtocolConfigurationError: Failed to parse runtime config YAML at ./invalid/runtime/runtime_config.yaml
        (correlation_id: 123e4567-e89b-12d3-a456-426614174000)
    """
    config_path = contracts_dir / DEFAULT_RUNTIME_CONFIG
    # Use passed correlation_id for consistent tracing, or generate new one
    effective_correlation_id = correlation_id or generate_correlation_id()
    context = ModelInfraErrorContext(
        transport_type=EnumInfraTransportType.RUNTIME,
        operation="load_config",
        target_name=str(config_path),
        correlation_id=effective_correlation_id,
    )

    if config_path.exists():
        logger.info(
            "Loading runtime config from %s (correlation_id=%s)",
            config_path,
            effective_correlation_id,
        )
        try:
            with config_path.open(encoding="utf-8") as f:
                raw_config = yaml.safe_load(f) or {}

            # Type guard: reject non-mapping YAML payloads
            # yaml.safe_load() can return list, str, int, etc. for valid YAML
            # but runtime config must be a dict (mapping) for model validation
            if not isinstance(raw_config, dict):
                raise ProtocolConfigurationError(
                    f"Runtime config at {config_path} must be a YAML mapping (dict), "
                    f"got {type(raw_config).__name__}",
                    context=context,
                    config_path=str(config_path),
                    error_details=f"Expected dict, got {type(raw_config).__name__}",
                )

            # Contract validation: validate against schema before Pydantic
            # This provides early, actionable error messages for pattern/range violations
            contract_errors = validate_runtime_config(raw_config)
            if contract_errors:
                error_count = len(contract_errors)
                # Create concise summary for log message (first 3 errors)
                error_summary = "; ".join(contract_errors[:3])
                if error_count > 3:
                    error_summary += f" (and {error_count - 3} more...)"
                raise ProtocolConfigurationError(
                    f"Contract validation failed at {config_path}: {error_count} error(s). "
                    f"First errors: {error_summary}",
                    context=context,
                    config_path=str(config_path),
                    # Full error list for structured debugging (not truncated)
                    validation_errors=contract_errors,
                    error_count=error_count,
                )
            logger.debug(
                "Contract validation passed (correlation_id=%s)",
                effective_correlation_id,
            )

            config = ModelRuntimeConfig.model_validate(raw_config)
            logger.debug(
                "Runtime config loaded successfully (correlation_id=%s)",
                effective_correlation_id,
                extra={
                    "input_topic": config.input_topic,
                    "output_topic": config.output_topic,
                    "consumer_group": config.consumer_group,
                    "event_bus_type": config.event_bus.type,
                },
            )
            return config
        except yaml.YAMLError as e:
            raise ProtocolConfigurationError(
                f"Failed to parse runtime config YAML at {config_path}: {e}",
                context=context,
                config_path=str(config_path),
                error_details=str(e),
            ) from e
        except ValidationError as e:
            # Extract validation error details for actionable error messages
            error_count = e.error_count()
            # Convert Pydantic errors to list[str] for consistency with contract validation
            # Both validation_errors fields should have the same type: list[str]
            pydantic_errors = [
                f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
                for err in e.errors()
            ]
            error_summary = "; ".join(pydantic_errors[:3])
            raise ProtocolConfigurationError(
                f"Runtime config validation failed at {config_path}: {error_count} error(s). "
                f"First errors: {error_summary}",
                context=context,
                config_path=str(config_path),
                validation_errors=pydantic_errors,
                error_count=error_count,
            ) from e
        except UnicodeDecodeError as e:
            raise ProtocolConfigurationError(
                f"Runtime config file contains binary or non-UTF-8 content: {config_path}",
                context=context,
                config_path=str(config_path),
                error_details=f"Encoding error at position {e.start}-{e.end}: {e.reason}",
            ) from e
        except OSError as e:
            raise ProtocolConfigurationError(
                f"Failed to read runtime config at {config_path}: {e}",
                context=context,
                config_path=str(config_path),
                error_details=str(e),
            ) from e

    # No config file - use environment variables and defaults
    logger.info(
        "No runtime config found at %s, using environment/defaults (correlation_id=%s)",
        config_path,
        effective_correlation_id,
    )
    config = ModelRuntimeConfig(
        input_topic=os.getenv("ONEX_INPUT_TOPIC", DEFAULT_INPUT_TOPIC),
        output_topic=os.getenv("ONEX_OUTPUT_TOPIC", DEFAULT_OUTPUT_TOPIC),
        consumer_group=os.getenv("ONEX_GROUP_ID", DEFAULT_GROUP_ID),
    )
    logger.debug(
        "Runtime config constructed from environment/defaults (correlation_id=%s)",
        effective_correlation_id,
        extra={
            "input_topic": config.input_topic,
            "output_topic": config.output_topic,
            "consumer_group": config.consumer_group,
        },
    )
    return config


async def bootstrap() -> int:
    """Bootstrap the ONEX runtime from contracts.

    This is the main async entrypoint that orchestrates the complete runtime
    initialization and lifecycle management. The bootstrap process follows a
    structured sequence to ensure proper resource initialization and cleanup.

    Bootstrap Sequence:
        1. Determine contracts directory from ONEX_CONTRACTS_DIR environment variable
        2. Load and validate runtime configuration from contracts or environment
        3. Create and initialize event bus (EventBusInmemory or EventBusKafka based on config)
        4. Create ModelONEXContainer and wire infrastructure services (async)
        5. Resolve RegistryProtocolBinding from container (async)
        6. Instantiate RuntimeHostProcess with validated configuration and pre-resolved registry
        7. Setup graceful shutdown signal handlers (SIGINT, SIGTERM)
        8. Start runtime and HTTP health server for Docker/Kubernetes health probes
        9. Run runtime until shutdown signal received
        10. Perform graceful shutdown with configurable timeout
        11. Clean up resources in finally block to prevent resource leaks

    Error Handling:
        - Configuration errors: Logged with full context and correlation_id
        - Runtime errors: Caught and logged with detailed error information
        - Unexpected errors: Logged with exception details for debugging
        - All errors include correlation_id for distributed tracing

    Shutdown Behavior:
        - Health server stopped first (fast, non-blocking operation)
        - Runtime stopped with configurable grace period (default: 30s)
        - Timeout enforcement prevents indefinite shutdown hangs
        - Finally block ensures cleanup even on unexpected errors

    Returns:
        Exit code (0 for success, non-zero for errors).
            - 0: Clean shutdown after successful operation
            - 1: Configuration error, runtime error, or unexpected failure

    Environment Variables:
        ONEX_CONTRACTS_DIR: Path to contracts directory (default: ./contracts)
        ONEX_HTTP_PORT: Port for health check server (default: 8085)
        ONEX_LOG_LEVEL: Logging level (default: INFO)
        ONEX_ENVIRONMENT: Environment name (default: local)
        ONEX_INPUT_TOPIC: Input topic override (default: requests)
        ONEX_OUTPUT_TOPIC: Output topic override (default: responses)
        ONEX_GROUP_ID: Consumer group override (default: onex-runtime)

    Example:
        >>> # Run bootstrap and handle exit code
        >>> exit_code = await bootstrap()
        >>> if exit_code == 0:
        ...     print("Runtime shutdown successfully")
        ... else:
        ...     print("Runtime encountered errors")

    Example Startup Log:
        ============================================================
        ONEX Runtime Kernel v0.1.0
        Environment: production
        Contracts: /app/contracts
        Event Bus: inmemory (group: onex-runtime)
        Topics: requests → responses
        Health endpoint: http://0.0.0.0:8085/health
        ============================================================
    """
    # Lazy import to break circular dependency chain - see "Circular Import Note"
    # comment near line 98 for detailed explanation of the import cycle.
    from omnibase_infra.services.service_health import (
        DEFAULT_HTTP_PORT,
        ServiceHealth,
    )

    # Initialize resources to None for cleanup guard in finally block
    runtime: RuntimeHostProcess | None = None
    health_server: ServiceHealth | None = None
    postgres_pool: asyncpg.Pool | None = None
    introspection_unsubscribe: Callable[[], Awaitable[None]] | None = None
    correlation_id = generate_correlation_id()
    bootstrap_start_time = time.time()

    try:
        # 1. Determine contracts directory
        contracts_dir = _get_contracts_dir()
        logger.info(
            "ONEX Kernel starting with contracts_dir=%s (correlation_id=%s)",
            contracts_dir,
            correlation_id,
        )

        # 2. Load runtime configuration (may raise ProtocolConfigurationError)
        # Pass correlation_id for consistent tracing across initialization sequence
        config_start_time = time.time()
        config = load_runtime_config(contracts_dir, correlation_id=correlation_id)
        config_duration = time.time() - config_start_time
        # Log only safe config fields (no credentials or sensitive data)
        # Full config.model_dump() could leak passwords, API keys, connection strings
        logger.debug(
            "Runtime config loaded in %.3fs (correlation_id=%s)",
            config_duration,
            correlation_id,
            extra={
                "duration_seconds": config_duration,
                "input_topic": config.input_topic,
                "output_topic": config.output_topic,
                "consumer_group": config.consumer_group,
                "event_bus_type": config.event_bus.type,
                "shutdown_grace_period": config.shutdown.grace_period_seconds,
            },
        )

        # 3. Create event bus
        # Dispatch based on configuration or environment variable:
        # - If KAFKA_BOOTSTRAP_SERVERS env var is set, use EventBusKafka
        # - If config.event_bus.type == "kafka", use EventBusKafka
        # - Otherwise, use EventBusInmemory for local development/testing
        # Environment override takes precedence over config for environment field.
        environment = os.getenv("ONEX_ENVIRONMENT") or config.event_bus.environment
        kafka_bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS")

        # Explicit bool evaluation (not truthy string) for kafka usage.
        # KAFKA_BOOTSTRAP_SERVERS env var takes precedence over config.event_bus.type.
        # This prevents implicit "kafka but localhost" fallback scenarios.
        use_kafka: bool = (
            bool(kafka_bootstrap_servers) or config.event_bus.type == "kafka"
        )

        # Validate bootstrap_servers is provided when kafka is requested via config
        # This prevents confusing implicit localhost:9092 fallback
        if use_kafka and not kafka_bootstrap_servers:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="configure_event_bus",
                correlation_id=correlation_id,
            )
            raise ProtocolConfigurationError(
                "Kafka event bus requested (config.event_bus.type='kafka') but "
                "KAFKA_BOOTSTRAP_SERVERS environment variable is not set. "
                "Set KAFKA_BOOTSTRAP_SERVERS to the broker address (e.g., 'kafka:9092') "
                "or use event_bus.type='inmemory' for local development.",
                context=context,
                parameter="KAFKA_BOOTSTRAP_SERVERS",
            )

        event_bus_start_time = time.time()
        event_bus: EventBusInmemory | EventBusKafka
        event_bus_type: str

        if use_kafka:
            # Use EventBusKafka for production/integration testing
            # NOTE: bootstrap_servers is guaranteed non-empty at this point due to validation
            # above, but mypy cannot narrow the Optional[str] type through control flow.
            kafka_config = ModelKafkaEventBusConfig(
                bootstrap_servers=kafka_bootstrap_servers,  # type: ignore[arg-type]  # NOTE: control flow narrowing limitation
                environment=environment,
                circuit_breaker_threshold=config.event_bus.circuit_breaker_threshold,
            )
            event_bus = EventBusKafka(config=kafka_config)
            event_bus_type = "kafka"

            # Start EventBusKafka to connect to Kafka/Redpanda and enable consumers
            # Without this, the event bus cannot publish or consume messages
            try:
                await event_bus.start()
                logger.debug(
                    "EventBusKafka started successfully (correlation_id=%s)",
                    correlation_id,
                )
            except Exception as e:
                context = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.KAFKA,
                    operation="start_event_bus",
                    correlation_id=correlation_id,
                    target_name=kafka_bootstrap_servers,
                )
                raise RuntimeHostError(
                    f"Failed to start EventBusKafka: {sanitize_error_message(e)}",
                    context=context,
                ) from e

            logger.info(
                "Using EventBusKafka (correlation_id=%s)",
                correlation_id,
                extra={
                    "bootstrap_servers": kafka_bootstrap_servers,
                    "environment": environment,
                    "consumer_group": config.consumer_group,
                },
            )
        else:
            # Use EventBusInmemory for local development/testing
            event_bus = EventBusInmemory(
                environment=environment,
                group=config.consumer_group,
            )
            event_bus_type = "inmemory"

        event_bus_duration = time.time() - event_bus_start_time
        logger.debug(
            "Event bus created in %.3fs (correlation_id=%s)",
            event_bus_duration,
            correlation_id,
            extra={
                "duration_seconds": event_bus_duration,
                "event_bus_type": event_bus_type,
                "environment": environment,
                "consumer_group": config.consumer_group,
            },
        )

        # 4. Create and wire container for dependency injection
        container_start_time = time.time()
        container = ModelONEXContainer()
        if container.service_registry is None:
            logger.warning(
                "DEGRADED_MODE: service_registry is None (omnibase_core circular import bug?), "
                "skipping container wiring (correlation_id=%s)",
                correlation_id,
                extra={
                    "error_type": "NoneType",
                    "correlation_id": correlation_id,
                    "degraded_mode": True,
                    "degraded_reason": "service_registry_unavailable",
                    "component": "container_wiring",
                },
            )
            wire_summary: dict[str, list[str] | str] = {
                "services": [],
                "status": "degraded",
            }  # Empty summary for degraded mode
        else:
            try:
                wire_summary = await wire_infrastructure_services(container)
            except ServiceResolutionError as e:
                # Service resolution failed during wiring - container configuration issue.
                logger.warning(
                    "DEGRADED_MODE: Container wiring failed due to service resolution error, "
                    "continuing in degraded mode (correlation_id=%s): %s",
                    correlation_id,
                    e,
                    extra={
                        "error_type": type(e).__name__,
                        "correlation_id": correlation_id,
                        "degraded_mode": True,
                        "degraded_reason": "service_resolution_error",
                        "component": "container_wiring",
                    },
                )
                wire_summary = {"services": [], "status": "degraded"}
            except (RuntimeError, AttributeError) as e:
                # Unexpected error during wiring - container internals issue.
                logger.warning(
                    "DEGRADED_MODE: Container wiring failed with unexpected error, "
                    "continuing in degraded mode (correlation_id=%s): %s",
                    correlation_id,
                    e,
                    extra={
                        "error_type": type(e).__name__,
                        "correlation_id": correlation_id,
                        "degraded_mode": True,
                        "degraded_reason": "wiring_error",
                        "component": "container_wiring",
                    },
                )
                wire_summary = {"services": [], "status": "degraded"}
        container_duration = time.time() - container_start_time
        logger.debug(
            "Container wired in %.3fs (correlation_id=%s)",
            container_duration,
            correlation_id,
            extra={
                "duration_seconds": container_duration,
                "services": wire_summary["services"],
            },
        )

        # 4.5. Create PostgreSQL pool for projections
        # Only create if POSTGRES_HOST is set (indicates registration should be enabled)
        projector: ProjectorShell | None = None
        introspection_dispatcher: DispatcherNodeIntrospected | None = None
        consul_handler = None  # Will be initialized if Consul is configured

        postgres_host = os.getenv("POSTGRES_HOST")
        if postgres_host:
            postgres_pool_start_time = time.time()
            try:
                postgres_pool = await asyncpg.create_pool(
                    user=os.getenv("POSTGRES_USER", "postgres"),
                    password=os.getenv("POSTGRES_PASSWORD", ""),
                    host=postgres_host,
                    port=int(os.getenv("POSTGRES_PORT", "5432")),
                    database=os.getenv("POSTGRES_DATABASE", "omninode_bridge"),
                    min_size=2,
                    max_size=10,
                )
                postgres_pool_duration = time.time() - postgres_pool_start_time
                logger.info(
                    "PostgreSQL pool created in %.3fs (correlation_id=%s)",
                    postgres_pool_duration,
                    correlation_id,
                    extra={
                        "host": postgres_host,
                        "port": os.getenv("POSTGRES_PORT", "5432"),
                        "database": os.getenv("POSTGRES_DATABASE", "omninode_bridge"),
                    },
                )

                # 4.6. Load projectors from contracts via ProjectorPluginLoader (OMN-1170/1169)
                #
                # This section implements fully contract-driven projector management. The
                # loader discovers projector contracts from the package's projectors/contracts
                # directory and creates ProjectorShell instances for runtime use.
                #
                # Contract-driven approach:
                # - Projector behavior defined in YAML contracts (registration_projector.yaml)
                # - ProjectorPluginLoader discovers and loads projectors from contracts
                # - ProjectorShell provides generic projection operations (project, partial_update)
                # - Schema initialization is decoupled - SQL executed directly from schema file
                #
                # The registration projector is identified by projector_id="registration-projector"
                # and is passed to wire_registration_handlers() for handler injection.
                #
                projector_contracts_dir = (
                    Path(__file__).parent.parent / "projectors" / "contracts"
                )

                # Try to discover projectors from contracts
                projector_loader = ProjectorPluginLoader(
                    config=ModelProjectorPluginLoaderConfig(graceful_mode=True),
                    container=container,
                    pool=postgres_pool,
                )

                discovered_projectors: list[ProtocolEventProjector] = []
                if projector_contracts_dir.exists():
                    try:
                        discovered_projectors = (
                            await projector_loader.load_from_directory(
                                projector_contracts_dir
                            )
                        )
                        if discovered_projectors:
                            logger.info(
                                "Discovered %d projector(s) from contracts (correlation_id=%s)",
                                len(discovered_projectors),
                                correlation_id,
                                extra={
                                    "discovered_count": len(discovered_projectors),
                                    "contracts_dir": str(projector_contracts_dir),
                                    "projector_ids": [
                                        getattr(p, "projector_id", "unknown")
                                        for p in discovered_projectors
                                    ],
                                },
                            )
                        else:
                            logger.warning(
                                "No projector contracts found in %s (correlation_id=%s)",
                                projector_contracts_dir,
                                correlation_id,
                                extra={
                                    "contracts_dir": str(projector_contracts_dir),
                                },
                            )
                    except Exception as discovery_error:
                        # Log warning but continue - projector discovery is best-effort
                        # Registration features will be unavailable if discovery fails
                        logger.warning(
                            "Projector contract discovery failed: %s (correlation_id=%s)",
                            sanitize_error_message(discovery_error),
                            correlation_id,
                            extra={
                                "error_type": type(discovery_error).__name__,
                                "contracts_dir": str(projector_contracts_dir),
                            },
                        )
                else:
                    logger.debug(
                        "Projector contracts directory not found, skipping discovery "
                        "(correlation_id=%s)",
                        correlation_id,
                        extra={
                            "contracts_dir": str(projector_contracts_dir),
                        },
                    )

                # Extract registration projector from discovered projectors (OMN-1169)
                # This replaces the legacy ProjectorRegistration with contract-loaded ProjectorShell
                registration_projector_id = "registration-projector"
                for discovered in discovered_projectors:
                    if (
                        getattr(discovered, "projector_id", None)
                        == registration_projector_id
                    ):
                        # Cast to ProjectorShell (loader creates ProjectorShell when pool provided)
                        if isinstance(discovered, ProjectorShell):
                            projector = discovered
                            logger.info(
                                "Using contract-loaded ProjectorShell for registration "
                                "(correlation_id=%s)",
                                correlation_id,
                                extra={
                                    "projector_id": registration_projector_id,
                                    "aggregate_type": projector.aggregate_type,
                                },
                            )
                        break

                if projector is None:
                    # Fallback: No registration projector discovered from contracts
                    logger.warning(
                        "Registration projector not found in contracts, "
                        "registration features will be unavailable (correlation_id=%s)",
                        correlation_id,
                        extra={
                            "expected_projector_id": registration_projector_id,
                            "discovered_count": len(discovered_projectors),
                        },
                    )

                # Initialize schema by executing SQL file directly
                # Schema initialization is decoupled from the projector - it just ensures
                # the table and indexes exist. The ProjectorShell uses the schema at runtime.
                schema_file = (
                    Path(__file__).parent.parent
                    / "schemas"
                    / "schema_registration_projection.sql"
                )
                if schema_file.exists():
                    try:
                        schema_sql = schema_file.read_text()
                        async with postgres_pool.acquire() as conn:
                            await conn.execute(schema_sql)
                        logger.info(
                            "Registration projection schema initialized (correlation_id=%s)",
                            correlation_id,
                        )
                    except Exception as schema_error:
                        # Log warning but continue - schema may already exist
                        logger.warning(
                            "Schema initialization encountered error: %s (correlation_id=%s)",
                            sanitize_error_message(schema_error),
                            correlation_id,
                            extra={
                                "error_type": type(schema_error).__name__,
                            },
                        )
                else:
                    logger.warning(
                        "Schema file not found: %s (correlation_id=%s)",
                        schema_file,
                        correlation_id,
                    )

                # 4.6.5. Initialize HandlerConsul if Consul is configured
                # CONSUL_HOST determines whether to enable Consul registration
                consul_host = os.getenv("CONSUL_HOST")
                if consul_host:
                    # Validate CONSUL_PORT environment variable
                    consul_port_str = os.getenv("CONSUL_PORT", "8500")
                    try:
                        consul_port = int(consul_port_str)
                        if not MIN_PORT <= consul_port <= MAX_PORT:
                            logger.warning(
                                "CONSUL_PORT %d outside valid range %d-%d, using default 8500 (correlation_id=%s)",
                                consul_port,
                                MIN_PORT,
                                MAX_PORT,
                                correlation_id,
                            )
                            consul_port = 8500
                    except ValueError:
                        logger.warning(
                            "Invalid CONSUL_PORT value '%s', using default 8500 (correlation_id=%s)",
                            consul_port_str,
                            correlation_id,
                        )
                        consul_port = 8500

                    try:
                        # Deferred import: Only load HandlerConsul when Consul is configured.
                        # This avoids loading the consul dependency (and its transitive deps)
                        # when Consul integration is disabled, reducing startup time.
                        from omnibase_infra.handlers import HandlerConsul

                        consul_handler = HandlerConsul(container)
                        await consul_handler.initialize(
                            {
                                "host": consul_host,
                                "port": consul_port,
                            }
                        )
                        logger.info(
                            "HandlerConsul initialized for dual registration (correlation_id=%s)",
                            correlation_id,
                            extra={
                                "consul_host": consul_host,
                                "consul_port": consul_port,
                            },
                        )
                    except Exception as consul_error:
                        # Log warning but continue without Consul (PostgreSQL is source of truth)
                        # Use sanitize_error_message to prevent credential leakage in logs
                        logger.warning(
                            "Failed to initialize HandlerConsul, proceeding without Consul: %s (correlation_id=%s)",
                            sanitize_error_message(consul_error),
                            correlation_id,
                            extra={
                                "error_type": type(consul_error).__name__,
                            },
                        )
                        consul_handler = None
                else:
                    logger.debug(
                        "CONSUL_HOST not set, Consul registration disabled (correlation_id=%s)",
                        correlation_id,
                    )

                # 4.7. Wire registration handlers with projector and consul_handler
                registration_summary = await wire_registration_handlers(
                    container,
                    postgres_pool,
                    projector=projector,
                    consul_handler=consul_handler,
                )
                logger.info(
                    "Registration handlers wired (correlation_id=%s)",
                    correlation_id,
                    extra={
                        "services": registration_summary["services"],
                    },
                )

                # 4.8. Create introspection dispatcher for routing events
                # Deferred import: HandlerNodeIntrospected depends on PostgreSQL and
                # registration infrastructure. Only loaded after PostgreSQL pool is
                # successfully created and registration handlers are wired.
                from omnibase_infra.nodes.node_registration_orchestrator.handlers import (
                    HandlerNodeIntrospected,
                )

                # Check if service_registry is available (may be None in omnibase_core 0.6.x)
                if container.service_registry is None:
                    logger.warning(
                        "DEGRADED_MODE: ServiceRegistry not available, skipping introspection dispatcher creation (correlation_id=%s)",
                        correlation_id,
                        extra={
                            "error_type": "NoneType",
                            "correlation_id": correlation_id,
                            "degraded_mode": True,
                            "degraded_reason": "service_registry_unavailable",
                            "component": "introspection_dispatcher",
                        },
                    )
                    # Set introspection_dispatcher to None and continue without it
                    introspection_dispatcher = None
                else:
                    logger.debug(
                        "Resolving HandlerNodeIntrospected from container (correlation_id=%s)",
                        correlation_id,
                    )
                    handler_introspected: HandlerNodeIntrospected = (
                        await container.service_registry.resolve_service(
                            HandlerNodeIntrospected
                        )
                    )
                    logger.debug(
                        "HandlerNodeIntrospected resolved successfully (correlation_id=%s)",
                        correlation_id,
                        extra={
                            "handler_class": handler_introspected.__class__.__name__,
                        },
                    )

                    introspection_dispatcher = DispatcherNodeIntrospected(
                        handler_introspected
                    )
                    logger.info(
                        "Introspection dispatcher created and wired (correlation_id=%s)",
                        correlation_id,
                        extra={
                            "dispatcher_class": introspection_dispatcher.__class__.__name__,
                            "handler_class": handler_introspected.__class__.__name__,
                        },
                    )

            except Exception as pool_error:
                # Log warning but continue without registration support
                # Use sanitize_error_message to prevent credential leakage in logs
                # (PostgreSQL connection errors may include DSN with password)
                logger.warning(
                    "Failed to initialize PostgreSQL pool for registration: %s (correlation_id=%s)",
                    sanitize_error_message(pool_error),
                    correlation_id,
                    extra={
                        "error_type": type(pool_error).__name__,
                    },
                )
                if postgres_pool is not None:
                    try:
                        await postgres_pool.close()
                    except Exception as cleanup_error:
                        # Sanitize cleanup errors to prevent credential leakage
                        # NOTE: Do NOT use exc_info=True here - tracebacks may contain
                        # connection strings with credentials from PostgreSQL errors
                        logger.warning(
                            "Cleanup failed for PostgreSQL pool close: %s (correlation_id=%s)",
                            sanitize_error_message(cleanup_error),
                            correlation_id,
                        )
                    postgres_pool = None
                projector = None
                introspection_dispatcher = None
        else:
            logger.debug(
                "POSTGRES_HOST not set, skipping registration handler wiring (correlation_id=%s)",
                correlation_id,
            )

        # 5. Resolve RegistryProtocolBinding from container or create new instance
        # NOTE: Fallback to creating new instance is intentional degraded mode behavior.
        # The handler registry is optional for basic runtime operation - core event
        # processing continues even without explicit handler bindings. However,
        # ProtocolConfigurationError should NOT be masked as it indicates invalid
        # configuration that would cause undefined behavior.
        handler_registry: RegistryProtocolBinding | None = None

        # Check if service_registry is available (may be None in omnibase_core 0.6.x)
        if container.service_registry is not None:
            try:
                handler_registry = await container.service_registry.resolve_service(
                    RegistryProtocolBinding
                )
            except ServiceResolutionError as e:
                # Service not registered - expected in minimal configurations.
                # Create a new instance directly as fallback.
                logger.warning(
                    "DEGRADED_MODE: RegistryProtocolBinding not registered in container, "
                    "creating new instance (correlation_id=%s): %s",
                    correlation_id,
                    e,
                    extra={
                        "error_type": type(e).__name__,
                        "correlation_id": correlation_id,
                        "degraded_mode": True,
                        "degraded_reason": "service_not_registered",
                        "component": "handler_registry",
                    },
                )
                handler_registry = RegistryProtocolBinding()
            except (RuntimeError, AttributeError) as e:
                # Unexpected resolution failure - container internals issue.
                # Log with more diagnostic context but still allow degraded operation.
                logger.warning(
                    "DEGRADED_MODE: Unexpected error resolving RegistryProtocolBinding, "
                    "creating new instance (correlation_id=%s): %s",
                    correlation_id,
                    e,
                    extra={
                        "error_type": type(e).__name__,
                        "correlation_id": correlation_id,
                        "degraded_mode": True,
                        "degraded_reason": "resolution_error",
                        "component": "handler_registry",
                    },
                )
                handler_registry = RegistryProtocolBinding()
            # NOTE: ProtocolConfigurationError is NOT caught here - configuration
            # errors should propagate and stop startup to prevent undefined behavior.
        else:
            # ServiceRegistry not available, create a new RegistryProtocolBinding directly
            logger.warning(
                "DEGRADED_MODE: ServiceRegistry not available, creating RegistryProtocolBinding directly (correlation_id=%s)",
                correlation_id,
                extra={
                    "error_type": "NoneType",
                    "correlation_id": correlation_id,
                    "degraded_mode": True,
                    "degraded_reason": "service_registry_unavailable",
                    "component": "handler_registry",
                },
            )
            handler_registry = RegistryProtocolBinding()

        # 6. Create runtime host process with config and pre-resolved registry
        # RuntimeHostProcess accepts config as dict; cast model_dump() result to
        # dict[str, object] to avoid implicit Any typing (Pydantic's model_dump()
        # returns dict[str, Any] but all our model fields are strongly typed)
        #
        # NOTE: RuntimeHostProcess expects 'service_name' and 'node_name' keys,
        # but ModelRuntimeConfig uses 'name'. Map 'name' -> 'service_name'/'node_name'
        # for compatibility. (OMN-1602)
        #
        # INVARIANT: In the current runtime model, `ModelRuntimeConfig.name` represents
        # both `service_name` and `node_name` by design; multi-node services require
        # schema expansion.
        #
        # TRIGGER FOR SPLIT: Split when ServiceKernel supports registering multiple
        # node contracts under one service runtime.
        #
        # Why both fields get the same value:
        # - For services using simplified config with just 'name', there's no semantic
        #   distinction between service and node - a single service hosts a single node
        # - RuntimeHostProcess uses these to construct ModelNodeIdentity for Kafka
        #   consumer group IDs and event routing
        # - The introspection consumer group format is:
        #   {env}.{service_name}.{node_name}.{purpose}.{version}
        #   e.g., "local.my-service.my-service.introspection.v1"
        # - When service_name == node_name, the format is intentionally redundant but
        #   maintains consistency with multi-node deployments where they would differ
        runtime_create_start_time = time.time()
        runtime_config_dict = cast("dict[str, object]", config.model_dump())
        if config.name:
            runtime_config_dict["service_name"] = config.name
            runtime_config_dict["node_name"] = config.name
        runtime = RuntimeHostProcess(
            container=container,
            event_bus=event_bus,
            input_topic=config.input_topic,
            output_topic=config.output_topic,
            config=runtime_config_dict,
            handler_registry=handler_registry,
            # Pass contracts directory for handler discovery (OMN-1317)
            # This enables contract-based handler registration instead of
            # falling back to wire_handlers() with an empty registry
            contract_paths=[str(contracts_dir)],
        )
        runtime_create_duration = time.time() - runtime_create_start_time
        logger.debug(
            "Runtime host process created in %.3fs (correlation_id=%s)",
            runtime_create_duration,
            correlation_id,
            extra={
                "duration_seconds": runtime_create_duration,
                "input_topic": config.input_topic,
                "output_topic": config.output_topic,
            },
        )

        # 7. Setup graceful shutdown
        shutdown_event = asyncio.Event()
        loop = asyncio.get_running_loop()

        def handle_shutdown(sig: signal.Signals) -> None:
            """Handle shutdown signal with correlation tracking."""
            logger.info(
                "Received %s, initiating graceful shutdown... (correlation_id=%s)",
                sig.name,
                correlation_id,
            )
            shutdown_event.set()

        # Register signal handlers for graceful shutdown
        if sys.platform != "win32":
            # Unix: Use asyncio's signal handler for proper event loop integration
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, handle_shutdown, sig)
        else:
            # Windows: asyncio signal handlers not supported, use signal.signal()
            # for SIGINT (Ctrl+C). Note: SIGTERM not available on Windows.
            #
            # Thread-safety: On Windows, signal.signal() handlers execute in a
            # different thread than the event loop. While asyncio.Event.set() is
            # documented as thread-safe, we use loop.call_soon_threadsafe() to
            # schedule the set() call on the event loop thread. This ensures
            # proper cross-thread communication and avoids potential race
            # conditions with any event loop state inspection.
            def windows_handler(signum: int, frame: object) -> None:
                """Windows-compatible signal handler wrapper.

                Uses call_soon_threadsafe to safely communicate with the event
                loop from the signal handler thread.
                """
                sig = signal.Signals(signum)
                logger.info(
                    "Received %s, initiating graceful shutdown... (correlation_id=%s)",
                    sig.name,
                    correlation_id,
                )
                loop.call_soon_threadsafe(shutdown_event.set)

            signal.signal(signal.SIGINT, windows_handler)

        # 8. Start runtime and health server
        runtime_start_time = time.time()
        logger.info(
            "Starting ONEX runtime... (correlation_id=%s)",
            correlation_id,
        )
        await runtime.start()
        runtime_start_duration = time.time() - runtime_start_time
        logger.debug(
            "Runtime started in %.3fs (correlation_id=%s)",
            runtime_start_duration,
            correlation_id,
            extra={
                "duration_seconds": runtime_start_duration,
            },
        )

        # 9. Start HTTP health server for Docker/K8s probes
        # Port can be configured via ONEX_HTTP_PORT environment variable
        http_port_str = os.getenv("ONEX_HTTP_PORT", str(DEFAULT_HTTP_PORT))
        try:
            http_port = int(http_port_str)
            if not MIN_PORT <= http_port <= MAX_PORT:
                logger.warning(
                    "ONEX_HTTP_PORT %d outside valid range %d-%d, using default %d (correlation_id=%s)",
                    http_port,
                    MIN_PORT,
                    MAX_PORT,
                    DEFAULT_HTTP_PORT,
                    correlation_id,
                )
                http_port = DEFAULT_HTTP_PORT
        except ValueError:
            logger.warning(
                "Invalid ONEX_HTTP_PORT value '%s', using default %d (correlation_id=%s)",
                http_port_str,
                DEFAULT_HTTP_PORT,
                correlation_id,
            )
            http_port = DEFAULT_HTTP_PORT

        health_server = ServiceHealth(
            container=container,
            runtime=runtime,
            port=http_port,
            version=KERNEL_VERSION,
        )
        health_start_time = time.time()
        await health_server.start()
        health_start_duration = time.time() - health_start_time
        logger.debug(
            "Health server started in %.3fs (correlation_id=%s)",
            health_start_duration,
            correlation_id,
            extra={
                "duration_seconds": health_start_duration,
                "port": http_port,
            },
        )

        # 9.5. Start introspection event consumer if dispatcher is available
        # This consumer subscribes to the input topic and routes introspection
        # events to the HandlerNodeIntrospected via DispatcherNodeIntrospected.
        # Unlike RuntimeHostProcess which routes based on handler_type field,
        # this consumer directly parses introspection events from JSON.
        #
        # The message handler is extracted to IntrospectionMessageHandler for
        # better testability and separation of concerns (PR #101 code quality).
        #
        # Duck typing approach per CLAUDE.md architectural guidelines:
        # Check for subscribe() capability via hasattr/callable instead of isinstance.
        # This enables any event bus implementing subscribe() to participate in
        # introspection event consumption, following protocol-based polymorphism.
        #
        # Production considerations:
        # - EventBusKafka: Uses distributed consumer groups for production workloads
        # - EventBusInmemory: subscribe() works for testing scenarios
        # - Other implementations: Will work if they implement subscribe()
        #
        # The duck typing approach allows new event bus implementations to
        # participate in introspection without modifying this code.
        has_subscribe = hasattr(event_bus, "subscribe") and callable(
            getattr(event_bus, "subscribe", None)
        )
        if introspection_dispatcher is not None and has_subscribe:
            # Create extracted event router with container-based DI pattern
            # Dependencies are passed explicitly since they are created at runtime
            # by the kernel and may not be registered in the container yet
            introspection_event_router = IntrospectionEventRouter(
                container=container,
                output_topic=config.output_topic,
                dispatcher=introspection_dispatcher,
                event_bus=event_bus,
            )

            # Create typed node identity for introspection subscription (OMN-1602)
            # Uses ModelNodeIdentity + EnumConsumerGroupPurpose instead of hardcoded
            # group_id suffix hack for proper semantic consumer group naming.
            #
            # Required fields from config (fail-fast if missing):
            # - service_name: from config.name (required for node identification)
            # - node_name: from config.name (required for node identification)
            # Optional with defaults:
            # - env: from environment variable or event_bus.environment
            # - version: from config.contract_version or "v1"
            if not config.name:
                context = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="create_node_identity",
                    correlation_id=correlation_id,
                )
                raise ProtocolConfigurationError(
                    "Runtime config requires 'name' field for service identification. "
                    "Add 'name: your-service-name' to runtime_config.yaml. "
                    "This is required for typed introspection subscription (OMN-1602).",
                    context=context,
                    parameter="name",
                )

            introspection_node_identity = ModelNodeIdentity(
                env=environment,
                service=config.name,
                node_name=config.name,
                version=config.contract_version or "v1",
            )

            # Subscribe with callback - returns unsubscribe function
            subscribe_start_time = time.time()
            logger.info(
                "Subscribing to introspection events on event bus (correlation_id=%s)",
                correlation_id,
                extra={
                    "topic": config.input_topic,
                    "node_identity": {
                        "env": introspection_node_identity.env,
                        "service": introspection_node_identity.service,
                        "node_name": introspection_node_identity.node_name,
                        "version": introspection_node_identity.version,
                    },
                    "purpose": EnumConsumerGroupPurpose.INTROSPECTION.value,
                    "event_bus_type": event_bus_type,
                },
            )

            introspection_unsubscribe = await event_bus.subscribe(
                topic=config.input_topic,
                node_identity=introspection_node_identity,
                on_message=introspection_event_router.handle_message,
                purpose=EnumConsumerGroupPurpose.INTROSPECTION,
            )
            subscribe_duration = time.time() - subscribe_start_time

            logger.info(
                "Introspection event consumer started successfully in %.3fs (correlation_id=%s)",
                subscribe_duration,
                correlation_id,
                extra={
                    "topic": config.input_topic,
                    "node_identity": {
                        "env": introspection_node_identity.env,
                        "service": introspection_node_identity.service,
                        "node_name": introspection_node_identity.node_name,
                        "version": introspection_node_identity.version,
                    },
                    "purpose": EnumConsumerGroupPurpose.INTROSPECTION.value,
                    "subscribe_duration_seconds": subscribe_duration,
                    "event_bus_type": event_bus_type,
                },
            )

        # Calculate total bootstrap time
        bootstrap_duration = time.time() - bootstrap_start_time

        # Display startup banner with key configuration
        if introspection_dispatcher is not None:
            if consul_handler is not None:
                registration_status = "enabled (PostgreSQL + Consul)"
            else:
                registration_status = "enabled (PostgreSQL only)"
        else:
            registration_status = "disabled"
        banner_lines = [
            "=" * 60,
            f"ONEX Runtime Kernel v{KERNEL_VERSION}",
            f"Environment: {environment}",
            f"Contracts: {contracts_dir}",
            f"Event Bus: {event_bus_type} (group: {config.consumer_group})",
            f"Topics: {config.input_topic} → {config.output_topic}",
            f"Registration: {registration_status}",
            f"Health endpoint: http://0.0.0.0:{http_port}/health",
            f"Bootstrap time: {bootstrap_duration:.3f}s",
            f"Correlation ID: {correlation_id}",
            "=" * 60,
        ]
        banner = "\n".join(banner_lines)
        logger.info("\n%s", banner)

        logger.info(
            "ONEX runtime started successfully in %.3fs (correlation_id=%s)",
            bootstrap_duration,
            correlation_id,
            extra={
                "bootstrap_duration_seconds": bootstrap_duration,
                "config_load_seconds": config_duration,
                "event_bus_create_seconds": event_bus_duration,
                "container_wire_seconds": container_duration,
                "runtime_create_seconds": runtime_create_duration,
                "runtime_start_seconds": runtime_start_duration,
                "health_start_seconds": health_start_duration,
            },
        )

        # Wait for shutdown signal
        await shutdown_event.wait()

        grace_period = config.shutdown.grace_period_seconds
        shutdown_start_time = time.time()
        logger.info(
            "Shutdown signal received, stopping runtime (timeout=%ss, correlation_id=%s)",
            grace_period,
            correlation_id,
        )

        # Stop introspection consumer first (fast)
        if introspection_unsubscribe is not None:
            try:
                await introspection_unsubscribe()
                logger.debug(
                    "Introspection consumer stopped (correlation_id=%s)",
                    correlation_id,
                )
            except Exception as consumer_stop_error:
                logger.warning(
                    "Failed to stop introspection consumer: %s (correlation_id=%s)",
                    sanitize_error_message(consumer_stop_error),
                    correlation_id,
                )
            introspection_unsubscribe = None

        # Stop health server (fast, non-blocking)
        if health_server is not None:
            try:
                health_stop_start_time = time.time()
                await health_server.stop()
                health_stop_duration = time.time() - health_stop_start_time
                logger.debug(
                    "Health server stopped in %.3fs (correlation_id=%s)",
                    health_stop_duration,
                    correlation_id,
                    extra={
                        "duration_seconds": health_stop_duration,
                    },
                )
            except Exception as health_stop_error:
                logger.warning(
                    "Failed to stop health server: %s (correlation_id=%s)",
                    health_stop_error,
                    correlation_id,
                    extra={
                        "error_type": type(health_stop_error).__name__,
                    },
                )
            health_server = None

        # Stop runtime with timeout
        try:
            runtime_stop_start_time = time.time()
            await asyncio.wait_for(runtime.stop(), timeout=grace_period)
            runtime_stop_duration = time.time() - runtime_stop_start_time
            logger.debug(
                "Runtime stopped in %.3fs (correlation_id=%s)",
                runtime_stop_duration,
                correlation_id,
                extra={
                    "duration_seconds": runtime_stop_duration,
                },
            )
        except TimeoutError:
            logger.warning(
                "Graceful shutdown timed out after %s seconds, forcing stop (correlation_id=%s)",
                grace_period,
                correlation_id,
            )
        runtime = None  # Mark as stopped to prevent double-stop in finally

        # Close PostgreSQL pool
        if postgres_pool is not None:
            try:
                pool_close_start_time = time.time()
                await postgres_pool.close()
                pool_close_duration = time.time() - pool_close_start_time
                logger.debug(
                    "PostgreSQL pool closed in %.3fs (correlation_id=%s)",
                    pool_close_duration,
                    correlation_id,
                )
            except Exception as pool_close_error:
                # Sanitize to prevent credential leakage
                logger.warning(
                    "Failed to close PostgreSQL pool: %s (correlation_id=%s)",
                    sanitize_error_message(pool_close_error),
                    correlation_id,
                )
            postgres_pool = None

        shutdown_duration = time.time() - shutdown_start_time
        logger.info(
            "ONEX runtime stopped successfully in %.3fs (correlation_id=%s)",
            shutdown_duration,
            correlation_id,
            extra={
                "shutdown_duration_seconds": shutdown_duration,
            },
        )
        return 0

    except ProtocolConfigurationError as e:
        # Configuration errors already have proper context and chaining
        error_code = getattr(getattr(e, "model", None), "error_code", None)
        error_code_name = getattr(error_code, "name", None)
        logger.exception(
            "ONEX runtime configuration failed (correlation_id=%s)",
            correlation_id,
            extra={
                "error_type": type(e).__name__,
                "error_code": str(error_code_name)
                if error_code_name is not None
                else None,
            },
        )
        return 1

    except RuntimeHostError as e:
        # Runtime host errors already have proper structure
        error_code = getattr(getattr(e, "model", None), "error_code", None)
        error_code_name = getattr(error_code, "name", None)
        logger.exception(
            "ONEX runtime host error (correlation_id=%s)",
            correlation_id,
            extra={
                "error_type": type(e).__name__,
                "error_code": str(error_code_name)
                if error_code_name is not None
                else None,
            },
        )
        return 1

    except Exception as e:
        # Unexpected errors: log with full context and return error code
        # (consistent with ProtocolConfigurationError and RuntimeHostError handlers)
        # Sanitize error message to prevent credential leakage
        logger.exception(
            "ONEX runtime failed with unexpected error: %s (correlation_id=%s)",
            sanitize_error_message(e),
            correlation_id,
            extra={
                "error_type": type(e).__name__,
            },
        )
        return 1

    finally:
        # Guard cleanup - stop all resources if not already stopped
        # Order: introspection consumer -> health server -> runtime -> pool

        if introspection_unsubscribe is not None:
            try:
                await introspection_unsubscribe()
            except Exception as cleanup_error:
                logger.warning(
                    "Failed to stop introspection consumer during cleanup: %s (correlation_id=%s)",
                    sanitize_error_message(cleanup_error),
                    correlation_id,
                )

        if health_server is not None:
            try:
                await health_server.stop()
            except Exception as cleanup_error:
                logger.warning(
                    "Failed to stop health server during cleanup: %s (correlation_id=%s)",
                    sanitize_error_message(cleanup_error),
                    correlation_id,
                )

        if runtime is not None:
            try:
                await runtime.stop()
            except Exception as cleanup_error:
                # Log cleanup failures with context instead of suppressing them
                # Sanitize to prevent potential credential leakage from runtime errors
                logger.warning(
                    "Failed to stop runtime during cleanup: %s (correlation_id=%s)",
                    sanitize_error_message(cleanup_error),
                    correlation_id,
                )

        if postgres_pool is not None:
            try:
                await postgres_pool.close()
            except Exception as cleanup_error:
                # Sanitize to prevent credential leakage from PostgreSQL errors
                logger.warning(
                    "Failed to close PostgreSQL pool during cleanup: %s (correlation_id=%s)",
                    sanitize_error_message(cleanup_error),
                    correlation_id,
                )


def configure_logging() -> None:
    """Configure logging for the kernel with structured format.

    Sets up structured logging with appropriate log level from the
    ONEX_LOG_LEVEL environment variable (default: INFO). This function
    must be called early in the bootstrap process to ensure logging
    is available for all subsequent operations.

    Logging Configuration:
        - Log Level: Controlled by ONEX_LOG_LEVEL environment variable
        - Format: Timestamp, level, logger name, message, extras
        - Date Format: ISO-8601 compatible (YYYY-MM-DD HH:MM:SS)
        - Structured Extras: Support for correlation_id and custom fields

    Bootstrap Order Rationale:
        This function is called BEFORE runtime config is loaded because logging
        must be available during config loading itself (to log errors, warnings,
        and info about config discovery). Therefore, logging configuration uses
        environment variables rather than contract-based config values.

        This is a deliberate chicken-and-egg solution:
        - Environment variables control early bootstrap logging
        - Contract config controls runtime behavior after bootstrap

    Environment Variables:
        ONEX_LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            Default: INFO

    Log Format Example:
        2025-01-15 10:30:45 [INFO] omnibase_infra.runtime.service_kernel: ONEX Kernel v0.1.0
        2025-01-15 10:30:45 [DEBUG] omnibase_infra.runtime.service_kernel: Runtime config loaded
            (correlation_id=123e4567-e89b-12d3-a456-426614174000)

    Structured Logging Extras:
        All log calls support structured extras for observability:
        - correlation_id: UUID for distributed tracing
        - duration_seconds: Operation timing metrics
        - error_type: Exception class name for error analysis
        - Custom fields: Any JSON-serializable data

    Example:
        >>> configure_logging()
        >>> logger.info("Operation completed", extra={"duration_seconds": 1.234})
    """
    log_level = os.getenv("ONEX_LOG_LEVEL", "INFO").upper()

    # Validate log level and provide helpful error if invalid
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if log_level not in valid_levels:
        print(
            f"Warning: Invalid ONEX_LOG_LEVEL '{log_level}', using INFO. "
            f"Valid levels: {', '.join(sorted(valid_levels))}",
            file=sys.stderr,
        )
        log_level = "INFO"

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    """Entry point for the ONEX runtime kernel.

    This is the synchronous entry point for the kernel. It configures
    logging, initiates the async bootstrap process, and handles the
    final exit code.

    Execution Flow:
        1. Configure logging from environment variables
        2. Log kernel version for startup identification
        3. Run async bootstrap function in event loop
        4. Exit with appropriate exit code (0=success, 1=error)

    Exit Codes:
        0: Successful startup and clean shutdown
        1: Configuration error, runtime error, or unexpected failure

    This function is the target for:
        - The installed entrypoint: `onex-runtime`
        - Direct module execution: `python -m omnibase_infra.runtime.service_kernel`
        - Docker CMD/ENTRYPOINT in container deployments

    Example:
        >>> # From command line
        >>> python -m omnibase_infra.runtime.service_kernel
        >>> # Or via installed entrypoint
        >>> onex-runtime

    Docker Usage:
        CMD ["onex-runtime"]
        # Container will start runtime and expose health endpoint
    """
    configure_logging()
    logger.info("ONEX Kernel v%s initializing...", KERNEL_VERSION)
    exit_code = asyncio.run(bootstrap())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()


__all__: list[str] = [
    "ENV_CONTRACTS_DIR",
    "bootstrap",
    "load_runtime_config",
    "main",
]
