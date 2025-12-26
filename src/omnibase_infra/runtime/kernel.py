# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Kernel - Minimal bootstrap for contract-driven runtime.

This is the kernel entrypoint for the ONEX runtime. It provides a contract-driven
bootstrap that wires configuration into the existing RuntimeHostProcess.

The kernel is responsible for:
    1. Loading runtime configuration from contracts or environment
    2. Creating and starting the InMemoryEventBus
    3. Building the dependency container (event_bus, config)
    4. Instantiating RuntimeHostProcess with contract-driven configuration
    5. Starting the HTTP health server for Docker/K8s probes
    6. Setting up graceful shutdown signal handlers
    7. Running the runtime until shutdown is requested

Usage:
    # Run with default contracts directory (./contracts)
    python -m omnibase_infra.runtime.kernel

    # Run with custom contracts directory
    CONTRACTS_DIR=/path/to/contracts python -m omnibase_infra.runtime.kernel

    # Or via the installed entrypoint
    onex-runtime

Environment Variables:
    CONTRACTS_DIR: Path to contracts directory (default: ./contracts)
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
from importlib.metadata import version as get_package_version
from pathlib import Path
from typing import TYPE_CHECKING, cast

import yaml
from omnibase_core.container import ModelONEXContainer
from pydantic import ValidationError

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    ModelInfraErrorContext,
    ProtocolConfigurationError,
    RuntimeHostError,
)
from omnibase_infra.event_bus.inmemory_event_bus import InMemoryEventBus
from omnibase_infra.runtime.container_wiring import wire_infrastructure_services
from omnibase_infra.runtime.health_server import DEFAULT_HTTP_PORT, HealthServer
from omnibase_infra.runtime.models import ModelRuntimeConfig
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess
from omnibase_infra.runtime.validation import validate_runtime_config
from omnibase_infra.utils.correlation import generate_correlation_id

if TYPE_CHECKING:
    from omnibase_core.types import JsonValue

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
DEFAULT_INPUT_TOPIC = "requests"
DEFAULT_OUTPUT_TOPIC = "responses"
DEFAULT_GROUP_ID = "onex-runtime"

# Port validation constants
MIN_PORT = 1
MAX_PORT = 65535


def load_runtime_config(contracts_dir: Path) -> ModelRuntimeConfig:
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
        - File-based config takes precedence over environment variables
        - Individual environment variables can override specific file settings
        - Defaults are used when neither file nor environment variables are set

    Args:
        contracts_dir: Path to the contracts directory containing runtime_config.yaml.
            Example: Path("./contracts") or Path("/app/contracts")

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
        'requests'
        >>> print(config.event_bus.type)
        'inmemory'

    Example Error:
        >>> # If runtime_config.yaml has invalid YAML syntax
        >>> load_runtime_config(Path("./invalid"))
        ProtocolConfigurationError: Failed to parse runtime config YAML at ./invalid/runtime/runtime_config.yaml
        (correlation_id: 123e4567-e89b-12d3-a456-426614174000)
    """
    config_path = contracts_dir / DEFAULT_RUNTIME_CONFIG
    correlation_id = generate_correlation_id()
    context = ModelInfraErrorContext(
        transport_type=EnumInfraTransportType.RUNTIME,
        operation="load_config",
        target_name=str(config_path),
        correlation_id=correlation_id,
    )

    if config_path.exists():
        logger.info(
            "Loading runtime config from %s (correlation_id=%s)",
            config_path,
            correlation_id,
        )
        try:
            with open(config_path, encoding="utf-8") as f:
                raw_config = yaml.safe_load(f) or {}

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
                correlation_id,
            )

            config = ModelRuntimeConfig.model_validate(raw_config)
            logger.debug(
                "Runtime config loaded successfully (correlation_id=%s)",
                correlation_id,
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
        correlation_id,
    )
    config = ModelRuntimeConfig(
        input_topic=os.getenv("ONEX_INPUT_TOPIC", DEFAULT_INPUT_TOPIC),
        output_topic=os.getenv("ONEX_OUTPUT_TOPIC", DEFAULT_OUTPUT_TOPIC),
        consumer_group=os.getenv("ONEX_GROUP_ID", DEFAULT_GROUP_ID),
    )
    logger.debug(
        "Runtime config constructed from environment/defaults (correlation_id=%s)",
        correlation_id,
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
        1. Determine contracts directory from CONTRACTS_DIR environment variable
        2. Load and validate runtime configuration from contracts or environment
        3. Create and initialize InMemoryEventBus for event-driven architecture
        4. Create ModelONEXContainer and wire infrastructure services (async)
        5. Resolve ProtocolBindingRegistry from container (async)
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
        CONTRACTS_DIR: Path to contracts directory (default: ./contracts)
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
    # Initialize runtime and health server to None for cleanup guard
    runtime: RuntimeHostProcess | None = None
    health_server: HealthServer | None = None
    correlation_id = generate_correlation_id()
    bootstrap_start_time = time.time()

    try:
        # 1. Determine contracts directory
        contracts_dir = Path(os.getenv("CONTRACTS_DIR", DEFAULT_CONTRACTS_DIR))
        logger.info(
            "ONEX Kernel starting with contracts_dir=%s (correlation_id=%s)",
            contracts_dir,
            correlation_id,
        )

        # 2. Load runtime configuration (may raise ProtocolConfigurationError)
        config_start_time = time.time()
        config = load_runtime_config(contracts_dir)
        config_duration = time.time() - config_start_time
        logger.debug(
            "Runtime config loaded in %.3fs (correlation_id=%s)",
            config_duration,
            correlation_id,
            extra={
                "duration_seconds": config_duration,
                "config": config.model_dump(),
            },
        )

        # 3. Create event bus
        # MVP limitation: Always creates InMemoryEventBus regardless of config.event_bus.type.
        # The config model supports "kafka" as a type value for future compatibility,
        # but Kafka event bus implementation is not yet available. When Kafka support
        # is added, this section should dispatch based on config.event_bus.type.
        # Environment override takes precedence over config for environment field.
        environment = os.getenv("ONEX_ENVIRONMENT") or config.event_bus.environment
        event_bus_start_time = time.time()
        event_bus = InMemoryEventBus(
            environment=environment,
            group=config.consumer_group,
        )
        event_bus_duration = time.time() - event_bus_start_time
        logger.debug(
            "Event bus created in %.3fs (correlation_id=%s)",
            event_bus_duration,
            correlation_id,
            extra={
                "duration_seconds": event_bus_duration,
                "environment": environment,
                "consumer_group": config.consumer_group,
            },
        )

        # 4. Create and wire container for dependency injection
        container_start_time = time.time()
        container = ModelONEXContainer()
        wire_summary = await wire_infrastructure_services(container)
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

        # 5. Resolve ProtocolBindingRegistry from container
        from omnibase_infra.runtime.handler_registry import ProtocolBindingRegistry

        handler_registry: ProtocolBindingRegistry = (
            await container.service_registry.resolve_service(ProtocolBindingRegistry)
        )

        # 6. Create runtime host process with config and pre-resolved registry
        # RuntimeHostProcess accepts config as dict; cast model_dump() result to
        # dict[str, JsonValue] to avoid implicit Any typing (Pydantic's model_dump()
        # returns dict[str, Any] but all our model fields are strongly typed)
        runtime_create_start_time = time.time()
        runtime = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic=config.input_topic,
            output_topic=config.output_topic,
            config=cast("dict[str, JsonValue]", config.model_dump()),
            handler_registry=handler_registry,
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

        health_server = HealthServer(
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

        # Calculate total bootstrap time
        bootstrap_duration = time.time() - bootstrap_start_time

        # Display startup banner with key configuration
        banner_lines = [
            "=" * 60,
            f"ONEX Runtime Kernel v{KERNEL_VERSION}",
            f"Environment: {environment}",
            f"Contracts: {contracts_dir}",
            f"Event Bus: {config.event_bus.type} (group: {config.consumer_group})",
            f"Topics: {config.input_topic} → {config.output_topic}",
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

        # Stop health server first (fast, non-blocking)
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
        logger.exception(
            "ONEX runtime configuration failed (correlation_id=%s)",
            correlation_id,
            extra={
                "error_type": type(e).__name__,
                "error_code": e.model.error_code.name if hasattr(e, "model") else None,
            },
        )
        return 1

    except RuntimeHostError as e:
        # Runtime host errors already have proper structure
        logger.exception(
            "ONEX runtime host error (correlation_id=%s)",
            correlation_id,
            extra={
                "error_type": type(e).__name__,
                "error_code": e.model.error_code.name if hasattr(e, "model") else None,
            },
        )
        return 1

    except Exception as e:
        # Unexpected errors: log with full context and return error code
        # (consistent with ProtocolConfigurationError and RuntimeHostError handlers)
        logger.exception(
            "ONEX runtime failed with unexpected error: %s (correlation_id=%s)",
            e,
            correlation_id,
            extra={
                "error_type": type(e).__name__,
            },
        )
        return 1

    finally:
        # Guard cleanup - stop health server and runtime if not already stopped
        if health_server is not None:
            try:
                await health_server.stop()
            except Exception as cleanup_error:
                logger.warning(
                    "Failed to stop health server during cleanup: %s (correlation_id=%s)",
                    cleanup_error,
                    correlation_id,
                )

        if runtime is not None:
            try:
                await runtime.stop()
            except Exception as cleanup_error:
                # Log cleanup failures with context instead of suppressing them
                logger.warning(
                    "Failed to stop runtime during cleanup: %s (correlation_id=%s)",
                    cleanup_error,
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
        2025-01-15 10:30:45 [INFO] omnibase_infra.runtime.kernel: ONEX Kernel v0.1.0
        2025-01-15 10:30:45 [DEBUG] omnibase_infra.runtime.kernel: Runtime config loaded
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
        - Direct module execution: `python -m omnibase_infra.runtime.kernel`
        - Docker CMD/ENTRYPOINT in container deployments

    Example:
        >>> # From command line
        >>> python -m omnibase_infra.runtime.kernel
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


__all__: list[str] = ["bootstrap", "main", "load_runtime_config"]
