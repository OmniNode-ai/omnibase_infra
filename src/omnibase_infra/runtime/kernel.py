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
    5. Setting up graceful shutdown signal handlers
    6. Running the runtime until shutdown is requested

Usage:
    # Run with default contracts directory (./contracts)
    python -m omnibase_infra.runtime.kernel

    # Run with custom contracts directory
    CONTRACTS_DIR=/path/to/contracts python -m omnibase_infra.runtime.kernel

    # Or via the installed entrypoint
    onex-runtime

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
from importlib.metadata import version as get_package_version
from pathlib import Path
from typing import cast
from uuid import uuid4

import yaml
from pydantic import ValidationError

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    ModelInfraErrorContext,
    ProtocolConfigurationError,
    RuntimeHostError,
)
from omnibase_infra.event_bus.inmemory_event_bus import InMemoryEventBus
from omnibase_infra.runtime.models import ModelRuntimeConfig
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess

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


def load_runtime_config(contracts_dir: Path) -> ModelRuntimeConfig:
    """Load runtime configuration from contract file or return defaults.

    Attempts to load runtime_config.yaml from the contracts directory.
    If the file doesn't exist, returns sensible defaults to allow
    the runtime to start without requiring a config file.

    Args:
        contracts_dir: Path to the contracts directory.

    Returns:
        ModelRuntimeConfig: Typed configuration model with runtime settings.

    Raises:
        ProtocolConfigurationError: If config file exists but cannot be parsed
            or fails validation.
    """
    config_path = contracts_dir / DEFAULT_RUNTIME_CONFIG
    context = ModelInfraErrorContext(
        transport_type=EnumInfraTransportType.RUNTIME,
        operation="load_config",
        target_name=str(config_path),
        correlation_id=uuid4(),
    )

    if config_path.exists():
        logger.info("Loading runtime config from %s", config_path)
        try:
            with open(config_path) as f:
                raw_config = yaml.safe_load(f) or {}
            return ModelRuntimeConfig.model_validate(raw_config)
        except yaml.YAMLError as e:
            raise ProtocolConfigurationError(
                f"Failed to parse runtime config YAML at {config_path}",
                context=context,
                config_path=str(config_path),
            ) from e
        except ValidationError as e:
            raise ProtocolConfigurationError(
                f"Runtime config validation failed at {config_path}: {e.error_count()} error(s)",
                context=context,
                config_path=str(config_path),
                validation_errors=str(e),
            ) from e
        except OSError as e:
            raise ProtocolConfigurationError(
                f"Failed to read runtime config at {config_path}",
                context=context,
                config_path=str(config_path),
            ) from e

    # No config file - use environment variables and defaults
    logger.info(
        "No runtime config found at %s, using environment/defaults", config_path
    )
    return ModelRuntimeConfig(
        input_topic=os.getenv("ONEX_INPUT_TOPIC", DEFAULT_INPUT_TOPIC),
        output_topic=os.getenv("ONEX_OUTPUT_TOPIC", DEFAULT_OUTPUT_TOPIC),
        consumer_group=os.getenv("ONEX_GROUP_ID", DEFAULT_GROUP_ID),
    )


async def bootstrap() -> int:
    """Bootstrap the ONEX runtime from contracts.

    This is the main async entrypoint that:
    1. Determines contracts directory from CONTRACTS_DIR env var
    2. Loads runtime configuration from contracts or defaults
    3. Creates and starts InMemoryEventBus
    4. Creates RuntimeHostProcess with configuration
    5. Sets up signal handlers for graceful shutdown
    6. Runs until shutdown signal received

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    # Initialize runtime to None for cleanup guard
    runtime: RuntimeHostProcess | None = None
    correlation_id = uuid4()

    # Create error context for bootstrap operations
    bootstrap_context = ModelInfraErrorContext(
        transport_type=EnumInfraTransportType.RUNTIME,
        operation="bootstrap",
        target_name="onex-kernel",
        correlation_id=correlation_id,
    )

    try:
        # 1. Determine contracts directory
        contracts_dir = Path(os.getenv("CONTRACTS_DIR", DEFAULT_CONTRACTS_DIR))
        logger.info("ONEX Kernel starting with contracts_dir=%s", contracts_dir)

        # 2. Load runtime configuration (may raise ProtocolConfigurationError)
        config = load_runtime_config(contracts_dir)
        logger.debug("Runtime config: %s", config.model_dump())

        # 3. Create event bus
        # MVP limitation: Always creates InMemoryEventBus regardless of config.event_bus.type.
        # The config model supports "kafka" as a type value for future compatibility,
        # but Kafka event bus implementation is not yet available. When Kafka support
        # is added, this section should dispatch based on config.event_bus.type.
        # Environment override takes precedence over config for environment field.
        environment = os.getenv("ONEX_ENVIRONMENT") or config.event_bus.environment
        event_bus = InMemoryEventBus(
            environment=environment,
            group=config.consumer_group,
        )

        # 4. Create runtime host process with config
        # RuntimeHostProcess accepts config as dict; cast model_dump() result to
        # dict[str, object] to avoid implicit Any typing (Pydantic's model_dump()
        # returns dict[str, Any] but all our model fields are strongly typed)
        runtime = RuntimeHostProcess(
            event_bus=event_bus,
            input_topic=config.input_topic,
            output_topic=config.output_topic,
            config=cast(dict[str, object], config.model_dump()),
        )

        # 5. Setup graceful shutdown
        shutdown_event = asyncio.Event()
        loop = asyncio.get_running_loop()

        def handle_shutdown(sig: signal.Signals) -> None:
            """Handle shutdown signal."""
            logger.info("Received %s, initiating graceful shutdown...", sig.name)
            shutdown_event.set()

        # Register signal handlers for graceful shutdown
        if sys.platform != "win32":
            # Unix: Use asyncio's signal handler for proper event loop integration
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, handle_shutdown, sig)
        else:
            # Windows: asyncio signal handlers not supported, use signal.signal()
            # for SIGINT (Ctrl+C). Note: SIGTERM not available on Windows.
            def windows_handler(signum: int, frame: object) -> None:
                """Windows-compatible signal handler wrapper."""
                handle_shutdown(signal.Signals(signum))

            signal.signal(signal.SIGINT, windows_handler)

        # 6. Start and run
        logger.info("Starting ONEX runtime...")
        await runtime.start()

        logger.info(
            "ONEX runtime started successfully. "
            "Listening on topic '%s', publishing to '%s'",
            runtime.input_topic,
            runtime.output_topic,
        )

        # Wait for shutdown signal
        await shutdown_event.wait()

        logger.info("Shutdown signal received, stopping runtime...")
        await runtime.stop()
        runtime = None  # Mark as stopped to prevent double-stop in finally

        logger.info("ONEX runtime stopped successfully.")
        return 0

    except ProtocolConfigurationError:
        # Configuration errors already have proper context and chaining
        logger.exception("ONEX runtime configuration failed")
        return 1

    except RuntimeHostError:
        # Runtime host errors already have proper structure
        logger.exception("ONEX runtime host error")
        return 1

    except Exception as e:
        # Unexpected errors: log with full context and return error code
        # (consistent with ProtocolConfigurationError and RuntimeHostError handlers)
        logger.exception(
            "ONEX runtime failed with unexpected error: %s (correlation_id=%s)",
            e,
            correlation_id,
        )
        return 1

    finally:
        # Guard cleanup - only attempt if runtime was initialized and not already stopped
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
    """Configure logging for the kernel.

    Sets up structured logging with appropriate log level from
    environment variable ONEX_LOG_LEVEL (default: INFO).

    Note on bootstrap order (intentional design):
        This function is called BEFORE runtime config is loaded because we need
        logging available during config loading itself (to log errors, warnings,
        and info about config discovery). Therefore, logging configuration uses
        environment variables rather than contract-based config values like
        config.logging.level or config.logging.format. This is a deliberate
        chicken-and-egg solution: env vars control early bootstrap logging,
        while contract config controls runtime behavior after bootstrap.
    """
    log_level = os.getenv("ONEX_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    """Entry point for the ONEX runtime kernel.

    Configures logging and runs the async bootstrap function.
    """
    configure_logging()
    logger.info("ONEX Kernel v%s", KERNEL_VERSION)
    exit_code = asyncio.run(bootstrap())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()


__all__: list[str] = ["bootstrap", "main", "load_runtime_config"]
