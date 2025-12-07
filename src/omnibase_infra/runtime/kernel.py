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
from pathlib import Path
from typing import Any

import yaml

from omnibase_infra.event_bus.inmemory_event_bus import InMemoryEventBus
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONTRACTS_DIR = "./contracts"
DEFAULT_RUNTIME_CONFIG = "runtime/runtime_config.yaml"
DEFAULT_INPUT_TOPIC = "requests"
DEFAULT_OUTPUT_TOPIC = "responses"
DEFAULT_GROUP_ID = "onex-runtime"


def load_runtime_config(contracts_dir: Path) -> dict[str, Any]:
    """Load runtime configuration from contract file or return defaults.

    Attempts to load runtime_config.yaml from the contracts directory.
    If the file doesn't exist, returns sensible defaults to allow
    the runtime to start without requiring a config file.

    Args:
        contracts_dir: Path to the contracts directory.

    Returns:
        Configuration dictionary with runtime settings.
    """
    config_path = contracts_dir / DEFAULT_RUNTIME_CONFIG

    if config_path.exists():
        logger.info("Loading runtime config from %s", config_path)
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        return config

    # No config file - use environment variables and defaults
    logger.info(
        "No runtime config found at %s, using environment/defaults", config_path
    )
    return {
        "input_topic": os.getenv("ONEX_INPUT_TOPIC", DEFAULT_INPUT_TOPIC),
        "output_topic": os.getenv("ONEX_OUTPUT_TOPIC", DEFAULT_OUTPUT_TOPIC),
        "group_id": os.getenv("ONEX_GROUP_ID", DEFAULT_GROUP_ID),
    }


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
    # 1. Determine contracts directory
    contracts_dir = Path(os.getenv("CONTRACTS_DIR", DEFAULT_CONTRACTS_DIR))
    logger.info("ONEX Kernel starting with contracts_dir=%s", contracts_dir)

    # 2. Load runtime configuration
    config = load_runtime_config(contracts_dir)
    logger.debug("Runtime config: %s", config)

    # 3. Create event bus
    event_bus = InMemoryEventBus(
        environment=os.getenv("ONEX_ENVIRONMENT", "local"),
        group=config.get("group_id", DEFAULT_GROUP_ID),
    )

    # 4. Create runtime host process with config
    runtime = RuntimeHostProcess(
        event_bus=event_bus,
        input_topic=config.get("input_topic", DEFAULT_INPUT_TOPIC),
        output_topic=config.get("output_topic", DEFAULT_OUTPUT_TOPIC),
        config=config,
    )

    # 5. Setup graceful shutdown
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def handle_shutdown(sig: signal.Signals) -> None:
        """Handle shutdown signal."""
        logger.info("Received %s, initiating graceful shutdown...", sig.name)
        shutdown_event.set()

    # Register signal handlers (Unix-only)
    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, handle_shutdown, sig)

    # 6. Start and run
    try:
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

        logger.info("ONEX runtime stopped successfully.")
        return 0

    except Exception as e:
        logger.exception("ONEX runtime failed: %s", e)
        # Ensure cleanup on error
        try:
            await runtime.stop()
        except Exception:
            pass  # Best effort cleanup
        return 1


def configure_logging() -> None:
    """Configure logging for the kernel.

    Sets up structured logging with appropriate log level from
    environment variable ONEX_LOG_LEVEL (default: INFO).
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
    logger.info("ONEX Kernel v0.1.0")
    exit_code = asyncio.run(bootstrap())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()


__all__: list[str] = ["bootstrap", "main", "load_runtime_config"]
