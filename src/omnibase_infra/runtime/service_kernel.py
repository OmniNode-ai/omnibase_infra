# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
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
    - EventBusKafka: For production use with Kafka/Redpanda (default)
    - EventBusInmemory: For testing only (via ONEX_EVENT_BUS_TYPE env override)

    Selection is determined by:
    - ONEX_EVENT_BUS_TYPE environment variable (highest priority, testing only)
    - KAFKA_BOOTSTRAP_SERVERS environment variable (if set, uses Kafka)
    - config.event_bus.type field in runtime_config.yaml (defaults to kafka)

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
import re
import signal
import sys
import time
from collections.abc import Awaitable, Callable
from importlib.metadata import version as get_package_version
from pathlib import Path
from typing import cast
from uuid import UUID

import yaml
from pydantic import ValidationError

from omnibase_core.container import ModelONEXContainer
from omnibase_core.protocols.event_bus import ProtocolEventBusPublisher
from omnibase_infra.enums import (
    EnumCircuitState,
    EnumConsumerGroupPurpose,
    EnumInfraTransportType,
)
from omnibase_infra.errors import (
    DbOwnershipMismatchError,
    DbOwnershipMissingError,
    EventRegistryFingerprintMismatchError,
    EventRegistryFingerprintMissingError,
    ModelInfraErrorContext,
    ProtocolConfigurationError,
    RuntimeHostError,
    SchemaFingerprintMismatchError,
    SchemaFingerprintMissingError,
    ServiceResolutionError,
)

# OMN-7077: EventBusInmemory is migrating to omnibase_core.
# Import from infra for type safety until core Part 1 merges; runtime bus
# selection in select_event_bus() handles the core→infra fallback.
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.models.health.model_llm_endpoint_health_config import (
    ModelLlmEndpointHealthConfig,
)
from omnibase_infra.nodes.node_autonomous_loop_orchestrator.plugin import (
    PluginBuildLoop,
)
from omnibase_infra.nodes.node_contract_registry_reducer.contract_registration_event_router import (
    ContractRegistrationEventRouter,
    ProtocolIntentEffect,
)
from omnibase_infra.nodes.node_contract_registry_reducer.reducer import (
    ContractRegistryReducer,
)
from omnibase_infra.nodes.node_delegation_orchestrator.plugin import (
    PluginDelegation,
)
from omnibase_infra.nodes.node_registration_orchestrator.plugin import (
    PluginRegistration,
)
from omnibase_infra.observability.runtime_log_event_bridge import RuntimeLogEventBridge
from omnibase_infra.observability.wiring_health.wiring_health_checker import (
    WiringHealthChecker,
)
from omnibase_infra.runtime.handler_registry import RegistryProtocolBinding
from omnibase_infra.runtime.models import (
    ModelDomainPluginConfig,
    ModelRuntimeConfig,
    ModelSecurityConfig,
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
from omnibase_infra.runtime.models.model_runtime_node_graph_config import (
    ModelRuntimeNodeGraphConfig,
)
from omnibase_infra.runtime.protocol_domain_plugin import (
    ProtocolDomainPlugin,
    RegistryDomainPlugin,
)
from omnibase_infra.runtime.service_runtime_host_process import RuntimeHostProcess
from omnibase_infra.runtime.util_container_wiring import (
    wire_infrastructure_services,
)
from omnibase_infra.runtime.util_validation import validate_runtime_config
from omnibase_infra.services.service_circuit_breaker_event_publisher import (
    CircuitBreakerEventPublisher,
)
from omnibase_infra.services.service_llm_endpoint_health import (
    ServiceLlmEndpointHealth,
)
from omnibase_infra.topics import (
    SUFFIX_CONTRACT_DEREGISTERED,
    SUFFIX_CONTRACT_REGISTERED,
    SUFFIX_NODE_HEARTBEAT,
    SUFFIX_RUNTIME_ERROR,
    TopicResolutionError,
    TopicResolver,
)
from omnibase_infra.utils.correlation import generate_correlation_id
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

logger = logging.getLogger(__name__)

# Kernel version - read from installed package metadata to avoid version drift
# between code and pyproject.toml. Falls back to "unknown" if package is not
# installed (e.g., during development without editable install).
try:
    KERNEL_VERSION = get_package_version("omnibase_infra")
except Exception:  # noqa: BLE001 — boundary: catch-all for resilience
    KERNEL_VERSION = "unknown"

# Default configuration
DEFAULT_CONTRACTS_DIR = "./contracts"
DEFAULT_RUNTIME_CONFIG = "runtime/runtime_config.yaml"

# Environment variable name for contracts directory
ENV_CONTRACTS_DIR = "ONEX_CONTRACTS_DIR"
# Environment variable for omniclaude skills root (cross-repo topic discovery)
ENV_OMNICLAUDE_SKILLS_ROOT = "OMNICLAUDE_SKILLS_ROOT"
DEFAULT_INPUT_TOPIC = "requests"
DEFAULT_OUTPUT_TOPIC = "responses"
DEFAULT_GROUP_ID = "onex-runtime"

# Port validation constants
MIN_PORT = 1
MAX_PORT = 65535

# Kafka broker allowlist validation
# Patterns that are unconditionally rejected — they point at local or
# container-internal brokers that cannot reach the production Redpanda cluster.
_KAFKA_BROKER_DENYLIST_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^localhost:"),
    re.compile(r"^redpanda:"),
    re.compile(r"^127\.0\.0\.1:"),
    re.compile(r"^0\.0\.0\.0:"),
)

# Environment variable name for the operator-supplied allowlist.
# Value: comma-separated host prefixes, e.g. "192.168.86.,10.0.0."
# When unset, only the built-in denylist is enforced.
ENV_KAFKA_BROKER_ALLOWLIST = "KAFKA_BROKER_ALLOWLIST"


def validate_kafka_broker_allowlist(
    bootstrap_servers: str,
    correlation_id: object | None = None,
) -> None:
    """Validate that a Kafka broker address is not a known-bad local target.

    Called during bootstrap() before any Kafka consumers or producers are
    started. Raises ProtocolConfigurationError immediately if the value
    matches any entry in the denylist, providing a clear error message
    rather than a confusing connection-refused timeout seconds later.

    The allowlist is configurable via KAFKA_BROKER_ALLOWLIST (comma-separated
    host prefixes). When set, *any* broker that matches at least one prefix
    passes validation regardless of the denylist. When unset only the
    denylist is applied — any non-denied value is accepted.

    Args:
        bootstrap_servers: Raw value of KAFKA_BOOTSTRAP_SERVERS.
        correlation_id: Optional correlation ID for structured error context.

    Raises:
        ProtocolConfigurationError: If the broker value matches a denylist
            pattern and does not match any allowlist prefix.
    """
    context = ModelInfraErrorContext(
        transport_type=EnumInfraTransportType.KAFKA,
        operation="validate_kafka_broker",
        correlation_id=correlation_id,
    )

    # Read operator-supplied allowlist (comma-separated host prefixes)
    raw_allowlist = os.getenv(ENV_KAFKA_BROKER_ALLOWLIST, "")
    allowlist_prefixes: list[str] = [
        p.strip() for p in raw_allowlist.split(",") if p.strip()
    ]

    # Validate each broker in the comma-separated list
    for broker in bootstrap_servers.split(","):
        broker = broker.strip()
        if not broker:
            continue

        if allowlist_prefixes:
            # Strict allowlist mode: when KAFKA_BROKER_ALLOWLIST is set, only
            # brokers whose address starts with a listed prefix are accepted.
            # Brokers not matching any prefix are rejected — this prevents
            # unintended connections to off-allowlist hosts.
            if any(broker.startswith(prefix) for prefix in allowlist_prefixes):
                continue
            raise ProtocolConfigurationError(
                f"KAFKA_BOOTSTRAP_SERVERS value '{broker}' is not permitted. "
                f"KAFKA_BROKER_ALLOWLIST is set but '{broker}' does not start with "
                f"any listed prefix ({', '.join(allowlist_prefixes)}). "
                f"Add the appropriate prefix to KAFKA_BROKER_ALLOWLIST to permit it.",
                context=context,
                rejected_broker=broker,
                parameter="KAFKA_BOOTSTRAP_SERVERS",
            )

        # Denylist-only mode (no allowlist configured): reject known-bad patterns
        for pattern in _KAFKA_BROKER_DENYLIST_PATTERNS:
            if pattern.match(broker):
                raise ProtocolConfigurationError(
                    f"KAFKA_BOOTSTRAP_SERVERS value '{broker}' is not allowed. "
                    f"Local/container broker addresses are rejected at boot to "
                    f"prevent silent misconfiguration. "
                    f"Set KAFKA_BROKER_ALLOWLIST to override (comma-separated prefixes, e.g., "
                    f"KAFKA_BROKER_ALLOWLIST=redpanda: for local Docker containers, "
                    f"or KAFKA_BROKER_ALLOWLIST=localhost: for host scripts and tests).",
                    context=context,
                    rejected_broker=broker,
                    parameter="KAFKA_BOOTSTRAP_SERVERS",
                )


def _load_node_graph_config() -> ModelRuntimeNodeGraphConfig:
    """Load typed runtime config from the 5 runtime contract YAMLs.

    Resolution uses ``get_runtime_contracts_dir()`` from omnibase_core, which checks
    ``ONEX_RUNTIME_CONTRACTS_DIR`` env var first, then falls back to the repository-
    relative path.

    Returns:
        Frozen config model with all runtime parameters.

    Raises:
        FileNotFoundError: If runtime contracts directory cannot be found.
    """
    from omnibase_core.contracts.runtime_contracts import get_runtime_contracts_dir

    contracts_dir = get_runtime_contracts_dir()
    return ModelRuntimeNodeGraphConfig.from_contracts_dir(contracts_dir)


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
        - File-based config is loaded and contract-validated first
        - Environment variables (ONEX_GROUP_ID, ONEX_INPUT_TOPIC, ONEX_OUTPUT_TOPIC)
          override corresponding YAML fields when a config file is present
        - Environment overrides are re-validated against the same contract rules
          as the YAML file, preventing invalid env-var values from bypassing checks
        - When no config file exists, environment variables and defaults are used
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
        kafka

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

            # Environment variable overrides (highest priority per contract header).
            # Env-var values are merged into raw_config and re-validated against
            # the same contract rules that the YAML file was validated against.
            # This prevents invalid env-var values from bypassing contract checks.
            env_overrides: dict[str, str] = {}
            env_group_id = os.getenv("ONEX_GROUP_ID")
            env_input_topic = os.getenv("ONEX_INPUT_TOPIC")
            env_output_topic = os.getenv("ONEX_OUTPUT_TOPIC")

            # Reject empty-string env var overrides with a clear diagnostic.
            # An empty string passes the ``is not None`` check but would
            # produce a confusing Pydantic validation error downstream.
            _env_override_names = {
                "ONEX_GROUP_ID": env_group_id,
                "ONEX_INPUT_TOPIC": env_input_topic,
                "ONEX_OUTPUT_TOPIC": env_output_topic,
            }
            for var_name, var_value in _env_override_names.items():
                if var_value is not None and var_value.strip() == "":
                    raise ProtocolConfigurationError(
                        f"Environment variable {var_name} is set but empty. "
                        f"Either unset it to use the YAML default or provide "
                        f"a non-empty value.",
                        context=context,
                        config_path=str(config_path),
                    )

            if env_group_id is not None:
                env_overrides["consumer_group"] = env_group_id
            if env_input_topic is not None:
                env_overrides["input_topic"] = env_input_topic
            if env_output_topic is not None:
                env_overrides["output_topic"] = env_output_topic
            if env_overrides:
                merged = {**raw_config, **env_overrides}
                # Remove the group_id alias key if consumer_group is being overridden,
                # because Pydantic gives alias keys precedence over field names when
                # both are present (populate_by_name=True). Without this, the YAML
                # group_id value would shadow the env-var consumer_group override.
                if "consumer_group" in env_overrides and "group_id" in merged:
                    del merged["group_id"]
                # Re-validate merged config to catch invalid env-var values
                override_errors = validate_runtime_config(merged)
                if override_errors:
                    error_count = len(override_errors)
                    error_summary = "; ".join(override_errors[:3])
                    if error_count > 3:
                        error_summary += f" (and {error_count - 3} more...)"
                    raise ProtocolConfigurationError(
                        f"Environment variable override validation failed: "
                        f"{error_count} error(s). "
                        f"First errors: {error_summary}",
                        context=context,
                        config_path=str(config_path),
                        validation_errors=override_errors,
                        error_count=error_count,
                        overridden_fields=list(env_overrides.keys()),
                    )
                config = ModelRuntimeConfig.model_validate(merged)
                logger.info(
                    "Applied environment variable overrides to runtime config",
                    extra={"overridden_fields": list(env_overrides.keys())},
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


def _topic_matches_pattern(topic: str, pattern: str) -> bool:
    """Check if a topic string matches a wildcard topic pattern.

    Supports ``*`` as a single-segment wildcard in ONEX topic patterns.
    Example: ``*.evt.platform.node-introspection.*`` matches
    ``onex.evt.platform.node-introspection.v1``.
    """
    topic_parts = topic.split(".")
    pattern_parts = pattern.split(".")
    if len(topic_parts) != len(pattern_parts):
        return False
    return all(
        pp == "*" or pp == tp for tp, pp in zip(topic_parts, pattern_parts, strict=True)
    )


# ai-slop-ok: pre-existing === separators in example startup log in docstring
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
        8. Start HTTP health server for Docker/Kubernetes health probes
        9. Start runtime (Kafka consumer joins — may take 10+ min)
        10. Run runtime until shutdown signal received
        11. Perform graceful shutdown with configurable timeout
        12. Clean up resources in finally block to prevent resource leaks

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
    # Plugin system owns resource lifecycle (pools, publishers, dispatchers)
    plugin_registry: RegistryDomainPlugin | None = None
    registration_plugin: PluginRegistration | None = None
    activated_plugins: list[ProtocolDomainPlugin] = []
    # ready_plugins tracks plugins that completed handler wiring successfully.
    # Only these plugins should have consumers started in Pass 2. Plugins in
    # activated_plugins but NOT in ready_plugins had successful init (so need
    # shutdown for cleanup) but failed wire_handlers/wire_dispatchers (so must
    # not start consumers with no handlers/dispatchers wired).
    ready_plugins: list[ProtocolDomainPlugin] = []
    plugin_unsubscribe_callbacks: list[Callable[[], Awaitable[None]]] = []
    # Contract registry unsubscribe functions and router (separate domain)
    contract_router: ContractRegistrationEventRouter | None = None
    contract_unsub_registered: Callable[[], Awaitable[None]] | None = None
    contract_unsub_deregistered: Callable[[], Awaitable[None]] | None = None
    contract_unsub_heartbeat: Callable[[], Awaitable[None]] | None = None
    plugin_config: ModelDomainPluginConfig | None = None
    runtime_log_bridge: RuntimeLogEventBridge | None = None
    llm_health_service: ServiceLlmEndpointHealth | None = None
    wiring_health_checker: WiringHealthChecker | None = None
    wiring_health_task: asyncio.Task[None] | None = None
    triage_unsub: Callable[[], Awaitable[None]] | None = None
    baselines_task: asyncio.Task[None] | None = None
    _baselines_pool = None  # asyncpg.Pool | None, assigned inside try block
    savings_task: asyncio.Task[None] | None = None
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

        # 1b. Load contract-driven runtime config [OMN-6339]
        # This loads typed configuration from the 5 runtime contract YAMLs
        # in omnibase_core. Values are source of truth; ONEX_RUNTIME_* env vars
        # are operator overrides on top. The node_graph_config is passed to
        # RuntimeHostProcess and other components that previously used DEFAULT_*
        # constants.
        try:
            node_graph_config = _load_node_graph_config()
        except (FileNotFoundError, ValueError) as exc:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                operation="load_node_graph_config",
                target_name=str(contracts_dir),
            )
            raise ProtocolConfigurationError(
                f"Failed to load runtime node graph config: {exc}",
                context=context,
            ) from exc
        logger.debug(
            "Runtime contract config loaded: startup_timeout=%dms, step_timeout=%dms "
            "(correlation_id=%s)",
            node_graph_config.startup_timeout_ms,
            node_graph_config.step_timeout_ms,
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

        # 2b. Assert config.event_bus.type is production-safe (OMN-4848)
        # The ModelEventBusConfig validator already rejects non-production-safe types
        # at model construction time. This runtime assertion is a defense-in-depth
        # guard that catches any bypass (e.g., mock configs in tests that skip
        # Pydantic validation). The ONEX_EVENT_BUS_TYPE env var override below
        # can still select inmemory for testing — that path is separate.
        if hasattr(config.event_bus.type, "is_production_safe"):
            if not config.event_bus.type.is_production_safe:
                context = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.KAFKA,
                    operation="validate_event_bus_config",
                    correlation_id=correlation_id,
                )
                raise ProtocolConfigurationError(
                    f"config.event_bus.type='{config.event_bus.type}' is not production-safe. "
                    f"Use 'kafka' or 'cloud' instead.",
                    context=context,
                    parameter="event_bus.type",
                )

        # 3. Create event bus
        # Dispatch based on configuration or environment variable:
        # - ONEX_EVENT_BUS_TYPE env var overrides config.event_bus.type
        # - If KAFKA_BOOTSTRAP_SERVERS env var is set, use EventBusKafka
        # - If config.event_bus.type == "kafka", use EventBusKafka
        # - Otherwise, use EventBusInmemory for testing (via ONEX_EVENT_BUS_TYPE override)
        # Environment override takes precedence over config for environment field.
        # KAFKA_ENVIRONMENT is the authoritative source for the Kafka topic prefix.
        # ONEX_ENVIRONMENT is a general environment name (not always a valid Kafka env value)
        # and is only used as a fallback if KAFKA_ENVIRONMENT is not set.
        # config.event_bus.environment is the final fallback (default: "local").
        _kafka_env_from_env = os.getenv("KAFKA_ENVIRONMENT") or os.getenv(
            "ONEX_ENVIRONMENT"
        )
        environment: str = _kafka_env_from_env or config.event_bus.environment
        kafka_bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
        if not kafka_bootstrap_servers:
            logger.warning(
                "KAFKA_BOOTSTRAP_SERVERS is not set. "
                "Kafka event bus will not be available unless ONEX_EVENT_BUS_TYPE=kafka "
                "is also requested, in which case startup will fail. "
                "Set KAFKA_BOOTSTRAP_SERVERS to the broker address "
                "(e.g., 'redpanda:9092' for local Docker, or a remote broker address) to enable Kafka. "
                "(correlation_id=%s)",
                correlation_id,
            )

        # Check for ONEX_EVENT_BUS_TYPE environment variable override
        # This allows CI/testing environments to force inmemory event bus
        # even when the config file defaults to kafka.
        event_bus_type_override = os.getenv("ONEX_EVENT_BUS_TYPE", "").lower()
        if event_bus_type_override:
            logger.debug(
                "Event bus type override from ONEX_EVENT_BUS_TYPE=%s (correlation_id=%s)",
                event_bus_type_override,
                correlation_id,
            )

        # Determine effective event bus type with override precedence:
        # 1. ONEX_EVENT_BUS_TYPE env var (highest priority)
        # 2. KAFKA_BOOTSTRAP_SERVERS env var (if set, implies kafka)
        # 3. config.event_bus.type (from runtime_config.yaml)
        # Both "kafka" and "cloud" types require a broker connection.
        _broker_required_types = {"kafka", "cloud"}
        if event_bus_type_override == "inmemory":
            # Explicit inmemory override - use inmemory regardless of other config
            use_kafka = False
            logger.info(
                "Using inmemory event bus (ONEX_EVENT_BUS_TYPE override) (correlation_id=%s)",
                correlation_id,
            )
        elif event_bus_type_override in _broker_required_types:
            # Explicit kafka/cloud override - validate that bootstrap_servers is available
            use_kafka = True
        elif event_bus_type_override and event_bus_type_override not in (
            "inmemory",
            *_broker_required_types,
        ):
            # Invalid override value - warn and fall back to config
            logger.warning(
                "Invalid ONEX_EVENT_BUS_TYPE value '%s', expected 'inmemory', 'kafka', or 'cloud'. "
                "Falling back to config.event_bus.type='%s' (correlation_id=%s)",
                event_bus_type_override,
                config.event_bus.type,
                correlation_id,
            )
            use_kafka = (
                bool(kafka_bootstrap_servers)
                or config.event_bus.type in _broker_required_types
            )
        else:
            # No override - use original logic
            # Explicit bool evaluation (not truthy string) for kafka usage.
            # KAFKA_BOOTSTRAP_SERVERS env var takes precedence over config.event_bus.type.
            # This prevents implicit "kafka but localhost" fallback scenarios.
            # Both "kafka" and "cloud" types require a broker connection.
            use_kafka = (
                bool(kafka_bootstrap_servers)
                or config.event_bus.type in _broker_required_types
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

        # Validate that the broker address is not a local/container broker.
        # This guard runs whenever Kafka is selected AND a bootstrap_servers
        # value is present. It fires *before* any connection attempt so that
        # misconfiguration is caught immediately at boot rather than producing
        # a confusing connection-refused error minutes later.
        # Warn-only when unset (unset case already handled above for Kafka mode).
        if kafka_bootstrap_servers:
            validate_kafka_broker_allowlist(kafka_bootstrap_servers, correlation_id)
        elif not use_kafka:
            # Inmemory mode with no broker configured: log at DEBUG only.
            logger.debug(
                "KAFKA_BOOTSTRAP_SERVERS is not set; using inmemory event bus "
                "(correlation_id=%s)",
                correlation_id,
            )

        event_bus_start_time = time.time()
        event_bus: EventBusInmemory | EventBusKafka
        event_bus_type: str

        # OMN-7076: Use registry auto-configuration for bus selection.
        # select_event_bus() probes onex.backends entry points and selects
        # the best available backend, replacing the previous inline if/else.
        from omnibase_infra.backends.auto_configure import select_event_bus

        event_bus = cast(  # type: ignore[redundant-cast]
            "EventBusInmemory | EventBusKafka",
            select_event_bus(
                kafka_bootstrap_servers=kafka_bootstrap_servers if use_kafka else None,
                environment=environment,
                consumer_group=config.consumer_group,
                circuit_breaker_threshold=config.event_bus.circuit_breaker_threshold,
            ),
        )
        event_bus_type = "kafka" if isinstance(event_bus, EventBusKafka) else "inmemory"

        # Start the bus and wire post-creation infrastructure
        if isinstance(event_bus, EventBusKafka):
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

            # 3.4a. Wire CircuitBreakerEventPublisher to EventBusKafka (OMN-6137)
            # Publishes state transition events to omnidash /circuit-breaker dashboard.
            try:
                _cb_publisher = CircuitBreakerEventPublisher(
                    event_bus=event_bus,  # type: ignore[arg-type]
                )

                async def _cb_transition_handler(
                    service_name: str,
                    new_state: EnumCircuitState,
                    previous_state: EnumCircuitState,
                    failure_count: int,
                    threshold: int,
                ) -> None:
                    await _cb_publisher.publish_transition(
                        service_name=service_name,
                        new_state=new_state,
                        previous_state=previous_state,
                        failure_count=failure_count,
                        threshold=threshold,
                    )

                event_bus.set_transition_callback(_cb_transition_handler)
                logger.info(
                    "CircuitBreakerEventPublisher wired to EventBusKafka "
                    "(correlation_id=%s)",
                    correlation_id,
                )
            except Exception:  # noqa: BLE001 — best-effort, never blocks startup
                logger.warning(
                    "Failed to wire CircuitBreakerEventPublisher, continuing "
                    "without circuit breaker event emission (correlation_id=%s)",
                    correlation_id,
                    exc_info=True,
                )

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

        # 3.5. Provision platform topics (best-effort, never blocks startup)
        if use_kafka:
            try:
                from omnibase_infra.event_bus.service_topic_manager import (
                    TopicProvisioner,
                )

                # Resolve optional cross-repo skill-manifest root for topic discovery.
                # OMNICLAUDE_SKILLS_ROOT: path to omniclaude plugins/onex/skills/
                _skills_root_env = os.environ.get(
                    ENV_OMNICLAUDE_SKILLS_ROOT, ""
                ).strip()
                _skill_manifests_root: Path | None = None
                if not _skills_root_env:
                    logger.info(
                        "Topic provisioning: skill-manifest discovery disabled"
                        " (OMNICLAUDE_SKILLS_ROOT not set)"
                    )
                else:
                    _skill_manifests_path = Path(_skills_root_env)
                    if not _skill_manifests_path.exists():
                        logger.warning(
                            "Topic provisioning: OMNICLAUDE_SKILLS_ROOT=%s not found,"
                            " skill topics will not be provisioned",
                            _skills_root_env,
                        )
                    else:
                        _skill_manifests_root = _skill_manifests_path

                _contracts_root = _get_contracts_dir()

                # Infra standalone manifests (cli/topics.yaml, services/topics.yaml)
                _infra_src = Path(__file__).resolve().parent.parent
                _extra_manifest_roots: list[Path] = []
                for _subdir in ("cli", "services"):
                    _candidate = _infra_src / _subdir
                    if _candidate.is_dir():
                        _extra_manifest_roots.append(_candidate)

                topic_provisioner = TopicProvisioner(
                    bootstrap_servers=kafka_bootstrap_servers,
                    contracts_root=_contracts_root,
                    skill_manifests_root=_skill_manifests_root,
                    skill_manifests_roots=_extra_manifest_roots,
                )
                provisioning_result = (
                    await topic_provisioner.ensure_provisioned_topics_exist(
                        correlation_id=correlation_id,
                    )
                )
                log_level = (
                    logging.WARNING
                    if provisioning_result["status"] != "success"
                    else logging.INFO
                )
                logger.log(
                    log_level,
                    "Topic provisioning: status=%s created=%d existing=%d failed=%d "
                    "failed_topics=%s (correlation_id=%s)",
                    provisioning_result["status"],
                    len(provisioning_result["created"]),
                    len(provisioning_result["existing"]),
                    len(provisioning_result["failed"]),
                    provisioning_result["failed"] or "none",
                    correlation_id,
                )
            except Exception:  # noqa: BLE001 — boundary: logs warning and degrades
                logger.warning(
                    "Topic provisioning failed (best-effort, non-blocking) "
                    "(correlation_id=%s)",
                    correlation_id,
                    exc_info=True,
                )

        # 3.6. Validate platform topics exist (best-effort by default)
        if use_kafka:
            try:
                from omnibase_infra.event_bus.service_topic_startup_validator import (
                    TopicStartupValidator,
                )

                validator = TopicStartupValidator(
                    bootstrap_servers=kafka_bootstrap_servers,
                )
                validation_result = await validator.validate(
                    correlation_id=correlation_id,
                )
                if not validation_result.is_valid:
                    if os.environ.get("STARTUP_VALIDATION_STRICT") == "1":
                        raise RuntimeError(
                            f"Missing topics: {validation_result.missing_topics}"
                        )
                    logger.warning(
                        "Topic validation: %d missing (non-blocking) "
                        "(correlation_id=%s)",
                        len(validation_result.missing_topics),
                        correlation_id,
                    )
            except RuntimeError:
                raise
            except Exception:  # noqa: BLE001 — boundary: logs warning and degrades
                logger.warning(
                    "Topic validation failed (best-effort, non-blocking) "
                    "(correlation_id=%s)",
                    correlation_id,
                    exc_info=True,
                )

        # 3.6. Initialize RuntimeLogEventBridge if enabled (OMN-5525)
        # Captures ERROR/WARNING log records from allowlisted loggers and emits
        # them as structured Kafka events. Requires a dedicated producer.
        if use_kafka and RuntimeLogEventBridge.is_enabled() and kafka_bootstrap_servers:
            try:
                from aiokafka import AIOKafkaProducer as _BridgeProducer

                _bridge_producer = _BridgeProducer(
                    bootstrap_servers=kafka_bootstrap_servers,
                )
                await _bridge_producer.start()

                runtime_log_bridge = RuntimeLogEventBridge(
                    producer=_bridge_producer,
                    hostname=os.environ.get("HOSTNAME", ""),
                    service_label="onex-kernel",
                )

                # Parse allowlist from env or use defaults
                allowlist_raw = os.environ.get(
                    "RUNTIME_LOG_BRIDGE_ALLOWLIST",
                    "aiokafka.consumer,asyncpg,aiohttp",
                )
                allowlist = [
                    name.strip() for name in allowlist_raw.split(",") if name.strip()
                ]
                runtime_log_bridge.attach_to_loggers(allowlist)
                await runtime_log_bridge.start()

                logger.info(
                    "RuntimeLogEventBridge started (loggers=%s, correlation_id=%s)",
                    allowlist,
                    correlation_id,
                )
            except Exception:  # noqa: BLE001 — best-effort, never blocks startup
                logger.warning(
                    "Failed to start RuntimeLogEventBridge, continuing without it "
                    "(correlation_id=%s)",
                    correlation_id,
                    exc_info=True,
                )
                runtime_log_bridge = None

        # 3.7. Initialize ServiceLlmEndpointHealth if LLM endpoints configured (OMN-6135)
        # Probes configured LLM endpoints at a regular interval and emits health
        # events to Kafka.  Best-effort: failure does not block kernel startup.
        if use_kafka:
            try:
                _llm_endpoints: dict[str, str] = {}
                for env_key, logical_name in [
                    ("LLM_CODER_URL", "coder"),
                    ("LLM_CODER_FAST_URL", "coder-fast"),
                    ("LLM_EMBEDDING_URL", "embedding"),
                    ("LLM_DEEPSEEK_R1_URL", "deepseek-r1"),
                    ("LLM_SMALL_URL", "small"),
                ]:
                    _url = os.environ.get(env_key, "").strip()
                    if _url:
                        _llm_endpoints[logical_name] = _url

                if _llm_endpoints:
                    _llm_health_config = ModelLlmEndpointHealthConfig(
                        endpoints=_llm_endpoints,
                        probe_interval_seconds=float(
                            os.environ.get("LLM_HEALTH_PROBE_INTERVAL", "30")
                        ),
                    )
                    llm_health_service = ServiceLlmEndpointHealth(
                        config=_llm_health_config,
                        event_bus=event_bus,  # type: ignore[arg-type]
                    )
                    await llm_health_service.start()
                    logger.info(
                        "ServiceLlmEndpointHealth started (endpoints=%d, correlation_id=%s)",
                        len(_llm_endpoints),
                        correlation_id,
                    )
                else:
                    logger.debug(
                        "No LLM endpoints configured, skipping health probes "
                        "(correlation_id=%s)",
                        correlation_id,
                    )
            except Exception:  # noqa: BLE001 — best-effort, never blocks startup
                logger.warning(
                    "Failed to start ServiceLlmEndpointHealth, continuing without it "
                    "(correlation_id=%s)",
                    correlation_id,
                    exc_info=True,
                )
                llm_health_service = None

        # 3.8. Wire baselines batch compute periodic task
        # Runs the 3-phase baselines computation at a configurable interval
        # and emits baselines-computed.v1 snapshot events for omnidash.
        # Creates its own asyncpg pool for isolation from plugin pools.
        # (baselines_task and _baselines_pool pre-declared before try block)
        if use_kafka:
            try:
                import asyncpg as _asyncpg

                from omnibase_core.models.events.model_event_envelope import (
                    ModelEventEnvelope,
                )

                _baselines_dsn = os.environ.get("OMNIBASE_INFRA_DB_URL", "").strip()
                if _baselines_dsn:
                    _baselines_interval = float(
                        os.environ.get("BASELINES_COMPUTE_INTERVAL", "300")
                    )
                    _baselines_pool = await _asyncpg.create_pool(
                        _baselines_dsn, min_size=1, max_size=2
                    )

                    from omnibase_infra.topics import topic_keys as _bl_topic_keys
                    from omnibase_infra.topics.service_topic_registry import (
                        ServiceTopicRegistry as _BlTopicRegistry,
                    )

                    _baselines_default_topic = _BlTopicRegistry.from_defaults().resolve(
                        _bl_topic_keys.BASELINES_COMPUTED
                    )

                    async def _baselines_publish(
                        event_type: str,
                        payload: object,
                        topic: str | None,
                        correlation_id: object,
                        **kwargs: object,
                    ) -> bool:
                        """Publisher callback for baselines batch compute."""
                        _env: ModelEventEnvelope[object] = ModelEventEnvelope(
                            payload=payload,
                            correlation_id=correlation_id,  # type: ignore[arg-type]
                            event_type=event_type,
                            source="baselines_batch_compute",
                        )
                        await event_bus.publish_envelope(
                            _env,  # type: ignore[arg-type]
                            topic=topic or _baselines_default_topic,
                        )
                        return True

                    from omnibase_infra.nodes.node_baselines_batch_compute.handlers.handler_baselines_batch_compute import (
                        HandlerBaselinesBatchCompute,
                    )
                    from omnibase_infra.nodes.node_baselines_batch_compute.models.model_baselines_batch_compute_command import (
                        ModelBaselinesBatchComputeCommand,
                    )

                    _baselines_handler = HandlerBaselinesBatchCompute(
                        pool=_baselines_pool,
                        publisher=_baselines_publish,
                    )

                    async def _baselines_loop() -> None:
                        """Periodic baselines batch computation loop."""
                        while True:
                            try:
                                await asyncio.sleep(_baselines_interval)
                                _cmd = ModelBaselinesBatchComputeCommand(
                                    correlation_id=generate_correlation_id(),
                                )
                                _output = await _baselines_handler.handle(_cmd)
                                logger.info(
                                    "Baselines batch compute completed "
                                    "(rows=%d, emitted=%s, correlation_id=%s)",
                                    _output.result.total_rows,
                                    _output.snapshot_emitted,
                                    _cmd.correlation_id,
                                )
                            except asyncio.CancelledError:
                                break
                            except Exception:  # noqa: BLE001 — never crash the loop
                                logger.warning(
                                    "Baselines batch compute failed (will retry in %ds)",
                                    int(_baselines_interval),
                                    exc_info=True,
                                )

                    baselines_task = asyncio.create_task(
                        _baselines_loop(), name="baselines-batch-compute"
                    )
                    logger.info(
                        "Baselines batch compute loop started "
                        "(interval=%ds, correlation_id=%s)",
                        int(_baselines_interval),
                        correlation_id,
                    )
                else:
                    logger.debug(
                        "OMNIBASE_INFRA_DB_URL not set, skipping baselines batch compute "
                        "(correlation_id=%s)",
                        correlation_id,
                    )
            except Exception:  # noqa: BLE001 — best-effort, never blocks startup
                logger.warning(
                    "Failed to start baselines batch compute loop, continuing "
                    "without it (correlation_id=%s)",
                    correlation_id,
                    exc_info=True,
                )
                baselines_task = None

        # 3.9. Wire savings estimation consumer
        # Correlates session events and produces savings-estimated.v1 events.
        # (savings_task pre-declared before try block)
        if use_kafka:
            try:
                from omnibase_core.models.events.model_event_envelope import (
                    ModelEventEnvelope,
                )
                from omnibase_infra.services.observability.savings_estimation.config import (
                    ConfigSavingsEstimation,
                )
                from omnibase_infra.services.observability.savings_estimation.consumer import (
                    ServiceSavingsEstimator,
                )
                from omnibase_infra.topics import topic_keys
                from omnibase_infra.topics.service_topic_registry import (
                    ServiceTopicRegistry,
                )

                _savings_config = ConfigSavingsEstimation()
                _savings_estimator = ServiceSavingsEstimator(
                    config=_savings_config,
                )
                _savings_topic = ServiceTopicRegistry.from_defaults().resolve(
                    topic_keys.SAVINGS_ESTIMATED
                )

                # Subscribe to input topics (resolved via topic registry)
                _savings_registry = ServiceTopicRegistry.from_defaults()
                _savings_input_topics = [
                    _savings_registry.resolve(topic_keys.LLM_CALL_COMPLETED),
                    _savings_registry.resolve(topic_keys.SESSION_OUTCOME_CANONICAL),
                    _savings_registry.resolve(topic_keys.HOOK_CONTEXT_INJECTED),
                    _savings_registry.resolve(topic_keys.VALIDATOR_CATCH),
                    _savings_registry.resolve(topic_keys.PATTERN_ENFORCEMENT),
                ]

                async def _savings_consumer_loop() -> None:
                    """Consume input events and produce savings estimates."""
                    import json as _json

                    _poll_interval = 2.0

                    for _input_topic in _savings_input_topics:
                        try:

                            async def _savings_on_message(
                                msg: object,
                                *,
                                _topic: str = _input_topic,
                            ) -> None:
                                _savings_estimator.ingest_event(
                                    _topic,
                                    _json.loads(msg)  # type: ignore[arg-type]
                                    if isinstance(msg, (str, bytes))
                                    else msg,
                                )

                            await event_bus.subscribe(
                                _input_topic,
                                on_message=_savings_on_message,  # type: ignore[arg-type]
                            )
                        except Exception:  # noqa: BLE001
                            logger.debug(
                                "Could not subscribe to %s for savings estimation",
                                _input_topic,
                            )

                    while True:
                        try:
                            await asyncio.sleep(_poll_interval)
                            estimates = (
                                await _savings_estimator.finalize_ready_sessions()
                            )
                            for estimate in estimates:
                                try:
                                    _envelope: ModelEventEnvelope[object] = (
                                        ModelEventEnvelope(
                                            payload=estimate,
                                            correlation_id=str(
                                                estimate.get("correlation_id", "")
                                            ),
                                            event_type="savings.estimated",
                                            source="savings_estimation_consumer",
                                        )
                                    )
                                    await event_bus.publish_envelope(
                                        _envelope,  # type: ignore[arg-type]
                                        topic=_savings_topic,
                                    )
                                except Exception:  # noqa: BLE001
                                    logger.warning(
                                        "Failed to emit savings estimate",
                                        exc_info=True,
                                    )
                        except asyncio.CancelledError:
                            break
                        except Exception:  # noqa: BLE001 — never crash the loop
                            logger.warning(
                                "Savings estimation loop iteration failed",
                                exc_info=True,
                            )

                savings_task = asyncio.create_task(
                    _savings_consumer_loop(), name="savings-estimation-consumer"
                )
                logger.info(
                    "Savings estimation consumer started (correlation_id=%s)",
                    correlation_id,
                )
            except Exception:  # noqa: BLE001 — best-effort, never blocks startup
                logger.warning(
                    "Failed to start savings estimation consumer, continuing "
                    "without it (correlation_id=%s)",
                    correlation_id,
                    exc_info=True,
                )
                savings_task = None

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

        # 4.1. Register event bus as ProtocolEventBusPublisher in service registry
        # Domain plugins (e.g., claude) resolve ProtocolEventBusPublisher from the
        # registry during wire_dispatchers(). Without this registration, plugin
        # handler wiring fails and no event consumers are started.
        if container.service_registry is not None:
            try:
                await container.service_registry.register_instance(
                    ProtocolEventBusPublisher,
                    event_bus,  # type: ignore[arg-type]  # EventBusKafka/Inmemory implements ProtocolEventBusPublisher via ProtocolEventBus
                )
                logger.info(
                    "Registered %s as ProtocolEventBusPublisher (correlation_id=%s)",
                    event_bus_type,
                    correlation_id,
                )
            except Exception:  # noqa: BLE001 — boundary: logs warning and degrades
                logger.warning(
                    "Failed to register event bus as ProtocolEventBusPublisher "
                    "(correlation_id=%s)",
                    correlation_id,
                    exc_info=True,
                )

        # 4.5. Activate domain plugins via RegistryDomainPlugin (OMN-1992)
        #
        # The plugin system replaces inline wiring of registration infrastructure.
        # Each domain plugin encapsulates its own resource creation, handler wiring,
        # dispatcher setup, and consumer startup. The kernel iterates registered
        # plugins and calls the standard lifecycle:
        #   should_activate() -> initialize() -> wire_handlers() ->
        #   wire_dispatchers() -> start_consumers()
        #
        # Plugins are shut down in LIFO order during kernel shutdown.
        plugin_registry = RegistryDomainPlugin()
        registration_plugin = PluginRegistration()
        plugin_registry.register(registration_plugin)

        # Register lightweight plugins BEFORE heavy ones. Plugin registration
        # order determines Pass 2 (start_consumers) order. PluginDelegation (3
        # topics) and PluginBuildLoop (1 topic) subscribe in seconds, while
        # PluginIntelligence (46 topics) takes ~12 minutes. If Intelligence
        # goes first, later plugins never get their consumers started before
        # the runtime is restarted or killed.

        # Try to register PluginDelegation (OMN-7040: delegation pipeline).
        try:
            plugin_registry.register(PluginDelegation())
            logger.info(
                "PluginDelegation registered (correlation_id=%s)",
                correlation_id,
            )
        except Exception:  # noqa: BLE001 — boundary: logs warning and degrades
            logger.warning(
                "PluginDelegation failed to initialize, continuing without it "
                "(correlation_id=%s)",
                correlation_id,
                exc_info=True,
            )

        # Try to register PluginBuildLoop (OMN-5113: autonomous build loop).
        try:
            plugin_registry.register(PluginBuildLoop())
            logger.info(
                "PluginBuildLoop registered (correlation_id=%s)",
                correlation_id,
            )
        except Exception:  # noqa: BLE001 — boundary: logs warning and degrades
            logger.warning(
                "PluginBuildLoop failed to initialize, continuing without it "
                "(correlation_id=%s)",
                correlation_id,
                exc_info=True,
            )

        # Try to register PluginEmitDaemon (OMN-7640: portable event publisher).
        # Gated by ONEX_EMIT_DAEMON_ENABLED=true (default false).
        try:
            from omnibase_infra.plugins.plugin_emit_daemon import PluginEmitDaemon

            plugin_registry.register(PluginEmitDaemon())
            logger.info(
                "PluginEmitDaemon registered (correlation_id=%s)",
                correlation_id,
            )
        except Exception:  # noqa: BLE001 — boundary: logs warning and degrades
            logger.warning(
                "PluginEmitDaemon failed to initialize, continuing without it "
                "(correlation_id=%s)",
                correlation_id,
                exc_info=True,
            )

        # Try to register PluginLlm (OMN-6600: LLM domain plugin).
        try:
            from omnibase_infra.adapters.llm.plugin_llm import PluginLlm

            plugin_registry.register(PluginLlm())
            logger.info(
                "PluginLlm registered (correlation_id=%s)",
                correlation_id,
            )
        except Exception:  # noqa: BLE001 — boundary: logs warning and degrades
            logger.warning(
                "PluginLlm failed to initialize, continuing without it "
                "(correlation_id=%s)",
                correlation_id,
                exc_info=True,
            )

        # Try to register PluginDlq (OMN-6601: DLQ + retry worker).
        try:
            from omnibase_infra.dlq.plugin_dlq import PluginDlq

            plugin_registry.register(PluginDlq())
            logger.info(
                "PluginDlq registered (correlation_id=%s)",
                correlation_id,
            )
        except Exception:  # noqa: BLE001 — boundary: logs warning and degrades
            logger.warning(
                "PluginDlq failed to initialize, continuing without it "
                "(correlation_id=%s)",
                correlation_id,
                exc_info=True,
            )

        # Try to import and register PluginIntelligence (graceful degradation).
        # omniintelligence is an optional dependency — kernel boots without it.
        # Registered LAST because it subscribes to 46 topics (~12 min startup).
        try:
            from omniintelligence.runtime.plugin import (  # type: ignore[import-not-found]
                PluginIntelligence,
            )

            plugin_registry.register(PluginIntelligence())
            logger.info(
                "PluginIntelligence registered (correlation_id=%s)",
                correlation_id,
            )
        except ImportError:
            logger.debug(
                "omniintelligence not installed, intelligence plugin not available "
                "(correlation_id=%s)",
                correlation_id,
            )
        except Exception:  # noqa: BLE001 — boundary: logs warning and degrades
            logger.warning(
                "PluginIntelligence failed to initialize, continuing without it "
                "(correlation_id=%s)",
                correlation_id,
                exc_info=True,
            )

        # 4.6. Discover domain plugins from entry_points (OMN-2000)
        #
        # After explicit registration, scan installed packages for plugins
        # declared under the "onex.domain_plugins" entry_point group.
        # Explicit registration takes precedence on duplicate plugin_id.
        #
        # Security: Discovery validates entry_point module paths against the
        # namespace allowlist BEFORE calling .load() (pre-import gate).
        # Post-import, isinstance(plugin, ProtocolDomainPlugin) is checked.
        try:
            security_config = ModelSecurityConfig()
            discovery_report = plugin_registry.discover_from_entry_points(
                security_config=security_config,
            )
            if discovery_report.has_errors:
                logger.warning(
                    "Plugin entry_point discovery had errors: %d entries with "
                    "import/instantiation failures (correlation_id=%s)",
                    len(
                        [
                            e
                            for e in discovery_report.entries
                            if e.status in ("import_error", "instantiation_error")
                        ]
                    ),
                    correlation_id,
                    extra={
                        "group": discovery_report.group,
                        "discovered_count": discovery_report.discovered_count,
                        "accepted": discovery_report.accepted,
                        "errors": [
                            {
                                "name": e.entry_point_name,
                                "status": e.status,
                                "reason": e.reason,
                            }
                            for e in discovery_report.entries
                            if e.status in ("import_error", "instantiation_error")
                        ],
                    },
                )
            elif discovery_report.accepted:
                logger.info(
                    "Plugin entry_point discovery: %d plugins discovered from "
                    "group '%s' (correlation_id=%s)",
                    len(discovery_report.accepted),
                    discovery_report.group,
                    correlation_id,
                    extra={
                        "accepted_plugins": discovery_report.accepted,
                        "discovered_count": discovery_report.discovered_count,
                    },
                )
            else:
                logger.debug(
                    "Plugin entry_point discovery: no new plugins found in "
                    "group '%s' (correlation_id=%s)",
                    discovery_report.group,
                    correlation_id,
                )
        except Exception:  # noqa: BLE001 — boundary: logs warning and degrades
            logger.warning(
                "Plugin entry_point discovery failed; continuing with "
                "explicitly registered plugins only (correlation_id=%s)",
                correlation_id,
                exc_info=True,
            )

        # Create typed node identity for plugin subscriptions (OMN-1602)
        plugin_node_identity: ModelNodeIdentity | None = None
        if config.name:
            plugin_node_identity = ModelNodeIdentity(
                env=environment,
                service=config.name,
                node_name=config.name,
                version=config.contract_version or "v1",
            )
        else:
            # Graceful degradation (OMN-1992): config.name absence is logged
            # rather than raising ProtocolConfigurationError so kernels with
            # optional introspection can still boot.  Plugins that require
            # node_identity (e.g. PluginRegistration.start_consumers) will
            # return a "skipped" result instead of failing.
            logger.error(
                "runtime_config.yaml missing 'name' field — plugin consumers "
                "will not subscribe to introspection events. "
                "Set 'name' in runtime_config.yaml to enable introspection "
                "(correlation_id=%s)",
                correlation_id,
            )

        # 4.7. Create MessageDispatchEngine (OMN-2050)
        #
        # The dispatch engine is the single routing component for all events.
        # It is instantiated here, set on plugin_config so plugins can register
        # dispatchers during wire_dispatchers(), then frozen after all plugins
        # have registered their dispatchers but BEFORE any consumers start.
        #
        # This two-pass lifecycle ensures:
        #   Pass 1: initialize -> wire_handlers -> wire_dispatchers (all plugins)
        #   Freeze: dispatch_engine.freeze()
        #   Pass 2: start_consumers (all plugins)
        from omnibase_infra.runtime.service_message_dispatch_engine import (
            MessageDispatchEngine,
        )

        dispatch_engine = MessageDispatchEngine(logger=logger)
        logger.debug(
            "MessageDispatchEngine created (correlation_id=%s)",
            correlation_id,
        )

        # Create shared plugin configuration
        plugin_config = ModelDomainPluginConfig(
            container=container,
            event_bus=event_bus,
            correlation_id=correlation_id,
            input_topic=config.input_topic,
            output_topic=config.output_topic,
            consumer_group=config.consumer_group,
            dispatch_engine=dispatch_engine,
            node_identity=plugin_node_identity,
            kafka_bootstrap_servers=kafka_bootstrap_servers,
        )

        # Activate plugins using two-pass lifecycle (OMN-2050, OMN-2089)
        #
        # Pass 1: should_activate -> initialize -> validate_handshake ->
        #         wire_handlers -> wire_dispatchers
        #   The handshake gate (OMN-2089) runs between initialize() and
        #   wire_handlers(). If validate_handshake() fails, the kernel
        #   aborts before wiring handlers/dispatchers/consumers.
        #
        #   Phase state machine:
        #   INITIALIZING -> HANDSHAKE_VALIDATE -> HANDSHAKE_ATTEST -> WIRING -> READY
        #
        #   All plugins register their dispatchers with the engine before it is frozen.
        #
        # Freeze: dispatch_engine.freeze() after all wire_dispatchers() complete
        #
        # Pass 2: start_consumers for all activated plugins
        #   Consumers only start after the engine is frozen and read-only.
        #
        # This ordering prevents a race where a late plugin's wire_dispatchers()
        # could modify the engine while an early plugin's consumer is already
        # dispatching messages through it.
        plugin_activation_start = time.time()

        # --- Pass 1: Initialize, validate handshake, wire handlers, wire dispatchers ---
        for plugin in plugin_registry.get_all():
            plugin_id = plugin.plugin_id

            try:
                # 1. Check activation
                if not plugin.should_activate(plugin_config):
                    logger.info(
                        "Plugin '%s' skipped (not activated) (correlation_id=%s)",
                        plugin_id,
                        correlation_id,
                    )
                    continue

                # 2. Initialize (create pools, connections, resources)
                init_result = await plugin.initialize(plugin_config)
                if not init_result:
                    logger.warning(
                        "Plugin '%s' initialization failed: %s (correlation_id=%s)",
                        plugin_id,
                        init_result.get_error_message_or_default(),
                        correlation_id,
                    )
                    continue

                # Track for shutdown immediately after successful init so
                # allocated resources (DB pools, Kafka producers) are always
                # cleaned up even if later lifecycle steps fail.
                activated_plugins.append(plugin)

                # 3. HANDSHAKE_VALIDATE: Run prerequisite checks (OMN-2089)
                # The handshake gate ensures all B1-B3 checks pass before
                # any consumers, dispatchers, or handlers are wired.
                # Plugins that don't implement validate_handshake() pass
                # by default (optional method).
                if hasattr(plugin, "validate_handshake") and callable(
                    getattr(plugin, "validate_handshake", None)
                ):
                    handshake_result = await plugin.validate_handshake(plugin_config)
                    if not handshake_result:
                        logger.error(
                            "Plugin '%s' handshake validation FAILED: %s — "
                            "aborting before wiring handlers (correlation_id=%s)",
                            plugin_id,
                            handshake_result.error_message or "unknown",
                            correlation_id,
                            extra={
                                "checks": [
                                    {
                                        "name": c.check_name,
                                        "passed": c.passed,
                                        "message": c.message,
                                    }
                                    for c in handshake_result.checks
                                ],
                            },
                        )
                        continue
                    logger.info(
                        "Plugin '%s' handshake ATTESTED (%d checks passed) "
                        "(correlation_id=%s)",
                        plugin_id,
                        len(handshake_result.checks),
                        correlation_id,
                    )
                else:
                    logger.debug(
                        "Plugin '%s' has no validate_handshake() — default pass "
                        "(correlation_id=%s)",
                        plugin_id,
                        correlation_id,
                    )

                # 4. Wire handlers (WIRING phase)
                wire_result = await plugin.wire_handlers(plugin_config)
                if not wire_result:
                    logger.warning(
                        "Plugin '%s' handler wiring failed: %s — consumers will "
                        "NOT be started for this plugin (correlation_id=%s)",
                        plugin_id,
                        wire_result.get_error_message_or_default(),
                        correlation_id,
                    )
                    continue

                # 5. Wire dispatchers (non-fatal if skipped)
                dispatch_result = await plugin.wire_dispatchers(plugin_config)
                if not dispatch_result:
                    logger.warning(
                        "Plugin '%s' dispatcher wiring failed: %s (correlation_id=%s)",
                        plugin_id,
                        dispatch_result.get_error_message_or_default(),
                        correlation_id,
                    )

                # Plugin completed handler wiring successfully — safe to start
                # consumers in Pass 2. Plugins that failed wire_handlers() are
                # excluded via the `continue` above, preventing consumers from
                # starting with no handlers/dispatchers wired.
                ready_plugins.append(plugin)

                logger.info(
                    "Plugin '%s' wiring completed (correlation_id=%s)",
                    plugin_id,
                    correlation_id,
                )
            except (
                DbOwnershipMismatchError,
                DbOwnershipMissingError,
                SchemaFingerprintMismatchError,
                SchemaFingerprintMissingError,
                EventRegistryFingerprintMismatchError,
                EventRegistryFingerprintMissingError,
            ):
                # Hard gates -- propagate to kill the kernel.
                # DB ownership errors (OMN-2085): wrong database.
                # Schema fingerprint errors (OMN-2087): schema drift.
                # Event registry fingerprint errors (OMN-2088): event drift.
                # These are raised by validate_handshake() and must not be
                # swallowed.
                raise
            except Exception:  # noqa: BLE001 — boundary: logs warning and degrades
                logger.warning(
                    "Plugin '%s' failed during lifecycle activation "
                    "(correlation_id=%s)",
                    plugin_id,
                    correlation_id,
                    exc_info=True,
                )
                # Safety: if exception occurred before the plugin was tracked
                # in activated_plugins (e.g. during should_activate or
                # initialize), attempt best-effort shutdown to prevent
                # resource leaks from partially-initialized plugins.
                if plugin not in activated_plugins:
                    try:
                        await plugin.shutdown(plugin_config)
                    except Exception:  # noqa: BLE001 — boundary: catch-all for resilience
                        logger.debug(
                            "Best-effort shutdown of untracked plugin '%s' "
                            "also failed (correlation_id=%s)",
                            plugin_id,
                            correlation_id,
                            exc_info=True,
                        )

        # --- Auto-wiring: Contract discovery + handler wiring (OMN-7656) ---
        #
        # After explicit plugins register their handlers/dispatchers, auto-wiring
        # discovers contracts from onex.nodes entry points and wires handlers
        # that are NOT already claimed by explicit plugins.
        #
        # Coexistence: explicit plugins always take precedence. Auto-wiring only
        # picks up contracts that have no plugin handler wired for the same topics.
        # Topic collision detection logs warnings for overlapping subscriptions.
        auto_wiring_report = None
        lifecycle_executor = None
        try:
            from omnibase_infra.runtime.auto_wiring import (
                LifecycleHookExecutor,
                discover_contracts,
                wire_from_manifest,
            )

            # 1. Discover all contracts from installed packages
            auto_wiring_start = time.time()
            manifest = discover_contracts()
            logger.info(
                "Auto-wiring discovery: %d contracts found, %d errors "
                "(correlation_id=%s)",
                manifest.total_discovered,
                manifest.total_errors,
                correlation_id,
                extra={
                    "discovered": [c.name for c in manifest.contracts],
                    "errors": [
                        {"entry_point": e.entry_point_name, "error": e.error}
                        for e in manifest.errors
                    ],
                },
            )

            if manifest.total_discovered > 0:
                # 2. Collect topics already claimed by explicit plugins
                #    by inspecting registered routes on the dispatch engine
                claimed_topic_patterns: set[str] = set()
                for route in dispatch_engine._routes.values():
                    claimed_topic_patterns.add(route.topic_pattern)

                # 3. Run lifecycle hooks (on_start + validate_handshake)
                lifecycle_executor = LifecycleHookExecutor()
                for contract in manifest.contracts:
                    if not contract.handler_routing:
                        continue
                    lifecycle_hooks_raw = getattr(contract, "lifecycle_hooks", None)
                    if (
                        lifecycle_hooks_raw is not None
                        and lifecycle_hooks_raw.has_hooks()
                    ):
                        context_kwargs: dict[str, object] = {
                            "handler_id": contract.name,
                            "node_kind": contract.node_type,
                            "contract_version": str(contract.contract_version),
                        }
                        hook_results = await lifecycle_executor.execute_startup(
                            lifecycle_hooks_raw, context_kwargs
                        )
                        for hr in hook_results:
                            if not hr.success:
                                logger.warning(
                                    "Auto-wiring lifecycle hook '%s' failed for "
                                    "'%s': %s (correlation_id=%s)",
                                    hr.hook_name,
                                    contract.name,
                                    hr.error_message,
                                    correlation_id,
                                )

                quarantined = lifecycle_executor.get_quarantined_contracts()
                quarantined_names = {q.handler_id for q in quarantined}
                if quarantined:
                    logger.warning(
                        "Auto-wiring: %d contracts quarantined after handshake "
                        "failure (correlation_id=%s)",
                        len(quarantined),
                        correlation_id,
                        extra={
                            "quarantined": [
                                {
                                    "handler_id": q.handler_id,
                                    "reason": q.failure_reason.value,
                                }
                                for q in quarantined
                            ],
                        },
                    )

                # 4. Filter manifest to exclude quarantined contracts
                from omnibase_infra.runtime.auto_wiring.models import (
                    ModelAutoWiringManifest,
                )

                filtered_contracts = tuple(
                    c for c in manifest.contracts if c.name not in quarantined_names
                )
                filtered_manifest = ModelAutoWiringManifest(
                    contracts=filtered_contracts,
                    errors=manifest.errors,
                )

                # 5. Wire handlers into dispatch engine
                auto_wiring_report = await wire_from_manifest(
                    manifest=filtered_manifest,
                    dispatch_engine=dispatch_engine,
                    event_bus=event_bus,
                    environment=environment,
                )

                # 6. Topic collision detection: warn for auto-wired topics
                #    that overlap with explicit plugin routes
                if auto_wiring_report:
                    for result in auto_wiring_report.results:
                        for topic in result.topics_subscribed:
                            for pattern in claimed_topic_patterns:
                                if _topic_matches_pattern(topic, pattern):
                                    logger.warning(
                                        "Topic collision: auto-wired topic '%s' "
                                        "(contract=%s) overlaps with explicit "
                                        "plugin route pattern '%s' "
                                        "(correlation_id=%s)",
                                        topic,
                                        result.contract_name,
                                        pattern,
                                        correlation_id,
                                    )

                auto_wiring_duration = time.time() - auto_wiring_start
                logger.info(
                    "Auto-wiring completed in %.3fs: wired=%d skipped=%d "
                    "failed=%d quarantined=%d (correlation_id=%s)",
                    auto_wiring_duration,
                    auto_wiring_report.total_wired if auto_wiring_report else 0,
                    auto_wiring_report.total_skipped if auto_wiring_report else 0,
                    auto_wiring_report.total_failed if auto_wiring_report else 0,
                    len(quarantined),
                    correlation_id,
                    extra={
                        "duration_seconds": auto_wiring_duration,
                        "wired": [
                            r.contract_name
                            for r in (
                                auto_wiring_report.results if auto_wiring_report else ()
                            )
                            if r.outcome.value == "wired"
                        ],
                    },
                )
        except Exception:  # noqa: BLE001 — boundary: auto-wiring degrades gracefully
            logger.warning(
                "Auto-wiring failed; continuing with explicitly registered "
                "plugins only (correlation_id=%s)",
                correlation_id,
                exc_info=True,
            )

        # --- Freeze dispatch engine ---
        # All plugins and auto-wired handlers have registered their dispatchers.
        # Freeze the engine to make it read-only and thread-safe for concurrent dispatch.
        dispatch_engine.freeze()
        logger.info(
            "MessageDispatchEngine frozen after all wire_dispatchers() "
            "and auto-wiring (correlation_id=%s)",
            correlation_id,
        )

        # --- Pass 2: Start consumers for ready plugins only ---
        # ready_plugins is a subset of activated_plugins: only plugins that
        # completed wire_handlers() successfully. This prevents starting
        # consumers for plugins with no handlers/dispatchers wired.
        for plugin in ready_plugins:
            plugin_id = plugin.plugin_id
            try:
                consumer_result = await plugin.start_consumers(plugin_config)
                if consumer_result and consumer_result.unsubscribe_callbacks:
                    plugin_unsubscribe_callbacks.extend(
                        consumer_result.unsubscribe_callbacks
                    )
                logger.info(
                    "Plugin '%s' consumers started (correlation_id=%s)",
                    plugin_id,
                    correlation_id,
                )
            except Exception:  # noqa: BLE001 — boundary: logs warning and degrades
                logger.warning(
                    "Plugin '%s' failed to start consumers (correlation_id=%s)",
                    plugin_id,
                    correlation_id,
                    exc_info=True,
                )

        plugin_activation_duration = time.time() - plugin_activation_start
        logger.info(
            "Plugin activation completed in %.3fs: %d/%d plugins activated "
            "(correlation_id=%s)",
            plugin_activation_duration,
            len(activated_plugins),
            len(plugin_registry),
            correlation_id,
            extra={
                "activated_plugins": [p.plugin_id for p in activated_plugins],
                "duration_seconds": plugin_activation_duration,
            },
        )

        # 4.9. Wire ContractRegistrationEventRouter if contract_registry.enabled
        # This router subscribes to contract lifecycle events (registration,
        # deregistration, heartbeat) and routes them to the ContractRegistryReducer.
        # The router also runs an internal tick timer for staleness computation.
        # Uses postgres_pool from the registration plugin.
        postgres_pool = registration_plugin.postgres_pool
        if config.contract_registry.enabled and postgres_pool is not None:
            # Import postgres handlers for contract persistence
            # Deferred import to avoid loading heavy dependencies when not needed
            from omnibase_infra.nodes.node_contract_persistence_effect.handlers import (
                HandlerPostgresCleanupTopics,
                HandlerPostgresContractUpsert,
                HandlerPostgresDeactivate,
                HandlerPostgresHeartbeat,
                HandlerPostgresMarkStale,
                HandlerPostgresTopicUpdate,
            )

            # Create effect handlers keyed by intent_type
            # These handlers execute PostgreSQL operations for intents from the reducer
            # Note: Handlers implement ProtocolIntentEffect duck-typing style with
            # more specific payload types. Cast tells mypy they satisfy the protocol.
            contract_effect_handlers: dict[str, ProtocolIntentEffect] = {
                "postgres.upsert_contract": cast(
                    "ProtocolIntentEffect",
                    HandlerPostgresContractUpsert(postgres_pool),
                ),
                "postgres.update_topic": cast(
                    "ProtocolIntentEffect",
                    HandlerPostgresTopicUpdate(postgres_pool),
                ),
                "postgres.mark_stale": cast(
                    "ProtocolIntentEffect",
                    HandlerPostgresMarkStale(postgres_pool),
                ),
                "postgres.update_heartbeat": cast(
                    "ProtocolIntentEffect",
                    HandlerPostgresHeartbeat(postgres_pool),
                ),
                "postgres.deactivate_contract": cast(
                    "ProtocolIntentEffect",
                    HandlerPostgresDeactivate(postgres_pool),
                ),
                "postgres.cleanup_topic_references": cast(
                    "ProtocolIntentEffect",
                    HandlerPostgresCleanupTopics(postgres_pool),
                ),
            }

            # Create reducer and router
            contract_reducer = ContractRegistryReducer()
            contract_router = ContractRegistrationEventRouter(
                container=container,
                reducer=contract_reducer,
                effect_handlers=contract_effect_handlers,
                event_bus=event_bus,
                tick_interval_seconds=config.contract_registry.tick_interval_seconds,
            )

            logger.info(
                "ContractRegistrationEventRouter created (correlation_id=%s)",
                correlation_id,
                extra={
                    "tick_interval_seconds": config.contract_registry.tick_interval_seconds,
                    "handler_count": len(contract_effect_handlers),
                },
            )
        else:
            logger.debug(
                "Contract registry disabled or no postgres_pool (correlation_id=%s)",
                correlation_id,
                extra={
                    "contract_registry_enabled": config.contract_registry.enabled,
                    "postgres_pool_available": postgres_pool is not None,
                },
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
        runtime_config_dict = cast("dict[str, object]", config.model_dump(mode="json"))
        if config.name:
            runtime_config_dict["service_name"] = config.name
            runtime_config_dict["node_name"] = config.name

        # 6.1 Create introspection service (OMN-5609)
        # Wire contract data into introspection so published events include
        # metadata.description, event_bus topics, and contract_capabilities.
        # Without this, RuntimeHostProcess receives introspection_service=None
        # and silently skips all introspection publishing.
        #
        # The registration orchestrator contract is used as the primary
        # contract source because it is the principal node in the kernel
        # and declares all subscribe/publish topics.
        introspection_service = None
        if config.name:
            try:
                from omnibase_infra.services.service_node_introspection import (
                    ServiceNodeIntrospection,
                )

                # Use registration orchestrator contract (primary kernel node)
                _registration_contract_dir = (
                    Path(__file__).resolve().parent.parent
                    / "nodes"
                    / "node_registration_orchestrator"
                )
                introspection_service = ServiceNodeIntrospection.from_contract_dir(
                    contracts_dir=_registration_contract_dir,
                    event_bus=event_bus,
                    node_name=config.name,
                    environment=environment,
                )
                logger.info(
                    "Introspection service created for node '%s' (correlation_id=%s)",
                    config.name,
                    correlation_id,
                )
            except Exception as intro_err:  # noqa: BLE001 — boundary: logs warning and degrades
                logger.warning(
                    "Failed to create introspection service, introspection "
                    "will be disabled: %s (correlation_id=%s)",
                    intro_err,
                    correlation_id,
                    extra={"error_type": type(intro_err).__name__},
                )

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
            # OMN-2050: Wire dispatch engine so RuntimeHostProcess skips the
            # legacy _on_message subscription and routes through
            # EventBusSubcontractWiring instead.
            dispatch_engine=dispatch_engine,
            # OMN-5609: Wire introspection service so the runtime publishes
            # introspection events with metadata, capabilities, and event_bus
            # fields from the contract.
            # NOTE(OMN-5609): ServiceNodeIntrospection structurally satisfies
            # ProtocolNodeIntrospection via MixinNodeIntrospection, but mypy
            # cannot verify structural protocol conformance across mixin chains.
            introspection_service=introspection_service,  # type: ignore[arg-type]
            # OMN-6334: Pass contract-driven runtime config so RuntimeHostProcess
            # uses contract values instead of DEFAULT_* constants.
            runtime_node_graph_config=node_graph_config,
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

        # Track which signal triggered shutdown for diagnostics (OMN-3591)
        shutdown_reason: str = "unknown"

        def handle_shutdown(sig: signal.Signals) -> None:
            """Handle shutdown signal with correlation tracking."""
            nonlocal shutdown_reason
            shutdown_reason = f"signal:{sig.name}"
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
                nonlocal shutdown_reason
                sig = signal.Signals(signum)
                shutdown_reason = f"signal:{sig.name}"
                logger.info(
                    "Received %s, initiating graceful shutdown... (correlation_id=%s)",
                    sig.name,
                    correlation_id,
                )
                loop.call_soon_threadsafe(shutdown_event.set)

            signal.signal(signal.SIGINT, windows_handler)

        # 8. Start HTTP health server for Docker/K8s probes
        # Started BEFORE runtime.start() so Docker health checks pass during
        # the slow Kafka consumer-join phase (runtime.start() can take 10+ min).
        # The health server has no dependency on the runtime being started.
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

        # 9. Start runtime
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

        # Initialize WiringHealthChecker with correct sources (OMN-6515)
        # MUST be after runtime.start() because EventBusSubcontractWiring
        # is created during RuntimeHostProcess.start().
        #
        # Source roles:
        #   emission_source  = event_bus (EventBusKafka -> MixinEmissionCounter)
        #   consumption_source = runtime.event_bus_wiring (EventBusSubcontractWiring -> MixinConsumptionCounter)
        #
        # Previously, event_bus was incorrectly used for both, causing
        # AttributeError on get_consumption_counts() (OMN-6515).
        if use_kafka:
            try:
                consumption_source = runtime.event_bus_wiring
                emission_source = event_bus
                if consumption_source is not None:
                    _wiring_health_interval = float(
                        os.environ.get("WIRING_HEALTH_EMIT_INTERVAL", "60")
                    )
                    wiring_health_checker = WiringHealthChecker(
                        emission_source=emission_source,  # type: ignore[arg-type]
                        consumption_source=consumption_source,
                        environment=environment,
                        event_bus=event_bus,  # type: ignore[arg-type]
                    )

                    async def _wiring_health_loop() -> None:
                        """Periodic loop that computes and emits wiring health snapshots."""
                        assert wiring_health_checker is not None
                        while True:
                            try:
                                _cid = generate_correlation_id()
                                metrics = wiring_health_checker.compute_health(
                                    correlation_id=_cid,
                                )
                                await wiring_health_checker.emit_snapshot(
                                    metrics=metrics,
                                    correlation_id=_cid,
                                )
                            except asyncio.CancelledError:
                                raise
                            except Exception:  # noqa: BLE001
                                logger.warning(
                                    "Wiring health snapshot emission failed",
                                    exc_info=True,
                                )
                            await asyncio.sleep(_wiring_health_interval)

                    wiring_health_task = asyncio.create_task(
                        _wiring_health_loop(),
                        name="wiring-health-emit",
                    )
                    logger.info(
                        "WiringHealthChecker periodic emission started "
                        "(interval=%ss, correlation_id=%s)",
                        _wiring_health_interval,
                        correlation_id,
                    )
                else:
                    logger.info(
                        "WiringHealthChecker skipped: no EventBusSubcontractWiring "
                        "(dispatch_engine not configured) (correlation_id=%s)",
                        correlation_id,
                    )
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to start WiringHealthChecker emission, continuing without it "
                    "(correlation_id=%s)",
                    correlation_id,
                    exc_info=True,
                )
                wiring_health_checker = None
                wiring_health_task = None

        # 9.5. Introspection event consumer is now started by domain plugins
        # during plugin activation (step 4.5). The PluginRegistration.start_consumers()
        # method handles subscription using node_identity and EnumConsumerGroupPurpose.

        # 9.6. Start contract registry event consumer if router is available
        # This consumer subscribes to 3 Kafka topics for contract lifecycle events
        # and routes them to the ContractRegistryReducer for projection.
        has_subscribe = hasattr(event_bus, "subscribe") and callable(
            getattr(event_bus, "subscribe", None)
        )
        if contract_router is not None and has_subscribe:
            # Create typed node identity for contract registry subscriptions
            contract_node_identity = ModelNodeIdentity(
                env=environment,
                service=config.name or "onex-kernel",
                node_name="contract-registry",
                version=config.contract_version or "v1",
            )

            # Subscribe to 3 contract lifecycle topics with same identity
            contract_subscribe_start_time = time.time()

            # Resolve realm-agnostic topic names via TopicResolver (no env prefix).
            # Topics are realm-agnostic in ONEX; the environment/realm is enforced
            # via envelope identity and consumer group naming, not topic names.
            topic_resolver = TopicResolver()
            try:
                contract_registered_topic = topic_resolver.resolve(
                    SUFFIX_CONTRACT_REGISTERED,
                    correlation_id=correlation_id,
                )
                contract_deregistered_topic = topic_resolver.resolve(
                    SUFFIX_CONTRACT_DEREGISTERED,
                    correlation_id=correlation_id,
                )
                node_heartbeat_topic = topic_resolver.resolve(
                    SUFFIX_NODE_HEARTBEAT,
                    correlation_id=correlation_id,
                )
            except TopicResolutionError as e:
                # TopicResolutionError is a ProtocolConfigurationError with a
                # guaranteed infra_context (including correlation_id). Log at
                # warning level so operators can diagnose configuration issues,
                # then re-raise with kernel-specific context message.
                logger.warning(
                    "TopicResolver rejected topic suffix during kernel bootstrap "
                    "(correlation_id=%s): %s",
                    e.infra_context.correlation_id,
                    e,
                    extra={
                        "correlation_id": str(e.infra_context.correlation_id),
                        "transport_type": "kafka",
                        "operation": "resolve_topic",
                    },
                )
                raise ProtocolConfigurationError(
                    f"Invalid topic suffix in runtime configuration: {e}",
                    context=e.infra_context,
                ) from e

            logger.info(
                "Subscribing to contract registry events on event bus (correlation_id=%s)",
                correlation_id,
                extra={
                    "topics": [
                        contract_registered_topic,
                        contract_deregistered_topic,
                        node_heartbeat_topic,
                    ],
                    "node_identity": {
                        "env": contract_node_identity.env,
                        "service": contract_node_identity.service,
                        "node_name": contract_node_identity.node_name,
                        "version": contract_node_identity.version,
                    },
                    "purpose": EnumConsumerGroupPurpose.CONTRACT_REGISTRY.value,
                },
            )

            contract_unsub_registered = await event_bus.subscribe(
                topic=contract_registered_topic,
                node_identity=contract_node_identity,
                on_message=contract_router.handle_message,
                purpose=EnumConsumerGroupPurpose.CONTRACT_REGISTRY,
                required_for_readiness=True,
            )
            contract_unsub_deregistered = await event_bus.subscribe(
                topic=contract_deregistered_topic,
                node_identity=contract_node_identity,
                on_message=contract_router.handle_message,
                purpose=EnumConsumerGroupPurpose.CONTRACT_REGISTRY,
                required_for_readiness=True,
            )
            contract_unsub_heartbeat = await event_bus.subscribe(
                topic=node_heartbeat_topic,
                node_identity=contract_node_identity,
                on_message=contract_router.handle_message,
                purpose=EnumConsumerGroupPurpose.CONTRACT_REGISTRY,
                required_for_readiness=True,
            )

            # Start the router's tick timer
            await contract_router.start()

            contract_subscribe_duration = time.time() - contract_subscribe_start_time
            logger.info(
                "Contract registry event consumers started successfully in %.3fs (correlation_id=%s)",
                contract_subscribe_duration,
                correlation_id,
                extra={
                    "topics_count": 3,
                    "tick_interval_seconds": contract_router.tick_interval_seconds,
                    "subscribe_duration_seconds": contract_subscribe_duration,
                    "event_bus_type": event_bus_type,
                },
            )

        # 9.7. Start runtime error triage consumer (OMN-5655)
        # Subscribes to runtime-error events and routes them to the
        # HandlerRuntimeErrorTriage for first-match-wins triage processing.
        # (triage_unsub pre-declared before try block)
        if postgres_pool is not None and has_subscribe:
            try:
                from omnibase_infra.nodes.node_runtime_error_triage_effect.handlers.handler_runtime_error_triage import (
                    HandlerRuntimeErrorTriage,
                )

                triage_handler = HandlerRuntimeErrorTriage(db_pool=postgres_pool)

                triage_topic_resolver = TopicResolver()
                runtime_error_topic = triage_topic_resolver.resolve(
                    SUFFIX_RUNTIME_ERROR,
                    correlation_id=correlation_id,
                )
                triage_node_identity = ModelNodeIdentity(
                    env=environment,
                    service=config.name or "onex-kernel",
                    node_name="runtime-error-triage",
                    version="v1",
                )

                from omnibase_infra.event_bus.models.model_event_message import (
                    ModelEventMessage,
                )

                async def _triage_on_message(
                    message: ModelEventMessage,
                ) -> None:
                    """Deserialize runtime error event and dispatch to triage handler."""
                    import json as _json

                    from omnibase_infra.models.health.model_runtime_error_event import (
                        ModelRuntimeErrorEvent,
                    )

                    try:
                        payload = _json.loads(message.value)
                        event = ModelRuntimeErrorEvent.model_validate(payload)
                        await triage_handler.handle(event)
                    except Exception:  # noqa: BLE001 — boundary: consumer must not crash
                        logger.warning(
                            "Failed to process runtime error triage event",
                            exc_info=True,
                        )

                triage_unsub = await event_bus.subscribe(
                    topic=runtime_error_topic,
                    node_identity=triage_node_identity,
                    on_message=_triage_on_message,
                    purpose=EnumConsumerGroupPurpose.CONSUME,
                    required_for_readiness=False,
                )

                logger.info(
                    "Runtime error triage consumer started (topic=%s, correlation_id=%s)",
                    runtime_error_topic,
                    correlation_id,
                )
            except Exception:  # noqa: BLE001 — boundary: triage is non-critical
                logger.warning(
                    "Failed to start runtime error triage consumer (correlation_id=%s)",
                    correlation_id,
                    exc_info=True,
                )

        # Calculate total bootstrap time
        bootstrap_duration = time.time() - bootstrap_start_time

        # Display startup banner with key configuration
        # Get registration status from plugin (encapsulates backend details)
        registration_status = registration_plugin.get_status_line()

        # Contract registry status for banner
        if contract_router is not None:
            contract_registry_status = (
                f"enabled (tick: {config.contract_registry.tick_interval_seconds}s)"
            )
        else:
            contract_registry_status = "disabled"

        # Plugin summary for banner
        plugin_names = [p.plugin_id for p in activated_plugins]

        # Runtime profile for operator disambiguation (OMN-3591)
        runtime_profile = os.getenv("RUNTIME_PROFILE", "default")

        banner_lines = [
            "=" * 60,
            f"ONEX Runtime Kernel v{KERNEL_VERSION}",
            f"Profile: {runtime_profile} (PID {os.getpid()})",
            f"Environment: {environment}",
            f"Contracts: {contracts_dir}",
            f"Event Bus: {event_bus_type} (group: {config.consumer_group})",
            f"Topics: {config.input_topic} -> {config.output_topic}",
            f"Registration: {registration_status}",
            f"Contract Registry: {contract_registry_status}",
            f"Plugins: {', '.join(plugin_names) if plugin_names else 'none'}",
            (
                f"Auto-wiring: {auto_wiring_report.total_wired} wired, "
                f"{auto_wiring_report.total_skipped} skipped, "
                f"{auto_wiring_report.total_failed} failed"
                if auto_wiring_report is not None
                else "Auto-wiring: disabled (no contracts discovered)"
            ),
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

        # Flush log handlers so Docker log driver captures all startup messages
        # before the process blocks on shutdown_event.wait(). Without this,
        # buffered log output may not appear in `docker logs` until the process
        # exits, making it look like the kernel never entered the run loop
        # (OMN-3591).
        for handler in logging.root.handlers:
            handler.flush()

        # Explicit run-loop entry log (OMN-3591)
        # This message confirms the kernel reached the blocking wait and did
        # not exit prematurely during bootstrap. If this message is absent
        # from container logs, the kernel exited before entering the run loop.
        logger.info(
            "RUN_LOOP_ENTERED: Kernel idle, waiting for shutdown signal "
            "(profile=%s, pid=%d, correlation_id=%s)",
            runtime_profile,
            os.getpid(),
            correlation_id,
        )
        # Flush again to guarantee the run-loop-entered message is visible
        for handler in logging.root.handlers:
            handler.flush()

        # Wait for shutdown signal
        await shutdown_event.wait()

        grace_period = config.shutdown.grace_period_seconds
        shutdown_start_time = time.time()
        logger.info(
            "RUN_LOOP_EXITED: Shutdown signal received (reason=%s, timeout=%ss, "
            "correlation_id=%s)",
            shutdown_reason,
            grace_period,
            correlation_id,
        )

        # Stop runtime FIRST so introspection tasks flush their final events
        # while the event bus is still active. Moving this before consumer
        # unsubscribe fixes the ~29 introspection errors per shutdown cycle
        # (OMN-3593).
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

        # Stop plugin consumers (unsubscribe callbacks from start_consumers)
        for unsub_callback in plugin_unsubscribe_callbacks:
            try:
                await unsub_callback()
            except Exception as consumer_stop_error:  # noqa: BLE001 — boundary: logs warning and degrades
                logger.warning(
                    "Failed to stop plugin consumer: %s (correlation_id=%s)",
                    sanitize_error_message(consumer_stop_error),
                    correlation_id,
                )
        plugin_unsubscribe_callbacks.clear()

        # Stop contract registry router and consumers
        if contract_router is not None:
            try:
                await contract_router.stop()
                logger.debug(
                    "Contract registry router stopped (correlation_id=%s)",
                    correlation_id,
                )
            except Exception as router_stop_error:  # noqa: BLE001 — boundary: logs warning and degrades
                logger.warning(
                    "Failed to stop contract registry router: %s (correlation_id=%s)",
                    sanitize_error_message(router_stop_error),
                    correlation_id,
                )
            contract_router = None

        # Unsubscribe from contract registry topics
        for unsub_name, unsub_func in [
            ("contract-registered", contract_unsub_registered),
            ("contract-deregistered", contract_unsub_deregistered),
            ("node-heartbeat", contract_unsub_heartbeat),
        ]:
            if unsub_func is not None:
                try:
                    await unsub_func()
                    logger.debug(
                        "Contract registry consumer %s stopped (correlation_id=%s)",
                        unsub_name,
                        correlation_id,
                    )
                except Exception as unsub_error:  # noqa: BLE001 — boundary: logs warning and degrades
                    logger.warning(
                        "Failed to stop contract registry consumer %s: %s (correlation_id=%s)",
                        unsub_name,
                        sanitize_error_message(unsub_error),
                        correlation_id,
                    )
        contract_unsub_registered = None
        contract_unsub_deregistered = None
        contract_unsub_heartbeat = None

        # Unsubscribe runtime error triage consumer (OMN-5655)
        if triage_unsub is not None:
            try:
                await triage_unsub()
                logger.debug(
                    "Runtime error triage consumer stopped (correlation_id=%s)",
                    correlation_id,
                )
            except Exception as unsub_error:  # noqa: BLE001 — boundary: logs warning and degrades
                logger.warning(
                    "Failed to stop runtime error triage consumer: %s (correlation_id=%s)",
                    sanitize_error_message(unsub_error),
                    correlation_id,
                )
            triage_unsub = None

        # Stop WiringHealthChecker periodic emission (OMN-6133)
        if wiring_health_task is not None:
            wiring_health_task.cancel()
            try:
                await wiring_health_task
            except asyncio.CancelledError:
                pass
            logger.debug(
                "WiringHealthChecker periodic emission stopped (correlation_id=%s)",
                correlation_id,
            )
            wiring_health_task = None

        # Stop baselines batch compute loop and close its pool
        if baselines_task is not None:
            baselines_task.cancel()
            try:
                await baselines_task
            except asyncio.CancelledError:
                pass
            logger.debug(
                "Baselines batch compute loop stopped (correlation_id=%s)",
                correlation_id,
            )
            baselines_task = None
        if _baselines_pool is not None:
            try:
                await _baselines_pool.close()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass
            _baselines_pool = None

        # Stop savings estimation consumer
        if savings_task is not None:
            savings_task.cancel()
            try:
                await savings_task
            except asyncio.CancelledError:
                pass
            logger.debug(
                "Savings estimation consumer stopped (correlation_id=%s)",
                correlation_id,
            )
            savings_task = None

        # Stop ServiceLlmEndpointHealth (OMN-6135)
        if llm_health_service is not None:
            try:
                await llm_health_service.stop()
                logger.debug(
                    "ServiceLlmEndpointHealth stopped (correlation_id=%s)",
                    correlation_id,
                )
            except Exception:  # noqa: BLE001 — boundary: best-effort cleanup
                logger.warning(
                    "Error stopping ServiceLlmEndpointHealth (correlation_id=%s)",
                    correlation_id,
                    exc_info=True,
                )
            llm_health_service = None

        # Stop RuntimeLogEventBridge (OMN-5525)
        if runtime_log_bridge is not None:
            try:
                allowlist_raw = os.environ.get(
                    "RUNTIME_LOG_BRIDGE_ALLOWLIST",
                    "aiokafka.consumer,asyncpg,aiohttp",
                )
                allowlist = [
                    name.strip() for name in allowlist_raw.split(",") if name.strip()
                ]
                runtime_log_bridge.detach_from_loggers(allowlist)
                await runtime_log_bridge.stop()
                # Stop the bridge's producer
                if runtime_log_bridge._producer is not None:
                    await runtime_log_bridge._producer.stop()
                logger.debug(
                    "RuntimeLogEventBridge stopped (correlation_id=%s)",
                    correlation_id,
                )
            except Exception:  # noqa: BLE001 — boundary: best-effort cleanup
                logger.warning(
                    "Error stopping RuntimeLogEventBridge (correlation_id=%s)",
                    correlation_id,
                    exc_info=True,
                )
            runtime_log_bridge = None

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
            except Exception as health_stop_error:  # noqa: BLE001 — boundary: logs warning and degrades
                logger.warning(
                    "Failed to stop health server: %s (correlation_id=%s)",
                    health_stop_error,
                    correlation_id,
                    extra={
                        "error_type": type(health_stop_error).__name__,
                    },
                )
            health_server = None

        # Shutdown plugins in LIFO order (Last In, First Out)
        # This ensures plugins activated later are shut down before plugins they
        # may depend on. Each plugin handles its own resource cleanup (pools,
        # publishers, connections).
        if plugin_config is not None:
            for plugin in reversed(activated_plugins):
                try:
                    shutdown_result = await plugin.shutdown(plugin_config)
                    if not shutdown_result:
                        logger.warning(
                            "Plugin '%s' shutdown reported errors: %s (correlation_id=%s)",
                            plugin.plugin_id,
                            shutdown_result.get_error_message_or_default(),
                            correlation_id,
                        )
                    else:
                        logger.debug(
                            "Plugin '%s' shut down (correlation_id=%s)",
                            plugin.plugin_id,
                            correlation_id,
                        )
                except Exception as plugin_shutdown_error:  # noqa: BLE001 — boundary: logs warning and degrades
                    logger.warning(
                        "Plugin '%s' shutdown failed: %s (correlation_id=%s)",
                        plugin.plugin_id,
                        sanitize_error_message(plugin_shutdown_error),
                        correlation_id,
                    )
            activated_plugins.clear()

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
        # Order: plugin consumers -> contract registry -> health server -> runtime -> plugins (LIFO)

        # Cleanup plugin consumer subscriptions
        for unsub_callback in plugin_unsubscribe_callbacks:
            try:
                await unsub_callback()
            except Exception as cleanup_error:  # noqa: BLE001 — boundary: logs warning and degrades
                logger.warning(
                    "Failed to stop plugin consumer during cleanup: %s (correlation_id=%s)",
                    sanitize_error_message(cleanup_error),
                    correlation_id,
                )

        # Cleanup contract registry router and consumers
        if contract_router is not None:
            try:
                await contract_router.stop()
            except Exception as cleanup_error:  # noqa: BLE001 — boundary: logs warning and degrades
                logger.warning(
                    "Failed to stop contract registry router during cleanup: %s (correlation_id=%s)",
                    sanitize_error_message(cleanup_error),
                    correlation_id,
                )

        for unsub_func in [
            contract_unsub_registered,
            contract_unsub_deregistered,
            contract_unsub_heartbeat,
        ]:
            if unsub_func is not None:
                try:
                    await unsub_func()
                except Exception as cleanup_error:  # noqa: BLE001 — boundary: logs warning and degrades
                    logger.warning(
                        "Failed to stop contract registry consumer during cleanup: %s (correlation_id=%s)",
                        sanitize_error_message(cleanup_error),
                        correlation_id,
                    )

        # Cleanup runtime error triage consumer (OMN-5655)
        if triage_unsub is not None:
            try:
                await triage_unsub()
            except Exception as cleanup_error:  # noqa: BLE001 — boundary: logs warning and degrades
                logger.warning(
                    "Failed to stop runtime error triage consumer during cleanup: %s (correlation_id=%s)",
                    sanitize_error_message(cleanup_error),
                    correlation_id,
                )

        # Cleanup WiringHealthChecker (OMN-6133)
        if wiring_health_task is not None:
            wiring_health_task.cancel()
            try:
                await wiring_health_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

        # Cleanup baselines batch compute loop and pool
        if baselines_task is not None:
            baselines_task.cancel()
            try:
                await baselines_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if _baselines_pool is not None:
            try:
                await _baselines_pool.close()
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

        # Cleanup savings estimation consumer
        if savings_task is not None:
            savings_task.cancel()
            try:
                await savings_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

        # Cleanup ServiceLlmEndpointHealth (OMN-6135)
        if llm_health_service is not None:
            try:
                await llm_health_service.stop()
            except Exception:  # noqa: BLE001 — boundary: best-effort cleanup
                logger.warning(
                    "Failed to stop ServiceLlmEndpointHealth during cleanup "
                    "(correlation_id=%s)",
                    correlation_id,
                )

        # Cleanup RuntimeLogEventBridge (OMN-5525)
        if runtime_log_bridge is not None:
            try:
                await runtime_log_bridge.stop()
                if runtime_log_bridge._producer is not None:
                    await runtime_log_bridge._producer.stop()
            except Exception:  # noqa: BLE001 — boundary: best-effort cleanup
                logger.warning(
                    "Failed to stop RuntimeLogEventBridge during cleanup (correlation_id=%s)",
                    correlation_id,
                )

        if health_server is not None:
            try:
                await health_server.stop()
            except Exception as cleanup_error:  # noqa: BLE001 — boundary: logs warning and degrades
                logger.warning(
                    "Failed to stop health server during cleanup: %s (correlation_id=%s)",
                    sanitize_error_message(cleanup_error),
                    correlation_id,
                )

        if runtime is not None:
            try:
                await runtime.stop()
            except Exception as cleanup_error:  # noqa: BLE001 — boundary: logs warning and degrades
                # Log cleanup failures with context instead of suppressing them
                # Sanitize to prevent potential credential leakage from runtime errors
                logger.warning(
                    "Failed to stop runtime during cleanup: %s (correlation_id=%s)",
                    sanitize_error_message(cleanup_error),
                    correlation_id,
                )

        # Shutdown plugins in LIFO order (handles pools, publishers, connections)
        # Uses minimal config for cleanup to avoid depending on resources that may
        # have been partially created during a failed bootstrap.
        if plugin_config is not None:
            for plugin in reversed(activated_plugins):
                try:
                    await plugin.shutdown(plugin_config)
                except Exception as cleanup_error:  # noqa: BLE001 — boundary: logs warning and degrades
                    logger.warning(
                        "Failed to shut down plugin '%s' during cleanup: %s (correlation_id=%s)",
                        plugin.plugin_id,
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

    # OMN-3811: Opt-in OTEL tracing — silently skips if no endpoint configured
    from omnibase_infra.runtime.tracing import configure_tracing

    configure_tracing()

    exit_code = asyncio.run(bootstrap())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()


__all__: list[str] = [
    "ENV_CONTRACTS_DIR",
    "ENV_OMNICLAUDE_SKILLS_ROOT",
    "bootstrap",
    "load_runtime_config",
    "main",
]
