# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed runtime configuration loaded from the 5 runtime contract YAMLs.

Not reusing ModelRuntimeContractConfig because: that model aggregates
contract load results (success/failure counts), not runtime configuration
values (timeouts, retry policies, FSM definitions).

Loading pattern:
    1. Contract YAML is always read (source of truth)
    2. For each field, check ONEX_RUNTIME_{FIELD_NAME} env var
    3. If env var set, use it as operator override
    4. Otherwise use the contract value
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


def _env_override_int(field_name: str, contract_value: int) -> int:
    """Check for ONEX_RUNTIME_{FIELD_NAME} env var override."""
    env_key = f"ONEX_RUNTIME_{field_name.upper()}"
    env_val = os.environ.get(env_key)
    if env_val is not None:
        return int(env_val)
    return contract_value


def _env_override_float(field_name: str, contract_value: float) -> float:
    """Check for ONEX_RUNTIME_{FIELD_NAME} env var override."""
    env_key = f"ONEX_RUNTIME_{field_name.upper()}"
    env_val = os.environ.get(env_key)
    if env_val is not None:
        return float(env_val)
    return contract_value


def _deep_get(data: dict, path: str, default: object = None) -> object:
    """Get a nested value from a dict by dot-separated path."""
    keys = path.split(".")
    current: object = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)  # type: ignore[union-attr]
    return current


class ModelRuntimeNodeGraphConfig(BaseModel):
    """Typed runtime configuration extracted from the 5 runtime contract YAMLs.

    Each field maps to a specific YAML path in one of the contracts. After loading
    from the contract, each field is checked for an ``ONEX_RUNTIME_{FIELD_NAME}``
    env var override. The contract is always the source of truth; env vars are
    operator overrides on top.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    # From runtime_orchestrator.yaml coordination_rules
    startup_timeout_ms: int = Field(..., description="Total startup timeout")
    step_timeout_ms: int = Field(..., description="Per-step timeout")
    max_step_retries: int = Field(..., description="Step retry count")
    retry_backoff_ms: int = Field(..., description="Initial retry backoff")
    retry_backoff_multiplier: float = Field(..., description="Retry multiplier")

    # From node_graph_reducer.yaml (no metadata.defaults section exists,
    # so these use sensible defaults matching the old DEFAULT_* constants)
    drain_timeout_ms: int = Field(..., description="Graceful drain timeout")
    max_concurrent_handlers: int = Field(..., description="Max concurrent handlers")
    handler_pool_size: int = Field(..., description="Handler pool size")
    health_check_timeout_ms: int = Field(..., description="Health check timeout")
    batch_response_size: int = Field(..., description="Batch response size")
    batch_flush_interval_ms: int = Field(..., description="Batch flush interval")

    # From event_bus_wiring_effect.yaml wiring_config
    topic_validation_pattern: str = Field(..., description="Topic name regex")
    topic_deny_patterns: tuple[str, ...] = Field(default_factory=tuple)
    max_topic_length: int = Field(default=255)
    max_subscriptions_per_node: int = Field(default=100)
    subscription_timeout_ms: int = Field(default=5000)
    circuit_breaker_failure_threshold: int = Field(default=5)
    circuit_breaker_timeout_ms: int = Field(default=30000)

    # From event_bus_wiring_effect.yaml retry_policy
    wiring_retry_max: int = Field(default=3)
    wiring_retry_base_delay_ms: int = Field(default=1000)
    wiring_retry_max_delay_ms: int = Field(default=10000)

    # From contract_loader_effect.yaml
    scan_exclude_patterns: tuple[str, ...] = Field(default_factory=tuple)
    scan_deny_paths: tuple[str, ...] = Field(default_factory=tuple)
    scan_timeout_ms: int = Field(default=60000)

    @classmethod
    def from_contracts_dir(cls, contracts_dir: Path) -> ModelRuntimeNodeGraphConfig:
        """Load config from the 5 runtime contract YAMLs.

        Each field is loaded from the contract YAML, then checked for an
        ``ONEX_RUNTIME_{FIELD_NAME}`` env var override.

        Args:
            contracts_dir: Path to directory containing the 5 runtime contract YAMLs.

        Returns:
            Frozen config model.

        Raises:
            FileNotFoundError: If any required contract YAML is missing.
        """
        # Load all contracts
        orchestrator = cls._load_yaml(contracts_dir / "runtime_orchestrator.yaml")
        node_graph = cls._load_yaml(contracts_dir / "node_graph_reducer.yaml")
        event_bus = cls._load_yaml(contracts_dir / "event_bus_wiring_effect.yaml")
        contract_loader = cls._load_yaml(contracts_dir / "contract_loader_effect.yaml")

        # Extract from runtime_orchestrator.yaml
        coord = orchestrator.get("workflow_coordination", {})
        coord_def = coord.get("workflow_definition", {})
        coord_rules = coord_def.get("coordination_rules", {})

        startup_timeout_ms = _env_override_int(
            "startup_timeout_ms",
            int(_deep_get(coord_rules, "timeout_ms", 120000)),  # type: ignore[arg-type]
        )
        step_timeout_ms = _env_override_int(
            "step_timeout_ms",
            int(_deep_get(coord_rules, "step_timeout_ms", 30000)),  # type: ignore[arg-type]
        )
        max_step_retries = _env_override_int(
            "max_step_retries",
            int(_deep_get(coord_rules, "max_retries", 3)),  # type: ignore[arg-type]
        )
        retry_backoff_ms = _env_override_int(
            "retry_backoff_ms",
            int(_deep_get(coord_rules, "retry_backoff_ms", 2000)),  # type: ignore[arg-type]
        )
        retry_backoff_multiplier = _env_override_float(
            "retry_backoff_multiplier",
            float(
                _deep_get(coord_rules, "retry_backoff_multiplier", 2.0)  # type: ignore[arg-type]
            ),
        )

        # Extract from node_graph_reducer.yaml
        # The node_graph contract defines FSM states but not operational defaults.
        # These values match the old DEFAULT_* constants from service_runtime_host_process.py.
        # The lifecycle section has graceful_shutdown_timeout_ms which maps to drain_timeout.
        lifecycle = coord.get("lifecycle", {})
        graceful_shutdown_ms = int(lifecycle.get("graceful_shutdown_timeout_ms", 60000))

        drain_timeout_ms = _env_override_int("drain_timeout_ms", graceful_shutdown_ms)
        max_concurrent_handlers = _env_override_int("max_concurrent_handlers", 10)
        handler_pool_size = _env_override_int("handler_pool_size", 4)
        health_check_timeout_ms = _env_override_int(
            "health_check_timeout_ms",
            int(lifecycle.get("health_check_interval_ms", 30000)),
        )
        batch_response_size = _env_override_int("batch_response_size", 100)
        batch_flush_interval_ms = _env_override_int("batch_flush_interval_ms", 500)

        # Extract from event_bus_wiring_effect.yaml
        wiring_config = event_bus.get("wiring_config", {})
        topic_validation = wiring_config.get("topic_validation", {})
        effect = event_bus.get("effect", {})
        retry_policy = effect.get("retry_policy", {})
        circuit_breaker = effect.get("circuit_breaker", {})

        topic_validation_pattern = str(
            topic_validation.get(
                "allowed_pattern", r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,254}$"
            )
        )
        topic_deny_patterns = tuple(topic_validation.get("deny_patterns", []))
        max_topic_length = _env_override_int(
            "max_topic_length",
            int(topic_validation.get("max_topic_length", 255)),
        )
        max_subscriptions_per_node = _env_override_int(
            "max_subscriptions_per_node",
            int(topic_validation.get("max_subscriptions_per_node", 100)),
        )
        subscription_timeout_ms = _env_override_int(
            "subscription_timeout_ms",
            int(wiring_config.get("subscription_timeout_ms", 5000)),
        )
        circuit_breaker_failure_threshold = _env_override_int(
            "circuit_breaker_failure_threshold",
            int(circuit_breaker.get("failure_threshold", 5)),
        )
        circuit_breaker_timeout_ms = _env_override_int(
            "circuit_breaker_timeout_ms",
            int(circuit_breaker.get("timeout_ms", 30000)),
        )
        wiring_retry_max = _env_override_int(
            "wiring_retry_max",
            int(retry_policy.get("max_retries", 3)),
        )
        wiring_retry_base_delay_ms = _env_override_int(
            "wiring_retry_base_delay_ms",
            int(retry_policy.get("base_delay_ms", 1000)),
        )
        wiring_retry_max_delay_ms = _env_override_int(
            "wiring_retry_max_delay_ms",
            int(retry_policy.get("max_delay_ms", 10000)),
        )

        # Extract from contract_loader_effect.yaml
        effect_ops = contract_loader.get("effect_operations", {})
        operations = effect_ops.get("operations", [])
        scan_op = next(
            (
                op
                for op in operations
                if op.get("operation_name") == "scan_contracts_directory"
            ),
            {},
        )
        scan_metadata = scan_op.get("metadata", {})
        scan_security = scan_op.get("io_config", {}).get("security", {})
        path_validation = scan_security.get("path_validation", {})

        scan_exclude_patterns = tuple(scan_metadata.get("exclude_patterns", []))
        scan_deny_paths = tuple(path_validation.get("deny_patterns", []))
        scan_timeout_ms = _env_override_int(
            "scan_timeout_ms",
            int(scan_op.get("operation_timeout_ms", 60000)),
        )

        return cls(
            startup_timeout_ms=startup_timeout_ms,
            step_timeout_ms=step_timeout_ms,
            max_step_retries=max_step_retries,
            retry_backoff_ms=retry_backoff_ms,
            retry_backoff_multiplier=retry_backoff_multiplier,
            drain_timeout_ms=drain_timeout_ms,
            max_concurrent_handlers=max_concurrent_handlers,
            handler_pool_size=handler_pool_size,
            health_check_timeout_ms=health_check_timeout_ms,
            batch_response_size=batch_response_size,
            batch_flush_interval_ms=batch_flush_interval_ms,
            topic_validation_pattern=topic_validation_pattern,
            topic_deny_patterns=topic_deny_patterns,
            max_topic_length=max_topic_length,
            max_subscriptions_per_node=max_subscriptions_per_node,
            subscription_timeout_ms=subscription_timeout_ms,
            circuit_breaker_failure_threshold=circuit_breaker_failure_threshold,
            circuit_breaker_timeout_ms=circuit_breaker_timeout_ms,
            wiring_retry_max=wiring_retry_max,
            wiring_retry_base_delay_ms=wiring_retry_base_delay_ms,
            wiring_retry_max_delay_ms=wiring_retry_max_delay_ms,
            scan_exclude_patterns=scan_exclude_patterns,
            scan_deny_paths=scan_deny_paths,
            scan_timeout_ms=scan_timeout_ms,
        )

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        """Load a YAML file, raising FileNotFoundError if missing."""
        if not path.exists():
            raise FileNotFoundError(f"Runtime contract not found: {path}")
        with path.open() as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            msg = f"Expected dict in {path}, got {type(data).__name__}"
            raise ValueError(msg)
        return data
