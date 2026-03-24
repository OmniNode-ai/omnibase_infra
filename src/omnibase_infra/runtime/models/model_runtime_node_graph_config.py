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

from omnibase_core.types import JsonType


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


def _get_dict(data: dict[str, JsonType], key: str) -> dict[str, JsonType]:
    """Get a nested dict, returning empty dict if missing or wrong type."""
    val = data.get(key)
    if isinstance(val, dict):
        return val
    return {}


def _get_int(data: dict[str, JsonType], key: str, default: int) -> int:
    """Get an int value from a dict, returning default if missing."""
    val = data.get(key, default)
    return int(val)  # type: ignore[arg-type]


def _get_float(data: dict[str, JsonType], key: str, default: float) -> float:
    """Get a float value from a dict, returning default if missing."""
    val = data.get(key, default)
    return float(val)  # type: ignore[arg-type]


def _get_str(data: dict[str, JsonType], key: str, default: str) -> str:
    """Get a str value from a dict, returning default if missing."""
    val = data.get(key, default)
    return str(val)


def _get_list(data: dict[str, JsonType], key: str) -> list[JsonType]:
    """Get a list value from a dict, returning empty list if missing."""
    val = data.get(key)
    if isinstance(val, list):
        return val
    return []


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

    # From node_graph_reducer.yaml (operational defaults matching old DEFAULT_* constants)
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
        coord = _get_dict(orchestrator, "workflow_coordination")
        coord_def = _get_dict(coord, "workflow_definition")
        coord_rules = _get_dict(coord_def, "coordination_rules")
        lifecycle = _get_dict(coord, "lifecycle")

        startup_timeout_ms = _env_override_int(
            "startup_timeout_ms", _get_int(coord_rules, "timeout_ms", 120000)
        )
        step_timeout_ms = _env_override_int(
            "step_timeout_ms", _get_int(coord_rules, "step_timeout_ms", 30000)
        )
        max_step_retries = _env_override_int(
            "max_step_retries", _get_int(coord_rules, "max_retries", 3)
        )
        retry_backoff_ms = _env_override_int(
            "retry_backoff_ms", _get_int(coord_rules, "retry_backoff_ms", 2000)
        )
        retry_backoff_multiplier = _env_override_float(
            "retry_backoff_multiplier",
            _get_float(coord_rules, "retry_backoff_multiplier", 2.0),
        )

        # Extract from lifecycle section (runtime_orchestrator.yaml)
        # and node_graph_reducer.yaml operational defaults
        node_graph_ops = _get_dict(node_graph, "operational_defaults")
        drain_timeout_ms = _env_override_int(
            "drain_timeout_ms",
            _get_int(lifecycle, "graceful_shutdown_timeout_ms", 60000),
        )
        max_concurrent_handlers = _env_override_int(
            "max_concurrent_handlers",
            _get_int(node_graph_ops, "max_concurrent_handlers", 10),
        )
        handler_pool_size = _env_override_int(
            "handler_pool_size",
            _get_int(node_graph_ops, "handler_pool_size", 4),
        )
        health_check_timeout_ms = _env_override_int(
            "health_check_timeout_ms",
            _get_int(lifecycle, "health_check_interval_ms", 30000),
        )
        batch_response_size = _env_override_int(
            "batch_response_size",
            _get_int(node_graph_ops, "batch_response_size", 100),
        )
        batch_flush_interval_ms = _env_override_int(
            "batch_flush_interval_ms",
            _get_int(node_graph_ops, "batch_flush_interval_ms", 500),
        )

        # Extract from event_bus_wiring_effect.yaml
        wiring_config = _get_dict(event_bus, "wiring_config")
        topic_validation = _get_dict(wiring_config, "topic_validation")
        effect = _get_dict(event_bus, "effect")
        retry_policy = _get_dict(effect, "retry_policy")
        circuit_breaker = _get_dict(effect, "circuit_breaker")

        topic_validation_pattern = _get_str(
            topic_validation,
            "allowed_pattern",
            r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,254}$",
        )
        deny_patterns_raw = _get_list(topic_validation, "deny_patterns")
        topic_deny_patterns = tuple(str(p) for p in deny_patterns_raw)

        max_topic_length = _env_override_int(
            "max_topic_length",
            _get_int(topic_validation, "max_topic_length", 255),
        )
        max_subscriptions_per_node = _env_override_int(
            "max_subscriptions_per_node",
            _get_int(topic_validation, "max_subscriptions_per_node", 100),
        )
        subscription_timeout_ms = _env_override_int(
            "subscription_timeout_ms",
            _get_int(wiring_config, "subscription_timeout_ms", 5000),
        )
        circuit_breaker_failure_threshold = _env_override_int(
            "circuit_breaker_failure_threshold",
            _get_int(circuit_breaker, "failure_threshold", 5),
        )
        circuit_breaker_timeout_ms = _env_override_int(
            "circuit_breaker_timeout_ms",
            _get_int(circuit_breaker, "timeout_ms", 30000),
        )
        wiring_retry_max = _env_override_int(
            "wiring_retry_max", _get_int(retry_policy, "max_retries", 3)
        )
        wiring_retry_base_delay_ms = _env_override_int(
            "wiring_retry_base_delay_ms",
            _get_int(retry_policy, "base_delay_ms", 1000),
        )
        wiring_retry_max_delay_ms = _env_override_int(
            "wiring_retry_max_delay_ms",
            _get_int(retry_policy, "max_delay_ms", 10000),
        )

        # Extract from contract_loader_effect.yaml
        effect_ops = _get_dict(contract_loader, "effect_operations")
        operations = _get_list(effect_ops, "operations")
        scan_op: dict[str, JsonType] = {}
        for op in operations:
            if (
                isinstance(op, dict)
                and op.get("operation_name") == "scan_contracts_directory"
            ):
                scan_op = op
                break
        scan_metadata = _get_dict(scan_op, "metadata")
        io_config = _get_dict(scan_op, "io_config")
        scan_security = _get_dict(io_config, "security")
        path_validation = _get_dict(scan_security, "path_validation")

        exclude_raw = _get_list(scan_metadata, "exclude_patterns")
        scan_exclude_patterns = tuple(str(p) for p in exclude_raw)
        deny_raw = _get_list(path_validation, "deny_patterns")
        scan_deny_paths = tuple(str(p) for p in deny_raw)
        scan_timeout_ms = _env_override_int(
            "scan_timeout_ms", _get_int(scan_op, "operation_timeout_ms", 60000)
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
    def _load_yaml(path: Path) -> dict[str, JsonType]:
        """Load a YAML file, raising FileNotFoundError if missing."""
        if not path.exists():
            raise FileNotFoundError(f"Runtime contract not found: {path}")
        with path.open() as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            msg = f"Expected dict in {path}, got {type(data).__name__}"
            raise ValueError(msg)
        return data  # type: ignore[return-value]
