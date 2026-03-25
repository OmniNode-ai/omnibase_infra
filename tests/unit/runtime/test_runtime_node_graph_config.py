# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Test ModelRuntimeNodeGraphConfig loads from runtime contract YAMLs.

These tests create temp contract directories with minimal YAML fixtures
rather than depending on the omnibase_core repo's contracts/runtime/ directory,
which is not available when core is installed from PyPI.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.runtime.models.model_runtime_node_graph_config import (
    ModelRuntimeNodeGraphConfig,
)

# Minimal contract YAML content for testing
_RUNTIME_ORCHESTRATOR = {
    "workflow_coordination": {
        "workflow_definition": {
            "coordination_rules": {
                "timeout_ms": 120000,
                "step_timeout_ms": 30000,
                "max_retries": 3,
                "retry_backoff_ms": 2000,
                "retry_multiplier": 2.0,
            }
        },
        "lifecycle": {},
    }
}

_NODE_GRAPH_REDUCER = {
    "state_management": {
        "drain_timeout_ms": 30000,
        "max_concurrent_handlers": 10,
        "handler_pool_size": 10,
        "health_check_timeout_ms": 5000,
        "batch_response_size": 100,
        "batch_flush_interval_ms": 1000,
    }
}

_EVENT_BUS_WIRING = {
    "wiring_config": {
        "topic_validation_pattern": r"^[a-z][a-z0-9._-]*$",
        "topic_deny_patterns": ["__consumer_offsets", "_schemas"],
        "max_topic_length": 255,
        "max_subscriptions_per_node": 100,
        "subscription_timeout_ms": 5000,
        "circuit_breaker": {
            "failure_threshold": 5,
            "timeout_ms": 30000,
        },
        "retry_policy": {
            "max_retries": 3,
            "base_delay_ms": 1000,
            "max_delay_ms": 10000,
        },
    }
}

_CONTRACT_LOADER = {
    "scan_config": {
        "exclude_patterns": ["__pycache__", ".git"],
        "deny_paths": ["/etc", "/var"],
        "timeout_ms": 60000,
    }
}


@pytest.fixture
def contracts_dir(tmp_path: Path) -> Path:
    """Create a temp directory with minimal runtime contract YAMLs."""
    for name, content in [
        ("runtime_orchestrator.yaml", _RUNTIME_ORCHESTRATOR),
        ("node_graph_reducer.yaml", _NODE_GRAPH_REDUCER),
        ("event_bus_wiring_effect.yaml", _EVENT_BUS_WIRING),
        ("contract_loader_effect.yaml", _CONTRACT_LOADER),
    ]:
        (tmp_path / name).write_text(yaml.dump(content, default_flow_style=False))
    return tmp_path


@pytest.mark.unit
def test_loads_from_runtime_contracts(contracts_dir: Path) -> None:
    """Load config from temp contract YAMLs."""
    config = ModelRuntimeNodeGraphConfig.from_contracts_dir(contracts_dir)

    # runtime_orchestrator values
    assert config.startup_timeout_ms == 120000
    assert config.step_timeout_ms == 30000
    assert config.max_step_retries == 3
    assert config.retry_backoff_ms == 2000
    assert config.retry_backoff_multiplier == 2.0

    # node_graph / lifecycle values
    assert config.drain_timeout_ms > 0
    assert config.max_concurrent_handlers > 0
    assert config.handler_pool_size > 0
    assert config.health_check_timeout_ms > 0
    assert config.batch_response_size > 0
    assert config.batch_flush_interval_ms > 0

    # event_bus_wiring values
    assert config.topic_validation_pattern is not None
    assert len(config.topic_deny_patterns) > 0
    assert config.max_topic_length == 255
    assert config.max_subscriptions_per_node == 100

    # contract_loader values
    assert len(config.scan_exclude_patterns) > 0
    assert len(config.scan_deny_paths) > 0
    assert config.scan_timeout_ms == 60000


@pytest.mark.unit
def test_env_var_override(contracts_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Env var ONEX_RUNTIME_DRAIN_TIMEOUT_MS overrides contract value."""
    monkeypatch.setenv("ONEX_RUNTIME_DRAIN_TIMEOUT_MS", "99999")
    config = ModelRuntimeNodeGraphConfig.from_contracts_dir(contracts_dir)
    assert config.drain_timeout_ms == 99999


@pytest.mark.unit
def test_config_is_frozen(contracts_dir: Path) -> None:
    """Config model must be frozen (immutable)."""
    config = ModelRuntimeNodeGraphConfig.from_contracts_dir(contracts_dir)

    with pytest.raises(Exception):
        config.startup_timeout_ms = 999  # type: ignore[misc]


@pytest.mark.unit
def test_missing_contract_raises() -> None:
    """Missing contract YAML must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        ModelRuntimeNodeGraphConfig.from_contracts_dir(Path("/nonexistent"))
