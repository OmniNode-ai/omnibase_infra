# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Test ModelRuntimeNodeGraphConfig loads from runtime contract YAMLs."""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_loads_from_runtime_contracts() -> None:
    """Load config from omnibase_core's runtime contracts."""
    from omnibase_core.contracts.runtime_contracts import get_runtime_contracts_dir
    from omnibase_infra.runtime.models.model_runtime_node_graph_config import (
        ModelRuntimeNodeGraphConfig,
    )

    contracts_dir = get_runtime_contracts_dir()
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
def test_env_var_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env var ONEX_RUNTIME_DRAIN_TIMEOUT_MS overrides contract value."""
    from omnibase_core.contracts.runtime_contracts import get_runtime_contracts_dir
    from omnibase_infra.runtime.models.model_runtime_node_graph_config import (
        ModelRuntimeNodeGraphConfig,
    )

    monkeypatch.setenv("ONEX_RUNTIME_DRAIN_TIMEOUT_MS", "99999")
    contracts_dir = get_runtime_contracts_dir()
    config = ModelRuntimeNodeGraphConfig.from_contracts_dir(contracts_dir)
    assert config.drain_timeout_ms == 99999


@pytest.mark.unit
def test_config_is_frozen() -> None:
    """Config model must be frozen (immutable)."""
    from omnibase_core.contracts.runtime_contracts import get_runtime_contracts_dir
    from omnibase_infra.runtime.models.model_runtime_node_graph_config import (
        ModelRuntimeNodeGraphConfig,
    )

    contracts_dir = get_runtime_contracts_dir()
    config = ModelRuntimeNodeGraphConfig.from_contracts_dir(contracts_dir)

    with pytest.raises(Exception):
        config.startup_timeout_ms = 999  # type: ignore[misc]


@pytest.mark.unit
def test_missing_contract_raises() -> None:
    """Missing contract YAML must raise FileNotFoundError."""
    from pathlib import Path

    from omnibase_infra.runtime.models.model_runtime_node_graph_config import (
        ModelRuntimeNodeGraphConfig,
    )

    with pytest.raises(FileNotFoundError):
        ModelRuntimeNodeGraphConfig.from_contracts_dir(Path("/nonexistent"))
