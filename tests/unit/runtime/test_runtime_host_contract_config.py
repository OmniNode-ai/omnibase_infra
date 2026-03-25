# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Test RuntimeHostProcess uses contract config when provided [OMN-6343].

Verifies that when a ModelRuntimeNodeGraphConfig is passed, the runtime host
uses its values as defaults instead of the module-level DEFAULT_* constants.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _make_runtime_config() -> dict[str, object]:
    """Create minimal runtime config dict for RuntimeHostProcess."""
    return {
        "service_name": "test_service",
        "node_name": "test_node",
    }


@pytest.mark.unit
def test_runtime_host_accepts_runtime_node_graph_config() -> None:
    """RuntimeHostProcess.__init__ must accept runtime_node_graph_config kwarg."""
    from omnibase_infra.runtime.models.model_runtime_node_graph_config import (
        ModelRuntimeNodeGraphConfig,
    )
    from omnibase_infra.runtime.service_runtime_host_process import (
        RuntimeHostProcess,
    )

    mock_config = MagicMock(spec=ModelRuntimeNodeGraphConfig)
    mock_config.drain_timeout_ms = 5000
    mock_config.max_concurrent_handlers = 8
    mock_config.health_check_timeout_ms = 3000
    mock_config.handler_pool_size = 4
    mock_config.batch_response_size = 50
    mock_config.batch_flush_interval_ms = 200

    host = RuntimeHostProcess(
        config=_make_runtime_config(),
        runtime_node_graph_config=mock_config,
    )
    # drain_timeout is stored in seconds, contract provides ms
    assert host._drain_timeout_seconds == 5.0
    assert host._max_concurrent_handlers == 8
    assert host._health_check_timeout_seconds == 3.0
    assert host._handler_pool_size == 4


@pytest.mark.unit
def test_runtime_host_without_config_uses_defaults() -> None:
    """Without runtime_node_graph_config, module-level defaults still work."""
    from omnibase_infra.runtime.service_runtime_host_process import (
        DEFAULT_DRAIN_TIMEOUT_SECONDS,
        DEFAULT_HEALTH_CHECK_TIMEOUT,
        RuntimeHostProcess,
    )

    host = RuntimeHostProcess(config=_make_runtime_config())
    assert host._drain_timeout_seconds == DEFAULT_DRAIN_TIMEOUT_SECONDS
    assert host._health_check_timeout_seconds == DEFAULT_HEALTH_CHECK_TIMEOUT


@pytest.mark.unit
def test_config_dict_overrides_contract_config() -> None:
    """Explicit config dict values override contract config values."""
    from omnibase_infra.runtime.models.model_runtime_node_graph_config import (
        ModelRuntimeNodeGraphConfig,
    )
    from omnibase_infra.runtime.service_runtime_host_process import (
        RuntimeHostProcess,
    )

    mock_config = MagicMock(spec=ModelRuntimeNodeGraphConfig)
    mock_config.drain_timeout_ms = 5000
    mock_config.max_concurrent_handlers = 8
    mock_config.health_check_timeout_ms = 3000
    mock_config.handler_pool_size = 4
    mock_config.batch_response_size = 50
    mock_config.batch_flush_interval_ms = 200

    runtime_config = _make_runtime_config()
    runtime_config["drain_timeout_seconds"] = 60.0
    runtime_config["max_concurrent_handlers"] = 16

    host = RuntimeHostProcess(
        config=runtime_config,
        runtime_node_graph_config=mock_config,
    )
    # Config dict values take precedence over contract config
    assert host._drain_timeout_seconds == 60.0
    assert host._max_concurrent_handlers == 16
    # Contract config values apply where config dict is silent
    assert host._health_check_timeout_seconds == 3.0
    assert host._handler_pool_size == 4
    # The runtime_node_graph_config is stored for downstream usage
    assert host._runtime_node_graph_config is not None
