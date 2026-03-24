# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Test that kernel bootstrap loads from runtime contract YAMLs."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
def test_load_node_graph_config_returns_config() -> None:
    """_load_node_graph_config must return ModelRuntimeNodeGraphConfig."""
    from omnibase_infra.runtime.models.model_runtime_node_graph_config import (
        ModelRuntimeNodeGraphConfig,
    )
    from omnibase_infra.runtime.service_kernel import _load_node_graph_config

    config = _load_node_graph_config()
    assert isinstance(config, ModelRuntimeNodeGraphConfig)
    assert config.startup_timeout_ms > 0
    assert config.step_timeout_ms > 0


@pytest.mark.unit
def test_load_node_graph_config_calls_from_contracts_dir() -> None:
    """_load_node_graph_config must call ModelRuntimeNodeGraphConfig.from_contracts_dir()."""
    from omnibase_infra.runtime.models.model_runtime_node_graph_config import (
        ModelRuntimeNodeGraphConfig,
    )

    mock_config = MagicMock(spec=ModelRuntimeNodeGraphConfig)
    with patch.object(
        ModelRuntimeNodeGraphConfig, "from_contracts_dir", return_value=mock_config
    ) as mock_load:
        from omnibase_infra.runtime.service_kernel import _load_node_graph_config

        # Need to reimport to pick up patched version
        result = _load_node_graph_config()
        mock_load.assert_called_once()
        assert result is mock_config
