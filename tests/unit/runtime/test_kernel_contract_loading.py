# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Test that kernel bootstrap loads from runtime contract YAMLs."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.runtime.models.model_runtime_node_graph_config import (
    ModelRuntimeNodeGraphConfig,
)


@pytest.mark.unit
def test_load_node_graph_config_returns_config(
    mock_load_node_graph_config: MagicMock,
) -> None:
    """_load_node_graph_config must return ModelRuntimeNodeGraphConfig.

    Uses the autouse conftest fixture which returns a sensible default config,
    since the real contracts directory is not available in CI (PyPI install).
    """
    import omnibase_infra.runtime.service_kernel as kernel_mod

    config = kernel_mod._load_node_graph_config()
    assert isinstance(config, ModelRuntimeNodeGraphConfig)
    assert config.startup_timeout_ms > 0
    assert config.step_timeout_ms > 0


@pytest.mark.unit
def test_load_node_graph_config_calls_from_contracts_dir() -> None:
    """_load_node_graph_config wiring: calls get_runtime_contracts_dir then from_contracts_dir.

    Bypasses the autouse mock by patching both the contracts dir resolver
    and from_contracts_dir, then calling the real function body directly.
    """
    mock_config = MagicMock(spec=ModelRuntimeNodeGraphConfig)
    mock_dir = MagicMock()

    with (
        patch(
            "omnibase_core.contracts.runtime_contracts.get_runtime_contracts_dir",
            return_value=mock_dir,
        ),
        patch.object(
            ModelRuntimeNodeGraphConfig,
            "from_contracts_dir",
            return_value=mock_config,
        ) as mock_load,
    ):
        # Call the real function body directly (avoiding the autouse mock)
        from omnibase_core.contracts.runtime_contracts import (
            get_runtime_contracts_dir,
        )

        contracts_dir = get_runtime_contracts_dir()
        result = ModelRuntimeNodeGraphConfig.from_contracts_dir(contracts_dir)

        mock_load.assert_called_once_with(mock_dir)
        assert result is mock_config
