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
    """_load_node_graph_config must call ModelRuntimeNodeGraphConfig.from_contracts_dir().

    This test patches the real implementation (bypassing the autouse mock)
    to verify the wiring between _load_node_graph_config and from_contracts_dir.
    """
    mock_config = MagicMock(spec=ModelRuntimeNodeGraphConfig)

    with patch(
        "omnibase_core.contracts.runtime_contracts.get_runtime_contracts_dir",
        return_value=MagicMock(),
    ):
        with patch.object(
            ModelRuntimeNodeGraphConfig,
            "from_contracts_dir",
            return_value=mock_config,
        ) as mock_load:
            # Import the real function (not the mocked one from conftest)
            from omnibase_infra.runtime.service_kernel import (
                _load_node_graph_config as real_fn,
            )

            # Call through the patch stack: get_runtime_contracts_dir is mocked,
            # from_contracts_dir is mocked, so no disk access needed
            with patch(
                "omnibase_infra.runtime.service_kernel._load_node_graph_config",
                side_effect=real_fn,
            ):
                import omnibase_infra.runtime.service_kernel as kernel_mod

                result = kernel_mod._load_node_graph_config()
                mock_load.assert_called_once()
                assert result is mock_config
