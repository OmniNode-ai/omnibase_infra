# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""HandlerMCP reads its server timeouts from the real handler contract (OMN-22130)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import omnibase_infra
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.handlers.handler_mcp import HandlerMCP
from omnibase_infra.handlers.models.mcp import ModelMcpServerTimeouts
from omnibase_infra.runtime.handler_contract_config_loader import (
    extract_handler_config,
    load_handler_contract_config,
)

pytestmark = pytest.mark.integration

_CONTRACT = (
    Path(omnibase_infra.__file__).parent
    / "contracts"
    / "handlers"
    / "mcp"
    / "handler_contract.yaml"
)


def test_shipped_contract_timeouts_reach_the_handler_config() -> None:
    contract = load_handler_contract_config(_CONTRACT, "mcp")
    config = extract_handler_config(contract, "mcp", require_basic_fields=False)
    timeouts = ModelMcpServerTimeouts.model_validate(config["server_timeouts"])
    assert timeouts == ModelMcpServerTimeouts(
        shutdown_seconds=5.0, cancel_seconds=1.0, startup_seconds=2.0
    )


def test_handler_is_built_from_the_container_alone() -> None:
    handler = HandlerMCP(container=MagicMock())
    assert handler is not None


@pytest.mark.parametrize(
    "bad",
    [
        "five",
        {"shutdown_seconds": 0},
        {"shutdown_seconds": -1.0},
        {"unknown_seconds": 1.0},
    ],
)
def test_malformed_server_timeouts_fail_at_contract_load(bad: object) -> None:
    contract = {
        "name": "x",
        "metadata": {"transport": {"server_timeouts": bad}},
    }
    with pytest.raises(ProtocolConfigurationError):
        extract_handler_config(contract, "mcp", require_basic_fields=False)
