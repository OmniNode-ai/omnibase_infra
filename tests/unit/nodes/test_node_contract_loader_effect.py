# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Test NodeContractLoaderEffect wraps RuntimeContractConfigLoader [OMN-6347]."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.mark.unit
def test_node_contract_loader_effect_exists() -> None:
    """NodeContractLoaderEffect must be importable."""
    from omnibase_infra.nodes.node_contract_loader_effect.node import (
        NodeContractLoaderEffect,
    )

    assert NodeContractLoaderEffect is not None


@pytest.mark.unit
def test_handler_contract_scan_exists() -> None:
    """HandlerContractScan must be importable and have handle method."""
    from omnibase_infra.nodes.node_contract_loader_effect.handlers.handler_contract_scan import (
        HandlerContractScan,
    )

    handler = HandlerContractScan(container=MagicMock())
    assert hasattr(handler, "handle")
