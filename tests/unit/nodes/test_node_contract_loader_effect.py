# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Test NodeContractLoaderEffect wraps RuntimeContractConfigLoader [OMN-6347]."""

from __future__ import annotations

from pathlib import Path
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


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handler_contract_scan_takes_typed_request(tmp_path: Path) -> None:
    """handle takes one typed request whose scan_paths override the default."""
    from omnibase_infra.nodes.node_contract_loader_effect.handlers.handler_contract_scan import (
        HandlerContractScan,
    )
    from omnibase_infra.nodes.node_contract_loader_effect.models import (
        ModelContractScanRequest,
    )

    handler = HandlerContractScan(
        container=MagicMock(), default_scan_path=str(tmp_path / "missing")
    )
    result = await handler.handle(ModelContractScanRequest(scan_paths=(str(tmp_path),)))
    assert result == {"status": "complete", "total_loaded": 0}
