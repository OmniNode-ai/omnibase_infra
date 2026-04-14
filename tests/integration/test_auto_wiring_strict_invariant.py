# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for strict auto-wiring startup invariant [OMN-8735].

Verifies that wire_from_manifest raises ModelOnexError when any contract fails
to wire, instead of silently continuing. This is the hard startup gate added
in OMN-8735.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)


def _make_contract(name: str, handler_module: str) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/contract.yaml"),
        entry_point_name=name,
        package_name="test-package",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(f"onex.evt.platform.{name}.v1",),
            publish_topics=(),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerNonexistent", module=handler_module
                    ),
                    event_model=None,
                    operation=None,
                ),
            ),
        ),
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_wire_from_manifest_raises_on_bad_contract() -> None:
    """wire_from_manifest must raise ModelOnexError when a contract fails to wire.

    OMN-8735 strict invariant: failures abort startup, no swallow-and-continue.
    """
    from omnibase_core.models.errors import ModelOnexError

    contract = _make_contract(
        name="node_bad",
        handler_module="nonexistent.module.that.does.not.exist",
    )
    manifest = ModelAutoWiringManifest(contracts=[contract])

    dispatch_engine = MagicMock()
    event_bus = AsyncMock()

    with pytest.raises(ModelOnexError, match="Auto-wiring failed"):
        await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=dispatch_engine,
            event_bus=event_bus,
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_wire_from_manifest_collects_all_failures_before_raising() -> None:
    """All failing contracts are reported together in the raised ModelOnexError.

    Ensures the error message lists all contract names, not just the first.
    """
    from omnibase_core.models.errors import ModelOnexError

    contracts = [
        _make_contract(
            name=f"node_bad_{i}",
            handler_module=f"nonexistent.module.{i}",
        )
        for i in range(3)
    ]
    manifest = ModelAutoWiringManifest(contracts=contracts)

    dispatch_engine = MagicMock()
    event_bus = AsyncMock()

    with pytest.raises(ModelOnexError) as exc_info:
        await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=dispatch_engine,
            event_bus=event_bus,
        )

    error_msg = str(exc_info.value)
    assert "3" in error_msg or "node_bad_0" in error_msg
