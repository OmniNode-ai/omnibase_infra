# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for strict auto-wiring startup invariant [OMN-8735, OMN-9126].

OMN-8735 introduced raise-on-failure as a hard startup gate.
OMN-9126 gates the raise behind ONEX_WIRING_STRICT_MODE (default OFF) so that
the runtime can start with non-compliant handlers while compliance is being
resolved, without requiring a full pre-strict rollback.

Non-strict mode (default): failures logged as WARNING, included in report.
Strict mode (ONEX_WIRING_STRICT_MODE=1): raises ModelOnexError as before.
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
async def test_wire_from_manifest_raises_in_strict_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In strict mode, wire_from_manifest raises ModelOnexError on bad contract.

    ONEX_WIRING_STRICT_MODE=1 restores the OMN-8735 hard startup gate.
    """
    from omnibase_core.models.errors import ModelOnexError

    monkeypatch.setenv("ONEX_WIRING_STRICT_MODE", "1")

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
async def test_wire_from_manifest_collects_all_failures_in_strict_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In strict mode, all failing contracts are reported together in the raised error.

    Ensures the error message lists all contract names, not just the first.
    """
    from omnibase_core.models.errors import ModelOnexError

    monkeypatch.setenv("ONEX_WIRING_STRICT_MODE", "1")

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
    assert "3 contract(s)" in error_msg, (
        f"Expected failure count '3 contract(s)' in error message: {error_msg!r}"
    )
    for i in range(3):
        assert f"node_bad_{i}" in error_msg, (
            f"Expected contract name 'node_bad_{i}' in error message: {error_msg!r}"
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_wire_from_manifest_non_strict_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In non-strict mode (default), wire_from_manifest returns report without raising.

    OMN-9126: failures are included in report.total_failed; runtime continues.
    """
    monkeypatch.delenv("ONEX_WIRING_STRICT_MODE", raising=False)

    contract = _make_contract(
        name="node_bad",
        handler_module="nonexistent.module.that.does.not.exist",
    )
    manifest = ModelAutoWiringManifest(contracts=[contract])

    dispatch_engine = MagicMock()
    event_bus = AsyncMock()

    report = await wire_from_manifest(
        manifest=manifest,
        dispatch_engine=dispatch_engine,
        event_bus=event_bus,
    )

    assert report.total_failed == 1, (
        f"Expected 1 failure in non-strict report, got {report.total_failed}"
    )
    assert report.total_wired == 0
