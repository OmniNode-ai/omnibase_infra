# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test for OMN-9457 async-incompat handler containment.

End-to-end verification that ``wire_from_manifest`` contains a handler whose
``__init__`` raises the CPython ``asyncio.run() cannot be called from a
running event loop`` signature, preventing a single async-incompatible
handler from poisoning ``runtime-effects`` boot.

Unlike the unit tests in ``tests/unit/runtime/auto_wiring/``, which patch
``_import_handler_class`` to inject synthetic handler classes, this test:

- writes a real Python module into the CPython module table (via
  ``sys.modules``) so ``importlib.import_module`` resolves it,
- runs the full ``wire_from_manifest`` pipeline against that module,
- asserts the manifest scan completes with the async-incompat handler
  quarantined rather than raised, and
- asserts the quarantined handler is surfaced on
  ``ModelAutoWiringReport.quarantined_handlers`` so follow-up migration
  tickets remain visible.

This is the CI gate that proves runtime-effects can reach healthy startup
in the presence of async-incompat handlers (OMN-9126 runtime-startup
invariant + OMN-9457 containment).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_infra.runtime.auto_wiring.enum_quarantine_reason import (
    EnumQuarantineReason,
)
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
from omnibase_infra.runtime.auto_wiring.report import EnumWiringOutcome

_MODULE_NAME = "tests.integration._async_incompat_handler_fixture_omn_9457"


class _HandlerAsyncIncompatIntegration:
    """Handler whose construction raises the exact CPython async-incompat signature."""

    def __init__(self) -> None:
        raise RuntimeError("asyncio.run() cannot be called from a running event loop")

    async def handle(self, envelope: object) -> None:
        return None


class _HandlerHealthyIntegration:
    """Baseline healthy handler — wires cleanly alongside a quarantined sibling."""

    async def handle(self, envelope: object) -> None:
        return None


@pytest.fixture
def installed_handler_module() -> types.ModuleType:
    """Install a real importable module containing the integration handlers.

    ``wire_from_manifest`` calls ``importlib.import_module(module_path)``
    to load handler classes. Registering the module in ``sys.modules``
    before the manifest scan satisfies that import without needing an
    on-disk package layout.
    """
    module = types.ModuleType(_MODULE_NAME)
    module._HandlerAsyncIncompatIntegration = _HandlerAsyncIncompatIntegration  # type: ignore[attr-defined]
    module._HandlerHealthyIntegration = _HandlerHealthyIntegration  # type: ignore[attr-defined]
    sys.modules[_MODULE_NAME] = module
    try:
        yield module
    finally:
        sys.modules.pop(_MODULE_NAME, None)


def _make_contract(
    *,
    node_name: str,
    handler_names: tuple[str, ...],
) -> ModelDiscoveredContract:
    entries = tuple(
        ModelHandlerRoutingEntry(
            handler=ModelHandlerRef(name=name, module=_MODULE_NAME),
            event_model=ModelHandlerRef(name=f"ModelEvt{idx}", module="fake.models"),
            operation=None,
        )
        for idx, name in enumerate(handler_names)
    )
    return ModelDiscoveredContract(
        name=node_name,
        node_type="ORCHESTRATOR_GENERIC",
        package_name="test_omn_9457_pkg",
        entry_point_name=node_name,
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/contract.yaml"),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=entries,
        ),
        event_bus=ModelEventBusWiring(
            subscribe_topics=(f"onex.evt.test.{node_name}.v1",),
            publish_topics=(),
        ),
    )


def _make_dispatch_engine() -> MagicMock:
    engine = MagicMock()
    engine._routes = {}
    engine._container = None
    engine.register_dispatcher = MagicMock()
    engine.register_route = MagicMock()
    engine.freeze = MagicMock()
    return engine


def _make_event_bus() -> MagicMock:
    bus = MagicMock()
    bus.subscribe = AsyncMock()
    return bus


@pytest.mark.integration
@pytest.mark.asyncio
async def test_async_incompat_handler_quarantined_end_to_end(
    installed_handler_module: types.ModuleType,
) -> None:
    """Full wire_from_manifest call: async-incompat handler is contained, not raised."""
    contract = _make_contract(
        node_name="node_async_incompat_integration",
        handler_names=("_HandlerAsyncIncompatIntegration",),
    )
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())

    report = await wire_from_manifest(
        manifest=manifest,
        dispatch_engine=_make_dispatch_engine(),
        event_bus=_make_event_bus(),
        container=None,
    )

    # No wiring failure — the async-incompat handler must not poison startup.
    assert report.total_failed == 0
    # The contract has no live handler, so it reports SKIPPED.
    assert report.total_skipped == 1
    assert report.total_quarantined == 1
    assert len(report.quarantined_handlers) == 1

    q = report.quarantined_handlers[0]
    assert q.handler_name == "_HandlerAsyncIncompatIntegration"
    assert q.handler_module == _MODULE_NAME
    assert q.contract_name == "node_async_incompat_integration"
    assert q.reason is EnumQuarantineReason.ASYNC_INCOMPATIBLE
    assert "asyncio.run" in q.detail

    result = report.results[0]
    assert result.outcome is EnumWiringOutcome.SKIPPED
    assert result.reason == "all handlers quarantined"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mixed_contract_wires_healthy_and_quarantines_incompat(
    installed_handler_module: types.ModuleType,
) -> None:
    """A contract with one healthy + one async-incompat handler still reports WIRED.

    This pins the behavior exposed by the PR #1380 review: the
    ``all_handlers_quarantined`` path must NOT fire unless every prepared
    handler is quarantined. Mixed contracts land on the WIRED path with
    the healthy handler's dispatcher registered and the quarantined
    handler surfaced in ``quarantined_handlers``.
    """
    contract = _make_contract(
        node_name="node_mixed_integration",
        handler_names=(
            "_HandlerHealthyIntegration",
            "_HandlerAsyncIncompatIntegration",
        ),
    )
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())

    report = await wire_from_manifest(
        manifest=manifest,
        dispatch_engine=_make_dispatch_engine(),
        event_bus=_make_event_bus(),
        container=None,
    )

    assert report.total_failed == 0
    assert report.total_wired == 1  # contract WIRED — at least one live handler
    assert report.total_quarantined == 1
    assert (
        report.quarantined_handlers[0].handler_name
        == "_HandlerAsyncIncompatIntegration"
    )

    result = report.results[0]
    assert result.outcome is EnumWiringOutcome.WIRED
    assert result.reason != "all handlers quarantined"
    assert len(result.dispatchers_registered) == 1
    assert len(result.quarantined_handlers) == 1
