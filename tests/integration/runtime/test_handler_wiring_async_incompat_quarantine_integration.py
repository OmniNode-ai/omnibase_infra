# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for async-incompatible handler quarantine [OMN-9457].

Exercises wire_from_manifest end-to-end with handlers that raise the exact CPython
asyncio.run() / Runner.run() error messages, confirming quarantine isolation:

  1. A handler raising "asyncio.run() cannot be called from a running event loop"
     (exact CPython 3.12 message) is quarantined — wiring does not fail.
  2. A handler raising "Runner.run() cannot be called from a running event loop"
     (CPython 3.11/3.12 Runner.run() message) is also quarantined.
  3. Mixed contract: one good zero-arg handler wires, one async-incompat handler
     quarantines; report is WIRED, quarantined_handlers has exactly 1 entry.
  4. Fully-quarantined contract: every handler quarantines → report outcome SKIPPED
     with reason "all handlers quarantined" and total_failed == 0.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_core.errors.error_service_resolution import ServiceResolutionError
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

# Exact CPython 3.11/3.12 messages (verified from asyncio/runners.py source).
_ASYNCIO_RUN_MSG = "asyncio.run() cannot be called from a running event loop"
_RUNNER_RUN_MSG = "Runner.run() cannot be called from a running event loop"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _GoodHandler:
    """Zero-arg handler that constructs cleanly."""

    async def handle(self, envelope: object) -> None:
        return None


def _make_entry(name: str, module: str) -> ModelHandlerRoutingEntry:
    return ModelHandlerRoutingEntry(
        handler=ModelHandlerRef(name=name, module=module),
        event_model=ModelHandlerRef(name="ModelTestEvent", module="fake.models"),
        operation=None,
    )


def _make_contract(
    name: str,
    entries: tuple[ModelHandlerRoutingEntry, ...],
    topic: str = "onex.evt.test.event.v1",
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="ORCHESTRATOR_GENERIC",
        package_name="omnibase_infra",
        entry_point_name=name,
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path(f"/fake/{name}_contract.yaml"),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=entries,
        ),
        event_bus=ModelEventBusWiring(
            subscribe_topics=(topic,),
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_asyncio_run_message_quarantines_handler() -> None:
    """Handler raising the exact CPython asyncio.run() message is quarantined [OMN-9457].

    CPython asyncio.run() raises "asyncio.run() cannot be called from a running
    event loop" (with parentheses). The detector must match this exact string.
    """

    class _AsyncIncompatHandler:
        def __init__(self) -> None:
            raise RuntimeError(_ASYNCIO_RUN_MSG)

        async def handle(self, envelope: object) -> None:
            return None

    entry = _make_entry("_AsyncIncompatHandler", __name__)
    contract = _make_contract("node_asyncio_run", (entry,))
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())

    container = MagicMock()
    container.get_service = MagicMock(side_effect=RuntimeError(_ASYNCIO_RUN_MSG))
    container.get_service_async = AsyncMock(side_effect=RuntimeError(_ASYNCIO_RUN_MSG))

    event_bus = MagicMock()
    event_bus.subscribe = AsyncMock()
    dispatch_engine = _make_dispatch_engine()

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=_AsyncIncompatHandler,
    ):
        report = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=dispatch_engine,
            event_bus=event_bus,
            container=container,
        )

    assert report.total_failed == 0, f"Unexpected failures: {report.results}"
    assert len(report.quarantined_handlers) == 1, (
        f"Expected 1 quarantined handler, got {report.quarantined_handlers}"
    )
    qh = report.quarantined_handlers[0]
    assert qh.reason == EnumQuarantineReason.ASYNC_INCOMPATIBLE
    assert "_AsyncIncompatHandler" in qh.handler_name


@pytest.mark.integration
@pytest.mark.asyncio
async def test_runner_run_message_quarantines_handler() -> None:
    """Handler raising Runner.run() message is quarantined [OMN-9457].

    CPython asyncio.Runner.run() raises "Runner.run() cannot be called from a
    running event loop" when invoked inside an active event loop.
    """

    class _RunnerIncompatHandler:
        def __init__(self) -> None:
            raise RuntimeError(_RUNNER_RUN_MSG)

        async def handle(self, envelope: object) -> None:
            return None

    entry = _make_entry("_RunnerIncompatHandler", __name__)
    contract = _make_contract(
        "node_runner_incompat", (entry,), topic="onex.evt.test.runner.v1"
    )
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())

    container = MagicMock()
    container.get_service = MagicMock(side_effect=RuntimeError(_RUNNER_RUN_MSG))
    container.get_service_async = AsyncMock(side_effect=RuntimeError(_RUNNER_RUN_MSG))

    event_bus = MagicMock()
    event_bus.subscribe = AsyncMock()
    dispatch_engine = _make_dispatch_engine()

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=_RunnerIncompatHandler,
    ):
        report = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=dispatch_engine,
            event_bus=event_bus,
            container=container,
        )

    assert report.total_failed == 0, f"Unexpected failures: {report.results}"
    assert len(report.quarantined_handlers) == 1
    assert (
        report.quarantined_handlers[0].reason == EnumQuarantineReason.ASYNC_INCOMPATIBLE
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mixed_contract_wires_good_quarantines_bad() -> None:
    """Mixed contract: good zero-arg handler wires, async-incompat handler quarantines [OMN-9457].

    The contract outcome is WIRED (at least one live handler); quarantined_handlers
    has exactly one entry for _BadHandler; total_failed == 0.

    _GoodHandler is zero-arg: the container raises ServiceResolutionError (not an
    asyncio error) so wiring falls through to the zero-arg construction path, which
    succeeds cleanly.
    """

    class _BadHandler:
        def __init__(self) -> None:
            raise RuntimeError(_ASYNCIO_RUN_MSG)

        async def handle(self, envelope: object) -> None:
            return None

    good_entry = _make_entry("_GoodHandler", __name__)
    bad_entry = _make_entry("_BadHandler", __name__)
    contract = _make_contract(
        "node_mixed",
        (good_entry, bad_entry),
        topic="onex.evt.test.mixed.v1",
    )
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())

    # Container raises ServiceResolutionError for good handler (triggers zero-arg
    # fallback) and asyncio RuntimeError for bad handler (triggers quarantine).
    async def _get_service_async(cls: type) -> object:
        if cls is _GoodHandler:
            raise ServiceResolutionError("no registration for _GoodHandler")
        raise RuntimeError(_ASYNCIO_RUN_MSG)

    def _get_service(cls: type) -> object:
        raise ServiceResolutionError("sync path not used")

    container = MagicMock()
    container.get_service = MagicMock(side_effect=_get_service)
    container.get_service_async = AsyncMock(side_effect=_get_service_async)

    event_bus = MagicMock()
    event_bus.subscribe = AsyncMock()
    dispatch_engine = _make_dispatch_engine()

    def _import_side_effect(module: str, name: str) -> type:
        if name == "_GoodHandler":
            return _GoodHandler
        return _BadHandler

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        side_effect=_import_side_effect,
    ):
        report = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=dispatch_engine,
            event_bus=event_bus,
            container=container,
        )

    assert report.total_failed == 0, f"Unexpected failures: {report.results}"
    assert len(report.quarantined_handlers) == 1, (
        f"Expected only _BadHandler quarantined, got {report.quarantined_handlers}"
    )
    assert (
        report.quarantined_handlers[0].reason == EnumQuarantineReason.ASYNC_INCOMPATIBLE
    )
    assert "_BadHandler" in report.quarantined_handlers[0].handler_name
    # Contract must be WIRED (good handler resolved via zero-arg path)
    wired_results = [r for r in report.results if r.outcome == EnumWiringOutcome.WIRED]
    assert len(wired_results) == 1, f"Expected 1 WIRED result, got {report.results}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_all_quarantined_contract_reports_skipped() -> None:
    """When all handlers quarantine, contract outcome is SKIPPED [OMN-9457].

    total_failed must remain 0. The reason must be "all handlers quarantined".
    No dispatcher or route registrations occur.
    """

    class _AsyncIncompatHandlerA:
        def __init__(self) -> None:
            raise RuntimeError(_ASYNCIO_RUN_MSG)

        async def handle(self, envelope: object) -> None:
            return None

    class _AsyncIncompatHandlerB:
        def __init__(self) -> None:
            raise RuntimeError(_RUNNER_RUN_MSG)

        async def handle(self, envelope: object) -> None:
            return None

    entry_a = _make_entry("_AsyncIncompatHandlerA", __name__)
    entry_b = _make_entry("_AsyncIncompatHandlerB", __name__)
    contract = _make_contract(
        "node_all_quarantined",
        (entry_a, entry_b),
        topic="onex.evt.test.all_quar.v1",
    )
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())

    async def _get_service_async(cls: type) -> object:
        if cls is _AsyncIncompatHandlerA:
            raise RuntimeError(_ASYNCIO_RUN_MSG)
        raise RuntimeError(_RUNNER_RUN_MSG)

    container = MagicMock()
    container.get_service = MagicMock(side_effect=RuntimeError(_ASYNCIO_RUN_MSG))
    container.get_service_async = AsyncMock(side_effect=_get_service_async)

    event_bus = MagicMock()
    event_bus.subscribe = AsyncMock()
    dispatch_engine = _make_dispatch_engine()

    def _import_side_effect(module: str, name: str) -> type:
        if name == "_AsyncIncompatHandlerA":
            return _AsyncIncompatHandlerA
        return _AsyncIncompatHandlerB

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        side_effect=_import_side_effect,
    ):
        report = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=dispatch_engine,
            event_bus=event_bus,
            container=container,
        )

    assert report.total_failed == 0, f"Unexpected failures: {report.results}"
    assert len(report.quarantined_handlers) == 2
    skipped_results = [
        r for r in report.results if r.outcome == EnumWiringOutcome.SKIPPED
    ]
    assert len(skipped_results) == 1
    assert skipped_results[0].reason == "all handlers quarantined"
    dispatch_engine.register_dispatcher.assert_not_called()
    dispatch_engine.register_route.assert_not_called()
