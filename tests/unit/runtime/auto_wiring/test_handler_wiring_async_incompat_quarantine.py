# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for async-incompatible handler quarantine in wire_from_manifest (OMN-9457).

After .201 runtime image refresh exposed 100+ contracts failing auto-wiring
with ``RuntimeError: asyncio.run() cannot be called from a running event loop``,
wire_from_manifest was extended with deterministic containment:

- Handlers whose construction raises the specific async-incompat signature are
  quarantined (dispatch engine receives nothing).
- Quarantined handlers are surfaced via
  ``ModelAutoWiringReport.quarantined_handlers`` so follow-up migration tickets
  are visible rather than silent.
- Normal handlers continue to wire cleanly in the same contract scan.
- A contract whose every handler quarantines reports ``SKIPPED`` with
  ``reason == "all handlers quarantined"``.

These tests cover:
  1. Positive: a normal zero-arg handler wires through without quarantine.
  2. The RuntimeError signature detector matches CPython's exact message,
     matches wrapped RuntimeErrors (via ``__cause__``), and does NOT match
     unrelated RuntimeErrors.
  3. Negative containment: a handler whose __init__ raises the exact
     ``asyncio.run() cannot be called from a running event loop`` message is
     quarantined, does NOT poison the wire_from_manifest call, and appears
     in ``report.quarantined_handlers``.
  4. Mixed: a contract with one good handler and one async-incompat handler
     still reports WIRED; the bad handler still lands in
     ``quarantined_handlers``.
  5. Fully-contained contract: when every handler quarantines, the contract
     reports SKIPPED.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_core.enums.enum_handler_resolution_outcome import (
    EnumHandlerResolutionOutcome,
)
from omnibase_infra.enums import EnumMessageCategory
from omnibase_infra.runtime.auto_wiring.enum_quarantine_reason import (
    EnumQuarantineReason,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    PreparedContractWiring,
    PreparedWiring,
    _commit_contract_wiring,
    _is_async_incompat_runtime_error,
    _skip_dispatcher,
    wire_from_manifest,
)
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


def _make_contract(
    *,
    node_name: str,
    handler_entries: tuple[tuple[str, str], ...],
    topic: str = "onex.evt.test.event.v1",
) -> ModelDiscoveredContract:
    entries = tuple(
        ModelHandlerRoutingEntry(
            handler=ModelHandlerRef(name=name, module=module),
            event_model=ModelHandlerRef(
                name=f"ModelTestEvent{idx}", module="fake.models"
            ),
            operation=None,
        )
        for idx, (module, name) in enumerate(handler_entries)
    )
    return ModelDiscoveredContract(
        name=node_name,
        node_type="ORCHESTRATOR_GENERIC",
        package_name="test_pkg",
        entry_point_name=node_name,
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/contract.yaml"),
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


def _make_event_bus() -> MagicMock:
    bus = MagicMock()
    bus.subscribe = AsyncMock()
    return bus


# ---------------------------------------------------------------------------
# _is_async_incompat_runtime_error tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_detector_matches_cpython_exact_message() -> None:
    """The detector keys on CPython's exact asyncio.run() message."""
    exc = RuntimeError("asyncio.run() cannot be called from a running event loop")
    assert _is_async_incompat_runtime_error(exc) is True


@pytest.mark.unit
def test_detector_matches_wrapped_via_cause() -> None:
    """Wrapped RuntimeError via ``raise X from original`` is still detected."""
    inner = RuntimeError("asyncio.run() cannot be called from a running event loop")
    outer = RuntimeError("handler init failed")
    outer.__cause__ = inner
    assert _is_async_incompat_runtime_error(outer) is True


@pytest.mark.unit
def test_detector_matches_wrapped_via_context() -> None:
    """Implicit exception chaining via ``__context__`` is detected."""
    try:
        try:
            raise RuntimeError(
                "asyncio.run() cannot be called from a running event loop"
            )
        except RuntimeError:
            raise RuntimeError("secondary failure") from None
    except RuntimeError as captured:
        # ``raise ... from None`` clears __cause__ but preserves __context__
        # only when suppress isn't used; we set __context__ manually to the
        # inner exception to simulate the implicit chain the production path
        # preserves.
        inner = RuntimeError("asyncio.run() cannot be called from a running event loop")
        captured.__context__ = inner
        assert _is_async_incompat_runtime_error(captured) is True


@pytest.mark.unit
def test_detector_rejects_unrelated_runtime_error() -> None:
    """Generic RuntimeErrors must NOT be classified as async-incompat."""
    assert (
        _is_async_incompat_runtime_error(RuntimeError("database connection lost"))
        is False
    )


@pytest.mark.unit
def test_detector_rejects_non_runtime_error() -> None:
    """ValueError / other types are never async-incompat."""
    assert _is_async_incompat_runtime_error(ValueError("not a runtime error")) is False


# ---------------------------------------------------------------------------
# Handler classes used in wiring tests
# ---------------------------------------------------------------------------


class _HandlerNormalZeroArg:
    """Normal async-safe handler — wires via zero-arg resolver step."""

    async def handle(self, envelope: object) -> None:
        return None


class _HandlerAsyncIncompatInInit:
    """Handler whose __init__ calls asyncio.run() inside running loop.

    Simulates the 100+ omnimarket/omnibase_infra handlers exposed by the
    2026-04-22 .201 image refresh.
    """

    def __init__(self) -> None:
        # Raise the exact signature wire_from_manifest must contain.
        raise RuntimeError("asyncio.run() cannot be called from a running event loop")

    async def handle(self, envelope: object) -> None:
        return None


class _HandlerAsyncIncompatInInitWithWrap:
    """Async-incompat handler that wraps the inner RuntimeError in a new one."""

    def __init__(self) -> None:
        try:
            raise RuntimeError(
                "asyncio.run() cannot be called from a running event loop"
            )
        except RuntimeError as exc:
            raise RuntimeError("HandlerFoo: startup failed") from exc

    async def handle(self, envelope: object) -> None:
        return None


# ---------------------------------------------------------------------------
# wire_from_manifest quarantine tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_normal_handler_wires_without_quarantine() -> None:
    """Baseline: a normal zero-arg handler wires, no quarantine entries."""
    contract = _make_contract(
        node_name="node_normal",
        handler_entries=(("fake.module", "_HandlerNormalZeroArg"),),
    )
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=_HandlerNormalZeroArg,
    ):
        report = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=_make_dispatch_engine(),
            event_bus=_make_event_bus(),
            container=None,
        )

    assert report.total_failed == 0
    assert report.total_wired == 1
    assert report.total_quarantined == 0
    assert report.quarantined_handlers == ()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_incompat_handler_is_quarantined_not_failed() -> None:
    """The OMN-9457 regression test: async-incompat does not poison startup."""
    contract = _make_contract(
        node_name="node_async_incompat",
        handler_entries=(("fake.module", "_HandlerAsyncIncompatInInit"),),
    )
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=_HandlerAsyncIncompatInInit,
    ):
        report = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=_make_dispatch_engine(),
            event_bus=_make_event_bus(),
            container=None,
        )

    assert report.total_failed == 0, (
        f"async-incompat handler must not fail wiring, got results={report.results}"
    )
    # Every handler was quarantined -> the contract is SKIPPED.
    assert report.total_wired == 0
    assert report.total_skipped == 1
    assert report.total_quarantined == 1
    assert len(report.quarantined_handlers) == 1
    q = report.quarantined_handlers[0]
    assert q.handler_name == "_HandlerAsyncIncompatInInit"
    assert q.handler_module == "fake.module"
    assert q.contract_name == "node_async_incompat"
    assert q.package_name == "test_pkg"
    assert q.reason is EnumQuarantineReason.ASYNC_INCOMPATIBLE
    assert "asyncio.run" in q.detail

    # Quarantined contract reports SKIPPED.
    result = report.results[0]
    assert result.outcome is EnumWiringOutcome.SKIPPED
    assert result.reason == "all handlers quarantined"
    assert len(result.quarantined_handlers) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_wrapped_async_incompat_is_quarantined() -> None:
    """Wrapped RuntimeError (raise X from original) is still contained."""
    contract = _make_contract(
        node_name="node_wrapped_incompat",
        handler_entries=(("fake.module", "_HandlerAsyncIncompatInInitWithWrap"),),
    )
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=_HandlerAsyncIncompatInInitWithWrap,
    ):
        report = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=_make_dispatch_engine(),
            event_bus=_make_event_bus(),
            container=None,
        )

    assert report.total_failed == 0
    assert report.total_quarantined == 1
    assert (
        report.quarantined_handlers[0].reason is EnumQuarantineReason.ASYNC_INCOMPATIBLE
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mixed_handlers_wire_good_quarantine_bad() -> None:
    """Good handler wires; bad handler is quarantined; contract is WIRED."""
    contract = _make_contract(
        node_name="node_mixed",
        handler_entries=(
            ("fake.module", "_HandlerNormalZeroArg"),
            ("fake.module", "_HandlerAsyncIncompatInInit"),
        ),
    )
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())

    def _import_side_effect(module: str, name: str) -> type:
        if name == "_HandlerNormalZeroArg":
            return _HandlerNormalZeroArg
        if name == "_HandlerAsyncIncompatInInit":
            return _HandlerAsyncIncompatInInit
        raise AssertionError(f"Unexpected import: {module}.{name}")

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        side_effect=_import_side_effect,
    ):
        report = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=_make_dispatch_engine(),
            event_bus=_make_event_bus(),
            container=None,
        )

    assert report.total_failed == 0
    assert (
        report.total_wired == 1
    )  # contract WIRED because at least one handler resolved
    assert report.total_quarantined == 1
    assert report.quarantined_handlers[0].handler_name == "_HandlerAsyncIncompatInInit"

    result = report.results[0]
    assert result.outcome is EnumWiringOutcome.WIRED
    # Dispatcher registered for the good handler only.
    assert len(result.dispatchers_registered) == 1
    assert len(result.quarantined_handlers) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unrelated_runtime_error_still_fails_contract() -> None:
    """A non-asyncio RuntimeError must NOT be quarantined (still a bug)."""

    class _HandlerGenericFailure:
        def __init__(self) -> None:
            raise RuntimeError("database connection string missing")

        async def handle(self, envelope: object) -> None:
            return None

    contract = _make_contract(
        node_name="node_generic_failure",
        handler_entries=(("fake.module", "_HandlerGenericFailure"),),
    )
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=_HandlerGenericFailure,
    ):
        report = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=_make_dispatch_engine(),
            event_bus=_make_event_bus(),
            container=None,
        )

    # Generic RuntimeError is still a wiring failure (collected, not quarantined).
    assert report.total_quarantined == 0
    assert report.total_failed == 1
    assert "database connection string" in report.results[0].reason


# ---------------------------------------------------------------------------
# Detector coverage added by the PR #1380 review feedback (OMN-9457).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_detector_matches_runner_run_message() -> None:
    """CPython ``asyncio.Runner.run`` raises a distinct message variant.

    Verified empirically on CPython 3.12: ``asyncio.Runner().run(...)`` from
    inside a running loop raises ``RuntimeError("Cannot run the event loop
    while another loop is running")``. Handlers that drive async work via
    ``asyncio.Runner`` (rather than the top-level ``asyncio.run``) must also
    be contained.
    """
    exc = RuntimeError("Cannot run the event loop while another loop is running")
    assert _is_async_incompat_runtime_error(exc) is True


@pytest.mark.unit
def test_detector_explores_both_cause_and_context() -> None:
    """Per PEP 3134 an exception may carry ``__cause__`` and ``__context__``
    simultaneously; the detector must explore both branches.

    Prior implementation short-circuited with ``current.__cause__ or
    current.__context__``, which skipped ``__context__`` whenever
    ``__cause__`` was truthy. This test pins the fix: an unrelated
    ``__cause__`` chain must not hide an async-incompat failure living on
    ``__context__``.
    """
    incompat = RuntimeError("asyncio.run() cannot be called from a running event loop")
    unrelated = RuntimeError("unrelated failure deeper down")
    outer = RuntimeError("handler init failed")
    # __cause__ branch leads nowhere relevant; the async-incompat signal
    # sits on __context__ and must still be discovered.
    outer.__cause__ = unrelated
    outer.__context__ = incompat
    assert _is_async_incompat_runtime_error(outer) is True


@pytest.mark.unit
def test_detector_explores_deeply_nested_context_under_cause() -> None:
    """The async-incompat signal may live arbitrarily deep in either branch.

    The detector walks both ``__cause__`` and ``__context__`` as a full
    traversal, not a linear chain that picks one per hop, so nested
    configurations like ``outer.__cause__.__context__ == incompat`` must be
    discovered.
    """
    incompat = RuntimeError("asyncio.run() cannot be called from a running event loop")
    mid = RuntimeError("intermediate wrapper")
    mid.__context__ = incompat
    outer = RuntimeError("outermost")
    outer.__cause__ = mid
    assert _is_async_incompat_runtime_error(outer) is True


@pytest.mark.unit
def test_detector_tolerates_self_referential_chain() -> None:
    """A self-referential exception chain must not cause an infinite loop.

    ``__cause__ is self`` is pathological but must still terminate.
    """
    exc = RuntimeError("loops back to itself")
    exc.__cause__ = exc
    exc.__context__ = exc
    # Terminates (no infinite loop) and, since no async-incompat message is
    # present anywhere in the chain, returns False.
    assert _is_async_incompat_runtime_error(exc) is False


# ---------------------------------------------------------------------------
# Strict "all handlers quarantined" check added by PR #1380 review feedback.
# ---------------------------------------------------------------------------


def _make_prepared_wiring_skip(
    *, handler_name: str, handler_module: str = "fake.module"
) -> PreparedWiring:
    """Construct a PreparedWiring representing a LOCAL_OWNERSHIP_SKIP entry."""
    return PreparedWiring(
        dispatcher_id="",
        dispatcher=_skip_dispatcher,
        category=EnumMessageCategory.EVENT,
        message_types={handler_name},
        handler_name=handler_name,
        handler_module=handler_module,
        resolution_outcome=EnumHandlerResolutionOutcome.RESOLVED_VIA_LOCAL_OWNERSHIP_SKIP,
        skip_reason="local ownership skip",
    )


def _make_prepared_wiring_quarantined(
    *, handler_name: str, handler_module: str = "fake.module"
) -> PreparedWiring:
    """Construct a PreparedWiring representing a quarantined entry."""
    return PreparedWiring(
        dispatcher_id="",
        dispatcher=_skip_dispatcher,
        category=EnumMessageCategory.EVENT,
        message_types={handler_name},
        handler_name=handler_name,
        handler_module=handler_module,
        resolution_outcome=EnumHandlerResolutionOutcome.UNRESOLVABLE,
        quarantine_reason=EnumQuarantineReason.ASYNC_INCOMPATIBLE,
        quarantine_detail="RuntimeError: asyncio.run() cannot be called ...",
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_commit_reports_skipped_when_every_handler_is_quarantined() -> None:
    """Strict 'all handlers quarantined' path: every prepared is quarantined."""
    contract = _make_contract(
        node_name="node_all_quarantined",
        handler_entries=(
            ("fake.module", "_HandlerQ1"),
            ("fake.module", "_HandlerQ2"),
        ),
    )
    pcw = PreparedContractWiring(
        contract=contract,
        prepared_wirings=[
            _make_prepared_wiring_quarantined(handler_name="_HandlerQ1"),
            _make_prepared_wiring_quarantined(handler_name="_HandlerQ2"),
        ],
        subscription_topics=[],
        environment="test",
    )

    result = await _commit_contract_wiring(pcw, _make_dispatch_engine(), None)

    assert result.outcome is EnumWiringOutcome.SKIPPED
    assert result.reason == "all handlers quarantined"
    assert len(result.quarantined_handlers) == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_commit_reports_wired_for_mixed_skip_and_quarantine() -> None:
    """Mixed resolver-skip + quarantine must NOT report 'all quarantined'.

    Prior implementation used ``not has_live_handler and quarantined``,
    which ALSO matched the case where some handlers were
    ``RESOLVED_VIA_LOCAL_OWNERSHIP_SKIP`` and the rest quarantined — that
    violates the stated meaning of reason ``"all handlers quarantined"``.
    The fixed predicate requires *every* prepared handler to be
    quarantined; otherwise the contract travels the normal WIRED path and
    skipped handlers land in ``skipped_handlers`` with their resolver
    reason preserved.
    """
    contract = _make_contract(
        node_name="node_mixed_skip_quarantine",
        handler_entries=(
            ("fake.module", "_HandlerSkipped"),
            ("fake.module", "_HandlerQuarantined"),
        ),
    )
    pcw = PreparedContractWiring(
        contract=contract,
        prepared_wirings=[
            _make_prepared_wiring_skip(handler_name="_HandlerSkipped"),
            _make_prepared_wiring_quarantined(handler_name="_HandlerQuarantined"),
        ],
        subscription_topics=[],
        environment="test",
    )

    result = await _commit_contract_wiring(pcw, _make_dispatch_engine(), None)

    # Mixed -> WIRED with the skipped and quarantined handlers each in their
    # dedicated collections, never collapsed into "all handlers quarantined".
    assert result.outcome is EnumWiringOutcome.WIRED
    assert result.reason != "all handlers quarantined"
    assert len(result.skipped_handlers) == 1
    assert result.skipped_handlers[0].handler_name == "_HandlerSkipped"
    assert len(result.quarantined_handlers) == 1
    assert result.quarantined_handlers[0].handler_name == "_HandlerQuarantined"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_commit_empty_prepared_wirings_does_not_report_all_quarantined() -> None:
    """A contract with zero prepared wirings must not take the quarantine path.

    ``all_handlers_quarantined`` explicitly requires ``bool(prepared_wirings)``
    so an empty list evaluates to False rather than the vacuous ``all(())``.
    """
    contract = _make_contract(
        node_name="node_no_handlers",
        handler_entries=(),
    )
    pcw = PreparedContractWiring(
        contract=contract,
        prepared_wirings=[],
        subscription_topics=[],
        environment="test",
    )

    result = await _commit_contract_wiring(pcw, _make_dispatch_engine(), None)

    assert result.outcome is EnumWiringOutcome.WIRED
    assert result.reason != "all handlers quarantined"
    assert result.quarantined_handlers == ()
