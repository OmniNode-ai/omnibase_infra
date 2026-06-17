# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for per-handler resolution-failure quarantine in wire_from_manifest (OMN-13203).

Before OMN-13203 the per-handler resolution-failure path crashed runtime boot:
``ServiceHandlerResolver`` raises a bare ``TypeError`` when a handler's
constructor cannot be satisfied (Step 6 unsatisfiable-ctor / Step 2
ctor-arg-mismatch). The bare ``raise`` at the resolver call site propagated
through the OMN-8735 ``except TypeError: raise`` guards, exited the kernel via
its ``except Exception: raise``, and the process died BEFORE binding the
``:8086`` health server — Docker then crash-looped the runtime-effects
container. A single bad handler took down every healthy handler in the manifest.

This containment quarantines exactly those deterministic, never-recoverable
per-handler wiring bugs:

- unsatisfiable-ctor / ctor-arg-mismatch ``TypeError`` (resolver Steps 2/6),
- not-handle-shaped / blank-required ``ValueError`` for a single handler entry,

so ``wire_from_manifest`` COMPLETES, the runtime binds its health server, and
the bad handler is REPORTED (``failed >= 1`` + WARNING log) — never silently
skipped. Truly-fatal infra errors (broker/DB unreachable, manifest unreadable)
surface as ``ModelOnexError`` / ``InfraConnectionError`` / ``ConnectionError`` /
``OSError`` — NOT a bare resolver ``TypeError``/``ValueError`` — and STILL crash
boot. ``ONEX_WIRING_STRICT_MODE=1`` re-raises, preserving the pre-OMN-13203
boot-crash invariant.

Tests:
  1. An unsatisfiable-ctor handler is quarantined, wire_from_manifest COMPLETES,
     ``total_failed >= 1``, reason recorded as UNRESOLVABLE_HANDLER + logged.
  2. A not-handle-shaped ValueError is quarantined identically.
  3. Mixed: a good handler wires, the unresolvable one is failed/quarantined.
  4. A truly-fatal infra error (broker/DB unreachable surfaced as a real
     InfraConnectionError, NOT a bare resolver TypeError) STILL propagates —
     the guard did not swallow it.
  5. ONEX_WIRING_STRICT_MODE=1 re-raises the resolver TypeError (boot fails).
  6. total_failed counts the resolution-quarantine; total_quarantined still
     counts it as a contained handler.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext
from omnibase_infra.protocols import ProtocolEventBusLike
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
    bus = MagicMock(spec=ProtocolEventBusLike)
    bus.subscribe = AsyncMock()
    return bus


# ---------------------------------------------------------------------------
# Handler classes that exercise the resolver failure surface
# ---------------------------------------------------------------------------


class _HandlerNormalZeroArg:
    """Normal async-safe handler — wires via zero-arg resolver step."""

    async def handle(self, envelope: object) -> None:
        return None


class _HandlerUnsatisfiableCtor:
    """Handler the resolver cannot construct — required ctor params no precedence
    path (node registry / container / known-injectable) can satisfy.

    The resolver's Step 6 raises a bare ``TypeError`` for this case (the exact
    path that crashed boot before OMN-13203).
    """

    def __init__(self, some_unresolvable_dependency: object) -> None:
        self._dep = some_unresolvable_dependency

    async def handle(self, envelope: object) -> None:
        return None


# ---------------------------------------------------------------------------
# 1. Unsatisfiable-ctor TypeError is quarantined; wire COMPLETES; failed >= 1.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unsatisfiable_handler_quarantined_completes_and_counts_failed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The OMN-13203 regression: an unsatisfiable handler no longer crashes boot.

    wire_from_manifest COMPLETES (no exception), the handler is contained, and
    it is REPORTED — failed >= 1 and the reason is logged loudly with the
    contract + handler + reason.
    """
    contract = _make_contract(
        node_name="node_unsatisfiable",
        handler_entries=(("fake.module", "_HandlerUnsatisfiableCtor"),),
    )
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())

    with (
        caplog.at_level(logging.WARNING),
        patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=_HandlerUnsatisfiableCtor,
        ),
    ):
        # No exception: wire_from_manifest COMPLETES so the kernel proceeds to
        # freeze() and the :8086 health-server bind.
        report = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=_make_dispatch_engine(),
            event_bus=_make_event_bus(),
            container=None,
        )

    # REPORTED, not silently skipped: failed >= 1.
    assert report.total_failed >= 1
    assert report.total_failed == 1
    # Still counted as a contained handler.
    assert report.total_quarantined == 1
    assert len(report.quarantined_handlers) == 1

    q = report.quarantined_handlers[0]
    assert q.handler_name == "_HandlerUnsatisfiableCtor"
    assert q.handler_module == "fake.module"
    assert q.contract_name == "node_unsatisfiable"
    assert q.package_name == "test_pkg"
    assert q.reason is EnumQuarantineReason.UNRESOLVABLE_HANDLER
    assert q.detail  # sanitized resolver TypeError text recorded

    # The contract is SKIPPED (its only handler was contained) but the handler
    # is still surfaced as failed at the report level.
    result = report.results[0]
    assert result.outcome is EnumWiringOutcome.SKIPPED
    assert len(result.quarantined_handlers) == 1

    # Logged loudly: contract + handler + reason all appear in the WARNING log.
    logged = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "_HandlerUnsatisfiableCtor" in logged
    assert "node_unsatisfiable" in logged
    assert EnumQuarantineReason.UNRESOLVABLE_HANDLER.value in logged


# ---------------------------------------------------------------------------
# 2. A per-handler ValueError (not-handle-shaped / blank-required) is contained.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_value_error_handler_quarantined_completes_and_counts_failed() -> None:
    """A per-handler ValueError from resolution is contained like the TypeError."""
    contract = _make_contract(
        node_name="node_value_error",
        handler_entries=(("fake.module", "_HandlerNotShaped"),),
    )
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())

    class _Resolver:
        def resolve(self, ctx: object) -> object:
            # Simulate a not-handle-shaped / blank-required per-handler bug.
            raise ValueError("handler entry is not handle-shaped: missing handle()")

    with (
        patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=_HandlerNormalZeroArg,
        ),
        patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring.ServiceHandlerResolver",
            return_value=_Resolver(),
        ),
    ):
        report = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=_make_dispatch_engine(),
            event_bus=_make_event_bus(),
            container=None,
        )

    assert report.total_failed == 1
    assert report.total_quarantined == 1
    q = report.quarantined_handlers[0]
    assert q.reason is EnumQuarantineReason.UNRESOLVABLE_HANDLER
    assert "handle-shaped" in q.detail


# ---------------------------------------------------------------------------
# 3. Mixed: good handler wires, unresolvable handler is failed/quarantined.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mixed_good_and_unresolvable_handler() -> None:
    """One good handler wires; the unresolvable one is contained + counted failed."""
    contract = _make_contract(
        node_name="node_mixed_unresolvable",
        handler_entries=(
            ("fake.module", "_HandlerNormalZeroArg"),
            ("fake.module", "_HandlerUnsatisfiableCtor"),
        ),
    )
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())

    def _import_side_effect(module: str, name: str) -> type:
        if name == "_HandlerNormalZeroArg":
            return _HandlerNormalZeroArg
        if name == "_HandlerUnsatisfiableCtor":
            return _HandlerUnsatisfiableCtor
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

    # Contract WIRED because at least one handler resolved; the bad handler is
    # still surfaced as failed + quarantined at the report level.
    assert report.total_wired == 1
    assert report.total_failed == 1
    assert report.total_quarantined == 1
    assert report.quarantined_handlers[0].handler_name == "_HandlerUnsatisfiableCtor"
    assert (
        report.quarantined_handlers[0].reason
        is EnumQuarantineReason.UNRESOLVABLE_HANDLER
    )
    result = report.results[0]
    assert result.outcome is EnumWiringOutcome.WIRED
    assert len(result.dispatchers_registered) == 1


# ---------------------------------------------------------------------------
# 4. Truly-fatal infra error STILL propagates (the guard did not swallow it).
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fatal_infra_error_still_crashes_boot() -> None:
    """A real infra outage (broker/DB unreachable) must NOT be quarantined.

    Infra failures surface as ModelOnexError / InfraConnectionError /
    ConnectionError / OSError — never a bare resolver TypeError/ValueError. The
    new resolution-quarantine arms catch only TypeError/ValueError, so an
    InfraConnectionError raised from resolution propagates: the contract fails
    (and the kernel would crash boot). The guard must NOT have widened to
    ``except Exception``.
    """
    contract = _make_contract(
        node_name="node_infra_down",
        handler_entries=(("fake.module", "_HandlerNormalZeroArg"),),
    )
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())

    class _ResolverInfraDown:
        def resolve(self, ctx: object) -> object:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="resolve_handler",
            )
            raise InfraConnectionError(
                "broker/DB unreachable during handler resolution",
                context=context,
            )

    with (
        patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=_HandlerNormalZeroArg,
        ),
        patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring.ServiceHandlerResolver",
            return_value=_ResolverInfraDown(),
        ),
    ):
        report = await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=_make_dispatch_engine(),
            event_bus=_make_event_bus(),
            container=None,
        )

    # InfraConnectionError is collected as a contract-level FAILED row (NOT
    # quarantined), so it is NOT demoted to a silent skip. In strict mode this
    # raises; in non-strict mode total_failed > 0 still fails the kernel
    # postcondition / gate. Crucially it was NOT swallowed into a quarantine.
    assert report.total_quarantined == 0
    assert report.total_failed == 1
    assert report.results[0].outcome is EnumWiringOutcome.FAILED
    assert "InfraConnectionError" in report.results[0].reason


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fatal_infra_error_raises_in_strict_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In strict mode an infra-down contract failure raises (boot fails)."""
    monkeypatch.setenv("ONEX_WIRING_STRICT_MODE", "1")
    contract = _make_contract(
        node_name="node_infra_down_strict",
        handler_entries=(("fake.module", "_HandlerNormalZeroArg"),),
    )
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())

    class _ResolverInfraDown:
        def resolve(self, ctx: object) -> object:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="resolve_handler",
            )
            raise InfraConnectionError("broker unreachable", context=context)

    with (
        patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=_HandlerNormalZeroArg,
        ),
        patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring.ServiceHandlerResolver",
            return_value=_ResolverInfraDown(),
        ),
        pytest.raises(Exception),
    ):
        await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=_make_dispatch_engine(),
            event_bus=_make_event_bus(),
            container=None,
        )


# ---------------------------------------------------------------------------
# 5. ONEX_WIRING_STRICT_MODE=1 re-raises the resolver TypeError (boot fails).
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_strict_mode_reraises_unresolvable_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict mode preserves the pre-OMN-13203 boot-crash invariant.

    The unsatisfiable-ctor TypeError must NOT be quarantined under strict mode;
    it propagates so the gate can still fail closed.
    """
    monkeypatch.setenv("ONEX_WIRING_STRICT_MODE", "1")
    contract = _make_contract(
        node_name="node_unsatisfiable_strict",
        handler_entries=(("fake.module", "_HandlerUnsatisfiableCtor"),),
    )
    manifest = ModelAutoWiringManifest(contracts=(contract,), errors=())

    with (
        patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=_HandlerUnsatisfiableCtor,
        ),
        pytest.raises(TypeError),
    ):
        await wire_from_manifest(
            manifest=manifest,
            dispatch_engine=_make_dispatch_engine(),
            event_bus=_make_event_bus(),
            container=None,
        )
