# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pool-taking contract handlers must resolve under strict wiring [OMN-17510].

The defect this file exists for, read live off ``onex-dev`` on
``omninode-runtime@sha256:589b8c3f`` (tag ``candidate-03fb690-20260901233601``)
at 2026-09-02T02:58Z::

    File ".../omnibase_infra/runtime/service_kernel.py", line 3233, in bootstrap
        auto_wiring_report = await wire_from_manifest(
    File ".../omnibase_core/services/service_handler_resolver.py", line 226, in resolve
        raise TypeError(
    TypeError: Handler ...handler_savings_correlation.HandlerSavingsCorrelation
    requires constructor parameters ['pool'] but no ownership_query, node
    registry explicit deps, container, or known injectable params could satisfy
    them.

``_build_runtime_handler_dependencies`` is ``ServiceHandlerResolver`` Step 2 —
the only precedence step that can supply a constructor parameter the resolver
has no provider for. Steps 3-5 cover the container, the three known injectables
(``event_bus`` / ``container`` / ``ownership_query``) and the zero-arg case; a
required ``pool`` matches none of them. OMN-16293 declared
``HandlerSavingsCorrelation`` in the node's ``handler_routing`` — which is what
makes ``wire_from_manifest`` resolve it — and wired it for its own use by
constructing it directly in the ``service_kernel`` §3.9 periodic loop, but never
registered its pool in that map. Under ``ONEX_WIRING_STRICT_MODE`` (what
onex-dev sets) the OMN-13203 per-handler quarantine is deliberately disabled and
the TypeError takes the whole runtime boot down with it.

Why the existing real-manifest gate did not catch it
----------------------------------------------------
``tests/integration/test_auto_wiring_real_manifest.py`` passes a ``MagicMock``
dispatch engine. ``handler_wiring._prepare_handler_wiring`` falls back to
``getattr(dispatch_engine, "_container", None)`` when no container is passed, so
the mock auto-creates a ``_container`` whose ``get_service`` returns a mock for
**every** handler class — resolver Step 3 succeeds unconditionally and no
constructor requirement is ever exercised. That gate is vacuous for this entire
defect class. These tests use a real ``MessageDispatchEngine`` (no ``_container``
attribute) and a real ``ModelONEXContainer`` (whose ``get_service`` raises
``ServiceResolutionError`` for an unregistered handler — the documented miss the
pod hit), so the precedence chain is exercised for real.

Scope: the class, not the one handler. Subjects are DISCOVERED — every contract
in the real ``onex.nodes`` manifest declaring a handler whose constructor
requires a ``pool`` — so the next EFFECT handler landed with an unregistered
pool fails here instead of at a strict-mode boot. There is no allowlist and no
skip set: a new pool-taking handler is either registered or this test is red.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

from omnibase_core.enums.enum_handler_resolution_outcome import (
    EnumHandlerResolutionOutcome,
)
from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_core.models.resolver.model_handler_resolver_context import (
    ModelHandlerResolverContext,
)
from omnibase_core.services.service_handler_resolver import ServiceHandlerResolver
from omnibase_infra.nodes.node_savings_estimation_compute.handlers.handler_savings_correlation import (
    HandlerSavingsCorrelation,
)
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.runtime.service_kernel import _build_runtime_handler_dependencies

_SAVINGS_CONTRACT = "node_savings_estimation_compute"
_SAVINGS_HANDLER = "HandlerSavingsCorrelation"


class _PoolSentinel:
    """Stands in for an ``asyncpg.Pool``.

    Every handler under test stores its pool and does no I/O in ``__init__``,
    so resolution is provable with no database. Two distinct instances are used
    so a test can assert WHICH pool a handler received, which is the OMN-16770
    half of this fix: ``HandlerSavingsCorrelation`` reads the ``application``
    database, not ``omnibase_infra``.
    """

    def __init__(self, label: str) -> None:
        self.label = label

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"_PoolSentinel({self.label!r})"


async def _noop_publisher(
    event_type: str,
    payload: object,
    topic: str | None,
    correlation_id: object,
    **kwargs: object,
) -> bool:
    return True


def _handler_requires_pool(handler_cls: type) -> bool:
    """True when ``pool`` is a REQUIRED constructor parameter.

    Mirrors the resolver's own required-parameter scan: concrete kinds only
    (a ``**kwargs``-shaped signature requires nothing) and no default.
    """
    try:
        signature = inspect.signature(handler_cls)
    except (TypeError, ValueError):  # pragma: no cover - unimportable signature
        return False
    parameter = signature.parameters.get("pool")
    return (
        parameter is not None
        and parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    )


def _pool_taking_contracts() -> list[ModelDiscoveredContract]:
    """Every real discovered contract declaring a pool-requiring handler."""
    selected: list[ModelDiscoveredContract] = []
    for contract in discover_contracts().contracts:
        if contract.handler_routing is None:
            continue
        for entry in contract.handler_routing.handlers:
            module = importlib.import_module(entry.handler.module)
            if _handler_requires_pool(getattr(module, entry.handler.name)):
                selected.append(contract)
                break
    return selected


def _kernel_dependencies(
    *,
    savings_correlation_pool: object | None,
) -> dict[str, dict[str, object]] | None:
    """The kernel's own dependency map, built by the kernel's own function.

    ``savings_correlation_pool`` is threaded rather than hardcoded so the
    non-vacuity test can prove that omitting it — the pre-OMN-17510 call site —
    is exactly what leaves the handler unresolvable.
    """
    return _build_runtime_handler_dependencies(
        _PoolSentinel("omnibase_infra"),
        savings_correlation_pool=savings_correlation_pool,
        savings_correlation_publisher=_noop_publisher,
    )


async def _wire(
    contracts: list[ModelDiscoveredContract],
    dependencies: dict[str, dict[str, object]] | None,
) -> object:
    """Run the production wiring path against real contracts.

    Real ``MessageDispatchEngine`` (carries no ``_container``, so the mock
    escape hatch is closed) and real ``ModelONEXContainer`` (raises
    ``ServiceResolutionError`` for unregistered handlers, the documented Step 3
    miss). ``event_bus=None`` + ``subscribe_immediately=False`` keep it offline.
    """
    return await wire_from_manifest(
        manifest=ModelAutoWiringManifest(contracts=contracts),
        dispatch_engine=MessageDispatchEngine(),
        event_bus=None,
        container=ModelONEXContainer(),
        subscribe_immediately=False,
        materialized_explicit_dependencies=dependencies,
    )


@pytest.mark.integration
def test_manifest_declares_at_least_one_pool_taking_handler() -> None:
    """Guard against a vacuous suite: the subject set must be non-empty.

    If discovery breaks or entry points are not installed, the wiring test
    below would pass over zero contracts and prove nothing.
    """
    contracts = _pool_taking_contracts()
    names = {contract.name for contract in contracts}
    assert names, (
        "No discovered contract declares a handler requiring a `pool` — "
        "discovery is broken or the onex.nodes entry points are not installed, "
        "and the strict-wiring gate below would be vacuous."
    )
    assert _SAVINGS_CONTRACT in names, (
        f"{_SAVINGS_CONTRACT} is missing from the pool-taking contract set. "
        f"It is the contract this gate was written for (OMN-17510); if its "
        f"handler_routing entry was removed, delete this assertion deliberately "
        f"rather than letting the gate silently stop covering it. Found: "
        f"{sorted(names)}"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pool_taking_handlers_resolve_under_strict_wiring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The onex-dev boot shape: strict wiring over every pool-taking contract.

    This is the assertion that was red before OMN-17510 — it reproduced the pod
    TypeError verbatim, on the same code path (``wire_from_manifest`` ->
    ``_prepare_handler_wiring`` -> ``ServiceHandlerResolver.resolve``).
    """
    monkeypatch.setenv("ONEX_WIRING_STRICT_MODE", "1")

    contracts = _pool_taking_contracts()
    dependencies = _kernel_dependencies(
        savings_correlation_pool=_PoolSentinel("application")
    )

    report = await _wire(contracts, dependencies)

    failed = [
        result
        for result in report.results  # type: ignore[attr-defined]
        if str(result.outcome).endswith("FAILED")
    ]
    assert not failed, (
        "Strict-mode wiring reported failures for contracts declaring a "
        "pool-requiring handler:\n"
        + "\n".join(f"  {r.contract_name}: {r.reason}" for r in failed)
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_strict_wiring_is_red_without_the_application_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-vacuity: the gate above detects the exact defect it was written for.

    Omitting ``savings_correlation_pool`` reproduces the pre-fix call site. If
    this stops raising, the gate above has stopped proving anything and the
    next unregistered pool would ship silently.
    """
    monkeypatch.setenv("ONEX_WIRING_STRICT_MODE", "1")

    contracts = _pool_taking_contracts()
    dependencies = _kernel_dependencies(savings_correlation_pool=None)

    with pytest.raises(TypeError) as excinfo:
        await _wire(contracts, dependencies)

    message = str(excinfo.value)
    assert _SAVINGS_HANDLER in message and "['pool']" in message, (
        f"Expected the resolver Step 6 exhaustion TypeError naming "
        f"{_SAVINGS_HANDLER} and its unsatisfied ['pool'] parameter; got: "
        f"{message}"
    )


@pytest.mark.integration
def test_savings_correlation_is_registered_with_the_application_pool() -> None:
    """OMN-16770 guard: the registered pool is NOT the omnibase_infra pool.

    ``HandlerSavingsCorrelation`` reads ``omninode_internal.savings_injection_signals``
    and joins ``llm_call_metrics`` / ``session_outcomes`` / ``savings_estimates``,
    all of which live in the ``application`` database (physical
    ``omnidash_analytics``). Satisfying its constructor from
    ``registration_service.postgres_pool`` (``OMNIBASE_INFRA_DB_URL``) would make
    wiring green while reinstating the OMN-16770 ``UndefinedTableError`` on every
    60s tick — a fix that trades a loud crash for a silent one.
    """
    application_pool = _PoolSentinel("application")
    dependencies = _kernel_dependencies(savings_correlation_pool=application_pool)

    assert dependencies is not None
    assert _SAVINGS_HANDLER in dependencies, (
        f"{_SAVINGS_HANDLER} has no entry in the kernel's explicit-dependency "
        f"map; ServiceHandlerResolver Step 2 is the only step that can satisfy "
        f"its `pool`. Registered: {sorted(dependencies)}"
    )
    registered_pool = dependencies[_SAVINGS_HANDLER]["pool"]
    assert registered_pool is application_pool, (
        "HandlerSavingsCorrelation must be registered with the "
        "OMNINODE_INTERNAL_DB_URL-bound application pool, not "
        f"{registered_pool!r} (OMN-16770)."
    )
    assert (
        registered_pool is not dependencies["HandlerBaselinesBatchCompute"]["pool"]
    ), (
        "HandlerSavingsCorrelation and HandlerBaselinesBatchCompute must not "
        "share a pool: they bind different databases (application vs "
        "omnibase_infra)."
    )


@pytest.mark.integration
def test_savings_correlation_resolves_via_node_registry_with_its_publisher() -> None:
    """Step 2 is the path taken, and the publisher rides along with the pool.

    Without the publisher the auto-wired instance computes an estimate and then
    logs ``no publisher configured, dropping computed estimate`` — the batch
    would run and emit nothing. The periodic §3.9 instance and the auto-wired
    instance must behave identically.
    """
    application_pool = _PoolSentinel("application")
    dependencies = _kernel_dependencies(savings_correlation_pool=application_pool)
    assert dependencies is not None

    resolution = ServiceHandlerResolver().resolve(
        ModelHandlerResolverContext(
            handler_cls=HandlerSavingsCorrelation,
            handler_module=HandlerSavingsCorrelation.__module__,
            handler_name=_SAVINGS_HANDLER,
            contract_name=_SAVINGS_CONTRACT,
            node_name=_SAVINGS_CONTRACT,
            explicit_dependency_shape=None,
            materialized_explicit_dependencies=dependencies,
            event_bus=None,
            container=ModelONEXContainer(),
            ownership_query=None,
        )
    )

    assert (
        resolution.outcome is EnumHandlerResolutionOutcome.RESOLVED_VIA_NODE_REGISTRY
    ), (
        f"Expected resolution through the explicit-dependency map (Step 2); got "
        f"{resolution.outcome}."
    )
    handler = resolution.handler_instance
    assert isinstance(handler, HandlerSavingsCorrelation)
    assert handler._pool is application_pool
    assert handler._publisher is _noop_publisher, (
        "The auto-wired instance received no publisher — a bus-triggered batch "
        "would drop every estimate it computes."
    )


@pytest.mark.integration
def test_kernel_binds_every_dependency_slot_it_declares() -> None:
    """The half that actually caused the outage: the call site must pass it.

    A dependency slot on ``_build_runtime_handler_dependencies`` is inert until
    ``bootstrap`` binds it. OMN-16293 landed the handler and its contract entry
    but no slot at all; the failure mode this pins is the next one over —
    a slot added and then never bound, which produces the identical resolver
    Step 6 exhaustion with a builder that *looks* correct in isolation.

    Every keyword-only parameter of the builder must appear as a keyword at the
    kernel's call site. Asserted over the AST rather than the source text, so
    reformatting, renaming the argument expression, or moving the call within
    ``service_kernel.py`` does not move this gate; only dropping the binding
    does.
    """
    builder_name = _build_runtime_handler_dependencies.__name__
    declared_slots = {
        name
        for name, parameter in inspect.signature(
            _build_runtime_handler_dependencies
        ).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }
    assert declared_slots, (
        f"{builder_name} declares no keyword-only dependency slots — the "
        f"signature this gate reads has changed shape; update the gate "
        f"deliberately rather than letting it pass vacuously."
    )

    kernel_source = Path(
        inspect.getsourcefile(_build_runtime_handler_dependencies) or ""
    )
    tree = ast.parse(kernel_source.read_text(encoding="utf-8"))
    call_sites = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == builder_name
    ]
    assert call_sites, (
        f"No call to {builder_name} found in {kernel_source.name}; the kernel "
        f"no longer builds a dependency map, or the call moved out of this "
        f"module."
    )

    for call in call_sites:
        bound = {keyword.arg for keyword in call.keywords if keyword.arg is not None}
        unbound = sorted(declared_slots - bound)
        assert not unbound, (
            f"{kernel_source.name}:{call.lineno} calls {builder_name} without "
            f"binding {unbound}. An unbound dependency slot registers nothing, "
            f"so any contract-declared handler needing it exhausts the resolver "
            f"precedence chain and — under ONEX_WIRING_STRICT_MODE — crashes "
            f"runtime boot (OMN-17510)."
        )
