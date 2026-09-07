# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The readiness gate may only require what the interleave will attempt (OMN-17372).

RED on the parent commit (``f6f68847``). The gate shipped by #3263 derived its
required set from ``manifest.contracts`` — every contract that subscribes an
``onex.cmd.*`` topic — while ``ModelContractAttachResult`` rows are produced
only for the ``eligible`` subset inside ``subscribe_wired_contract_topics``,
which is SIX filters narrower. The gate hand-mirrored exactly one of the six
(``plugin_managed``). A contract caught by any of the other five was required,
could never be recorded, sat in ``pending`` forever, and pinned ``/ready`` at
503 for the life of the process: no timeout to expire, and — for the one filter
that logged nothing — not a single line to read.

Measured on the .201 dev lane at 0.38.21, 2026-09-07T15:1xZ::

    required_contracts 155   attached_contracts 154
    not_ready_contracts 0    failed_contracts 0
    pending_contracts 1      PENDING: ['node_contract_resolver_bridge']
    phase wiring_in_progress
    event_bus_readiness: is_ready=True, required_topics_ready=True

Zero NOT_READY, zero FAILED, the event bus fully ready: the ONLY thing holding
the runtime at 503 was one contract that can never report.

The fix is at the condition, never a timeout and never a default-open:

* the two exclusions decidable from the CONTRACT alone live in
  ``contract_subscribes_a_command_topic``;
* the rest are reported by the interleave itself through
  ``ContractAttachReadinessGate.exclude``, so eligibility has ONE authority;
* :func:`test_every_eligibility_filter_reports_its_exclusion` closes the CLASS
  by failing if a future filter is added to the interleave without one.

Related Tickets:
    - OMN-17372: readiness must require the wired command topics.
    - OMN-17534: the candidate boot gate this wedged.
    - OMN-13237: the per-contract provision -> confirm-ready -> attach interleave.
    - OMN-15474 / OMN-10864 / OMN-17562: three of the six eligibility filters.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from omnibase_infra.event_bus.enum_contract_attach_exclusion_reason import (
    EnumContractAttachExclusionReason,
)
from omnibase_infra.event_bus.enum_contract_attach_status import (
    EnumContractAttachStatus,
)
from omnibase_infra.event_bus.model_contract_attach_exclusion import (
    ModelContractAttachExclusion,
)
from omnibase_infra.event_bus.model_contract_attach_result import (
    ModelContractAttachResult,
)
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    subscribe_wired_contract_topics,
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
from omnibase_infra.runtime.enums.enum_contract_attach_gate_phase import (
    EnumContractAttachGatePhase,
)
from omnibase_infra.runtime.health.contract_attach_readiness_gate import (
    ContractAttachReadinessGate,
    contract_subscribes_a_command_topic,
    derive_required_contract_names,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

pytestmark = pytest.mark.unit

_THIS_MODULE = (
    "tests.unit.runtime.health.test_omn17372_required_set_matches_the_interleave"
)

#: A real ONEX command topic shape. The bridge below subscribes the literal
#: topic the shipped contract declares.
_BRIDGE_COMMAND_TOPIC = "onex.cmd.platform.contract-resolve-requested.v1"
_ROUTER_COMMAND_TOPIC = "onex.cmd.omnibase-infra.delegation-request.v1"

_BRIDGE = "node_contract_resolver_bridge_fixture"
_ROUTER = "node_delegation_router_fixture"

_PATCH_IMPORT_HANDLER = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class"
)


# ---------------------------------------------------------------------------
# Handlers — shaped, not mocked. A MagicMock satisfies every ``callable()``
# probe by accident and would make these tests pass against a broken predicate.
# ---------------------------------------------------------------------------


class DelegationRouterHandler:
    """A live in-process handler: the kernel really dispatches this one."""

    def handle(self, input_data: dict[str, Any]) -> dict[str, Any]:
        return {"routed": True, "echo": input_data}


def _import_by_name(_module: str, class_name: str) -> type:
    return {"DelegationRouterHandler": DelegationRouterHandler}[class_name]


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


def _bridge_contract() -> ModelDiscoveredContract:
    """The shipped ``node_contract_resolver_bridge`` shape.

    Subscribes a command topic and declares NO ``handler_routing``: it is a
    transitional HTTP bridge served by its own process (OMN-2756), and nothing
    anywhere attaches a Kafka consumer to its command topic.
    """
    return ModelDiscoveredContract(
        name=_BRIDGE,
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake") / _BRIDGE / "contract.yaml",
        entry_point_name=_BRIDGE,
        package_name="omnibase_infra",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(_BRIDGE_COMMAND_TOPIC,),
            publish_topics=(),
        ),
        handler_routing=None,
    )


def _router_contract() -> ModelDiscoveredContract:
    """A genuine command contract the interleave really does attach."""
    return ModelDiscoveredContract(
        name=_ROUTER,
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake") / _ROUTER / "contract.yaml",
        entry_point_name=_ROUTER,
        package_name="omnibase_infra",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(_ROUTER_COMMAND_TOPIC,),
            publish_topics=(),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="DelegationRouterHandler", module=_THIS_MODULE
                    ),
                    event_model=ModelHandlerRef(
                        name="ModelDelegationRequest", module=_THIS_MODULE
                    ),
                    operation=None,
                ),
            ),
        ),
    )


def _event_bus() -> MagicMock:
    bus = MagicMock(spec=ProtocolEventBusLike)
    bus.subscribe = AsyncMock(return_value=AsyncMock())
    return bus


async def _boot(
    manifest: ModelAutoWiringManifest,
) -> tuple[
    list[ModelContractAttachResult],
    list[ModelContractAttachExclusion],
]:
    """Drive the REAL deferred boot path the kernel takes.

    ``subscribe_immediately=False`` then ``subscribe_wired_contract_topics`` is
    the deployed sequence (subscribe after the dispatch engine is frozen). A
    fix proven on any other path would change nothing on any lane.
    """
    bus = _event_bus()
    engine = MessageDispatchEngine()
    with patch(_PATCH_IMPORT_HANDLER, side_effect=_import_by_name):
        report = await wire_from_manifest(
            manifest,
            engine,
            event_bus=bus,
            environment="local",
            subscribe_immediately=False,
        )
    attach_results: list[ModelContractAttachResult] = []
    exclusions: list[ModelContractAttachExclusion] = []
    await subscribe_wired_contract_topics(
        manifest,
        report,
        engine,
        bus,
        environment="local",
        attach_results_out=attach_results,
        exclusions_out=exclusions,
    )
    return attach_results, exclusions


# ===========================================================================
# A. The wedge, reproduced end to end through the real boot path
# ===========================================================================


@pytest.mark.asyncio
async def test_a_command_contract_the_interleave_skips_does_not_wedge_ready() -> None:
    """THE defect: 154 of 155 attached, zero failures, /ready 503 forever.

    RED on the parent two ways over: ``derive_required_contract_names`` puts
    the bridge in the required set, and ``subscribe_wired_contract_topics``
    has no ``exclusions_out`` to say it will never attempt it.
    """
    manifest = ModelAutoWiringManifest(
        contracts=(_bridge_contract(), _router_contract())
    )

    gate = ContractAttachReadinessGate(derive_required_contract_names(manifest))
    attach_results, exclusions = await _boot(manifest)

    # The interleave really did attach the router and really did skip the bridge.
    assert [r.contract_name for r in attach_results] == [_ROUTER]
    assert attach_results[0].status is EnumContractAttachStatus.ATTACHED

    gate.exclude(tuple(exclusions))
    gate.record(tuple(attach_results))
    status = gate.status()

    assert status.pending_contracts == (), (
        f"a required contract is pending with nothing left to report it: "
        f"{status.pending_contracts} — /ready is 503 for the life of the process"
    )
    assert status.ready is True
    assert status.phase is EnumContractAttachGatePhase.ATTACHED
    assert status.attached_contracts == (_ROUTER,)


@pytest.mark.asyncio
async def test_the_skipped_contract_is_named_and_reasoned_on_ready() -> None:
    """A 503 (and a 200) must be diagnosable: name the contract and the reason."""
    manifest = ModelAutoWiringManifest(
        contracts=(_bridge_contract(), _router_contract())
    )
    _attach_results, exclusions = await _boot(manifest)

    bridge = [e for e in exclusions if e.contract_name == _BRIDGE]
    assert len(bridge) == 1, (
        f"the interleave silently dropped the bridge; exclusions={exclusions}"
    )
    assert bridge[0].reason is EnumContractAttachExclusionReason.NOT_WIRED
    assert "No handler_routing declared in contract" in bridge[0].detail

    # Arm the gate with the bridge REQUIRED. This is the shape the required set
    # takes for the five filters that are not decidable from the contract alone
    # (zero dispatchers, no live dispatcher, raw projection with no applier):
    # the gate has no way to know, and the interleave's report is what unwedges
    # it. /ready must then name the contract AND the reason.
    gate = ContractAttachReadinessGate({_BRIDGE, _ROUTER})
    gate.exclude(tuple(exclusions))
    ready, detail = gate.probe()

    excluded = detail["excluded_contracts"]
    assert isinstance(excluded, tuple | list)
    assert [e["contract_name"] for e in excluded] == [_BRIDGE]
    assert [e["reason"] for e in excluded] == [
        EnumContractAttachExclusionReason.NOT_WIRED.value
    ]
    assert detail["required_contracts"] == [_ROUTER]
    assert ready is False, "the router has not reported yet — still 503"
    assert detail["pending_contracts"] == [_ROUTER]


def test_a_command_contract_with_no_handler_routing_is_not_required() -> None:
    """Decidable from the contract alone, so it never enters the required set.

    Same documented reason as ``plugin_managed``: ``_prepare_contract_wiring``
    marks it SKIPPED, so it can never report an attach result.
    """
    assert contract_subscribes_a_command_topic(_bridge_contract()) is False


def test_control_a_command_contract_with_handler_routing_is_still_required() -> None:
    """POSITIVE CONTROL: the exclusion above is narrow, not a blanket opt-out."""
    assert contract_subscribes_a_command_topic(_router_contract()) is True
    manifest = ModelAutoWiringManifest(
        contracts=(_bridge_contract(), _router_contract())
    )
    assert derive_required_contract_names(manifest) == frozenset({_ROUTER})


# ===========================================================================
# B. The live instance — the shipped contract that actually broke every lane
# ===========================================================================


def _repo_contract(node_dir: str) -> dict[str, Any]:
    path = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "omnibase_infra"
        / "nodes"
        / node_dir
        / "contract.yaml"
    )
    assert path.is_file(), f"contract not found: {path}"
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return parsed


def test_the_shipped_resolver_bridge_is_the_shape_this_ticket_describes() -> None:
    """Pin the live instance, so a contract change re-opens this test, not /ready."""
    contract = _repo_contract("node_contract_resolver_bridge")

    subscribe_topics = contract["event_bus"]["subscribe_topics"]
    assert _BRIDGE_COMMAND_TOPIC in subscribe_topics, (
        "the fixture above no longer models the shipped contract"
    )
    assert "handler_routing" not in contract, (
        "node_contract_resolver_bridge now declares handler_routing — it is no "
        "longer structurally skipped, and this ticket's premise needs re-reading"
    )


def test_control_a_shipped_contract_that_does_declare_handler_routing() -> None:
    """POSITIVE CONTROL: the probe above is not vacuously true of every contract."""
    contract = _repo_contract("node_merge_gate_effect")
    assert "handler_routing" in contract


# ===========================================================================
# C. Fail-closed: an exclusion can never open a contract that really failed
# ===========================================================================


def test_an_exclusion_cannot_unblock_a_contract_that_reported_not_ready() -> None:
    """Never a default-open: only a contract the interleave never TRIED is dropped."""
    gate = ContractAttachReadinessGate({"node_a"})
    gate.record(
        (
            ModelContractAttachResult(
                contract_name="node_a",
                status=EnumContractAttachStatus.NOT_READY,
                detail="topics never became ready",
            ),
        )
    )
    gate.exclude(
        (
            ModelContractAttachExclusion(
                contract_name="node_a",
                reason=EnumContractAttachExclusionReason.NOT_WIRED,
            ),
        )
    )

    status = gate.status()
    assert status.ready is False
    assert status.not_ready_contracts == ("node_a",)
    assert status.excluded_contracts == ()


def test_an_exclusion_cannot_unblock_a_contract_that_reported_failed() -> None:
    gate = ContractAttachReadinessGate({"node_a"})
    gate.record(
        (
            ModelContractAttachResult(
                contract_name="node_a",
                status=EnumContractAttachStatus.FAILED,
                detail="attach raised",
            ),
        )
    )
    gate.exclude(
        (
            ModelContractAttachExclusion(
                contract_name="node_a",
                reason=EnumContractAttachExclusionReason.NO_LIVE_DISPATCHER,
            ),
        )
    )

    assert gate.status().ready is False
    assert gate.status().failed_contracts == ("node_a",)


def test_control_the_gate_still_closes_on_a_contract_that_never_reports() -> None:
    """POSITIVE CONTROL: the fix must not turn the gate into a default-open.

    A required contract with no result AND no exclusion still pins /ready at
    503 — that is the whole OMN-17372 property and it is unchanged.
    """
    gate = ContractAttachReadinessGate({"node_a", "node_b"})
    gate.record(
        (
            ModelContractAttachResult(
                contract_name="node_a", status=EnumContractAttachStatus.ATTACHED
            ),
        )
    )

    status = gate.status()
    assert status.ready is False
    assert status.phase is EnumContractAttachGatePhase.WIRING_IN_PROGRESS
    assert status.pending_contracts == ("node_b",)


def test_exclusions_for_contracts_the_gate_never_required_are_inert() -> None:
    gate = ContractAttachReadinessGate({"node_a"})
    gate.exclude(
        (
            ModelContractAttachExclusion(
                contract_name="node_unrelated",
                reason=EnumContractAttachExclusionReason.PLUGIN_MANAGED,
            ),
        )
    )
    assert gate.status().pending_contracts == ("node_a",)
    assert gate.status().excluded_contracts == ()


# ===========================================================================
# D. The class, not the instance: every filter must report
# ===========================================================================


def _eligibility_loop() -> ast.For:
    """The ``for result in _prioritize_subscription_results(...)`` loop."""
    import omnibase_infra.runtime.auto_wiring.handler_wiring as hw

    tree = ast.parse(Path(hw.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AsyncFunctionDef)
            and node.name == "subscribe_wired_contract_topics"
        ):
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.For) and "_prioritize_subscription_results" in (
                    ast.dump(stmt.iter)
                ):
                    return stmt
    raise AssertionError("eligibility loop not found in handler_wiring")


def test_every_eligibility_filter_reports_its_exclusion() -> None:
    """The durable half: a filter added later cannot silently re-open the wedge.

    The gate and the interleave were two hand-maintained copies of the same
    eligibility rule, and the copies had diverged five ways out of six. This
    asserts the interleave REPORTS every branch that drops a contract, so the
    gate never has to guess a second time.
    """
    loop = _eligibility_loop()

    unreported: list[int] = []
    for branch in loop.body:
        if not isinstance(branch, ast.If):
            continue
        if not any(isinstance(s, ast.Continue) for s in branch.body):
            continue
        reports = [
            s
            for s in branch.body
            if isinstance(s, ast.Expr)
            and isinstance(s.value, ast.Call)
            and isinstance(s.value.func, ast.Name)
            and s.value.func.id == "_exclude"
        ]
        if not reports:
            unreported.append(branch.lineno)

    assert unreported == [], (
        "handler_wiring.subscribe_wired_contract_topics drops a contract at "
        f"line(s) {unreported} without calling _exclude(...). That contract can "
        "never produce a ModelContractAttachResult, so any readiness gate that "
        "requires it wedges /ready at 503 forever — the OMN-17372 defect, "
        "re-opened."
    )


def test_control_the_ast_probe_finds_the_real_filters() -> None:
    """POSITIVE CONTROL: an empty result above must mean 'all report', not 'none found'."""
    loop = _eligibility_loop()
    dropping = [
        b
        for b in loop.body
        if isinstance(b, ast.If) and any(isinstance(s, ast.Continue) for s in b.body)
    ]
    assert len(dropping) >= 5, (
        f"expected the documented eligibility filters, found {len(dropping)}"
    )


def test_the_silent_filter_now_logs() -> None:
    """The outcome-is-not-WIRED filter was the one of six that logged nothing.

    That silence is why a permanently-503 runtime produced no diagnostic at
    all for 29.5 minutes per boot.
    """
    loop = _eligibility_loop()
    first = loop.body[0]
    assert isinstance(first, ast.If)
    assert any(isinstance(s, ast.Continue) for s in first.body)
    assert any(
        isinstance(s, ast.Expr)
        and isinstance(s.value, ast.Call)
        and isinstance(s.value.func, ast.Attribute)
        and s.value.func.attr in {"info", "warning"}
        for s in first.body
    ), "the outcome-is-not-WIRED filter still drops a contract without a log line"


# ===========================================================================
# E. The kernel wires the exclusion path
# ===========================================================================


def _kernel_source() -> str:
    import omnibase_infra.runtime.service_kernel as kernel_module

    return Path(kernel_module.__file__).read_text(encoding="utf-8")


def test_kernel_passes_the_exclusion_sink_and_folds_it_in_before_recording() -> None:
    source = _kernel_source()

    assert "exclusions_out=_attach_exclusions," in source, (
        "the kernel does not ask the interleave for its exclusions, so the gate "
        "still requires contracts that can never report"
    )
    subscribe = source.index("await subscribe_wired_contract_topics(")
    exclude = source.index("_contract_attach_gate.exclude(tuple(_attach_exclusions))")
    record = source.index("_contract_attach_gate.record(tuple(_attach_results))")

    assert subscribe < exclude < record


def test_control_the_kernel_probe_is_not_vacuous() -> None:
    """POSITIVE CONTROL: the searches above raise on a source that lacks them."""
    with pytest.raises(ValueError):
        "".index("_contract_attach_gate.exclude(")


def test_the_interleave_signature_carries_the_exclusion_sink() -> None:
    params = inspect.signature(subscribe_wired_contract_topics).parameters
    assert "exclusions_out" in params
    assert params["exclusions_out"].default is None, (
        "the sink must be optional so every existing caller is unaffected"
    )
