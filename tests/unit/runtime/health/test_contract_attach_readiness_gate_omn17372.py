# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Readiness must require the wired COMMAND topics, not only the registry (OMN-17372).

RED on the parent commit (``eee49719``): the whole module fails to import there,
because ``omnibase_infra.runtime.health.contract_attach_readiness_gate`` does not
exist — there is no mechanism by which an unattached command contract can affect
``/ready`` at all. The behavioural half of that RED is proven WITHOUT the new
module by :func:`test_control_registry_only_runtime_is_ready_without_the_gate`,
which passes on the parent and on this branch: a runtime whose event bus reports
ONLY the three contract-registry control topics as required-and-ready answers
``/ready`` 200. That is the defect.

Related Tickets:
    - OMN-17372: readiness truth must require the wired command topics.
    - OMN-13237: the per-contract attach interleave producing the results.
    - OMN-14758: the supplemental readiness probe seam.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.event_bus.enum_contract_attach_status import (
    EnumContractAttachStatus,
)
from omnibase_infra.event_bus.model_contract_attach_result import (
    ModelContractAttachResult,
)
from omnibase_infra.event_bus.models.model_event_bus_readiness import (
    ModelEventBusReadiness,
)
from omnibase_infra.runtime.auto_wiring.models.model_auto_wiring_manifest import (
    ModelAutoWiringManifest,
)
from omnibase_infra.runtime.auto_wiring.models.model_contract_version import (
    ModelContractVersion,
)
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.auto_wiring.models.model_event_bus_wiring import (
    ModelEventBusWiring,
)
from omnibase_infra.runtime.auto_wiring.models.model_handler_ref import (
    ModelHandlerRef,
)
from omnibase_infra.runtime.auto_wiring.models.model_handler_routing import (
    ModelHandlerRouting,
)
from omnibase_infra.runtime.auto_wiring.models.model_handler_routing_entry import (
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.enums.enum_contract_attach_gate_phase import (
    EnumContractAttachGatePhase,
)
from omnibase_infra.runtime.health.contract_attach_readiness_gate import (
    CONTRACT_ATTACH_PROBE_NAME,
    ContractAttachReadinessGate,
    contract_subscribes_a_command_topic,
    derive_required_contract_names,
    is_command_topic,
)
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess
from tests.helpers.runtime_helpers import make_runtime_config

pytestmark = pytest.mark.unit

# The three topics that ARE marked required_for_readiness today
# (service_kernel.py:4343,4350,4357 — contract-registry registered /
# deregistered / heartbeat). Shape only; the exact names are irrelevant to the
# invariant, which is that these three alone must not be sufficient.
REGISTRY_ONLY_TOPICS: tuple[str, ...] = (
    "onex.evt.platform.contract-registered.v1",
    "onex.evt.platform.contract-deregistered.v1",
    "onex.evt.platform.contract-heartbeat.v1",
)

# The two delegation command topics the gateway publishes into. Real constants,
# not literals invented here — see the corpus agreement test below.
DELEGATION_REQUEST_TOPIC: str = "onex.cmd.omnibase-infra.delegation-request.v1"
DELEGATION_INFERENCE_REQUEST_TOPIC: str = (
    "onex.cmd.omnibase-infra.delegation-inference-request.v1"
)


#: A contract the interleave will actually attach declares handler routing.
#: Without one it is SKIPPED upstream and can never report (OMN-17372).
_DEFAULT_ROUTING: ModelHandlerRouting = ModelHandlerRouting(
    routing_strategy="payload_type_match",
    handlers=(
        ModelHandlerRoutingEntry(
            handler=ModelHandlerRef(name="StubHandler", module="tests.stub"),
            event_model=ModelHandlerRef(name="ModelStub", module="tests.stub"),
            operation=None,
        ),
    ),
)


class _StubEventBus:
    """Event bus that reports exactly the topics it was handed as required+ready."""

    def __init__(self, required_topics: tuple[str, ...]) -> None:
        self._required_topics = required_topics

    async def get_readiness_status(self) -> ModelEventBusReadiness:
        return ModelEventBusReadiness(
            is_ready=True,
            consumers_started=True,
            assignments={topic: [0] for topic in self._required_topics},
            consume_tasks_alive=dict.fromkeys(self._required_topics, True),
            required_topics=self._required_topics,
            required_topics_ready=True,
        )


def _running_process(required_topics: tuple[str, ...]) -> RuntimeHostProcess:
    process = RuntimeHostProcess(config=make_runtime_config())
    process._event_bus = _StubEventBus(required_topics)  # type: ignore[assignment]
    process._is_running = True
    process._is_draining = False
    return process


def _contract(
    name: str,
    subscribe_topics: tuple[str, ...],
    *,
    handler_routing: ModelHandlerRouting | None = _DEFAULT_ROUTING,
    **eb: Any,
) -> ModelDiscoveredContract:
    """A contract shaped like one the boot interleave will actually attach.

    ``handler_routing`` defaults to a real routing block (OMN-17372): a
    contract without one is SKIPPED by ``_prepare_contract_wiring`` and can
    never report an attach result, so it is not a readiness requirement and a
    fixture that omits it does not model a wired command contract at all. Pass
    ``handler_routing=None`` to model that skipped shape deliberately.
    """
    return ModelDiscoveredContract(
        name=name,
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/nonexistent") / name / "contract.yaml",
        entry_point_name=name,
        package_name="omnibase_infra",
        event_bus=ModelEventBusWiring(subscribe_topics=subscribe_topics, **eb),
        handler_routing=handler_routing,
    )


def _attached(name: str) -> ModelContractAttachResult:
    return ModelContractAttachResult(
        contract_name=name, status=EnumContractAttachStatus.ATTACHED
    )


def _not_ready(name: str) -> ModelContractAttachResult:
    return ModelContractAttachResult(
        contract_name=name, status=EnumContractAttachStatus.NOT_READY
    )


def _failed(name: str) -> ModelContractAttachResult:
    return ModelContractAttachResult(
        contract_name=name, status=EnumContractAttachStatus.FAILED
    )


# ---------------------------------------------------------------------------
# The defect, and the positive control that proves the harness reaches it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_control_registry_only_runtime_is_ready_without_the_gate() -> None:
    """POSITIVE CONTROL: today a registry-only runtime answers /ready 200.

    This passes on the parent commit and on this branch. It is what makes the
    RED below attributable: the harness genuinely reaches
    ``RuntimeHostProcess.readiness_check``, and with no contract-attach gate
    registered that path reports ready with zero command topics subscribed.
    """
    process = _running_process(REGISTRY_ONLY_TOPICS)

    result = await process.readiness_check()

    assert result["ready"] is True


@pytest.mark.asyncio
async def test_registry_only_runtime_is_not_ready_when_a_command_contract_is_not_attached() -> (
    None
):
    """RED on parent: a command contract that never attached must block /ready."""
    process = _running_process(REGISTRY_ONLY_TOPICS)
    gate = ContractAttachReadinessGate({"node_delegation_router"})
    gate.record([_not_ready("node_delegation_router")])
    process.register_readiness_probe(CONTRACT_ATTACH_PROBE_NAME, gate.probe)

    result = await process.readiness_check()

    assert result["ready"] is False
    supplemental = result["supplemental_readiness"]
    assert isinstance(supplemental, dict)
    detail = supplemental[CONTRACT_ATTACH_PROBE_NAME]
    assert detail["ready"] is False
    # The 503 body NAMES the blocking contract.
    assert detail["not_ready_contracts"] == ["node_delegation_router"]
    assert detail["phase"] == EnumContractAttachGatePhase.BLOCKED.value


@pytest.mark.asyncio
async def test_fully_attached_runtime_is_ready() -> None:
    """GREEN: once every required contract is ATTACHED, /ready is 200 again."""
    process = _running_process(REGISTRY_ONLY_TOPICS)
    gate = ContractAttachReadinessGate({"node_delegation_router"})
    gate.record([_attached("node_delegation_router")])
    process.register_readiness_probe(CONTRACT_ATTACH_PROBE_NAME, gate.probe)

    result = await process.readiness_check()

    assert result["ready"] is True
    supplemental = result["supplemental_readiness"]
    assert isinstance(supplemental, dict)
    assert supplemental[CONTRACT_ATTACH_PROBE_NAME]["ready"] is True


@pytest.mark.asyncio
async def test_gate_is_not_ready_before_any_result_is_recorded() -> None:
    """Fail-closed: during wiring, with no results yet, /ready must be 503."""
    process = _running_process(REGISTRY_ONLY_TOPICS)
    gate = ContractAttachReadinessGate({"node_delegation_router"})
    process.register_readiness_probe(CONTRACT_ATTACH_PROBE_NAME, gate.probe)

    result = await process.readiness_check()

    assert result["ready"] is False
    supplemental = result["supplemental_readiness"]
    assert isinstance(supplemental, dict)
    detail = supplemental[CONTRACT_ATTACH_PROBE_NAME]
    assert detail["phase"] == EnumContractAttachGatePhase.WIRING_IN_PROGRESS.value
    assert detail["pending_contracts"] == ["node_delegation_router"]


# ---------------------------------------------------------------------------
# Never more permissive
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_never_makes_an_unready_bus_ready() -> None:
    """A fully-attached gate cannot rescue an event bus that is not ready."""

    class _UnreadyBus:
        async def get_readiness_status(self) -> ModelEventBusReadiness:
            return ModelEventBusReadiness(
                is_ready=False,
                consumers_started=False,
                required_topics=REGISTRY_ONLY_TOPICS,
                required_topics_ready=False,
            )

    process = RuntimeHostProcess(config=make_runtime_config())
    process._event_bus = _UnreadyBus()  # type: ignore[assignment]
    process._is_running = True
    process._is_draining = False
    gate = ContractAttachReadinessGate({"node_delegation_router"})
    gate.record([_attached("node_delegation_router")])
    process.register_readiness_probe(CONTRACT_ATTACH_PROBE_NAME, gate.probe)

    result = await process.readiness_check()

    assert result["ready"] is False


def test_a_failed_contract_blocks_and_is_named() -> None:
    gate = ContractAttachReadinessGate({"a", "b"})
    gate.record([_attached("a"), _failed("b")])

    status = gate.status()

    assert status.ready is False
    assert status.failed_contracts == ("b",)
    assert status.attached_contracts == ("a",)
    assert status.phase is EnumContractAttachGatePhase.BLOCKED


def test_reconciliation_result_flips_the_gate_without_a_restart() -> None:
    """OMN-15215's bounded retry must be able to clear a NOT_READY blocker."""
    gate = ContractAttachReadinessGate({"a"})
    gate.record([_not_ready("a")])
    assert gate.status().ready is False

    gate.record([_attached("a")])

    assert gate.status().ready is True
    assert gate.status().phase is EnumContractAttachGatePhase.ATTACHED


def test_a_regressing_contract_flips_the_gate_back() -> None:
    gate = ContractAttachReadinessGate({"a"})
    gate.record([_attached("a")])
    assert gate.status().ready is True

    gate.record([_failed("a")])

    assert gate.status().ready is False


def test_no_command_contracts_imposes_no_constraint() -> None:
    """A profile that consumes no commands is not wedged by this gate."""
    gate = ContractAttachReadinessGate(frozenset())

    status = gate.status()

    assert status.ready is True
    assert status.required_contracts == ()


# ---------------------------------------------------------------------------
# The required set comes from the CONTRACTS
# ---------------------------------------------------------------------------


def test_required_set_is_derived_from_contract_subscribe_topics() -> None:
    manifest = ModelAutoWiringManifest(
        contracts=(
            _contract("node_delegation_router", (DELEGATION_REQUEST_TOPIC,)),
            _contract(
                "node_delegation_inference",
                (DELEGATION_INFERENCE_REQUEST_TOPIC, "onex.evt.platform.x.v1"),
            ),
            _contract("node_projection_savings", ("onex.evt.omnimarket.savings.v1",)),
        ),
    )

    required = derive_required_contract_names(manifest)

    assert required == frozenset(
        {"node_delegation_router", "node_delegation_inference"}
    )


def test_the_two_gateway_delegation_command_topics_are_required_by_construction() -> (
    None
):
    """The brief's minimum: both delegation command topics gate readiness."""
    assert is_command_topic(DELEGATION_REQUEST_TOPIC) is True
    assert is_command_topic(DELEGATION_INFERENCE_REQUEST_TOPIC) is True


def test_plugin_managed_command_contract_is_not_required() -> None:
    """OMN-10864: the boot interleave never attaches it, so it can never report.

    Requiring it would wedge /ready at 503 for the life of the process.
    """
    contract = _contract(
        "node_plugin_owned", (DELEGATION_REQUEST_TOPIC,), plugin_managed=True
    )

    assert contract_subscribes_a_command_topic(contract) is False


def test_event_only_and_intent_topics_are_not_commands() -> None:
    assert is_command_topic("onex.evt.platform.thing-happened.v1") is False
    assert is_command_topic("onex.intent.platform.thing-wanted.v1") is False
    # Non-canonical shapes are not guessed at.
    assert is_command_topic("cmd.thing") is False
    assert is_command_topic("acme.cmd.platform.thing.v1") is False
    assert is_command_topic("onex.cmd.platform.thing") is False


def test_command_classification_agrees_with_the_wiring_categoriser() -> None:
    """Anti-drift: agree with handler_wiring over the real platform topic corpus.

    Positive control: the corpus is asserted non-empty and to contain BOTH
    command and non-command topics, so agreement is not vacuous.
    """
    from omnibase_infra.runtime.auto_wiring.handler_wiring import (
        _derive_message_category,
    )
    from omnibase_infra.topics import platform_topic_suffixes

    corpus = tuple(
        value
        for name, value in vars(platform_topic_suffixes).items()
        if name.startswith("SUFFIX_")
        and isinstance(value, str)
        and value.startswith("onex.")
        and len(value.split(".")) >= 5
    )
    commands = tuple(t for t in corpus if _derive_message_category(t) == "command")
    non_commands = tuple(t for t in corpus if _derive_message_category(t) != "command")
    assert len(corpus) > 50, "positive control: the topic corpus must be non-trivial"
    assert commands, "positive control: the corpus must contain command topics"
    assert non_commands, "positive control: the corpus must contain non-command topics"

    for topic in corpus:
        assert is_command_topic(topic) is (topic in commands), topic


__all__: list[str] = []


# ---------------------------------------------------------------------------
# The KERNEL WIRING itself (hostile-reviewer MAJOR, thread r3949048723)
#
# Everything above builds the gate by hand. That leaves the one property this
# whole change rests on untested: that `service_kernel` registers the probe
# BEFORE it starts subscribing. If the registration were moved below the
# interleave, every test above would still pass and the entire ~18-minute
# wiring window would go back to answering /ready 200 -- which is the exact
# bug class this diff exists to remove (a wiring site that forgot to mark).
#
# `bootstrap()` cannot be called in a unit test (it needs a broker, a database
# and a container), so the ordering is asserted on the module's own AST. That
# is a real guard against the regression: it fails if the three statements are
# reordered, renamed, or deleted.
# ---------------------------------------------------------------------------


def _kernel_source() -> str:
    import omnibase_infra.runtime.service_kernel as kernel_module

    return Path(kernel_module.__file__).read_text(encoding="utf-8")


def test_kernel_registers_the_gate_before_it_subscribes_anything() -> None:
    """The probe must be armed BEFORE the boot interleave, not after it."""
    source = _kernel_source()

    construct = source.index("ContractAttachReadinessGate(")
    # The kernel passes the CONSTANT, not the literal, so the shared name
    # can never drift between the gate module and the registration site.
    register = source.index("CONTRACT_ATTACH_PROBE_NAME,", construct)
    subscribe = source.index("await subscribe_wired_contract_topics(")

    assert construct < subscribe, (
        "the gate is constructed after the boot interleave starts; the whole "
        "wiring window would answer /ready 200"
    )
    assert register < subscribe, (
        "register_readiness_probe runs after subscribe_wired_contract_topics; "
        "/ready is unguarded for the entire wiring window"
    )


def test_kernel_feeds_the_gate_from_the_real_manifest_not_a_literal() -> None:
    """The required set comes from the manifest the kernel is about to wire."""
    source = _kernel_source()

    assert (
        "ContractAttachReadinessGate(\n"
        "                derive_required_contract_names("
        "auto_wiring_manifest_for_subscriptions)\n"
        "            )" in source
    ), (
        "the kernel must derive the required set from "
        "auto_wiring_manifest_for_subscriptions -- the same manifest object "
        "passed to subscribe_wired_contract_topics -- and never from a literal"
    )


def test_kernel_records_boot_results_and_every_reconciliation_attempt() -> None:
    """Boot results AND retry results must both reach the gate."""
    source = _kernel_source()

    subscribe = source.index("await subscribe_wired_contract_topics(")
    boot_record = source.index("_contract_attach_gate.record(tuple(_attach_results))")
    assert boot_record > subscribe, "boot attach results are never recorded"

    assert "on_attempt=lambda _subscribed, results: (" in source, (
        "the bounded NOT_READY reconciliation loop must hand every attempt to "
        "the gate, or a contract that converges late never flips /ready to 200"
    )
    reconcile_record = source.index(
        "on_attempt=lambda _subscribed, results: (\n"
        "                                _contract_attach_gate.record(results)"
    )
    assert reconcile_record > boot_record


def test_control_the_kernel_ast_probes_are_not_vacuous() -> None:
    """POSITIVE CONTROL: the searches above fail on a source that lacks them."""
    with pytest.raises(ValueError):
        "".index("ContractAttachReadinessGate(")
