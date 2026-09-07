# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Readiness requires the wired COMMAND topics, not just the registry (OMN-17372).

Before this module, ``required_for_readiness=True`` was passed at exactly three
sites in ``service_kernel`` — the contract-registry ``registered`` /
``deregistered`` / ``heartbeat`` control topics — and at NONE of the several
hundred auto-wired command/event topics. ``EventBusKafka.get_readiness_status``
computes ``is_ready = consumers_started and required_topics_ready`` over exactly
that marked set, so ``/ready`` returned 200 as soon as those three control
topics held partition assignments: the runtime advertised itself ready with zero
command topics subscribed. A gateway POST then returned 202 (a truthful
produce-ack — see ``omninode_infra docker/onex-api/gateway_publisher.py``) and
nothing consumed the command for the next ~18 minutes. That is the "hang".

``subscribe_wired_contract_topics`` already returns a
:class:`~omnibase_infra.event_bus.model_contract_attach_result.ModelContractAttachResult`
per contract (ATTACHED / NOT_READY / FAILED) and nobody read it for readiness.
This gate reads it.

Which contracts are REQUIRED is derived from the CONTRACTS, never from a hand
list: a wired contract is required when it subscribes at least one ONEX
**command** topic (``onex.cmd.<producer>.<event-name>.v<n>``). The delegation
command topics the gateway publishes into
(``onex.cmd.omnibase-infra.delegation-request.v1`` and
``onex.cmd.omnibase-infra.delegation-inference-request.v1``) are members of that
set by construction, not by enumeration, so a contract added later is covered
without editing this module.

A required contract must be one the boot interleave will ACTUALLY ATTEMPT.
The first revision of this module derived the required set from
``manifest.contracts`` while results were produced only for the ``eligible``
subset inside ``subscribe_wired_contract_topics`` — six filters narrower — and
this module hand-mirrored exactly one of those six. One contract caught by any
of the other five was required, never recorded, and pending forever:
``/ready`` answered 503 for the life of the process, on every lane, with no
timeout to expire and no log line to read. The two filters decidable from the
contract alone live in :func:`contract_subscribes_a_command_topic`; the rest are
reported by the interleave itself through
:meth:`ContractAttachReadinessGate.exclude`, so there is ONE authority on
eligibility and a filter added to the interleave later cannot silently re-open
the wedge.

Direction of travel is one-way: this gate can only make ``/ready`` LESS
permissive. It is ANDed into ``RuntimeHostProcess.readiness_check`` through the
existing OMN-14758 supplemental-probe seam; it never marks anything ready that
the event bus considers unready, and the three contract-registry requirements
are untouched.

Related Tickets:
    - OMN-17372: readiness truth must require the wired command topics.
    - OMN-13237: the per-contract provision -> confirm-ready -> attach interleave.
    - OMN-15215: the bounded NOT_READY reconciliation loop whose later attempts
      are folded in here, so a contract that converges late flips /ready to 200
      without a restart.
    - OMN-14758: ``register_readiness_probe``, the seam this rides.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from omnibase_infra.event_bus.enum_contract_attach_status import (
    EnumContractAttachStatus,
)
from omnibase_infra.event_bus.model_contract_attach_exclusion import (
    ModelContractAttachExclusion,
)
from omnibase_infra.runtime.enums.enum_contract_attach_gate_phase import (
    EnumContractAttachGatePhase,
)
from omnibase_infra.runtime.models.model_contract_attach_gate_status import (
    ModelContractAttachGateStatus,
)

if TYPE_CHECKING:
    from omnibase_infra.event_bus.model_contract_attach_result import (
        ModelContractAttachResult,
    )
    from omnibase_infra.runtime.auto_wiring.models import (
        ModelAutoWiringManifest,
        ModelDiscoveredContract,
    )

#: Segment index 1 of an ONEX topic carries the message kind.
_ONEX_TOPIC_PREFIX: str = "onex"
_ONEX_COMMAND_KIND: str = "cmd"
_ONEX_MIN_SEGMENTS: int = 5

#: Probe name registered on ``RuntimeHostProcess``; also the key this gate's
#: status appears under in ``/ready``'s ``supplemental_readiness`` block.
CONTRACT_ATTACH_PROBE_NAME: str = "contract_attach"


def is_command_topic(topic: str) -> bool:
    """Return True when *topic* is an ONEX command topic.

    Convention: ``onex.<kind>.<producer>.<event-name>.v<n>`` with ``<kind>``
    one of ``evt`` / ``cmd`` / ``intent``. Only ``cmd`` is a command.

    Deliberately stricter than
    ``handler_wiring._derive_message_category``: that function reads segment 1
    without checking the ``onex`` prefix or the 5-segment shape, because it
    only ever classifies topics that already passed topic-name validation. A
    readiness requirement is a fail-closed decision about which contracts may
    block traffic, so it demands the full canonical shape and treats anything
    else as not-a-command (i.e. not required) rather than guessing.
    """
    parts = topic.split(".")
    return (
        len(parts) >= _ONEX_MIN_SEGMENTS
        and parts[0] == _ONEX_TOPIC_PREFIX
        and parts[1] == _ONEX_COMMAND_KIND
    )


def contract_subscribes_a_command_topic(contract: ModelDiscoveredContract) -> bool:
    """Return True when *contract* declares at least one command subscribe topic.

    The two exclusions below are the ones decidable from the CONTRACT ALONE,
    before any wiring runs. Both drop a contract the boot interleave provably
    never attempts, so requiring it would wedge ``/ready`` at 503 forever. The
    exclusions that are only decidable from the wiring REPORT (zero registered
    dispatchers, a raw projection with no applier, an all-no-op dispatch) are
    NOT guessed at here — the interleave reports those itself through
    :meth:`ContractAttachReadinessGate.exclude`, which is the single authority.
    """
    event_bus = contract.event_bus
    if event_bus is None:
        return False
    if event_bus.plugin_managed:
        # OMN-10864: a plugin-managed contract owns its own subscription and is
        # never attached by the boot interleave, so it can never report a
        # result here. Requiring it would wedge /ready at 503 forever.
        return False
    if contract.handler_routing is None:
        # OMN-17372: a contract with no handler_routing is SKIPPED by
        # ``_prepare_contract_wiring`` ("No handler_routing declared in
        # contract"), so ``subscribe_wired_contract_topics`` drops it at the
        # outcome-is-not-WIRED filter and it can never report a result — the
        # identical failure mode as plugin_managed above.
        #
        # This was not hypothetical: ``node_contract_resolver_bridge``
        # subscribed ``onex.cmd.platform.contract-resolve-requested.v1`` and
        # declared no handler_routing (it is a transitional HTTP bridge served
        # by its own process, OMN-2756). It made every runtime from 0.38.21
        # onward serve /ready 503 permanently — 154 of 155 required contracts
        # ATTACHED, zero NOT_READY, zero FAILED, one contract pending forever.
        #
        # #3281 (OMN-18013) has since deleted that subscribe declaration, so
        # the measured instance is gone. The exclusion stays: it is decidable
        # from the contract alone, it is the identical failure mode as
        # plugin_managed, and the next contract to declare a command topic
        # without handler_routing must not wedge /ready a second time.
        return False
    return any(is_command_topic(topic) for topic in event_bus.subscribe_topics)


def derive_required_contract_names(
    manifest: ModelAutoWiringManifest,
) -> frozenset[str]:
    """Derive the readiness-required contract set FROM THE CONTRACTS.

    A contract is required when it subscribes at least one ONEX command topic.
    No hand list, no environment variable, no allowlist file: adding a contract
    that consumes a command automatically makes it a readiness requirement, and
    deleting one automatically stops it from blocking.

    Args:
        manifest: The auto-wiring manifest the kernel is about to subscribe.

    Returns:
        The required contract names. Empty when the runtime profile consumes no
        commands at all (e.g. a pure projection writer), in which case this gate
        imposes no constraint and readiness is decided entirely by the existing
        event-bus required-topic check.
    """
    return frozenset(
        contract.name
        for contract in manifest.contracts
        if contract_subscribes_a_command_topic(contract)
    )


class ContractAttachReadinessGate:
    """AND-term for ``/ready``: every required contract must be ATTACHED.

    The gate starts NOT ready with every required contract ``pending`` and only
    becomes ready once a result of status ATTACHED has been recorded for each
    one. Results arrive from the boot interleave and, later, from the bounded
    NOT_READY reconciliation loop; the newest result for a contract wins, so a
    contract that converges on retry flips the gate without a restart, and a
    contract that regresses flips it back.
    """

    __slots__ = ("_excluded", "_required", "_status_by_contract")

    def __init__(self, required_contract_names: Iterable[str]) -> None:
        self._required: frozenset[str] = frozenset(required_contract_names)
        self._status_by_contract: dict[str, EnumContractAttachStatus] = {}
        self._excluded: dict[str, ModelContractAttachExclusion] = {}

    @property
    def required_contract_names(self) -> frozenset[str]:
        """The contract names this gate requires to be ATTACHED."""
        return self._required

    def exclude(self, exclusions: Sequence[ModelContractAttachExclusion]) -> None:
        """Fold in the contracts the boot interleave says it will NEVER attempt.

        This is the fix for the class of defect OMN-17372 shipped: the required
        set is derived statically from the contracts, but results are produced
        only for the subset the interleave finds eligible, and that eligibility
        depends on facts (registered dispatchers, live dispatch targets, applier
        registration) that do not exist until wiring has run. Mirroring those
        filters here — guessing a second time, in a second place — is what
        broke; the interleave reporting its own exclusions is what fixes it.

        This is NOT a default-open and NOT a timeout. Exclusion applies only to
        a contract the interleave never TRIED: a contract with a recorded
        NOT_READY or FAILED result keeps blocking readiness no matter what
        arrives here, so nothing that actually failed to attach can be excluded
        into readiness.
        """
        for exclusion in exclusions:
            self._excluded[exclusion.contract_name] = exclusion

    def record(self, results: Sequence[ModelContractAttachResult]) -> None:
        """Fold attach results in. Results for non-required contracts are kept.

        Non-required results are still recorded so the gate's status body can
        stay truthful if the required set is ever widened at runtime; they
        never affect ``ready``.
        """
        for result in results:
            self._status_by_contract[result.contract_name] = result.status

    def status(self) -> ModelContractAttachGateStatus:
        """Return the typed gate status, naming the blocking contracts."""
        attached: list[str] = []
        not_ready: list[str] = []
        failed: list[str] = []
        pending: list[str] = []
        effective_required: list[str] = []
        excluded: list[ModelContractAttachExclusion] = []
        for name in sorted(self._required):
            status = self._status_by_contract.get(name)
            exclusion = self._excluded.get(name)
            # Fail-closed: an exclusion never overrides a contract the
            # interleave actually TRIED and could not attach. Only a contract
            # with no result, or one already ATTACHED, can be excluded.
            if exclusion is not None and status in (
                None,
                EnumContractAttachStatus.ATTACHED,
            ):
                excluded.append(exclusion)
                continue
            effective_required.append(name)
            if status is None:
                pending.append(name)
            elif status is EnumContractAttachStatus.ATTACHED:
                attached.append(name)
            elif status is EnumContractAttachStatus.NOT_READY:
                not_ready.append(name)
            else:
                failed.append(name)

        if pending:
            phase = EnumContractAttachGatePhase.WIRING_IN_PROGRESS
        elif not_ready or failed:
            phase = EnumContractAttachGatePhase.BLOCKED
        else:
            phase = EnumContractAttachGatePhase.ATTACHED

        return ModelContractAttachGateStatus(
            phase=phase,
            ready=phase is EnumContractAttachGatePhase.ATTACHED,
            required_contracts=tuple(effective_required),
            attached_contracts=tuple(attached),
            not_ready_contracts=tuple(not_ready),
            failed_contracts=tuple(failed),
            pending_contracts=tuple(pending),
            excluded_contracts=tuple(
                sorted(excluded, key=lambda item: item.contract_name)
            ),
        )

    def probe(self) -> tuple[bool, dict[str, object]]:
        """The ``register_readiness_probe`` callable (OMN-14758 shape).

        Returns ``(ready, detail)``. The detail is the dumped
        :class:`ModelContractAttachGateStatus`, so a 503 body names every
        NOT_READY / FAILED / still-pending contract rather than only asserting
        that the runtime is not ready.
        """
        gate_status = self.status()
        detail: dict[str, object] = gate_status.model_dump(mode="json")
        # ``ready`` is supplied by the probe tuple and re-added by the caller;
        # dropping it here keeps the merged supplemental block single-valued.
        detail.pop("ready", None)
        return gate_status.ready, detail


__all__: list[str] = [
    "CONTRACT_ATTACH_PROBE_NAME",
    "ContractAttachReadinessGate",
    "contract_subscribes_a_command_topic",
    "derive_required_contract_names",
    "is_command_topic",
]
