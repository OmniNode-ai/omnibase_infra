# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Resolve — and prove — where a delegation's orchestrator will run.

OMN-17295 / OMN-17304. ``--bus`` chooses the TRANSPORT. It never chose the
executor, but it was read as though it did, so a "dev-lane probe" issued as
``onex delegate ... --bus kafka`` ran the orchestrator in-process out of the
caller's own venv and its result was reported as a statement about ``.201``.
The instrument produced a confident answer about a machine it never touched.

Three rules make that impossible here:

1. **Locus is resolved, never assumed.** It follows the transport by default —
   a shared bus means somebody else consumes, an in-memory bus means nobody
   can — and an explicit ``--locus`` says so in the record.
2. **A dispatched run fails closed.** Before publishing, the exact command
   topic (read from the contract, not from a constant here) must have a live
   ``STABLE`` consumer group owned by the contract's node bound to it
   (OMN-20235). No consumer, or an unanswerable
   broker, refuses the run. It never silently degrades to in-process, because
   a degraded lane probe is worse than no lane probe: it still prints a
   receipt.
3. **An in-process run never shares a broker with a deployed orchestrator**
   (OMN-20236).

.. versionadded:: OMN-17304
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import time
from pathlib import Path

import yaml

from omnibase_infra.backends.backend_probe import (
    ConsumerGroupDescribeDeniedError,
    ConsumerGroupLivenessUnknownError,
    live_chain_consumer_groups,
    live_consumer_groups,
)
from omnibase_infra.backends.model_consumer_group_owner import ModelConsumerGroupOwner
from omnibase_infra.cli.model_delegate_locus_decision import ModelDelegateLocusDecision
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus
from omnibase_infra.models.delegation.model_delegate_downstream_chain import (
    ModelDelegateDownstreamChain,
)

__all__ = [
    "DOWNSTREAM_CHAIN_STAGE",
    "DelegateDownstreamChainRefusedError",
    "REBIND_WINDOW_FAILURE_CLASS",
    "DelegateLocusAclRefusedError",
    "DelegateLocusRefusedError",
    "contract_command_topic",
    "contract_consumer_owner",
    "contract_downstream_chain",
    "contract_terminal_topic",
    "orchestrator_distribution",
    "resolve_delegate_locus",
]

logger = logging.getLogger(__name__)

# The distribution that ships the delegate orchestrator contract and handler.
_ORCHESTRATOR_DISTRIBUTION = "omnimarket"

# Seconds allowed for the consumer-group liveness question. Generous relative
# to the 2s health probe: this answer gates the whole run, and a timeout here
# refuses rather than degrades, so being impatient turns a slow broker into a
# false refusal.
_LIVENESS_TIMEOUT_SECONDS = 5.0

# OMN-18843. The named failure class this wait exists for: every
# runtime-affecting merge to ``dev`` force-recreates the containers that bind
# the delegate command topic (operating rule 24(a)), so for a bounded window
# after each rebuild the broker answers and NOTHING is bound. A client that
# asks during that window is not talking to a missing lane; it is talking to a
# lane that is rebinding. The class name is in the refusal text and the wait
# log so a search for either lands on the runbook entry.
REBIND_WINDOW_FAILURE_CLASS = "delegate-consumer-rebind-window"

# OMN-20209. The stage a refusal names when the first hop is live and the
# chain it hands the request to is not; see ``contract_downstream_chain``.
DOWNSTREAM_CHAIN_STAGE = "downstream-delegation-chain"

# How long a dispatched run waits for a consumer group to bind before it
# refuses. Measured on the .201 dev lane after omnibase_infra#3939 removed the
# health-gated hold: container created 2026-09-23T17:58:02.183Z, delegate-skill
# orchestrator group joined 17:59:40.214Z, 98.0 s. The pre-#3939 windows ran
# 4 m 20 s to 7 m 52 s. 180 s covers the measured post-fix window with room for
# a slow host and is still a hard bound: past it the run refuses exactly as it
# did before, because a lane that is still unbound after three minutes is down,
# not rebinding.
_REBIND_WAIT_SECONDS = 180.0

# Seconds between re-probes inside that window. The liveness probe itself is
# bounded by ``_LIVENESS_TIMEOUT_SECONDS``, so one re-probe costs at most this
# plus that.
_REBIND_POLL_SECONDS = 5.0

# Indirection so the wait can be driven by a fake clock in unit tests.
_sleep = time.sleep
_monotonic = time.monotonic


class DelegateLocusRefusedError(RuntimeError):
    """The requested locus cannot be honoured, so the run does not start.

    Always a refusal, never a downgrade. The whole defect being fixed is a
    probe that answers about the wrong executor; falling back to in-process
    when the lane cannot be reached would reproduce it exactly, with the
    added insult of having been asked not to.
    """


class DelegateDownstreamChainRefusedError(DelegateLocusRefusedError):
    """The first hop is live but the chain it hands the request to is not.

    The deployed orchestrator would accept the command and then wait its whole
    execution budget, reported to the caller as a hang. The class name is what
    the transport refusal records as ``transport_error_type``.
    """


class DelegateLocusAclRefusedError(DelegateLocusRefusedError):
    """The refusal is a missing broker grant, not a missing lane (OMN-19914).

    Raised when every consumer group bound to the command topic was hidden
    from this principal by the broker. Its class name is what the written
    transport refusal records as ``transport_error_type``, so a reader of the
    receipt sees "grant missing" without parsing prose, and its message names
    the group(s) and the exact DESCRIBE grant that would let the probe answer.
    """

    def __init__(self, message: str, *, group_ids: tuple[str, ...]) -> None:
        super().__init__(message)
        self.group_ids = group_ids


def _load_contract(text: str) -> object:
    """Parse contract YAML: the one place this module reads a contract.

    Every reader here parses installed package contracts, so they share one
    loader rather than each carrying its own exemption.
    """
    return yaml.safe_load(text)  # yaml-safe-load-ok: contract is trusted package data


def contract_terminal_topic(contract_path: Path) -> str:
    """Read the topic the delegation terminal arrives on, or ``""`` (OMN-17516).

    ``terminal_event`` — the key ``RuntimeLocal`` itself subscribes the terminal
    listener to.

    Unlike :func:`contract_command_topic` this **never raises**. Its only caller
    is the timeout-refusal builder, which runs while reporting a failure: a
    reader that raised there would replace a legible refusal with a traceback
    about the refusal, which is strictly worse than the empty stdout it exists
    to remove. An unreadable or silent contract yields ``""`` and the refusal
    says so in that field, which is itself a finding worth having.
    """
    try:
        raw = _load_contract(contract_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return ""
    if not isinstance(raw, dict):
        return ""
    terminal = raw.get("terminal_event")
    return terminal if isinstance(terminal, str) else ""


def contract_command_topic(contract_path: Path) -> str:
    """Read the topic the delegate command is published to, from the contract.

    ``event_bus.subscribe_topics[0]`` — the same element ``RuntimeLocal``
    publishes to. Read here rather than named as a constant because the two
    must not be able to disagree: ``resolve_default_bus`` has been asserting
    consumer liveness against ``SUFFIX_DELEGATION_REQUEST``
    (``onex.cmd.omnibase-infra.delegation-request.v1``) while this path
    publishes to ``onex.cmd.omnimarket.delegate-skill.v1``, so its
    AUTHORITATIVE grade was decided by consumers of a topic this command never
    reaches. Verified live 2026-08-31: both topics carry ``STABLE`` groups, and
    they are different groups on different topics.

    Raises:
        DelegateLocusRefusedError: the contract declares no command topic.
    """
    try:
        raw = _load_contract(contract_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DelegateLocusRefusedError(
            f"cannot read the delegate contract at {contract_path}: {exc}"
        ) from exc
    event_bus = raw.get("event_bus") if isinstance(raw, dict) else None
    topics = event_bus.get("subscribe_topics") if isinstance(event_bus, dict) else None
    if not isinstance(topics, list) or not topics or not isinstance(topics[0], str):
        raise DelegateLocusRefusedError(
            f"the delegate contract at {contract_path} declares no "
            "event_bus.subscribe_topics, so there is no command topic to "
            "publish to or to check for consumers"
        )
    return topics[0]


def contract_consumer_owner(contract_path: Path) -> ModelConsumerGroupOwner:
    """Read the node identity whose groups a deployed dispatch needs (OMN-20235).

    The distribution shipping the contract is the runtime's package_name.

    Raises:
        DelegateLocusRefusedError: the contract is unreadable or has no node name.
    """
    try:
        raw = _load_contract(contract_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError, UnicodeError) as exc:
        raise DelegateLocusRefusedError(
            f"cannot read the delegate contract at {contract_path}: {exc}"
        ) from exc
    name = raw.get("name") if isinstance(raw, dict) else None
    if not isinstance(name, str) or not name:
        raise DelegateLocusRefusedError(
            f"the delegate contract at {contract_path} declares no string name, "
            "so its consumer group owner cannot be checked"
        )
    return ModelConsumerGroupOwner(service=_ORCHESTRATOR_DISTRIBUTION, node=name)


def contract_downstream_chain(
    contract_path: Path,
) -> ModelDelegateDownstreamChain | None:
    """Resolve the chain the deployed orchestrator hands work to (OMN-20209).

    The delegate-skill orchestrator is the first hop only: it republishes every
    command to the topic its contract declares under
    ``delegation_runtime_dispatch.topics.command`` and waits its whole
    execution budget for that chain's terminal. Measured on the .201 dev lane
    on 2026-10-02 between 06:52Z and 08:08Z: the first hop was ``Stable`` and
    picked every command up within 168 ms, the runtime hosting the downstream
    consumer was crash-looping, and each run came back ``status=timeout``
    with ``attempts=[]`` after 300 s, by which time every caller had killed
    the CLI. A probe of the first hop alone admitted all of them.

    The consumer is found among the sibling node contracts, never by name: the
    one contract that subscribes to the downstream command topic and publishes
    its completion topic. Its live group id cannot be derived from that name
    (a runtime plugin names it), so the probe identifies it by the contract's
    whole subscription footprint instead.

    Returns:
        The chain, or ``None`` when the contract declares no downstream
        dispatch, in which case only the first hop can be proven.

    Raises:
        DelegateLocusRefusedError: the declaration is malformed, or zero or
            several installed contracts match it.
    """
    try:
        raw = _load_contract(contract_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError, UnicodeError) as exc:
        raise DelegateLocusRefusedError(
            f"cannot read the delegate contract at {contract_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict) or "delegation_runtime_dispatch" not in raw:
        return None
    dispatch = raw["delegation_runtime_dispatch"]
    topics = dispatch.get("topics") if isinstance(dispatch, dict) else None
    resolved_topics: dict[str, str] = {}
    for field in ("command", "completed"):
        value = topics.get(field) if isinstance(topics, dict) else None
        if not isinstance(value, str) or not value:
            raise DelegateLocusRefusedError(
                f"the delegate contract at {contract_path} declares no non-empty "
                f"string delegation_runtime_dispatch.topics.{field}"
            )
        resolved_topics[field] = value
    command = resolved_topics["command"]
    completed = resolved_topics["completed"]
    matches: list[ModelDelegateDownstreamChain] = []
    for path in sorted(contract_path.parent.parent.glob("*/contract.yaml")):
        if path == contract_path:
            continue
        try:
            text = path.read_text(encoding="utf-8")
            if command not in text:
                continue
            sibling = _load_contract(text)
        except (OSError, yaml.YAMLError, UnicodeError):
            continue
        if not isinstance(sibling, dict):
            continue
        event_bus = sibling.get("event_bus")
        if not isinstance(event_bus, dict):
            continue
        subscribed = event_bus.get("subscribe_topics")
        published = event_bus.get("publish_topics")
        if (
            not isinstance(subscribed, list)
            or command not in subscribed
            or not isinstance(published, list)
            or completed not in published
        ):
            continue
        name = sibling.get("name")
        if not isinstance(name, str) or not name:
            raise DelegateLocusRefusedError(
                f"the downstream consumer contract at {path} declares no string name"
            )
        if any(not isinstance(topic, str) or not topic for topic in subscribed):
            raise DelegateLocusRefusedError(
                f"the downstream consumer contract {name} at {path} declares "
                "invalid event_bus.subscribe_topics"
            )
        matches.append(
            ModelDelegateDownstreamChain(
                command_topic=command,
                completed_topic=completed,
                consumer_contract=name,
                consumer_contract_path=str(path),
                subscribe_topics=tuple(subscribed),
            )
        )
    if len(matches) != 1:
        names = (
            ": " + ", ".join(match.consumer_contract for match in matches)
            if len(matches) > 1
            else ""
        )
        raise DelegateLocusRefusedError(
            f"downstream command topic '{command}' has {len(matches)} matching "
            f"consumer contracts{names}; a dispatched run would wait on a chain "
            "no single installed contract declares"
        )
    return matches[0]


def orchestrator_distribution() -> str:
    """Name the installed distribution the orchestrator contract came from.

    ``'<name> <version> (<location>)'``. The location matters as much as the
    version: the OMN-17295 probe's receipt named a contract under
    ``omnibase_infra/.venv/.../omnimarket/`` while being read as a statement
    about a container on another host, and nothing in the output said so.
    """
    try:
        version = importlib.metadata.version(_ORCHESTRATOR_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        version = "(not installed)"
    spec = importlib.util.find_spec(_ORCHESTRATOR_DISTRIBUTION)
    search_locations = spec.submodule_search_locations if spec is not None else None
    locations = list(search_locations) if search_locations is not None else []
    location = locations[0] if locations else "(unresolved)"
    return f"{_ORCHESTRATOR_DISTRIBUTION} {version} ({location})"


def resolve_delegate_locus(
    *,
    requested: EnumDelegateLocus,
    bus: str,
    kafka_bootstrap: str | None,
    contract_path: Path,
    shared_bus_value: str,
) -> ModelDelegateLocusDecision:
    """Decide the locus and, for a dispatched run, prove it is viable.

    Args:
        requested: ``--locus``. ``AUTO`` follows the transport.
        bus: The already-resolved transport (``inmemory`` / ``kafka``).
        kafka_bootstrap: Explicit broker override, if one was given.
        contract_path: The delegate orchestrator contract being dispatched.
        shared_bus_value: The bus value that denotes a shared broker, passed
            in rather than hardcoded so this module never becomes a second
            place that knows the transport vocabulary.

    Raises:
        DelegateLocusRefusedError: an incoherent request (a deployed lane over
            an in-process bus), or a dispatched run with no live consumer on
            the command topic.
    """
    distribution = orchestrator_distribution()
    on_shared_bus = bus == shared_bus_value

    if requested is EnumDelegateLocus.AUTO:
        locus = (
            EnumDelegateLocus.DEPLOYED_LANE
            if on_shared_bus
            else EnumDelegateLocus.IN_PROCESS
        )
        resolved_from = (
            f"resolved from transport={bus}: a shared bus has other consumers, "
            "an in-process bus has none"
        )
    else:
        locus = requested
        resolved_from = (
            f"explicit --locus {requested.value} OVERRIDES the transport-derived "
            f"locus (transport={bus})"
        )

    if locus is EnumDelegateLocus.DEPLOYED_LANE and not on_shared_bus:
        raise DelegateLocusRefusedError(
            f"--locus deployed-lane is incoherent with transport {bus!r}: an "
            "in-process event bus exists only inside this process, so no "
            "deployed runtime can consume the command. Use --bus "
            f"{shared_bus_value} to reach a shared broker, or --locus "
            "in-process to run it here."
        )

    if locus is EnumDelegateLocus.IN_PROCESS:
        # Best-effort for an in-process run: the topic is informational only
        # on an in-memory bus, where the runtime publishes and consumes it
        # inside this process. A contract without a topic is legitimate for a
        # single-handler workflow. A dispatched run is the opposite — the
        # topic IS the interface to the other runtime — so it refuses below.
        try:
            command_topic = contract_command_topic(contract_path)
        except DelegateLocusRefusedError as exc:
            logger.debug(
                "onex delegate: no command topic on %s (%s) — informational "
                "only for an in-process run",
                contract_path,
                exc,
            )
            command_topic = ""
        if on_shared_bus and command_topic:
            raise DelegateLocusRefusedError(
                f"--locus in-process on a shared bus causes double dispatch: "
                f"the start command lands on the shared topic '{command_topic}', "
                "which the deployed orchestrator consumes now or from its "
                "committed offset when it next starts, so the run executes "
                "twice and the second copy overwrites the first in the "
                "projection (OMN-20236). Use --locus deployed-lane for one "
                "dispatch on the lane, or --bus inmemory for a true in-process "
                "run that publishes nothing to a shared broker."
            )
        logger.info(
            "onex delegate: execution locus IN-PROCESS (%s) — the accept/climb "
            "decision is made by %s in THIS process; this run is not evidence "
            "about any deployed lane",
            resolved_from,
            distribution,
        )
        return ModelDelegateLocusDecision(
            locus=locus,
            resolved_from=resolved_from,
            orchestrator_contract=str(contract_path),
            orchestrator_distribution=distribution,
            command_topic=command_topic,
            broker="",
            lane_consumer_groups=(),
        )

    command_topic = contract_command_topic(contract_path)
    owner = contract_consumer_owner(contract_path)
    downstream = contract_downstream_chain(contract_path)
    if downstream is None:
        logger.info(
            "onex delegate: contract %s declares no downstream delegation chain; "
            "only the first hop is proven",
            contract_path,
        )
    if kafka_bootstrap is None:
        # OMN-16871. This used to mean "let the client read
        # KAFKA_BOOTSTRAP_SERVERS", so the probe asked whichever lane the
        # shell happened to name while the publish went somewhere else — a
        # probe of a lane the run never reaches, which is the OMN-17295
        # instrument defect in a second place. It is now a refusal, which also
        # makes the recorded ``broker`` a real address rather than a
        # placeholder string standing in for one.
        raise DelegateLocusRefusedError(
            f"cannot probe a deployed orchestrator on '{command_topic}': no "
            "broker address was resolved for this run. Select the lane with "
            "--lane <lane id> so the probe and the publish name the same "
            "broker (OMN-16871)."
        )
    groups, downstream_groups, bind_wait_seconds = _assert_dispatch_viable(
        command_topic=command_topic,
        kafka_bootstrap=kafka_bootstrap,
        owner=owner,
        downstream=downstream,
    )
    broker = kafka_bootstrap
    logger.info(
        "onex delegate: execution locus DEPLOYED-LANE (%s) — publishing to "
        "'%s' on %s; %d live consumer group(s) bound: %s. The accept/climb "
        "decision is made THERE, not by %s, which supplies only the wire "
        "description; downstream consumer groups: %s",
        resolved_from,
        command_topic,
        broker,
        len(groups),
        ", ".join(groups),
        distribution,
        ", ".join(downstream_groups),
    )
    return ModelDelegateLocusDecision(
        locus=locus,
        resolved_from=resolved_from,
        orchestrator_contract=str(contract_path),
        orchestrator_distribution=distribution,
        command_topic=command_topic,
        broker=broker,
        lane_consumer_groups=groups,
        downstream_command_topic=downstream.command_topic if downstream else "",
        downstream_consumer_groups=downstream_groups,
        consumer_bind_wait_seconds=bind_wait_seconds,
    )


def _assert_dispatch_viable(
    *,
    command_topic: str,
    kafka_bootstrap: str,
    owner: ModelConsumerGroupOwner,
    downstream: ModelDelegateDownstreamChain | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...], float]:
    """Refuse unless the first hop and any declared downstream chain are live.

    Returns first-hop groups, downstream groups, and seconds spent waiting.

    Three outcomes, two of which refuse:

    * live groups → proceed, and they go in the record.
    * broker answered, nothing bound → wait up to ``_REBIND_WAIT_SECONDS``,
      re-probing every ``_REBIND_POLL_SECONDS``, then refuse. A lane rebuild
      leaves the topic unbound for a bounded window (OMN-18843, the
      ``delegate-consumer-rebind-window`` class), and refusing on the first
      empty answer turned every rebuild into refused delegations. Past the
      bound it refuses exactly as before: publishing would succeed and the run
      would then sit until its timeout, reported as "the lane was slow" rather
      than "there was no lane".
    * broker could not be asked → refuse at once, never retried. UNKNOWN is
      not permission; that conflation is what let a probe report on an
      executor it never reached.

    The address is required (OMN-16871): the caller resolves it from the
    selected lane, so this function can never probe a broker the publish will
    not use.

    Args:
        command_topic: Exact topic this dispatch publishes to.
        kafka_bootstrap: Broker address resolved for this run.
        owner: Contract service and node whose live groups are required.
        downstream: Optional declared chain sharing the same rebind wait bound.
    """
    started = _monotonic()
    probes = 0
    while True:
        probes += 1
        try:
            groups = live_consumer_groups(
                topic=command_topic,
                bootstrap_servers=kafka_bootstrap,
                timeout=_LIVENESS_TIMEOUT_SECONDS,
                owner=owner,
            )
        except ConsumerGroupDescribeDeniedError as exc:
            raise DelegateLocusAclRefusedError(
                f"cannot confirm a deployed orchestrator is consuming "
                f"'{command_topic}': {exc} Refusing rather than running here and "
                "reporting it as a lane result. Add the grant to the lane's "
                "declared broker ACLs and apply them, or pass --bus inmemory --locus "
                "in-process to run it locally on purpose.",
                group_ids=exc.group_ids,
            ) from exc
        except ConsumerGroupLivenessUnknownError as exc:
            raise DelegateLocusRefusedError(
                "cannot confirm a deployed orchestrator is consuming "
                f"'{command_topic}' ({exc}). A lane probe that cannot verify "
                "the lane is not a lane probe — refusing rather than running "
                "here and reporting it as a lane result. Fix the broker "
                "address, or pass --bus inmemory --locus in-process to run it locally on "
                "purpose."
            ) from exc
        downstream_groups: tuple[str, ...] = ()
        if groups and downstream is not None:
            try:
                downstream_groups = live_chain_consumer_groups(
                    command_topic=downstream.command_topic,
                    subscribe_topics=downstream.subscribe_topics,
                    bootstrap_servers=kafka_bootstrap,
                    timeout=_LIVENESS_TIMEOUT_SECONDS,
                )
            except ConsumerGroupDescribeDeniedError as exc:
                raise DelegateLocusAclRefusedError(
                    f"cannot confirm {DOWNSTREAM_CHAIN_STAGE} is consuming "
                    f"'{downstream.command_topic}': {exc} Add the grant to the "
                    "lane's declared broker ACLs and apply them, or pass "
                    "--bus inmemory --locus in-process to run it here on purpose.",
                    group_ids=exc.group_ids,
                ) from exc
            except ConsumerGroupLivenessUnknownError as exc:
                raise DelegateDownstreamChainRefusedError(
                    f"cannot confirm {DOWNSTREAM_CHAIN_STAGE} is consuming "
                    f"'{downstream.command_topic}' for {downstream.consumer_contract} "
                    f"({exc}). The first-hop groups were live: {', '.join(groups)}. "
                    "Fix the broker probe, or pass --bus inmemory --locus "
                    "in-process to run it here on purpose."
                ) from exc
        waited = _monotonic() - started
        if groups and (downstream is None or downstream_groups):
            if probes > 1 and downstream is not None:
                logger.warning(
                    "onex delegate: %s consumer %s bound to '%s' after %.1f s "
                    "and %d probes (%s): the lane was rebinding, not down",
                    DOWNSTREAM_CHAIN_STAGE,
                    downstream.consumer_contract,
                    downstream.command_topic,
                    waited,
                    probes,
                    REBIND_WINDOW_FAILURE_CLASS,
                )
            elif probes > 1:
                logger.warning(
                    "onex delegate: a consumer group bound to '%s' after "
                    "%.1f s and %d probes (%s, OMN-18843): the lane was "
                    "rebinding, not down",
                    command_topic,
                    waited,
                    probes,
                    REBIND_WINDOW_FAILURE_CLASS,
                )
            return groups, downstream_groups, waited
        if waited + _REBIND_POLL_SECONDS > _REBIND_WAIT_SECONDS:
            if groups and downstream is not None:
                raise DelegateDownstreamChainRefusedError(
                    f"no live consumer group is bound at {DOWNSTREAM_CHAIN_STAGE} "
                    f"to '{downstream.command_topic}' for "
                    f"{downstream.consumer_contract}. The first-hop groups WERE "
                    f"live: {', '.join(groups)}. Waited {waited:.1f} s over "
                    f"{probes} probes, the bound for the "
                    f"{REBIND_WINDOW_FAILURE_CLASS} class "
                    f"({_REBIND_WAIT_SECONDS:.0f} s). The deployed orchestrator "
                    "would accept the command and then wait its whole execution "
                    "budget, reported to the caller as a hang. Start the runtime "
                    "that hosts that consumer, or pass --bus inmemory --locus "
                    "in-process to run it here on purpose."
                )
            raise DelegateLocusRefusedError(
                f"no live consumer group is bound to '{command_topic}' for "
                f"{owner.service}.{owner.node} — there "
                "is no deployed orchestrator to make the accept/climb "
                "decision. The command would be published and nothing would "
                f"answer it. Waited {waited:.1f} s over {probes} probes, the "
                f"bound for the {REBIND_WINDOW_FAILURE_CLASS} class "
                f"({_REBIND_WAIT_SECONDS:.0f} s, OMN-18843), so this is a lane "
                "that is down, not one that is rebinding. Start the runtime "
                "that consumes this topic, or pass --bus inmemory --locus in-process to run "
                "it here on purpose."
            )
        if groups and downstream is not None:
            logger.warning(
                "onex delegate: %s has no consumer group bound to '%s' for %s "
                "yet (%s): waited %.1f of %.0f s, re-probing in %.0f s",
                DOWNSTREAM_CHAIN_STAGE,
                downstream.command_topic,
                downstream.consumer_contract,
                REBIND_WINDOW_FAILURE_CLASS,
                waited,
                _REBIND_WAIT_SECONDS,
                _REBIND_POLL_SECONDS,
            )
        else:
            logger.warning(
                "onex delegate: no consumer group is bound to '%s' yet (%s, "
                "OMN-18843): waited %.1f of %.0f s, re-probing in %.0f s",
                command_topic,
                REBIND_WINDOW_FAILURE_CLASS,
                waited,
                _REBIND_WAIT_SECONDS,
                _REBIND_POLL_SECONDS,
            )
        _sleep(_REBIND_POLL_SECONDS)
