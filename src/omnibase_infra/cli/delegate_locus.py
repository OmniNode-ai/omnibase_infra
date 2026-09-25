# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Resolve — and prove — where a delegation's orchestrator will run.

OMN-17295 / OMN-17304. ``--bus`` chooses the TRANSPORT. It never chose the
executor, but it was read as though it did, so a "dev-lane probe" issued as
``onex delegate ... --bus kafka`` ran the orchestrator in-process out of the
caller's own venv and its result was reported as a statement about ``.201``.
The instrument produced a confident answer about a machine it never touched.

Two rules make that impossible here:

1. **Locus is resolved, never assumed.** It follows the transport by default —
   a shared bus means somebody else consumes, an in-memory bus means nobody
   can — and an explicit ``--locus`` says so in the record.
2. **A dispatched run fails closed.** Before publishing, the exact command
   topic (read from the contract, not from a constant here) must have a live
   ``STABLE`` consumer group bound to it. No consumer, or an unanswerable
   broker, refuses the run. It never silently degrades to in-process, because
   a degraded lane probe is worse than no lane probe: it still prints a
   receipt.

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
    ConsumerGroupLivenessUnknownError,
    live_consumer_groups,
)
from omnibase_infra.cli.model_delegate_locus_decision import ModelDelegateLocusDecision
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus

__all__ = [
    "REBIND_WINDOW_FAILURE_CLASS",
    "DelegateLocusRefusedError",
    "contract_command_topic",
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
        raw = yaml.safe_load(  # yaml-safe-load-ok: contract is trusted package data
            contract_path.read_text(encoding="utf-8")
        )
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
        raw = yaml.safe_load(  # yaml-safe-load-ok: contract is trusted package data
            contract_path.read_text(encoding="utf-8")
        )
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
        # Best-effort for an in-process run: the topic is informational there
        # (the runtime publishes and consumes it inside this process), and a
        # contract that does not declare one is a legitimate shape for a
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
    groups, bind_wait_seconds = _assert_dispatch_viable(
        command_topic=command_topic,
        kafka_bootstrap=kafka_bootstrap,
    )
    broker = kafka_bootstrap
    logger.info(
        "onex delegate: execution locus DEPLOYED-LANE (%s) — publishing to "
        "'%s' on %s; %d live consumer group(s) bound: %s. The accept/climb "
        "decision is made THERE, not by %s, which supplies only the wire "
        "description",
        resolved_from,
        command_topic,
        broker,
        len(groups),
        ", ".join(groups),
        distribution,
    )
    return ModelDelegateLocusDecision(
        locus=locus,
        resolved_from=resolved_from,
        orchestrator_contract=str(contract_path),
        orchestrator_distribution=distribution,
        command_topic=command_topic,
        broker=broker,
        lane_consumer_groups=groups,
        consumer_bind_wait_seconds=bind_wait_seconds,
    )


def _assert_dispatch_viable(
    *, command_topic: str, kafka_bootstrap: str
) -> tuple[tuple[str, ...], float]:
    """Refuse unless something is provably consuming the command topic.

    Returns the live groups and the seconds spent waiting for them to bind.

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
            )
        except ConsumerGroupLivenessUnknownError as exc:
            raise DelegateLocusRefusedError(
                "cannot confirm a deployed orchestrator is consuming "
                f"'{command_topic}' ({exc}). A lane probe that cannot verify "
                "the lane is not a lane probe — refusing rather than running "
                "here and reporting it as a lane result. Fix the broker "
                "address, or pass --locus in-process to run it locally on "
                "purpose."
            ) from exc
        waited = _monotonic() - started
        if groups:
            if probes > 1:
                logger.warning(
                    "onex delegate: a consumer group bound to '%s' after "
                    "%.1f s and %d probes (%s, OMN-18843): the lane was "
                    "rebinding, not down",
                    command_topic,
                    waited,
                    probes,
                    REBIND_WINDOW_FAILURE_CLASS,
                )
            return groups, waited
        if waited + _REBIND_POLL_SECONDS > _REBIND_WAIT_SECONDS:
            raise DelegateLocusRefusedError(
                f"no live consumer group is bound to '{command_topic}' — there "
                "is no deployed orchestrator to make the accept/climb "
                "decision. The command would be published and nothing would "
                f"answer it. Waited {waited:.1f} s over {probes} probes, the "
                f"bound for the {REBIND_WINDOW_FAILURE_CLASS} class "
                f"({_REBIND_WAIT_SECONDS:.0f} s, OMN-18843), so this is a lane "
                "that is down, not one that is rebinding. Start the runtime "
                "that consumes this topic, or pass --locus in-process to run "
                "it here on purpose."
            )
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
