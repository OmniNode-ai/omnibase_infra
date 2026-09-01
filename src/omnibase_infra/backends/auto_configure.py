# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Auto-configuration for backend registry using onex.backends entry points.

Discovers installed backends via entry points, probes them for health,
and registers the best available backend for each protocol in the
container's service registry.

When omnibase_core's auto_configure_registry() is available (Part 1 merged),
this module delegates to it. Until then, it provides equivalent functionality
using the same entry point group and probe model.
"""

from __future__ import annotations

import logging
import os
from importlib.metadata import entry_points

from omnibase_infra.backends.backend_probe import (
    probe_kafka,
    probe_postgres,
)
from omnibase_infra.backends.enum_probe_state import EnumProbeState
from omnibase_infra.backends.model_probe_result import ModelProbeResult

logger = logging.getLogger(__name__)

# Canonical event-bus target names. Every resolution path in the repo speaks
# these two values and nothing else — ``BUS_CHOICES`` in ``cli/cli_delegate.py``
# and ``RuntimeLocal.SUPPORTED_EVENT_BUS_VALUES`` are the same closed set.
BUS_INMEMORY = "inmemory"
BUS_KAFKA = "kafka"
SUPPORTED_BUS_TYPES: tuple[str, ...] = (BUS_INMEMORY, BUS_KAFKA)

# OMN-17304: this env var holds NO tier in transport resolution any more.
# Per the operator ruling, env vars may bootstrap WHERE configuration is found
# (ONEX_CONTRACTS_DIR), never WHAT the transport is — a shell profile deciding
# the transport of every delegation was exactly the environmental accident the
# ruling removed. The name is retained solely so :func:`resolve_bus_type` can
# WARN when it is set-and-ignored, giving operators a signal to delete the
# export rather than a silent behaviour change.
BUS_TYPE_OVERRIDE_ENV = "ONEX_EVENT_BUS_TYPE"

# The ONE accepted vocabulary, shared by EVERY tier (OMN-16693). ``cloud`` is
# broker-backed and resolves to the Kafka transport; ``EnumEventBusType.CLOUD``
# is a production-safe value a runtime contract may legally declare, so a tier
# that rejected it would turn a valid config into a hard boot error.
#
# Before OMN-16693 this table was consulted only by the env-override tier, so
# ``cloud`` resolved through ``ONEX_EVENT_BUS_TYPE`` and raised through
# ``explicit_bus`` — the same "two tiers disagree about the same word" defect
# OMN-16678 was opened to remove, one layer down.
_BUS_ALIASES: dict[str, str] = {
    "inmemory": BUS_INMEMORY,
    "kafka": BUS_KAFKA,
    "cloud": BUS_KAFKA,
}


def _normalize_bus_value(raw: str, *, source: str, remedy: str) -> str:
    """Map one tier's raw value onto :data:`SUPPORTED_BUS_TYPES`, or raise.

    Args:
        raw: The value as the tier received it (case/whitespace insensitive).
        source: Human-readable origin, named in the error so the operator knows
            WHICH surface to correct (an env var, a flag, a contract field).
        remedy: The concrete next action for this tier.

    Returns:
        The canonical bus name.

    Raises:
        ValueError: ``raw`` names no known transport. Never degrades to "probe
            and hope" — a typo must not read as "no selection"
            (``feedback_no_defensive_no_defaults``).
    """
    resolved = _BUS_ALIASES.get(raw.strip().lower())
    if resolved is None:
        raise ValueError(
            f"{source} {raw!r} is not a recognised event bus. "
            f"Valid values: {', '.join(sorted(_BUS_ALIASES))}. {remedy}"
        )
    return resolved


class EventBusResolutionAmbiguousError(RuntimeError):
    """The event-bus transport could not be resolved to ONE repeatable answer.

    Raised when the Kafka probe returns an INDETERMINATE result — TCP connected
    but the broker's serving state could not be established (metadata timeout,
    auth failure, or a missing client library). Selecting a transport from that
    state is a coin flip: measured over 20 consecutive calls with an unchanged
    environment and a healthy broker, the same probe produced ``kafka`` 14
    times and ``inmemory`` 6 times (``OmniNode-ai/knowledge-base#59``), because
    a transient ``AdminClient.list_topics`` timeout degrades to ``REACHABLE``.

    Failing here — naming the ambiguity and the deterministic remedies — is the
    only outcome that is repeatable when the underlying signal is not
    (``feedback_no_defensive_no_defaults``).
    """


def _import_event_bus_inmemory() -> type:
    """Import the thin infra EventBusInmemory adapter over the core transport.

    OMN-13419: the in-memory transport lives once in omnibase_core; the infra
    adapter (omnibase_infra.event_bus.event_bus_inmemory) wraps it to provide
    the infra-shaped health_check / consumer-group surface the runtime kernel
    needs when Kafka is unavailable.
    """
    from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory as _Cls

    return _Cls


# Probe functions keyed by entry point name
_PROBE_REGISTRY: dict[str, object] = {
    "event_bus_kafka": probe_kafka,
    "state_postgres": probe_postgres,
}


def discover_backends() -> list[ModelProbeResult]:
    """Discover and probe all installed onex.backends entry points.

    Returns:
        List of probe results, one per discovered backend.
    """
    results: list[ModelProbeResult] = []
    backends = entry_points(group="onex.backends")

    for ep in backends:
        probe_fn = _PROBE_REGISTRY.get(ep.name)
        if probe_fn is not None and callable(probe_fn):
            result = probe_fn()
            results.append(result)
        else:
            results.append(
                ModelProbeResult(
                    state=EnumProbeState.DISCOVERED,
                    reason=f"No probe registered for backend '{ep.name}'",
                    backend_label=ep.name,
                )
            )

    return results


def resolve_bus_type(
    *,
    explicit_bus: str | None = None,
    config_bus: str | None = None,
    config_source: str | None = None,
    kafka_bootstrap: str | None = None,
    authority_topic: str | None = None,
) -> tuple[str, str]:
    """Resolve WHICH event-bus transport to use, deterministically (OMN-16678).

    The single authority every call site shares — :func:`select_event_bus` (the
    runtime kernel's path) and ``cli/cli_delegate.py::resolve_default_bus``
    (the ``onex delegate`` path) both route through this function, so the two
    can no longer disagree about what the operator asked for.

    Resolution order, highest first (OMN-17304):

    1. **Explicit argument** (``--bus`` at the CLI, ``bus_type=`` in-process).
       Never second-guessed and never probed against.
    2. **Declared config** (``config.event_bus.type``, OMN-16693). The
       runtime's own configured statement of intent, honoured without probing:
       the config already stated the answer, and probing it only reintroduces
       the chance to contradict it. Every runtime has one — the deployed lanes
       ship ``contracts/runtime/runtime_config.yaml``, and an unconfigured
       install resolves the SHIPPED tier-0 default
       (``runtime/tier0_runtime_config.yaml``: in-memory bus, local profile),
       so for config-resolved callers the probe tier below is structurally
       unreachable.
    3. **Live broker probe** (:func:`~omnibase_infra.backends.backend_probe.probe_kafka`),
       reached only by callers that genuinely pass no config, mapped totally:

       * ``AUTHORITATIVE`` / ``HEALTHY`` -> ``kafka``. Determinate positive.
       * ``DISCOVERED`` -> ``inmemory``. Determinate negative: reached only by
         branches that concluded something definite (no bootstrap configured,
         unparseable broker address, refused TCP connect).
       * ``REACHABLE`` -> :class:`EventBusResolutionAmbiguousError`.
         Indeterminate: TCP is up but the metadata call did not complete, so
         whether this broker will serve this caller is UNKNOWN. This is the
         state a transient 2s ``list_topics`` timeout against a *healthy*
         broker lands in, and mapping it to a transport is what made
         resolution non-repeatable (14 kafka / 6 inmemory over 20 calls,
         unchanged env — ``OmniNode-ai/knowledge-base#59``).

    **``ONEX_EVENT_BUS_TYPE`` holds NO tier (OMN-17304).** The pre-ruling
    ladder ranked it above config, which made one line in an operator's shell
    profile the transport authority for every delegation on the machine. Per
    the operator ruling, env vars may bootstrap WHERE configuration is found
    (``ONEX_CONTRACTS_DIR``), never WHAT the transport is. A set value is
    ignored — with a WARNING naming the removal, so the export is deleted
    rather than silently dead. This applies to every caller: configured
    binding outranks env everywhere, because env stopped being a peer
    authority at all.

    Args:
        explicit_bus: Caller-supplied transport, or ``None`` to auto-resolve.
        config_bus: The transport declared by the runtime's configuration
            (``config.event_bus.type``), or ``None`` when the caller genuinely
            has no config to speak for.
        config_source: Optional human-readable provenance of WHERE
            ``config_bus`` came from (a file path, "shipped tier-0 default
            ..."), appended to the returned reason so a receipt can name the
            authority, not merely the tier.
        kafka_bootstrap: Broker override. ``None`` lets ``probe_kafka`` resolve
            ``KAFKA_BOOTSTRAP_SERVERS`` itself — the already-approved boundary
            for that lookup.
        authority_topic: Topic whose live consumer-group liveness decides the
            AUTHORITATIVE tier (OMN-16529). Pass the exact topic the caller is
            about to publish to, or ``None``.

    Returns:
        ``(bus_type, reason)`` — ``bus_type`` is one of
        :data:`SUPPORTED_BUS_TYPES`; ``reason`` is human-readable provenance
        naming which tier decided, for logging and receipts.

    Raises:
        ValueError: ``explicit_bus`` or ``config_bus`` names a transport that
            does not exist.
        EventBusResolutionAmbiguousError: the probe result is indeterminate and
            nothing above it declared an intent to disambiguate it.
    """
    # OMN-17304: warn on a set-and-ignored ONEX_EVENT_BUS_TYPE. The env-var
    # name is spelled as a LITERAL here, not as BUS_TYPE_OVERRIDE_ENV:
    # `check-env-reads` grandfathers an added read only when it can extract a
    # literal name and match it against the same file's base version, and it
    # deliberately blocks the constant-name (non-literal) read form outright.
    # This read pre-dates OMN-16678 in this file; it no longer participates in
    # resolution — it exists solely to tell the operator their export stopped
    # doing anything (the value is not even validated: a dead variable must
    # not retain the power to fail a boot).
    raw_ignored = os.getenv("ONEX_EVENT_BUS_TYPE", "").strip()
    if raw_ignored:
        logger.warning(
            "%s=%s is set but is ignored (OMN-17304): it no longer holds any "
            "tier in transport resolution. The transport resolves from the "
            "runtime's configured authority (explicit argument > "
            "config.event_bus.type > broker probe); env vars may bootstrap "
            "where config is found (ONEX_CONTRACTS_DIR), never what the "
            "transport is. Remove the export.",
            BUS_TYPE_OVERRIDE_ENV,
            raw_ignored,
        )

    # Tier 1 — explicit caller selection.
    if explicit_bus is not None:
        normalized = _normalize_bus_value(
            explicit_bus,
            source="Explicit event bus selection",
            remedy="Pass one of those values instead.",
        )
        return normalized, f"explicit bus selection: {normalized}"

    # Tier 2 — the transport the runtime's configuration declares (OMN-16693,
    # promoted to the top non-explicit tier by OMN-17304). Honoured without
    # probing: the config already stated the answer, and probing it only
    # reintroduces the chance to contradict it.
    if config_bus is not None:
        resolved_config = _normalize_bus_value(
            config_bus,
            source="config.event_bus.type",
            remedy="Correct the runtime config's event_bus.type field.",
        )
        provenance = f"config.event_bus.type={config_bus}"
        if config_source:
            provenance = f"{provenance} ({config_source})"
        return resolved_config, provenance

    # Tier 3 — probe. Total over EnumProbeState; no implicit default branch.
    probe = probe_kafka(
        bootstrap_servers=kafka_bootstrap, authority_topic=authority_topic
    )
    if probe.state in (EnumProbeState.HEALTHY, EnumProbeState.AUTHORITATIVE):
        return BUS_KAFKA, probe.reason
    if probe.state is EnumProbeState.DISCOVERED:
        return BUS_INMEMORY, f"{probe.state.name}: {probe.reason}"
    raise EventBusResolutionAmbiguousError(
        f"Kafka probe returned {probe.state.name} ({probe.reason}); the broker "
        f"accepted a TCP connection but its serving state could not be "
        f"established, so the transport cannot be resolved repeatably. "
        f"Select one explicitly: pass the bus argument (e.g. "
        f"'--bus {BUS_KAFKA}' / '--bus {BUS_INMEMORY}'), or declare "
        f"event_bus.type in the runtime config (OMN-17304 — "
        f"{BUS_TYPE_OVERRIDE_ENV} no longer selects a transport)."
    )


def select_event_bus(
    *,
    bus_type: str | None = None,
    kafka_bootstrap_servers: str | None = None,
    environment: str = "local",
    consumer_group: str = "onex-runtime",
    circuit_breaker_threshold: int = 5,
) -> object:
    """Construct the event bus for the transport :func:`resolve_bus_type` selects.

    This function owns CONSTRUCTION only. The decision of which transport to
    build belongs to :func:`resolve_bus_type` (OMN-16678), which is shared with
    ``cli/cli_delegate.py::resolve_default_bus`` so both paths apply one
    resolution order (OMN-17304): explicit argument >
    ``config.event_bus.type`` > probe (``ONEX_EVENT_BUS_TYPE`` holds no tier).

    The runtime kernel resolves first and passes the answer back in as
    ``bus_type`` (OMN-16693), so the config tier is applied there and tier 1
    short-circuits here — one decision per boot, never two.

    Behavior change (OMN-16678): the old ``REACHABLE`` branch here built a Kafka
    bus "despite the probe result" while the delegate path mapped the identical
    state to in-memory. That state is indeterminate — a plain metadata timeout
    against a healthy broker lands in it — so it now raises
    :class:`EventBusResolutionAmbiguousError` instead of either side guessing.

    Args:
        bus_type: Explicit transport, bypassing override + probe entirely.
        kafka_bootstrap_servers: Kafka broker addresses.
        environment: Runtime environment identifier.
        consumer_group: Consumer group for the bus.
        circuit_breaker_threshold: Circuit breaker threshold.

    Returns:
        An event bus instance (EventBusKafka or EventBusInmemory).

    Raises:
        EventBusResolutionAmbiguousError: the probe result is indeterminate and
            neither ``bus_type`` nor ``ONEX_EVENT_BUS_TYPE`` disambiguates it.
    """
    # Resolve bootstrap servers (match probe_kafka fallback logic)
    resolved_bootstrap = kafka_bootstrap_servers or os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS", ""
    )

    resolved_bus, reason = resolve_bus_type(
        explicit_bus=bus_type,
        kafka_bootstrap=resolved_bootstrap or None,
    )

    if resolved_bus == BUS_KAFKA:
        logger.info("Event bus resolved to kafka (%s) — using EventBusKafka", reason)
        from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
        from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

        kafka_config = ModelKafkaEventBusConfig(
            bootstrap_servers=resolved_bootstrap,
            environment=environment,
            circuit_breaker_threshold=circuit_breaker_threshold,
        ).apply_environment_overrides()
        return EventBusKafka(config=kafka_config)

    logger.info("Event bus resolved to inmemory (%s) — using EventBusInmemory", reason)
    _InmemoryBus = _import_event_bus_inmemory()
    return _InmemoryBus(
        environment=environment,
        group=consumer_group,
    )
