# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Backend health probes for onex.backends entry point discovery.

Each backend (Kafka, Postgres) provides a probe function that returns a
ModelProbeResult with a 4-state severity:

    DISCOVERED  — entry point found, no connectivity attempted
    REACHABLE   — TCP connect succeeded (or auth failed after connect)
    HEALTHY     — basic operations succeed (topic list, SELECT 1)
    AUTHORITATIVE — safe to replace the local default for this protocol

Authority doctrine:
    AUTHORITATIVE means the backend is ready to be the *sole* provider of
    its protocol at runtime. For Kafka this means brokers match env config
    and at least topic listing works. For Postgres this means the required
    schema tables exist.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from typing import Protocol, cast

from omnibase_infra.backends.enum_probe_state import EnumProbeState
from omnibase_infra.backends.model_probe_result import ModelProbeResult

logger = logging.getLogger(__name__)

_MAX_LIVE_CONSUMER_GROUP_DESCRIBE_CANDIDATES = 16


class ConsumerGroupDescribeResponse(Protocol):
    groups: list[tuple[object, ...]]


def _tcp_reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    """Return True if a TCP connection to host:port succeeds."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, TimeoutError):
        return False


def _has_live_consumer_group(admin: object, topic: str, *, timeout: float) -> bool:
    """Return True when a ``Stable`` consumer group is bound to ``topic``.

    Group IDs minted by ``derive_consumer_group_id`` / ``derive_service_group_id``
    (``omnibase_core.event_bus.util_consumer_group``) carry a ``.__t.<topic>``
    scope suffix for every topic-scoped subscription (``TOPIC_SCOPE_INFIX``). A
    live, ``Stable`` group ending in that suffix is real wiring-truth evidence
    that *something* is actively consuming this topic right now — independent
    of which service/node owns it, and independent of what host/IP the caller
    used to reach the broker.

    Best-effort and probe-safe: any failure (timeout, unsupported broker API,
    permission error, missing optional dependency) returns ``False`` rather
    than raising. "Cannot confirm liveness" is never conflated with
    "confirmed dead" — callers must treat a ``False`` return as ambiguous and
    fall back to a weaker signal, never as a negative health verdict on its
    own (see :func:`probe_kafka`'s Stage 3 authority precedence).
    """
    try:
        from omnibase_core.event_bus.util_consumer_group import TOPIC_SCOPE_INFIX

        future = admin.list_consumer_groups(  # type: ignore[attr-defined]
            request_timeout=timeout
        )
        listing = future.result(timeout=timeout + 1.0)
    except Exception:  # noqa: BLE001 — probe must never raise
        return False

    suffix = f"{TOPIC_SCOPE_INFIX}{topic}"
    for group in getattr(listing, "valid", []):
        try:
            if group.group_id.endswith(suffix) and group.state.name == "STABLE":
                return True
        except AttributeError:
            continue
    return False


class ConsumerGroupLivenessUnknownError(RuntimeError):
    """The broker could not be asked which consumers are bound to a topic.

    Distinct from "no live consumer": a caller that is about to hand work to
    somebody else must be able to tell "nobody is listening" from "I could not
    find out". :func:`_has_live_consumer_group` deliberately collapses both
    into ``False`` because it feeds a health *grade* where a weaker signal is
    an acceptable fallback. A fail-closed dispatch gate cannot use that
    collapse — UNKNOWN has to refuse, not proceed (OMN-17295 AC1).
    """


def live_consumer_groups(
    *,
    topic: str,
    bootstrap_servers: str | None = None,
    timeout: float = 5.0,
) -> tuple[str, ...]:
    """Return the ids of every ``Stable`` consumer group bound to *topic*.

    Wiring truth, not configuration: a group id ending in the
    ``TOPIC_SCOPE_INFIX`` scope suffix for *topic*, in state ``STABLE``, is a
    process that is subscribed and joined RIGHT NOW. This is the only
    un-forgeable answer to "if I publish here, is anything going to pick it
    up" available to an off-box caller — broker-identity string matching is
    structurally blind through any address the broker does not advertise
    (OMN-16529), and a topic's existence says nothing about consumers.

    Unlike :func:`_has_live_consumer_group` this is NOT probe-safe: it raises
    rather than reporting a bare ``False`` for a question it could not ask.
    Callers gating a dispatch on the answer must fail closed on UNKNOWN.

    Args:
        topic: The exact topic the caller is about to publish to. Liveness on
            a different topic proves nothing about this one — passing a
            related-sounding topic is how a gate ends up green while the
            actual command lands nowhere.
        bootstrap_servers: Comma-separated broker addresses. Defaults to
            ``KAFKA_BOOTSTRAP_SERVERS``.
        timeout: Admin request timeout in seconds.

    Returns:
        Group ids, sorted, possibly empty. Empty is a real answer: the broker
        was asked and no ``STABLE`` group is bound.

    Raises:
        ConsumerGroupLivenessUnknownError: the question could not be answered
            — no broker configured, the lane transport could not be resolved,
            or the admin call failed.

    Note:
        Synchronous by contract because the delegate CLI's locus gate is, so
        the aiokafka admin round trip runs under :func:`asyncio.run`. There is
        no async caller today; one would pass through
        :func:`_live_consumer_groups_async` directly rather than nest a loop.
    """
    resolved = bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "")
    if not resolved:
        raise ConsumerGroupLivenessUnknownError(
            "no broker address: neither an explicit bootstrap nor "
            "KAFKA_BOOTSTRAP_SERVERS is set"
        )
    try:
        return asyncio.run(
            _live_consumer_groups_async(
                topic=topic, bootstrap_servers=resolved, timeout=timeout
            )
        )
    except ConsumerGroupLivenessUnknownError:
        raise
    except Exception as exc:
        raise ConsumerGroupLivenessUnknownError(
            f"could not list consumer groups on {resolved}: {exc}"
        ) from exc


async def _live_consumer_groups_async(
    *, topic: str, bootstrap_servers: str, timeout: float
) -> tuple[str, ...]:
    """Ask the broker the liveness question with the runtime's own client.

    OMN-18418: this was the one bus caller in the repository built on the
    synchronous ``confluent_kafka`` family, while every producer, consumer and
    admin client the runtime wires is ``aiokafka`` via
    :func:`build_aiokafka_auth_kwargs`. That split is not cosmetic. Token-
    callback mechanisms — ``AWS_MSK_IAM`` and ``OAUTHBEARER`` — cannot be
    expressed as librdkafka config entries at all, so
    :func:`build_confluent_auth_config` refuses them by design, and the refusal
    arrived here as an exception and left as UNKNOWN. On onex-dev, the only
    ``AWS_MSK_IAM`` lane, that refused EVERY dispatched delegation with a
    message about the lane rather than about the client: measured in-cluster
    2026-09-15 in ``omninode-runtime-7946cbb694-wv8v7`` on the dev-system
    cluster, where the runtime in the same pod was consuming the command topic
    throughout.

    OMN-17304 hit this class once already and fixed it by threading confluent
    credentials into the same construction, which is why it reproduced the
    moment a mechanism arrived the family cannot express. One client
    resolution path is the fix, not a second set of credentials: the MSK token
    callback has exactly one implementation and the probe shares it.
    """
    from aiokafka.admin import AIOKafkaAdminClient

    from omnibase_core.event_bus.util_consumer_group import TOPIC_SCOPE_INFIX
    from omnibase_infra.event_bus.kafka_auth import (
        build_aiokafka_auth_kwargs_for,
    )

    admin = AIOKafkaAdminClient(
        bootstrap_servers=bootstrap_servers,
        request_timeout_ms=int(timeout * 1000),
        # OMN-18432: the probe authenticates as whoever the publish will. A
        # bound lane transport answers for its own address; everywhere else
        # this is the environment-sourced answer it has always been.
        **build_aiokafka_auth_kwargs_for(bootstrap_servers),
    )
    await admin.start()
    try:
        # Metadata FIRST, deliberately. A cluster that cannot describe itself
        # cannot be asked about consumers, and a client that answers the group
        # question anyway answers it EMPTY — which reads as "nobody is bound"
        # and is exactly the UNKNOWN/absent conflation this function exists to
        # prevent. Observed live 2026-08-31 against a closed port.
        await admin.describe_cluster()
        listing = await admin.list_consumer_groups()

        # Narrow by name before describing. The group id carries the topic it
        # is scoped to, so the describe — one coordinator round trip per group
        # — runs over the handful of candidates rather than the ~600 groups a
        # real cluster lists.
        suffix = f"{TOPIC_SCOPE_INFIX}{topic}"
        candidates = sorted(
            {
                str(entry[0])
                for entry in listing
                if entry and str(entry[0]).endswith(suffix)
            }
        )
        if not candidates:
            return ()
        if len(candidates) > _MAX_LIVE_CONSUMER_GROUP_DESCRIBE_CANDIDATES:
            raise ConsumerGroupLivenessUnknownError(
                f"consumer group liveness for {topic!r} on {bootstrap_servers} "
                f"matched {len(candidates)} candidate groups; refusing to run "
                "unbounded serial DescribeGroups probes"
            )

        # ONE CANDIDATE PER DESCRIBE, deliberately. ``describe_consumer_groups``
        # batches every group that shares a coordinator into a single
        # ``DescribeGroupsRequest`` and gathers the responses concurrently, and
        # against MSK that batched form fails to decode: measured in-cluster on
        # onex-dev 2026-09-16, a describe of the three groups bound to
        # ``onex.cmd.omnimarket.delegate-skill.v1`` died with
        # ``ValueError: Buffer underrun decoding string`` and took the broker
        # connection down with it, while the SAME three groups described ONE AT
        # A TIME returned cleanly (``Stable``/1 member, ``Empty``/0, ``Stable``/1).
        # Serial describes are intentionally capped above so this fail-closed
        # gate cannot turn a broad group listing into an unbounded hot-path
        # probe. If a later aiokafka/MSK combination proves batched describes
        # safe, this loop can be collapsed only with a lane proof that covers
        # the same three-group shape. Until then, any describe failure or
        # missing candidate response still leaves the answer UNKNOWN rather
        # than returning a partial consumer set.
        described: list[ConsumerGroupDescribeResponse] = []
        for candidate in candidates:
            responses = cast(
                "list[ConsumerGroupDescribeResponse]",
                await admin.describe_consumer_groups([candidate]),
            )
            response_group_ids = {
                str(group[1])
                for response in responses
                for group in getattr(response, "groups", [])
                if len(group) >= 2
            }
            if candidate not in response_group_ids:
                raise ConsumerGroupLivenessUnknownError(
                    f"describing consumer group {candidate!r} on "
                    f"{bootstrap_servers} returned no matching group; "
                    "the answer is incomplete"
                )
            described.extend(responses)
    finally:
        await admin.close()

    # Wiring truth needs the group's STATE, which the listing does not carry.
    # ``Stable`` is Kafka's own spelling on the wire; the confluent enum spelled
    # it ``STABLE``, so the comparison is case-folded rather than literal.
    found: set[str] = set()
    for response in described:
        for group in getattr(response, "groups", []):
            error_code, group_id, state = group[0], group[1], group[2]
            if error_code:
                # A per-group error makes the listing partial, and a partial
                # answer to "is anything consuming this" is not an answer.
                raise ConsumerGroupLivenessUnknownError(
                    f"describing consumer group {group_id!r} on "
                    f"{bootstrap_servers} returned error code {error_code}; "
                    "the answer is incomplete"
                )
            if str(state).upper() == "STABLE":
                found.add(str(group_id))
    return tuple(sorted(found))


async def consumer_group_topic_backlog(
    *,
    topic: str,
    consumer_group: str,
    bootstrap_servers: str,
) -> int:
    """Return ``consumer_group``'s uncommitted backlog on ``topic``.

    OMN-18852. The second broker question the delegation path asks, and it
    lives beside the first on purpose: :func:`_live_consumer_groups_async`
    resolves WHICH groups are bound to the command topic, and this resolves
    HOW FAR BEHIND one of them is. Both are read-only observations made by a
    CLI that owns no runtime bus, both are answered with the runtime's own
    ``aiokafka`` family, and both authenticate through
    :func:`build_aiokafka_auth_kwargs_for` so a bound lane transport answers
    for its own address. Splitting them across two modules would have put the
    second one's client construction in a module with no business owning a
    transport at all, and would have reproduced the OMN-18418 split this
    module's sibling docstring records: one client resolution path is the fix.

    The figure is broker-reported throughout -- committed offsets against
    log-end offsets, through the ``ServiceConsumerLagObserver`` /
    ``AdapterKafkaAdminLag`` pair the topic-migration drain gate uses. Nothing
    here accepts a depth from a caller.

    Args:
        topic: Topic whose backlog is wanted.
        consumer_group: Group whose committed offsets are the low-water mark.
        bootstrap_servers: Broker address the caller was itself bound to.

    Returns:
        Total uncommitted records across every partition of ``topic``.

    Raises:
        ValueError: the group reported no observed partitions on ``topic``,
            so there is no committed offset to measure a backlog against and
            a zero would be a fabrication.
    """
    from aiokafka import AIOKafkaConsumer
    from aiokafka.admin import AIOKafkaAdminClient

    from omnibase_infra.event_bus.kafka_auth import build_aiokafka_auth_kwargs_for
    from omnibase_infra.migration.adapter_kafka_admin_lag import AdapterKafkaAdminLag
    from omnibase_infra.migration.service_consumer_lag_observer import (
        ServiceConsumerLagObserver,
    )

    auth_kwargs = build_aiokafka_auth_kwargs_for(bootstrap_servers)
    admin = AIOKafkaAdminClient(bootstrap_servers=bootstrap_servers, **auth_kwargs)
    consumer = AIOKafkaConsumer(
        bootstrap_servers=bootstrap_servers,
        # No group_id and no auto-commit: join no group, commit nothing,
        # perturb no offset the run being measured depends on.
        group_id=None,
        enable_auto_commit=False,
        **auth_kwargs,
    )
    await admin.start()
    try:
        await consumer.start()
        try:
            # ``AIOKafkaAdminClient`` satisfies the committed-offset half of
            # ``ProtocolKafkaAdminLike`` structurally; the adapter supplies
            # the ``list_offsets`` half the pinned 0.13.0 client omits
            # (OMN-12632), and is itself the full surface the observer needs.
            observer = ServiceConsumerLagObserver(AdapterKafkaAdminLag(admin, consumer))
            lag = await observer.observe(consumer_group)
            if not lag.has_partitions_for_topic(topic):
                raise ValueError(
                    f"group {consumer_group!r} has no observed partitions on {topic!r}"
                )
            return lag.lag_for_topic(topic)
        finally:
            await consumer.stop()
    finally:
        await admin.close()


def probe_kafka(
    *,
    bootstrap_servers: str | None = None,
    timeout: float = 2.0,
    authority_topic: str | None = None,
) -> ModelProbeResult:
    """Probe Kafka/Redpanda backend health.

    Probe stages:
        1. TCP connect to first broker → REACHABLE
        2. Topic list via confluent_kafka AdminClient → HEALTHY
        3. Authority check → AUTHORITATIVE, in precedence order:
           3a. ``authority_topic`` given and a ``Stable`` consumer group is
               bound to it (:func:`_has_live_consumer_group`) — real
               wiring-truth liveness, works identically on-box or off-box.
           3b. Otherwise, brokers returned by the cluster match the
               configured host string (the original OMN-7075 check).

    Auth failure at any stage results in REACHABLE (not HEALTHY).

    OMN-16529: stage 3b (broker-identity string match) is structurally blind
    for any off-box caller reaching the broker via an address the broker does
    not itself advertise — e.g. a Tailscale/MagicDNS-fronted broker dialed by
    its plain LAN IP always advertises the MagicDNS hostname back, never the
    caller's dialed IP, so ``returned_brokers & configured_hosts`` is empty
    even when the broker is genuinely healthy and fully authoritative for
    this caller (confirmed live against ``.201``'s dev-lane redpanda: probe
    state pinned at HEALTHY, reason "broker mismatch", 100% reproducible).
    That check compares an off-box-unreachable SURFACE (a hostname string),
    not a real signal of whether traffic published here will actually be
    served. Stage 3a fixes this for any caller that names the specific topic
    it intends to use (``resolve_default_bus`` does, for the delegation path)
    by checking actual consumer-group liveness instead — the node-liveness
    doctrine's "consumer groups are wiring truth". Callers that omit
    ``authority_topic`` keep the exact pre-OMN-16529 behavior (stage 3b only).

    Args:
        bootstrap_servers: Comma-separated broker addresses.
            Defaults to KAFKA_BOOTSTRAP_SERVERS env var.
        timeout: TCP connection timeout in seconds.
        authority_topic: Optional topic name. When supplied, a live ``Stable``
            consumer group bound to this topic is sufficient for
            AUTHORITATIVE, ahead of (and independent of) the broker-identity
            check. Pass the exact topic the caller is about to publish to —
            liveness on a different topic proves nothing about this one.

    Returns:
        ModelProbeResult with probe state and reason.
    """
    backend_name = "event_bus_kafka"

    if bootstrap_servers is None:
        bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "")

    if not bootstrap_servers:
        return ModelProbeResult(
            state=EnumProbeState.DISCOVERED,
            reason="KAFKA_BOOTSTRAP_SERVERS not set",
            backend_label=backend_name,
        )

    # Parse first broker for TCP check
    first_broker = bootstrap_servers.split(",")[0].strip()
    parts = first_broker.rsplit(":", 1)
    if len(parts) != 2:
        return ModelProbeResult(
            state=EnumProbeState.DISCOVERED,
            reason=f"Cannot parse broker address: {first_broker}",
            backend_label=backend_name,
        )

    host, port_str = parts
    try:
        port = int(port_str)
    except ValueError:
        return ModelProbeResult(
            state=EnumProbeState.DISCOVERED,
            reason=f"Invalid port in broker address: {first_broker}",
            backend_label=backend_name,
        )

    # Stage 1: TCP reachability
    if not _tcp_reachable(host, port, timeout=timeout):
        return ModelProbeResult(
            state=EnumProbeState.DISCOVERED,
            reason=f"TCP connect to {host}:{port} failed",
            backend_label=backend_name,
        )

    # Stage 2: Topic listing via AdminClient
    try:
        from confluent_kafka.admin import AdminClient

        from omnibase_infra.event_bus.kafka_auth import (
            build_confluent_auth_config_from_env,
        )

        # OMN-18012: honour the lane transport. A PLAINTEXT lane resolves to
        # {} and keeps the previous construction; a SASL lane gets the
        # credentials instead of a handshake failure reported as "auth
        # failure -> REACHABLE".
        admin = AdminClient(
            {
                "bootstrap.servers": bootstrap_servers,
                "socket.timeout.ms": int(timeout * 1000),
                "request.timeout.ms": int(timeout * 1000),
                **build_confluent_auth_config_from_env(),
            }
        )
        cluster_metadata = admin.list_topics(timeout=timeout)
        topic_count = len(cluster_metadata.topics)
    except ImportError:
        return ModelProbeResult(
            state=EnumProbeState.REACHABLE,
            reason="confluent_kafka not installed; TCP reachable but cannot list topics",
            backend_label=backend_name,
        )
    except Exception as exc:  # noqa: BLE001 — probe must never raise
        reason = str(exc)
        # Auth failures are REACHABLE, not HEALTHY
        if "auth" in reason.lower() or "sasl" in reason.lower():
            return ModelProbeResult(
                state=EnumProbeState.REACHABLE,
                reason=f"Auth failure: {reason}",
                backend_label=backend_name,
            )
        return ModelProbeResult(
            state=EnumProbeState.REACHABLE,
            reason=f"TCP reachable but topic list failed: {reason}",
            backend_label=backend_name,
        )

    # Stage 3a: Authority check — live consumer-group liveness (OMN-16529).
    # Takes precedence over the broker-identity check below because it is a
    # real liveness signal rather than a string comparison that is
    # structurally blind off-box (see the OMN-16529 docstring note above).
    if authority_topic and _has_live_consumer_group(
        admin, authority_topic, timeout=timeout
    ):
        return ModelProbeResult(
            state=EnumProbeState.AUTHORITATIVE,
            reason=(
                f"Kafka healthy with {topic_count} topics; a Stable consumer "
                f"group is bound to {authority_topic!r}"
            ),
            backend_label=backend_name,
        )

    # Stage 3b: Authority check — brokers returned match env config
    try:
        returned_brokers = {b.host for b in cluster_metadata.brokers.values()}
        configured_hosts = {
            addr.split(",")[0].rsplit(":", 1)[0].strip() for addr in [bootstrap_servers]
        }
        brokers_match = bool(returned_brokers & configured_hosts)
    except Exception:  # noqa: BLE001 — best-effort authority check
        brokers_match = False

    if brokers_match and topic_count >= 0:
        return ModelProbeResult(
            state=EnumProbeState.AUTHORITATIVE,
            reason=f"Kafka healthy with {topic_count} topics, brokers match config",
            backend_label=backend_name,
        )

    return ModelProbeResult(
        state=EnumProbeState.HEALTHY,
        reason=f"Kafka reachable with {topic_count} topics but broker mismatch",
        backend_label=backend_name,
    )


def probe_postgres(
    *,
    host: str | None = None,
    port: int | None = None,
    user: str | None = None,
    password: str | None = None,
    dbname: str | None = None,
    timeout: float = 2.0,
    required_tables: tuple[str, ...] = ("snapshots", "projections"),
) -> ModelProbeResult:
    """Probe PostgreSQL backend health.

    Probe stages:
        1. TCP connect → REACHABLE
        2. SELECT 1 via psycopg2 → HEALTHY
        3. Required schema tables exist → AUTHORITATIVE

    Auth failure at any stage results in REACHABLE (not HEALTHY).

    Args:
        host: Postgres host. Defaults to localhost.
        port: Postgres port. Defaults to PGPORT env var or 5436.
        user: Postgres user. Defaults to PGUSER or "postgres".
        password: Postgres password. Defaults to POSTGRES_PASSWORD env var.
        dbname: Database name. Defaults to PGDATABASE or "omnibase_infra".
        timeout: TCP connection timeout in seconds.
        required_tables: Tables that must exist for AUTHORITATIVE state.

    Returns:
        ModelProbeResult with probe state and reason.
    """
    backend_name = "state_postgres"

    effective_host: str = host or os.environ["PGHOST"]
    try:
        effective_port = port or int(os.getenv("PGPORT", "5436"))
    except ValueError:
        return ModelProbeResult(
            state=EnumProbeState.DISCOVERED,
            reason=f"Invalid PGPORT value: {os.getenv('PGPORT', '')}",
            backend_label=backend_name,
        )
    effective_user = user or os.getenv("PGUSER", "postgres")
    effective_password = password or os.getenv("POSTGRES_PASSWORD", "")
    effective_dbname = dbname or os.getenv("PGDATABASE", "omnibase_infra")

    # Stage 1: TCP reachability
    if not _tcp_reachable(effective_host, effective_port, timeout=timeout):
        return ModelProbeResult(
            state=EnumProbeState.DISCOVERED,
            reason=f"TCP connect to {effective_host}:{effective_port} failed",
            backend_label=backend_name,
        )

    # Stage 2: SELECT 1 via psycopg2
    try:
        import psycopg2

        conn = psycopg2.connect(
            host=effective_host,
            port=effective_port,
            user=effective_user,
            password=effective_password,
            dbname=effective_dbname,
            connect_timeout=int(timeout),
        )
    except ImportError:
        return ModelProbeResult(
            state=EnumProbeState.REACHABLE,
            reason="psycopg2 not installed; TCP reachable but cannot query",
            backend_label=backend_name,
        )
    except Exception as exc:  # noqa: BLE001 — probe must never raise
        reason = str(exc)
        if "auth" in reason.lower() or "password" in reason.lower():
            return ModelProbeResult(
                state=EnumProbeState.REACHABLE,
                reason=f"Auth failure: {reason}",
                backend_label=backend_name,
            )
        return ModelProbeResult(
            state=EnumProbeState.REACHABLE,
            reason=f"TCP reachable but connect failed: {reason}",
            backend_label=backend_name,
        )

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
    except Exception as exc:  # noqa: BLE001 — probe must never raise
        conn.close()
        return ModelProbeResult(
            state=EnumProbeState.REACHABLE,
            reason=f"Connected but SELECT 1 failed: {exc}",
            backend_label=backend_name,
        )

    # Stage 3: Schema table check for authority
    if not required_tables:
        conn.close()
        return ModelProbeResult(
            state=EnumProbeState.HEALTHY,
            reason="SELECT 1 succeeded, no required tables specified",
            backend_label=backend_name,
        )

    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(required_tables))
            cur.execute(
                f"SELECT table_name FROM information_schema.tables "  # noqa: S608 — parameterized via %s
                f"WHERE table_schema = 'public' AND table_name IN ({placeholders})",
                required_tables,
            )
            found_tables = {row[0] for row in cur.fetchall()}
    except Exception as exc:  # noqa: BLE001 — probe must never raise
        conn.close()
        return ModelProbeResult(
            state=EnumProbeState.HEALTHY,
            reason=f"SELECT 1 succeeded but schema check failed: {exc}",
            backend_label=backend_name,
        )
    finally:
        conn.close()

    missing = set(required_tables) - found_tables
    if missing:
        return ModelProbeResult(
            state=EnumProbeState.HEALTHY,
            reason=f"Missing required tables: {sorted(missing)}",
            backend_label=backend_name,
        )

    return ModelProbeResult(
        state=EnumProbeState.AUTHORITATIVE,
        reason=f"Postgres healthy with all required tables: {sorted(required_tables)}",
        backend_label=backend_name,
    )
