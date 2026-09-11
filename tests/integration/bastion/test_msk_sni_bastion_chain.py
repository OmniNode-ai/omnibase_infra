# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Chain tests for the lab -> MSK SNI bastion -> MSK path (OMN-15744).

WHY THIS FILE EXISTS
    On 2026-08-06 MSK `omninode-dev-msk` ran INCREASE_BROKER_COUNT 2 -> 6. The
    SNI bastion's nginx stream map still described the 2-broker cluster and
    carried `default b1`, so SNI for b-3..b-6 fell through onto broker 1.

    Nothing detected this for a MONTH, and the reason is precise: bootstrap and
    metadata are served by ANY broker, so every liveness surface stayed green.
    Only LEADER-ROUTED requests failed, and they failed with
    NotLeaderForPartition, which reads as transient Kafka noise. Eight of 21
    tenant topic-partitions were silently unwritable, including
    delegation-request / delegation-completed / delegation-failed.

    These two tests exist so the next resize is caught in one scheduled run.

WHY NOT A TLS-ONLY TEST — measured, not assumed
    The obvious cheap test is "handshake against each broker SNI". It is
    WORTHLESS here, and this was verified rather than reasoned about: on
    2026-09-10 all six broker SNIs presented ONE wildcard certificate,
    sha256[:16] e4224daa249fdb90, identical across b-1..b-6. A TLS-only golden
    test would therefore have been GREEN for the entire month-long outage. Any
    test that cannot tell WHICH broker answered does not test this defect.

    So the golden test speaks the Kafka protocol and discriminates on the
    broker's own answer. See `_leader_discriminating_error` below.

WHY NOT "produce to a topic led by each of the six brokers"
    That was the first design and it is not achievable here, for two reasons
    established against the live system rather than guessed:
      1. No visible topic is led by broker 4 (leader census 2026-09-10:
         broker 1 -> 7 partitions, 2 -> 6, 3 -> 2, 5 -> 5, 6 -> 1, 4 -> 0).
      2. The sanctioned topic-creation path (`scripts/create_kafka_topics.py`
         with `ModelTopicProvisioningPolicy`) governs REPLICATION FACTOR only.
         Nothing in this codebase can request or influence partition LEADER
         placement, so a topic created to sit on broker 4 would land wherever
         the cluster chose and the test would be flaky by construction.
    The error-code discriminator below covers all six brokers deterministically,
    including broker 4, with no topic creation and no dependence on where
    leaders happen to sit. It is a strictly stronger test than the one it
    replaces, not a weaker substitute for it.

RUNNING
    Scheduled by .github/workflows/msk-bastion-canary.yml on the self-hosted
    `omnibase-deploy` runner — the only runner with network reach to the
    bastion. Not hand-run, and deliberately not part of the PR-time suite:
    it dials live AWS.

        uv run pytest tests/integration/bastion -v -m integration
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import socket
import ssl
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.kafka,
    pytest.mark.serial,
    pytest.mark.bastion,
]

# The bastion is addressed by IP + SNI so the test does not depend on which
# resolver the caller happens to have. Overridable for a second lab.
BASTION_HOST = os.environ.get("MSK_SNI_BASTION_HOST", "100.53.215.198")
BASTION_PORT = int(os.environ.get("MSK_SNI_BASTION_PORT", "9098"))
MSK_CLUSTER_NAME = os.environ.get("MSK_CLUSTER_NAME", "omninode-dev-msk")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# An SNI that is deliberately not, and must never be, in the map.
UNMAPPED_SNI_TEMPLATE = "b-{n}.omninodedevmsk.7ozyd3.c14.kafka.us-east-1.amazonaws.com"
UNMAPPED_BROKER_ORDINAL = 99

NOT_LEADER_FOR_PARTITION = 6


@dataclass(frozen=True)
class BrokerEndpoint:
    """One broker as the MSK control plane reports it."""

    broker_id: int
    endpoint: str


def _live_brokers() -> list[BrokerEndpoint]:
    """Enumerate brokers from the MSK control plane.

    Deliberately live rather than a fixture: a hardcoded broker list is the
    exact artifact that went stale and caused this outage. A test that asserts
    against a list it carries in its own source cannot detect a resize.
    """
    import boto3

    client = boto3.client("kafka", region_name=AWS_REGION)
    arns = [
        c["ClusterArn"]
        for c in client.list_clusters_v2()["ClusterInfoList"]
        if c["ClusterName"] == MSK_CLUSTER_NAME
    ]
    assert arns, (
        f"MSK cluster {MSK_CLUSTER_NAME!r} not found in {AWS_REGION}. "
        "Refusing to pass: an empty cluster list is a broken query, not a "
        "cluster with no brokers."
    )
    nodes = client.list_nodes(ClusterArn=arns[0])["NodeInfoList"]
    brokers = [
        BrokerEndpoint(
            broker_id=int(n["BrokerNodeInfo"]["BrokerId"]),
            endpoint=n["BrokerNodeInfo"]["Endpoints"][0],
        )
        for n in nodes
    ]
    assert brokers, (
        "list_nodes returned zero brokers. Failing closed: an empty result is "
        "a broken query, not evidence of absence."
    )
    return sorted(brokers, key=lambda b: b.broker_id)


@contextlib.contextmanager
def _dns_pinned_to_bastion(names: list[str]) -> Iterator[None]:
    """Resolve the MSK broker names to the bastion for the duration of a block.

    WHY THIS IS NOT A CHEAT. The broker DNS names resolve to VPC-private
    addresses that nothing outside the VPC can reach; inside the deployed
    gateway they are pinned to the bastion by a dnsmasq sidecar (OMN-16449).
    Without pinning, this test would measure THE CALLER'S DNS rather than the
    bastion's routing, and would fail identically whether the map was correct
    or not — a test whose red says nothing.

    Pinning here makes the connection go to the bastion while still presenting
    the broker name as the TLS SNI, which is exactly the packet the bastion
    routes on. The assertion is then purely about the bastion's map.
    """
    pinned = set(names)
    real_getaddrinfo = socket.getaddrinfo

    def fake_getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
        if host in pinned:
            return real_getaddrinfo(BASTION_HOST, port, *args, **kwargs)
        return real_getaddrinfo(host, port, *args, **kwargs)

    socket.getaddrinfo = fake_getaddrinfo
    try:
        yield
    finally:
        socket.getaddrinfo = real_getaddrinfo


def _tls_probe(sni: str, timeout: float = 10.0) -> tuple[bool, str, str]:
    """Handshake against the bastion with `sni`.

    Returns (handshake_ok, detail, cert_sha256_16). Never raises.
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    sock = socket.create_connection((BASTION_HOST, BASTION_PORT), timeout=timeout)
    try:
        wrapped = ctx.wrap_socket(sock, server_hostname=sni)
        der = wrapped.getpeercert(True) or b""
        return True, "handshake completed", hashlib.sha256(der).hexdigest()[:16]
    except Exception as exc:  # noqa: BLE001 - any failure is the signal
        return False, f"{type(exc).__name__}: {exc}", ""
    finally:
        try:
            sock.close()
        except OSError:
            pass


async def _leader_discriminating_error(
    sni: str, topic: str, partition: int
) -> int | str:
    """Ask the broker behind `sni` for the offsets of one partition.

    This is the discriminator the whole golden test rests on. Kafka answers a
    leader-routed request differently depending on WHICH broker receives it:

        * the partition's leader     -> error_code 0
        * any other broker           -> error_code 6 (NotLeaderForPartition)

    So the broker's own error code reports its identity, which a shared
    wildcard certificate cannot. Under the stale `default b1` map every SNI
    reached broker 1, so a leader-1 partition returned 0 for ALL six SNIs —
    which is exactly the failure this asserts against.

    Returns the error code, or a string describing a transport failure.
    """
    from aiokafka.conn import create_conn
    from aiokafka.protocol.offset import OffsetRequest

    from omnibase_infra.event_bus.kafka_auth import MSKTokenProvider

    conn = await create_conn(
        sni,
        BASTION_PORT,
        security_protocol="SASL_SSL",
        ssl_context=ssl.create_default_context(),
        sasl_mechanism="OAUTHBEARER",
        sasl_oauth_token_provider=MSKTokenProvider(region=AWS_REGION),
        client_id="msk-sni-bastion-canary",
        request_timeout_ms=15000,
    )
    try:
        response = await conn.send(
            OffsetRequest(
                replica_id=-1, isolation_level=0, topics=[(topic, [(partition, -1)])]
            )
        )
        _topic_name, partitions = response.topics[0]
        return int(partitions[0][1])
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return f"{type(exc).__name__}: {exc}"
    finally:
        conn.close()


async def _pick_reference_partition() -> tuple[str, int, int]:
    """Return (topic, partition, leader_id) for any live tenant partition.

    Chosen live rather than pinned: a pinned topic is another stale list.
    """
    from aiokafka import AIOKafkaConsumer
    from aiokafka.structs import TopicPartition

    from omnibase_infra.event_bus.kafka_auth import MSKTokenProvider

    brokers = _live_brokers()
    bootstrap = ",".join(f"{b.endpoint}:{BASTION_PORT}" for b in brokers[:3])
    consumer = AIOKafkaConsumer(
        bootstrap_servers=bootstrap,
        security_protocol="SASL_SSL",
        ssl_context=ssl.create_default_context(),
        sasl_mechanism="OAUTHBEARER",
        sasl_oauth_token_provider=MSKTokenProvider(region=AWS_REGION),
        client_id="msk-sni-bastion-canary-meta",
        group_id=None,
        enable_auto_commit=False,
        request_timeout_ms=20000,
    )
    await consumer.start()
    try:
        topics = sorted(t for t in await consumer.topics() if t.startswith("tenant-"))
        assert topics, (
            "metadata returned zero tenant topics. Failing closed rather than "
            "reporting a vacuous pass on an empty query."
        )
        metadata = await consumer._client.fetch_all_metadata()
        for topic in topics:
            for partition in sorted(metadata.partitions_for_topic(topic) or []):
                leader = metadata.leader_for_partition(TopicPartition(topic, partition))
                if leader is not None and leader >= 0:
                    return topic, partition, int(leader)
        raise AssertionError("no tenant partition with a live leader was found")
    finally:
        try:
            await consumer.stop()
        except BaseException:  # noqa: BLE001 - shutdown artifact, not a result
            pass


@pytest.mark.asyncio
async def test_bastion_golden_every_broker_is_reachable_and_distinct() -> None:
    """GOLDEN: every live MSK broker is individually reachable through the map.

    Emits a per-broker table as the evidence artifact. Fails if ANY live broker
    is missing from the map, or if two different broker SNIs turn out to be the
    same broker (which is precisely what `default b1` did).
    """
    brokers = _live_brokers()
    names = [b.endpoint for b in brokers]

    with _dns_pinned_to_bastion(names):
        topic, partition, leader = await _pick_reference_partition()

        table: list[tuple[int, str, int | str, str]] = []
        for broker in brokers:
            code = await _leader_discriminating_error(broker.endpoint, topic, partition)
            if broker.broker_id == leader:
                ok = code == 0
                expectation = "0 (is the leader)"
            else:
                ok = code == NOT_LEADER_FOR_PARTITION
                expectation = f"{NOT_LEADER_FOR_PARTITION} (is NOT the leader)"
            table.append(
                (broker.broker_id, broker.endpoint, code, "PASS" if ok else "FAIL")
            )

    print(f"\nreference partition: {topic}[{partition}], leader={leader}")
    print(f"{'broker':>6}  {'answer':>28}  verdict")
    for broker_id, _endpoint, code, verdict in table:
        print(f"{broker_id:>6}  {code!s:>28}  {verdict}")

    misrouted = [
        broker_id for broker_id, _endpoint, code, verdict in table if verdict == "FAIL"
    ]
    assert not misrouted, (
        f"brokers {misrouted} did not answer as themselves through the bastion. "
        f"Reference partition {topic}[{partition}] is led by broker {leader}: the "
        f"leader must answer 0 and every other broker must answer "
        f"{NOT_LEADER_FOR_PARTITION}. A broker answering 0 when it is not the "
        "leader means its SNI was routed to the leader instead — the stale-map "
        "defect of OMN-15744. Regenerate the map with "
        "aws/cluster-dev/msk-sni-bastion/render-nginx-conf.sh in omninode_infra "
        "and apply it."
    )

    # The positive control for the whole table: the leader itself answered 0.
    # Without this, a run in which EVERY broker errored would read as a pass on
    # the "not the leader" rows alone.
    leader_rows = [row for row in table if row[0] == leader]
    assert leader_rows and leader_rows[0][2] == 0, (
        "positive control failed: the partition's own leader did not answer 0, "
        "so the error codes above carry no information about routing."
    )


def test_bastion_error_chain_unmapped_sni_fails_closed() -> None:
    """ERROR: an SNI the map does not carry must fail loudly, not silently work.

    This is the half that makes the golden test trustworthy. The original
    defect was not "a broker was unreachable" — it was that an UNKNOWN broker
    name still got a working TCP+TLS session and a real Kafka response, from
    the wrong broker. `default b1` made a misconfiguration indistinguishable
    from a healthy path.

    The map now sends anything unmapped to a dead upstream, so an unknown name
    fails at connect. This test proves that, and proves it is not just "the
    bastion is down" by requiring a mapped SNI to succeed in the same run.
    """
    brokers = _live_brokers()
    mapped_sni = brokers[0].endpoint
    unmapped_sni = UNMAPPED_SNI_TEMPLATE.format(n=UNMAPPED_BROKER_ORDINAL)

    mapped_ok, mapped_detail, mapped_fp = _tls_probe(mapped_sni)
    unmapped_ok, unmapped_detail, _ = _tls_probe(unmapped_sni)

    print(f"\n  POSITIVE CONTROL mapped   {mapped_sni}: {mapped_detail}")
    print(f"  UNDER TEST       unmapped {unmapped_sni}: {unmapped_detail}")

    # Positive control first: without it, a bastion that is simply down would
    # satisfy the assertion below and report a green error-chain.
    assert mapped_ok, (
        f"positive control FAILED: mapped SNI {mapped_sni} did not complete a "
        f"handshake ({mapped_detail}). The bastion is unreachable or broken, so "
        "the unmapped-SNI result below carries no information. This is not a "
        "pass."
    )
    assert not unmapped_ok, (
        f"unmapped SNI {unmapped_sni} COMPLETED a TLS handshake (cert "
        f"{mapped_fp}). The fail-closed default is gone and the bastion is "
        "silently routing unknown broker names to a real broker again — the "
        "OMN-15744 defect. Check the `default sni_unmapped` line in "
        "aws/cluster-dev/msk-sni-bastion/nginx.conf in omninode_infra."
    )


def test_bastion_map_covers_every_live_broker() -> None:
    """RESIZE DETECTOR: every broker the control plane reports is routable.

    This is the test that would have caught the 2026-08-06 scale-out inside one
    scheduled run instead of a month. It compares the LIVE control-plane broker
    list against what the bastion will actually route, so adding brokers to the
    cluster turns this red until the map is regenerated.

    It is separate from the golden test on purpose: this one needs no data-plane
    credentials, so it still reports if the MSK IAM identity is unavailable.
    """
    brokers = _live_brokers()
    print(f"\n  control plane reports {len(brokers)} broker(s)")

    unroutable: list[str] = []
    for broker in brokers:
        ok, detail, _ = _tls_probe(broker.endpoint)
        print(f"    broker {broker.broker_id}: {'routable' if ok else detail}")
        if not ok:
            unroutable.append(
                f"broker {broker.broker_id} ({broker.endpoint}): {detail}"
            )

    assert not unroutable, (
        "the bastion cannot route to every live MSK broker:\n  "
        + "\n  ".join(unroutable)
        + "\n\nIf the cluster was resized, regenerate and apply the map: "
        "aws/cluster-dev/msk-sni-bastion/render-nginx-conf.sh --write then "
        "apply-nginx-conf.sh --execute, in omninode_infra."
    )

    # Negative control: a name outside the map must NOT be routable, otherwise
    # "everything is routable" is vacuously true and this test proves nothing.
    unmapped_ok, _detail, _fp = _tls_probe(
        UNMAPPED_SNI_TEMPLATE.format(n=UNMAPPED_BROKER_ORDINAL)
    )
    assert not unmapped_ok, (
        "negative control FAILED: an unmapped broker name is also routable, so "
        "'every live broker is routable' above is vacuous — the map is matching "
        "everything, not the six brokers specifically."
    )
