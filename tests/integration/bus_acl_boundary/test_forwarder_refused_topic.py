# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19928: the gateway forwarder survives a topic its broker refuses.

What happened (regression (a) of 2026-09-28). omnibase_infra#4227 added
``onex.cmd.github.webhook-delivery.v1`` to the forwarder's cloud inbound set.
The dev cloud broker did not admit the tenant-prefixed topic, the consumer's
``start()`` raised a topic-authorization error, and the forwarder exited and
restarted in a loop on the lab. Every broker in CI missed it: the plaintext
ones enforce nothing, and the SASL harness connects as a superuser, which
skips every grant. omnibase_infra#4254 (OMN-15629) is the fix: a refused
inbound topic is peeled off the subscription, logged and retried.

What this pins, against a real broker that enforces grants:

* the process started the way it is deployed stays up for 30 s;
* the refused topic is reported by name, in the forwarder's own
  ``kafka_transport_topic_refused`` line;
* one record produced on an admitted inbound topic reaches the local lane.

Negative control (plan task S3): this file and the harness, copied onto the
#4227 merge (c6c1c47c8) with nothing else changed, fail at the liveness step
with the forwarder exiting on ``TopicAuthorizationFailedError``, never at
setup. The run is cited in the PR body.

Topology. Two throwaway brokers that this test starts and removes: the LANE
broker (the forwarder's local leg) and the CLOUD broker (its trust-boundary
leg). It never adopts a declared broker (``start_redpanda_sasl``, not
``resolve_broker``), so it cannot touch a lab lane's broker. On the cloud
broker the harness superuser only bootstraps; the forwarder connects as a
runtime principal whose grants are derived here from the node contract's
mirror topic set, with exactly one inbound topic left out.

RESIDUAL: the cloud leg speaks SCRAM here, not MSK IAM (see
``forwarder_process.py``). The lane leg's grants are a wildcard, because the
lane leg is not the boundary under test.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
import yaml

from omnibase_core.models.core.model_envelope_metadata import ModelEnvelopeMetadata
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.nodes.node_bus_forwarder_effect.models.model_gateway_forwarder_runtime_config import (
    ModelGatewayForwarderRuntimeConfig,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_topic_transform import (
    prefix_topic,
)
from omnibase_infra.runtime import gateway_forwarder
from tests.integration.customer_path import redpanda_sasl_harness as harness

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.kafka,
    # Two broker boots, a 30 s liveness window and a delivery readback. The
    # repository-wide 60 s watchdog is sized for unit tests (see the same
    # marker in tests/integration/customer_path/).
    pytest.mark.timeout(600),
    pytest.mark.xdist_group("omn19928_bus_acl_boundary"),
]

# The topic omnibase_infra#4227 added to the cloud inbound set: the one the dev
# cloud broker refused on 2026-09-28. The test asserts it is still declared, so
# a contract that drops it fails here instead of silently testing nothing.
REFUSED_CANONICAL_TOPIC = "onex.cmd.github.webhook-delivery.v1"

# Plan task S3: "the process is alive after 30 s".
ALIVE_AFTER_SECONDS = 30.0
READY_DEADLINE_SECONDS = 120.0
DELIVERY_DEADLINE_SECONDS = 90.0

# A synthetic tenant. Nothing outside the two throwaway brokers ever sees it.
TENANT_ID = UUID("19928000-0000-4000-8000-000000000928")
TENANT_SLUG = "acl-boundary-199280000928"
TENANT_PRINCIPAL_ID = f"t-{TENANT_ID.hex}"

CLOUD_PRINCIPAL = harness.RuntimePrincipal(
    "omn19928-forwarder-cloud", "omn19928-cloud-pw"
)
LANE_PRINCIPAL = harness.RuntimePrincipal("omn19928-forwarder-lane", "omn19928-lane-pw")
LANE_CREDENTIAL_REF = "omn19928.lane.runtime"

FORWARDER_PROCESS = Path(__file__).with_name("forwarder_process.py")


@dataclass(frozen=True)
class _Brokers:
    lane: harness.RedpandaSasl
    cloud: harness.RedpandaSasl


@dataclass(frozen=True)
class _Deployment:
    config_path: Path
    broker_ref_map: Path
    lane_credential_map: Path
    cloud_leg_auth: Path
    runtime: ModelGatewayForwarderRuntimeConfig


@pytest.fixture(scope="module")
def brokers() -> Iterator[_Brokers]:
    """Two throwaway auth-required brokers, started here and removed here."""
    if not harness.docker_available():
        if os.environ.get("OMN18012_REQUIRE_HARNESS") == "1":
            pytest.fail(
                "OMN18012_REQUIRE_HARNESS=1 and no docker daemon is reachable; the "
                "bus ACL boundary test did not run"
            )
        pytest.skip("no docker daemon (local developer skip)")
    with ThreadPoolExecutor(max_workers=2) as pool:
        lane_future = pool.submit(harness.start_redpanda_sasl)
        cloud_future = pool.submit(harness.start_redpanda_sasl)
        started: list[harness.RedpandaSasl] = []
        try:
            lane = lane_future.result()
            started.append(lane)
            cloud = cloud_future.result()
            started.append(cloud)
        except BaseException:
            for broker in started:
                harness.stop_redpanda(broker)
            raise
    try:
        yield _Brokers(lane=lane, cloud=cloud)
    finally:
        harness.stop_redpanda(cloud)
        harness.stop_redpanda(lane)


def _contract_cloud_leg() -> dict[str, object]:
    contract = yaml.safe_load(
        gateway_forwarder._DEFAULT_GATEWAY_CONTRACT_PATH.read_text(encoding="utf-8")
    )
    cloud_leg = contract["config"]["gateway_forwarder"]["cloud_leg"]
    return {
        key: cloud_leg[key]
        for key in (
            "broker_provider_id",
            "cloud_broker_ref",
            "cloud_auth_ref",
            "acl_provisioner_ref",
            "msk_region_ref",
            "security_protocol",
            "sasl_mechanism",
        )
    }


def _write_deployment(brokers: _Brokers, root: Path) -> _Deployment:
    """The resolved files a forwarder deployment mounts, pointed at our brokers."""
    cloud_leg = _contract_cloud_leg()
    config = {
        "forwarder": {
            "tenant_identity": {
                "tenant_id": str(TENANT_ID),
                "tenant_slug": TENANT_SLUG,
                "principal_id": TENANT_PRINCIPAL_ID,
            },
            "cloud_bus": cloud_leg,
            "local_transport_flavor": "containerized",
            "mirror_topic_set": "node_bus_forwarder_effect",
            "canary_topic_set": "node_bus_forwarder_effect",
            "heartbeat_interval_seconds": 5,
            "dedupe_store_path": str(root / "delivery.sqlite3"),
        },
        "local_bus": {
            "bootstrap_servers": brokers.lane.bootstrap,
            "environment": "omn19928-lane",
            "auto_offset_reset": "earliest",
            "enable_auto_commit": False,
            "security_protocol": "SASL_PLAINTEXT",
            "sasl_mechanism": brokers.lane.mechanism,
            "sasl_credential_ref": LANE_CREDENTIAL_REF,
        },
        # The declared cloud leg, exactly as a deployment resolves it: the
        # address comes from the broker-ref map, never a literal here.
        "cloud_bus": {
            "environment": "omn19928-cloud",
            "auto_offset_reset": "earliest",
            "enable_auto_commit": False,
            "security_protocol": cloud_leg["security_protocol"],
            "sasl_mechanism": cloud_leg["sasl_mechanism"],
            "msk_region": "us-east-1",
        },
    }
    config_path = root / "forwarder.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    broker_ref_map = root / "broker-ref-map.yaml"
    broker_ref_map.write_text(
        yaml.safe_dump({str(cloud_leg["cloud_broker_ref"]): brokers.cloud.bootstrap}),
        encoding="utf-8",
    )
    lane_credential_map = root / "lane-credentials.yaml"
    lane_credential_map.write_text(
        yaml.safe_dump(
            {
                LANE_CREDENTIAL_REF: {
                    "username": LANE_PRINCIPAL.username,
                    "password": LANE_PRINCIPAL.password,
                }
            }
        ),
        encoding="utf-8",
    )
    cloud_leg_auth = root / "cloud-leg-auth.yaml"
    cloud_leg_auth.write_text(
        yaml.safe_dump(
            {
                "security_protocol": "SASL_PLAINTEXT",
                "sasl_mechanism": brokers.cloud.mechanism,
                "sasl_plain_username": CLOUD_PRINCIPAL.username,
                "sasl_plain_password": CLOUD_PRINCIPAL.password,
                "msk_region": None,
            }
        ),
        encoding="utf-8",
    )
    # The same loader the process runs, so the topic sets below are the ones
    # the process subscribes to, read from the node contract.
    runtime = gateway_forwarder.load_gateway_forwarder_runtime_config(
        config_path,
        broker_ref_map_path=broker_ref_map,
        lane_credential_map_path=lane_credential_map,
    )
    return _Deployment(
        config_path=config_path,
        broker_ref_map=broker_ref_map,
        lane_credential_map=lane_credential_map,
        cloud_leg_auth=cloud_leg_auth,
        runtime=runtime,
    )


def _wire(topic: str) -> str:
    return prefix_topic(TENANT_SLUG, topic)


def _provision(brokers: _Brokers, deployment: _Deployment) -> None:
    """Topics on both brokers; grants for every topic but the refused one."""
    mirror = deployment.runtime.forwarder.mirror_topics
    assert mirror is not None
    canonical = sorted({*mirror.inbound, *mirror.outbound})
    for topic in canonical:
        brokers.lane.create_topic(topic)
        brokers.cloud.create_topic(_wire(topic))

    # Lane leg: not the boundary under test, so one wildcard grant.
    brokers.lane.create_principal(LANE_PRINCIPAL)
    brokers.lane.grant(
        LANE_PRINCIPAL, operations=("all",), topics=("*",), groups=("*",)
    )
    brokers.lane.grant(LANE_PRINCIPAL, operations=("idempotent_write",), cluster=True)

    # Cloud leg: exactly what the forwarder uses, minus one inbound topic.
    admitted_inbound = tuple(
        _wire(topic) for topic in mirror.inbound if topic != REFUSED_CANONICAL_TOPIC
    )
    brokers.cloud.create_principal(CLOUD_PRINCIPAL)
    brokers.cloud.grant(
        CLOUD_PRINCIPAL,
        operations=("read", "describe"),
        topics=admitted_inbound,
        groups=(f"tenant-{TENANT_SLUG}-gateway-forwarder-inbound",),
    )
    brokers.cloud.grant(
        CLOUD_PRINCIPAL,
        operations=("write", "describe"),
        topics=tuple(_wire(topic) for topic in mirror.outbound),
    )
    brokers.cloud.grant(CLOUD_PRINCIPAL, operations=("idempotent_write",), cluster=True)


def _forwarder_env() -> dict[str, str]:
    """The test's environment minus anything that could redirect a bus leg."""
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("KAFKA_") and not key.startswith("OMN18012_BROKER_")
    }


def _log_tail(log_path: Path, lines: int = 60) -> str:
    if not log_path.exists():
        return "<no forwarder output>"
    return "\n".join(
        log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    )


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def _inbound_record(canonical_topic: str) -> tuple[str, str]:
    """One cloud command on ``canonical_topic``, as the cloud side stamps it."""
    envelope_id = uuid4()
    envelope = ModelEventEnvelope[dict[str, object]](
        envelope_id=envelope_id,
        envelope_timestamp=datetime.now(UTC),
        correlation_id=envelope_id,
        source_tool="omn19928-acl-boundary",
        event_type="omn19928.acl-boundary-probe",
        payload={"probe": "omn19928"},
        metadata=ModelEnvelopeMetadata(
            tags={
                "source_tenant_id": str(TENANT_ID),
                "source_tenant_principal_id": TENANT_PRINCIPAL_ID,
            }
        ),
    )
    return str(envelope_id), envelope.model_dump_json(exclude_none=True)


def _read_lane_topic(
    broker: harness.RedpandaSasl, topic: str
) -> list[dict[str, object]]:
    proc = broker.rpk(
        "topic", "consume", topic, "-o", ":end", "-f", "%v\n", check=False
    )
    records: list[dict[str, object]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def test_the_forwarder_survives_a_topic_the_cloud_broker_refuses(
    brokers: _Brokers, tmp_path: Path
) -> None:
    deployment = _write_deployment(brokers, tmp_path)
    mirror = deployment.runtime.forwarder.mirror_topics
    assert mirror is not None
    assert REFUSED_CANONICAL_TOPIC in mirror.inbound, (
        f"{REFUSED_CANONICAL_TOPIC} is no longer a declared inbound topic; pick "
        "the inbound topic this test refuses again, or the test proves nothing"
    )
    _provision(brokers, deployment)

    # --- the broker is ours, and it enforces grants -------------------------
    for broker in (brokers.lane, brokers.cloud):
        assert broker.owned, "the test adopted a broker instead of starting one"
        assert broker.container_id(), "the broker container has no id"
    assert brokers.cloud.container_id() != brokers.lane.container_id()
    superusers = brokers.cloud.superusers()
    assert superusers == (harness.SASL_USERNAME,), superusers
    assert CLOUD_PRINCIPAL.username not in superusers
    users = brokers.cloud.rpk("security", "user", "list").stdout
    assert CLOUD_PRINCIPAL.username in users, users
    acls = brokers.cloud.acl_listing()
    assert f"User:{CLOUD_PRINCIPAL.username}" in acls, acls
    assert _wire(REFUSED_CANONICAL_TOPIC) not in acls, (
        f"the refused topic carries a grant:\n{acls}"
    )
    refused = brokers.cloud.rpk_as(
        CLOUD_PRINCIPAL,
        "topic",
        "describe",
        "--print-summary",
        _wire(REFUSED_CANONICAL_TOPIC),
    )
    assert "TOPIC_AUTHORIZATION_FAILED" in refused.stdout + refused.stderr, (
        "positive control: the cloud broker did not refuse the runtime principal "
        f"on the refused topic, so it is not enforcing grants.\n{refused.stdout}"
        f"\n{refused.stderr}\nACLs:\n{brokers.cloud.acl_listing()}"
    )
    admitted_topic = next(
        topic
        for topic in mirror.inbound
        if topic != REFUSED_CANONICAL_TOPIC
        and not topic.endswith(".gateway-heartbeat.v1")
    )
    admitted = brokers.cloud.rpk_as(
        CLOUD_PRINCIPAL, "topic", "describe", "--print-summary", _wire(admitted_topic)
    )
    assert admitted.returncode == 0 and "AUTHORIZATION_FAILED" not in admitted.stdout, (
        f"the runtime principal cannot describe an admitted topic:\n{admitted.stdout}"
        f"\n{admitted.stderr}"
    )

    # --- start the forwarder as its own process -----------------------------
    ready_file = tmp_path / "ready"
    log_path = tmp_path / "forwarder.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(FORWARDER_PROCESS),
                "--config",
                str(deployment.config_path),
                "--broker-ref-map",
                str(deployment.broker_ref_map),
                "--lane-credential-map",
                str(deployment.lane_credential_map),
                "--cloud-leg-auth",
                str(deployment.cloud_leg_auth),
                "--ready-file",
                str(ready_file),
                "--egress-health-file",
                str(tmp_path / "egress-health.json"),
                "--lane-mirror-health-file",
                str(tmp_path / "lane-mirror-health.json"),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=_forwarder_env(),
        )
    try:
        started = time.monotonic()
        while not ready_file.exists() and process.poll() is None:
            if time.monotonic() - started > READY_DEADLINE_SECONDS:
                pytest.fail(
                    f"the forwarder wrote no ready file in {READY_DEADLINE_SECONDS}s "
                    f"and is still running:\n{_log_tail(log_path)}"
                )
            time.sleep(0.5)
        while (
            process.poll() is None and time.monotonic() - started < ALIVE_AFTER_SECONDS
        ):
            time.sleep(0.5)

        output = log_path.read_text(encoding="utf-8", errors="replace")
        exit_code = process.poll()
        assert exit_code is None, (
            f"the forwarder exited (rc={exit_code}) within {ALIVE_AFTER_SECONDS:.0f}s "
            f"of start; topic-authorization exit: "
            f"{'TopicAuthorizationFailedError' in output}. A broker refusing one "
            f"inbound topic ({_wire(REFUSED_CANONICAL_TOPIC)}) took the process "
            f"down.\n{_log_tail(log_path)}"
        )
        assert ready_file.exists(), f"alive but never ready:\n{_log_tail(log_path)}"

        # --- the refused topic is reported by name ---------------------------
        refusal_line = (
            f"kafka_transport_topic_refused topic={_wire(REFUSED_CANONICAL_TOPIC)} "
        )
        assert refusal_line in output, (
            f"the forwarder did not report the refused topic by name:\n{_log_tail(log_path)}"
        )

        # --- an admitted inbound topic still delivers ------------------------
        envelope_id, value = _inbound_record(admitted_topic)
        brokers.cloud.produce(_wire(admitted_topic), [value])
        deadline = time.monotonic() + DELIVERY_DEADLINE_SECONDS
        delivered: dict[str, object] | None = None
        while delivered is None and time.monotonic() < deadline:
            assert process.poll() is None, (
                f"the forwarder exited while delivering:\n{_log_tail(log_path)}"
            )
            for record in _read_lane_topic(brokers.lane, admitted_topic):
                if record.get("envelope_id") == envelope_id:
                    delivered = record
                    break
            else:
                time.sleep(2)
        assert delivered is not None, (
            f"the record on admitted inbound topic {admitted_topic} did not reach "
            f"the lane broker in {DELIVERY_DEADLINE_SECONDS:.0f}s:\n{_log_tail(log_path)}"
        )
        metadata = delivered.get("metadata")
        assert isinstance(metadata, dict)
        tags = metadata.get("tags")
        assert isinstance(tags, dict)
        assert tags.get("gateway_direction") == "cloud-to-local"
        assert tags.get("gateway_canonical_topic") == admitted_topic
        assert process.poll() is None, f"the forwarder exited:\n{_log_tail(log_path)}"
    finally:
        _stop(process)
