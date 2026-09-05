# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16459: contract-declared HTTPS ingest leg for the forwarder's OUTBOUND leg.

Operator ruling 2026-08-24 (verbatim, via team-lead): *"the gateway on the cloud
should be configurable to point at whatever env we want. we can use the forwarder
to accelerate moving our work to the cloud, but it should be replaced with the
https doors as soon as possible."*

Operator scope ruling 2026-08-30 (verbatim): *"the cloud leg shouldn't need
anything new. all we need is one ingress for all calls right?"* -- ONE
authenticated batch-ingest route on the existing cloud gateway, idempotent on the
content-addressed envelope id, no new service and no per-event-class endpoint.

What these tests pin:

* the node CONTRACT is the sole authority for the ingest refs, batching and
  idempotency key -- the resolved deployment YAML may not carry topic/URL
  literals of its own (same authority split ``mirror_topics`` and
  ``cloud_broker_ref`` already use);
* the auth material is a REFERENCE, never a value in config;
* the leg is fail-closed -- an unresolvable URL ref, a plaintext URL, or an
  ingest door pointed at the broker is refused at load, not at first publish;
* the HTTPS leg replaces the OUTBOUND publish boundary ONLY. The inbound leg is
  still a Kafka pull from the cloud broker, so this ticket alone does not retire
  the OMN-16449 bastion (recorded as an explicit assertion so the AC5 claim
  cannot be made by accident);
* this ticket does not widen ``mirror_topics.outbound`` -- that is OMN-16979,
  gated behind OMN-17209 / OMN-16019.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

import httpx
import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayCanaryConfig,
    ModelGatewayCloudBusConfig,
    ModelGatewayForwarderConfig,
    ModelGatewayForwarderRuntimeConfig,
    ModelGatewayHttpsIngestConfig,
    ModelGatewayMirrorTopics,
    ModelGatewayTenantIdentity,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_forwarder import (
    ProtocolGatewayPublisher,
)
from omnibase_infra.runtime import gateway_forwarder
from omnibase_infra.runtime.gateway_forwarder import TransportGatewayHttpsIngest

REPO_ROOT = Path(__file__).parents[4]
CONTRACT_PATH = (
    REPO_ROOT
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_bus_forwarder_effect"
    / "contract.yaml"
)
RESOLVED_DEPLOYMENT_PATH = REPO_ROOT / "docker" / "gateway" / "beta-gateway-canary.yaml"

INGEST_URL_REF = "gateway.cloud.https.ingest_url"
INGEST_AUTH_REF = "gateway.cloud.https.gateway_token"


def _contract_https_ingest_block() -> dict[str, object]:
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    return dict(contract["config"]["gateway_forwarder"]["https_ingest"])


def _https_ingest_config(**overrides: object) -> ModelGatewayHttpsIngestConfig:
    base: dict[str, object] = {
        "ingest_url": "https://dev.api.omninode.ai/v1/gateway/ingest",
        "ingest_url_ref": INGEST_URL_REF,
        "ingest_auth_ref": INGEST_AUTH_REF,
        "idempotency_key": "envelope_id",
        "max_batch_records": 100,
        "request_timeout_seconds": 15.0,
        "retry_initial_seconds": 1.0,
        "retry_max_seconds": 30.0,
    }
    base.update(overrides)
    return ModelGatewayHttpsIngestConfig.model_validate(base)


def _forwarder_config(
    https_ingest: ModelGatewayHttpsIngestConfig | None = None,
) -> ModelGatewayForwarderConfig:
    return ModelGatewayForwarderConfig(
        tenant_identity=ModelGatewayTenantIdentity(
            tenant_id=UUID("79afa726-3852-464f-b7a4-d4b8b9c75ee7"),
            tenant_slug="beta-gateway-canary-79afa7263852",
            principal_id="t-79afa7263852464fb7a4d4b8b9c75ee7",
        ),
        cloud_bus=ModelGatewayCloudBusConfig(
            broker_provider_id=UUID("22222222-2222-2222-2222-222222222222"),
            cloud_broker_ref="gateway.cloud.kafka.broker",
            cloud_auth_ref="gateway.cloud.kafka.msk_iam",
            acl_provisioner_ref="gateway.cloud.kafka.authorization",
            msk_region_ref="gateway.cloud.kafka.msk_region",
            sasl_mechanism="AWS_MSK_IAM",
        ),
        local_transport_flavor="containerized",
        dedupe_store_path=Path("/app/data/gateway/delivery.sqlite3"),
        mirror_topics=ModelGatewayMirrorTopics(
            inbound=("onex.cmd.omnibase-infra.delegation-request.v1",),
            outbound=(
                "onex.evt.omnibase-infra.delegation-completed.v1",
                "onex.evt.omnibase-infra.gateway-heartbeat.v1",
            ),
        ),
        canary=ModelGatewayCanaryConfig(
            topic="onex.evt.omnibase-infra.gateway-canary.v1",
            cadence_seconds=30,
            produce_deadline_seconds=15,
            readback_deadline_seconds=12,
        ),
        https_ingest=https_ingest,
    )


def _runtime_config(
    https_ingest: ModelGatewayHttpsIngestConfig | None = None,
) -> ModelGatewayForwarderRuntimeConfig:
    return ModelGatewayForwarderRuntimeConfig(
        forwarder=_forwarder_config(https_ingest),
        local_bus=ModelKafkaEventBusConfig(
            bootstrap_servers="omnibase-infra-redpanda:9092",
            environment="gateway-local",
            enable_auto_commit=False,
            auto_offset_reset="earliest",
        ),
        cloud_bus=ModelKafkaEventBusConfig(
            bootstrap_servers="b-1.example.kafka.amazonaws.com:9098",
            environment="gateway-cloud",
            security_protocol="SASL_SSL",
            sasl_mechanism="AWS_MSK_IAM",
            msk_region="us-east-1",
            enable_auto_commit=False,
            auto_offset_reset="earliest",
        ),
    )


# --------------------------------------------------------------------------
# CONTRACT is the authority
# --------------------------------------------------------------------------


def test_contract_declares_the_https_ingest_leg() -> None:
    """AC1: the ingest leg exists as a contract-declared block, not as code."""
    block = _contract_https_ingest_block()
    assert block["ingest_url_ref"] == INGEST_URL_REF
    assert block["ingest_auth_ref"] == INGEST_AUTH_REF


def test_contract_ingest_is_idempotent_on_the_content_addressed_envelope_id() -> None:
    """2026-08-30 ruling clause 2: idempotency belongs on the route, keyed on the
    content-addressed envelope id -- not on a dedupe service and not on the sink."""
    assert _contract_https_ingest_block()["idempotency_key"] == "envelope_id"


def test_contract_declares_one_batch_route_not_a_route_per_event_class() -> None:
    """2026-08-30 ruling clause 1: batch, one route. A single ``ingest_url_ref``
    with a batch bound -- never a mapping of topic to endpoint."""
    block = _contract_https_ingest_block()
    assert isinstance(block["max_batch_records"], int)
    assert block["max_batch_records"] >= 1
    assert not any(key.endswith("_url_refs") for key in block)


def test_contract_carries_no_ingest_url_or_token_literal() -> None:
    """The contract names refs. Neither a URL nor a credential value is a literal
    anywhere in the contract's ingest block."""
    rendered = yaml.safe_dump(_contract_https_ingest_block())
    assert "https://" not in rendered
    assert "http://" not in rendered


def test_the_live_canary_deployment_has_not_opted_in_yet() -> None:
    """The leg ships DARK, and that is deliberate.

    ``docker/gateway/beta-gateway-canary.yaml`` is the config mounted into the
    running ``omninode-gateway-forwarder`` on .201. Opting it in here would take
    effect on that container's next restart -- before the cloud gateway's ingest
    route exists and before the credential reference resolves -- and the loader
    is fail-closed, so the live forwarder would refuse to start. Two things must
    land before this line is added: the single ingest route on the existing
    cloud gateway, and a resolvable ``gateway.cloud.https.gateway_token``.

    Until then every deployment keeps the direct-MSK Kafka outbound leg with no
    behavioural change at all, which is also why this PR cannot be read as
    satisfying OMN-16459's live-proof AC."""
    resolved = yaml.safe_load(RESOLVED_DEPLOYMENT_PATH.read_text(encoding="utf-8"))
    forwarder = resolved["forwarder"]
    assert "https_ingest_set" not in forwarder
    assert "https_ingest" not in forwarder


def test_resolved_deployment_opts_in_by_named_set_not_by_literals(
    tmp_path: Path,
) -> None:
    """Same authority split ``mirror_topic_set`` uses: a resolved deployment
    that DOES opt in names the contract by set, and restates none of its
    values -- no URL, no credential, no batch bound, no idempotency key."""
    config_path, _ = _resolved_yaml_with_https(tmp_path)
    forwarder = yaml.safe_load(config_path.read_text(encoding="utf-8"))["forwarder"]
    assert forwarder["https_ingest_set"] == "node_bus_forwarder_effect"
    assert "https_ingest" not in forwarder
    rendered = yaml.safe_dump(forwarder)
    assert "https://" not in rendered
    assert "ingest_url" not in rendered


def test_this_ticket_does_not_widen_the_cloud_mirror_set() -> None:
    """OMN-16979 owns the widening; changing the TRANSPORT must not smuggle in
    new event classes.

    OMN-16979 has since landed, so the two hook classes are now present. The
    guard is preserved in its still-falsifiable form: whatever is widened must
    be governed by `egress_redaction`. A transport change that added an
    UNGOVERNED class would still fail here.
    """
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    forwarder = contract["config"]["gateway_forwarder"]
    outbound = set(forwarder["mirror_topics"]["outbound"])
    governed = set(forwarder.get("egress_redaction", {}).get("governed_topics", ()))
    content_bearing = {
        "onex.evt.omniclaude.tool-executed.v1",
        "onex.evt.omniclaude.prompt-submitted.v1",
        "onex.evt.omniclaude.tool-output-captured.v1",
        "onex.evt.omniclaude.skill-started.v1",
        "onex.evt.omniclaude.skill-completed.v1",
    }
    assert (outbound & content_bearing) <= governed


# --------------------------------------------------------------------------
# MODEL fails closed
# --------------------------------------------------------------------------


def test_ingest_config_rejects_a_plaintext_url() -> None:
    """The leg is the HTTPS door. A cleartext door is refused at load."""
    with pytest.raises(ValidationError, match="https"):
        _https_ingest_config(ingest_url="http://dev.api.omninode.ai/v1/gateway/ingest")


def test_ingest_config_rejects_an_auth_value_in_place_of_a_reference() -> None:
    """``ingest_auth_ref`` is a reference. Anything that is not a dotted ref --
    a bearer token, a base64 blob -- is refused so a credential cannot be pasted
    into a config file."""
    with pytest.raises(ValidationError, match="reference"):
        _https_ingest_config(ingest_auth_ref="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9")


def test_ingest_config_rejects_a_non_envelope_id_idempotency_key() -> None:
    with pytest.raises(ValidationError):
        _https_ingest_config(idempotency_key="topic")


def test_ingest_config_rejects_a_zero_batch_bound() -> None:
    with pytest.raises(ValidationError):
        _https_ingest_config(max_batch_records=0)


def test_ingest_config_requires_retry_max_at_or_above_retry_initial() -> None:
    with pytest.raises(ValidationError, match="retry_max_seconds"):
        _https_ingest_config(retry_initial_seconds=30.0, retry_max_seconds=1.0)


# --------------------------------------------------------------------------
# RUNTIME CONFIG fails closed
# --------------------------------------------------------------------------


def test_runtime_config_accepts_a_forwarder_with_no_https_leg() -> None:
    """The HTTPS leg is opt-in. Every existing deployment that has not opted in
    keeps the Kafka outbound leg with no change."""
    assert _runtime_config().forwarder.https_ingest is None


def test_runtime_config_rejects_an_ingest_door_pointed_at_the_cloud_broker() -> None:
    """The HTTPS door is not the broker. A URL whose host is the resolved cloud
    bootstrap host means the operator wired the ingest ref to the broker ref."""
    with pytest.raises(ValueError, match="must not resolve to the cloud broker"):
        _runtime_config(
            _https_ingest_config(
                ingest_url="https://b-1.example.kafka.amazonaws.com/v1/gateway/ingest"
            )
        )


def test_runtime_config_keeps_the_inbound_kafka_leg_when_https_is_declared() -> None:
    """OMN-16459 replaces the OUTBOUND leg only. ``mirror_topics.inbound`` is
    still pulled from the cloud broker, so declaring the HTTPS leg must NOT be
    readable as 'the bastion can now be deleted' (ticket AC5 is not satisfied by
    this leg alone)."""
    config = _runtime_config(_https_ingest_config())
    assert config.forwarder.https_ingest is not None
    assert config.forwarder.mirror_topics.inbound
    assert config.cloud_bus.bootstrap_servers


def test_an_inbound_free_forwarder_is_unrepresentable_so_no_guard_is_needed() -> None:
    """A stronger fact than a runtime-config guard: ``ModelGatewayMirrorTopics``
    already declares ``inbound`` with ``min_length=1``, so a forwarder that has
    dropped its inbound Kafka pull cannot be CONSTRUCTED at all.

    This is recorded as an assertion because the first revision of this lane
    added a fail-closed validator for that state, and the pre-existing model
    made it unreachable -- so the validator was deleted rather than kept as
    decoration. The consequence for OMN-16459 is unchanged and is the point:
    declaring the HTTPS leg cannot be read as 'the OMN-16449 bastion is now
    deletable', because every valid config still carries an inbound set that is
    consumed from the cloud broker."""
    with pytest.raises(ValidationError, match="at least 1 item"):
        ModelGatewayMirrorTopics(
            inbound=(),
            outbound=("onex.evt.omnibase-infra.gateway-heartbeat.v1",),
        )


# --------------------------------------------------------------------------
# MATERIALIZATION from the node contract + the operator-supplied ref map
# --------------------------------------------------------------------------


def _resolved_yaml_with_https(tmp_path: Path) -> tuple[Path, Path]:
    resolved = yaml.safe_load(RESOLVED_DEPLOYMENT_PATH.read_text(encoding="utf-8"))
    resolved["forwarder"]["https_ingest_set"] = "node_bus_forwarder_effect"
    config_path = tmp_path / "gateway-forwarder.yaml"
    config_path.write_text(yaml.safe_dump(resolved), encoding="utf-8")
    ref_map_path = tmp_path / "broker-ref-map.yaml"
    ref_map_path.write_text(
        yaml.safe_dump(
            {
                "gateway.cloud.kafka.broker": "b-1.example.kafka.amazonaws.com:9098",
                INGEST_URL_REF: "https://dev.api.omninode.ai/v1/gateway/ingest",
            }
        ),
        encoding="utf-8",
    )
    return config_path, ref_map_path


def test_loader_resolves_the_ingest_url_from_the_operator_supplied_ref_map(
    tmp_path: Path,
) -> None:
    """No new mount and no new env var: the ingest URL resolves from the SAME
    operator-supplied ref map the cloud broker ref already resolves from."""
    config_path, ref_map_path = _resolved_yaml_with_https(tmp_path)
    config = gateway_forwarder.load_gateway_forwarder_runtime_config(
        config_path,
        contract_path=CONTRACT_PATH,
        broker_ref_map_path=ref_map_path,
    )
    assert config.forwarder.https_ingest is not None
    assert (
        config.forwarder.https_ingest.ingest_url
        == "https://dev.api.omninode.ai/v1/gateway/ingest"
    )
    assert config.forwarder.https_ingest.ingest_auth_ref == INGEST_AUTH_REF


def test_loader_fails_closed_when_the_ingest_url_ref_is_unresolvable(
    tmp_path: Path,
) -> None:
    """An opted-in deployment whose ref map has no ingest entry must refuse to
    start, not silently fall back to the direct-MSK leg."""
    config_path, ref_map_path = _resolved_yaml_with_https(tmp_path)
    ref_map_path.write_text(
        yaml.safe_dump(
            {"gateway.cloud.kafka.broker": "b-1.example.kafka.amazonaws.com:9098"}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=INGEST_URL_REF):
        gateway_forwarder.load_gateway_forwarder_runtime_config(
            config_path,
            contract_path=CONTRACT_PATH,
            broker_ref_map_path=ref_map_path,
        )


def test_loader_refuses_an_inline_https_ingest_block(tmp_path: Path) -> None:
    """The resolved deployment may not redeclare the contract's ingest values."""
    resolved = yaml.safe_load(RESOLVED_DEPLOYMENT_PATH.read_text(encoding="utf-8"))
    resolved["forwarder"]["https_ingest"] = {"ingest_url_ref": INGEST_URL_REF}
    config_path, ref_map_path = _resolved_yaml_with_https(tmp_path)
    config_path.write_text(yaml.safe_dump(resolved), encoding="utf-8")
    with pytest.raises(ValueError, match="https_ingest_set"):
        gateway_forwarder.load_gateway_forwarder_runtime_config(
            config_path,
            contract_path=CONTRACT_PATH,
            broker_ref_map_path=ref_map_path,
        )


def test_a_round_tripped_null_https_ingest_is_not_treated_as_a_redeclaration(
    tmp_path: Path,
) -> None:
    """``https_ingest`` is optional, so any config round-tripped through
    ``model_dump()`` emits an explicit ``https_ingest: null``. Refusing on key
    PRESENCE rather than on a POPULATED block breaks every such round trip --
    the exact defect the OMN-17034 lane hit and fixed."""
    resolved = yaml.safe_load(RESOLVED_DEPLOYMENT_PATH.read_text(encoding="utf-8"))
    resolved["forwarder"]["https_ingest_set"] = "node_bus_forwarder_effect"
    resolved["forwarder"]["https_ingest"] = None
    config_path, ref_map_path = _resolved_yaml_with_https(tmp_path)
    config_path.write_text(yaml.safe_dump(resolved), encoding="utf-8")
    config = gateway_forwarder.load_gateway_forwarder_runtime_config(
        config_path,
        contract_path=CONTRACT_PATH,
        broker_ref_map_path=ref_map_path,
    )
    assert config.forwarder.https_ingest is not None


def test_loader_leaves_the_kafka_outbound_leg_alone_when_the_set_is_absent(
    tmp_path: Path,
) -> None:
    """Backwards path: a deployment that has not opted in resolves with no HTTPS
    leg and an unchanged cloud broker leg."""
    resolved = yaml.safe_load(RESOLVED_DEPLOYMENT_PATH.read_text(encoding="utf-8"))
    resolved["forwarder"].pop("https_ingest_set", None)
    config_path, ref_map_path = _resolved_yaml_with_https(tmp_path)
    resolved["forwarder"].pop("https_ingest_set", None)
    config_path.write_text(yaml.safe_dump(resolved), encoding="utf-8")
    config = gateway_forwarder.load_gateway_forwarder_runtime_config(
        config_path,
        contract_path=CONTRACT_PATH,
        broker_ref_map_path=ref_map_path,
    )
    assert config.forwarder.https_ingest is None
    assert config.cloud_bus.bootstrap_servers == "b-1.example.kafka.amazonaws.com:9098"


# --------------------------------------------------------------------------
# TRANSPORT: the publish boundary swap
# --------------------------------------------------------------------------


def _envelope_bytes(envelope_id: str) -> bytes:
    return json.dumps(
        {
            "envelope_id": envelope_id,
            "payload": {"hello": "world"},
        }
    ).encode("utf-8")


def _ingest_transport(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    config: ModelGatewayHttpsIngestConfig | None = None,
) -> TransportGatewayHttpsIngest:
    resolved = config if config is not None else _https_ingest_config()
    return TransportGatewayHttpsIngest(
        config=resolved,
        tenant_slug="beta-gateway-canary-79afa7263852",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        auth_token="test-gateway-token",
    )


def test_transport_satisfies_the_existing_publisher_boundary() -> None:
    """OMN-12912 protocol-ownership: this leg introduces NO new protocol. It
    satisfies the node's existing ``ProtocolGatewayPublisher`` structurally,
    exactly as ``TransportGatewayBus`` does for the Kafka leg."""
    transport = _ingest_transport(lambda request: httpx.Response(202))
    publisher: ProtocolGatewayPublisher = transport
    assert publisher is transport


@pytest.mark.asyncio
async def test_transport_posts_one_record_to_the_single_contract_declared_route() -> (
    None
):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(202, json={"accepted": 1})

    transport = _ingest_transport(handler)
    await transport.publish(
        topic="beta-gateway-canary-79afa7263852.onex.evt.omnibase-infra.gateway-heartbeat.v1",
        key=b"tenant-key",
        value=_envelope_bytes("11111111-1111-1111-1111-111111111111"),
    )
    assert len(seen) == 1
    assert str(seen[0].url) == "https://dev.api.omninode.ai/v1/gateway/ingest"
    body = json.loads(seen[0].content)
    assert len(body["records"]) == 1
    assert (
        body["records"][0]["topic"]
        == "beta-gateway-canary-79afa7263852.onex.evt.omnibase-infra.gateway-heartbeat.v1"
    )


@pytest.mark.asyncio
async def test_transport_preserves_the_record_bytes_exactly() -> None:
    """The cloud route republishes under the same topic names; a lossy transport
    would break deterministic replay of the mirrored record."""
    captured: dict[str, object] = {}
    original = _envelope_bytes("22222222-2222-2222-2222-222222222222")

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content)["records"][0])
        return httpx.Response(202)

    await _ingest_transport(handler).publish(
        topic="t.onex.evt.omnibase-infra.gateway-heartbeat.v1",
        key=b"tenant-key",
        value=original,
    )
    assert base64.b64decode(str(captured["value"])) == original
    assert base64.b64decode(str(captured["key"])) == b"tenant-key"


@pytest.mark.asyncio
async def test_transport_sends_the_content_addressed_envelope_id_as_idempotency_key() -> (
    None
):
    """2026-08-30 ruling clause 2 -- idempotency is asserted by the caller on the
    route, keyed on the content-addressed envelope id."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(202)

    await _ingest_transport(handler).publish(
        topic="t.onex.evt.omnibase-infra.gateway-heartbeat.v1",
        key=None,
        value=_envelope_bytes("33333333-3333-3333-3333-333333333333"),
    )
    assert seen[0].headers["idempotency-key"] == "33333333-3333-3333-3333-333333333333"


@pytest.mark.asyncio
async def test_transport_presents_the_forwarders_own_verified_identity() -> None:
    """AC1: authenticated with the forwarder's existing tenant credential. The
    token is supplied by reference resolution at the process boundary and is
    never read from this transport's own config."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(202)

    await _ingest_transport(handler).publish(
        topic="t.onex.evt.omnibase-infra.gateway-heartbeat.v1",
        key=None,
        value=_envelope_bytes("44444444-4444-4444-4444-444444444444"),
    )
    assert seen[0].headers["authorization"] == "Bearer test-gateway-token"


@pytest.mark.asyncio
async def test_transport_raises_a_retryable_error_on_a_server_side_failure() -> None:
    """A 5xx must surface as the forwarder's retryable class so the delivery node
    retains the source message and does NOT commit the offset -- the same
    at-least-once guarantee the direct-MSK leg gets from the broker."""
    transport = _ingest_transport(lambda request: httpx.Response(503))
    with pytest.raises(InfraUnavailableError):
        await transport.publish(
            topic="t.onex.evt.omnibase-infra.gateway-heartbeat.v1",
            key=None,
            value=_envelope_bytes("55555555-5555-5555-5555-555555555555"),
        )


@pytest.mark.asyncio
async def test_transport_raises_a_retryable_error_on_a_transport_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    with pytest.raises(InfraUnavailableError):
        await _ingest_transport(handler).publish(
            topic="t.onex.evt.omnibase-infra.gateway-heartbeat.v1",
            key=None,
            value=_envelope_bytes("66666666-6666-6666-6666-666666666666"),
        )


@pytest.mark.asyncio
async def test_transport_does_not_retry_forever_on_a_rejected_record() -> None:
    """A 4xx is the route refusing this record. Retrying it is an infinite loop
    against a permanent failure, so it must NOT be raised as the retryable class."""
    transport = _ingest_transport(lambda request: httpx.Response(400))
    with pytest.raises(Exception) as excinfo:
        await transport.publish(
            topic="t.onex.evt.omnibase-infra.gateway-heartbeat.v1",
            key=None,
            value=_envelope_bytes("77777777-7777-7777-7777-777777777777"),
        )
    assert not isinstance(excinfo.value, InfraUnavailableError)


@pytest.mark.asyncio
async def test_transport_refuses_a_record_with_no_envelope_id() -> None:
    """Fail closed: without the content-addressed id there is no idempotency key,
    so the route could not deduplicate a redelivery."""
    transport = _ingest_transport(lambda request: httpx.Response(202))
    with pytest.raises(ValueError, match="envelope_id"):
        await transport.publish(
            topic="t.onex.evt.omnibase-infra.gateway-heartbeat.v1",
            key=None,
            value=json.dumps({"payload": {}}).encode("utf-8"),
        )


# --------------------------------------------------------------------------
# PROCESS WIRING
# --------------------------------------------------------------------------


def test_outbound_publish_boundary_selection_is_contract_driven() -> None:
    """The process picks the outbound publish boundary from the resolved
    contract, not from an environment variable or a command-line flag."""
    assert (
        gateway_forwarder.select_outbound_publish_transport.__module__
        == "omnibase_infra.runtime.gateway_forwarder"
    )
    parser_help = gateway_forwarder._build_parser().format_help()
    assert "--https" not in parser_help
    assert "--ingest" not in parser_help


@pytest.mark.asyncio
async def test_selector_returns_the_kafka_leg_when_no_https_leg_is_declared() -> None:
    """A deployment that has not opted in never consults the secret store and
    keeps the direct-MSK outbound leg byte-unchanged."""
    consulted: list[str] = []

    async def _resolve(reference: str) -> str | None:
        consulted.append(reference)
        return "unused"

    kafka_bus = object()
    publisher, client = await gateway_forwarder.select_outbound_publish_transport(
        _runtime_config(),
        kafka_bus=kafka_bus,  # type: ignore[arg-type]
        resolve_secret=_resolve,
    )
    assert publisher is kafka_bus
    assert client is None
    assert consulted == []


@pytest.mark.asyncio
async def test_selector_fails_closed_when_the_credential_ref_does_not_resolve() -> None:
    """The whole point of ruling 39 (OMN-15692) is that the direct-MSK leg stops
    being used. A declared HTTPS leg whose credential is unresolvable must refuse
    to start, NOT quietly fall back to the leg being retired."""

    async def _resolve(_reference: str) -> str | None:
        return None

    with pytest.raises(ValueError, match=INGEST_AUTH_REF):
        await gateway_forwarder.select_outbound_publish_transport(
            _runtime_config(_https_ingest_config()),
            kafka_bus=object(),  # type: ignore[arg-type]
            resolve_secret=_resolve,
        )


@pytest.mark.asyncio
async def test_selector_builds_the_https_leg_from_the_resolved_credential() -> None:
    clients: list[httpx.AsyncClient] = []

    async def _resolve(reference: str) -> str | None:
        assert reference == INGEST_AUTH_REF
        return "resolved-token"

    def _factory() -> httpx.AsyncClient:
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(202))
        )
        clients.append(client)
        return client

    publisher, client = await gateway_forwarder.select_outbound_publish_transport(
        _runtime_config(_https_ingest_config()),
        kafka_bus=object(),  # type: ignore[arg-type]
        resolve_secret=_resolve,
        client_factory=_factory,
    )
    assert isinstance(publisher, TransportGatewayHttpsIngest)
    assert client is clients[0]
    await client.aclose()


def test_the_credential_reference_never_appears_as_a_value_in_the_repo() -> None:
    """AC1's 'no new credential type without justification' has a companion
    obligation: the credential itself is a reference everywhere in tracked
    files. The contract and the live resolved deployment carry the ref name and
    never a token."""
    for path in (CONTRACT_PATH, RESOLVED_DEPLOYMENT_PATH):
        text = path.read_text(encoding="utf-8")
        assert "Bearer " not in text
        assert "eyJ" not in text
