# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayCanaryConfig,
    ModelGatewayCloudBusConfig,
    ModelGatewayForwarderConfig,
    ModelGatewayMirrorTopics,
    ModelGatewayTenantIdentity,
)

BROKER_PROVIDER_ID = UUID("22222222-2222-2222-2222-222222222222")
PRINCIPAL_ID = "t-33333333333333333333333333333333"

CONTRACT_PATH = (
    Path(__file__).parents[4]
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_bus_forwarder_effect"
    / "contract.yaml"
)

# OMN-16204: operator OD-9 ruling 2026-08-18 ~12:40Z allows EXACTLY this bare
# session-lifecycle pair (session id + timestamps, content-free) to cross to
# cloud via node_bus_forwarder_effect's mirror_topics.outbound.
OD9_ALLOWED_SESSION_LIFECYCLE_TOPICS = (
    "onex.evt.omniclaude.session-started.v1",
    "onex.evt.omniclaude.session-ended.v1",
)

# Content-bearing omniclaude topics that OD-9 explicitly keeps DENIED pending
# the scrubbing/projection-transform layer OMN-14323 still owns.
OD9_DENIED_OMNICLAUDE_TOPICS = (
    "onex.evt.omniclaude.prompt-submitted.v1",
    "onex.evt.omniclaude.tool-executed.v1",
    "onex.evt.omniclaude.skill-started.v1",
    "onex.evt.omniclaude.skill-completed.v1",
    "onex.evt.omniclaude.tool-output-captured.v1",
)


@pytest.mark.parametrize("topic", OD9_ALLOWED_SESSION_LIFECYCLE_TOPICS)
def test_mirror_topics_model_accepts_od9_session_lifecycle_topic(topic: str) -> None:
    """Per-topic proof: each OD-9-allowed session-lifecycle topic independently
    passes ``ModelGatewayMirrorTopics`` shape validation as an outbound entry."""
    mirror_topics = ModelGatewayMirrorTopics(
        inbound=("onex.cmd.omnibase-infra.delegation-request.v1",),
        outbound=("onex.evt.omnibase-infra.inference-response.v1", topic),
    )
    assert topic in mirror_topics.outbound


@pytest.mark.parametrize("topic", OD9_ALLOWED_SESSION_LIFECYCLE_TOPICS)
def test_contract_declares_od9_session_lifecycle_topic_in_outbound(
    topic: str,
) -> None:
    """Per-topic proof against the REAL contract.yaml on disk: each OD-9
    session-lifecycle topic is declared exactly once under
    ``config.gateway_forwarder.mirror_topics.outbound``."""
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    outbound = contract["config"]["gateway_forwarder"]["mirror_topics"]["outbound"]
    assert outbound.count(topic) == 1


@pytest.mark.parametrize("topic", OD9_DENIED_OMNICLAUDE_TOPICS)
def test_contract_does_not_widen_beyond_od9_session_lifecycle_pair(
    topic: str,
) -> None:
    """OMN-16204 scope guard: no other omniclaude topic (prompt/tool/skill
    content) may be added to mirror_topics.outbound by this change -- those
    stay DENIED pending OMN-14323's scrubbing layer, per OD-9."""
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    outbound = contract["config"]["gateway_forwarder"]["mirror_topics"]["outbound"]
    assert topic not in outbound


def test_contract_outbound_gains_exactly_two_new_topics() -> None:
    """Falsifiable count check: outbound grew from the pre-OMN-16204 baseline
    of 6 topics to exactly 8 -- proving nothing beyond the two OD-9 topics
    was added."""
    contract = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    outbound = contract["config"]["gateway_forwarder"]["mirror_topics"]["outbound"]
    assert len(outbound) == 8
    for topic in OD9_ALLOWED_SESSION_LIFECYCLE_TOPICS:
        assert topic in outbound


def _cloud_bus() -> ModelGatewayCloudBusConfig:
    return ModelGatewayCloudBusConfig(
        broker_provider_id=BROKER_PROVIDER_ID,
        cloud_broker_ref="gateway.cloud.kafka.broker",
        cloud_auth_ref="gateway.cloud.kafka.oauth",
        acl_provisioner_ref="gateway.cloud.kafka.authorization",
        client_id_ref="gateway.cloud.kafka.oauth.client_id",
        client_secret_api_key_ref="infisical://gateway/redpanda-events",
    )


def _canary() -> ModelGatewayCanaryConfig:
    return ModelGatewayCanaryConfig(
        topic="onex.evt.omnibase-infra.gateway-canary.v1",
        cadence_seconds=30,
        produce_deadline_seconds=8,
        readback_deadline_seconds=12,
    )


def test_config_rejects_tenant_prefixed_contract_topic() -> None:
    with pytest.raises(ValidationError, match="tenant prefix"):
        ModelGatewayMirrorTopics(
            inbound=(
                "tenant-acme.onex.cmd.omnibase-infra.delegation-inference-request.v1",
            ),
            outbound=("onex.evt.omnibase-infra.inference-response.v1",),
        )


def test_config_rejects_reserved_tenant_slug() -> None:
    with pytest.raises(ValidationError, match="reserved"):
        ModelGatewayTenantIdentity(
            tenant_id=UUID("11111111-1111-1111-1111-111111111111"),
            tenant_slug="system",
            principal_id=PRINCIPAL_ID,
        )


def test_config_requires_silence_window_above_heartbeat() -> None:
    with pytest.raises(ValidationError, match="max_silence_window_seconds"):
        ModelGatewayForwarderConfig(
            tenant_identity=ModelGatewayTenantIdentity(
                tenant_id=UUID("11111111-1111-1111-1111-111111111111"),
                tenant_slug="acme",
                principal_id=PRINCIPAL_ID,
            ),
            cloud_bus=_cloud_bus(),
            canary=_canary(),
            local_transport_flavor="containerized",
            dedupe_store_path=Path.cwd() / "gateway-test.sqlite3",
            mirror_topics=ModelGatewayMirrorTopics(
                inbound=("onex.cmd.omnibase-infra.delegation-inference-request.v1",),
                outbound=("onex.evt.omnibase-infra.inference-response.v1",),
            ),
            heartbeat_interval_seconds=60,
            max_silence_window_seconds=60,
        )


def test_config_requires_retry_max_at_least_initial_delay() -> None:
    with pytest.raises(ValidationError, match="forward_retry_max_seconds"):
        ModelGatewayForwarderConfig(
            tenant_identity=ModelGatewayTenantIdentity(
                tenant_id=UUID("11111111-1111-1111-1111-111111111111"),
                tenant_slug="acme",
                principal_id=PRINCIPAL_ID,
            ),
            cloud_bus=_cloud_bus(),
            canary=_canary(),
            local_transport_flavor="containerized",
            dedupe_store_path=Path.cwd() / "gateway-test.sqlite3",
            mirror_topics=ModelGatewayMirrorTopics(
                inbound=("onex.cmd.omnibase-infra.delegation-request.v1",),
                outbound=("onex.evt.omnibase-infra.inference-response.v1",),
            ),
            forward_retry_initial_seconds=10,
            forward_retry_max_seconds=5,
        )


def test_config_requires_reconnect_backoff_max_at_least_initial_delay() -> None:
    with pytest.raises(ValidationError, match="reconnect_backoff_max_seconds"):
        ModelGatewayForwarderConfig(
            tenant_identity=ModelGatewayTenantIdentity(
                tenant_id=UUID("11111111-1111-1111-1111-111111111111"),
                tenant_slug="acme",
                principal_id=PRINCIPAL_ID,
            ),
            cloud_bus=_cloud_bus(),
            canary=_canary(),
            local_transport_flavor="containerized",
            dedupe_store_path=Path.cwd() / "gateway-test.sqlite3",
            mirror_topics=ModelGatewayMirrorTopics(
                inbound=("onex.cmd.omnibase-infra.delegation-request.v1",),
                outbound=("onex.evt.omnibase-infra.inference-response.v1",),
            ),
            reconnect_backoff_initial_seconds=10,
            reconnect_backoff_max_seconds=5,
        )


def test_config_reconnect_defaults_match_contract() -> None:
    config = ModelGatewayForwarderConfig(
        tenant_identity=ModelGatewayTenantIdentity(
            tenant_id=UUID("11111111-1111-1111-1111-111111111111"),
            tenant_slug="acme",
            principal_id=PRINCIPAL_ID,
        ),
        cloud_bus=_cloud_bus(),
        canary=_canary(),
        local_transport_flavor="containerized",
        dedupe_store_path=Path.cwd() / "gateway-test.sqlite3",
        mirror_topics=ModelGatewayMirrorTopics(
            inbound=("onex.cmd.omnibase-infra.delegation-request.v1",),
            outbound=("onex.evt.omnibase-infra.inference-response.v1",),
        ),
    )
    assert config.reconnect_backoff_initial_seconds == 1.0
    assert config.reconnect_backoff_max_seconds == 30.0
    assert config.reconnect_backoff_jitter_seconds == 0.5
    assert config.degraded_after_seconds == 60


def test_config_requires_absolute_durable_store_path() -> None:
    with pytest.raises(ValidationError, match="dedupe_store_path must be absolute"):
        ModelGatewayForwarderConfig(
            tenant_identity=ModelGatewayTenantIdentity(
                tenant_id=UUID("11111111-1111-1111-1111-111111111111"),
                tenant_slug="acme",
                principal_id=PRINCIPAL_ID,
            ),
            cloud_bus=_cloud_bus(),
            canary=_canary(),
            local_transport_flavor="containerized",
            dedupe_store_path=Path("relative/delivery.sqlite3"),
            mirror_topics=ModelGatewayMirrorTopics(
                inbound=("onex.cmd.omnibase-infra.delegation-request.v1",),
                outbound=("onex.evt.omnibase-infra.inference-response.v1",),
            ),
        )


def test_config_enforces_twenty_four_hour_dedupe_floor() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 24"):
        ModelGatewayForwarderConfig(
            tenant_identity=ModelGatewayTenantIdentity(
                tenant_id=UUID("11111111-1111-1111-1111-111111111111"),
                tenant_slug="acme",
                principal_id=PRINCIPAL_ID,
            ),
            cloud_bus=_cloud_bus(),
            canary=_canary(),
            local_transport_flavor="containerized",
            dedupe_store_path=Path.cwd() / "gateway-test.sqlite3",
            dedupe_retention_hours=23,
            mirror_topics=ModelGatewayMirrorTopics(
                inbound=("onex.cmd.omnibase-infra.delegation-request.v1",),
                outbound=("onex.evt.omnibase-infra.inference-response.v1",),
            ),
        )


def test_cloud_bus_config_rejects_ambient_kafka_env_refs() -> None:
    with pytest.raises(ValidationError, match=r"KAFKA_\* env"):
        ModelGatewayCloudBusConfig(
            broker_provider_id=BROKER_PROVIDER_ID,
            cloud_broker_ref="KAFKA_BOOTSTRAP_SERVERS",
            cloud_auth_ref="gateway.cloud.kafka.oauth",
            acl_provisioner_ref="gateway.cloud.kafka.authorization",
            client_id_ref="gateway.cloud.kafka.oauth.client_id",
            client_secret_api_key_ref="infisical://gateway/redpanda-events",
        )


def test_cloud_bus_config_rejects_literal_bootstrap_server_field() -> None:
    with pytest.raises(ValidationError, match="bootstrap_servers"):
        ModelGatewayCloudBusConfig(
            broker_provider_id=BROKER_PROVIDER_ID,
            cloud_broker_ref="gateway.cloud.kafka.broker",
            cloud_auth_ref="gateway.cloud.kafka.oauth",
            acl_provisioner_ref="gateway.cloud.kafka.authorization",
            client_id_ref="gateway.cloud.kafka.oauth.client_id",
            client_secret_api_key_ref="infisical://gateway/redpanda-events",
            bootstrap_servers=("kafka.omninode.ai:9093",),
        )
