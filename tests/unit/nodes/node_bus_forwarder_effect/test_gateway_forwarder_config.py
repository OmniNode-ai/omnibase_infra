# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_bus_forwarder_effect.models import (
    ModelGatewayCloudBusConfig,
    ModelGatewayForwarderConfig,
    ModelGatewayMirrorTopics,
    ModelGatewayTenantIdentity,
)


def _cloud_bus() -> ModelGatewayCloudBusConfig:
    return ModelGatewayCloudBusConfig(
        broker_provider_id="redpanda-dogfood",
        cloud_broker_ref="gateway.cloud.kafka.broker",
        cloud_auth_ref="gateway.cloud.kafka.oauth",
        acl_provisioner_ref="gateway.cloud.kafka.authorization",
        client_id_ref="gateway.cloud.kafka.oauth.client_id",
        client_secret_api_key_ref="infisical://gateway/redpanda-events",
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
            principal_id="tenant:11111111-1111-1111-1111-111111111111",
        )


def test_config_requires_silence_window_above_heartbeat() -> None:
    with pytest.raises(ValidationError, match="max_silence_window_seconds"):
        ModelGatewayForwarderConfig(
            tenant_identity=ModelGatewayTenantIdentity(
                tenant_id=UUID("11111111-1111-1111-1111-111111111111"),
                tenant_slug="acme",
                principal_id="tenant:11111111-1111-1111-1111-111111111111",
            ),
            cloud_bus=_cloud_bus(),
            local_transport_flavor="containerized",
            mirror_topics=ModelGatewayMirrorTopics(
                inbound=("onex.cmd.omnibase-infra.delegation-inference-request.v1",),
                outbound=("onex.evt.omnibase-infra.inference-response.v1",),
            ),
            heartbeat_interval_seconds=60,
            max_silence_window_seconds=60,
        )


def test_cloud_bus_config_rejects_ambient_kafka_env_refs() -> None:
    with pytest.raises(ValidationError, match=r"KAFKA_\* env"):
        ModelGatewayCloudBusConfig(
            broker_provider_id="redpanda-dogfood",
            cloud_broker_ref="KAFKA_BOOTSTRAP_SERVERS",
            cloud_auth_ref="gateway.cloud.kafka.oauth",
            acl_provisioner_ref="gateway.cloud.kafka.authorization",
            client_id_ref="gateway.cloud.kafka.oauth.client_id",
            client_secret_api_key_ref="infisical://gateway/redpanda-events",
        )


def test_cloud_bus_config_rejects_literal_bootstrap_server_field() -> None:
    with pytest.raises(ValidationError, match="bootstrap_servers"):
        ModelGatewayCloudBusConfig(
            broker_provider_id="redpanda-dogfood",
            cloud_broker_ref="gateway.cloud.kafka.broker",
            cloud_auth_ref="gateway.cloud.kafka.oauth",
            acl_provisioner_ref="gateway.cloud.kafka.authorization",
            client_id_ref="gateway.cloud.kafka.oauth.client_id",
            client_secret_api_key_ref="infisical://gateway/redpanda-events",
            bootstrap_servers=("kafka.omninode.ai:9093",),
        )
