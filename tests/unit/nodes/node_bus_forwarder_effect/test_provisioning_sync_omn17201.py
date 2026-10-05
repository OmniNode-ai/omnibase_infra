# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Provisioning follows resolved wire bindings, never a second topic list.

The contract determines mirror membership; the transport builder resolves the
physical topics and groups. This pin compares that configuration with the
pure grant derivation, so editing a test-side literal list cannot mask a
missing grant. Topic additions are exercised by the two-version derivation
fixture in node_broker_grant_derive_compute.
"""

from pathlib import Path

import pytest

from omnibase_infra.handlers.handler_broker_grant_derive import (
    HandlerBrokerGrantDerive,
)
from omnibase_infra.nodes.node_bus_forwarder_effect.services.service_gateway_topic_transform import (
    prefix_topic,
)
from omnibase_infra.runtime.gateway_forwarder import (
    build_gateway_transports,
    load_gateway_forwarder_runtime_config,
)


@pytest.mark.unit
def test_cloud_provisioning_is_exactly_the_resolved_contract(
    tmp_path, monkeypatch
) -> None:
    """Every mirrored physical topic has exactly the configured operations."""
    monkeypatch.delenv("KAFKA_TOPIC_NAMESPACE", raising=False)
    root = Path(__file__).parents[4]
    credentials = tmp_path / "credentials.yaml"
    credentials.write_text(
        "lane.dev.kafka.scram:\n  username: fixture\n  password: fixture-not-a-secret\n"
    )
    config = load_gateway_forwarder_runtime_config(
        root / "docker/gateway/beta-gateway-canary.yaml",
        broker_ref_map_path=root
        / "tests/fixtures/gateway/beta-gateway-canary-broker-ref-map.yaml",
        lane_credential_map_path=credentials,
    )
    resolved, transports = build_gateway_transports(config)
    derived = HandlerBrokerGrantDerive().handle(resolved)
    cloud = config.forwarder.cloud_bus.cloud_broker_ref
    slug = config.forwarder.tenant_identity.tenant_slug
    inbound = {prefix_topic(slug, t) for t in config.forwarder.declared_inbound_topics}
    outbound = {
        prefix_topic(slug, t) for t in config.forwarder.declared_outbound_topics
    }
    grants = {
        (g.resource, g.operation)
        for g in derived.grants
        if g.broker == cloud and g.resource_type == "TOPIC"
    }
    assert grants == (
        {(topic, "READ") for topic in inbound}
        | {(topic, "WRITE") for topic in outbound}
        | {(topic, "DESCRIBE") for topic in inbound | outbound}
    )
    assert set(transports["cloud"]._physical_topics) == inbound
    group_grants = {
        (g.resource, g.operation)
        for g in derived.grants
        if g.broker == cloud and g.resource_type == "GROUP"
    }
    assert group_grants == {(transports["cloud"]._group, "READ")}
