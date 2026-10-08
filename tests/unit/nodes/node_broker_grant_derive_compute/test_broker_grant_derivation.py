# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Broker grants must follow the transport builders, including physical names."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import yaml
from aiokafka import AIOKafkaProducer

from omnibase_core.models.event_bus.model_bus_binding import ModelBusBinding
from omnibase_core.models.event_bus.model_bus_group_describe import (
    ModelBusGroupDescribe,
)
from omnibase_core.models.event_bus.model_resolved_bus_bindings import (
    ModelResolvedBusBindings,
)
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models import ModelEventHeaders
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.handlers.handler_broker_grant_derive import (
    HandlerBrokerGrantDerive,
)
from omnibase_infra.runtime import gateway_forwarder
from omnibase_infra.topics.topic_namespace import apply_topic_namespace

pytestmark = pytest.mark.unit
ROOT = Path(__file__).parents[4]
CONTRACT = ROOT / "src/omnibase_infra/nodes/node_bus_forwarder_effect/contract.yaml"


def shipped_config(tmp_path: Path, *, contract_path: Path = CONTRACT):
    credentials = tmp_path / "credentials.yaml"
    credentials.write_text(
        "lane.dev.kafka.scram:\n  username: fixture\n  password: fixture-not-a-secret\n"
    )
    return gateway_forwarder.load_gateway_forwarder_runtime_config(
        ROOT / "docker/gateway/beta-gateway-canary.yaml",
        contract_path=contract_path,
        broker_ref_map_path=ROOT
        / "tests/fixtures/gateway/beta-gateway-canary-broker-ref-map.yaml",
        lane_credential_map_path=credentials,
    )


def grant_keys(resolved):
    return {
        (g.broker, g.resource_type, g.resource, g.operation)
        for g in HandlerBrokerGrantDerive().handle(resolved).grants
    }


@pytest.mark.parametrize("namespace", ["", "isolated"])
def test_gateway_builder_bindings_cover_every_transport(
    tmp_path, monkeypatch, namespace
):
    monkeypatch.setenv("KAFKA_TOPIC_NAMESPACE", namespace)
    config = shipped_config(tmp_path)
    resolved, transports = gateway_forwarder.build_gateway_transports(config)
    grants = grant_keys(resolved)
    consumed = {
        (b.physical_topic, b.consumer_group)
        for b in resolved.bindings
        if b.direction == "consume"
    }
    built = {
        (topic, transport._group)
        for transport in transports.values()
        for topic in transport._physical_topics
    }
    assert built == consumed
    for binding in resolved.bindings:
        op = "READ" if binding.direction == "consume" else "WRITE"
        assert (binding.broker, "TOPIC", binding.physical_topic, op) in grants
        assert (binding.broker, "TOPIC", binding.physical_topic, "DESCRIBE") in grants
        if binding.consumer_group:
            assert (binding.broker, "GROUP", binding.consumer_group, "READ") in grants
    cloud = config.forwarder.cloud_bus.cloud_broker_ref
    slug = config.forwarder.tenant_identity.tenant_slug
    topic = "onex.cmd.github.webhook-delivery.v1"
    assert (
        cloud,
        "TOPIC",
        apply_topic_namespace(f"tenant-{slug}.{topic}"),
        "READ",
    ) in grants
    assert (
        cloud,
        "GROUP",
        f"tenant-{slug}-gateway-forwarder-inbound",
        "READ",
    ) in grants
    # Consuming alone does not grant administrative group inspection.
    assert not any(g[1] == "GROUP" and g[3] == "DESCRIBE" for g in grants)


@pytest.mark.parametrize("direction", ["inbound", "outbound"])
def test_contract_topic_addition_changes_exactly_its_grants(
    tmp_path, monkeypatch, direction
):
    monkeypatch.delenv("KAFKA_TOPIC_NAMESPACE", raising=False)
    before, _ = gateway_forwarder.build_gateway_transports(shipped_config(tmp_path))
    contract = yaml.safe_load(CONTRACT.read_text())
    added = "onex.evt.fixture.additional.v1"
    contract["config"]["gateway_forwarder"]["mirror_topics"][direction].append(added)
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(contract))
    config = shipped_config(tmp_path, contract_path=path)
    after, _ = gateway_forwarder.build_gateway_transports(config)
    cloud = config.forwarder.cloud_bus.cloud_broker_ref
    physical = f"tenant-{config.forwarder.tenant_identity.tenant_slug}.{added}"
    assert grant_keys(before) - grant_keys(after) == set()
    local_op, cloud_op = (
        ("WRITE", "READ") if direction == "inbound" else ("READ", "WRITE")
    )
    assert grant_keys(after) - grant_keys(before) == {
        ("local", "TOPIC", added, local_op),
        ("local", "TOPIC", added, "DESCRIBE"),
        (cloud, "TOPIC", physical, cloud_op),
        (cloud, "TOPIC", physical, "DESCRIBE"),
    }


def test_group_describe_is_exact_and_broker_scoped():
    resolved = ModelResolvedBusBindings(
        principal="inspector",
        bindings=(),
        described_groups=(
            ModelBusGroupDescribe(broker="cloud", group="observed"),
            ModelBusGroupDescribe(broker="local", group="other"),
        ),
    )
    assert grant_keys(resolved) == {
        ("cloud", "GROUP", "observed", "DESCRIBE"),
        ("local", "GROUP", "other", "DESCRIBE"),
    }


def test_grants_are_deduplicated_sorted_and_preserve_principal():
    a = ModelBusBinding(
        broker="cloud",
        physical_topic="fixture",
        direction="consume",
        consumer_group="worker",
    )
    b = ModelBusBinding(broker="cloud", physical_topic="fixture", direction="produce")
    handler = HandlerBrokerGrantDerive()
    result = handler.handle(
        ModelResolvedBusBindings(principal="tenant", bindings=(a, b))
    )
    assert result == handler.handle(
        ModelResolvedBusBindings(principal="tenant", bindings=(b, a))
    )
    assert result.principal == "tenant"
    assert len(result.grants) == 4


@pytest.mark.parametrize("namespace", ["", "isolated"])
@pytest.mark.parametrize("instance_id", [None, "worker"])
def test_runtime_consumer_builder_matches_derived_physical_binding(
    monkeypatch, namespace, instance_id
):
    monkeypatch.setenv("KAFKA_TOPIC_NAMESPACE", namespace)
    bus = EventBusKafka(
        ModelKafkaEventBusConfig(
            bootstrap_servers="fixture:9092", environment="dev", instance_id=instance_id
        )
    )
    topics = ("onex.evt.fixture.input.v1", "onex.cmd.fixture.request.v1")
    resolved = bus.resolve_bus_bindings(
        principal="runtime",
        subscribe_topics=(*topics, topics[0]),
        publish_topics=(),
        group_id="runtime",
    )
    calls = []

    def consumer(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer", consumer
    )
    for topic in topics:
        group = bus._resolve_effective_group_id(
            "runtime", topic, uuid4(), (topic, "runtime")
        )
        bus._build_consumer(topic, group, "instance", "earliest", group_id="runtime")
    built = {(args[0], kwargs["group_id"]) for args, kwargs in calls}
    assert built == {
        (binding.physical_topic, binding.consumer_group)
        for binding in resolved.bindings
    }
    assert grant_keys(resolved) == (
        {
            ("dev", "TOPIC", topic, op)
            for topic, _ in built
            for op in ("READ", "DESCRIBE")
        }
        | {("dev", "GROUP", group, "READ") for _, group in built}
    )


def test_https_egress_requires_no_cloud_kafka_write(tmp_path, monkeypatch):
    from omnibase_infra.nodes.node_bus_forwarder_effect.models.model_gateway_https_ingest_config import (
        ModelGatewayHttpsIngestConfig,
    )

    monkeypatch.delenv("KAFKA_TOPIC_NAMESPACE", raising=False)
    config = shipped_config(tmp_path)
    raw = config.model_dump(exclude_computed_fields=True)
    raw["forwarder"]["https_ingest"] = ModelGatewayHttpsIngestConfig(
        ingest_url="https://fixture.invalid/api/ingest",
        ingest_url_ref="gateway.cloud.https.ingest",
        ingest_auth_ref="gateway.cloud.https.auth",
        idempotency_key="event_id",
        max_batch_records=1,
        request_timeout_seconds=1,
        retry_initial_seconds=1,
        retry_max_seconds=2,
    )
    config = type(config).model_validate(raw)
    resolved, transports = gateway_forwarder.build_gateway_transports(config)
    cloud = config.forwarder.cloud_bus.cloud_broker_ref
    assert not any(g[0] == cloud and g[3] == "WRITE" for g in grant_keys(resolved))
    assert transports["cloud"]._topics


def test_runtime_publish_and_group_truncation_use_the_same_binding(monkeypatch):
    monkeypatch.setenv("KAFKA_TOPIC_NAMESPACE", "isolated")
    bus = EventBusKafka(
        ModelKafkaEventBusConfig(
            bootstrap_servers="fixture:9092", environment="dev", instance_id="worker"
        )
    )
    topic = "onex.evt.fixture.input.v1"
    resolved = bus.resolve_bus_bindings(
        principal="runtime",
        subscribe_topics=(topic,),
        publish_topics=(topic, topic),
        group_id="worker-" * 60,
    )
    consuming = next(b for b in resolved.bindings if b.direction == "consume")
    producing = next(b for b in resolved.bindings if b.direction == "produce")
    assert len(consuming.consumer_group) <= 255
    assert consuming.consumer_group == bus._resolve_effective_group_id(
        "worker-" * 60, topic, uuid4(), (topic, "worker-" * 60)
    )
    assert producing == bus.resolve_bus_binding(topic)
    assert len(resolved.bindings) == 2
    assert len(grant_keys(resolved)) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("namespace", ["", "isolated"])
async def test_runtime_published_wire_topics_match_exact_derived_grants(
    monkeypatch, namespace
):
    monkeypatch.setenv("KAFKA_TOPIC_NAMESPACE", namespace)
    bus = EventBusKafka(
        ModelKafkaEventBusConfig(bootstrap_servers="fixture:9092", environment="dev")
    )
    topics = ("onex.evt.fixture.output.v1", "onex.cmd.fixture.request.v1")
    resolved = bus.resolve_bus_bindings(
        principal="runtime",
        subscribe_topics=(),
        publish_topics=(*topics, topics[0]),
        group_id="runtime",
    )
    wire_topics = set()

    async def send(topic, **kwargs):
        wire_topics.add(topic)
        acknowledgement = asyncio.get_running_loop().create_future()
        acknowledgement.set_result(SimpleNamespace(topic=topic, partition=0, offset=0))
        return acknowledgement

    producer = AsyncMock(spec=AIOKafkaProducer)
    producer.send.side_effect = send
    bus._producer = producer
    bus._started = True
    try:
        for topic in topics:
            await bus.publish(
                topic,
                None,
                b"{}",
                ModelEventHeaders(
                    timestamp=datetime(2026, 1, 1, tzinfo=UTC),
                    source="fixture",
                    event_type="fixture.output",
                ),
            )
        assert wire_topics == {binding.physical_topic for binding in resolved.bindings}
        assert grant_keys(resolved) == {
            ("dev", "TOPIC", topic, operation)
            for topic in wire_topics
            for operation in ("WRITE", "DESCRIBE")
        }
    finally:
        await bus.close()


def test_provisioning_pin_contains_no_topic_literals():
    import ast

    pin = (
        ROOT
        / "tests/unit/nodes/node_bus_forwarder_effect/test_provisioning_sync_omn17201.py"
    )
    literals = [
        node.value
        for node in ast.walk(ast.parse(pin.read_text()))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    assert not any(value.startswith("onex.") or ".onex." in value for value in literals)


def test_discovered_contract_dispatches_resolved_wire_payload(tmp_path):
    import importlib
    import importlib.resources
    import tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    module = project["project"]["entry-points"]["onex.nodes"][
        "node_broker_grant_derive_compute"
    ]
    contract = yaml.safe_load(
        importlib.resources.files(module).joinpath("contract.yaml").read_text()
    )
    handler_ref = contract["handler_routing"]["handlers"][0]["handler"]
    handler_class = getattr(
        importlib.import_module(handler_ref["module"]), handler_ref["name"]
    )
    input_ref = contract["input_model"]
    input_class = getattr(
        importlib.import_module(input_ref["module"]), input_ref["name"]
    )
    resolved, _ = gateway_forwarder.build_gateway_transports(shipped_config(tmp_path))
    request = input_class.model_validate_json(resolved.model_dump_json())
    result = handler_class().handle(request)
    assert result.principal == resolved.principal
    assert {
        (g.broker, g.resource_type, g.resource, g.operation) for g in result.grants
    } == grant_keys(resolved)
