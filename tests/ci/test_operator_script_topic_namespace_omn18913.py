# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Operator broker boundaries must honor the shared topic namespace (OMN-18913).

Walk this repository's scripts, including future additions. Broker observers
without subscriptions (lag/alarm readers) do not consume and need no topic
argument at construction. Producer factories are checked at their send seams.
No other clone, suppression list, or service is needed for this commit gate.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import yaml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
BROKER_CLASSES = frozenset(
    {"KafkaConsumer", "KafkaProducer", "AIOKafkaConsumer", "AIOKafkaProducer"}
)


def unrouted_topics(source: str) -> list[int]:
    """Reject raw topic arguments even when another call in the file is routed."""
    tree = ast.parse(source)
    clients: dict[str, str] = {}
    helpers: dict[str, str] = {}
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname or alias.name
                if node.module in {"kafka", "aiokafka"}:
                    clients[name] = alias.name
                if node.module == "omnibase_infra.topics.topic_namespace":
                    helpers[name] = alias.name
                if (
                    node.module == "omnibase_infra.topics"
                    and alias.name == "topic_namespace"
                ):
                    modules.add(name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"kafka", "aiokafka"}:
                    modules.add(alias.asname or alias.name)
                if alias.name == "omnibase_infra.topics.topic_namespace":
                    modules.add(alias.asname or alias.name)

    def call_name(node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return clients.get(node.id, helpers.get(node.id, ""))
        if isinstance(node, ast.Attribute) and ast.unparse(node.value) in modules:
            return node.attr
        return ""

    def routed(node: ast.expr, *, many: bool = False) -> bool:
        if isinstance(node, ast.Starred):
            return routed(node.value, many=True)
        if isinstance(node, ast.Call):
            expected = "apply_topic_namespace_all" if many else "apply_topic_namespace"
            return call_name(node.func) == expected
        if many and isinstance(node, (ast.List, ast.Tuple)):
            return all(routed(item) for item in node.elts)
        return False

    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    constructions = [node for node in calls if call_name(node.func) in BROKER_CLASSES]
    if not constructions:
        return []
    violations: list[int] = []
    sends = []
    for node in calls:
        name = call_name(node.func)
        if name in {"KafkaConsumer", "AIOKafkaConsumer"}:
            if any(not routed(arg) for arg in node.args):
                violations.append(node.lineno)
        elif isinstance(node.func, ast.Attribute):
            method = node.func.attr
            if method in {"send", "send_and_wait", "subscribe"}:
                topics = node.args[:1] or [
                    kw.value for kw in node.keywords if kw.arg in {"topic", "topics"}
                ]
                if method != "subscribe":
                    sends.append(node)
                if not topics or any(
                    not routed(t, many=method == "subscribe") for t in topics
                ):
                    violations.append(node.lineno)
    if not sends:
        violations.extend(
            node.lineno
            for node in constructions
            if call_name(node.func).endswith("Producer")
        )
    return sorted(set(violations))


def test_all_script_topic_arguments_are_routed() -> None:
    offenders = [
        f"{path.relative_to(ROOT)}:{line}"
        for path in sorted((ROOT / "scripts").rglob("*.py"))
        if not any(part in {"tests", ".venv", "__pycache__"} for part in path.parts)
        for line in unrouted_topics(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, "Unrouted operator broker topics:\n" + "\n".join(offenders)


@pytest.mark.parametrize("client", sorted(BROKER_CLASSES))
def test_planted_unrouted_construction_is_refused(client: str) -> None:
    module = "aiokafka" if client.startswith("AIO") else "kafka"
    source = f"from {module} import {client} as Client\n"
    if client.endswith("Consumer"):
        source += 'consumer = Client("raw")\n'
    else:
        source += 'producer = Client()\nproducer.send("raw", b"value")\n'
    assert unrouted_topics(source)


def test_planted_script_fails_the_same_commit_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "planted.py").write_text(
        'from kafka import KafkaConsumer\nclient = KafkaConsumer("shared")\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    with pytest.raises(AssertionError, match=r"planted\.py:2"):
        test_all_script_topic_arguments_are_routed()


def test_guard_runs_at_commit_and_in_the_required_ci_umbrella() -> None:
    from scripts.ci.ci_summary_gate import STRICT_GATE_JOBS

    filename = "tests/ci/test_operator_script_topic_namespace_omn18913.py"
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    hook = next(
        hook
        for repo in config["repos"]
        for hook in repo.get("hooks", [])
        if hook["id"] == "operator-script-topic-namespace"
    )
    assert filename in hook["entry"]
    assert hook["pass_filenames"] is False
    assert "pre-commit" in hook["stages"]
    for path in (
        "scripts/planted.py",
        "src/omnibase_infra/topics/topic_namespace.py",
        filename,
        ".pre-commit-config.yaml",
        ".github/workflows/ci.yml",
    ):
        assert re.search(hook["files"], path)
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    job = workflow["jobs"]["arch-invariants"]
    assert job["name"] in STRICT_GATE_JOBS
    assert any(filename in step.get("run", "") for step in job["steps"])


def test_standalone_agent_declares_and_locks_the_shared_namespace_provider() -> None:
    project = tomllib.loads((ROOT / "scripts/deploy-agent/pyproject.toml").read_text())
    assert "omnibase-infra>=0.38.36,<0.39" in project["project"]["dependencies"]
    lock = tomllib.loads((ROOT / "scripts/deploy-agent/uv.lock").read_text())
    agent = next(p for p in lock["package"] if p["name"] == "deploy-agent")
    assert {"name": "omnibase-infra"} in agent["dependencies"]
    assert any(p["name"] == "omnibase-infra" for p in lock["package"])


def test_half_routed_consumer_and_unrouted_second_send_are_refused() -> None:
    assert unrouted_topics(
        "from aiokafka import AIOKafkaConsumer, AIOKafkaProducer\n"
        "from omnibase_infra.topics.topic_namespace import apply_topic_namespace\n"
        'consumer = AIOKafkaConsumer(apply_topic_namespace("one"), "two")\n'
        "producer = AIOKafkaProducer()\n"
        'producer.send_and_wait(apply_topic_namespace("one"), b"ok")\n'
        'producer.send_and_wait("two", b"bad")\n'
    ) == [3, 6]


def test_routed_aliases_and_subscription_are_accepted() -> None:
    assert (
        unrouted_topics(
            "import kafka as broker\n"
            "import omnibase_infra.topics.topic_namespace as ns\n"
            "consumer = broker.KafkaConsumer()\n"
            'consumer.subscribe(ns.apply_topic_namespace_all(["one", "two"]))\n'
            "producer = broker.KafkaProducer()\n"
            'producer.send(topic=ns.apply_topic_namespace("one"), value=b"ok")\n'
        )
        == []
    )


def _load_script(name: str, path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=[None, "prepr-omn18913"])
def prefix(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.delenv("KAFKA_TOPIC_NAMESPACE", raising=False)
    if request.param:
        monkeypatch.setenv("KAFKA_TOPIC_NAMESPACE", request.param)
        return request.param + "."
    return ""


@pytest.mark.asyncio
async def test_operator_publishers_preserve_parent_topics_and_apply_prefix(
    prefix: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import aiokafka

    from omnibase_core.models.core.model_deployment_topology import (
        ModelDeploymentTopology,
    )

    producer = SimpleNamespace(
        start=AsyncMock(), stop=AsyncMock(), send_and_wait=AsyncMock()
    )
    monkeypatch.setattr(aiokafka, "AIOKafkaProducer", MagicMock(return_value=producer))
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "broker:9092")
    setup = _load_script("operator_setup", "scripts/onex-setup.py")
    await setup._publish_setup_command(
        ModelDeploymentTopology(schema_version="1.0", services={}), "compose.yml", True
    )
    assert producer.send_and_wait.call_args.args[0] == (
        prefix + "onex.cmd.omnibase-infra.setup-orchestration-start.v1"
    )
    baselines = _load_script(
        "operator_baselines", "scripts/emit_baselines_raw_snapshot.py"
    )
    await baselines._emit_to_kafka({}, "broker:9092")
    assert producer.send_and_wait.call_args.args[0] == (
        prefix + "onex.evt.omnibase-infra.baselines-computed.v1"
    )


def test_deploy_agent_transport_topics_and_group(
    prefix: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import kafka
    from deploy_agent import agent, consumer, publisher, trigger
    from deploy_agent.events import EnumRuntimeLane
    from deploy_agent.kafka_config import ModelDeployAgentKafkaConfig

    config = ModelDeployAgentKafkaConfig(
        bootstrap_servers="broker:9092", security_protocol="PLAINTEXT"
    )
    producer = MagicMock()
    factory = MagicMock(return_value=producer)
    monkeypatch.setattr(kafka, "KafkaProducer", factory)
    monkeypatch.setattr(publisher, "KafkaProducer", factory)
    trigger.publish_signed_command({"correlation_id": "test"}, config)
    assert (
        producer.send.call_args.args[0]
        == prefix + "onex.cmd.deploy.rebuild-requested.v1"
    )
    assert publisher.publish_result({"correlation_id": "test"}, config)
    assert (
        producer.send.call_args.args[0]
        == prefix + "onex.evt.deploy.rebuild-completed.v1"
    )

    daemon = object.__new__(agent.DeployAgent)
    daemon._kafka_config = config
    assert daemon._publish_rejection_event(SimpleNamespace(to_wire=dict))
    assert (
        producer.send.call_args.args[0]
        == prefix + "onex.evt.deploy.rebuild-rejected.v1"
    )

    consumer_factory = MagicMock()
    monkeypatch.setattr(consumer, "KafkaConsumer", consumer_factory)
    reader = consumer.DeployConsumer(
        config,
        SimpleNamespace(state_dir=tmp_path),
        frozenset({EnumRuntimeLane.DEV}),
        lambda *_: None,
    )
    assert consumer_factory.call_args.args == (
        prefix + "onex.cmd.deploy.rebuild-requested.v1",
    )
    assert consumer_factory.call_args.kwargs[
        "group_id"
    ] == prefix + consumer.consumer_group_for(None)
    reader._publish_dlq({})
    assert (
        producer.send.call_args.args[0]
        == prefix + "onex.dlq.omnibase-infra.deploy-command.v1"
    )


@pytest.mark.asyncio
async def test_worker_routes_all_topics_and_keeps_physical_ack_coordinates(
    prefix: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import aiokafka
    from aiokafka.structs import TopicPartition

    from scripts.edge_delegation_worker import delegation_channel as channel
    from scripts.edge_delegation_worker.topic_constants import (
        OUTBOUND_FAILURE_TOPICS,
        OUTBOUND_RESULT_TOPICS,
    )

    consumer = SimpleNamespace(getone=AsyncMock(), commit=AsyncMock())
    producer = SimpleNamespace(send_and_wait=AsyncMock())
    factory = MagicMock(return_value=consumer)
    monkeypatch.setattr(aiokafka, "AIOKafkaConsumer", factory)
    monkeypatch.setattr(aiokafka, "AIOKafkaProducer", MagicMock(return_value=producer))
    worker = channel.build_kafka_channel(
        brokers="broker:9092", consumer_group="operator-worker"
    )
    assert factory.call_args.args == (
        prefix + "onex.cmd.omnibase-infra.delegation-request.v1",
        prefix + "onex.cmd.omnibase-infra.delegation-inference-request.v1",
    )
    assert factory.call_args.kwargs["group_id"] == prefix + "operator-worker"
    for topic in (*OUTBOUND_RESULT_TOPICS, *OUTBOUND_FAILURE_TOPICS):
        await worker.publish_result(
            topic=topic, correlation_id=uuid4(), event_type="test", payload={}
        )
        assert producer.send_and_wait.call_args.args[0] == prefix + topic
    physical = prefix + "onex.cmd.omnibase-infra.delegation-request.v1"
    consumer.getone.return_value = SimpleNamespace(
        value=b'{"correlation_id":"11111111-1111-4111-8111-111111111111"}',
        topic=physical,
        partition=2,
        offset=9,
        headers=[],
    )
    envelope = await worker.claim()
    assert envelope is not None
    assert envelope.source_topic == physical
    await worker.ack(envelope)
    consumer.commit.assert_awaited_once_with({TopicPartition(physical, 2): 10})
