# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration guard for savings estimator runtime subscriptions."""

from __future__ import annotations

import ast
from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.services.observability.savings_estimation.config import (
    ConfigSavingsEstimation,
)
from omnibase_infra.utils import compute_consumer_group_id

SERVICE_KERNEL_PATH = Path("src/omnibase_infra/runtime/service_kernel.py")
TEST_BOOTSTRAP_SERVERS = "localhost:9092"


def _load_service_kernel_ast() -> ast.Module:
    return ast.parse(SERVICE_KERNEL_PATH.read_text(encoding="utf-8"))


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


@pytest.mark.integration
def test_savings_estimator_subscribe_calls_use_canonical_identity() -> None:
    """Savings subscriptions derive their groups from a typed node identity."""
    tree = _load_service_kernel_ast()
    savings_identity_assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_savings_node_identity"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and _call_name(node.value.func) == "ModelNodeIdentity"
    ]
    assert len(savings_identity_assignments) == 1
    identity_call = savings_identity_assignments[0].value
    assert isinstance(identity_call, ast.Call)
    identity_keywords = {
        keyword.arg: keyword.value for keyword in identity_call.keywords
    }
    assert isinstance(identity_keywords["env"], ast.Name)
    assert identity_keywords["env"].id == "environment"
    assert isinstance(identity_keywords["node_name"], ast.Constant)
    assert identity_keywords["node_name"].value == "savings-estimator"
    assert isinstance(identity_keywords["service"], ast.BoolOp)
    assert isinstance(identity_keywords["service"].op, ast.Or)
    assert isinstance(identity_keywords["service"].values[0], ast.Attribute)
    assert identity_keywords["service"].values[0].attr == "name"
    assert isinstance(identity_keywords["service"].values[1], ast.Constant)
    assert identity_keywords["service"].values[1].value == "onex-kernel"
    assert isinstance(identity_keywords["version"], ast.Constant)
    assert identity_keywords["version"].value == "v1"

    savings_input_loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_input_topic"
    ]
    assert len(savings_input_loops) == 1

    savings_subscribe_calls = [
        node
        for node in ast.walk(savings_input_loops[0])
        if isinstance(node, ast.Call) and _call_name(node.func) == "subscribe"
        if any(
            keyword.arg == "node_identity"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "_savings_node_identity"
            for keyword in node.keywords
        )
    ]

    assert len(savings_subscribe_calls) == 1
    assert all(
        keyword.arg != "group_id" for keyword in savings_subscribe_calls[0].keywords
    )


@pytest.mark.integration
def test_savings_estimator_topics_keep_distinct_effective_consumer_groups() -> None:
    """Canonical base identity remains isolated per topic at the Kafka boundary."""
    savings_config = ConfigSavingsEstimation(
        kafka_bootstrap_servers=TEST_BOOTSTRAP_SERVERS,
    )
    identity = ModelNodeIdentity(
        env="dev",
        service="onex-kernel",
        node_name="savings-estimator",
        version="v1",
    )
    base_group_id = compute_consumer_group_id(identity)
    event_bus = EventBusKafka(
        config=ModelKafkaEventBusConfig(
            bootstrap_servers=TEST_BOOTSTRAP_SERVERS,
            environment="dev",
        )
    )

    effective_group_ids = {
        topic: event_bus._resolve_effective_group_id(
            base_group_id,
            topic,
            uuid4(),
            (topic, base_group_id),
        )
        for topic in savings_config.consumed_topics
    }

    assert len(effective_group_ids) >= 3
    assert len(set(effective_group_ids.values())) == len(effective_group_ids)
    for topic, effective_group_id in effective_group_ids.items():
        assert effective_group_id == f"{base_group_id}.__t.{topic}"


@pytest.mark.integration
def test_savings_estimator_subscription_failures_are_warning_logged() -> None:
    """Subscription failures should be visible in runtime logs."""
    tree = _load_service_kernel_ast()
    warning_messages = [
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "warning"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ]

    assert "Could not subscribe to %s for savings estimation" in warning_messages
