# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration-facing golden tests for runner usage savings estimates."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.pricing import ModelPricingTable
from omnibase_infra.nodes.node_runner_usage_effect.handlers import (
    HandlerRunnerUsageSavings,
)
from omnibase_infra.nodes.node_runner_usage_effect.models import ModelRunnerUsageEvent
from omnibase_infra.topics import topic_keys
from omnibase_infra.topics.service_topic_registry import ServiceTopicRegistry

FIXTURE_DIR = Path(__file__).parent.parent.parent / "fixtures" / "cost_observability"
RUNNER_FIXTURE = FIXTURE_DIR / "task-12-runner.fixtures.jsonl"
RUNNER_GOLDEN = FIXTURE_DIR / "task-12-runner.golden.json"


def _load_runner_events(path: Path = RUNNER_FIXTURE) -> list[ModelRunnerUsageEvent]:
    return [
        ModelRunnerUsageEvent(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _serialize_replay(
    handler: HandlerRunnerUsageSavings,
    events: list[ModelRunnerUsageEvent],
) -> list[dict[str, object]]:
    return [estimate.to_kafka_payload() for estimate in handler.replay(events)]


@pytest.mark.integration
def test_runner_usage_10_minutes_emits_one_exact_savings_estimate() -> None:
    table = ModelPricingTable.from_dict(
        {
            "schema_version": "1.0.0",
            "models": {},
            "runner_cost": {"github_hosted_per_minute_usd": 0.008},
        }
    )
    handler = HandlerRunnerUsageSavings(pricing_table=table)
    event = _load_runner_events()[0]

    first = handler.compute_savings(event)
    duplicate = handler.compute_savings(event)

    assert first is not None
    assert duplicate is None
    assert first.workflow_run_id == "103430001"
    assert first.job_id == "build-and-test"
    assert first.model_local == "self_hosted_runner"
    assert first.model_cloud_baseline == "github_runner"
    assert first.runner_minutes == Decimal("10.000000")
    assert first.local_cost_usd == Decimal("0.000000")
    assert first.cloud_cost_usd == Decimal("0.080000")
    assert first.savings_usd == Decimal("0.080000")
    assert first.source_event_id == "runner-usage:103430001:build-and-test"


@pytest.mark.integration
def test_runner_usage_dedupe_cache_is_bounded() -> None:
    table = ModelPricingTable.from_dict(
        {
            "schema_version": "1.0.0",
            "models": {},
            "runner_cost": {"github_hosted_per_minute_usd": 0.008},
        }
    )
    handler = HandlerRunnerUsageSavings(pricing_table=table, emitted_key_limit=1)
    first_event = _load_runner_events()[0]
    second_event = first_event.model_copy(update={"job_id": "lint"})

    assert handler.compute_savings(first_event) is not None
    assert handler.compute_savings(second_event) is not None
    assert handler.compute_savings(first_event) is not None


@pytest.mark.integration
def test_runner_usage_missing_runner_cost_raises_correlated_config_error() -> None:
    table = ModelPricingTable.from_dict({"schema_version": "1.0.0", "models": {}})
    handler = HandlerRunnerUsageSavings(pricing_table=table)

    with pytest.raises(ProtocolConfigurationError) as exc_info:
        handler.compute_savings(_load_runner_events()[0])

    context = exc_info.value.model.context
    assert context["operation"] == "compute_runner_usage_savings"
    assert context["target_name"] == "handler-runner-usage-savings"
    assert context["parameter"] == "runner_cost.github_hosted_per_minute_usd"
    assert exc_info.value.model.correlation_id is not None


@pytest.mark.integration
def test_runner_usage_replay_fixture_matches_golden_field_by_field() -> None:
    handler = HandlerRunnerUsageSavings()
    actual = _serialize_replay(handler, _load_runner_events())
    expected = json.loads(RUNNER_GOLDEN.read_text(encoding="utf-8"))

    assert actual == expected


@pytest.mark.integration
def test_runner_usage_topics_are_registered() -> None:
    registry = ServiceTopicRegistry.from_defaults()

    assert registry.resolve(topic_keys.RUNNER_USAGE_RECORDED) == (
        "onex.evt.omninode.runner-usage-recorded.v1"
    )
    assert registry.resolve(topic_keys.SAVINGS_ESTIMATED) == (
        "onex.evt.omnibase-infra.savings-estimated.v1"
    )
