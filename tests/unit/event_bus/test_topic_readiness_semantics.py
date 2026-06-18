# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Deterministic topic-readiness semantics tests (OMN-13237, §3.7).

``evaluate_topic_readiness`` classifies broker metadata into a per-topic
readiness outcome. A topic is READY only when broker metadata returns it, its
partition count matches the expected spec, every partition has a leader, and the
reported replication factor matches the spec (where inspectable). Each failure
carries a CLASSIFIED reason, not a bare boolean.
"""

from __future__ import annotations

import pytest

from omnibase_infra.event_bus.enum_topic_readiness_failure_reason import (
    EnumTopicReadinessFailureReason,
)
from omnibase_infra.event_bus.enum_topic_readiness_status import (
    EnumTopicReadinessStatus,
)
from omnibase_infra.event_bus.service_topic_manager import evaluate_topic_readiness
from omnibase_infra.topics.model_topic_spec import ModelTopicSpec


def _meta(
    topic: str,
    partitions: int,
    *,
    leader: int = 1,
    replicas: tuple[int, ...] | None = (1,),
    error_code: int = 0,
) -> dict[str, object]:
    part_list: list[dict[str, object]] = []
    for i in range(partitions):
        entry: dict[str, object] = {"partition": i, "leader": leader}
        if replicas is not None:
            entry["replicas"] = list(replicas)
        part_list.append(entry)
    return {"topic": topic, "error_code": error_code, "partitions": part_list}


class TestReadyClassification:
    def test_topic_ready_when_metadata_converged(self) -> None:
        result = evaluate_topic_readiness(
            ("topic.a",),
            [_meta("topic.a", 6)],
            expected_specs={"topic.a": ModelTopicSpec(suffix="topic.a")},
        )
        assert result.status is EnumTopicReadinessStatus.READY
        assert result.is_ready
        assert result.ready_topics == ("topic.a",)
        assert result.failures == ()

    def test_ready_without_spec_uses_existence_and_leader_only(self) -> None:
        result = evaluate_topic_readiness(("topic.a",), [_meta("topic.a", 3)])
        assert result.is_ready


class TestFailureClassification:
    def test_topic_absent(self) -> None:
        result = evaluate_topic_readiness(("topic.a",), [])
        assert result.status is EnumTopicReadinessStatus.NOT_READY
        assert result.failures[0].reason is (
            EnumTopicReadinessFailureReason.TOPIC_ABSENT
        )

    def test_error_code_classified_as_absent(self) -> None:
        result = evaluate_topic_readiness(
            ("topic.a",), [_meta("topic.a", 0, error_code=3)]
        )
        assert result.failures[0].reason is (
            EnumTopicReadinessFailureReason.TOPIC_ABSENT
        )

    def test_partition_mismatch(self) -> None:
        result = evaluate_topic_readiness(
            ("topic.a",),
            [_meta("topic.a", 3)],
            expected_specs={"topic.a": ModelTopicSpec(suffix="topic.a", partitions=6)},
        )
        assert result.failures[0].reason is (
            EnumTopicReadinessFailureReason.PARTITION_MISMATCH
        )

    def test_no_leader(self) -> None:
        result = evaluate_topic_readiness(
            ("topic.a",), [_meta("topic.a", 2, leader=-1)]
        )
        assert result.failures[0].reason is (EnumTopicReadinessFailureReason.NO_LEADER)

    def test_replication_mismatch(self) -> None:
        result = evaluate_topic_readiness(
            ("topic.a",),
            [_meta("topic.a", 1, replicas=(1,))],
            expected_specs={
                "topic.a": ModelTopicSpec(
                    suffix="topic.a", partitions=1, replication_factor=3
                )
            },
        )
        assert result.failures[0].reason is (
            EnumTopicReadinessFailureReason.REPLICATION_MISMATCH
        )

    def test_replication_skipped_when_not_inspectable(self) -> None:
        # No "replicas" key -> RF check is skipped (not a failure).
        result = evaluate_topic_readiness(
            ("topic.a",),
            [_meta("topic.a", 1, replicas=None)],
            expected_specs={
                "topic.a": ModelTopicSpec(
                    suffix="topic.a", partitions=1, replication_factor=3
                )
            },
        )
        assert result.is_ready

    def test_zero_partitions_is_mismatch(self) -> None:
        result = evaluate_topic_readiness(("topic.a",), [_meta("topic.a", 0)])
        assert result.failures[0].reason is (
            EnumTopicReadinessFailureReason.PARTITION_MISMATCH
        )


class TestMixedSet:
    def test_one_ready_one_failed_is_not_ready(self) -> None:
        result = evaluate_topic_readiness(
            ("topic.a", "topic.b"),
            [_meta("topic.a", 6)],
        )
        assert result.status is EnumTopicReadinessStatus.NOT_READY
        assert result.ready_topics == ("topic.a",)
        assert {f.topic for f in result.failures} == {"topic.b"}

    def test_empty_topic_set_is_skipped(self) -> None:
        result = evaluate_topic_readiness((), [])
        assert result.status is EnumTopicReadinessStatus.READY  # vacuously ready


@pytest.mark.parametrize("attempts", [1, 5, 60])
def test_attempts_recorded(attempts: int) -> None:
    result = evaluate_topic_readiness(
        ("topic.a",), [_meta("topic.a", 6)], attempts=attempts
    )
    assert result.attempts == attempts
