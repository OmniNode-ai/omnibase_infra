# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The capacity ceiling comes from a measurement, or it does not come at all.

OMN-15395. ``probe_broker_count`` is the seam that replaced an inference
("``sasl_mechanism != AWS_MSK_IAM`` therefore one node") with a live
``describe_cluster`` read. Every failure mode here resolves to ``None`` —
*unmeasured* — rather than to a guessed node count, because a guessed ceiling
silently downgrades a contract-declared replication factor while a missing
ceiling merely fails loudly at ``CreateTopics``.
"""

from __future__ import annotations

from typing import Any

import pytest

from omnibase_infra.topics.broker_capacity_probe import (
    bind_policy_to_broker_capacity,
    probe_broker_count,
)
from omnibase_infra.topics.model_topic_provisioning_policy import (
    MANAGED_MINIMUM_REPLICATION_FACTOR,
    ModelTopicProvisioningPolicy,
)

pytestmark = [pytest.mark.unit]


class _Admin:
    """Minimal stand-in for the ``describe_cluster`` surface of the admin client."""

    def __init__(self, described: object) -> None:
        self._described = described
        self.calls = 0

    async def describe_cluster(self) -> Any:
        self.calls += 1
        if isinstance(self._described, Exception):
            raise self._described
        return self._described


class _AdminWithoutDescribeCluster:
    """An admin client that predates / lacks the cluster-metadata surface."""


def _cluster(broker_count: int) -> dict[str, object]:
    return {
        "cluster_id": "c",
        "controller_id": 1,
        "brokers": [{"node_id": i} for i in range(broker_count)],
    }


class TestProbeBrokerCount:
    async def test_counts_the_reported_brokers(self) -> None:
        assert await probe_broker_count(_Admin(_cluster(3))) == 3

    async def test_missing_describe_cluster_is_unmeasured(self) -> None:
        assert await probe_broker_count(_AdminWithoutDescribeCluster()) is None

    async def test_describe_cluster_failure_is_unmeasured(self) -> None:
        assert await probe_broker_count(_Admin(RuntimeError("boom"))) is None

    @pytest.mark.parametrize(
        "described",
        [
            "not-a-mapping",
            {"cluster_id": "c"},
            {"brokers": "not-a-sequence"},
            {"brokers": []},
        ],
    )
    async def test_unusable_responses_are_unmeasured(self, described: object) -> None:
        assert await probe_broker_count(_Admin(described)) is None


class TestBindPolicyToBrokerCapacity:
    async def test_measurement_installs_the_ceiling(self) -> None:
        bound = await bind_policy_to_broker_capacity(
            _Admin(_cluster(3)), ModelTopicProvisioningPolicy.self_hosted()
        )
        assert bound.broker_count == 3
        assert bound.capacity_replication_factor == 3

    async def test_unmeasured_policy_is_returned_unchanged(self) -> None:
        policy = ModelTopicProvisioningPolicy.self_hosted()
        bound = await bind_policy_to_broker_capacity(
            _AdminWithoutDescribeCluster(), policy
        )
        assert bound is policy
        assert bound.capacity_replication_factor is None

    async def test_managed_floor_is_never_weakened_by_a_measurement(self) -> None:
        bound = await bind_policy_to_broker_capacity(
            _Admin(_cluster(1)), ModelTopicProvisioningPolicy.managed()
        )
        assert bound.minimum_replication_factor == MANAGED_MINIMUM_REPLICATION_FACTOR
        assert bound.capacity_replication_factor is None
