# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the topic replication policy + provisioning diff (OMN-15395).

The policy is the single seam where an undeclared replication factor becomes a
concrete number, and the only place a concrete number is checked against the
environment's durability floor. These tests pin both halves.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.errors import TopicReplicationPolicyError
from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)
from omnibase_infra.topics.enum_topic_provisioning_profile import (
    EnumTopicProvisioningProfile,
)
from omnibase_infra.topics.model_topic_provisioning_diff import (
    build_provisioning_diff,
)
from omnibase_infra.topics.model_topic_provisioning_policy import (
    MANAGED_MINIMUM_REPLICATION_FACTOR,
    ModelTopicProvisioningPolicy,
)
from omnibase_infra.topics.model_topic_spec import ModelTopicSpec

pytestmark = [pytest.mark.unit]

TOPIC = "onex.evt.test-producer.example-event.v1"  # onex-topic-allow: unit fixture


class TestProfileDerivation:
    """The profile comes from the live Kafka config, not a caller-supplied label."""

    def test_msk_iam_auth_derives_managed_profile(self) -> None:
        config = ModelKafkaEventBusConfig(
            bootstrap_servers="b-1.msk.example:9098",
            security_protocol="SASL_SSL",
            sasl_mechanism="AWS_MSK_IAM",
            msk_region="us-east-1",
        )
        policy = ModelTopicProvisioningPolicy.from_kafka_config(config)
        assert policy.profile is EnumTopicProvisioningProfile.MANAGED
        assert policy.is_managed
        assert policy.minimum_replication_factor == MANAGED_MINIMUM_REPLICATION_FACTOR
        # No implicit default: undeclared RF must fail closed, not inherit.
        assert policy.default_replication_factor is None

    def test_plaintext_broker_derives_self_hosted_profile(self) -> None:
        config = ModelKafkaEventBusConfig(bootstrap_servers="redpanda:9092")
        policy = ModelTopicProvisioningPolicy.from_kafka_config(config)
        assert policy.profile is EnumTopicProvisioningProfile.SELF_HOSTED
        assert not policy.is_managed
        assert policy.default_replication_factor == 1

    def test_from_env_reads_the_runtime_kafka_configuration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "b-1.msk.example:9098")
        monkeypatch.setenv("KAFKA_SECURITY_PROTOCOL", "SASL_SSL")
        monkeypatch.setenv("KAFKA_SASL_MECHANISM", "AWS_MSK_IAM")
        monkeypatch.setenv("KAFKA_MSK_REGION", "us-east-1")
        assert ModelTopicProvisioningPolicy.from_env().is_managed


class TestManagedPolicyResolution:
    """Managed staging: RF1 rejected, undeclared refused, RF>=2 untouched."""

    def test_rf1_is_rejected(self) -> None:
        policy = ModelTopicProvisioningPolicy.managed()
        with pytest.raises(TopicReplicationPolicyError) as excinfo:
            policy.resolve_replication_factor(topic=TOPIC, declared=1)
        assert "replication_factor=1" in str(excinfo.value)

    def test_undeclared_is_refused_not_defaulted(self) -> None:
        policy = ModelTopicProvisioningPolicy.managed()
        with pytest.raises(TopicReplicationPolicyError) as excinfo:
            policy.resolve_replication_factor(topic=TOPIC, declared=None)
        assert "no replication_factor declared" in str(excinfo.value)

    def test_declared_rf2_resolves_unchanged(self) -> None:
        policy = ModelTopicProvisioningPolicy.managed()
        assert policy.resolve_replication_factor(topic=TOPIC, declared=2) == 2

    def test_resolve_spec_preserves_partitions_and_config(self) -> None:
        policy = ModelTopicProvisioningPolicy.managed()
        spec = ModelTopicSpec(
            suffix=TOPIC,
            partitions=7,
            replication_factor=3,
            kafka_config={"cleanup.policy": "compact"},
            provisioning_priority=5,
        )
        resolved = policy.resolve_spec(spec)
        assert resolved.replication_factor == 3
        assert resolved.partitions == 7
        assert dict(resolved.kafka_config or {}) == {"cleanup.policy": "compact"}
        assert resolved.provisioning_priority == 5


class TestSelfHostedPolicyResolution:
    """Self-hosted single-broker deployments keep working at RF1."""

    def test_undeclared_resolves_to_declared_default(self) -> None:
        policy = ModelTopicProvisioningPolicy.self_hosted()
        assert policy.resolve_replication_factor(topic=TOPIC, declared=None) == 1

    def test_declared_rf1_is_allowed(self) -> None:
        policy = ModelTopicProvisioningPolicy.self_hosted()
        assert policy.resolve_replication_factor(topic=TOPIC, declared=1) == 1

    def test_declared_rf3_is_passed_through(self) -> None:
        policy = ModelTopicProvisioningPolicy.self_hosted()
        assert policy.resolve_replication_factor(topic=TOPIC, declared=3) == 3


class TestPolicyValidation:
    """A policy cannot declare a default it would itself reject."""

    def test_default_below_floor_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="below"):
            ModelTopicProvisioningPolicy(
                profile=EnumTopicProvisioningProfile.MANAGED,
                minimum_replication_factor=2,
                default_replication_factor=1,
            )


class TestSpecModel:
    """``ModelTopicSpec`` no longer carries a silent RF default."""

    def test_undeclared_replication_factor_is_none_not_one(self) -> None:
        assert ModelTopicSpec(suffix=TOPIC).replication_factor is None

    def test_zero_replication_factor_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelTopicSpec(suffix=TOPIC, replication_factor=0)


class TestProvisioningDiff:
    """The shared desired-vs-live diff behind every provisioning path."""

    def test_splits_missing_and_present(self) -> None:
        diff = build_provisioning_diff(["a", "b", "c"], ["b", "z"])
        assert diff.missing_topics == ("a", "c")
        assert diff.present_topics == ("b",)
        assert diff.desired_topics == ("a", "b", "c")
        assert diff.has_missing

    def test_fully_provisioned_reports_nothing_missing(self) -> None:
        diff = build_provisioning_diff(["a", "b"], ["a", "b", "extra"])
        assert diff.missing_topics == ()
        assert not diff.has_missing

    def test_desired_order_is_preserved_and_deduplicated(self) -> None:
        diff = build_provisioning_diff(["z", "a", "z"], [])
        assert diff.desired_topics == ("z", "a")
        assert diff.missing_topics == ("z", "a")

    def test_empty_desired_set_is_a_no_op(self) -> None:
        diff = build_provisioning_diff([], ["a"])
        assert diff.missing_topics == ()
        assert diff.present_topics == ()
