# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the topic replication policy + provisioning diff (OMN-15395).

The policy is the single seam where an undeclared replication factor becomes a
concrete number, and the only place a concrete number is checked against the
environment's durability floor. These tests pin both halves.
"""

from __future__ import annotations

import logging

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
        # An undeclared RF resolves to the managed durability floor — the
        # cluster's own broker default — never to the old module constant of 1.
        assert policy.default_replication_factor == MANAGED_MINIMUM_REPLICATION_FACTOR
        assert policy.default_replication_factor != 1
        # No capacity ceiling on a real multi-broker cluster.
        assert policy.capacity_replication_factor is None

    def test_plaintext_broker_derives_self_hosted_profile(self) -> None:
        config = ModelKafkaEventBusConfig(bootstrap_servers="redpanda:9092")
        policy = ModelTopicProvisioningPolicy.from_kafka_config(config)
        assert policy.profile is EnumTopicProvisioningProfile.SELF_HOSTED
        assert not policy.is_managed
        assert policy.default_replication_factor == 1
        # The auth mechanism says NOTHING about node count, so a
        # config-derived policy carries no ceiling until something measures
        # the cluster. See TestSaslClusterIsNotAssumedSingleNode.
        assert policy.capacity_replication_factor is None
        assert policy.broker_count is None

    @pytest.mark.parametrize(
        "mechanism",
        ["PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512"],
    )
    def test_non_iam_mechanisms_carry_no_capacity_ceiling(self, mechanism: str) -> None:
        """Every non-IAM mechanism the config accepts is unmeasured, not RF1.

        RED-before: ``self_hosted()`` used to hardcode
        ``capacity_replication_factor = 1``, so every one of these clusters had
        its contract-declared RF silently clamped to 1.
        """
        config = ModelKafkaEventBusConfig(
            bootstrap_servers="broker-1.example:9096",
            security_protocol="SASL_SSL",
            sasl_mechanism=mechanism,
        )
        policy = ModelTopicProvisioningPolicy.from_kafka_config(config)
        assert policy.capacity_replication_factor is None
        assert policy.broker_count is None


class TestSaslClusterIsNotAssumedSingleNode:
    """The capacity ceiling is MEASURED, never inferred from the auth mechanism.

    RED-before (the durability regression this class exists to stop): the first
    revision classified every ``sasl_mechanism != "AWS_MSK_IAM"`` cluster as a
    single node and reduced its declared replication factor to 1. A multi-broker
    cluster reached over SCRAM — including an MSK cluster fronted by SCRAM — had
    its contract-declared RF2/RF3 clamped to RF1, which is the exact
    ``AWS_KAFKA_HIGH_RISK_CONFIG_RF_EQUALS_ONE`` condition OMN-15395 exists to
    eliminate.
    """

    @staticmethod
    def _scram_policy() -> ModelTopicProvisioningPolicy:
        config = ModelKafkaEventBusConfig(
            bootstrap_servers="b-1.example:9096,b-2.example:9096,b-3.example:9096",
            security_protocol="SASL_SSL",
            sasl_mechanism="SCRAM-SHA-512",
        )
        return ModelTopicProvisioningPolicy.from_kafka_config(config)

    def test_unmeasured_scram_cluster_does_not_clamp_declared_rf(self) -> None:
        policy = self._scram_policy()
        assert policy.resolve_replication_factor(topic=TOPIC, declared=2) == 2
        assert policy.resolve_replication_factor(topic=TOPIC, declared=3) == 3

    def test_three_broker_scram_cluster_preserves_declared_rf(self) -> None:
        policy = self._scram_policy().with_broker_capacity(3)
        assert policy.broker_count == 3
        assert policy.capacity_replication_factor == 3
        assert policy.resolve_replication_factor(topic=TOPIC, declared=2) == 2
        assert policy.resolve_replication_factor(topic=TOPIC, declared=3) == 3

    def test_three_broker_cluster_defaults_undeclared_rf_to_the_durable_value(
        self,
    ) -> None:
        """A multi-node self-hosted cluster has no business minting RF1 either."""
        policy = self._scram_policy().with_broker_capacity(3)
        assert policy.resolve_replication_factor(topic=TOPIC, declared=None) == (
            MANAGED_MINIMUM_REPLICATION_FACTOR
        )


class TestMeasuredBrokerCapacity:
    """``with_broker_capacity`` may install a ceiling; it may never weaken a floor."""

    def test_single_node_measurement_reduces_declared_rf(self) -> None:
        policy = ModelTopicProvisioningPolicy.self_hosted(broker_count=1)
        assert policy.capacity_replication_factor == 1
        assert policy.resolve_replication_factor(topic=TOPIC, declared=2) == 1
        assert policy.resolve_replication_factor(topic=TOPIC, declared=3) == 1

    def test_reduction_is_logged_at_warning_not_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A silent durability downgrade below WARNING is invisible in prod logs."""
        policy = ModelTopicProvisioningPolicy.self_hosted(broker_count=1)
        with caplog.at_level(
            logging.WARNING,
            logger="omnibase_infra.topics.model_topic_provisioning_policy",
        ):
            assert policy.resolve_replication_factor(topic=TOPIC, declared=2) == 1
        assert [record for record in caplog.records if "Reducing" in record.message], (
            "the capacity reduction must be emitted at WARNING; records seen: "
            f"{[(r.levelname, r.message) for r in caplog.records]}"
        )

    def test_managed_floor_survives_a_measurement_below_it(self) -> None:
        """A 1-broker managed cluster REFUSES; it does not clamp to RF1."""
        policy = ModelTopicProvisioningPolicy.managed(broker_count=1)
        assert policy.minimum_replication_factor == MANAGED_MINIMUM_REPLICATION_FACTOR
        # No ceiling installed — a ceiling under the floor would be a bypass of
        # the entire RF1 rejection.
        assert policy.capacity_replication_factor is None
        with pytest.raises(TopicReplicationPolicyError):
            policy.resolve_replication_factor(topic=TOPIC, declared=1)

    def test_managed_multi_broker_measurement_caps_an_impossible_rf(self) -> None:
        policy = ModelTopicProvisioningPolicy.managed(broker_count=3)
        assert policy.resolve_replication_factor(topic=TOPIC, declared=5) == 3
        assert policy.resolve_replication_factor(topic=TOPIC, declared=2) == 2

    def test_measurement_never_lowers_the_durability_floor(self) -> None:
        for broker_count in (1, 2, 3, 9):
            measured = ModelTopicProvisioningPolicy.managed(broker_count=broker_count)
            assert measured.minimum_replication_factor == (
                MANAGED_MINIMUM_REPLICATION_FACTOR
            )
            assert measured.profile is EnumTopicProvisioningProfile.MANAGED

    def test_zero_or_negative_broker_count_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not a live node count"):
            ModelTopicProvisioningPolicy.self_hosted().with_broker_capacity(0)

    def test_from_env_reads_the_runtime_kafka_configuration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "b-1.msk.example:9098")
        monkeypatch.setenv("KAFKA_SECURITY_PROTOCOL", "SASL_SSL")
        monkeypatch.setenv("KAFKA_SASL_MECHANISM", "AWS_MSK_IAM")
        monkeypatch.setenv("KAFKA_MSK_REGION", "us-east-1")
        assert ModelTopicProvisioningPolicy.from_env().is_managed


class TestManagedPolicyResolution:
    """Managed staging: RF1 rejected, undeclared floored, RF>=2 untouched."""

    def test_rf1_is_rejected(self) -> None:
        policy = ModelTopicProvisioningPolicy.managed()
        with pytest.raises(TopicReplicationPolicyError) as excinfo:
            policy.resolve_replication_factor(topic=TOPIC, declared=1)
        assert "replication_factor=1" in str(excinfo.value)

    def test_undeclared_resolves_to_the_floor_never_to_one(self) -> None:
        """The defect was a module constant of 1, not "a default exists".

        Refusing outright was tried and measured: 168 of 168 production topics
        declare no replication factor and 75 have no producing contract in this
        repo at all, so refusal made provisioning a permanent no-op on MSK.
        Resolving to the floor keeps provisioning working and still makes an
        RF1 topic unreachable through this path.
        """
        policy = ModelTopicProvisioningPolicy.managed()
        resolved = policy.resolve_replication_factor(topic=TOPIC, declared=None)
        assert resolved == MANAGED_MINIMUM_REPLICATION_FACTOR
        assert resolved != 1

    def test_a_policy_with_no_default_still_refuses(self) -> None:
        """The refuse-on-undeclared branch is still reachable and still works.

        The managed profile chooses to carry a default; the mechanism that
        refuses when a profile declares none is retained and tested, so a future
        strict profile is a constructor argument rather than a code change.
        """
        policy = ModelTopicProvisioningPolicy(
            profile=EnumTopicProvisioningProfile.MANAGED,
            minimum_replication_factor=MANAGED_MINIMUM_REPLICATION_FACTOR,
            default_replication_factor=None,
        )
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

    def test_declared_rf_above_measured_capacity_is_reduced_not_rejected(
        self,
    ) -> None:
        """A broker MEASURED at one node cannot host RF3; reduce, don't fail create.

        This is what lets the contract tree declare the production-durable RF2
        while local Redpanda, CI, and the ``.201`` lanes keep provisioning. Note
        the measurement: the reduction is licensed by the node count, not by the
        cluster's auth mechanism.
        """
        policy = ModelTopicProvisioningPolicy.self_hosted(broker_count=1)
        assert policy.resolve_replication_factor(topic=TOPIC, declared=3) == 1
        assert policy.resolve_replication_factor(topic=TOPIC, declared=2) == 1

    def test_unmeasured_policy_reduces_nothing(self) -> None:
        """No measurement, no ceiling: a declared RF reaches the broker intact.

        RED-before for the durability regression — this returned 1 for every
        declared value because ``self_hosted()`` hardcoded the ceiling.
        """
        policy = ModelTopicProvisioningPolicy.self_hosted()
        assert policy.capacity_replication_factor is None
        assert policy.resolve_replication_factor(topic=TOPIC, declared=2) == 2
        assert policy.resolve_replication_factor(topic=TOPIC, declared=3) == 3

    def test_reduction_is_one_way(self) -> None:
        """Capacity never RAISES a declared value; it only ever reduces."""
        policy = ModelTopicProvisioningPolicy(
            profile=EnumTopicProvisioningProfile.SELF_HOSTED,
            minimum_replication_factor=1,
            default_replication_factor=1,
            capacity_replication_factor=3,
        )
        assert policy.resolve_replication_factor(topic=TOPIC, declared=1) == 1


class TestPolicyValidation:
    """A policy cannot declare bounds that contradict each other."""

    def test_default_below_floor_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="below"):
            ModelTopicProvisioningPolicy(
                profile=EnumTopicProvisioningProfile.MANAGED,
                minimum_replication_factor=2,
                default_replication_factor=1,
            )

    def test_capacity_ceiling_may_not_undercut_the_durability_floor(self) -> None:
        """Otherwise a capacity reduction would silently mint RF1 on MSK.

        The reduction happens before the floor check, so a ceiling below the
        floor would be a bypass of the entire RF1 rejection. It is refused at
        construction time instead.
        """
        with pytest.raises(ValidationError, match="may never undercut"):
            ModelTopicProvisioningPolicy(
                profile=EnumTopicProvisioningProfile.MANAGED,
                minimum_replication_factor=2,
                default_replication_factor=2,
                capacity_replication_factor=1,
            )

    def test_default_above_capacity_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="cannot default to a value"):
            ModelTopicProvisioningPolicy(
                profile=EnumTopicProvisioningProfile.SELF_HOSTED,
                minimum_replication_factor=1,
                default_replication_factor=3,
                capacity_replication_factor=2,
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
