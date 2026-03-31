# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for subscription verification probe [OMN-7040]."""

from __future__ import annotations

import pytest

from omnibase_infra.enums.enum_check_severity import EnumCheckSeverity
from omnibase_infra.enums.enum_contract_check_type import EnumContractCheckType
from omnibase_infra.enums.enum_validation_verdict import EnumValidationVerdict
from omnibase_infra.verification.contract_parser import (
    ModelParsedContractForVerification,
)
from omnibase_infra.verification.probes.probe_subscription import (
    check_subscriptions,
)

CONTRACT_NAME = "node_registration_orchestrator"

SAMPLE_TOPICS = (
    "onex.evt.platform.node-introspection.v1",
    "onex.evt.platform.registry-request-introspection.v1",
    "onex.intent.platform.runtime-tick.v1",
)


def _make_contract(
    subscribe_topics: tuple[str, ...] = SAMPLE_TOPICS,
) -> ModelParsedContractForVerification:
    return ModelParsedContractForVerification(
        name=CONTRACT_NAME,
        node_type="ORCHESTRATOR_GENERIC",
        subscribe_topics=subscribe_topics,
    )


def _make_admin_fn(
    subscribed: set[str] | None = None,
    error: Exception | None = None,
) -> object:
    """Create a mock kafka_admin_fn.

    Args:
        subscribed: Set of topics to return. If None, returns all SAMPLE_TOPICS.
        error: If set, the function raises this exception.
    """
    if subscribed is None:
        subscribed = set(SAMPLE_TOPICS)

    def admin_fn(group_id: str) -> set[str]:
        if error is not None:
            raise error
        return subscribed

    return admin_fn


@pytest.mark.unit
class TestCheckSubscriptionsPass:
    """Happy-path subscription checks."""

    def test_all_topics_subscribed(self) -> None:
        contract = _make_contract()
        results = check_subscriptions(contract, kafka_admin_fn=_make_admin_fn())
        assert len(results) == len(SAMPLE_TOPICS)
        assert all(r.verdict == EnumValidationVerdict.PASS for r in results)
        assert all(r.check_type == EnumContractCheckType.SUBSCRIPTION for r in results)

    def test_no_topics_declared(self) -> None:
        contract = _make_contract(subscribe_topics=())
        results = check_subscriptions(contract, kafka_admin_fn=_make_admin_fn())
        assert len(results) == 1
        assert results[0].verdict == EnumValidationVerdict.PASS
        assert "No subscribe_topics" in results[0].evidence

    def test_contract_name_preserved(self) -> None:
        contract = _make_contract()
        results = check_subscriptions(contract, kafka_admin_fn=_make_admin_fn())
        assert all(r.contract_name == CONTRACT_NAME for r in results)


@pytest.mark.unit
class TestCheckSubscriptionsFail:
    """Failure-path subscription checks."""

    def test_consumer_group_empty(self) -> None:
        contract = _make_contract()
        results = check_subscriptions(
            contract, kafka_admin_fn=_make_admin_fn(subscribed=set())
        )
        assert len(results) == len(SAMPLE_TOPICS)
        assert all(r.verdict == EnumValidationVerdict.FAIL for r in results)
        assert all("no subscribed topics" in r.evidence.lower() for r in results)

    def test_partial_subscriptions(self) -> None:
        subscribed = {SAMPLE_TOPICS[0]}
        contract = _make_contract()
        results = check_subscriptions(
            contract, kafka_admin_fn=_make_admin_fn(subscribed=subscribed)
        )
        pass_count = sum(1 for r in results if r.verdict == EnumValidationVerdict.PASS)
        fail_count = sum(1 for r in results if r.verdict == EnumValidationVerdict.FAIL)
        assert pass_count == 1
        assert fail_count == 2

    def test_missing_topic_evidence(self) -> None:
        subscribed = {SAMPLE_TOPICS[0]}
        contract = _make_contract()
        results = check_subscriptions(
            contract, kafka_admin_fn=_make_admin_fn(subscribed=subscribed)
        )
        fail_results = [r for r in results if r.verdict == EnumValidationVerdict.FAIL]
        for r in fail_results:
            assert "NOT subscribed" in r.evidence


@pytest.mark.unit
class TestCheckSubscriptionsQuarantine:
    """Degraded-state checks: admin API unavailable -> QUARANTINE."""

    def test_admin_api_unavailable(self) -> None:
        contract = _make_contract()
        results = check_subscriptions(
            contract,
            kafka_admin_fn=_make_admin_fn(error=ConnectionError("Kafka unavailable")),
        )
        assert len(results) == len(SAMPLE_TOPICS)
        assert all(r.verdict == EnumValidationVerdict.QUARANTINE for r in results)

    def test_quarantine_evidence_includes_error(self) -> None:
        contract = _make_contract()
        results = check_subscriptions(
            contract,
            kafka_admin_fn=_make_admin_fn(error=ConnectionError("Connection refused")),
        )
        assert all("Connection refused" in r.evidence for r in results)

    def test_quarantine_severity_is_required(self) -> None:
        contract = _make_contract()
        results = check_subscriptions(
            contract,
            kafka_admin_fn=_make_admin_fn(error=RuntimeError("timeout")),
        )
        assert all(r.severity == EnumCheckSeverity.REQUIRED for r in results)


@pytest.mark.unit
class TestCheckSubscriptionsOnePerTopic:
    """One result per subscribe_topic."""

    def test_result_count_matches_topic_count(self) -> None:
        topics = (
            "topic.a",
            "topic.b",
            "topic.c",
            "topic.d",
        )
        contract = _make_contract(subscribe_topics=topics)
        results = check_subscriptions(
            contract, kafka_admin_fn=_make_admin_fn(subscribed=set(topics))
        )
        assert len(results) == 4

    def test_single_topic(self) -> None:
        contract = _make_contract(subscribe_topics=("only.topic",))
        results = check_subscriptions(
            contract,
            kafka_admin_fn=_make_admin_fn(subscribed={"only.topic"}),
        )
        assert len(results) == 1
        assert results[0].verdict == EnumValidationVerdict.PASS
