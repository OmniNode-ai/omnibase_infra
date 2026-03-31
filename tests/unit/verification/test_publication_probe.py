# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for publication verification probe [OMN-7040]."""

from __future__ import annotations

import pytest

from omnibase_infra.enums.enum_check_severity import EnumCheckSeverity
from omnibase_infra.enums.enum_contract_check_type import EnumContractCheckType
from omnibase_infra.enums.enum_validation_verdict import EnumValidationVerdict
from omnibase_infra.verification.contract_parser import (
    ModelParsedContractForVerification,
)
from omnibase_infra.verification.probes.probe_publication import (
    CORE_REGISTRATION_SUFFIXES,
    check_publications,
)

CONTRACT_NAME = "node_registration_orchestrator"

SAMPLE_TOPICS = (
    "onex.evt.platform.node-registration-result.v1",
    "onex.evt.platform.node-registration-initiated.v1",
    "onex.evt.platform.node-became-active.v1",
)


def _make_contract(
    publish_topics: tuple[str, ...] = SAMPLE_TOPICS,
) -> ModelParsedContractForVerification:
    return ModelParsedContractForVerification(
        name=CONTRACT_NAME,
        node_type="ORCHESTRATOR_GENERIC",
        publish_topics=publish_topics,
    )


def _make_watermark_fn(
    high_offset: int = 100,
    error: Exception | None = None,
    per_topic: dict[str, int] | None = None,
) -> object:
    """Create a mock watermark_fn.

    Args:
        high_offset: Default high watermark to return.
        error: If set, the function raises this exception.
        per_topic: Per-topic high watermark overrides.
    """

    def watermark_fn(topic: str) -> tuple[int, int]:
        if error is not None:
            raise error
        if per_topic is not None and topic in per_topic:
            return (0, per_topic[topic])
        return (0, high_offset)

    return watermark_fn


@pytest.mark.unit
class TestCheckPublicationsPass:
    """Happy-path publication checks."""

    def test_all_topics_have_data(self) -> None:
        contract = _make_contract()
        results = check_publications(contract, watermark_fn=_make_watermark_fn())
        assert len(results) == len(SAMPLE_TOPICS)
        assert all(r.verdict == EnumValidationVerdict.PASS for r in results)
        assert all(r.check_type == EnumContractCheckType.PUBLICATION for r in results)

    def test_no_topics_declared(self) -> None:
        contract = _make_contract(publish_topics=())
        results = check_publications(contract, watermark_fn=_make_watermark_fn())
        assert len(results) == 1
        assert results[0].verdict == EnumValidationVerdict.PASS
        assert "No publish_topics" in results[0].evidence

    def test_contract_name_preserved(self) -> None:
        contract = _make_contract()
        results = check_publications(contract, watermark_fn=_make_watermark_fn())
        assert all(r.contract_name == CONTRACT_NAME for r in results)


@pytest.mark.unit
class TestCheckPublicationsFail:
    """Failure-path publication checks."""

    def test_all_topics_zero_offset(self) -> None:
        contract = _make_contract()
        results = check_publications(
            contract, watermark_fn=_make_watermark_fn(high_offset=0)
        )
        assert all(r.verdict == EnumValidationVerdict.FAIL for r in results)
        assert all("zero offset" in r.evidence.lower() for r in results)

    def test_partial_offsets(self) -> None:
        per_topic = {
            "onex.evt.platform.node-registration-result.v1": 50,
            "onex.evt.platform.node-registration-initiated.v1": 0,
            "onex.evt.platform.node-became-active.v1": 10,
        }
        contract = _make_contract()
        results = check_publications(
            contract, watermark_fn=_make_watermark_fn(per_topic=per_topic)
        )
        pass_count = sum(1 for r in results if r.verdict == EnumValidationVerdict.PASS)
        fail_count = sum(1 for r in results if r.verdict == EnumValidationVerdict.FAIL)
        assert pass_count == 2
        assert fail_count == 1


@pytest.mark.unit
class TestCheckPublicationsQuarantine:
    """Degraded-state checks: watermark query fails -> QUARANTINE."""

    def test_watermark_unavailable(self) -> None:
        contract = _make_contract()
        results = check_publications(
            contract,
            watermark_fn=_make_watermark_fn(error=ConnectionError("Kafka unavailable")),
        )
        assert len(results) == len(SAMPLE_TOPICS)
        assert all(r.verdict == EnumValidationVerdict.QUARANTINE for r in results)

    def test_quarantine_evidence_includes_error(self) -> None:
        contract = _make_contract()
        results = check_publications(
            contract,
            watermark_fn=_make_watermark_fn(
                error=ConnectionError("Connection refused")
            ),
        )
        assert all("Connection refused" in r.evidence for r in results)


@pytest.mark.unit
class TestCheckPublicationsSeverity:
    """Severity classification: core registration topics vs others."""

    def test_core_topics_are_required(self) -> None:
        core_topic = "onex.evt.platform.node-registration-result.v1"
        contract = _make_contract(publish_topics=(core_topic,))
        results = check_publications(
            contract, watermark_fn=_make_watermark_fn(high_offset=0)
        )
        assert results[0].severity == EnumCheckSeverity.REQUIRED

    def test_non_core_topics_are_recommended(self) -> None:
        non_core_topic = "onex.evt.platform.node-became-active.v1"
        contract = _make_contract(publish_topics=(non_core_topic,))
        results = check_publications(
            contract, watermark_fn=_make_watermark_fn(high_offset=0)
        )
        assert results[0].severity == EnumCheckSeverity.RECOMMENDED

    def test_core_registration_suffixes(self) -> None:
        assert "node-registration-result" in CORE_REGISTRATION_SUFFIXES
        assert "node-registration-initiated" in CORE_REGISTRATION_SUFFIXES
        assert len(CORE_REGISTRATION_SUFFIXES) == 2

    def test_mixed_severity_in_results(self) -> None:
        contract = _make_contract()
        results = check_publications(contract, watermark_fn=_make_watermark_fn())
        severities = {r.severity for r in results}
        assert EnumCheckSeverity.REQUIRED in severities
        assert EnumCheckSeverity.RECOMMENDED in severities


@pytest.mark.unit
class TestCheckPublicationsOnePerTopic:
    """One result per publish_topic."""

    def test_result_count_matches_topic_count(self) -> None:
        topics = ("topic.a", "topic.b", "topic.c", "topic.d", "topic.e")
        contract = _make_contract(publish_topics=topics)
        results = check_publications(contract, watermark_fn=_make_watermark_fn())
        assert len(results) == 5

    def test_single_topic(self) -> None:
        contract = _make_contract(publish_topics=("only.topic",))
        results = check_publications(contract, watermark_fn=_make_watermark_fn())
        assert len(results) == 1
        assert results[0].verdict == EnumValidationVerdict.PASS
