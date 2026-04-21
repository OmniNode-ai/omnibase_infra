# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for registration contract verification with mocked dependencies [OMN-7040]."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.enums.enum_contract_check_type import EnumContractCheckType
from omnibase_infra.enums.enum_validation_verdict import EnumValidationVerdict
from omnibase_infra.verification.verify_registration import (
    verify_registration_contract,
)

# Path to the actual registration orchestrator contract
_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_registration_orchestrator"
    / "contract.yaml"
)


def _write_runtime_config(
    tmp_path: Path,
    *,
    name: str = "runtime_config",
    environment: str = "local",
    contract_version: str = "1.0.0",
) -> Path:
    path = tmp_path / "runtime_config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": name,
                "contract_version": contract_version,
                "event_bus": {"environment": environment},
            }
        )
    )
    return path


def _make_db_query_fn(
    orchestrator_state: str | None = "active",
    projection_rows: list[dict[str, Any]] | None = None,
) -> Any:
    """Create a mock db_query_fn.

    Args:
        orchestrator_state: State for the orchestrator row. None = no row.
        projection_rows: Rows returned for the full projection query.
            Defaults to a single active row if not specified.
    """
    if projection_rows is None:
        projection_rows = [
            {"entity_id": "some_id", "current_state": "active", "node_type": "effect"},
        ]

    schema_columns: list[dict[str, Any]] = [
        {"column_name": "entity_id"},
        {"column_name": "domain"},
        {"column_name": "current_state"},
        {"column_name": "node_type"},
        {"column_name": "node_version"},
        {"column_name": "capabilities"},
    ]

    def db_query_fn(sql: str) -> list[dict[str, Any]]:
        if "information_schema" in sql:
            return schema_columns
        if "WHERE node_type = 'orchestrator'" in sql:
            if orchestrator_state is None:
                return []
            return [
                {
                    "entity_id": "abc-123",
                    "current_state": orchestrator_state,
                    "node_type": "orchestrator",
                }
            ]
        # Full projection query
        return projection_rows

    return db_query_fn


def _make_kafka_admin_fn(
    subscribed_topics: set[str] | None = None,
) -> Any:
    """Create a mock kafka_admin_fn returning subscribed topics."""
    if subscribed_topics is None:
        # Return all 7 declared subscribe topics
        subscribed_topics = {
            "onex.evt.platform.node-introspection.v1",
            "onex.evt.platform.registry-request-introspection.v1",
            "onex.intent.platform.runtime-tick.v1",
            "onex.cmd.platform.node-registration-acked.v1",
            "onex.evt.platform.node-heartbeat.v1",
            "onex.cmd.platform.topic-catalog-query.v1",
            "onex.cmd.platform.request-introspection.v1",
        }

    def kafka_admin_fn() -> set[str]:
        return subscribed_topics

    return kafka_admin_fn


def _make_watermark_fn(
    high_offset: int = 100,
) -> Any:
    """Create a mock watermark_fn returning (0, high_offset) for all topics."""

    def watermark_fn(topic: str) -> tuple[int, int]:
        return (0, high_offset)

    return watermark_fn


@pytest.mark.unit
class TestVerifyRegistrationContractAllPass:
    """All checks pass with healthy mocks."""

    def test_overall_verdict_is_pass(self) -> None:
        report = verify_registration_contract(
            db_query_fn=_make_db_query_fn(),
            kafka_admin_fn=_make_kafka_admin_fn(),
            watermark_fn=_make_watermark_fn(),
            contract_path=_CONTRACT_PATH,
        )
        assert report.overall_verdict == EnumValidationVerdict.PASS

    def test_four_checks_returned(self) -> None:
        report = verify_registration_contract(
            db_query_fn=_make_db_query_fn(),
            kafka_admin_fn=_make_kafka_admin_fn(),
            watermark_fn=_make_watermark_fn(),
            contract_path=_CONTRACT_PATH,
        )
        assert len(report.checks) == 4

    def test_check_types_present(self) -> None:
        report = verify_registration_contract(
            db_query_fn=_make_db_query_fn(),
            kafka_admin_fn=_make_kafka_admin_fn(),
            watermark_fn=_make_watermark_fn(),
            contract_path=_CONTRACT_PATH,
        )
        check_types = {c.check_type for c in report.checks}
        assert EnumContractCheckType.REGISTRATION in check_types
        assert EnumContractCheckType.SUBSCRIPTION in check_types
        assert EnumContractCheckType.PUBLICATION in check_types
        assert EnumContractCheckType.PROJECTION_STATE in check_types

    def test_contract_name(self) -> None:
        report = verify_registration_contract(
            db_query_fn=_make_db_query_fn(),
            kafka_admin_fn=_make_kafka_admin_fn(),
            watermark_fn=_make_watermark_fn(),
            contract_path=_CONTRACT_PATH,
        )
        assert report.contract_name == "node_registration_orchestrator"

    def test_node_type(self) -> None:
        report = verify_registration_contract(
            db_query_fn=_make_db_query_fn(),
            kafka_admin_fn=_make_kafka_admin_fn(),
            watermark_fn=_make_watermark_fn(),
            contract_path=_CONTRACT_PATH,
        )
        assert report.node_type == "ORCHESTRATOR_GENERIC"

    def test_fingerprint_not_empty(self) -> None:
        report = verify_registration_contract(
            db_query_fn=_make_db_query_fn(),
            kafka_admin_fn=_make_kafka_admin_fn(),
            watermark_fn=_make_watermark_fn(),
            contract_path=_CONTRACT_PATH,
        )
        assert len(report.report_fingerprint) == 64  # SHA-256 hex

    def test_duration_ms_nonneg(self) -> None:
        report = verify_registration_contract(
            db_query_fn=_make_db_query_fn(),
            kafka_admin_fn=_make_kafka_admin_fn(),
            watermark_fn=_make_watermark_fn(),
            contract_path=_CONTRACT_PATH,
        )
        assert report.duration_ms >= 0


@pytest.mark.unit
class TestVerifyRegistrationFails:
    """Individual check failures propagate correctly."""

    def test_fail_when_orchestrator_not_registered(self) -> None:
        report = verify_registration_contract(
            db_query_fn=_make_db_query_fn(orchestrator_state=None),
            kafka_admin_fn=_make_kafka_admin_fn(),
            watermark_fn=_make_watermark_fn(),
            contract_path=_CONTRACT_PATH,
        )
        assert report.overall_verdict == EnumValidationVerdict.FAIL
        reg_check = next(
            c
            for c in report.checks
            if c.check_type == EnumContractCheckType.REGISTRATION
        )
        assert reg_check.verdict == EnumValidationVerdict.FAIL

    def test_fail_when_orchestrator_idle(self) -> None:
        report = verify_registration_contract(
            db_query_fn=_make_db_query_fn(orchestrator_state="idle"),
            kafka_admin_fn=_make_kafka_admin_fn(),
            watermark_fn=_make_watermark_fn(),
            contract_path=_CONTRACT_PATH,
        )
        reg_check = next(
            c
            for c in report.checks
            if c.check_type == EnumContractCheckType.REGISTRATION
        )
        assert reg_check.verdict == EnumValidationVerdict.FAIL

    def test_fail_when_missing_subscriptions(self) -> None:
        report = verify_registration_contract(
            db_query_fn=_make_db_query_fn(),
            kafka_admin_fn=_make_kafka_admin_fn(subscribed_topics=set()),
            watermark_fn=_make_watermark_fn(),
            contract_path=_CONTRACT_PATH,
        )
        sub_check = next(
            c
            for c in report.checks
            if c.check_type == EnumContractCheckType.SUBSCRIPTION
        )
        assert sub_check.verdict == EnumValidationVerdict.FAIL

    def test_quarantine_when_runtime_identity_unavailable(self, tmp_path: Path) -> None:
        report = verify_registration_contract(
            db_query_fn=_make_db_query_fn(),
            kafka_admin_fn=_make_kafka_admin_fn(subscribed_topics=set()),
            watermark_fn=_make_watermark_fn(),
            contract_path=_CONTRACT_PATH,
            runtime_config_path=tmp_path / "missing-runtime-config.yaml",
        )
        sub_check = next(
            c
            for c in report.checks
            if c.check_type == EnumContractCheckType.SUBSCRIPTION
        )
        assert sub_check.verdict == EnumValidationVerdict.QUARANTINE
        assert "grounding=FABRICATED" in sub_check.evidence

    def test_exact_runtime_identity_preserves_fail_semantics(
        self, tmp_path: Path
    ) -> None:
        runtime_config_path = _write_runtime_config(tmp_path)
        report = verify_registration_contract(
            db_query_fn=_make_db_query_fn(),
            kafka_admin_fn=_make_kafka_admin_fn(subscribed_topics=set()),
            watermark_fn=_make_watermark_fn(),
            contract_path=_CONTRACT_PATH,
            runtime_config_path=runtime_config_path,
        )
        sub_check = next(
            c
            for c in report.checks
            if c.check_type == EnumContractCheckType.SUBSCRIPTION
        )
        assert sub_check.verdict == EnumValidationVerdict.FAIL
        assert "registration-orchestrator" in sub_check.evidence
        assert "grounding=EXACT" in sub_check.evidence

    def test_fail_when_partial_subscriptions(self) -> None:
        report = verify_registration_contract(
            db_query_fn=_make_db_query_fn(),
            kafka_admin_fn=_make_kafka_admin_fn(
                subscribed_topics={"onex.evt.platform.node-introspection.v1"}
            ),
            watermark_fn=_make_watermark_fn(),
            contract_path=_CONTRACT_PATH,
        )
        sub_check = next(
            c
            for c in report.checks
            if c.check_type == EnumContractCheckType.SUBSCRIPTION
        )
        assert sub_check.verdict == EnumValidationVerdict.FAIL
        assert "6/7" in sub_check.evidence  # 6 missing out of 7

    def test_fail_when_no_publication_data(self) -> None:
        report = verify_registration_contract(
            db_query_fn=_make_db_query_fn(),
            kafka_admin_fn=_make_kafka_admin_fn(),
            watermark_fn=_make_watermark_fn(high_offset=0),
            contract_path=_CONTRACT_PATH,
        )
        pub_check = next(
            c
            for c in report.checks
            if c.check_type == EnumContractCheckType.PUBLICATION
        )
        assert pub_check.verdict == EnumValidationVerdict.FAIL

    def test_fail_when_no_projection_rows(self) -> None:
        report = verify_registration_contract(
            db_query_fn=_make_db_query_fn(projection_rows=[]),
            kafka_admin_fn=_make_kafka_admin_fn(),
            watermark_fn=_make_watermark_fn(),
            contract_path=_CONTRACT_PATH,
        )
        proj_check = next(
            c
            for c in report.checks
            if c.check_type == EnumContractCheckType.PROJECTION_STATE
        )
        assert proj_check.verdict == EnumValidationVerdict.FAIL

    def test_fail_when_db_raises(self) -> None:
        def failing_db(sql: str) -> list[dict[str, Any]]:
            raise ConnectionError("Connection refused")

        report = verify_registration_contract(
            db_query_fn=failing_db,
            kafka_admin_fn=_make_kafka_admin_fn(),
            watermark_fn=_make_watermark_fn(),
            contract_path=_CONTRACT_PATH,
        )
        assert report.overall_verdict == EnumValidationVerdict.FAIL

    def test_fail_when_kafka_admin_raises(self) -> None:
        def failing_kafka() -> set[str]:
            raise ConnectionError("Kafka unavailable")

        report = verify_registration_contract(
            db_query_fn=_make_db_query_fn(),
            kafka_admin_fn=failing_kafka,
            watermark_fn=_make_watermark_fn(),
            contract_path=_CONTRACT_PATH,
        )
        sub_check = next(
            c
            for c in report.checks
            if c.check_type == EnumContractCheckType.SUBSCRIPTION
        )
        assert sub_check.verdict == EnumValidationVerdict.QUARANTINE
        assert "Kafka unavailable" in sub_check.evidence

    def test_fail_when_watermark_raises(self) -> None:
        def failing_watermark(topic: str) -> tuple[int, int]:
            raise ConnectionError("Kafka unavailable")

        report = verify_registration_contract(
            db_query_fn=_make_db_query_fn(),
            kafka_admin_fn=_make_kafka_admin_fn(),
            watermark_fn=failing_watermark,
            contract_path=_CONTRACT_PATH,
        )
        pub_check = next(
            c
            for c in report.checks
            if c.check_type == EnumContractCheckType.PUBLICATION
        )
        assert pub_check.verdict == EnumValidationVerdict.FAIL


@pytest.mark.unit
class TestReportBoolConversion:
    """Test __bool__ on report."""

    def test_truthy_when_pass(self) -> None:
        report = verify_registration_contract(
            db_query_fn=_make_db_query_fn(),
            kafka_admin_fn=_make_kafka_admin_fn(),
            watermark_fn=_make_watermark_fn(),
            contract_path=_CONTRACT_PATH,
        )
        assert bool(report) is True

    def test_falsy_when_fail(self) -> None:
        report = verify_registration_contract(
            db_query_fn=_make_db_query_fn(orchestrator_state=None),
            kafka_admin_fn=_make_kafka_admin_fn(),
            watermark_fn=_make_watermark_fn(),
            contract_path=_CONTRACT_PATH,
        )
        assert bool(report) is False
