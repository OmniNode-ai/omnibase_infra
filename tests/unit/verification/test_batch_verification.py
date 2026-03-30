# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Tests for multi-contract batch verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.enums.enum_check_severity import EnumCheckSeverity
from omnibase_infra.enums.enum_contract_check_type import EnumContractCheckType
from omnibase_infra.enums.enum_validation_verdict import EnumValidationVerdict
from omnibase_infra.verification.batch import (
    BatchVerificationConfig,
    _compute_overall_verdict,
    _discover_contracts,
    _get_probe_requirement,
    _verify_single_contract,
    run_batch_verification,
)
from omnibase_infra.verification.contract_parser import (
    ModelParsedContractForVerification,
)
from omnibase_infra.verification.models import (
    ModelContractCheckResult,
)

# -- Fixtures --


def _make_contract_yaml(node_dir: Path, node_type: str, name: str) -> Path:
    """Write a minimal contract.yaml into the given directory."""
    node_dir.mkdir(parents=True, exist_ok=True)
    contract_path = node_dir / "contract.yaml"
    data = {
        "name": name,
        "node_type": node_type,
        "event_bus": {
            "subscribe_topics": ["onex.evt.test.input.v1"],
            "publish_topics": ["onex.evt.test.output.v1"],
        },
    }
    contract_path.write_text(yaml.dump(data))
    return contract_path


def _stub_db_query(sql: str) -> list[dict[str, Any]]:
    """Stub DB query returning a registration row."""
    return [{"node_name": "test_node", "current_state": "active"}]


def _stub_kafka_admin(group_id: str) -> set[str]:
    """Stub Kafka admin returning subscribed topics."""
    return {"onex.evt.test.input.v1"}


def _stub_watermark(topic: str) -> tuple[int, int]:
    """Stub watermark returning non-zero offsets."""
    return (0, 42)


def _make_parsed(
    name: str = "test_orch",
    node_type: str = "ORCHESTRATOR_GENERIC",
) -> ModelParsedContractForVerification:
    return ModelParsedContractForVerification(
        name=name,
        node_type=node_type,
        subscribe_topics=("onex.evt.test.input.v1",),
        publish_topics=("onex.evt.test.output.v1",),
    )


# -- Probe matrix tests --


@pytest.mark.unit
class TestProbeMatrix:
    def test_orchestrator_registration_required(self) -> None:
        assert (
            _get_probe_requirement("ORCHESTRATOR_GENERIC", "REGISTRATION") == "REQUIRED"
        )

    def test_orchestrator_subscription_required(self) -> None:
        assert (
            _get_probe_requirement("ORCHESTRATOR_GENERIC", "SUBSCRIPTION") == "REQUIRED"
        )

    def test_orchestrator_handler_optional(self) -> None:
        assert (
            _get_probe_requirement("ORCHESTRATOR_GENERIC", "HANDLER_EXECUTION")
            == "OPTIONAL"
        )

    def test_reducer_registration_skip(self) -> None:
        assert _get_probe_requirement("REDUCER_GENERIC", "REGISTRATION") == "SKIP"

    def test_reducer_subscription_required(self) -> None:
        assert _get_probe_requirement("REDUCER_GENERIC", "SUBSCRIPTION") == "REQUIRED"

    def test_effect_publication_recommended(self) -> None:
        assert _get_probe_requirement("EFFECT_GENERIC", "PUBLICATION") == "RECOMMENDED"

    def test_effect_projection_required(self) -> None:
        assert (
            _get_probe_requirement("EFFECT_GENERIC", "PROJECTION_STATE") == "REQUIRED"
        )

    def test_compute_all_skip(self) -> None:
        for probe in (
            "REGISTRATION",
            "SUBSCRIPTION",
            "PUBLICATION",
            "HANDLER_EXECUTION",
            "PROJECTION_STATE",
        ):
            assert _get_probe_requirement("COMPUTE_GENERIC", probe) == "SKIP"

    def test_unknown_type_falls_back_to_compute(self) -> None:
        assert _get_probe_requirement("BANANA_NODE", "REGISTRATION") == "SKIP"


# -- Verdict aggregation tests --


@pytest.mark.unit
class TestComputeOverallVerdict:
    def test_all_pass(self) -> None:
        checks = (
            ModelContractCheckResult(
                check_type=EnumContractCheckType.REGISTRATION,
                severity=EnumCheckSeverity.REQUIRED,
                verdict=EnumValidationVerdict.PASS,
                evidence="ok",
                contract_name="test",
                message="ok",
            ),
        )
        assert _compute_overall_verdict(checks) == EnumValidationVerdict.PASS

    def test_required_fail_yields_fail(self) -> None:
        checks = (
            ModelContractCheckResult(
                check_type=EnumContractCheckType.REGISTRATION,
                severity=EnumCheckSeverity.REQUIRED,
                verdict=EnumValidationVerdict.FAIL,
                evidence="missing",
                contract_name="test",
                message="fail",
            ),
        )
        assert _compute_overall_verdict(checks) == EnumValidationVerdict.FAIL

    def test_recommended_fail_does_not_yield_fail(self) -> None:
        checks = (
            ModelContractCheckResult(
                check_type=EnumContractCheckType.PUBLICATION,
                severity=EnumCheckSeverity.RECOMMENDED,
                verdict=EnumValidationVerdict.FAIL,
                evidence="not published",
                contract_name="test",
                message="fail",
            ),
        )
        assert _compute_overall_verdict(checks) == EnumValidationVerdict.PASS

    def test_quarantine_propagates(self) -> None:
        checks = (
            ModelContractCheckResult(
                check_type=EnumContractCheckType.SUBSCRIPTION,
                severity=EnumCheckSeverity.REQUIRED,
                verdict=EnumValidationVerdict.QUARANTINE,
                evidence="infra down",
                contract_name="test",
                message="quarantine",
            ),
        )
        assert _compute_overall_verdict(checks) == EnumValidationVerdict.QUARANTINE


# -- Contract discovery tests --


@pytest.mark.unit
class TestDiscoverContracts:
    def test_discovers_contracts(self, tmp_path: Path) -> None:
        _make_contract_yaml(tmp_path / "node_a", "EFFECT_GENERIC", "a")
        _make_contract_yaml(tmp_path / "node_b", "REDUCER_GENERIC", "b")
        paths = _discover_contracts(tmp_path)
        assert len(paths) == 2

    def test_empty_dir(self, tmp_path: Path) -> None:
        paths = _discover_contracts(tmp_path)
        assert paths == []

    def test_nonexistent_dir(self, tmp_path: Path) -> None:
        paths = _discover_contracts(tmp_path / "nope")
        assert paths == []


# -- Single contract verification tests --


@pytest.mark.unit
class TestVerifySingleContract:
    def test_compute_node_skips_all_probes(self) -> None:
        parsed = _make_parsed(name="compute_node", node_type="COMPUTE_GENERIC")
        config = BatchVerificationConfig()
        report = _verify_single_contract(parsed, config)
        assert report.overall_verdict == EnumValidationVerdict.PASS
        assert len(report.checks) == 0

    def test_orchestrator_with_stubs_passes(self) -> None:
        parsed = _make_parsed(name="test_orch", node_type="ORCHESTRATOR_GENERIC")
        config = BatchVerificationConfig(
            db_query_fn=_stub_db_query,
            kafka_admin_fn=_stub_kafka_admin,
            watermark_fn=_stub_watermark,
        )
        report = _verify_single_contract(parsed, config)
        assert len(report.checks) > 0
        assert report.contract_name == "test_orch"
        assert report.duration_ms >= 0

    def test_orchestrator_without_db_quarantines(self) -> None:
        parsed = _make_parsed(name="test_orch", node_type="ORCHESTRATOR_GENERIC")
        config = BatchVerificationConfig(
            kafka_admin_fn=_stub_kafka_admin,
            watermark_fn=_stub_watermark,
        )
        report = _verify_single_contract(parsed, config)
        # Registration and projection checks should quarantine without DB
        quarantined = [
            c for c in report.checks if c.verdict == EnumValidationVerdict.QUARANTINE
        ]
        assert len(quarantined) >= 1

    def test_effect_runs_subscription_and_projection(self) -> None:
        parsed = _make_parsed(name="test_effect", node_type="EFFECT_GENERIC")
        config = BatchVerificationConfig(
            db_query_fn=_stub_db_query,
            kafka_admin_fn=_stub_kafka_admin,
            watermark_fn=_stub_watermark,
        )
        report = _verify_single_contract(parsed, config)
        check_types = {c.check_type for c in report.checks}
        assert EnumContractCheckType.SUBSCRIPTION in check_types
        assert EnumContractCheckType.PROJECTION_STATE in check_types

    def test_report_has_fingerprint(self) -> None:
        parsed = _make_parsed(name="test_orch", node_type="ORCHESTRATOR_GENERIC")
        config = BatchVerificationConfig(
            db_query_fn=_stub_db_query,
            kafka_admin_fn=_stub_kafka_admin,
            watermark_fn=_stub_watermark,
        )
        report = _verify_single_contract(parsed, config)
        assert len(report.report_fingerprint) == 64  # SHA-256 hex


# -- Batch run tests --


@pytest.mark.unit
class TestRunBatchVerification:
    def test_batch_walks_all_contracts(self, tmp_path: Path) -> None:
        _make_contract_yaml(tmp_path / "node_a", "EFFECT_GENERIC", "node_a")
        _make_contract_yaml(tmp_path / "node_b", "COMPUTE_GENERIC", "node_b")
        _make_contract_yaml(tmp_path / "node_c", "REDUCER_GENERIC", "node_c")

        config = BatchVerificationConfig(
            db_query_fn=_stub_db_query,
            kafka_admin_fn=_stub_kafka_admin,
            watermark_fn=_stub_watermark,
        )
        reports = run_batch_verification(tmp_path, config)
        assert len(reports) == 3

    def test_batch_skips_unparseable_contracts(self, tmp_path: Path) -> None:
        _make_contract_yaml(tmp_path / "node_good", "EFFECT_GENERIC", "good")
        bad_dir = tmp_path / "node_bad"
        bad_dir.mkdir()
        (bad_dir / "contract.yaml").write_text("{{not valid yaml")

        config = BatchVerificationConfig()
        reports = run_batch_verification(tmp_path, config)
        assert len(reports) == 1
        assert reports[0].contract_name == "good"

    def test_batch_one_contract_error_does_not_stop_others(
        self, tmp_path: Path
    ) -> None:
        _make_contract_yaml(tmp_path / "node_a", "EFFECT_GENERIC", "node_a")
        _make_contract_yaml(tmp_path / "node_b", "EFFECT_GENERIC", "node_b")

        call_count = 0

        def flaky_db(sql: str) -> list[dict[str, Any]]:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("DB down")
            return [{"node_name": "x", "current_state": "active"}]

        config = BatchVerificationConfig(
            db_query_fn=flaky_db,
            kafka_admin_fn=_stub_kafka_admin,
            watermark_fn=_stub_watermark,
        )
        reports = run_batch_verification(tmp_path, config)
        assert len(reports) == 2

    def test_batch_empty_dir(self, tmp_path: Path) -> None:
        config = BatchVerificationConfig()
        reports = run_batch_verification(tmp_path, config)
        assert reports == []

    def test_compute_nodes_all_pass(self, tmp_path: Path) -> None:
        for i in range(3):
            _make_contract_yaml(
                tmp_path / f"node_{i}", "COMPUTE_GENERIC", f"compute_{i}"
            )
        config = BatchVerificationConfig()
        reports = run_batch_verification(tmp_path, config)
        assert all(r.overall_verdict == EnumValidationVerdict.PASS for r in reports)
