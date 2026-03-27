# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for post-merge consumer models.

Related Tickets:
    - OMN-6727: post-merge consumer chain
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from omnibase_infra.services.post_merge.enum_check_stage import EnumCheckStage
from omnibase_infra.services.post_merge.enum_finding_severity import (
    EnumFindingSeverity,
)
from omnibase_infra.services.post_merge.model_post_merge_finding import (
    ModelPostMergeFinding,
)
from omnibase_infra.services.post_merge.model_post_merge_result import (
    ModelPostMergeResult,
)


@pytest.mark.unit
class TestModelPostMergeFinding:
    """Tests for ModelPostMergeFinding."""

    def test_create_minimal(self) -> None:
        finding = ModelPostMergeFinding(
            stage=EnumCheckStage.HOSTILE_REVIEW,
            severity=EnumFindingSeverity.HIGH,
            title="Test finding",
            description="A test finding description",
        )
        assert finding.stage == EnumCheckStage.HOSTILE_REVIEW
        assert finding.severity == EnumFindingSeverity.HIGH
        assert finding.file_path is None
        assert finding.line_number is None

    def test_create_with_file_info(self) -> None:
        finding = ModelPostMergeFinding(
            stage=EnumCheckStage.CONTRACT_SWEEP,
            severity=EnumFindingSeverity.CRITICAL,
            title="Drift detected",
            description="Topic drift in contract",
            file_path="src/foo.py",
            line_number=42,
        )
        assert finding.file_path == "src/foo.py"
        assert finding.line_number == 42

    def test_frozen(self) -> None:
        finding = ModelPostMergeFinding(
            stage=EnumCheckStage.HOSTILE_REVIEW,
            severity=EnumFindingSeverity.LOW,
            title="Test",
            description="Test",
        )
        with pytest.raises(Exception):
            finding.title = "modified"  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(Exception):
            ModelPostMergeFinding(
                stage=EnumCheckStage.HOSTILE_REVIEW,
                severity=EnumFindingSeverity.LOW,
                title="Test",
                description="Test",
                unknown_field="bad",  # type: ignore[call-arg]
            )

    def test_line_number_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            ModelPostMergeFinding(
                stage=EnumCheckStage.HOSTILE_REVIEW,
                severity=EnumFindingSeverity.LOW,
                title="Test",
                description="Test",
                line_number=0,
            )

    def test_serialization_roundtrip(self) -> None:
        finding = ModelPostMergeFinding(
            stage=EnumCheckStage.INTEGRATION_CHECK,
            severity=EnumFindingSeverity.MEDIUM,
            title="API change",
            description="Public API modified",
            file_path="src/__init__.py",
        )
        data = finding.model_dump()
        restored = ModelPostMergeFinding.model_validate(data)
        assert restored == finding


@pytest.mark.unit
class TestModelPostMergeResult:
    """Tests for ModelPostMergeResult."""

    def test_create_minimal(self) -> None:
        now = datetime.now(tz=UTC)
        result = ModelPostMergeResult(
            repo="OmniNode-ai/omnibase_infra",
            pr_number=42,
            merge_sha="abc123",
            started_at=now,
            completed_at=now,
        )
        assert result.repo == "OmniNode-ai/omnibase_infra"
        assert result.findings == []
        assert result.stages_completed == []
        assert result.tickets_created == []

    def test_with_findings(self) -> None:
        now = datetime.now(tz=UTC)
        finding = ModelPostMergeFinding(
            stage=EnumCheckStage.HOSTILE_REVIEW,
            severity=EnumFindingSeverity.HIGH,
            title="Secret found",
            description="AWS key detected",
        )
        result = ModelPostMergeResult(
            repo="OmniNode-ai/omnibase_infra",
            pr_number=99,
            merge_sha="def456",
            findings=[finding],
            stages_completed=[EnumCheckStage.HOSTILE_REVIEW],
            tickets_created=["OMN-9999"],
            started_at=now,
            completed_at=now,
        )
        assert len(result.findings) == 1
        assert result.tickets_created == ["OMN-9999"]

    def test_json_roundtrip(self) -> None:
        now = datetime.now(tz=UTC)
        result = ModelPostMergeResult(
            repo="OmniNode-ai/test",
            pr_number=1,
            merge_sha="aaa",
            started_at=now,
            completed_at=now,
        )
        json_str = result.model_dump_json()
        restored = ModelPostMergeResult.model_validate_json(json_str)
        assert restored.repo == result.repo


@pytest.mark.unit
class TestEnumCheckStage:
    """Tests for EnumCheckStage values."""

    def test_all_stages(self) -> None:
        assert len(EnumCheckStage) == 3
        assert "hostile_review" in [s.value for s in EnumCheckStage]
        assert "contract_sweep" in [s.value for s in EnumCheckStage]
        assert "integration_check" in [s.value for s in EnumCheckStage]


@pytest.mark.unit
class TestEnumFindingSeverity:
    """Tests for EnumFindingSeverity values."""

    def test_all_severities(self) -> None:
        assert len(EnumFindingSeverity) == 5
        expected = {"critical", "high", "medium", "low", "info"}
        assert {s.value for s in EnumFindingSeverity} == expected
