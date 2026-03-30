# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for eval task models and enums."""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.enums.enum_eval_finding_category import EnumEvalFindingCategory
from omnibase_infra.enums.enum_eval_finding_severity import EnumEvalFindingSeverity
from omnibase_infra.enums.enum_eval_task_status import EnumEvalTaskStatus
from omnibase_infra.enums.enum_eval_task_type import EnumEvalTaskType
from omnibase_infra.models.eval.model_eval_budget_cap import ModelEvalBudgetCap
from omnibase_infra.models.eval.model_eval_finding import ModelEvalFinding
from omnibase_infra.models.eval.model_eval_result import ModelEvalResult
from omnibase_infra.models.eval.model_eval_task import ModelEvalTask


@pytest.mark.unit
class TestEnumEvalTaskType:
    """Tests for EnumEvalTaskType."""

    def test_all_values_are_strings(self) -> None:
        for member in EnumEvalTaskType:
            assert isinstance(member.value, str)

    def test_expected_members(self) -> None:
        expected = {
            "CODE_REVIEW",
            "TECH_DEBT_SCAN",
            "DOC_FRESHNESS",
            "REGRESSION_TEST",
            "HOSTILE_REVIEW",
        }
        actual = {m.name for m in EnumEvalTaskType}
        assert actual == expected


@pytest.mark.unit
class TestModelEvalTask:
    """Tests for ModelEvalTask."""

    def test_creates_with_defaults(self) -> None:
        task = ModelEvalTask(
            task_type=EnumEvalTaskType.CODE_REVIEW,
            target_repo="omnibase_infra",
        )
        assert task.task_type == EnumEvalTaskType.CODE_REVIEW
        assert task.target_repo == "omnibase_infra"
        assert task.status == EnumEvalTaskStatus.PENDING
        assert task.max_tokens == 4096

    def test_serialization_roundtrip(self) -> None:
        task = ModelEvalTask(
            task_type=EnumEvalTaskType.TECH_DEBT_SCAN,
            target_repo="omnibase_core",
            target_path="src/omnibase_core/nodes",
            max_tokens=8192,
        )
        data = task.model_dump_json()
        restored = ModelEvalTask.model_validate_json(data)
        assert restored == task

    def test_frozen(self) -> None:
        task = ModelEvalTask(
            task_type=EnumEvalTaskType.CODE_REVIEW,
            target_repo="test",
        )
        with pytest.raises(Exception):
            task.status = EnumEvalTaskStatus.RUNNING  # type: ignore[misc]


@pytest.mark.unit
class TestModelEvalFinding:
    """Tests for ModelEvalFinding."""

    def test_creates_with_required_fields(self) -> None:
        finding = ModelEvalFinding(
            severity=EnumEvalFindingSeverity.HIGH,
            category=EnumEvalFindingCategory.BUG,
            description="Potential null deref",
        )
        assert finding.severity == EnumEvalFindingSeverity.HIGH
        assert finding.category == EnumEvalFindingCategory.BUG

    def test_frozen(self) -> None:
        finding = ModelEvalFinding(
            severity=EnumEvalFindingSeverity.LOW,
            category=EnumEvalFindingCategory.STYLE,
            description="Minor style issue",
        )
        with pytest.raises(Exception):
            finding.description = "changed"  # type: ignore[misc]


@pytest.mark.unit
class TestModelEvalResult:
    """Tests for ModelEvalResult."""

    def test_creates_with_findings(self) -> None:
        finding = ModelEvalFinding(
            severity=EnumEvalFindingSeverity.HIGH,
            category=EnumEvalFindingCategory.BUG,
            description="Potential null deref",
            file_path="src/foo.py",
            line_number=42,
        )
        result = ModelEvalResult(
            task_id=uuid4(),
            findings=(finding,),
            high_count=1,
            tokens_used=1500,
            llm_model_label="gemini-2.0-flash",
            llm_provider_label="gemini",
        )
        assert result.high_count == 1
        assert len(result.findings) == 1

    def test_serialization_roundtrip(self) -> None:
        result = ModelEvalResult(
            task_id=uuid4(),
            tokens_used=2000,
            actual_cost_usd=0.001,
            duration_ms=350,
        )
        data = result.model_dump_json()
        restored = ModelEvalResult.model_validate_json(data)
        assert restored == result


@pytest.mark.unit
class TestModelEvalBudgetCap:
    """Tests for ModelEvalBudgetCap."""

    def test_defaults(self) -> None:
        cap = ModelEvalBudgetCap()
        assert cap.max_tokens_per_window == 500_000
        assert cap.max_cost_usd_per_window == 1.0
        assert cap.window_hours == 24

    def test_frozen(self) -> None:
        cap = ModelEvalBudgetCap()
        with pytest.raises(Exception):
            cap.window_hours = 48  # type: ignore[misc]
