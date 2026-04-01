# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for ServiceEvalRunner [OMN-6773]."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from onex_change_control.enums.enum_eval_metric_type import EnumEvalMetricType
from onex_change_control.enums.enum_eval_mode import EnumEvalMode
from onex_change_control.models.model_eval_task import ModelEvalSuite, ModelEvalTask

from omnibase_infra.services.eval.service_eval_runner import (
    ServiceEvalRunner,
    _capture_env_snapshot,
    _set_mode_flags,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Create a workspace with a test_repo directory."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    return tmp_path


@pytest.fixture
def sample_task() -> ModelEvalTask:
    return ModelEvalTask(
        task_id="test-001",
        name="Test task",
        category="bug-fix",
        prompt="Fix the bug",
        repo="test_repo",
        setup_commands=[],
        success_criteria=["true"],
        max_duration_seconds=30,
    )


@pytest.fixture
def sample_suite(sample_task: ModelEvalTask) -> ModelEvalSuite:
    return ModelEvalSuite(
        suite_id="test-suite",
        name="Test Suite",
        tasks=[sample_task],
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        version="1.0.0",
    )


@pytest.mark.unit
class TestSetModeFlags:
    def test_onex_on_sets_all_true(self) -> None:
        _set_mode_flags(EnumEvalMode.ONEX_ON)
        for key in _capture_env_snapshot():
            assert os.environ[key] == "true"

    def test_onex_off_sets_all_false(self) -> None:
        _set_mode_flags(EnumEvalMode.ONEX_OFF)
        for key in _capture_env_snapshot():
            assert os.environ[key] == "false"


@pytest.mark.unit
class TestCaptureEnvSnapshot:
    def test_returns_dict_of_flags(self) -> None:
        _set_mode_flags(EnumEvalMode.ONEX_ON)
        snapshot = _capture_env_snapshot()
        assert isinstance(snapshot, dict)
        assert "ENABLE_REAL_TIME_EVENTS" in snapshot


@pytest.mark.unit
class TestServiceEvalRunner:
    def test_run_task_with_passing_criteria(
        self, sample_task: ModelEvalTask, workspace: Path
    ) -> None:
        runner = ServiceEvalRunner(workspace_root=str(workspace))
        with patch(
            "omnibase_infra.services.eval.service_eval_runner._get_git_sha",
            return_value="abc123",
        ):
            run = runner.run_task(sample_task, EnumEvalMode.ONEX_ON)

        assert run.task_id == "test-001"
        assert run.mode == EnumEvalMode.ONEX_ON
        assert run.success is True
        assert run.git_sha == "abc123"
        assert any(m.metric_type == EnumEvalMetricType.LATENCY_MS for m in run.metrics)
        assert any(
            m.metric_type == EnumEvalMetricType.SUCCESS_RATE for m in run.metrics
        )

    def test_run_task_with_failing_criteria(self, workspace: Path) -> None:
        task = ModelEvalTask(
            task_id="test-fail",
            name="Failing task",
            category="bug-fix",
            prompt="Fix it",
            repo="test_repo",
            setup_commands=[],
            success_criteria=["false"],
            max_duration_seconds=30,
        )
        runner = ServiceEvalRunner(workspace_root=str(workspace))
        with patch(
            "omnibase_infra.services.eval.service_eval_runner._get_git_sha",
            return_value="abc123",
        ):
            run = runner.run_task(task, EnumEvalMode.ONEX_OFF)

        assert run.success is False
        assert run.mode == EnumEvalMode.ONEX_OFF

    def test_run_task_setup_failure(self, workspace: Path) -> None:
        task = ModelEvalTask(
            task_id="test-setup-fail",
            name="Setup fail task",
            category="bug-fix",
            prompt="Fix it",
            repo="test_repo",
            setup_commands=["exit 1"],
            success_criteria=["true"],
            max_duration_seconds=30,
        )
        runner = ServiceEvalRunner(workspace_root=str(workspace))
        with patch(
            "omnibase_infra.services.eval.service_eval_runner._get_git_sha",
            return_value="abc123",
        ):
            run = runner.run_task(task, EnumEvalMode.ONEX_ON)

        assert run.success is False
        assert run.error_message is not None
        assert "Setup command failed" in run.error_message

    def test_run_suite(self, sample_suite: ModelEvalSuite, workspace: Path) -> None:
        runner = ServiceEvalRunner(workspace_root=str(workspace))
        with patch(
            "omnibase_infra.services.eval.service_eval_runner._get_git_sha",
            return_value="abc123",
        ):
            runs = runner.run_suite(sample_suite, EnumEvalMode.ONEX_ON)

        assert len(runs) == 1
        assert runs[0].task_id == "test-001"

    def test_run_ab_suite(self, sample_suite: ModelEvalSuite, workspace: Path) -> None:
        runner = ServiceEvalRunner(workspace_root=str(workspace))
        with patch(
            "omnibase_infra.services.eval.service_eval_runner._get_git_sha",
            return_value="abc123",
        ):
            on_runs, off_runs = runner.run_ab_suite(sample_suite)

        assert len(on_runs) == 1
        assert len(off_runs) == 1
        assert on_runs[0].mode == EnumEvalMode.ONEX_ON
        assert off_runs[0].mode == EnumEvalMode.ONEX_OFF
