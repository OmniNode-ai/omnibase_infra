# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A/B Eval Runner Service.

Executes eval tasks from a ModelEvalSuite in both ONEX_ON and ONEX_OFF modes,
collecting metrics for each run. The runner toggles ENABLE_* feature flags
between modes and records environment state per run.

Related:
    - OMN-6773: Build eval runner service
    - OMN-6770: ModelEvalTask / ModelEvalSuite (onex_change_control)
    - OMN-6771: ModelEvalRun / EnumEvalMode (onex_change_control)
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from onex_change_control.enums.enum_eval_metric_type import EnumEvalMetricType
from onex_change_control.enums.enum_eval_mode import EnumEvalMode
from onex_change_control.models.model_eval_run import ModelEvalMetric, ModelEvalRun
from onex_change_control.models.model_eval_task import ModelEvalSuite, ModelEvalTask

logger = logging.getLogger(__name__)

# Feature flags toggled between ONEX_ON and ONEX_OFF modes.
_ONEX_FEATURE_FLAGS: list[str] = [
    "ENABLE_REAL_TIME_EVENTS",
    "ENABLE_PATTERN_ENFORCEMENT",
    "ENABLE_CONSUMER_HEALTH_EMITTER",
    "ENABLE_CONSUMER_HEALTH_TRIAGE",
    "ENABLE_LOCAL_INFERENCE_PIPELINE",
]


def _capture_env_snapshot() -> dict[str, str]:
    """Capture current state of all ENABLE_* flags."""
    return {key: os.environ.get(key, "") for key in sorted(_ONEX_FEATURE_FLAGS)}


def _set_mode_flags(mode: EnumEvalMode) -> None:
    """Set ENABLE_* flags for the given eval mode.

    ONEX_ON: all flags set to 'true'.
    ONEX_OFF: all flags set to 'false'.
    """
    value = "true" if mode == EnumEvalMode.ONEX_ON else "false"
    for flag in _ONEX_FEATURE_FLAGS:
        os.environ[flag] = value


def _get_git_sha(repo_path: str) -> str:
    """Get current git SHA for the repo, or 'unknown' on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown"


def _run_setup_commands(commands: list[str], cwd: str) -> str | None:
    """Run setup commands sequentially. Returns error message or None."""
    for cmd in commands:
        try:
            result = subprocess.run(  # noqa: S602 -- eval tasks require shell
                cmd,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if result.returncode != 0:
                return f"Setup command failed: {cmd!r} -> {result.stderr.strip()}"
        except subprocess.TimeoutExpired:
            return f"Setup command timed out: {cmd!r}"
    return None


def _check_success_criteria(
    criteria: list[str],
    cwd: str,
    timeout: int,
) -> tuple[bool, int, int]:
    """Check machine-checkable success criteria.

    Returns (all_passed, passed_count, total_count).
    """
    if not criteria:
        return True, 0, 0

    passed = 0
    for criterion in criteria:
        # Criteria are shell commands that should exit 0 on success
        try:
            result = subprocess.run(  # noqa: S602 -- eval criteria are shell commands
                criterion,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if result.returncode == 0:
                passed += 1
        except subprocess.TimeoutExpired:
            pass  # Criterion timed out — counted as not passed
    return passed == len(criteria), passed, len(criteria)


class ServiceEvalRunner:
    """Runs eval tasks in ONEX_ON and ONEX_OFF modes, collecting metrics.

    Args:
        workspace_root: Path to the workspace root containing repo directories.
    """

    def __init__(self, workspace_root: str) -> None:
        self._workspace_root = workspace_root

    def _repo_path(self, repo: str) -> str:
        """Resolve absolute path for a repo name."""
        return str(Path(self._workspace_root) / repo)

    def run_task(
        self,
        task: ModelEvalTask,
        mode: EnumEvalMode,
    ) -> ModelEvalRun:
        """Execute a single eval task in the specified mode.

        Sets ENABLE_* flags, runs setup commands, executes success criteria,
        and collects timing metrics.
        """
        run_id = f"eval-run-{uuid4().hex[:12]}"
        repo_path = self._repo_path(task.repo)
        git_sha = _get_git_sha(repo_path)

        # Toggle feature flags
        _set_mode_flags(mode)
        env_snapshot = _capture_env_snapshot()

        started_at = datetime.now(UTC)
        start_mono = time.monotonic()

        # Run setup commands
        setup_error = _run_setup_commands(task.setup_commands, repo_path)
        if setup_error is not None:
            return ModelEvalRun(
                run_id=run_id,
                task_id=task.task_id,
                mode=mode,
                started_at=started_at,
                completed_at=datetime.now(UTC),
                success=False,
                error_message=setup_error,
                git_sha=git_sha,
                env_snapshot=env_snapshot,
            )

        # Check success criteria
        all_passed, passed_count, total_count = _check_success_criteria(
            task.success_criteria,
            repo_path,
            task.max_duration_seconds,
        )

        elapsed_ms = (time.monotonic() - start_mono) * 1000.0
        completed_at = datetime.now(UTC)

        metrics: list[ModelEvalMetric] = [
            ModelEvalMetric(
                metric_type=EnumEvalMetricType.LATENCY_MS,
                value=elapsed_ms,
                unit="ms",
            ),
        ]
        if total_count > 0:
            metrics.append(
                ModelEvalMetric(
                    metric_type=EnumEvalMetricType.SUCCESS_RATE,
                    value=passed_count / total_count,
                    unit="ratio",
                )
            )

        return ModelEvalRun(
            run_id=run_id,
            task_id=task.task_id,
            mode=mode,
            started_at=started_at,
            completed_at=completed_at,
            success=all_passed,
            metrics=metrics,
            git_sha=git_sha,
            env_snapshot=env_snapshot,
        )

    def run_suite(
        self,
        suite: ModelEvalSuite,
        mode: EnumEvalMode,
    ) -> list[ModelEvalRun]:
        """Run all tasks in a suite under a single mode.

        Returns one ModelEvalRun per task.
        """
        runs: list[ModelEvalRun] = []
        for task in suite.tasks:
            logger.info(
                "Running task %s in %s mode",
                task.task_id,
                mode.value,
            )
            run = self.run_task(task, mode)
            runs.append(run)
            logger.info(
                "Task %s %s (%.0f ms)",
                task.task_id,
                "PASSED" if run.success else "FAILED",
                next(
                    (
                        m.value
                        for m in run.metrics
                        if m.metric_type == EnumEvalMetricType.LATENCY_MS
                    ),
                    0.0,
                ),
            )
        return runs

    def run_ab_suite(
        self,
        suite: ModelEvalSuite,
    ) -> tuple[list[ModelEvalRun], list[ModelEvalRun]]:
        """Run the full A/B eval: each task in both ONEX_ON and ONEX_OFF.

        Returns (on_runs, off_runs).
        """
        logger.info(
            "Starting A/B eval for suite %s (%d tasks)",
            suite.suite_id,
            len(suite.tasks),
        )
        on_runs = self.run_suite(suite, EnumEvalMode.ONEX_ON)
        off_runs = self.run_suite(suite, EnumEvalMode.ONEX_OFF)
        logger.info("A/B eval complete for suite %s", suite.suite_id)
        return on_runs, off_runs


__all__: list[str] = ["ServiceEvalRunner"]
