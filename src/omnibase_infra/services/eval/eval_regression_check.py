# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Eval regression check for close-out autopilot.

Checks whether the latest eval report shows ONEX performing worse than
baseline on more than a configurable threshold of tasks. If so, returns
a regression alert that the close-out skill should act on (e.g., create
a Linear ticket).

Related:
    - OMN-6782: Build eval regression check for close-out
    - OMN-6776: Eval orchestrator skill
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from onex_change_control.enums.enum_eval_verdict import EnumEvalVerdict
from onex_change_control.models.model_eval_report import ModelEvalReport

logger = logging.getLogger(__name__)

# Default: flag regression if ONEX is worse on >30% of tasks
DEFAULT_REGRESSION_THRESHOLD = 0.30


@dataclass(frozen=True)
class EvalRegressionResult:
    """Result of an eval regression check.

    Attributes:
        is_regression: True if ONEX is worse on more than threshold% of tasks.
        worse_count: Number of tasks where ONEX performed worse.
        total_tasks: Total tasks in the report.
        worse_ratio: Fraction of tasks where ONEX was worse.
        threshold: The regression threshold used.
        report_id: The eval report that was checked.
        suite_id: The eval suite that was evaluated.
    """

    is_regression: bool
    worse_count: int
    total_tasks: int
    worse_ratio: float
    threshold: float
    report_id: str
    suite_id: str

    @property
    def summary(self) -> str:
        """Human-readable summary of the regression check."""
        if self.is_regression:
            return (
                f"REGRESSION: ONEX worse on {self.worse_count}/{self.total_tasks} "
                f"tasks ({self.worse_ratio:.0%} > {self.threshold:.0%} threshold) "
                f"[report={self.report_id}]"
            )
        return (
            f"OK: ONEX worse on {self.worse_count}/{self.total_tasks} "
            f"tasks ({self.worse_ratio:.0%} <= {self.threshold:.0%} threshold) "
            f"[report={self.report_id}]"
        )


def check_eval_regression(
    report: ModelEvalReport,
    threshold: float = DEFAULT_REGRESSION_THRESHOLD,
) -> EvalRegressionResult:
    """Check if the latest eval report shows a regression.

    Args:
        report: The eval report to check.
        threshold: Fraction of tasks that must be worse to flag regression.
            Must be in [0.0, 1.0].

    Returns:
        EvalRegressionResult with the check outcome.

    Raises:
        ValueError: If threshold is outside [0.0, 1.0].
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")

    total = report.summary.total_tasks
    if total == 0:
        logger.warning(
            "Empty eval report %s, skipping regression check", report.report_id
        )
        return EvalRegressionResult(
            is_regression=False,
            worse_count=0,
            total_tasks=0,
            worse_ratio=0.0,
            threshold=threshold,
            report_id=report.report_id,
            suite_id=report.suite_id,
        )

    worse_count = report.summary.onex_worse_count
    worse_ratio = worse_count / total

    result = EvalRegressionResult(
        is_regression=worse_ratio > threshold,
        worse_count=worse_count,
        total_tasks=total,
        worse_ratio=worse_ratio,
        threshold=threshold,
        report_id=report.report_id,
        suite_id=report.suite_id,
    )

    if result.is_regression:
        logger.warning(result.summary)
    else:
        logger.info(result.summary)

    return result


__all__: list[str] = [
    "DEFAULT_REGRESSION_THRESHOLD",
    "EvalRegressionResult",
    "check_eval_regression",
]
