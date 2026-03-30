# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Auto-eval models for autonomous LLM evaluation (OMN-6795).

Exports:
    ModelAutoEvalTask: Defines a single evaluation task to run against an LLM endpoint.
    ModelAutoEvalResult: Result of executing an eval task, with scores and cost.
    ModelAutoEvalBudgetCap: Budget constraints for eval execution windows.
"""

from omnibase_infra.models.eval.model_auto_eval_budget_cap import (
    ModelAutoEvalBudgetCap,
)
from omnibase_infra.models.eval.model_auto_eval_result import ModelAutoEvalResult
from omnibase_infra.models.eval.model_auto_eval_task import ModelAutoEvalTask

__all__: list[str] = [
    "ModelAutoEvalTask",
    "ModelAutoEvalResult",
    "ModelAutoEvalBudgetCap",
]
