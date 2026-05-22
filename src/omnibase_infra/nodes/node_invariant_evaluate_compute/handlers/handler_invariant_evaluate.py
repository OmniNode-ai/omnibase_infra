# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Runtime handlers for invariant evaluation compute."""

from __future__ import annotations

from omnibase_core.models.invariant import ModelEvaluationSummary, ModelInvariantResult
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_invariant_evaluate_compute.evaluator_invariant import (
    InvariantEvaluator,
)
from omnibase_infra.nodes.node_invariant_evaluate_compute.models import (
    ModelInvariantEvaluateAllInput,
    ModelInvariantEvaluateBatchInput,
    ModelInvariantEvaluateInput,
)


class HandlerInvariantEvaluate:
    """Handler descriptor for deterministic invariant evaluation operations."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE


def _new_evaluator(
    allowed_import_paths: list[str] | None,
) -> InvariantEvaluator:
    """Create a per-request evaluator to avoid sharing mutable schema cache."""
    return InvariantEvaluator(allowed_import_paths=allowed_import_paths)


async def handle_invariant_evaluate(
    input_data: ModelInvariantEvaluateInput,
) -> ModelInvariantResult:
    """Evaluate one invariant against an output payload."""
    evaluator = _new_evaluator(input_data.allowed_import_paths)
    return evaluator.evaluate(input_data.invariant, input_data.output)


async def handle_invariant_evaluate_batch(
    input_data: ModelInvariantEvaluateBatchInput,
) -> list[ModelInvariantResult]:
    """Evaluate an invariant set and return ordered per-invariant results."""
    evaluator = _new_evaluator(input_data.allowed_import_paths)
    return evaluator.evaluate_batch(
        input_data.invariant_set,
        input_data.output,
        enabled_only=input_data.enabled_only,
    )


async def handle_invariant_evaluate_all(
    input_data: ModelInvariantEvaluateAllInput,
) -> ModelEvaluationSummary:
    """Evaluate an invariant set and return aggregate summary statistics."""
    evaluator = _new_evaluator(input_data.allowed_import_paths)
    return evaluator.evaluate_all(
        input_data.invariant_set,
        input_data.output,
        fail_fast=input_data.fail_fast,
    )


__all__ = [
    "HandlerInvariantEvaluate",
    "handle_invariant_evaluate",
    "handle_invariant_evaluate_all",
    "handle_invariant_evaluate_batch",
]
