# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Runtime handlers for invariant evaluation compute (canonical definition B)."""

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


def _new_evaluator(
    allowed_import_paths: list[str] | None,
) -> InvariantEvaluator:
    """Create a per-request evaluator to avoid sharing mutable schema cache."""
    return InvariantEvaluator(allowed_import_paths=allowed_import_paths)


class HandlerInvariantEvaluate:
    """Canonical def-B handler: evaluate one invariant against an output payload."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self, request: ModelInvariantEvaluateInput
    ) -> ModelInvariantResult:
        """Evaluate one invariant against an output payload."""
        evaluator = _new_evaluator(request.allowed_import_paths)
        return evaluator.evaluate(request.invariant, request.output)


class HandlerInvariantEvaluateBatch:
    """Canonical def-B handler: evaluate an invariant set, ordered per-invariant results."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self, request: ModelInvariantEvaluateBatchInput
    ) -> list[ModelInvariantResult]:
        """Evaluate an invariant set and return ordered per-invariant results."""
        evaluator = _new_evaluator(request.allowed_import_paths)
        return evaluator.evaluate_batch(
            request.invariant_set,
            request.output,
            enabled_only=request.enabled_only,
        )


class HandlerInvariantEvaluateAll:
    """Canonical def-B handler: evaluate an invariant set, aggregate summary statistics."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.COMPUTE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.COMPUTE

    async def handle(
        self, request: ModelInvariantEvaluateAllInput
    ) -> ModelEvaluationSummary:
        """Evaluate an invariant set and return aggregate summary statistics."""
        evaluator = _new_evaluator(request.allowed_import_paths)
        return evaluator.evaluate_all(
            request.invariant_set,
            request.output,
            fail_fast=request.fail_fast,
        )


__all__ = [
    "HandlerInvariantEvaluate",
    "HandlerInvariantEvaluateAll",
    "HandlerInvariantEvaluateBatch",
]
