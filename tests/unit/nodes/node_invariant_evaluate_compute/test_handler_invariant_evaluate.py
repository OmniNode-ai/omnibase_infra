# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for node_invariant_evaluate_compute."""

from __future__ import annotations

import pytest

from omnibase_core.enums import EnumInvariantType, EnumSeverity
from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_core.models.invariant import ModelInvariant, ModelInvariantSet
from omnibase_infra.nodes.node_invariant_evaluate_compute.evaluator_invariant import (
    InvariantEvaluator,
)
from omnibase_infra.nodes.node_invariant_evaluate_compute.handlers import (
    handle_invariant_evaluate,
    handle_invariant_evaluate_all,
    handle_invariant_evaluate_batch,
)
from omnibase_infra.nodes.node_invariant_evaluate_compute.models import (
    ModelInvariantEvaluateAllInput,
    ModelInvariantEvaluateBatchInput,
    ModelInvariantEvaluateInput,
)
from omnibase_infra.nodes.node_invariant_evaluate_compute.node import (
    NodeInvariantEvaluateCompute,
)


def _field_presence_invariant() -> ModelInvariant:
    return ModelInvariant(
        name="Field Check",
        type=EnumInvariantType.FIELD_PRESENCE,
        severity=EnumSeverity.CRITICAL,
        config={"fields": ["status"]},
    )


def _threshold_invariant() -> ModelInvariant:
    return ModelInvariant(
        name="Threshold Check",
        type=EnumInvariantType.THRESHOLD,
        severity=EnumSeverity.WARNING,
        config={"metric_name": "score", "min_value": 0.5},
    )


@pytest.mark.unit
def test_node_invariant_evaluate_compute_is_declarative_shell() -> None:
    node = NodeInvariantEvaluateCompute(
        ModelONEXContainer(enable_service_registry=False)
    )

    assert isinstance(node, NodeInvariantEvaluateCompute)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_invariant_evaluate_returns_single_result() -> None:
    result = await handle_invariant_evaluate(
        ModelInvariantEvaluateInput(
            invariant=_field_presence_invariant(),
            output={"status": "ok"},
        )
    )

    assert result.passed is True
    assert result.invariant_name == "Field Check"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_invariant_evaluate_batch_preserves_order() -> None:
    invariant_set = ModelInvariantSet(
        name="Batch",
        target="test",
        invariants=[
            _field_presence_invariant(),
            _threshold_invariant(),
        ],
    )

    results = await handle_invariant_evaluate_batch(
        ModelInvariantEvaluateBatchInput(
            invariant_set=invariant_set,
            output={"status": "ok", "score": 0.8},
        )
    )

    assert [result.invariant_name for result in results] == [
        "Field Check",
        "Threshold Check",
    ]
    assert all(result.passed for result in results)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_invariant_evaluate_all_summarizes_failures() -> None:
    invariant_set = ModelInvariantSet(
        name="Summary",
        target="test",
        invariants=[
            _field_presence_invariant(),
            _threshold_invariant(),
        ],
    )

    summary = await handle_invariant_evaluate_all(
        ModelInvariantEvaluateAllInput(
            invariant_set=invariant_set,
            output={"score": 0.2},
            fail_fast=False,
        )
    )

    assert summary.overall_passed is False
    assert summary.critical_failures == 1
    assert summary.warning_failures == 1


@pytest.mark.unit
def test_invariant_evaluator_blocks_redos_patterns() -> None:
    evaluator = InvariantEvaluator()

    is_safe, error_msg = evaluator._is_regex_safe(r"(a+)+")

    assert is_safe is False
    assert "nested quantifiers" in error_msg.lower()


@pytest.mark.unit
def test_invariant_evaluator_reuses_schema_validator_cache() -> None:
    evaluator = InvariantEvaluator()
    invariant = ModelInvariant(
        name="Schema Check",
        type=EnumInvariantType.SCHEMA,
        severity=EnumSeverity.CRITICAL,
        config={
            "json_schema": {
                "type": "object",
                "required": ["status"],
                "properties": {"status": {"type": "string"}},
            }
        },
    )

    first = evaluator.evaluate(invariant, {"status": "ok"})
    second = evaluator.evaluate(invariant, {"status": "ready"})

    assert first.passed is True
    assert second.passed is True
    assert evaluator.get_validator_cache_size() == 1
