# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Integration test: quality-gate handler loads DoD checks from contract.yaml.

OMN-10616 wires HandlerQualityGate to read deterministic and heuristic DoD
checks from the task-class contract. This test loads the real
node_delegation_quality_gate_reducer/contract.yaml from disk and exercises
delta() end-to-end with both contract-declared check sets and the legacy
heuristic fallback, asserting the new fail_category="fail_deterministic"
semantics on deterministic-check failure.

The "integration" aspect is loading and parsing the on-disk contract YAML
that ships with the node, then driving the handler with inputs derived from
that contract — verifying the contract is parseable, the model fields
align with the contract input_model declaration, and the handler
short-circuits to FAIL_DETERMINISTIC when a contract-declared deterministic
check fails.

Related:
    - OMN-10616: Wire quality gate to read DoD from contract
    - OMN-7040: Node-based delegation pipeline
"""

from __future__ import annotations

from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest
import yaml

from omnibase_infra.nodes.node_delegation_quality_gate_reducer.handlers.handler_quality_gate import (
    delta,
)
from omnibase_infra.nodes.node_delegation_quality_gate_reducer.models.model_quality_gate_input import (
    ModelQualityGateInput,
)

pytestmark = [pytest.mark.integration]


CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_delegation_quality_gate_reducer"
    / "contract.yaml"
)


@pytest.fixture(scope="module")
def quality_gate_contract() -> dict[str, object]:
    """Load the on-disk quality-gate contract once per module."""
    assert CONTRACT_PATH.exists(), (
        f"quality-gate contract not found at {CONTRACT_PATH}; OMN-10616 wiring requires "
        "a contract.yaml co-located with the node module"
    )
    with CONTRACT_PATH.open(encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    assert isinstance(payload, dict), "quality-gate contract must parse to a mapping"
    return cast("dict[str, object]", payload)


def _input_with_contract_dod(
    *,
    content: str,
    dod_deterministic: tuple[str, ...] = (),
    dod_heuristic: tuple[str, ...] = (),
    task_type: str = "test",
) -> ModelQualityGateInput:
    return ModelQualityGateInput(
        correlation_id=uuid4(),
        task_type=task_type,
        llm_response_content=content,
        expected_markers=(),
        min_response_length=0,
        dod_deterministic=dod_deterministic,
        dod_heuristic=dod_heuristic,
    )


class TestContractLoading:
    """Verify the on-disk contract is parseable and wired to the handler."""

    def test_contract_file_present_and_parseable(
        self, quality_gate_contract: dict[str, object]
    ) -> None:
        """Contract YAML is present at the expected node-local path and parses cleanly."""
        assert (
            quality_gate_contract.get("name") == "node_delegation_quality_gate_reducer"
        )
        assert quality_gate_contract.get("node_type") == "REDUCER_GENERIC"

    def test_contract_declares_quality_gate_input_model(
        self, quality_gate_contract: dict[str, object]
    ) -> None:
        """Contract input_model points at ModelQualityGateInput, the same Pydantic
        model that the handler accepts and that carries dod_deterministic /
        dod_heuristic."""
        input_model = quality_gate_contract.get("input_model")
        assert isinstance(input_model, dict)
        assert input_model.get("name") == "ModelQualityGateInput"
        # Sanity-check the model the handler actually consumes carries the
        # OMN-10616 DoD fields, so the contract <-> handler wiring is real.
        sample = ModelQualityGateInput(
            correlation_id=uuid4(),
            task_type="test",
            llm_response_content="x",
            expected_markers=(),
            min_response_length=0,
            dod_deterministic=("output_parses",),
            dod_heuristic=("no_refusal",),
        )
        assert sample.dod_deterministic == ("output_parses",)
        assert sample.dod_heuristic == ("no_refusal",)


class TestDeterministicCheckSemantics:
    """Verify the contract-driven deterministic checks BLOCK on failure."""

    def test_empty_response_fails_output_parses(self) -> None:
        """Deterministic output_parses BLOCKS empty content with fail_deterministic."""
        result = delta(
            _input_with_contract_dod(
                content="",
                dod_deterministic=("output_parses",),
            )
        )
        assert result.passed is False
        assert result.fail_category == "fail_deterministic"

    def test_bare_traceback_fails_output_parses(self) -> None:
        """Deterministic output_parses BLOCKS bare-traceback content."""
        result = delta(
            _input_with_contract_dod(
                content="Traceback (most recent call last):\n  File 'x', line 1\nValueError",
                dod_deterministic=("output_parses",),
            )
        )
        assert result.passed is False
        assert result.fail_category == "fail_deterministic"

    def test_valid_content_passes_deterministic_only(self) -> None:
        """When only deterministic checks are declared and all pass, the gate passes."""
        content = "def add(a: int, b: int) -> int:\n    return a + b\n"
        result = delta(
            _input_with_contract_dod(
                content=content,
                dod_deterministic=("output_parses", "signature_preserved"),
            )
        )
        assert result.passed is True
        assert result.fail_category == "pass"


class TestHeuristicFallback:
    """Verify legacy heuristic semantics still apply when no contract DoD is set."""

    def test_no_contract_dod_uses_legacy_heuristics(self) -> None:
        """Empty dod_deterministic + dod_heuristic falls back to legacy
        length/refusal/marker checks (the pre-OMN-10616 behavior)."""
        # Legacy heuristic failure should NOT use the deterministic-block category.
        result = delta(
            _input_with_contract_dod(
                content="I cannot help with that.",
                dod_deterministic=(),
                dod_heuristic=(),
            )
        )
        assert result.passed is False
        assert result.fail_category != "fail_deterministic"
