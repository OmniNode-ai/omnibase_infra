# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Golden-equivalence proof for the node_auth_gate_compute def-B hand-flip (OMN-14818).

RSD canonical-rewrite lane (parent OMN-14355). ``HandlerAuthGate`` was hand-flipped
to canonical definition B via a PURE RENAME ``evaluate(request) -> handle(request)``;
the 10-step authorization cascade body is preserved byte-identically and the legacy
``execute(envelope)`` wrapper was removed.

The goldens under ``tests/fixtures/golden/node_auth_gate_compute/*.json`` pin the
exact ``ModelAuthGateRequest -> ModelAuthGateDecision`` mapping (recorded from the
live def-B ``handle()``, which is the byte-identical legacy ``evaluate()`` body).
Replaying each recorded input through ``handle()`` and asserting zero output diffs
is a durable regression: a future edit to the cascade cannot silently change
behavior without failing here. The comparator is non-vacuous — a perturbed golden
is detected (``test_golden_comparator_detects_divergence``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from omnibase_infra.nodes.node_auth_gate_compute.handlers import HandlerAuthGate
from omnibase_infra.nodes.node_auth_gate_compute.models import (
    ModelAuthGateDecision,
    ModelAuthGateRequest,
)
from scripts.ci.compute_golden import compare_output

pytestmark = pytest.mark.unit

_GOLDEN_DIR = (
    Path(__file__).resolve().parents[4]
    / "tests"
    / "fixtures"
    / "golden"
    / "node_auth_gate_compute"
)


def _golden_files() -> list[Path]:
    files = sorted(_GOLDEN_DIR.glob("*.json"))
    assert files, f"no golden fixtures found under {_GOLDEN_DIR}"
    return files


@pytest.mark.parametrize("golden_path", _golden_files(), ids=lambda p: p.stem)
def test_handle_reproduces_recorded_golden(golden_path: Path) -> None:
    """def-B ``handle(request)`` on the recorded input == the recorded output."""
    golden: dict[str, Any] = json.loads(golden_path.read_text(encoding="utf-8"))
    request = ModelAuthGateRequest.model_validate(golden["input"])
    fresh_output = HandlerAuthGate(MagicMock()).handle(request)
    assert isinstance(fresh_output, ModelAuthGateDecision)
    diffs = compare_output(golden, fresh_output)
    assert diffs == [], f"{golden_path.name}: handle() output diverged: {diffs}"


def test_golden_comparator_detects_divergence() -> None:
    """Non-vacuity guard: a perturbed golden output is flagged (RED-vs-exists-wrong)."""
    golden_path = _golden_files()[0]
    golden: dict[str, Any] = json.loads(golden_path.read_text(encoding="utf-8"))
    request = ModelAuthGateRequest.model_validate(golden["input"])
    fresh_output = HandlerAuthGate(MagicMock()).handle(request)
    perturbed = {
        **golden,
        "output": {**golden["output"], "step": golden["output"]["step"] + 99},
    }
    diffs = compare_output(perturbed, fresh_output)
    assert diffs, "comparator must detect a perturbed golden output"


def test_golden_fixture_count_matches_expected_pool() -> None:
    """Regression guard: the recorded scenario pool has a known, reviewed size."""
    assert len(_golden_files()) == 6
