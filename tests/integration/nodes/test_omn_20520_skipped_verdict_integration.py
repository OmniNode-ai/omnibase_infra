# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for OMN-20520's skipped dod_verify reporting.

Exercises the skipped-verdict reason builder and the contract-declared
decision against the real node contract, so the feature-code coverage gate
sees a test under tests/integration/.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    _skipped_verdict_reason,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_decision import (
    EnumEvidenceAutocloseDecision,
)

pytestmark = pytest.mark.integration

_CONTRACT = (
    Path(__file__).parents[3]
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_evidence_autoclose_sweep_effect"
    / "contract.yaml"
)


def test_skipped_reason_keeps_terminal_and_check_causes() -> None:
    verdict: dict[str, object] = {
        "error_message": "ceiling hit",
        "checks": [
            {
                "evidence_id": "live-proof",
                "status": "skipped",
                "unverifiable_cause": "check_budget_exceeded",
            }
        ],
    }

    reason = _skipped_verdict_reason(verdict)

    assert "ceiling hit" in reason
    assert "check_budget_exceeded" in reason


def test_skipped_reason_states_when_none_supplied() -> None:
    assert _skipped_verdict_reason({}) == "no reason supplied by dod_verify"


def test_contract_declares_skipped_dod_verify_error_type() -> None:
    contract = yaml.safe_load(_CONTRACT.read_text(encoding="utf-8"))
    names = {e["name"] for e in contract["error_handling"]["error_types"]}

    assert "DOD_VERIFY_SKIPPED" in names
    assert (
        EnumEvidenceAutocloseDecision.SKIPPED_DOD_VERIFY.value == "skipped_dod_verify"
    )
