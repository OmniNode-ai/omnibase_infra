# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for projection state verification probe [OMN-7046]."""

from __future__ import annotations

import pytest

from omnibase_infra.enums.enum_contract_check_type import EnumContractCheckType
from omnibase_infra.enums.enum_validation_verdict import EnumValidationVerdict
from omnibase_infra.verification.probes.probe_projection import (
    TERMINAL_STATES,
    check_projection_state,
)

CONTRACT_NAME = "node_registration_orchestrator"


@pytest.mark.unit
class TestCheckProjectionState:
    """Tests for check_projection_state probe."""

    def test_pass_with_active_row(self) -> None:
        rows = [{"current_state": "active", "entity_id": "abc"}]
        result = check_projection_state(CONTRACT_NAME, rows)
        assert result.verdict == EnumValidationVerdict.PASS
        assert result.check_type == EnumContractCheckType.PROJECTION_STATE

    def test_pass_with_ack_received_row(self) -> None:
        rows = [{"current_state": "ack_received", "entity_id": "abc"}]
        result = check_projection_state(CONTRACT_NAME, rows)
        assert result.verdict == EnumValidationVerdict.PASS

    def test_pass_with_mixed_states(self) -> None:
        rows = [
            {"current_state": "pending_registration"},
            {"current_state": "active"},
            {"current_state": "rejected"},
        ]
        result = check_projection_state(CONTRACT_NAME, rows)
        assert result.verdict == EnumValidationVerdict.PASS
        assert "1/3" in result.evidence

    def test_fail_with_no_rows(self) -> None:
        result = check_projection_state(CONTRACT_NAME, [])
        assert result.verdict == EnumValidationVerdict.FAIL
        assert "No rows" in result.evidence

    def test_fail_with_all_pending(self) -> None:
        rows = [
            {"current_state": "pending_registration"},
            {"current_state": "pending_registration"},
        ]
        result = check_projection_state(CONTRACT_NAME, rows)
        assert result.verdict == EnumValidationVerdict.FAIL
        assert "non-terminal" in result.evidence

    def test_fail_with_all_non_terminal(self) -> None:
        rows = [
            {"current_state": "pending_registration"},
            {"current_state": "awaiting_ack"},
            {"current_state": "rejected"},
        ]
        result = check_projection_state(CONTRACT_NAME, rows)
        assert result.verdict == EnumValidationVerdict.FAIL

    def test_evidence_includes_state_counts(self) -> None:
        rows = [
            {"current_state": "active"},
            {"current_state": "active"},
            {"current_state": "pending_registration"},
        ]
        result = check_projection_state(CONTRACT_NAME, rows)
        assert "active=2" in result.evidence
        assert "pending_registration=1" in result.evidence

    def test_contract_name_preserved(self) -> None:
        rows = [{"current_state": "active"}]
        result = check_projection_state(CONTRACT_NAME, rows)
        assert result.contract_name == CONTRACT_NAME

    def test_terminal_states_are_correct(self) -> None:
        assert {"active", "ack_received"} == TERMINAL_STATES
