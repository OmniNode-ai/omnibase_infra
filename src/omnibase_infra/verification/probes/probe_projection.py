# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Projection state verification probe.

Verifies that the registration FSM has completed at least one full cycle
by checking the registration_projections table for rows in terminal states.
"""

from __future__ import annotations

from typing import Any

from omnibase_infra.enums.enum_check_severity import EnumCheckSeverity
from omnibase_infra.enums.enum_contract_check_type import EnumContractCheckType
from omnibase_infra.enums.enum_validation_verdict import EnumValidationVerdict
from omnibase_infra.verification.models.model_contract_check_result import (
    ModelContractCheckResult,
)

# Terminal FSM states indicating a completed registration cycle.
# 'active' = node fully registered and healthy.
# 'ack_received' = node acknowledged registration (precursor to active).
TERMINAL_STATES: frozenset[str] = frozenset({"active", "ack_received"})


# ONEX_EXCLUDE: any_type - DB rows are untyped dicts from SQL queries
def check_projection_state(
    contract_name: str,
    db_rows: list[dict[str, Any]],
) -> ModelContractCheckResult:
    """Check registration projection state for evidence of completed FSM cycles.

    Args:
        contract_name: Name of the contract being verified.
        db_rows: Rows from registration_projections query. Each row must have
            at least a 'current_state' key.

    Returns:
        A ModelContractCheckResult with PASS if at least one row is in a
        terminal state, FAIL otherwise.
    """
    if not db_rows:
        return ModelContractCheckResult(
            check_type=EnumContractCheckType.PROJECTION_STATE,
            severity=EnumCheckSeverity.REQUIRED,
            verdict=EnumValidationVerdict.FAIL,
            evidence="No rows found in registration_projections table.",
            contract_name=contract_name,
            message="Projection state check failed: no registrations exist.",
        )

    # Count states
    state_counts: dict[str, int] = {}
    for row in db_rows:
        state = str(row.get("current_state", "unknown"))
        state_counts[state] = state_counts.get(state, 0) + 1

    terminal_count = sum(
        count for state, count in state_counts.items() if state in TERMINAL_STATES
    )
    total_count = len(db_rows)

    state_summary = ", ".join(
        f"{state}={count}" for state, count in sorted(state_counts.items())
    )

    if terminal_count > 0:
        return ModelContractCheckResult(
            check_type=EnumContractCheckType.PROJECTION_STATE,
            severity=EnumCheckSeverity.REQUIRED,
            verdict=EnumValidationVerdict.PASS,
            evidence=(
                f"Found {terminal_count}/{total_count} rows in terminal states. "
                f"State distribution: {state_summary}"
            ),
            contract_name=contract_name,
            message=(
                f"Projection state check passed: {terminal_count} nodes in "
                f"terminal state ({state_summary})."
            ),
        )

    return ModelContractCheckResult(
        check_type=EnumContractCheckType.PROJECTION_STATE,
        severity=EnumCheckSeverity.REQUIRED,
        verdict=EnumValidationVerdict.FAIL,
        evidence=(
            f"All {total_count} rows stuck in non-terminal states. "
            f"State distribution: {state_summary}. "
            f"Expected at least one row in: {', '.join(sorted(TERMINAL_STATES))}"
        ),
        contract_name=contract_name,
        message=(
            f"Projection state check failed: {total_count} rows, "
            f"none in terminal state."
        ),
    )


__all__: list[str] = ["check_projection_state", "TERMINAL_STATES"]
