# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration guard for OMN-6679 projection row typing."""

from __future__ import annotations

from omnibase_infra.enums.enum_validation_verdict import EnumValidationVerdict
from omnibase_infra.verification.probes.probe_projection import check_projection_state


def test_omn_6679_projection_probe_accepts_string_rows() -> None:
    rows: list[dict[str, str]] = [
        {
            "current_state": "active",
            "entity_id": "node-registration-orchestrator",
            "node_type": "effect",
        }
    ]

    result = check_projection_state("node_registration_orchestrator", rows)

    assert result.verdict == EnumValidationVerdict.PASS
