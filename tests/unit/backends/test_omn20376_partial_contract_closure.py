# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A partial liveness fix cannot close the carrier's published-plugin work."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_core.nodes.node_done_write_receipt_gate_compute.handler_done_write_receipt_gate import (
    evaluate_done_write_receipt,
)

pytestmark = pytest.mark.unit
CONTRACT = Path(__file__).resolve().parents[3] / "contracts/OMN-20376.yaml"


def test_passing_partial_contract_cannot_close_unbound_carrier_criteria() -> None:
    contract = yaml.safe_load(CONTRACT.read_text())
    # A synthetic ticket description retains the two known open criteria beside
    # the exact AC3/AC4 text declared by this contract. No live ticket is edited.
    criteria = [
        "AC1: the C30 published-plugin walk reaches terminal_ok 1",
        "AC2: the published-plugin walk's negative control refuses",
        *(
            criterion["statement"]
            for requirement in contract["requirements"]
            for criterion in requirement["acceptance"]
        ),
    ]
    description = "## Acceptance criteria\n" + "\n".join(
        f"- [ ] {criterion}" for criterion in criteria
    )
    # Give every shipped check the strongest possible outcome. Passing the
    # whole partial contract still cannot discharge an omitted criterion.
    checks = [
        {
            "evidence_id": item["id"],
            "status": "verified",
            "proof_class": "behavior",
            "binds_ac": item.get("binds_ac", []),
        }
        for item in contract["dod_evidence"]
    ]

    def decide(rows: list[dict[str, Any]]):
        return evaluate_done_write_receipt(
            ticket_id=contract["ticket_id"],
            description=description,
            verdict={
                "status": "verified",
                "total_checks": len(rows),
                "verified_count": len(rows),
                "failed_count": 0,
                "non_probative_count": 0,
                "checks": rows,
            },
        )

    refused = decide(checks)
    assert not refused.allowed
    assert {row.label for row in refused.rows if not row.bound} == {"AC1", "AC2"}
    assert {row.label for row in refused.rows if row.bound} == {"AC3", "AC4"}
    assert "AC1" in refused.reason and "AC2" in refused.reason

    # Positive control: the rule can allow the same ticket when distinct,
    # verified evidence actually binds the two missing labels. These fixture
    # records are not added to the real contract or claimed as delivered work.
    completed = decide(
        checks
        + [
            {
                "evidence_id": "fixture-published-plugin-proof",
                "status": "verified",
                "proof_class": "behavior",
                "binds_ac": ["AC1", "AC2"],
            }
        ]
    )
    assert completed.allowed
