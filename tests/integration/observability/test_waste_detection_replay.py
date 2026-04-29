# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Replay tests for waste detection projection rows."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.nodes.node_waste_detection_compute.handlers.handler_waste_detection import (
    HandlerWasteDetection,
)
from omnibase_infra.nodes.node_waste_detection_compute.models import (
    ModelWasteCall,
    ModelWasteDetectionInput,
)

FIXTURE_PATH = (
    Path(__file__).parents[2]
    / "fixtures"
    / "cost_observability"
    / "task-10-waste.fixtures.jsonl"
)
GOLDEN_PATH = (
    Path(__file__).parents[2]
    / "fixtures"
    / "cost_observability"
    / "task-10-waste.golden.json"
)
DETECTED_AT = datetime(2026, 4, 29, 12, 1, 0, tzinfo=UTC)
RULE_IDS = (
    "tool_failure_waste",
    "agent_loop",
    "retry_waste",
    "high_output",
    "model_overkill",
    "low_cache",
)


class FakeConnection:
    """Capture projected DB rows without a live database."""

    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    async def execute(self, _query: str, *args: object) -> None:
        evidence_raw = args[5]
        assert isinstance(evidence_raw, str)
        self.rows.append(
            {
                "session_id": args[0],
                "rule_id": args[1],
                "severity": args[2],
                "waste_tokens": args[3],
                "waste_cost_usd": args[4],
                "evidence": json.loads(evidence_raw),
                "evidence_hash": args[6],
                "dedup_key": args[7],
                "recommendation": args[8],
                "repo_name": args[9],
                "machine_id": args[10],
                "detected_at": args[11].isoformat()
                if isinstance(args[11], datetime)
                else args[11],
            }
        )


def _load_calls() -> tuple[ModelWasteCall, ...]:
    calls: list[ModelWasteCall] = []
    with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                calls.append(ModelWasteCall.model_validate_json(line))
    return tuple(calls)


def _detect_rows() -> list[dict[str, Any]]:
    detection_input = ModelWasteDetectionInput(
        session_id="sess-task-10-waste",
        calls=_load_calls(),
        detected_at=DETECTED_AT,
    )
    findings = HandlerWasteDetection().detect(detection_input)
    return [finding.to_db_row() for finding in findings]


@pytest.mark.integration
@pytest.mark.parametrize("rule_id", RULE_IDS)
def test_fixture_replay_projects_waste_finding_for_each_rule(rule_id: str) -> None:
    rows = _detect_rows()

    matching = [row for row in rows if row["rule_id"] == rule_id]
    assert matching, f"expected fixture to produce {rule_id}"
    assert matching[0]["waste_tokens"] > 0
    assert matching[0]["waste_cost_usd"] > 0
    assert matching[0]["severity"] in {"LOW", "MEDIUM", "HIGH"}
    assert matching[0]["dedup_key"] == (
        f"{matching[0]['session_id']}:{rule_id}:{matching[0]['evidence_hash']}"
    )


@pytest.mark.integration
def test_fixture_replay_matches_golden_rows_byte_for_byte() -> None:
    rows = _detect_rows()
    expected = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    assert rows == expected


@pytest.mark.integration
async def test_project_findings_upserts_waste_findings_rows() -> None:
    detection_input = ModelWasteDetectionInput(
        session_id="sess-task-10-waste",
        calls=_load_calls(),
        detected_at=DETECTED_AT,
    )
    handler = HandlerWasteDetection()
    findings = handler.detect(detection_input)
    connection = FakeConnection()

    await handler.project_findings(connection, findings)

    assert len(connection.rows) == len(findings)
    for row in connection.rows:
        assert row["dedup_key"] == (
            f"{row['session_id']}:{row['rule_id']}:{row['evidence_hash']}"
        )
