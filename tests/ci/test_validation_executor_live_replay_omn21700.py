# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Replay the validation executor request captured on the h101 lab runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from omnibase_core.enums.enum_workflow_result import EnumWorkflowResult
from omnibase_core.runtime.runtime_local import RuntimeLocal


@pytest.mark.live_contact(
    "contracts/artifacts/OMN-21700/validation-executor-runtime.json"
)
async def test_executor_contract_replays_recorded_runtime_request(
    recorded_response: dict[str, Any], tmp_path: Path
) -> None:
    root = Path(__file__).resolve().parents[2]
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(recorded_response["request"]))
    runtime = RuntimeLocal(
        root / "src/omnibase_infra/nodes/node_validation_executor/contract.yaml",
        input_path=input_path,
        state_root=tmp_path / "state",
        backend_overrides={"event_bus": "inmemory"},
        timeout=5,
    )

    assert await runtime.run_async() is EnumWorkflowResult.COMPLETED
    readback = json.loads((tmp_path / "state/workflow_result.json").read_text())
    recorded = recorded_response["response"]
    assert readback["result"] == recorded["result"] == "completed"
    assert readback["exit_code"] == recorded["exit_code"] == 0
    assert readback["handler_locus"] == recorded["handler_locus"] == "in_process"
    actual = readback["terminal_payload"]
    expected = recorded["terminal_payload"]
    for key in ("plan_id", "candidate_id", "score_threshold"):
        assert actual[key] == expected[key]
    assert len(actual["check_results"]) == len(expected["check_results"]) == 2
    for actual_check, recorded_check in zip(
        actual["check_results"], expected["check_results"], strict=True
    ):
        for key in ("check_code", "label", "severity", "passed", "skipped", "message"):
            assert actual_check[key] == recorded_check[key]
