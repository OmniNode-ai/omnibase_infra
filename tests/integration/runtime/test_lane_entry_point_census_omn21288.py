# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The census node runs through the local runtime and persists its result."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnibase_core.runtime.runtime_local import RuntimeLocal
from omnibase_infra.models.lane_entry_point_census import (
    ModelLaneEntryPointCensusRequest,
    ModelLaneEntryPointCensusResult,
)

pytestmark = pytest.mark.integration
REPO = Path(__file__).resolve().parents[3]
CONTRACT = (
    REPO / "src/omnibase_infra/nodes/node_lane_entry_point_census_effect/contract.yaml"
)
POLICY = REPO / "deploy/lane-census/runtime-entry-points.yaml"
FIXTURE = (
    REPO
    / "tests/unit/nodes/node_lane_entry_point_census_effect/fixtures/inspect_planted.json"
)


def test_runtime_reports_the_planted_python_dash_m_container(tmp_path: Path) -> None:
    request = ModelLaneEntryPointCensusRequest(
        policy_path=str(POLICY), inspect_json_path=str(FIXTURE)
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json(), encoding="utf-8")
    state = tmp_path / "runtime-state"
    runtime = RuntimeLocal(
        workflow_path=CONTRACT,
        state_root=state,
        input_path=request_path,
        backend_overrides={"event_bus": "inmemory"},
        timeout=10,
    )
    runtime.run()
    readback = json.loads((state / "workflow_result.json").read_text())
    result = ModelLaneEntryPointCensusResult.model_validate(readback["handler_result"])
    assert runtime.exit_code == readback["exit_code"] == 0
    lane = next(c for c in result.censuses if c.lane == "proj-a")
    assert "proj-a-ctx-consumer" in lane.remove


def test_runtime_exits_nonzero_when_the_policy_is_unreadable(tmp_path: Path) -> None:
    request = ModelLaneEntryPointCensusRequest(
        policy_path=str(tmp_path / "absent.yaml"), inspect_json_path=str(FIXTURE)
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json(), encoding="utf-8")
    state = tmp_path / "runtime-state"
    runtime = RuntimeLocal(
        workflow_path=CONTRACT,
        state_root=state,
        input_path=request_path,
        backend_overrides={"event_bus": "inmemory"},
        timeout=10,
    )
    runtime.run()
    readback = json.loads((state / "workflow_result.json").read_text())
    assert runtime.exit_code == readback["exit_code"] == 1
    assert readback["handler_result"]["success"] is False
