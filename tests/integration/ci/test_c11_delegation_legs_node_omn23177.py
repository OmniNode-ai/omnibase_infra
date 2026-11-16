# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Golden and error chain events of the C11 delegation-legs node (OMN-23177).

Runs the packaged node through the generic ``onex node`` command, exactly as the
workflow does, over the committed observations: the healthy file completes, and
the same file with the canary planted in the answer fails the node, so a canary
leak cannot read green.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.integration

ONEX = Path(sys.executable).parent / "onex"
REPO = Path(__file__).resolve().parents[3]


def _run(tmp_path: Path, *, plant: bool) -> tuple[int, dict[str, Any], Path]:
    record = tmp_path / "record.json"
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "config_path": str(REPO / "config" / "c11_delegation_legs.yaml"),
                "replay_path": str(
                    REPO / "tests" / "fixtures" / "omn23177" / "legs_healthy.json"
                ),
                "record_path": str(record),
                "plant_canary_in_answer": plant,
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "state"
    done = subprocess.run(
        [
            str(ONEX),
            "node",
            "node_c11_delegation_legs_effect",
            "--input",
            str(request),
            "--backend",
            "event_bus=inmemory",
            "--state-root",
            str(state),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        cwd=REPO,
    )
    result = json.loads((state / "workflow_result.json").read_text(encoding="utf-8"))
    return done.returncode, result, record


def test_golden_the_committed_observations_complete_the_node(tmp_path: Path) -> None:
    code, result, record = _run(tmp_path, plant=False)
    assert code == 0
    assert result["result"] == "completed"
    assert result["terminal_payload"]["outcome"] == "PASS"
    assert json.loads(record.read_text(encoding="utf-8"))["verdict"] == "pass"


def test_error_a_canary_planted_in_the_answer_fails_the_node(tmp_path: Path) -> None:
    code, result, record = _run(tmp_path, plant=True)
    assert code != 0
    assert result["result"] == "failed"
    assert result["terminal_payload"]["outcome"] == "FAIL"
    assert "canary" in result["terminal_payload"]["error_message"]
    assert json.loads(record.read_text(encoding="utf-8"))["planted_canary"] is True
