# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Golden and error chain events of the C16 pinned-backend node (OMN-23176).

Runs the packaged node through the generic ``onex node`` command, exactly as the
workflow does, over recorded observations: three pinned runs that agree with their
rows complete, and the same set with the paid role's run missing fails with the
named backend, so a backend the lane lacks cannot read green.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.scripts.test_c16_pinned_backends import CONFIG, _payload

pytestmark = pytest.mark.integration

ONEX = Path(sys.executable).parent / "onex"
QUERY = "SELECT 1 WHERE {correlation_id} IS NULL"


def _run(tmp_path: Path, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump({"pinned_backends": dict(CONFIG), "row_query": QUERY}),
        encoding="utf-8",
    )
    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps(payload), encoding="utf-8")
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "config_path": str(config),
                "replay_path": str(replay),
                "record_path": str(tmp_path / "record.json"),
                "timeout_seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "state"
    done = subprocess.run(
        [
            str(ONEX),
            "node",
            "node_c16_pinned_backend_effect",
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
    )
    result = json.loads((state / "workflow_result.json").read_text(encoding="utf-8"))
    return done.returncode, result


def test_golden_agreeing_receipts_and_rows_complete_the_node(tmp_path: Path) -> None:
    code, result = _run(tmp_path, _payload())
    assert code == 0
    assert result["result"] == "completed"
    assert result["terminal_payload"]["outcome"] == "PASS"
    record = json.loads((tmp_path / "record.json").read_text(encoding="utf-8"))
    assert record["probes"][0]["outcome"] == "PASS"


def test_error_a_role_with_no_run_fails_the_node_naming_its_backend(
    tmp_path: Path,
) -> None:
    def drop_paid(runs: list[dict[str, Any]], rows: dict[str, Any]) -> None:
        runs[:] = [r for r in runs if r["role"] != "paid"]

    code, result = _run(tmp_path, _payload(drop_paid))
    assert code != 0
    assert result["result"] == "failed"
    assert result["terminal_payload"]["outcome"] == "FAIL"
    assert "backend-c" in result["terminal_payload"]["error_message"]
