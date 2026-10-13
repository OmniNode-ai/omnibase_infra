# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""``onex node`` hosts the stages an orchestrator's execution graph names (OMN-21071).

End to end through the real click entry point on the in-memory bus: the
orchestrator publishes a command per stage and no test supplies a consumer, so
both correlated stage terminals can only come from stage nodes the runtime
hosted for the run.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from click.testing import CliRunner

from omnibase_infra.cli.cli_node import run_node_by_name
from tests.helpers.execution_graph_stand_in import install_execution_graph_stand_ins

pytestmark = pytest.mark.integration

SWEEP = "node_stand_in_sweep_orchestrator"


def test_one_node_call_runs_the_sweep_to_its_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_execution_graph_stand_ins(monkeypatch)
    correlation_id = uuid.uuid4()
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "correlation_id": str(correlation_id),
                "operand": "fleet-7",
                "stage_deadline_s": 10,
            }
        )
    )

    result = CliRunner().invoke(
        run_node_by_name,
        [
            SWEEP,
            "--input",
            str(request),
            "--timeout",
            "900",
            "--state-root",
            str(tmp_path / "state"),
        ],
    )

    assert result.exit_code == 0, result.output
    state = json.loads((tmp_path / "state" / "workflow_result.json").read_text())
    terminal = state["terminal_payload"]
    assert terminal["correlation_id"] == str(correlation_id)
    assert terminal["status"] == "completed", terminal
    assert terminal["stage_values"] == {
        "alpha": "node_stand_in_stage_alpha:fleet-7",
        "beta": "node_stand_in_stage_beta:fleet-7",
    }
