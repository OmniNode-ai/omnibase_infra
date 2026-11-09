# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-to-handler and command-line integration for the dev-lane return to dev (OMN-22996).

Loads node_dev_lane_return_to_dev_effect's contract from disk, resolves each routed
operation to its declared handler class and its ``handle`` entry point, then drives the
handler's command line as a subprocess exactly as the deploy agent does: one JSON request
as the argument, one JSON result on stdout. No Docker daemon, broker or deploy agent is
touched.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.integration

_NODE_DIR = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_dev_lane_return_to_dev_effect"
)
_MODULE = "omnibase_infra.handlers.handler_dev_lane_return_to_dev"
_HEAD = "29dd9aa2c765929561db684e4ac9374c98547b19"


def _cli(request: dict[str, object]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", _MODULE, json.dumps(request)],
        capture_output=True,
        text=True,
        check=False,
    )


def _build() -> dict[str, object]:
    now = datetime.now(UTC)
    return {
        "repo": "omnibase_infra",
        "pr": 5755,
        "branch": "omn-22808-estimator-follows-web",
        "head": _HEAD,
        "requested_by": "estimator-deploy-2340",
        "job_id": "8dd5eb7f",
        "recorded_at": now.isoformat(),
        "expires_at": (now + timedelta(days=2)).isoformat(),
    }


def test_contract_routes_each_operation_to_the_handler_entry_point() -> None:
    contract = yaml.safe_load((_NODE_DIR / "contract.yaml").read_text(encoding="utf-8"))
    routed = contract["handler_routing"]["handlers"]
    assert {entry["operation"] for entry in routed} == {"record", "settle"}
    for entry in routed:
        module = importlib.import_module(entry["handler"]["module"])
        assert callable(getattr(module, entry["handler"]["name"]).handle)


def test_record_then_settle_through_the_command_line(tmp_path: Path) -> None:
    state = tmp_path / "branch-builds"
    base = {"state_dir": str(state), "git_timeout_seconds": 30}

    recorded = _cli({**base, "operation": "record", "build": _build()})
    assert recorded.returncode == 0, recorded.stderr
    assert json.loads(recorded.stdout)["recorded"][0]["pr"] == 5755

    settled = _cli(
        {
            **base,
            "operation": "settle",
            "jobs_dir": str(tmp_path),
            "now": (datetime.now(UTC) + timedelta(days=3)).isoformat(),
            "dev_ref": "origin/dev",
        }
    )
    assert settled.returncode == 0, settled.stderr
    result = json.loads(settled.stdout)
    assert [d["reason"] for d in result["due"]] == ["expired"]
    assert result["command"]["requested_by"] == "agent/return-to-dev"
    assert json.loads((state / "records.json").read_text())["last_return_attempt"]


@pytest.mark.parametrize(
    "request_",
    [
        {"operation": "record"},
        {"operation": "settle"},
        {"operation": "explode"},
    ],
)
def test_a_request_the_node_cannot_run_is_refused_and_writes_nothing(
    tmp_path: Path, request_: dict[str, object]
) -> None:
    result = _cli({**request_, "state_dir": str(tmp_path), "git_timeout_seconds": 30})
    assert result.returncode == 2
    assert "return-to-dev refused" in result.stderr
    assert not (tmp_path / "records.json").exists()
