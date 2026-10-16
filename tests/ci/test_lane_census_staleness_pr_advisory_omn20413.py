# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Replay the recorded pull-request red of the lane census age gate (OMN-20413).

omnibase_infra#4988 failed 'Lane Census Staleness + E-5 Env Drift (OMN-13034)'
on the age of the private overlay census, which no pull request here can move.
The replay drives the real checker at the recorded instant: without the
advisory flag it reproduces the recorded failure, with it the pull request is
warned and passes, and the workflow passes the flag on pull_request only.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/lane-census-staleness.yml"


def _checker() -> Any:
    spec = importlib.util.spec_from_file_location(
        "check_lane_census_age", ROOT / "scripts" / "check_lane_census_age.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.live_contact(
    "tests/ci/fixtures/ci_live_contact/lane_census_staleness_pr_omn20413.json"
)
def test_recorded_pull_request_red_is_advisory_on_pull_requests_only(
    recorded_response: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = recorded_response["response"]
    assert run["event_name"] == "pull_request"
    assert run["conclusion"] == "failure"
    snapshot = tmp_path / "census-snapshot.json"
    snapshot.write_text(
        json.dumps({"schema_version": "1.0.0", "emitted_at": run["census_emitted_at"]}),
        encoding="utf-8",
    )
    now = datetime.fromisoformat(run["completed_at"].replace("Z", "+00:00"))
    mod = _checker()

    assert mod.check_census_age(snapshot, run["max_age_days"], now=now) == 1
    assert "Census snapshot is 7 days old" in capsys.readouterr().err
    assert (
        mod.check_census_age(
            snapshot, run["max_age_days"], now=now, stale_is_advisory=True
        )
        == 0
    )
    assert "::warning title=LANE-CENSUS-STALE" in capsys.readouterr().err

    steps = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"][
        "lane-census-staleness"
    ]["steps"]
    step = next(s for s in steps if s["name"] == "Check census snapshot staleness")
    assert step["env"]["STALE_ADVISORY"] == "${{ github.event_name == 'pull_request' }}"
    assert "--stale-advisory" in step["run"]
