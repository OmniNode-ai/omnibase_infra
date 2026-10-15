# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The census refresh leg can push its bump and survives drift (OMN-20413).

THE DEFECT THIS PINS. After OMN-20934 moved the census into the CI overlay,
every run of `lane-census-refresh.yml` failed, so the committed census aged
past the 7-day limit and `Lane Census Staleness` refused every pull request
whose paths trigger it. The verify runners' worker logs on the lab host show two
failures, both in this file:

1. "Open the census bump PR in the CI overlay" exited 128. `actions/checkout`
   persists its credential behind an `includeIf "gitdir:<checkout path>/.git"`,
   and the next step moves the clone to `../ci-overlay`, so `git push origin`
   ran with no credential at all.
2. "Collect the census from the lab docker daemon" read
   "lane-census-check.sh exited 8". Since OMN-20798 drift whose event was not
   published exits 8, and this job deliberately names no broker container, so
   every drift collection read 8 and was refused as uncollectable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO / ".github" / "workflows" / "lane-census-refresh.yml"


def _steps() -> list[dict[str, object]]:
    doc = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    steps = doc["jobs"]["refresh"]["steps"]
    assert isinstance(steps, list)
    return steps


def _step(name: str) -> dict[str, object]:
    matches = [s for s in _steps() if s.get("name") == name]
    assert len(matches) == 1, f"expected one step named {name!r}"
    return matches[0]


def _collect_acceptance_line() -> str:
    run = str(_step("Collect the census from the lab docker daemon")["run"])
    lines = [line for line in run.splitlines() if "if [[ $rc -ne 0" in line]
    assert len(lines) == 1, lines
    return lines[0].strip()


def test_the_overlay_checkout_persists_no_credential() -> None:
    """A persisted credential is bound to the pre-move gitdir and never applies."""
    with_block = _step(
        "Check out the CI overlay (app token so the bot PR triggers CI)"
    )["with"]
    assert isinstance(with_block, dict)
    assert with_block.get("path") == ".ci-overlay-checkout"
    assert with_block.get("persist-credentials") is False


def test_the_bump_push_carries_the_app_token() -> None:
    """The push names its credential, because the moved clone has none."""
    run = str(_step("Open the census bump PR in the CI overlay")["run"])
    pushes = [
        line.strip() for line in run.splitlines() if line.strip().startswith("git push")
    ]
    assert pushes == [
        'git push "https://x-access-token:${GH_TOKEN}@github.com/${OVERLAY_REPO}.git" "$BRANCH"'
    ]


@pytest.mark.parametrize(
    ("rc", "accepted"),
    [(0, True), (30, True), (8, True), (2, False), (3, False), (4, False), (5, False)],
)
def test_the_collect_step_accepts_drift_it_did_not_publish(
    rc: int, accepted: bool
) -> None:
    """0, 30 and 8 each wrote a snapshot; the rest produced no census."""
    line = _collect_acceptance_line()
    script = f"rc={rc}\n{line}\n  exit 1\nfi\nexit 0\n"
    result = subprocess.run(
        ["/bin/bash", "-c", script], check=False, capture_output=True, text=True
    )
    assert (result.returncode == 0) is accepted, (rc, line)


def _extraheader(repo: Path) -> str:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "config",
            "--get",
            "http.https://github.com/.extraheader",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(os.environ),
    )
    return result.stdout.strip()


@pytest.mark.live_contact(
    "tests/ci/fixtures/lane_census_refresh_overlay_push_omn20413.json"
)
def test_the_recorded_checkout_credential_does_not_survive_the_move(
    recorded_response: dict[str, Any], tmp_path: Path
) -> None:
    """Replay the runner's own overlay config through real git, before and after the move.

    The recording is the moved clone's .git/config from the lab runner. Its only
    credential is an includeIf on the pre-move gitdir, so once the clone sits at
    ../ci-overlay git resolves no credential and the push exits 128, which is
    what the recorded push step reported.
    """
    failures = {
        str(f["step"]): list(f["errorMessages"])
        for f in recorded_response["failed_steps"]
    }
    assert failures["Open the census bump PR in the CI overlay"] == [
        "Process completed with exit code 128."
    ]
    assert failures["Collect the census from the lab docker daemon"][0] == (
        "lane-census-check.sh exited 8; no census collected, no PR opened."
    )

    workspace = str(recorded_response["workspace"])
    config = str(recorded_response["overlay_git_config"])
    include_paths = {
        line.split("=", 1)[1].strip()
        for line in config.splitlines()
        if line.strip().startswith("path = ")
    }
    assert len(include_paths) == 2
    replay_workspace = tmp_path / "work" / "omnibase_infra" / "omnibase_infra"
    clone = replay_workspace / ".ci-overlay-checkout"
    clone.mkdir(parents=True)
    subprocess.run(
        ["git", "init", "-q", str(clone)],
        check=True,
        env=scrub_git_location_env(os.environ),
    )
    temp_dir = tmp_path / "work" / "_temp"
    temp_dir.mkdir()
    replayed = config.replace(workspace, str(replay_workspace))
    for original in include_paths:
        credential = temp_dir / Path(original).name
        credential.write_text(
            '[http "https://github.com/"]\n\textraheader = AUTHORIZATION: basic replay\n',
            encoding="utf-8",
        )
        replayed = replayed.replace(original, str(credential))
    (clone / ".git" / "config").write_text(replayed, encoding="utf-8")

    assert _extraheader(clone) == "AUTHORIZATION: basic replay"
    moved = (
        tmp_path
        / "work"
        / "omnibase_infra"
        / Path(str(recorded_response["moved_to"])).name
    )
    shutil.move(str(clone), str(moved))
    assert _extraheader(moved) == ""

    # The fix: nothing persisted, and the push names the App token itself.
    with_block = _step(
        "Check out the CI overlay (app token so the bot PR triggers CI)"
    )["with"]
    assert isinstance(with_block, dict)
    assert with_block.get("persist-credentials") is False
    run = str(_step("Open the census bump PR in the CI overlay")["run"])
    assert "x-access-token:${GH_TOKEN}@github.com/${OVERLAY_REPO}.git" in run
