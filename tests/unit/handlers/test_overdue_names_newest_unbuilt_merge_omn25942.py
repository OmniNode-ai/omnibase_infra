# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The overdue build names the newest unbuilt runtime merge once the oldest is contained (OMN-25942).

THE RECORD
----------
``tests/fixtures/overdue_build_oldest_contained_omn25942/build_decision_rows.json``: at
2026-11-26T23:10Z the overdue publisher kept naming the oldest unbuilt merge 9f77818e
(omnibase_infra#6317, "no dev build by 17:56Z"), which the pinned 22:01Z build already
contains, so each hourly window ended superseded and built nothing while five newer heads sat
unbuilt. Once the merge it names is contained in the running build the publisher moves to the
newest runtime merge the running build does not contain.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.handlers.handler_dev_merge_build_overdue import request_id_for
from omnibase_infra.models.model_overdue_dev_merge_job import ModelOverdueDevMergeJob
from omnibase_infra.nodes.node_deploy_job_reading_effect import (
    runtime_deploy_job_reading as caller,
)

from .test_dev_merge_build_one_request_omn26316 import _SETTINGS, _decision, _Sender

pytestmark = pytest.mark.unit

_FIXTURE = (
    Path(__file__).parents[2]
    / "fixtures"
    / "overdue_build_oldest_contained_omn25942"
    / "build_decision_rows.json"
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(os.environ),
    ).stdout.strip()


def _history(tmp_path: Path, subjects: list[str]) -> tuple[SimpleNamespace, list[str]]:
    repo = tmp_path / "clone"
    state = tmp_path / "state"
    repo.mkdir()
    state.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    shas = []
    for subject in subjects:
        _git(repo, "commit", "--allow-empty", "-q", "-m", subject)
        shas.append(_git(repo, "rev-parse", "HEAD"))
    return SimpleNamespace(
        STATE_DIR=str(state), deploy_repo_dir=lambda: str(repo)
    ), shas


def _running(built_sha: str) -> ModelOverdueDevMergeJob:
    return ModelOverdueDevMergeJob.model_validate(
        {
            "job_id": "00000000-0000-5000-8000-000000000009",
            "runtime_lane": "dev",
            "accepted_at": "2026-11-26T22:01:00Z",
            "status": "success",
            "git_ref": "dev-20261126220100-0123456789ab",
            "requested_by": "agent/idle-restore",
            "built_sha": built_sha,
        }
    )


@pytest.fixture
def sender(monkeypatch: pytest.MonkeyPatch) -> _Sender:
    for relative in caller._OVERDUE_MODULES:
        caller._load(relative)
    fake = _Sender()
    monkeypatch.setattr(caller, "AgentDevMergeSender", lambda *_a: fake)
    monkeypatch.setattr(caller, "_TRACKED_HEAD", "HEAD")
    return fake


def test_the_fixture_records_the_oldest_merge_as_the_one_named() -> None:
    record = json.loads(_FIXTURE.read_text())
    assert record["overdue_publisher"]["names"] == "9f77818e"
    assert record["running_build"]["contains"] == ["9f77818e"]
    assert len(record["omnibase_infra_heads"]) == 6


def test_a_contained_merge_moves_the_request_to_the_newest_unbuilt_runtime_merge(
    tmp_path: Path, sender: _Sender
) -> None:
    # 6317 is contained in the running build; 6318 and 6320 are newer; 6325 is a not-runtime head.
    execution, shas = _history(
        tmp_path,
        [
            "no dev build by 17:56Z (#6317)",
            "newer runtime merge (#6318)",
            "newest runtime merge (#6320)",
            "docs only (#6325)",
        ],
    )
    context = caller.OverdueContext(
        (_running(shas[0]),),
        dict.fromkeys(shas, "omnibase_infra"),
        {
            "omnibase_infra": (
                (shas[3], "NON_RUNTIME"),
                (shas[2], "RUNTIME"),
                (shas[1], "RUNTIME"),
                (shas[0], "RUNTIME"),
            )
        },
    )
    subjects = {
        shas[0]: "no dev build by 17:56Z (#6317)",
        shas[1]: "newer runtime merge (#6318)",
        shas[2]: "newest runtime merge (#6320)",
        shas[3]: "docs only (#6325)",
    }
    caller._request_overdue_builds(
        execution, _SETTINGS, [_decision(shas[0])], subjects, context
    )
    assert len(sender.commands) == 1
    command = sender.commands[0]
    assert command.pr_number == 6320
    assert shas[2] in command.rollback_reason
    assert command.correlation_id == request_id_for("omnibase_infra", shas[2])
    assert sender.events[0].merge_sha == shas[2]


def test_with_no_newer_unbuilt_runtime_merge_nothing_is_requested(
    tmp_path: Path, sender: _Sender
) -> None:
    execution, shas = _history(tmp_path, ["old (#6317)", "docs only (#6325)"])
    context = caller.OverdueContext(
        (_running(shas[0]),),
        dict.fromkeys(shas, "omnibase_infra"),
        {"omnibase_infra": ((shas[1], "NON_RUNTIME"), (shas[0], "RUNTIME"))},
    )
    caller._request_overdue_builds(
        execution,
        _SETTINGS,
        [_decision(shas[0])],
        {shas[0]: "old (#6317)"},
        context,
    )
    assert sender.commands == []


def test_an_unbuilt_overdue_merge_is_still_named_as_before(
    tmp_path: Path, sender: _Sender
) -> None:
    execution, shas = _history(tmp_path, ["a (#1)", "b (#2)", "c (#3)"])
    context = caller.OverdueContext((), dict.fromkeys(shas, "omnibase_infra"), {})
    caller._request_overdue_builds(
        execution, _SETTINGS, [_decision(shas[0])], {shas[0]: "a (#1)"}, context
    )
    assert len(sender.commands) == 1
    assert sender.commands[0].pr_number == 1
