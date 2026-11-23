# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The notes node stages the recorded 0.4.325 cut's CHANGELOG.md in a real git clone (OMN-26500)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)
from omnibase_infra.handlers.handler_release_notes_apply import (
    HandlerReleaseNotesApply,
)
from omnibase_infra.models.model_release_notes_apply_request import (
    ModelReleaseNotesApplyRequest,
)

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "release_notes"
    / "cut_0_4_325.json"
)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=train",
            "-c",
            "user.email=train@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-C",
            str(root),
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(os.environ),
    ).stdout


@pytest.mark.integration
@pytest.mark.asyncio
async def test_the_recorded_cut_of_a_gated_repo_commits_the_changelog(
    tmp_path: Path,
) -> None:
    recorded = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    clone = tmp_path / "clone"
    gate = clone / recorded["gate_source_path"]
    gate.parent.mkdir(parents=True)
    gate.write_text(recorded["gate_source_excerpt"], encoding="utf-8")
    (clone / "README.md").write_text("# omnimarket\n", encoding="utf-8")
    _git(clone, "init", "-q", "-b", "dev")
    _git(clone, "add", "--", ".")
    _git(clone, "commit", "-q", "-m", "dev")
    base = _git(clone, "rev-parse", "HEAD").strip()
    entry_file = tmp_path / "entry.md"
    entry_file.write_text(recorded["rendered_entry"], encoding="utf-8")

    result = await HandlerReleaseNotesApply().handle(
        ModelReleaseNotesApplyRequest(
            clone_path=str(clone),
            entry_file=str(entry_file),
            repo_name="omnimarket",
            repository_visibility="private",
        )
    )
    _git(clone, "add", "--", *result.staged)
    _git(clone, "commit", "-q", "--allow-empty", "-m", "chore: release")

    assert result.success is True
    assert result.staged == ("CHANGELOG.md",)
    changed = _git(clone, "diff", "--name-only", base, "HEAD").split()
    assert changed == ["CHANGELOG.md"]
