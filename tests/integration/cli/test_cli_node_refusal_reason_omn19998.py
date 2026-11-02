# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A handler's own refusal reaches the ``SkillRoutingError`` envelope (OMN-19998).

The ``ci-live-contact`` pre-commit hook runs ``onex node
node_ci_live_contact_effect``. When a commit stages a CI-tooling change with no
recorded evidence, the handler refuses with a reason that names the fix, but
the envelope printed only ``run did not complete (result=failed)`` and the
reason stayed in the state file. These tests run the real packaged node on a
throwaway repository and read the envelope the hook prints.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.cli.cli_node import run_node_by_name

pytestmark = pytest.mark.integration

_NODE = "node_ci_live_contact_effect"


def _repo_with_staged(tmp_path: Path, relative: str) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            env=scrub_git_location_env(os.environ),
        )

    git("init", "-q")
    target = repo / relative
    target.parent.mkdir(parents=True)
    target.write_text("print('changed')\n", encoding="utf-8")
    git("add", relative)
    return repo


def _run(repo: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[int, str]:
    # The handler runs git with the process environment; a git hook's GIT_DIR
    # would retarget it from this repository to the invoking worktree.
    scrubbed = scrub_git_location_env(os.environ)
    for name in set(os.environ) - set(scrubbed):
        monkeypatch.delenv(name)
    monkeypatch.chdir(repo)
    result = CliRunner().invoke(
        run_node_by_name,
        [
            _NODE,
            "--backend",
            "event_bus=inmemory",
            "--state-root",
            str(repo.parent / "state"),
            "--timeout",
            "30",
        ],
    )
    return result.exit_code, result.output


def _envelope(output: str) -> dict[str, str]:
    start = output.rfind('{\n  "error_type"')
    assert start != -1, f"no SkillRoutingError envelope in output: {output!r}"
    envelope: dict[str, str] = json.loads(output[start:])
    return envelope


def test_refused_ci_tooling_change_names_its_fix_in_the_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo_with_staged(tmp_path, "scripts/ci/check_example.py")

    exit_code, output = _run(repo, monkeypatch)

    assert exit_code == 1
    envelope = _envelope(output)
    assert envelope["error_type"] == "SkillRoutingError"
    assert envelope["node_id"] == _NODE
    assert envelope["result"] == "failed"
    assert "CI-tooling change requires a live-contact test" in envelope["message"]
    assert ".github/live-contact-exception.md" in envelope["message"]


def test_non_tooling_change_completes_without_an_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo_with_staged(tmp_path, "src/example.py")

    exit_code, output = _run(repo, monkeypatch)

    assert exit_code == 0, output
    assert "SkillRoutingError" not in output
