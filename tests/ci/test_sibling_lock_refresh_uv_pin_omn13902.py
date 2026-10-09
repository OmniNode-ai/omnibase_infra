# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A lockfile format revision alone is not a sibling refresh (OMN-13902).

``[tool.uv]`` pins ``omnibase-core``, ``omnibase-spi`` and ``omnibase-compat`` to
exact versions, so ``uv lock --upgrade-package`` has nothing to move for them. The
refresh job installs whatever ``astral-sh/setup-uv`` calls latest, and a uv newer
than the one that wrote the committed lock raises the ``revision =`` header instead
(``revision = 3`` became ``revision = 5`` under uv 0.12.24 against this repo's lock).
That one line is a diff, so the job opened a "refresh omninode sibling locks" PR
whose only content was the format revision, and the runner image lock digest, which
hashes ``uv.lock``, moved with it.

These tests run the refresh step's own shell against a throwaway git repository
with a stub ``uv``, so the step is exercised as written rather than re-implemented
here.
"""

import os
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "sibling-lock-refresh.yml"

COMMITTED_LOCK = 'version = 1\nrevision = 3\nrequires-python = ">=3.12"\n\n[[package]]\nname = "omnimarket"\nversion = "0.4.1"\n'


def _refresh_script() -> str:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["refresh"]["steps"]
    (step,) = [s for s in steps if s.get("id") == "refresh"]
    return str(step["run"]).replace("${{ github.event.inputs.packages }}", "")


def _run_refresh(tmp_path: Path, committed: str, produced: str) -> tuple[str, str]:
    """Run the refresh step's shell over ``committed`` with a uv that writes ``produced``."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "uv.lock").write_text(committed, encoding="utf-8")
    identity = ["-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    for argv in (
        ["init", "-q"],
        ["add", "uv.lock"],
        [*identity, "commit", "-q", "-m", "base"],
    ):
        subprocess.run(
            ["git", *argv],
            cwd=repo,
            env=scrub_git_location_env(os.environ),
            check=True,
        )

    produced_path = tmp_path / "produced.lock"
    produced_path.write_text(produced, encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "uv"
    stub.write_text(f'#!/bin/sh\ncp "{produced_path}" uv.lock\n', encoding="utf-8")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    output = tmp_path / "github_output"
    output.touch()
    subprocess.run(
        ["bash", "-c", _refresh_script()],
        cwd=repo,
        env={
            **scrub_git_location_env(os.environ),
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "GITHUB_OUTPUT": str(output),
        },
        check=True,
        capture_output=True,
        text=True,
    )
    return output.read_text(encoding="utf-8"), (repo / "uv.lock").read_text(
        encoding="utf-8"
    )


@pytest.mark.live_contact("tests/fixtures/omn13902/uv-lock-revision-rewrite.json")
def test_revision_only_change_opens_no_refresh_pr(
    tmp_path: Path, recorded_response: dict[str, object]
) -> None:
    # The lock uv wrote is the recorded one: the committed lock with the header
    # line the real uv replaced swapped for the line it wrote.
    removed = str(recorded_response["removed_line"])
    added = str(recorded_response["added_line"])
    committed = COMMITTED_LOCK.replace("revision = 3", removed)
    assert committed != COMMITTED_LOCK.replace("revision = 3", added)

    outputs, lock = _run_refresh(tmp_path, committed, committed.replace(removed, added))

    assert "changed=false" in outputs
    assert "changed=true" not in outputs
    assert lock == committed, "a format-only rewrite must not be left in the tree"


def test_moved_sibling_pin_still_opens_a_refresh_pr(tmp_path: Path) -> None:
    produced = COMMITTED_LOCK.replace("revision = 3", "revision = 5").replace(
        "0.4.1", "0.4.2"
    )
    outputs, lock = _run_refresh(tmp_path, COMMITTED_LOCK, produced)

    assert "changed=true" in outputs
    assert lock == produced


def test_unchanged_lock_opens_no_refresh_pr(tmp_path: Path) -> None:
    outputs, lock = _run_refresh(tmp_path, COMMITTED_LOCK, COMMITTED_LOCK)

    assert "changed=false" in outputs
    assert lock == COMMITTED_LOCK
