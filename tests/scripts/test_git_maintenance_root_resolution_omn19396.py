# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Root resolution in `scripts/git-maintenance.sh` (OMN-19396).

The script used to default OMNI_HOME and WORKTREE_ROOT to machine paths on an
old external disk, the second of them a sibling `omni_worktrees` beside the
registry root. That sibling is a stray root (operator ruling 2026-09-24).

Now OMNI_HOME is required (rule 8), and the worktree phase has no default root
at all: it refuses unless WORKTREE_ROOT is set explicitly. That refusal, and the
rest of the destructive-phase safety, is covered in
`test_git_maintenance_prune_safety_omn19396.py`.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "git-maintenance.sh"


def test_refuses_to_run_without_omni_home() -> None:
    """Fail fast: no OMNI_HOME means no guessed registry and no guessed root.
    Dry run only, so no version of the script can delete anything here."""
    env = dict(os.environ)
    for key in ("OMNI_HOME", "WORKTREE_ROOT", "ONEX_LEDGER_PATH", "GIT_PREFIX"):
        env.pop(key, None)
    result = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"],
        env=scrub_git_location_env(env),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, result.stdout
    assert "OMNI_HOME must be set" in result.stderr, result.stderr
    assert "=== Worktree Cleanup ===" not in result.stdout, result.stdout


def test_script_names_no_machine_path() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "/Volumes/" not in text
    assert "/Users/" not in text
