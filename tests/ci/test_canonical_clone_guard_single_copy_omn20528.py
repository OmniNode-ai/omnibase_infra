# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""omnibase_infra holds no copy of the canonical-clone git guard (OMN-20528).

The guard is the `node_canonical_clone_guard_effect` node, which this repository does not hold. Its refusal
path writes with `printf` only; the old shell guard that this repository used to track wrote
its refusal with `cat >&2 <<EOF`, and bash 5.3 can block forever writing that here-document
into a pipe the same process reads later (a `git checkout --force -B` hung for 46 hours).
This repository's installer also copied the shell guard into the registry root and pointed
`core.hooksPath` at it, so a host could end up running a second, stale copy beside the node.

The guard therefore exists in one place. These tests fail if a shell guard, its installer or
its hook-type directory comes back here.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD_BASENAMES = (
    "canonical_clone_guard.sh",
    "canonical_clone_ref_guard.sh",
    "canonical_clone_paths.sh",
    "install-canonical-clone-git-hooks.sh",
)


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files"],
        capture_output=True,
        text=True,
        check=True,
        env=scrub_git_location_env(),
    )
    return result.stdout.splitlines()


def test_no_shell_copy_of_the_canonical_clone_guard_is_tracked() -> None:
    offenders = [
        path for path in _tracked_files() if Path(path).name in GUARD_BASENAMES
    ]
    assert offenders == [], (
        "the canonical-clone guard lives in the guard node only; remove "
        f"these copies: {offenders}"
    )


def test_no_canonical_clone_hook_type_directory_is_tracked() -> None:
    offenders = [
        path for path in _tracked_files() if "git-hooks/canonical-clone/" in path
    ]
    assert offenders == [], (
        "hook-type links for the canonical-clone guard belong to the guard node's "
        f"installer; remove: {offenders}"
    )


def test_control_the_tracked_file_listing_is_not_empty() -> None:
    """Positive control: an empty listing would make both checks above pass vacuously."""
    assert len(_tracked_files()) > 1000
