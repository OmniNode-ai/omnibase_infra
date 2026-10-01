# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Real Git references, package provenance and receipt boundaries (OMN-17427)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.cli import omnimarket_drift_guard as guard
from omnibase_infra.cli.omnimarket_drift_guard import (
    CanonicalCloneAttachment,
    OmnimarketDriftError,
    canonical_clone_attachment,
    canonical_local_omnimarket_commit,
    check_omnimarket_drift,
)
from tests.unit.cli.test_omnimarket_drift_guard import (
    _detach_head,
    _make_git_repo,
    _Reconciler,
)

pytestmark = pytest.mark.integration


def _scrubbed_git_env() -> dict[str, str]:
    env = scrub_git_location_env(os.environ)
    for key in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ):
        env.pop(key, None)
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_EDITOR"] = "true"
    return env


@pytest.mark.parametrize("behind", [0, 1])
@pytest.mark.parametrize("branch", ["main", "dev"])
def test_detached_registry_resolves_installed_code_against_default_without_mutating(
    tmp_path: Path, behind: int, branch: str
) -> None:
    """OMN-17427: a registry checkout is not the CLI's code authority."""
    clone = tmp_path / "omnimarket"
    clone.mkdir()
    installed = _make_git_repo(clone)
    if behind:
        subprocess.run(
            [
                "git",
                "-C",
                str(clone),
                "commit",
                "--quiet",
                "--allow-empty",
                "-m",
                "merged",
            ],
            check=True,
            env=_scrubbed_git_env(),
        )
    reference = canonical_local_omnimarket_commit(str(tmp_path))
    reference_ref = f"refs/remotes/origin/{branch}"
    subprocess.run(
        ["git", "-C", str(clone), "update-ref", reference_ref, reference],
        check=True,
        env=_scrubbed_git_env(),
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(clone),
            "symbolic-ref",
            "refs/remotes/origin/HEAD",
            reference_ref,
        ],
        check=True,
        env=_scrubbed_git_env(),
    )
    # The mirror's checkout may contain unmerged work; dispatch executes the
    # git-installed package, proved against the remote default, never this checkout.
    subprocess.run(
        [
            "git",
            "-C",
            str(clone),
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "registry-only",
        ],
        check=True,
        env=_scrubbed_git_env(),
    )
    checkout = canonical_local_omnimarket_commit(str(tmp_path))
    _detach_head(clone)
    reconciler = _Reconciler(ok=True)
    with patch.object(guard, "installed_omnimarket_commit", return_value=installed):
        verdict = check_omnimarket_drift(str(tmp_path), reconcile=reconciler)
    assert verdict is not None
    fields = verdict.as_receipt_fields()
    assert fields["mode"] == "registry-reference"
    assert fields["installed_commit"] == installed
    assert fields["reference_commit"] == reference
    assert fields["checkout_commit"] == checkout
    assert fields["reference_ref"] == reference_ref
    assert fields["commits_behind"] == behind
    assert reconciler.calls == 0
    assert canonical_local_omnimarket_commit(str(tmp_path)) == checkout
    assert (
        canonical_clone_attachment(str(tmp_path)) is CanonicalCloneAttachment.DETACHED
    )
    json.dumps(fields)


def test_detached_registry_rejects_installed_unmerged_checkout(tmp_path: Path) -> None:
    """An origin/main ref must not become a blanket detached-head bypass."""
    clone = tmp_path / "omnimarket"
    clone.mkdir()
    reference = _make_git_repo(clone)
    subprocess.run(
        ["git", "-C", str(clone), "update-ref", "refs/remotes/origin/main", reference],
        check=True,
        env=_scrubbed_git_env(),
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(clone),
            "symbolic-ref",
            "refs/remotes/origin/HEAD",
            "refs/remotes/origin/main",
        ],
        check=True,
        env=_scrubbed_git_env(),
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(clone),
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "unmerged",
        ],
        check=True,
        env=_scrubbed_git_env(),
    )
    installed = canonical_local_omnimarket_commit(str(tmp_path))
    _detach_head(clone)
    with patch.object(guard, "installed_omnimarket_commit", return_value=installed):
        with pytest.raises(OmnimarketDriftError):
            check_omnimarket_drift(str(tmp_path))
