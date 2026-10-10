# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20696 AC4: the change that vendored migration 0004 added no non-canonical file.

The criterion is a property of a diff: no script, plugin, allowlist entry or
suppression, and no grown baseline (RULING 2026-10-01T12:36:54Z). The change
landed here as omnibase_infra#4665. The canonical-file-shape ratchet
(omnibase_core ``canonical_file_shape``, OMN-20304) judges a diff by its paths,
so this test judges the change's recorded paths with the ratchet's own
classifiers, and, when this checkout holds the commit, reads the paths back
from git and runs the ratchet's ``check`` over that commit as well.

The manifest is checked everywhere; the git readback only adds to it. A planted
script, allowlist and baseline path must each be refused, so a clean verdict is
never the only evidence that the evaluator can fail.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import pytest

from omnibase_core.validators.canonical_file_shape import (
    DEFAULT_BASELINE,
    GitRepo,
    check,
    is_canonical_location,
    is_code_file,
    is_exception_file,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]

#: omnibase_infra#4665, the squash commit that vendored migration 0004.
CHANGE_COMMIT = "eed1c81562f601f7d15fb9a9a0810da1b2b381cf"

#: ``git show --name-status`` of that commit.
CHANGE_MANIFEST: tuple[tuple[str, str], ...] = (
    ("M", "config/migration_classes.yaml"),
    ("M", "docker/migrations/forward/_ledger/application-migrations.tsv"),
    (
        "A",
        "docker/migrations/forward/nodes/node_projection_dod_verdict/"
        "0004_dod_verify_runs_contract_subject.sql",
    ),
    ("M", "tests/integration/migrations/test_dod_verify_runs_omn18900.py"),
)


def shape_violations(changes: Iterable[tuple[str, str]]) -> list[str]:
    """Every path in a change the ratchet would refuse."""
    found: list[str] = []
    for status, path in changes:
        if status == "D":
            continue
        if is_code_file(path, None) and not is_canonical_location(path):
            found.append(f"{path}: code file outside a canonical location")
        if path == DEFAULT_BASELINE:
            found.append(f"{path}: the baseline is touched by this change")
        if status == "A" and is_exception_file(path, DEFAULT_BASELINE):
            found.append(f"{path}: new allowlist, baseline or suppression file")
    return found


def test_the_recorded_change_adds_no_noncanonical_file() -> None:
    assert shape_violations(CHANGE_MANIFEST) == []


def test_planted_script_allowlist_and_baseline_paths_are_refused() -> None:
    assert shape_violations([("A", "scripts/zz_planted.py")])
    assert shape_violations([("A", "config/zz_planted_allowlist.yaml")])
    assert shape_violations([("M", DEFAULT_BASELINE)])
    assert shape_violations([("A", "src/omnibase_infra/nodes/node_x/handler.py")]) == []


def test_the_recorded_manifest_is_what_git_holds_for_the_commit() -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    repo = GitRepo(REPO_ROOT, env)
    parent = f"{CHANGE_COMMIT}^"
    if not repo.has_revision(CHANGE_COMMIT) or not repo.has_revision(parent):
        # A shallow checkout cannot read the commit back; the recorded
        # manifest above is then the whole check, and a full checkout adds this.
        return
    recorded = {
        (status, path)
        for status, path, _old in repo.changed_paths(parent, CHANGE_COMMIT)
    }
    assert recorded == set(CHANGE_MANIFEST)
    assert check(repo, CHANGE_COMMIT, parent) == []
