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

from omnibase_core.models.validation.model_canonical_file_shape_change import (
    ModelCanonicalFileShapeChange,
)
from omnibase_core.models.validation.model_canonical_file_shape_facts import (
    ModelCanonicalFileShapeFacts,
)
from omnibase_core.models.validation.model_canonical_file_shape_tree import (
    ModelCanonicalFileShapeTree,
)
from omnibase_core.nodes.node_canonical_file_shape_check_compute.runtime_canonical_file_shape_check import (
    gather_facts,
)
from omnibase_core.validation.checker_canonical_file_shape import (
    check,
    load_rule_set,
)
from omnibase_core.validation.validator_canonical_file_shape import (
    ValidatorCanonicalFileShape,
)
from omnibase_core.validation.validator_git_repo import GitRepo
from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
RULES = load_rule_set(ValidatorCanonicalFileShape().contract)
DEFAULT_BASELINE = RULES.parameters.baseline_path

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
    """Judge recorded paths with typed facts and retain the no-baseline-touch rule.

    The manifest carries no blob contents. Empty texts let the checker classify
    added exception files without inventing their contents. The criterion also
    forbids touching the baseline at all, stronger than the checker's growth rule.
    """
    kept = tuple((status, path) for status, path in changes if status != "D")
    facts = ModelCanonicalFileShapeFacts(
        head_tree=ModelCanonicalFileShapeTree(files=tuple(path for _, path in kept)),
        has_base=True,
        changes=tuple(
            ModelCanonicalFileShapeChange(
                status=status,
                path=path,
                new_text="",
                old_text=None if status == "A" else "",
            )
            for status, path in kept
        ),
    )
    found = [f"{issue.file_path}: {issue.code}" for issue in check(facts, RULES)]
    for _, path in kept:
        if path == DEFAULT_BASELINE:
            found.append(f"{path}: the baseline is touched by this change")
    return found


def test_the_recorded_change_adds_no_noncanonical_file() -> None:
    assert shape_violations(CHANGE_MANIFEST) == []


def test_planted_script_allowlist_and_baseline_paths_are_refused() -> None:
    assert shape_violations([("A", "scripts/zz_planted.py")])
    assert shape_violations([("A", "config/zz_planted_allowlist.yaml")])
    assert shape_violations([("M", DEFAULT_BASELINE)])
    assert shape_violations([("A", "src/omnibase_infra/nodes/node_x/handler.py")]) == []


def test_the_recorded_manifest_is_what_git_holds_for_the_commit() -> None:
    env = scrub_git_location_env()
    # A lab runner may hide .git and declare this checkout explicitly instead.
    # Keep that identity only when it names this tree, never a caller's fixture.
    if os.environ.get("GIT_WORK_TREE") == str(REPO_ROOT):
        env.update(
            {
                key: os.environ[key]
                for key in ("GIT_DIR", "GIT_WORK_TREE")
                if key in os.environ
            }
        )
    repo = GitRepo(
        REPO_ROOT, env, cat_file_timeout_s=RULES.parameters.cat_file_timeout_s
    )
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
    # The criterion is a property of the change: a file the parent tree already
    # held is not the change's. A newer classifier (core 0.47.55, OMN-21332:
    # every shell script) refuses files the parent already carried, so judge
    # only the findings the commit adds over its parent's own tree.
    inherited = {
        (f.file_path, f.code)
        for f in check(gather_facts(repo, RULES.parameters, parent, None, None), RULES)
    }
    added = [
        f
        for f in check(
            gather_facts(repo, RULES.parameters, CHANGE_COMMIT, parent, None), RULES
        )
        if (f.file_path, f.code) not in inherited
    ]
    assert added == []
