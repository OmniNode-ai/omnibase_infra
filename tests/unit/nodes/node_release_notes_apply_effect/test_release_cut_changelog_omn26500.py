# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A cut for a repo whose release-cut gate needs ``CHANGELOG.md`` stages the entry (OMN-26500).

The release pull request of the marketplace repository for 0.4.325 (``chore: release
omnimarket 0.4.325 (OMN-18595)``, head ``e1fd1c07`` against base ``586bf466``) changed
no file: the repo has no ``.readme-only-docs.yaml`` and no root ``CHANGELOG.md``, so
the notes node answered ``release-pr-body`` and staged nothing (OMN-25958), and the
train committed with ``--allow-empty``. The repo's release-cut gate (the ticket-contract
prerequisite) admits a cut only with ``CHANGELOG.md`` in its diff, so the pull request
was red on ``lint``. The recorded facts are in
``tests/fixtures/release_notes/cut_0_4_325.json``; the gate source excerpt in it is the
repo's own, and the train reads it from the checkout instead of keeping a key of its own.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)
from omnibase_infra.handlers.handler_release_notes_apply import (
    HandlerReleaseNotesApply,
    apply_release_notes,
)
from omnibase_infra.models.model_release_notes_apply_request import (
    ModelReleaseNotesApplyRequest,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "release_notes" / "cut_0_4_325.json"
_PR_BODY_TARGET = "release-pr-body"


def _recorded() -> dict[str, Any]:
    recorded: dict[str, Any] = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    return recorded


def _gate_cut_paths(recorded: dict[str, Any]) -> frozenset[str]:
    """``_RELEASE_CUT_PATHS`` as the recorded gate excerpt defines it."""
    for node in ast.walk(ast.parse(recorded["gate_source_excerpt"])):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_RELEASE_CUT_PATHS"
            and isinstance(node.value, ast.Call)
        ):
            return frozenset(ast.literal_eval(node.value.args[0]))
    msg = "the recorded gate excerpt defines no _RELEASE_CUT_PATHS"
    raise AssertionError(msg)


def _gate_admits(recorded: dict[str, Any], paths: set[str]) -> bool:
    """The recorded gate's ``_is_release_cut`` rule over a cut's changed paths."""
    return "CHANGELOG.md" in paths and paths <= _gate_cut_paths(recorded)


def _base_tree(root: Path, recorded: dict[str, Any], *, with_gate: bool) -> Path:
    """The cut's base tree as far as the notes node reads it."""
    root.mkdir(parents=True)
    (root / "README.md").write_text("# omnimarket\n", encoding="utf-8")
    if recorded["base_tree_has_changelog"]:
        (root / "CHANGELOG.md").write_text("recorded\n", encoding="utf-8")
    if recorded["base_tree_has_readme_only_docs_yaml"]:
        (root / ".readme-only-docs.yaml").write_text("exceptions: []\n", "utf-8")
    if with_gate:
        gate = root / recorded["gate_source_path"]
        gate.parent.mkdir(parents=True)
        gate.write_text(recorded["gate_source_excerpt"], encoding="utf-8")
    return root


def _commit_base(root: Path) -> None:
    """Commit the tree as dev: the cut's diff is read against this commit."""
    _git(root, "init", "-q", "-b", "dev")
    _git(root, "add", "--", ".")
    _git(root, "commit", "-q", "-m", "dev")


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
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
    )
    return done.stdout


def _cut_diff(root: Path, staged: list[str]) -> set[str]:
    """Commit the cut the way the cut step does (``git add``, ``--allow-empty``)."""
    base = _git(root, "rev-parse", "HEAD").strip()
    if staged:
        _git(root, "add", "--", *staged)
    _git(root, "commit", "-q", "--allow-empty", "-m", "chore: release")
    return {
        path
        for path in _git(root, "diff", "--name-only", "-z", base, "HEAD").split("\0")
        if path
    }


@pytest.mark.unit
class TestTheRecordedFacts:
    def test_the_recorded_cut_is_the_one_in_the_pull_request(self) -> None:
        recorded = _recorded()
        pull_request = recorded["pull_request"]
        assert pull_request["title"] == "chore: release omnimarket 0.4.325 (OMN-18595)"
        assert pull_request["head_sha"].startswith("e1fd1c07")
        assert pull_request["base_sha"].startswith("586bf466")
        assert pull_request["changed_paths"] == []
        assert recorded["rendered_entry"].startswith("## v0.4.325 (2026-11-23)\n")

    def test_the_recorded_gate_refuses_the_empty_cut(self) -> None:
        recorded = _recorded()
        assert _gate_cut_paths(recorded) == {
            "CHANGELOG.md",
            "pyproject.toml",
            "uv.lock",
        }
        assert not _gate_admits(
            recorded, set(recorded["pull_request"]["changed_paths"])
        )


@pytest.mark.unit
class TestACutForAGatedRepoStagesTheChangelog:
    def test_the_recorded_cut_stages_the_entry_the_gate_needs(
        self, tmp_path: Path
    ) -> None:
        recorded = _recorded()
        clone = _base_tree(tmp_path / "clone", recorded, with_gate=True)
        _commit_base(clone)

        target, staged = apply_release_notes(clone, recorded["rendered_entry"])

        assert (target, staged) == ("CHANGELOG.md", ["CHANGELOG.md"])
        changelog = (clone / "CHANGELOG.md").read_text(encoding="utf-8")
        assert changelog == recorded["rendered_entry"].rstrip("\n") + "\n\n"
        assert _gate_admits(recorded, _cut_diff(clone, staged))

    @pytest.mark.asyncio
    async def test_the_node_reports_the_staged_changelog_and_no_pr_body_notes(
        self, tmp_path: Path
    ) -> None:
        recorded = _recorded()
        clone = _base_tree(tmp_path / "clone", recorded, with_gate=True)
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

        assert result.success is True
        assert result.target == "CHANGELOG.md"
        assert result.staged == ("CHANGELOG.md",)
        assert result.notes == ""

    def test_a_second_cut_of_the_same_version_is_refused(self, tmp_path: Path) -> None:
        recorded = _recorded()
        clone = _base_tree(tmp_path / "clone", recorded, with_gate=True)
        apply_release_notes(clone, recorded["rendered_entry"])

        with pytest.raises(ValueError, match=r"release 0\.4\.325 already has notes"):
            apply_release_notes(clone, recorded["rendered_entry"])


@pytest.mark.unit
class TestARepoWithoutAGateStillStagesNothing:
    def test_no_gate_file_keeps_the_notes_in_the_release_pr_body(
        self, tmp_path: Path
    ) -> None:
        recorded = _recorded()
        clone = _base_tree(tmp_path / "clone", recorded, with_gate=False)
        _commit_base(clone)

        target, staged = apply_release_notes(clone, recorded["rendered_entry"])

        assert (target, staged) == (_PR_BODY_TARGET, [])
        assert _cut_diff(clone, staged) == set()
        assert not (clone / "CHANGELOG.md").exists()

    @pytest.mark.parametrize(
        "gate_source",
        [
            pytest.param(
                '_RELEASE_CUT_PATHS = frozenset({"pyproject.toml", "uv.lock"})\n',
                id="gate-lists-no-changelog",
            ),
            pytest.param(
                'OTHER = frozenset({"CHANGELOG.md"})\n', id="changelog-under-other-name"
            ),
            pytest.param("def broken(:\n", id="gate-does-not-parse"),
        ],
    )
    def test_a_gate_that_does_not_need_the_changelog_is_not_one(
        self, tmp_path: Path, gate_source: str
    ) -> None:
        recorded = _recorded()
        clone = _base_tree(tmp_path / "clone", recorded, with_gate=False)
        gate = clone / recorded["gate_source_path"]
        gate.parent.mkdir(parents=True)
        gate.write_text(gate_source, encoding="utf-8")

        target, staged = apply_release_notes(clone, recorded["rendered_entry"])
        assert (target, staged) == (_PR_BODY_TARGET, [])

    def test_a_readme_only_policy_still_routes_to_the_readme(
        self, tmp_path: Path
    ) -> None:
        recorded = _recorded()
        clone = _base_tree(tmp_path / "clone", recorded, with_gate=True)
        (clone / ".readme-only-docs.yaml").write_text("exceptions: []\n", "utf-8")

        target, staged = apply_release_notes(clone, recorded["rendered_entry"])

        assert target == "README.md"
        assert "CHANGELOG.md" not in staged
