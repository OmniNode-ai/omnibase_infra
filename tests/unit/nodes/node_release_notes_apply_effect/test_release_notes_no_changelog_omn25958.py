# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A release cut never writes CHANGELOG.md into a repo that removed it (OMN-25958).

The 0.4.324 cut of the marketplace repository added a 52-line root ``CHANGELOG.md``
to its default branch. That repository deleted the file under the repository
prose-removal rule (OMN-17427) and its own test refuses the file, but it carries no
``.readme-only-docs.yaml``, and the notes node treated "no policy file" as "keep the
changelog" and created the file. These tests rebuild that cut from the recorded
inputs in ``tests/fixtures/release_notes/cut_0_4_324.json``: the cut's first parent
had no ``CHANGELOG.md`` and no policy file, and the train rendered the entry stored
there.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.handlers.handler_release_notes_apply import (
    HandlerReleaseNotesApply,
    apply_release_notes,
)
from omnibase_infra.models.model_release_notes_apply_request import (
    ModelReleaseNotesApplyRequest,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "release_notes" / "cut_0_4_324.json"
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "release-train-nightly.yml"
_PR_BODY_TARGET = "release-pr-body"


def _recorded_cut() -> dict[str, Any]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _base_tree_of_the_cut(root: Path, recorded: dict[str, Any]) -> Path:
    """The cut's first parent as far as the notes node reads it."""
    root.mkdir(parents=True)
    (root / "README.md").write_text(recorded["base_readme"], encoding="utf-8")
    if recorded["base_tree_has_changelog"]:
        (root / "CHANGELOG.md").write_text("recorded\n", encoding="utf-8")
    if recorded["base_tree_has_readme_only_docs_yaml"]:
        (root / ".readme-only-docs.yaml").write_text("exceptions: []\n", "utf-8")
    return root


def _tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.mark.unit
class TestRecordedCutInputs:
    def test_the_recorded_base_tree_is_the_one_the_cut_ran_on(self) -> None:
        recorded = _recorded_cut()
        assert recorded["base_tree_has_changelog"] is False
        assert recorded["base_tree_has_readme_only_docs_yaml"] is False
        assert recorded["rendered_entry"].startswith("## v0.4.324 (2026-11-21)\n")


@pytest.mark.unit
class TestACutNeverWritesTheChangelogOfARepoThatRemovedIt:
    def test_the_recorded_cut_leaves_the_tree_unchanged(self, tmp_path: Path) -> None:
        recorded = _recorded_cut()
        clone = _base_tree_of_the_cut(tmp_path / "clone", recorded)
        before = _tree(clone)

        target, staged = apply_release_notes(clone, recorded["rendered_entry"])

        assert not (clone / "CHANGELOG.md").exists()
        assert _tree(clone) == before
        assert (target, staged) == (_PR_BODY_TARGET, [])

    @pytest.mark.asyncio
    async def test_the_recorded_cut_returns_its_block_for_the_release_pr_body(
        self, tmp_path: Path
    ) -> None:
        recorded = _recorded_cut()
        clone = _base_tree_of_the_cut(tmp_path / "clone", recorded)
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
        assert result.target == _PR_BODY_TARGET
        assert result.staged == ()
        assert result.notes == recorded["rendered_entry"].rstrip("\n")
        assert not (clone / "CHANGELOG.md").exists()

    def test_a_repo_that_still_has_a_changelog_keeps_prepending_to_it(
        self, tmp_path: Path
    ) -> None:
        recorded = _recorded_cut()
        clone = _base_tree_of_the_cut(tmp_path / "clone", recorded)
        (clone / "CHANGELOG.md").write_text("## v0.4.322 (2026-11-19)\n", "utf-8")

        target, staged = apply_release_notes(clone, recorded["rendered_entry"])

        assert (target, staged) == ("CHANGELOG.md", ["CHANGELOG.md"])
        text = (clone / "CHANGELOG.md").read_text(encoding="utf-8")
        assert text.startswith("## v0.4.324 (2026-11-21)\n")


def _cut_step_script() -> str:
    steps = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))["jobs"]["cut"][
        "steps"
    ]
    by_name = {step["name"]: step for step in steps}
    run: str = by_name["Open the release pull request"]["run"]
    return "\n".join(
        line for line in run.splitlines() if not line.lstrip().startswith("#")
    )


@pytest.mark.unit
class TestTheCutStepCommitsAnEmptyStagedList:
    def test_the_cut_step_adds_staged_paths_only_when_there_are_some(self) -> None:
        script = _cut_step_script()
        guarded = re.search(
            r'if \[ "\$\{#staged\[@\]\}" -gt 0 \]; then\s+git add -- "\$\{staged\[@\]\}"\s+fi',
            script,
        )
        assert guarded is not None
        assert script.count('git add -- "${staged[@]}"') == 1

    def test_the_cut_step_commits_even_when_nothing_is_staged(self) -> None:
        assert "git commit --allow-empty -m" in _cut_step_script()

    def test_the_cut_step_puts_the_receipts_notes_in_the_release_pr_body(self) -> None:
        script = _cut_step_script()
        assert "(.result.handler_result // .result).notes" in script
        assert "${notes_section:-}" in script
