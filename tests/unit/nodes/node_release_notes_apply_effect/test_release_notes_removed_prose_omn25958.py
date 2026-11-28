# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A cut never writes CHANGELOG.md into a repo whose removed-prose policy lists it (OMN-25958).

The 0.4.330 release pull request of the marketplace repository (omnimarket#7055)
re-added the root ``CHANGELOG.md``: its release-cut gate still listed the file among
its cut paths, so ``release_cut_gate_needs_changelog`` answered yes and the notes node
wrote it. The same repository's ``REMOVED_PROSE`` list in
``tests/ci/test_repository_prose_removed_omn17427.py`` names ``CHANGELOG.md`` as a file
that must not exist, and three dev tests failed until omnimarket#7069 deleted it.
The removed-prose list is the repo's declared changelog-free policy, and it outranks
the gate listing and an existing file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.handlers.handler_release_notes_apply import (
    apply_release_notes,
    release_notes_target,
)

_PROSE_TEST = "tests/ci/test_repository_prose_removed_omn17427.py"
_GATE = (
    "src/omnimarket/nodes/node_ticket_contract_prerequisite_effect/handlers/"
    "handler_ticket_contract_prerequisite.py"
)
_ENTRY = (
    "## v0.4.330 (2026-11-28)\n\n### Included Since v0.4.329\n- fix: something (#1)\n"
)


def _tree(root: Path, *, prose_lists_changelog: bool, gate_lists: bool) -> Path:
    """The 0.4.330 cut's base tree as far as the notes node reads it."""
    root.mkdir(parents=True)
    (root / "README.md").write_text("# omnimarket\n", encoding="utf-8")
    prose = root / _PROSE_TEST
    prose.parent.mkdir(parents=True)
    listed = '    "CHANGELOG.md",\n' if prose_lists_changelog else ""
    prose.write_text(
        f'REMOVED_PROSE = (\n    "SECURITY.md",\n{listed})\n', encoding="utf-8"
    )
    gate = root / _GATE
    gate.parent.mkdir(parents=True)
    paths = '"CHANGELOG.md", "pyproject.toml", "uv.lock"' if gate_lists else '"uv.lock"'
    gate.write_text(f"_RELEASE_CUT_PATHS = frozenset({{{paths}}})\n", encoding="utf-8")
    return root


@pytest.mark.unit
class TestRemovedProsePolicyOutranksTheGate:
    def test_the_0_4_330_cut_writes_no_changelog(self, tmp_path: Path) -> None:
        clone = _tree(tmp_path / "c", prose_lists_changelog=True, gate_lists=True)

        target, staged = apply_release_notes(clone, _ENTRY)

        assert (target, staged) == ("release-pr-body", [])
        assert not (clone / "CHANGELOG.md").exists()

    def test_an_existing_changelog_is_not_prepended_to_either(
        self, tmp_path: Path
    ) -> None:
        clone = _tree(tmp_path / "c", prose_lists_changelog=True, gate_lists=False)
        (clone / "CHANGELOG.md").write_text("old\n", encoding="utf-8")

        assert release_notes_target(clone) == "release-pr-body"
        apply_release_notes(clone, _ENTRY)
        assert (clone / "CHANGELOG.md").read_text(encoding="utf-8") == "old\n"

    def test_a_repo_that_does_not_list_it_keeps_the_gate_rule(
        self, tmp_path: Path
    ) -> None:
        clone = _tree(tmp_path / "c", prose_lists_changelog=False, gate_lists=True)

        assert release_notes_target(clone) == "CHANGELOG.md"
