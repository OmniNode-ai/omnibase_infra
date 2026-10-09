# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pin the PR-title exemption mirror still used by Contract Compliance.

The OCC-only conditional sweep tests were removed with their caller jobs.
The shared title-rule helper remains live and must match its pinned source.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from scripts.ci.ci_summary_gate import title_rule_exempts_ticket

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]

# The ref .github/workflows/pr-title-check.yml in THIS repo pins. Read from
# that file rather than retyped, so the two cannot disagree.
_TITLE_CHECK_CALLER = REPO_ROOT / ".github/workflows/pr-title-check.yml"
_UPSTREAM_SLUG = "OmniNode-ai/onex_change_control"
_UPSTREAM_PATH = ".github/workflows/pr-title-check-reusable.yml"


def _pinned_title_check_ref() -> str:
    text = _TITLE_CHECK_CALLER.read_text(encoding="utf-8")
    match = re.search(
        rf"{re.escape(_UPSTREAM_SLUG)}/{re.escape(_UPSTREAM_PATH)}@([0-9a-f]{{40}})",
        text,
    )
    assert match, (
        f"no pinned 40-hex ref for the title reusable in {_TITLE_CHECK_CALLER}"
    )
    return match.group(1)


class TestTheTitleRuleMirrorIsPinnedToItsSource:
    """AC6 — an upstream edit is a red test, not silent drift."""

    def test_the_caller_still_pins_the_ref_this_mirror_was_read_from(self) -> None:
        assert _pinned_title_check_ref() == (
            "326ebee0561ab42b3d970e533ae9f447abed2292"
        ), (
            "the PR-title reusable pin moved. Re-read its exemption arms at the new "
            "ref and update title_rule_exempts_ticket and this pin together, in one "
            "change — the mirror is only honest while these two agree."
        )

    def test_the_module_comment_names_the_source_repo_path_and_pin(self) -> None:
        source = (REPO_ROOT / "scripts/ci/ci_summary_gate.py").read_text(
            encoding="utf-8"
        )
        for needle in (_UPSTREAM_SLUG, _UPSTREAM_PATH, _pinned_title_check_ref()):
            assert needle in source, needle

    @pytest.mark.parametrize(
        ("author", "title", "expected"),
        [
            # Arm 1 — any login ending in the bot suffix.
            ("dependabot[bot]", "literally anything", True),
            ("renovate[bot]", "feat: a feature", True),
            ("coderabbitai[bot]", "whatever", True),
            # Arm 2 — dependency bump titles, case-insensitive.
            ("jonahgabriel", "chore(deps): bump x from 1 to 2", True),
            ("jonahgabriel", "CHORE(DEPS): bump x", True),
            ("jonahgabriel", "build(deps-dev): bump y", True),
            ("jonahgabriel", "Bump actions/checkout from 4 to 5", True),
            # ...and the upstream requires the trailing space on "bump ".
            ("jonahgabriel", "bumpy road ahead", False),
            # Arm 3 — release titles.
            ("jonahgabriel", "chore: release 1.2.3", True),
            ("jonahgabriel", "chore(release): 1.2.3", True),
            ("jonahgabriel", "release: 1.2.3", True),
            # Not exempt: ordinary work, ticketed or not. Arm 4 (the OMN token)
            # is COMPLIANCE, not exemption, and is deliberately not mirrored.
            ("jonahgabriel", "feat(OMN-19167): a change", False),
            ("jonahgabriel", "feat: a change", False),
            ("jonahgabriel", "fix: deps got bumped", False),
            # Empty inputs resolve nothing.
            ("", "chore(deps): bump x", False),
            ("jonahgabriel", "", False),
        ],
    )
    def test_the_mirror_matches_the_upstream_arms(
        self, author: str, title: str, expected: bool
    ) -> None:
        assert title_rule_exempts_ticket(author=author, title=title) is expected

    def test_the_mirror_agrees_with_the_upstream_shell_on_every_row(self) -> None:
        """Differential control: run the upstream's own bash against the table.

        A hand-written table can encode the same misreading twice. This runs
        the upstream logic as bash, exactly as the reusable does, and compares.
        Skipped where bash is unavailable rather than silently passing.
        """

        script = r"""
        TITLE_LOWER=$(echo "$PR_TITLE" | tr '[:upper:]' '[:lower:]')
        if [[ -z "$PR_TITLE" ]]; then exit 1; fi
        if [[ "$PR_AUTHOR" == *"[bot]" ]]; then exit 0; fi
        if [[ "$TITLE_LOWER" =~ ^(chore\(deps|build\(deps|bump ) ]]; then exit 0; fi
        if [[ "$TITLE_LOWER" =~ ^(chore:\ release|chore\(release\)|release:) ]]; then exit 0; fi
        exit 1
        """
        cases = [
            ("dependabot[bot]", "literally anything"),
            ("jonahgabriel", "chore(deps): bump x from 1 to 2"),
            ("jonahgabriel", "CHORE(DEPS): bump x"),
            ("jonahgabriel", "build(deps-dev): bump y"),
            ("jonahgabriel", "Bump actions/checkout from 4 to 5"),
            ("jonahgabriel", "bumpy road ahead"),
            ("jonahgabriel", "chore: release 1.2.3"),
            ("jonahgabriel", "chore(release): 1.2.3"),
            ("jonahgabriel", "release: 1.2.3"),
            ("jonahgabriel", "feat(OMN-19167): a change"),
            ("jonahgabriel", "feat: a change"),
            ("jonahgabriel", ""),
        ]
        for author, title in cases:
            try:
                proc = subprocess.run(
                    ["bash", "-c", script],
                    env={
                        "PR_AUTHOR": author,
                        "PR_TITLE": title,
                        "PATH": "/usr/bin:/bin",
                    },
                    capture_output=True,
                    check=False,
                )
            except FileNotFoundError:  # pragma: no cover - bash is present in CI
                pytest.skip("bash unavailable")
            upstream = proc.returncode == 0
            mirror = title_rule_exempts_ticket(author=author, title=title)
            assert mirror is upstream, (author, title, mirror, upstream)
