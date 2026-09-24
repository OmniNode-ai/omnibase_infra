# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The generated release CHANGELOG must never trip the doc-content scan (OMN-18595).

The scheduled release train's changelog generator (``render_changelog_entry`` in
``scripts/ci/release_train.py``) embeds each unreleased commit's subject verbatim.
Squash-merge subjects in this fleet routinely carry a bare ``OMN-<digits>`` ticket
reference, either as the conventional-commit scope (``feat(OMN-19339): ...``) or
parenthetically in the body (``... (OMN-18157) (#1745)``). ``CHANGELOG.md`` is not
under ``onex_change_control/`` or ``contracts/``, so the omnibase_core doc-content
scan's TICKET_REFERENCE rule (``omnibase_core.validation.doc_content_scan.handler``)
flags every such line -- confirmed live on omnibase_core#1746 (2026-09-24), where the
scan was suppressed by hand, per-line, on the release branch. That fix does not
survive the next cut. This fixes the generator at the source instead: strip the bare
ticket reference before it is written, mirroring the precedent every hand-cut release
entry already set (PR-number-only traceability, e.g. ``(#1736)``).

Every hand-cut CHANGELOG entry omits the ticket id already -- this generator is the
one path that was embedding it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE = _REPO_ROOT / "scripts" / "ci" / "release_train.py"


def _load() -> Any:
    """Import the module under test by path, matching the sibling test files."""
    name = "release_train_changelog_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _MODULE)
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        msg = f"cannot load {_MODULE}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rt = _load()


def _scan_source() -> Any:
    """The real doc-content scanner -- reproduces the flag, not a stand-in."""
    from omnibase_core.validation.doc_content_scan.handler import scan_source

    return scan_source


# Real squash-merge subjects observed on omnibase_core dev (2026-09-24), the exact
# shape that flagged omnibase_core#1746.
_TICKET_AS_SCOPE = "feat(OMN-19339): sim-202 lane, an isolated stack on .202 (#4041)"
_TICKET_PARENTHETICAL = (
    "fix(ci): Contract Compliance reads contracts at the PR's own "
    "evidence commit (OMN-18157) (#1745)"
)
_NO_TICKET = "chore(deps): bump the actions group with 8 updates (#1742)"


class TestStripTicketReferences:
    """Unit coverage of the extracted helper -- exact text, not just "still flagged"."""

    def test_strips_ticket_used_as_conventional_commit_scope(self) -> None:
        result = rt._strip_ticket_references(_TICKET_AS_SCOPE)
        assert "OMN-19339" not in result
        assert result == "feat: sim-202 lane, an isolated stack on .202 (#4041)"

    def test_strips_ticket_cited_parenthetically(self) -> None:
        result = rt._strip_ticket_references(_TICKET_PARENTHETICAL)
        assert "OMN-18157" not in result
        assert result == (
            "fix(ci): Contract Compliance reads contracts at the PR's own "
            "evidence commit (#1745)"
        )

    def test_subject_with_no_ticket_is_unchanged(self) -> None:
        assert rt._strip_ticket_references(_NO_TICKET) == _NO_TICKET

    def test_bare_mention_with_no_parens_is_stripped(self) -> None:
        result = rt._strip_ticket_references("touches OMN-4242 inline, no parens")
        assert "OMN-4242" not in result
        assert "  " not in result


class TestRenderChangelogEntryClearsTheScan:
    """RED reproduces the omnibase_core#1746 failure; GREEN proves the source fix.

    Before the fix, ``render_changelog_entry`` embedded ``subjects`` verbatim and
    this test's first assertion (``flagged is False``) failed with a
    TICKET_REFERENCE finding on the ``feat(OMN-19339): ...`` line -- reproduced
    directly against the real scanner, not asserted from the diff alone.
    """

    def test_generated_entry_is_clean_against_the_real_doc_content_scan(self) -> None:
        scan_source = _scan_source()
        entry = rt.render_changelog_entry(
            package="omnibase-core",
            version="0.47.24",
            previous_tag="v0.47.23",
            subjects=(_TICKET_AS_SCOPE, _TICKET_PARENTHETICAL, _NO_TICKET),
        )
        result = scan_source(entry, "CHANGELOG.md")
        assert result.flagged is False, result.findings
        # The PR-number citations survive -- traceability is not lost, only the
        # internal ticket id is.
        assert "(#4041)" in entry
        assert "(#1745)" in entry
        assert "(#1742)" in entry

    def test_positive_control_the_scan_still_catches_a_real_violation(self) -> None:
        """A zero finding above is meaningful only if the scanner still fires.

        Feeds the scanner a line the generator does NOT produce (a raw, unstripped
        ticket reference) to prove the clean result above is a real pass, not a
        scanner that stopped looking.
        """
        scan_source = _scan_source()
        result = scan_source(f"- {_TICKET_AS_SCOPE}\n", "CHANGELOG.md")
        assert result.flagged is True
        assert any(
            f.violation_type.value == "ticket_reference" for f in result.findings
        )
