# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""PR-time gate: a cascade bump PR body must carry no foreign evidence stamp.

Why this file exists
---------------------
``dependency-cascade.yml`` used to write the RELEASING PR's own
``Evidence-Ticket`` / ``Evidence-Source`` into every downstream bump PR body,
verbatim. That citation is honest about where the release's own evidence
lives, but it does not bind to the downstream PR: ``occ-preflight`` resolves
the stamped commit and finds zero receipts for the downstream PR's own number
or head sha, failing permanently with ``reason=pr_ticket_mismatch`` -- no
rerun clears it, because the stamped tree never changes.

This is not a one-off. It was hit on the v0.47.10 wave (OMN-18202, the ticket
this file closes AC1 of), the 0.47.17/0.47.18 wave (OMN-18853),
``omnibase_infra#3970``, and three cascade legs each on 2026-09-22 and
2026-09-23 (OMN-17427) -- past the second-occurrence bar the last of those
filed a FRICTION row over.

The fix (OMN-18202 AC1): the generator emits NO Evidence-Ticket /
Evidence-Source line at all. Each downstream repo's own OCC autobind fires on
``opened`` (its condition is a ticket token in the branch name, which this
generator still writes), mints a companion carrying receipts bound to the
downstream PR's own commit, and PATCHes the stamp in itself -- leaving exactly
one writer of that field instead of two that can disagree.

Pinned two ways, both against the shipped file rather than a copied string:

1. the body template the generator writes must carry no stamp line (RED
   before the fix, GREEN after -- see the git history of this file for the
   removed content this replaced);
2. the workflow must carry a structural step that reads back a PR it just
   created and fails the run if a stamp is present anyway, so a future edit
   that reintroduces a stamp in the template fails loud in CI rather than
   waiting for a downstream ``occ-preflight`` failure to surface it days
   later.

Ticket: OMN-18202 (AC1). Related, not duplicates: OMN-18853 (the restamp /
idempotent-commit fix for a DIFFERENT arm of this same defect class -- the
inherited-stamp problem when a stamp IS present and needs superseding --
explicitly out of scope here, which deliberately does NOT change the
cascade's citation convention; this file's fix removes the convention that
one is downstream of).
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "dependency-cascade.yml"

_BODY_OPEN = "# >>> OMN-18202 no-self-stamp body >>>"
_BODY_CLOSE = "# <<< OMN-18202 no-self-stamp body <<<"

# The exact shape occ-preflight, the Receipt Gate and OCC Companion Merged
# Gate all parse (rule 15: this is the parsing logic itself, the one place the
# literal is legitimate -- never in a PR body or commit message).
_STAMP_LINE = re.compile(r"^\s*Evidence-(Ticket|Source):", re.MULTILINE)


def _body_block() -> str:
    """Lift the delimited PR-body-writing block out of the shipped workflow."""
    text = _WORKFLOW.read_text(encoding="utf-8")
    start = text.index(_BODY_OPEN)
    end = text.index(_BODY_CLOSE, start)
    return text[start:end]


def _heredoc_content(block: str) -> str:
    """Extract just the text between the ``<<'PREOF'`` markers.

    That is what actually becomes the PR body -- the surrounding bash
    (``gh pr create --body "$(cat <<'PREOF' ... PREOF)"``) never reaches
    GitHub.
    """
    start = block.index("<<'PREOF'\n") + len("<<'PREOF'\n")
    end = block.index("\n          PREOF", start)
    return block[start:end]


class TestCascadeBodyCarriesNoForeignStamp:
    def test_the_generated_body_has_no_evidence_stamp_line(self) -> None:
        heredoc = _heredoc_content(_body_block())
        match = _STAMP_LINE.search(heredoc)
        assert match is None, (
            "the cascade's PR body template writes an Evidence-Ticket or "
            f"Evidence-Source line ({match.group(0)!r} if not None) -- this "
            "is the OMN-18202/OMN-18853/OMN-17427 pr_ticket_mismatch "
            "regression. The generator must never stamp a downstream bump PR "
            "with the upstream release's companion; let the downstream "
            "repo's own occ-autobind mint and PATCH its own."
        )

    def test_the_body_block_still_creates_a_real_pr(self) -> None:
        """A body with no stamp is not the same defect as no PR at all."""
        block = _body_block()
        assert "gh pr create" in block
        assert '--head "${{ steps.vars.outputs.branch }}"' in block
        assert '--base "${{ steps.vars.outputs.base }}"' in block

    def test_the_create_step_records_whether_it_actually_created_a_pr(self) -> None:
        """The verify step below must be able to tell 'created' from 'skipped'.

        Without this output, the verify step cannot distinguish a PR this run
        just opened from one found already open (which may predate this fix
        and carry a legacy stamp this run had no part in writing) -- flagging
        the latter would fail a cascade run over a pre-existing PR rather than
        the regression this guards.
        """
        block = _body_block()
        assert 'echo "created=true" >> "$GITHUB_OUTPUT"' in block


class TestCascadeHasAStructuralNoForeignStampGuard:
    def test_a_verification_step_reads_back_the_pr_it_just_opened(self) -> None:
        text = _WORKFLOW.read_text(encoding="utf-8")
        assert "Verify no foreign Evidence-Source stamp was emitted" in text, (
            "the workflow no longer carries the OMN-18202 structural guard "
            "step -- a future template edit that reintroduces a stamp would "
            "then only be caught days later by a downstream occ-preflight "
            "failure, exactly the discovery cost this ticket is closing"
        )

    def test_the_guard_step_greps_for_the_stamp_line_shape(self) -> None:
        text = _WORKFLOW.read_text(encoding="utf-8")
        start = text.index("Verify no foreign Evidence-Source stamp was emitted")
        end = text.index("\n\n", start)
        step = text[start:end]
        assert "grep -qE '^Evidence-(Ticket|Source):'" in step
        assert "exit 1" in step

    def test_the_guard_step_only_runs_on_a_pr_it_actually_created(self) -> None:
        """The guard's `if:` must key off the create step's own output.

        Not off `env.SKIP` alone -- SKIP only tells you the lockfile changed,
        it says nothing about whether THIS run opened a new PR versus found
        one already open under one of the three legacy branch shapes.
        """
        text = _WORKFLOW.read_text(encoding="utf-8")
        start = text.index("Verify no foreign Evidence-Source stamp was emitted")
        end = text.index("\n        run:", start)
        header = text[start:end]
        assert "steps.open_pr.outputs.created == 'true'" in header


class TestCascadeCreateStepHasAStableId:
    def test_the_open_pull_request_step_is_named_open_pr(self) -> None:
        """The guard step's `if:` resolves `steps.open_pr` -- pin the id exists."""
        text = _WORKFLOW.read_text(encoding="utf-8")
        idx = text.index("- name: Open pull request")
        # The `id:` line must appear before the next `- name:` step boundary.
        next_step = text.index("\n      - name:", idx + 1)
        segment = text[idx:next_step]
        assert "id: open_pr" in segment
