# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""PR-time gate: a cascade bump branch carries a ticket, so autobind fires.

Why this file exists
--------------------
OCC autobind does not decline dependency-bump PRs on the merits. It never runs
on them at all. Every repo's ``call-occ-autobind.yml`` gates the job on::

    contains(pull_request.title, 'OMN-') || contains(head.ref, 'OMN-')

The cascade generated the title ``chore(deps): bump omnibase-infra to 0.38.29``
and the branch ``automation/bump-omnibase-infra-0.38.29``. Neither carries a
ticket, so the condition is false and the job reports ``skipping``.

The title cannot carry one: the pr-title gate exempts the ``chore(deps``
prefix precisely so dependency bumps need no ticket. That leaves the branch,
which is the arm this change fixes.

The cost was paid by people. ``onex_change_control#10026``, the companion for
``omnimemory#511``'s bump, was authored by a human at 2026-09-17T15:04:42Z --
hand-writing a companion a machine should have minted.
``onex_change_control#10047`` for ``omniclaude#2219`` was App-authored but only
at 17:05:14Z, against a PR opened at 13:50:03Z, leaving it blocked meanwhile.

Precedent, ported rather than invented
--------------------------------------
``omnimarket#2395`` ("bind generated bump branches to ticket") solved exactly
this. Its ``release-on-merge.yml`` generates
``automation/omn-18010-post-release-dev-bump-<version>``, satisfying the
``head.ref`` arm. GitHub's ``contains()`` is case-insensitive, which the
consecutive automated omnimarket bump PRs prove live.

``omnimarket#2366`` ("a bump PR guard must see the peer job's branch") is the
other half and is why the dedupe assertions below exist: when the branch shape
changes, an already-open check that knows only the new shape opens a DUPLICATE
beside an in-flight PR on the old one.

Ticket: OMN-18596
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "dependency-cascade.yml"

# The condition every repo's call-occ-autobind.yml applies to the head ref.
# Case-insensitive, because GitHub expression `contains()` is.
_AUTOBIND_TICKET = re.compile(r"omn-\d+", re.IGNORECASE)

_OPEN = "# >>> OMN-18596 bump branch >>>"
_CLOSE = "# <<< OMN-18596 bump branch <<<"


def _branch_program() -> str:
    """Lift the delimited branch builder out of the shipped workflow."""
    body = _WORKFLOW.read_text(encoding="utf-8")
    start = body.index(_OPEN)
    end = body.index(_CLOSE, start)
    block = body[start:end]
    lines = [
        line[10:] if line.startswith(" " * 10) else line for line in block.splitlines()
    ]
    return "\n".join(lines)


def _run_builder(
    package: str,
    version: str,
    ticket: str = "OMN-15449",
    evidence_source: str = "OCC#6695",
) -> tuple[int, str, dict[str, str]]:
    """Run the shipped builder; return (exit code, stdout, parsed outputs)."""
    import os
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "gh_output"
        out.touch()
        result = subprocess.run(
            ["bash", "-c", _branch_program()],
            capture_output=True,
            text=True,
            check=False,
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "GITHUB_OUTPUT": str(out),
                "PACKAGE": package,
                "VERSION": version,
                "TICKET": ticket,
                "EVIDENCE_SOURCE": evidence_source,
            },
        )
        parsed = {}
        for line in out.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                parsed[k] = v
        return result.returncode, result.stdout, parsed


def _rendered_branch(package: str, version: str, ticket: str = "OMN-15449") -> str:
    code, stdout, parsed = _run_builder(package, version, ticket=ticket)
    assert code == 0, stdout
    assert "branch" in parsed, "the branch builder wrote no branch= output"
    return parsed["branch"]


class TestTheBumpBranchSatisfiesAutobind:
    def test_the_generated_branch_carries_the_releases_own_ticket(self) -> None:
        """The whole defect in one assertion, now on the converged convention.

        Asserted against the SAME regex shape autobind's condition applies,
        not against a literal, so a branch that merely looks ticketed to a
        human but not to the gate still fails. And asserted with a ticket that
        is NOT this lane's own, so a fixed ticket baked back in would fail.
        """
        branch = _rendered_branch("omnibase_infra", "0.38.31", ticket="OMN-15449")
        assert _AUTOBIND_TICKET.search(branch), (
            f"the cascade bump branch {branch!r} carries no OMN- ticket, so "
            "every repo's call-occ-autobind.yml condition is false and the "
            "companion mint never runs. A person then hand-authors it, which "
            "is what happened on onex_change_control#10026"
        )
        assert branch.endswith("-omn-15449"), (
            "OMN-16286's convention carries the RELEASE's ticket as a SUFFIX; "
            f"{branch!r} does not, so this repo has drifted from omnibase_core "
            "and the fleet is carrying two conventions for one problem"
        )
        assert "omn-18596" not in branch, (
            "the interim fixed-prefix shape is back; the ticket must be the "
            "release's own, not the one that built the mechanism"
        )

    def test_a_different_release_ticket_produces_a_different_branch(self) -> None:
        """Proves the ticket is really an input and not a constant."""
        a = _rendered_branch("omnibase_infra", "0.38.31", ticket="OMN-15449")
        b_ = _rendered_branch("omnibase_infra", "0.38.31", ticket="OMN-18295")
        assert a != b_
        assert a.endswith("-omn-15449") and b_.endswith("-omn-18295")

    def test_the_branch_still_identifies_its_package_and_version(self) -> None:
        """Adding the ticket must not cost the branch its meaning."""
        branch = _rendered_branch("omnibase_infra", "0.38.31")
        assert "omnibase-infra" in branch
        assert "0.38.31" in branch

    def test_two_packages_do_not_collide_on_one_branch(self) -> None:
        a = _rendered_branch("omnibase_core", "0.47.17")
        b = _rendered_branch("omnibase_spi", "0.23.4")
        assert a != b
        assert _AUTOBIND_TICKET.search(a) and _AUTOBIND_TICKET.search(b)

    def test_a_leading_v_is_stripped_from_the_version(self) -> None:
        """Release inputs arrive tagged; the branch must not carry the v."""
        assert _rendered_branch("omnibase_infra", "v0.38.31") == _rendered_branch(
            "omnibase_infra", "0.38.31"
        )


class TestTheRenameCannotOpenADuplicate:
    """omnimarket#2366's lesson, which is the half a rename usually forgets."""

    def test_the_already_open_check_matches_the_legacy_branch_shape_too(
        self,
    ) -> None:
        body = _WORKFLOW.read_text(encoding="utf-8")
        assert "legacy_branch" in body, (
            "changing the branch shape without teaching the already-open check "
            "the OLD shape opens a second bump PR beside any in-flight one; "
            "that is the defect omnimarket#2366 fixed"
        )

    def test_the_legacy_shape_is_the_one_actually_in_flight(self) -> None:
        """Pinned to the real shape, so a typo in it is a red test.

        `automation/bump-<pkg>-<version>` is what the cascade generated until
        this change, and what any PR opened before it merges still carries.
        """
        body = _WORKFLOW.read_text(encoding="utf-8")
        assert "automation/bump-${PKG_HYPHEN}-${VERSION}" in body, (
            "the legacy branch shape the dedupe falls back to must be the "
            "shape the cascade actually generated, character for character"
        )

    def test_the_dedupe_exits_without_opening_when_either_shape_is_open(
        self,
    ) -> None:
        body = _WORKFLOW.read_text(encoding="utf-8")
        open_pr_step = body[body.index("- name: Open pull request") :]
        creates_at = open_pr_step.index("gh pr create")
        guard = open_pr_step[:creates_at]
        assert guard.count("--head") >= 2, (
            "the already-open check must query BOTH branch shapes before "
            "creating anything; one query cannot see a PR on the other shape"
        )
        assert "exit 0" in guard


class TestEveryInterpolatedFieldIsValidatedFirst:
    """OMN-16286. A cascade that interpolates an unvalidated input into a ref
    is how a malformed release input becomes a malformed ref, a malformed PR
    title, or worse. Each refusal is asserted separately so one over-broad
    pattern cannot mask another field going unchecked."""

    def test_an_unticketed_release_refuses_rather_than_guessing(self) -> None:
        code, stdout, parsed = _run_builder("omnibase_infra", "0.38.31", ticket="")
        assert code == 1
        assert "inputs.ticket" in stdout
        assert "branch" not in parsed, "refused, so it must write no branch"

    def test_a_malformed_ticket_refuses(self) -> None:
        code, stdout, _ = _run_builder("omnibase_infra", "0.38.31", ticket="JIRA-7")
        assert code == 1
        assert "inputs.ticket" in stdout

    def test_a_malformed_version_refuses(self) -> None:
        code, stdout, _ = _run_builder("omnibase_infra", "not-a-version")
        assert code == 1
        assert "inputs.version" in stdout

    def test_a_malformed_package_refuses(self) -> None:
        code, stdout, _ = _run_builder("omni base;rm -rf /", "0.38.31")
        assert code == 1
        assert "inputs.package" in stdout

    def test_a_malformed_evidence_source_refuses(self) -> None:
        code, stdout, _ = _run_builder(
            "omnibase_infra", "0.38.31", evidence_source="trust me"
        )
        assert code == 1
        assert "inputs.evidence_source" in stdout

    def test_the_documented_evidence_source_shapes_are_accepted(self) -> None:
        """Positive control: the refusals above are not refusing everything."""
        for ok in ("OCC#6695", "a40877eae47d46684458ded78483ee29b644f3ad"):
            code, stdout, parsed = _run_builder(
                "omnibase_infra", "0.38.31", evidence_source=ok
            )
            assert code == 0, stdout
            assert parsed["evidence_source"] == ok


class TestBothPredecessorShapesAreDeduped:
    """This is the SECOND rename of this branch, so both predecessors are in
    flight, not just the most recent one."""

    def test_the_dedupe_knows_all_three_shapes(self) -> None:
        body = _WORKFLOW.read_text(encoding="utf-8")
        assert "legacy_branch_omn18596" in body, (
            "the interim OMN-18596-prefixed shape is still in flight on "
            "omniclaude and omnimemory; a dedupe that forgets it opens a "
            "duplicate beside those PRs"
        )
        open_pr_step = body[body.index("- name: Open pull request") :]
        guard = open_pr_step[: open_pr_step.index("gh pr create")]
        assert guard.count("--head") >= 3, (
            "the already-open check must query the current shape and BOTH "
            "predecessors before creating anything"
        )
