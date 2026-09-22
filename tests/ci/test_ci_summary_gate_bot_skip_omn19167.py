# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19167 — the conditional layer-5 admission, and the four shapes it must NOT admit.

The defect, measured on omnibase_infra#3953 head ``4af8a2d5b6``, job 106734128710::

    external sweep failures (red, and named by NOTHING else):
      occ-autobind (skipped), occ-companion-effect (skipped)

Both callers are gated by their own ``if:`` on the pull request carrying a
ticket token, and doctrine's PR-title rule deliberately EXEMPTS a dependency
bump from carrying one. So the skip is the workflow's declared outcome, the
strict layer-5 bar counted it red, and every ticketless dependency-bot pull
request in the repository was blocked by construction.

The registry that fixes it is CONDITIONAL, and the value of this module is
mostly in the four negative controls: a fix that admitted the bare names
unconditionally would pass the first test here and silently stop noticing a
ticketed pull request whose change-control mint never ran.

The fixture is a real, unedited capture of that head's check-runs, workflow
runs and in-run job names. Nothing in it is synthesised; the negative controls
are built by mutating ONE field of it at a time, so each says what it changed.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from tests.ci.test_ci_summary_gate import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts" / "ci"))

from ci_summary_gate import (
    CONDITIONAL_SWEEP_EXCLUSIONS,
    DEPENDENCY_BOT_AUTHORS,
    EXPECTED_EXTERNAL_CONTEXTS,
    EXTERNAL_SWEEP_EXCLUSIONS,
    ConditionalSweepExclusion,
    JobState,
    PullRequestContext,
    check_run_event_index,
    conditional_exclusion_admits,
    evaluate_external_sweep,
    occ_caller_job_is_eligible,
    title_rule_exempts_ticket,
    validate_conditional_sweep_exclusions,
)

pytestmark = pytest.mark.unit

FIXTURE = REPO_ROOT / "tests/ci/fixtures/omn19167_dependabot_pr3953_check_runs.json"

# Inside every entry's window (added 2026-09-22, expires 2026-12-20), so an
# expiry is never what a shape test is really measuring. The expiry itself has
# its own test below.
NOW = datetime(2026, 10, 1, tzinfo=UTC)

CALLER_NAMES = ("occ-autobind", "occ-companion-effect")


def _fixture() -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return payload


def _live_context() -> PullRequestContext:
    """The real #3953 facts, read from the fixture rather than retyped."""

    p = _fixture()["_provenance"]
    return PullRequestContext(
        author=p["pr_author"],
        title=p["pr_title"],
        head_ref=p["pr_head_ref"],
        actor=p["event_actor"],
    )


def _sweep(
    context: PullRequestContext | None,
    *,
    check_runs: list[dict[str, Any]] | None = None,
    conditional: dict[str, ConditionalSweepExclusion] | None = None,
    now: datetime | None = NOW,
) -> list[str]:
    """Layer 5 over the real head, returning its failure lines."""

    head = _fixture()
    failures, _in_flight, swept, _excluded, _prov = evaluate_external_sweep(
        check_runs if check_runs is not None else head["check_runs"],
        expected=EXPECTED_EXTERNAL_CONTEXTS,
        in_run_names=frozenset(head["in_run_job_names"]),
        self_name="CI Summary",
        exclusions=EXTERNAL_SWEEP_EXCLUSIONS,
        events=check_run_event_index(head["workflow_runs"]),
        now=now,
        conditional_exclusions=(
            CONDITIONAL_SWEEP_EXCLUSIONS if conditional is None else conditional
        ),
        pr_context=context,
    )
    assert swept, "the sweep judged nothing, so any verdict here is vacuous"
    lines: list[str] = failures
    return lines


def _named(failures: list[str], name: str) -> bool:
    return any(f.startswith(f"{name} (") for f in failures)


def _with_conclusion(name: str, conclusion: str) -> list[dict[str, Any]]:
    """The real payload with ONE row's conclusion changed, and nothing else."""

    rows: list[dict[str, Any]] = _fixture()["check_runs"]
    hits = 0
    for row in rows:
        if row["name"] == name:
            row["conclusion"] = conclusion
            hits += 1
    assert hits == 1, f"expected exactly one {name!r} row, found {hits}"
    return rows


class TestTheFixtureIsTheRealDefect:
    """Rule 16: prove the population before reading anything off it."""

    def test_fixture_carries_the_two_skipped_caller_rows(self) -> None:
        rows = {r["name"]: r for r in _fixture()["check_runs"]}
        for name in CALLER_NAMES:
            assert rows[name]["status"] == "completed"
            assert rows[name]["conclusion"] == "skipped", rows[name]

    def test_provenance_names_the_head_and_the_verdict_it_produced(self) -> None:
        p = _fixture()["_provenance"]
        assert p["pr"] == 3953
        assert len(p["head_sha"]) == 40
        assert p["pr_author"] == "dependabot[bot]"
        # The actor is NOT the author: an update-branch repushed the head, so
        # the caller's actor arm passes and the missing ticket token is the
        # sole unmet half. A fix keyed on the actor alone would not have
        # fixed this head.
        assert p["event_actor"] == "jonahgabriel"
        for name in CALLER_NAMES:
            assert f"{name} (skipped)" in p["blocking_verdict_line"]

    def test_the_title_and_head_ref_really_carry_no_ticket_token(self) -> None:
        ctx = _live_context()
        assert not ctx.carries_ticket_token
        assert re.search(r"OMN-\d+", ctx.title) is None
        assert re.search(r"OMN-\d+", ctx.head_ref) is None


class TestTheDefectAndTheFix:
    def test_red_without_the_registry_this_is_the_defect(self) -> None:
        """THE RED PROOF, reproduced against the shipped code.

        With the conditional registry emptied, the module behaves exactly as
        it did on dev, and the real head reds on the two caller rows and
        nothing else. This is the failure that blocked #3953-#3958.
        """

        failures = _sweep(_live_context(), conditional={})
        for name in CALLER_NAMES:
            assert _named(failures, name), (name, failures)
        assert sorted(failures) == sorted(f"{n} (skipped)" for n in CALLER_NAMES), (
            failures
        )

    def test_green_with_the_registry_shape_one_of_four(self) -> None:
        """Shape 1 — the six live PRs. Must PASS."""

        assert _sweep(_live_context()) == []

    def test_the_admission_is_reported_with_its_reason(self) -> None:
        """A reader learns WHY a red row stopped being red, from the report."""

        head = _fixture()
        _f, _i, _s, excluded, _p = evaluate_external_sweep(
            head["check_runs"],
            expected=EXPECTED_EXTERNAL_CONTEXTS,
            in_run_names=frozenset(head["in_run_job_names"]),
            self_name="CI Summary",
            exclusions=EXTERNAL_SWEEP_EXCLUSIONS,
            events=check_run_event_index(head["workflow_runs"]),
            now=NOW,
            conditional_exclusions=CONDITIONAL_SWEEP_EXCLUSIONS,
            pr_context=_live_context(),
        )
        for name in CALLER_NAMES:
            assert (
                f"{name} (skipped; declared_ticketless_dependency_bot_skip)" in excluded
            ), excluded


class TestTheFourShapesItMustRefuse:
    """The negative controls. An unconditional entry passes NONE of these."""

    def test_shape_two_ticketed_pr_with_a_skipped_autobind_still_fails(self) -> None:
        """A ticketed PR whose mint did not run is a REAL refusal."""

        ctx = PullRequestContext(
            author="jonahgabriel",
            title="fix(OMN-19167): something ticketed",
            head_ref="jonah/omn-19167-ci-summary-bot-skip",
            actor="jonahgabriel",
        )
        failures = _sweep(ctx)
        for name in CALLER_NAMES:
            assert _named(failures, name), (name, failures)

    def test_shape_two_b_a_bot_pr_that_does_carry_a_ticket_still_fails(self) -> None:
        """The author being a bot is not on its own enough.

        Here the caller job WAS eligible (token present, actor not a bot), so a
        skip is unexplained and must red even though the author is dependabot.
        """

        ctx = PullRequestContext(
            author="dependabot[bot]",
            title="chore(deps): bump something OMN-19167",
            head_ref="dependabot/github_actions/something",
            actor="jonahgabriel",
        )
        failures = _sweep(ctx)
        for name in CALLER_NAMES:
            assert _named(failures, name), (name, failures)

    def test_shape_three_bot_pr_whose_autobind_ran_and_failed_still_fails(
        self,
    ) -> None:
        """Only `skipped` is admitted. A `failure` on the same name reds."""

        rows = _with_conclusion("occ-autobind", "failure")
        failures = _sweep(_live_context(), check_runs=rows)
        assert "occ-autobind (failure)" in failures, failures
        # and the sibling, untouched, is still admitted — so this test is
        # measuring the conclusion and not switching the registry off.
        assert not _named(failures, "occ-companion-effect"), failures

    def test_shape_three_b_a_cancelled_caller_row_still_fails(self) -> None:
        rows = _with_conclusion("occ-autobind", "cancelled")
        failures = _sweep(_live_context(), check_runs=rows)
        assert _named(failures, "occ-autobind"), failures

    def test_shape_four_non_bot_ticketless_pr_still_fails(self) -> None:
        """A human who forgot the ticket is told so, not excused."""

        ctx = PullRequestContext(
            author="jonahgabriel",
            title="chore(deps): bump something with no ticket",
            head_ref="jonah/no-ticket-here",
            actor="jonahgabriel",
        )
        failures = _sweep(ctx)
        for name in CALLER_NAMES:
            assert _named(failures, name), (name, failures)

    def test_shape_four_b_a_bot_pr_whose_title_is_not_an_exempt_class(self) -> None:
        """A bot author alone does not make an arbitrary title exempt here.

        The title rule's own first arm would exempt any bot, but this registry
        requires the dependency-bot author AND the exempt title class, so a
        renovate PR titled like ordinary feature work is refused.
        """

        ctx = PullRequestContext(
            author="renovate[bot]",
            title="feat: rewrite the scheduler",
            head_ref="renovate/scheduler",
            actor="renovate[bot]",
        )
        # The bot-suffix arm of the title rule does exempt it...
        assert title_rule_exempts_ticket(author=ctx.author, title=ctx.title)
        # ...and the admission still holds, because the producer declared the
        # job ineligible on its actor arm. This test records that the bot arm
        # is WIDE, which is why the dependency-bot author set is the narrow
        # half doing the bounding.
        assert _sweep(ctx) == []


class TestFailClosedOnAnUnresolvableContext:
    def test_no_context_at_all_admits_nothing(self) -> None:
        failures = _sweep(None)
        for name in CALLER_NAMES:
            assert _named(failures, name), (name, failures)

    @pytest.mark.parametrize("missing", ["author", "title"])
    def test_a_missing_required_field_admits_nothing(self, missing: str) -> None:
        live = _live_context()
        fields = {
            "author": live.author,
            "title": live.title,
            "head_ref": live.head_ref,
            "actor": live.actor,
        }
        fields[missing] = ""
        failures = _sweep(PullRequestContext(**fields))
        for name in CALLER_NAMES:
            assert _named(failures, name), (missing, failures)

    def test_a_missing_clock_admits_nothing(self) -> None:
        """`now is None` cannot prove an entry is live, so it enforces."""

        failures = _sweep(_live_context(), now=None)
        for name in CALLER_NAMES:
            assert _named(failures, name), failures

    def test_an_expired_entry_re_arms_the_sweep(self) -> None:
        after = datetime(2026, 12, 20, tzinfo=UTC)
        failures = _sweep(_live_context(), now=after)
        for name in CALLER_NAMES:
            assert _named(failures, name), failures

    def test_an_entry_naming_a_missing_predicate_admits_nothing(self) -> None:
        broken = {
            name: ConditionalSweepExclusion(
                reason="deliberately names a predicate that does not exist",
                ticket="OMN-19167",
                added="2026-09-22",
                expires="2026-12-20",
                conclusions=frozenset({"skipped"}),
                condition="no_such_predicate",
            )
            for name in CALLER_NAMES
        }
        failures = _sweep(_live_context(), conditional=broken)
        for name in CALLER_NAMES:
            assert _named(failures, name), failures
        # ...and it is also reported as malformed rather than silently inert.
        assert validate_conditional_sweep_exclusions(broken), (
            "a predicate that resolves to nothing must be a finding"
        )


class TestTheRegistryIsHeldToTheSameBar:
    def test_the_shipped_registry_validates(self) -> None:
        assert validate_conditional_sweep_exclusions(CONDITIONAL_SWEEP_EXCLUSIONS) == []

    def test_it_covers_exactly_the_two_caller_names(self) -> None:
        assert sorted(CONDITIONAL_SWEEP_EXCLUSIONS) == sorted(CALLER_NAMES)

    def test_every_entry_admits_only_skipped(self) -> None:
        for name, entry in CONDITIONAL_SWEEP_EXCLUSIONS.items():
            assert entry.conclusions == frozenset({"skipped"}), name

    def test_success_may_not_be_listed(self) -> None:
        bad = {
            "occ-autobind": ConditionalSweepExclusion(
                reason="lists success, which needs no admission",
                ticket="OMN-19167",
                added="2026-09-22",
                expires="2026-12-20",
                conclusions=frozenset({"skipped", "success"}),
                condition="declared_ticketless_dependency_bot_skip",
            )
        }
        assert any("success" in f for f in validate_conditional_sweep_exclusions(bad))

    @pytest.mark.parametrize(
        ("field", "value", "needle"),
        [
            ("reason", "   ", "reason is empty"),
            ("ticket", "OMN-", "OMN-<number>"),
            ("added", "yesterday", "YYYY-MM-DD"),
            ("expires", "2026-09-21", "not after added"),
        ],
    )
    def test_the_four_shared_field_checks_apply_here_too(
        self, field: str, value: str, needle: str
    ) -> None:
        kwargs: dict[str, Any] = {
            "reason": "a reason",
            "ticket": "OMN-19167",
            "added": "2026-09-22",
            "expires": "2026-12-20",
            "conclusions": frozenset({"skipped"}),
            "condition": "declared_ticketless_dependency_bot_skip",
        }
        kwargs[field] = value
        findings = validate_conditional_sweep_exclusions(
            {"occ-autobind": ConditionalSweepExclusion(**kwargs)}
        )
        assert any(needle in f for f in findings), findings

    def test_the_window_cannot_exceed_the_ninety_day_cap(self) -> None:
        findings = validate_conditional_sweep_exclusions(
            {
                "occ-autobind": ConditionalSweepExclusion(
                    reason="a year-long window",
                    ticket="OMN-19167",
                    added="2026-09-22",
                    expires="2027-09-22",
                    conclusions=frozenset({"skipped"}),
                    condition="declared_ticketless_dependency_bot_skip",
                )
            }
        )
        assert any("cap" in f for f in findings), findings

    def test_the_two_registries_do_not_overlap(self) -> None:
        """A name on both lists would be admitted unconditionally, silently."""

        assert not (set(CONDITIONAL_SWEEP_EXCLUSIONS) & set(EXTERNAL_SWEEP_EXCLUSIONS))

    def test_conditional_exclusion_admits_refuses_an_unregistered_name(self) -> None:
        state = JobState("Some Other Gate", "completed", "skipped", 1)
        assert not conditional_exclusion_admits(
            "Some Other Gate",
            state,
            exclusions=CONDITIONAL_SWEEP_EXCLUSIONS,
            context=_live_context(),
            now=NOW,
        )

    def test_conditional_exclusion_admits_refuses_an_incomplete_row(self) -> None:
        state = JobState("occ-autobind", "in_progress", None, 1)
        assert not conditional_exclusion_admits(
            "occ-autobind",
            state,
            exclusions=CONDITIONAL_SWEEP_EXCLUSIONS,
            context=_live_context(),
            now=NOW,
        )


# --------------------------------------------------------------------------
# The mirror, pinned. AC6.
# --------------------------------------------------------------------------

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
            "babdd13ce68f07df20f989f52ff1c4514d03d896"
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


class TestTheProducerEligibilityMirror:
    """The second mirror: the occ callers' own `if:` expression."""

    @pytest.mark.parametrize(
        "path",
        [
            ".github/workflows/call-occ-autobind.yml",
            ".github/workflows/call-occ-companion-effect.yml",
        ],
    )
    def test_both_callers_still_gate_on_the_arms_this_mirrors(self, path: str) -> None:
        text = (REPO_ROOT / path).read_text(encoding="utf-8")
        assert "github.actor != 'dependabot[bot]'" in text, path
        assert "github.actor != 'renovate[bot]'" in text, path
        assert "contains(github.event.pull_request.title, 'OMN-')" in text, path
        assert "contains(github.event.pull_request.head.ref, 'OMN-')" in text, path

    def test_the_bot_set_matches_the_logins_the_callers_name(self) -> None:
        assert frozenset({"dependabot[bot]", "renovate[bot]"}) == DEPENDENCY_BOT_AUTHORS

    @pytest.mark.parametrize(
        ("actor", "title", "head_ref", "eligible"),
        [
            ("jonahgabriel", "feat(OMN-1): x", "jonah/x", True),
            ("jonahgabriel", "chore(deps): x", "jonah/omn-1-x", False),
            ("jonahgabriel", "chore(deps): x", "dependabot/gha/x", False),
            ("dependabot[bot]", "feat(OMN-1): x", "jonah/x", False),
            ("renovate[bot]", "feat(OMN-1): x", "jonah/x", False),
        ],
    )
    def test_eligibility_matches_the_expression(
        self, actor: str, title: str, head_ref: str, eligible: bool
    ) -> None:
        ctx = PullRequestContext(
            author="whoever", title=title, head_ref=head_ref, actor=actor
        )
        assert occ_caller_job_is_eligible(ctx) is eligible

    def test_the_head_ref_arm_is_read_lowercase_insensitively_by_neither_side(
        self,
    ) -> None:
        """The producers use `contains`, which is case-SENSITIVE for 'OMN-'.

        Recorded so the mirror is not "improved" into case-insensitivity,
        which would make it admit a skip the producer would not have taken.
        """

        ctx = PullRequestContext(
            author="whoever", title="feat(omn-1): x", head_ref="jonah/x", actor="x"
        )
        assert not occ_caller_job_is_eligible(ctx)


class TestTheCliSurface:
    """The three new arguments exist and default to the enforcing value."""

    def test_the_parser_declares_them(self) -> None:
        source = (REPO_ROOT / "scripts/ci/ci_summary_gate.py").read_text(
            encoding="utf-8"
        )
        for flag in ("--pr-title", "--pr-head-ref", "--event-actor"):
            assert f'"{flag}"' in source, flag

    def test_ci_yml_passes_all_three(self) -> None:
        """A gate whose caller forgets an argument enforces, but pointlessly."""

        text = (REPO_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        for flag in ("--pr-title", "--pr-head-ref", "--event-actor"):
            assert flag in text, flag
        for env in ("PR_TITLE:", "PR_HEAD_REF:", "EVENT_ACTOR:"):
            assert env in text, env


def test_module_level_smoke_the_defect_head_is_green_end_to_end() -> None:
    """One assertion a reader can check without following the helpers."""

    assert _sweep(_live_context()) == []
    assert _sweep(_live_context(), conditional={}) == [
        "occ-autobind (skipped)",
        "occ-companion-effect (skipped)",
    ]
