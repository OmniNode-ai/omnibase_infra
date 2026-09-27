# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A named prose output vetoes a compilation-graded class (OMN-18831, the residual).

THE DEFECT. OMN-18831's first fix gated the ambiguous VERBS ("write a",
"assertion") on a nearby code artifact. The ticket's own 2026-09-20 comment
recorded what that leaves open: an UNAMBIGUOUS test phrase present because the
prompt DESCRIBES code work. A PR-body request listing which checks were run
said "unit tests", and ``test`` claimed it at priority 40 (correlation
``03953fba-18a9-4c85-a77b-bcc3bb860a01``); three local rungs returned prose and
were refused on the deterministic floor.

It is still live after every earlier fix. Read on 2026-09-23 against the
dispatch venv (omnibase_infra 0.38.57, omnimarket 0.4.211), run
``1a67961a-664e-4a98-8680-378e8c393921`` still resolves to ``test`` on
"pytest": it asked for "a markdown table ... for a pull request description",
whose first row counted "collectable pytest modules". Its three local rungs
returned the table and were refused at 0.333.

THE FIX. The contract's ``selection.vetoed_by`` names the REQUESTED OUTPUT,
which the prompt states outright: a class that declares it does not claim a
prompt naming one of those phrases. It is declared on the three public classes
graded by deterministic acceptance. Qualifying "unit tests" would not help,
because the prompt genuinely mentions unit tests; it is just not asking for
one.

Every row here runs against the digest-pinned production mirror, so it is a
statement about the live contract, not a hand-written one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.cli.task_class_selection import (
    EnumTaskTypeResolution,
    ModelSelectableTaskClass,
    load_selectable_task_classes,
    resolve_task_type,
)

pytestmark = pytest.mark.unit

_MIRROR = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "delegation"
    / "omn18831"
    / "task_class_selection_production_mirror.yaml"
)

_COMPILATION_GRADED = frozenset({"code_generation", "refactor", "test"})

#: Run 1a67961a-664e-4a98-8680-378e8c393921, verbatim from its run.json.
_RECORDED_TABLE_REQUEST = (
    "Reformat these measured numbers into a markdown table of exactly four data "
    "rows plus a header, for a pull request description. Columns: Metric, "
    "Before, After, Delta. Output only the table, no prose. Numbers: collectable "
    "pytest modules summed over the last 20 merged pull requests, before 7014, "
    "after 7054, delta plus 40 which is plus 0.57 percent. Pull requests whose "
    "shard count changed, before 0, after 0, delta 0. Pull requests escalated to "
    "the whole suite, before 0, after 0, delta 0. Worst-case selector wall time "
    "in seconds, before 0.072, after 0.344, delta plus 0.272."
)

#: Run 771582cc-f1e0-4a17-84a3-94832ff70070, opening verbatim. A genuine code
#: request from the same corpus; it says "no prose" about the answer's form,
#: which is deliberately not a veto.
_RECORDED_CODE_REQUEST = (
    "Write a POSIX-bash function named parse_reset_epoch. Input: one string "
    "argument, a refusal banner line. Output: print epoch seconds of the reset "
    "moment on stdout and return 0, or print nothing and return 1 when no reset "
    "time can be found. Return only the bash function body, no prose, no "
    "markdown fence."
)

# (prompt, class it resolved to before this change, why)
_DESCRIBED_CODE_WORK_ROWS: tuple[tuple[str, str, str], ...] = (
    (
        _RECORDED_TABLE_REQUEST,
        "test",
        "run 1a67961a verbatim: 'pytest' inside a pull request description table",
    ),
    (
        "Draft a GitHub PR body in markdown. Sections: Summary, Tests. Tests: "
        "ran the unit tests for the handler and ruff; all pass. Output only the "
        "body.",
        "test",
        "the 2026-09-20 comment's shape: 'unit tests' listed inside a PR body",
    ),
    (
        "Draft a concise Linear comment reporting two measured root causes. The "
        "unit tests passed locally. Keep it under 200 words, factual.",
        "test",
        "'unit tests' inside a ticket comment request",
    ),
    (
        "Draft the commit message for this change: it adds a pytest fixture "
        "that resets the broker between cases.",
        "test",
        "'pytest' inside a commit message request",
    ),
    (
        "Summarize in prose why we scaffold every new node from the template, "
        "for the onboarding page.",
        "code_generation",
        "'scaffold' inside an in-prose explanation",
    ),
)

# (prompt, expected class, why) -- the controls the veto must not touch.
_GENUINE_CODE_ROWS: tuple[tuple[str, str, str], ...] = (
    (
        _RECORDED_CODE_REQUEST,
        "code_generation",
        "run 771582cc verbatim: a real code request that says 'no prose'",
    ),
    (
        "Write unit tests for the retry helper in retry.py using pytest.",
        "test",
        "a real test request",
    ),
    (
        "Implement a parser for the lane manifest file.",
        "code_generation",
        "a real code request",
    ),
    (
        "Refactor the dispatcher to remove the duplicated topic lookup.",
        "refactor",
        "a real refactor request",
    ),
)


@pytest.fixture(name="production")
def _production() -> tuple[ModelSelectableTaskClass, ...]:
    return load_selectable_task_classes(_MIRROR)


class TestTheMirrorCarriesTheVeto:
    def test_exactly_the_compilation_graded_classes_declare_it(
        self, production: tuple[ModelSelectableTaskClass, ...]
    ) -> None:
        declaring = {entry.name for entry in production if entry.vetoed_by}
        assert declaring == _COMPILATION_GRADED

    def test_the_loader_reads_the_declared_list(
        self, production: tuple[ModelSelectableTaskClass, ...]
    ) -> None:
        by_name = {entry.name: entry for entry in production}
        assert "pull request description" in by_name["test"].vetoed_by
        assert by_name["test"].vetoed_by == by_name["code_generation"].vetoed_by


class TestDescribedCodeWorkIsNotACodeRequest:
    @pytest.mark.parametrize(
        ("prompt", "before", "why"),
        _DESCRIBED_CODE_WORK_ROWS,
        ids=[row[2] for row in _DESCRIBED_CODE_WORK_ROWS],
    )
    def test_the_phrase_still_matches_but_the_class_does_not_claim_it(
        self,
        production: tuple[ModelSelectableTaskClass, ...],
        prompt: str,
        before: str,
        why: str,
    ) -> None:
        """Both halves: the old claim is still there, and the veto overrides it."""
        by_name = {entry.name: entry for entry in production}
        assert by_name[before].matching_phrase(prompt.lower()) is not None, (
            f"{why}: precondition, the phrase that caused the misroute must "
            "still occur, or this row proves nothing"
        )

        resolution = resolve_task_type(prompt, explicit=None, classes=production)

        assert resolution.task_type not in _COMPILATION_GRADED, (
            f"{why}: {resolution.reason}"
        )
        assert f"vetoed: {before!r} matched" in resolution.reason, resolution.reason

    def test_the_recorded_request_lands_on_the_prose_fallback(
        self, production: tuple[ModelSelectableTaskClass, ...]
    ) -> None:
        resolution = resolve_task_type(
            _RECORDED_TABLE_REQUEST, explicit=None, classes=production
        )
        assert resolution.task_type == "document"
        assert resolution.resolution is EnumTaskTypeResolution.FALLBACK
        assert "'pull request description'" in resolution.reason


class TestGenuineRequestsStillRoute:
    @pytest.mark.parametrize(
        ("prompt", "expected", "why"),
        _GENUINE_CODE_ROWS,
        ids=[row[2] for row in _GENUINE_CODE_ROWS],
    )
    def test_no_veto_fires(
        self,
        production: tuple[ModelSelectableTaskClass, ...],
        prompt: str,
        expected: str,
        why: str,
    ) -> None:
        resolution = resolve_task_type(prompt, explicit=None, classes=production)
        assert resolution.task_type == expected, f"{why}: {resolution.reason}"
        assert "vetoed" not in resolution.reason

    def test_an_explicit_task_type_is_never_vetoed(
        self, production: tuple[ModelSelectableTaskClass, ...]
    ) -> None:
        """The veto governs inference only; a caller who names the class gets it."""
        resolution = resolve_task_type(
            _RECORDED_TABLE_REQUEST, explicit="test", classes=production
        )
        assert resolution.task_type == "test"
        assert resolution.resolution is EnumTaskTypeResolution.EXPLICIT


class TestTheKnownResidual:
    def test_a_code_request_whose_subject_is_a_prose_artifact_is_vetoed_too(
        self, production: tuple[ModelSelectableTaskClass, ...]
    ) -> None:
        """KNOWN RESIDUAL, pinned in its current direction so it is not latent.

        The veto reads the phrase, not its grammatical role, so a code request
        that names a prose artifact as its SUBJECT is refused too and lands on
        the prose fallback. That is the chosen direction of failure: the
        fallback grades on shape-agnostic prose floors, where the defect being
        removed graded prose on Python compilation, and an explicit
        --task-type restores the class. The day the evaluator learns the
        difference, this test goes red and the prompt moves to the controls.
        """
        resolution = resolve_task_type(
            "Write pytest unit tests for the function that parses a PR body.",
            explicit=None,
            classes=production,
        )
        assert resolution.task_type == "document", resolution.reason
        assert "'pr body'" in resolution.reason

    def test_the_same_request_without_the_artifact_is_still_a_test_request(
        self, production: tuple[ModelSelectableTaskClass, ...]
    ) -> None:
        """The control that isolates the row above to the named artifact."""
        resolution = resolve_task_type(
            "Write pytest unit tests for the function that parses a header.",
            explicit=None,
            classes=production,
        )
        assert resolution.task_type == "test", resolution.reason
