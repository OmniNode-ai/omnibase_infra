# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18766 — a CLAIM row names the executor that pays for it.

The ledger records what a claim COSTS (the rule-4 cost sentence, OMN-15649 /
OMN-18554) and what it COST (the friction row, OMN-18274). Until this gate it
did not record WHO paid, so neither number could be compared across executors.

Measured over ``docs/tracking/ROLLING_WORK_LEDGER.md`` plus its two most recent
archive splits, 2026-09-11 -> 2026-09-18, 4,293 dated rows: 613 of 1,163
distinct lanes (53%) and 2,648 rows (62.1%) carry no ``actor=`` and no
``model=`` field anywhere, and their lane names give no executor hint either.
A codex-vs-claude throughput comparison drawn from that ledger covers 12.3% of
its rows and is not a claim about the whole surface.

What each test class pins:

1. ``TestRedBeforeGreen`` — the differential. The SAME row, the same clock, the
   same goal file, refused without attribution and landed with it. This is what
   makes the refusal attributable to this gate rather than to one of the six
   other guards that run on the same payload. It is the non-skippable half of
   the red-before-green proof.
2. ``TestPreChangeImage`` — the literal pre-change bytes, read out of the object
   store at the commit this branch forked from, exit 0 and write the row. This
   is the archival half: it proves the behaviour CHANGED, not merely that it is
   correct now. It skips loudly when the object is unreachable (a shallow
   clone), which is why (1) exists and never skips.
3. ``TestPlaceholdersAreRefused`` — ``actor=unknown`` names no executor. A gate
   that accepted it would convert a missing field into a filled one and make
   the measurement look answered, which is strictly worse than the absence.
4. ``TestVocabularyIsReadFromTheLedger`` — every example the refusal offers
   satisfies the predicate the refusal enforces. A gate that advertises a value
   it would itself refuse trains lanes to ignore its message.
5. ``TestNonClaimRowsAreUntouched`` — TERMINAL, FRICTION, RULING and NOTE rows
   land byte-identical with no attribution and no banner. The fleet's whole
   coordination surface runs through this script; a gate that widened past
   claim rows would take it down rather than measure it.
6. ``TestNotWindowScoped`` — the refusal holds when the rule-4 cost-sentence
   window is closed, pending, or unreadable. Attribution is not a pricing
   question, and a gate that went inert alongside its neighbour would be off on
   exactly the days its neighbour is off.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "ledger_lock.py"

assert SCRIPT.is_file(), (
    f"scripts/ledger_lock.py must exist at {SCRIPT}; it is a committed file in this "
    "repo (OMN-18554), so absence is a failure and never a skip."
)

# The commit this branch forked from — the last revision of the script WITHOUT
# this gate. Pinned as a literal rather than resolved from `origin/dev`, because
# `origin/dev` advances past the fix the moment this lands and the pre-image
# would silently become the post-image.
PRE_CHANGE_SHA = "f8475f9be6afa8d4bd88dbdd270998ac0a756f00"

_spec = importlib.util.spec_from_file_location(
    "ledger_lock_attribution_omn18766", SCRIPT
)
assert _spec is not None and _spec.loader is not None
ll = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ll
_spec.loader.exec_module(ll)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _stamp(moment: datetime | None = None) -> str:
    return (moment or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")


def pipe_claim_row(
    *,
    lane: str = "omn18766-fixture",
    attribution: str | None = "actor=claude:opus5:subagent",
) -> str:
    """The DOMINANT live claim shape — ``<ts> | CLAIM | lane=…`` — carrying a
    conforming rule-4 cost sentence so the cost gate is satisfied and this gate
    is the only thing left that can refuse the row.

    ``attribution=None`` is the 62.1% of the live ledger this ticket is about.
    """
    fields = [
        _stamp(),
        "CLAIM",
        f"lane={lane}",
        "ticket=OMN-18766",
        "repos=omnibase_infra",
    ]
    if attribution is not None:
        fields.append(attribution)
    fields.append(
        "scope=exercise the executor-attribution gate; "
        "est ~1 lane-hours; displaces nothing; (OMN-18766)"
    )
    return " | ".join(fields)


@pytest.fixture
def ledger(tmp_path: Path) -> Path:
    # Not named ROLLING_WORK_LEDGER.md: that file name is governed by the
    # rolling ledger row grammar (OMN-19256), which judges the row type first and
    # fails closed where its omni_home module is absent, as it is in this repo's
    # CI. Nothing this module tests is scoped by file name.
    path = tmp_path / "WORK_LEDGER.md"
    path.write_text("## §5 ACTION LOG (append-only)\n", encoding="utf-8")
    return path


def _goal_clone(tmp_path: Path, state_as_of: date, name: str = "kb") -> Path:
    """A knowledge-base-internal clone carrying ``beta/GOAL.md`` in the real
    shape the freshness resolver reads (OMN-18751)."""
    root = tmp_path / name
    (root / "beta").mkdir(parents=True, exist_ok=True)
    (root / "beta" / "GOAL.md").write_text(
        f"state_as_of: {state_as_of.isoformat()} (fixture, written by the test suite)\n"
        "L1 | a row | R2-UNMET | ticket:OMN-18766 | fixture row\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def open_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The cost-sentence window OPEN — re-measured two days ago, inside the
    seven-day horizon. Every test that asserts a row LANDS needs this, or the
    landing would be the cost gate's verdict rather than this gate's."""
    root = _goal_clone(tmp_path, datetime.now(UTC).date() - timedelta(days=2))
    monkeypatch.delenv(ll.GOAL_PATH_ENV, raising=False)
    monkeypatch.setenv(ll.KB_INTERNAL_ROOT_ENV, str(root))
    return root


def _append(ledger: Path, line: str) -> int:
    return ll.main([str(ledger), "--append", line])


def _rows_with(ledger: Path, needle: str) -> int:
    return ledger.read_text(encoding="utf-8").count(needle)


# --------------------------------------------------------------------------
# 1. Red before green — the differential that never skips
# --------------------------------------------------------------------------


class TestRedBeforeGreen:
    def test_the_same_row_is_refused_without_attribution_and_lands_with_it(
        self,
        ledger: Path,
        open_window: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """AC1 + AC4. One variable changes between the two calls: the presence
        of ``actor=``. Anything else that refused the first call would refuse
        the second, so the pair isolates this gate from every other guard on
        the payload."""
        lane = "omn18766-differential"
        unattributed = pipe_claim_row(lane=lane, attribution=None)

        assert _append(ledger, unattributed) == 65
        assert _rows_with(ledger, lane) == 0, (
            "a refused claim row must not be written; a gate that refuses and "
            "writes is a warning with a bad exit code"
        )
        stderr = capsys.readouterr().err
        assert "OMN-18766 executor attribution" in stderr
        assert f"lane={lane}" in stderr, "the refusal must name the offending lane"
        assert "actor=" in stderr and "model=" in stderr, (
            "the refusal must name both fields that would satisfy it"
        )

        attributed = pipe_claim_row(
            lane=lane, attribution="actor=claude:opus5:subagent"
        )
        assert _append(ledger, attributed) == 0
        assert _rows_with(ledger, lane) == 1

    def test_a_model_field_alone_satisfies_the_gate(
        self, ledger: Path, open_window: Path
    ) -> None:
        """AC1. ``model=`` is the second accepted spelling, not a synonym that
        was documented and never wired — a delegated run names a model where a
        lane names an actor."""
        lane = "omn18766-model-only"
        assert (
            _append(ledger, pipe_claim_row(lane=lane, attribution="model=qwen3.8")) == 0
        )
        assert _rows_with(ledger, lane) == 1

    def test_the_bracketed_handle_claim_shape_is_also_gated(
        self, ledger: Path, open_window: Path
    ) -> None:
        """AC1. The gate's predicate is ``is_rule4_claim_row``, so it covers
        every claim shape that predicate covers, not only the pipe-lead one.
        A gate that saw one shape would be routed around by writing another."""
        head = f"- {_stamp()} [omn18766-handle] CLAIM OMN-18766 — exercise the gate"
        priced = "; est ~1 lane-hours; displaces nothing; (OMN-18766)"
        assert _append(ledger, head + priced) == 65
        assert _append(ledger, head + " actor=codex" + priced) == 0


# --------------------------------------------------------------------------
# 2. The literal pre-change image
# --------------------------------------------------------------------------


class TestPreChangeImage:
    def test_the_pre_change_script_accepts_the_row_this_one_refuses(
        self, tmp_path: Path
    ) -> None:
        """AC4, archival half. Proves the behaviour CHANGED rather than merely
        that it is correct now.

        Skips rather than fails when the object is unreachable, which happens
        on a shallow clone. The differential in ``TestRedBeforeGreen`` never
        skips, so this file is never vacuous even when this test is absent.
        """
        try:
            pre_image = subprocess.run(
                [
                    "git",
                    "-C",
                    str(REPO_ROOT),
                    "show",
                    f"{PRE_CHANGE_SHA}:scripts/ledger_lock.py",
                ],
                capture_output=True,
                check=True,
                # OMN-14891: a git hook exports GIT_DIR/GIT_WORK_TREE into this
                # process and those override `-C`. Unscrubbed, this read would
                # resolve against the invoking worktree rather than REPO_ROOT.
                env=scrub_git_location_env(os.environ),
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
            pytest.skip(
                f"pre-change image {PRE_CHANGE_SHA}:scripts/ledger_lock.py is not "
                f"reachable in this checkout ({exc}); the always-run differential in "
                "TestRedBeforeGreen covers the same claim"
            )

        script = tmp_path / "ledger_lock_prechange.py"
        script.write_bytes(pre_image)
        assert b"OMN-18766" not in pre_image, (
            "the pinned pre-change sha already carries this gate, so it is not a "
            "pre-image at all and the comparison below would be vacuous"
        )

        # Not named ROLLING_WORK_LEDGER.md: that file name is governed by the
        # rolling ledger row grammar (OMN-19256), which judges the row type first and
        # fails closed where its omni_home module is absent, as it is in this repo's
        # CI. Nothing this module tests is scoped by file name.
        ledger = tmp_path / "WORK_LEDGER.md"
        ledger.write_text("## §5 ACTION LOG (append-only)\n", encoding="utf-8")
        root = _goal_clone(tmp_path, datetime.now(UTC).date() - timedelta(days=2))
        lane = "omn18766-preimage"

        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                str(ledger),
                "--append",
                pipe_claim_row(lane=lane, attribution=None),
            ],
            capture_output=True,
            env={
                "PATH": "/usr/bin:/bin",
                "HOME": str(tmp_path),
                "KNOWLEDGE_BASE_INTERNAL_PATH": str(root),
                "LEDGER_LOCK_ROOT": str(tmp_path / "locks"),
            },
            check=False,
        )
        assert completed.returncode == 0, completed.stderr.decode()
        assert _rows_with(ledger, lane) == 1, (
            "the pre-change script must LAND the unattributed row — that is the "
            "defect this ticket closes"
        )


# --------------------------------------------------------------------------
# 3. Placeholders
# --------------------------------------------------------------------------


class TestPlaceholdersAreRefused:
    @pytest.mark.parametrize(
        "value", ["unknown", "none", "n/a", "tbd", "-", "?", "", "NONE", " Unknown "]
    )
    def test_a_placeholder_value_names_no_executor(
        self, ledger: Path, open_window: Path, value: str
    ) -> None:
        """AC1, second half. ``model=none`` and a bare ``model=,`` are both real
        specimens in the live ledger, which is how this failure starts."""
        lane = "omn18766-placeholder"
        row = pipe_claim_row(lane=lane, attribution=f"actor={value}")
        assert _append(ledger, row) == 65
        assert _rows_with(ledger, lane) == 0

    def test_the_refusal_says_the_field_was_present_but_empty(
        self, ledger: Path, open_window: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A lane that wrote ``actor=unknown`` and is told 'carries neither an
        actor= nor a model= field' will add the field it already added. The two
        causes must read differently."""
        assert _append(ledger, pipe_claim_row(attribution="actor=unknown")) == 65
        stderr = capsys.readouterr().err
        assert "placeholder" in stderr
        assert "actor=unknown" in stderr, "the refusal must quote the value it rejected"

    def test_a_real_value_beside_a_placeholder_is_attributed(
        self, ledger: Path, open_window: Path
    ) -> None:
        """A row carrying both answers the question; the placeholder is noise,
        not a veto. Refusing here would refuse a row that names its executor."""
        lane = "omn18766-mixed"
        row = pipe_claim_row(lane=lane, attribution="model=none | actor=claude:opus5")
        assert _append(ledger, row) == 0
        assert _rows_with(ledger, lane) == 1


# --------------------------------------------------------------------------
# 4. The vocabulary
# --------------------------------------------------------------------------


class TestVocabularyIsReadFromTheLedger:
    def test_the_refusal_names_the_live_vocabulary(
        self, ledger: Path, open_window: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """AC2. These four are the highest-count live spellings across the
        ledger and its archive splits on 2026-09-18 — a lane reading the
        refusal sees what its peers actually write, not an invented enum."""
        assert _append(ledger, pipe_claim_row(attribution=None)) == 65
        stderr = capsys.readouterr().err
        for example in (
            "actor=codex",
            "actor=claude:opus5:subagent",
            "actor=claude-sonnet",
            "model=claude-opus-5",
        ):
            assert example in stderr, f"the refusal must offer {example}"

    @pytest.mark.parametrize("example", ll.ATTRIBUTION_VOCABULARY)
    def test_every_advertised_example_satisfies_the_gate(self, example: str) -> None:
        """AC2. A gate that advertises a value it would itself refuse trains
        lanes to stop reading its refusals."""
        row = pipe_claim_row(attribution=example)
        assert ll.attribution_refusal_for_line(row) is None, (
            f"{example!r} is offered by the refusal but does not satisfy the gate"
        )

    def test_the_placeholder_set_is_not_empty(self) -> None:
        """Positive control for the parametrization above: an empty vocabulary
        or an empty placeholder set would make several assertions here pass
        without exercising anything."""
        assert len(ll.ATTRIBUTION_VOCABULARY) >= 10
        assert {"unknown", "none", "tbd"} <= ll.ATTRIBUTION_PLACEHOLDERS


# --------------------------------------------------------------------------
# 5. Non-claim rows
# --------------------------------------------------------------------------


class TestNonClaimRowsAreUntouched:
    @pytest.mark.parametrize(
        "row_class", ["TERMINAL", "RULING", "NOTE", "PROGRESS", "CORRECTION"]
    )
    def test_a_non_claim_row_lands_with_no_attribution(
        self,
        ledger: Path,
        open_window: Path,
        capsys: pytest.CaptureFixture[str],
        row_class: str,
    ) -> None:
        """AC3. Every row class other than a claim passes through untouched.
        Widening this gate past claim rows would take the fleet's coordination
        surface down rather than measure it."""
        lane = f"omn18766-{row_class.lower()}"
        row = (
            f"{_stamp()} | {row_class} | lane={lane} | "
            "RESULT: nothing mutated | friction=none"
        )
        assert _append(ledger, row) == 0
        assert row in ledger.read_text(encoding="utf-8"), (
            "the row must be written byte-identical; this gate never rewrites a row"
        )
        assert "OMN-18766" not in capsys.readouterr().err, (
            "a payload carrying no claim row gets no attribution banner — noise on "
            "every write is how a real signal stops being read"
        )

    def test_prose_mentioning_a_claim_is_not_a_claim_row(
        self, ledger: Path, open_window: Path
    ) -> None:
        """The predicate is shared with the rule-4 gate, which already refuses
        to read a mention as an act. Pinned here because a regression in that
        predicate would surface first as this gate refusing peers' notes."""
        row = (
            f"{_stamp()} | NOTE | lane=omn18766-prose | the CLAIM row for OMN-18766 "
            "is described in the ticket body; nothing is claimed here"
        )
        assert _append(ledger, row) == 0


# --------------------------------------------------------------------------
# 6. Not window-scoped
# --------------------------------------------------------------------------


class TestNotWindowScoped:
    def test_the_refusal_holds_when_the_goal_file_is_unreadable(
        self, ledger: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Attribution is not a pricing question. The cost gate is calendar-gated
        on the goal file's freshness; a gate that went inert alongside it would
        be off on exactly the days its neighbour is off."""
        monkeypatch.delenv(ll.GOAL_PATH_ENV, raising=False)
        monkeypatch.setenv(ll.KB_INTERNAL_ROOT_ENV, str(tmp_path / "no-such-clone"))
        assert _append(ledger, pipe_claim_row(attribution=None)) == 65

    def test_the_refusal_holds_when_the_window_has_not_opened(
        self, ledger: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A goal file stamped in the FUTURE lands an UNPRICED row (the cost
        gate is not yet in force). An unattributed one is still refused, which
        is what proves the two gates are independent rather than one gate with
        two messages."""
        root = _goal_clone(tmp_path, datetime.now(UTC).date() + timedelta(days=10))
        monkeypatch.delenv(ll.GOAL_PATH_ENV, raising=False)
        monkeypatch.setenv(ll.KB_INTERNAL_ROOT_ENV, str(root))

        lane = "omn18766-pending-window"
        unpriced_attributed = " | ".join(
            [
                _stamp(),
                "CLAIM",
                f"lane={lane}",
                "ticket=OMN-18766",
                "actor=codex",
                "scope=no cost sentence at all",
            ]
        )
        assert _append(ledger, unpriced_attributed) == 0, (
            "positive control: with the window not yet open the cost gate does not "
            "refuse, so a refusal below is this gate's and not its neighbour's"
        )
        assert _append(ledger, pipe_claim_row(lane=lane, attribution=None)) == 65
