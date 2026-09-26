# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18976, the residual gap: a queued merge the lane moved past unwatched.

WHAT WAS LEFT
-------------
The re-emission window answers for merges strictly between the lane revision a
verify run saw when it STARTED and the one it converged to. That lower bound is
a live observation. Measured 2026-09-26: ``84a1e7c3ec`` (#4160) sampled QUEUED;
the next two merges were NON_RUNTIME, so their runs skipped the verify job
entirely; the lane was rebuilt past it by another repository's push; and at
19:11Z the next runtime merge's run re-emitted for the three merges above its
live floor and left ``84a1e7c3ec`` with no receipt and no path back to one.

WHAT THIS PINS
--------------
* The floor is read from the receipts: the newest receipt, of any verdict, at
  or below the oldest receipt-less merge. Receipts ABOVE an orphan never floor
  it -- that is the leapfrog the 19:11Z run performed.
* Only RUNTIME-AFFECTING receipt-less merges are answered for.
* A merge that published no rebuild still answers (``--observe``): only the
  wait is skippable.
* Every unresolved case answers for nothing, and the walk is bounded.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from scripts.ci import check_dev_lane_staleness as guard
from scripts.ci.check_dev_lane_staleness import (
    REEMISSION_LOOKBACK_COMMITS,
    EnumReceiptPresence,
    ModelQueueFacts,
    durable_backfill,
    reemission_window,
)

pytestmark = pytest.mark.unit

# First-parent history, oldest first. A has a receipt from its own run; Q was
# QUEUED and never got one; N1 and N2 are NON_RUNTIME merges whose runs skipped
# the verify job; R is the next runtime merge, which converges.
_A = "a" * 40
_Q = "b" * 40
_N1 = "c" * 40
_N2 = "d" * 40
_R = "e" * 40
_HISTORY_OLDEST_FIRST = (_A, _Q, _N1, _N2, _R)
_RUNTIME = {_A, _Q, _R}


def _below(history: Sequence[str]):  # type: ignore[no-untyped-def]
    newest_first = list(reversed(history))

    def read(revision: str, limit: int) -> tuple[str, ...]:
        index = newest_first.index(revision)
        return tuple(newest_first[index + 1 : index + 1 + limit])

    return read


def _presence(receipted: set[str]):  # type: ignore[no-untyped-def]
    def read(sha: str) -> EnumReceiptPresence:
        return (
            EnumReceiptPresence.PRESENT
            if sha in receipted
            else EnumReceiptPresence.ABSENT
        )

    return read


def _runtime(sha: str) -> bool:
    return sha in _RUNTIME


class TestQueuedThenNonRuntimeThenRuntime:
    """The 2026-09-26 shape, end to end through the pure planner."""

    def test_the_live_window_alone_never_reaches_the_orphan(self) -> None:
        # R's run starts with the lane already at N2 (another repository's push
        # rebuilt it), so the live window is empty -- this is the defect.
        window = reemission_window(
            previous_revision=_N2,
            converged_revision=_R,
            resolve_contained=lambda _p, _c: (_R,),
        )
        assert _Q not in window

    def test_the_durable_floor_backfills_the_orphan(self) -> None:
        plan = durable_backfill(
            converged_revision=_R,
            already_answered=(),
            first_parent_below=_below(_HISTORY_OLDEST_FIRST),
            receipt_presence=_presence({_A, _R}),
            runtime_affecting=_runtime,
        )
        assert plan.backfill == (_Q,), (
            "Q is the one runtime-affecting merge between the last receipt on "
            "record and the lane; N1 and N2 need no receipt of their own"
        )
        assert plan.floor == _A

    def test_receipts_above_the_orphan_do_not_floor_it(self) -> None:
        """The leapfrog: at 19:11Z receipts landed for merges ABOVE 84a1e7c3ec."""
        above = ("f" * 40, "0" * 40)
        history = (*_HISTORY_OLDEST_FIRST, *above)
        plan = durable_backfill(
            converged_revision=above[-1],
            already_answered=(),
            first_parent_below=_below(history),
            receipt_presence=_presence({_A, _N1, _N2, _R, *above}),
            runtime_affecting=_runtime,
        )
        assert plan.backfill == (_Q,)
        assert plan.floor == _A

    def test_a_merge_with_a_receipt_of_any_verdict_is_never_backfilled(self) -> None:
        # A FAIL receipt is a verdict. Its re-emission, if any, is the live
        # window's binding-failure path, never an absent-receipt backfill.
        plan = durable_backfill(
            converged_revision=_R,
            already_answered=(),
            first_parent_below=_below(_HISTORY_OLDEST_FIRST),
            receipt_presence=_presence({_A, _Q}),
            runtime_affecting=_runtime,
        )
        assert plan.backfill == ()

    def test_live_window_members_are_not_answered_for_twice(self) -> None:
        plan = durable_backfill(
            converged_revision=_R,
            already_answered=(_Q,),
            first_parent_below=_below(_HISTORY_OLDEST_FIRST),
            receipt_presence=_presence({_A}),
            runtime_affecting=_runtime,
        )
        assert _Q not in plan.backfill

    def test_the_cli_plan_answers_for_the_orphan(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The same planner, through the read-only CLI the workflow's script ships."""
        monkeypatch.setattr(guard, "read_contained_commits", lambda *_a: (_R,))
        monkeypatch.setattr(
            guard,
            "read_first_parent_below",
            lambda _clone, revision, limit: _below(_HISTORY_OLDEST_FIRST)(
                revision, limit
            ),
        )
        monkeypatch.setattr(
            guard,
            "read_receipt_presence",
            lambda _repo, _lane, sha: _presence({_A, _R})(sha),
        )
        monkeypatch.setattr(
            guard, "load_runtime_affecting_predicate", lambda *_a: _runtime
        )
        code = guard.main(
            [
                "--plan-reemission",
                "--previous-revision",
                _N2,
                "--converged-revision",
                _R,
                "--runtime-path-validator",
                "validator.py",
            ]
        )
        assert code == 0
        plan = json.loads(capsys.readouterr().out)
        assert plan["live_window"] == []
        assert plan["durable_backfill"] == [_Q]
        assert plan["durable_floor"] == _A


class TestCrossRepoConvergenceOnANonRuntimeMerge:
    """The lane was rebuilt past Q by another repository; the next merge here is
    NON_RUNTIME, publishes nothing, and must still answer for Q."""

    @staticmethod
    def _wire(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        *,
        queue: ModelQueueFacts,
        receipted: set[str],
    ) -> Path:
        outputs = tmp_path / "outputs"
        outputs.write_text("", encoding="utf-8")
        monkeypatch.setenv("GITHUB_OUTPUT", str(outputs))
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
        monkeypatch.setattr(guard, "read_agent_queue", lambda *_a, **_k: queue)
        monkeypatch.setattr(
            guard,
            "read_first_parent_below",
            lambda _clone, revision, limit: _below(_HISTORY_OLDEST_FIRST)(
                revision, limit
            ),
        )
        monkeypatch.setattr(
            guard,
            "read_receipt_presence",
            lambda _repo, _lane, sha: _presence(receipted)(sha),
        )
        monkeypatch.setattr(
            guard, "load_runtime_affecting_predicate", lambda *_a: _runtime
        )
        monkeypatch.setattr(guard, "read_agent_loaded_code_sha", lambda *_a: "")
        monkeypatch.setattr(
            guard,
            "_contains_on_branch",
            lambda _repo, _branch, candidate, observed: (
                _HISTORY_OLDEST_FIRST.index(candidate)
                <= _HISTORY_OLDEST_FIRST.index(observed)
            ),
        )
        monkeypatch.setattr(guard, "_read_generation_or_warn", lambda _c: None)
        return outputs

    @staticmethod
    def _outputs(path: Path) -> dict[str, str]:
        values: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            values[key] = value
        return values

    @staticmethod
    def _observe(lane_revision: str) -> int:
        return guard.main(
            [
                "--observe",
                "--observed-merge",
                "9" * 40,
                "--deployed-revision",
                lane_revision,
                "--agent-url",
                "http://agent.invalid",
                "--runtime-path-validator",
                "validator.py",
            ]
        )

    _IDLE = ModelQueueFacts(
        commands_ahead=0,
        mean_service_time_seconds=None,
        service_sample_size=0,
        in_flight_correlation_id=None,
        unread_reason="",
        source="test",
    )

    def test_an_idle_lane_answers_for_the_orphan_with_no_wait(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # The lane sits at N2, put there by another repository's push. N2 is
        # non-runtime, so only Q is owed a receipt.
        outputs = self._wire(monkeypatch, tmp_path, queue=self._IDLE, receipted={_A})
        assert self._observe(_N2) == 0
        values = self._outputs(outputs)
        candidates = json.loads(values["reemit_candidates"])
        assert [c["sha"] for c in candidates] == [_Q]
        assert candidates[0]["endpoint"] is False
        assert candidates[0]["converged_via"] == f"{_N2} via container-revision"
        assert values["emit_receipt"] == "true"
        assert values["probe_lane"] == "true"
        assert values["verdict"] == "pass"
        assert values["mode"] == "observe"
        assert "OBSERVATION" in values["evidence"]

    def test_a_lane_sitting_on_the_orphan_answers_for_the_lane_revision_itself(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # The cross-repository push named Q's ref exactly. No run in this job
        # writes Q's receipt, so the observation includes the lane revision.
        outputs = self._wire(monkeypatch, tmp_path, queue=self._IDLE, receipted={_A})
        assert self._observe(_Q) == 0
        candidates = json.loads(self._outputs(outputs)["reemit_candidates"])
        assert [c["sha"] for c in candidates] == [_Q]

    def test_a_busy_agent_answers_for_nothing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        busy = ModelQueueFacts(
            commands_ahead=1,
            mean_service_time_seconds=None,
            service_sample_size=0,
            in_flight_correlation_id="corr-1",
            unread_reason="",
            source="test",
        )
        outputs = self._wire(monkeypatch, tmp_path, queue=busy, receipted={_A})
        assert self._observe(_N2) == 0
        values = self._outputs(outputs)
        assert values["reemit_candidates"] == "[]"
        assert values["emit_receipt"] == "false"
        assert values["probe_lane"] == "false"

    def test_an_unread_queue_is_not_idle(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        outputs = self._wire(
            monkeypatch,
            tmp_path,
            queue=ModelQueueFacts.unread("agent unreachable"),
            receipted={_A},
        )
        assert self._observe(_N2) == 0
        assert self._outputs(outputs)["emit_receipt"] == "false"

    def test_nothing_owed_probes_nothing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        outputs = self._wire(
            monkeypatch, tmp_path, queue=self._IDLE, receipted={_A, _Q}
        )
        assert self._observe(_N2) == 0
        values = self._outputs(outputs)
        assert values["reemit_candidates"] == "[]"
        assert values["probe_lane"] == "false"


class TestNoReceiptEverIsBounded:
    def test_no_receipt_in_the_lookback_answers_for_nothing(self) -> None:
        plan = durable_backfill(
            converged_revision=_R,
            already_answered=(),
            first_parent_below=_below(_HISTORY_OLDEST_FIRST),
            receipt_presence=_presence(set()),
            runtime_affecting=_runtime,
        )
        assert plan.backfill == ()
        assert plan.floor == ""
        assert "no compose-dev receipt of any verdict" in plan.reason

    def test_the_walk_never_reads_past_the_lookback(self) -> None:
        requested: list[int] = []
        history = [f"{i:040x}" for i in range(REEMISSION_LOOKBACK_COMMITS * 3)]
        seen: list[str] = []

        def below(revision: str, limit: int) -> Sequence[str]:
            requested.append(limit)
            # A reader that ignores the limit must still be bounded.
            return list(reversed(history[:-1]))

        def presence(sha: str) -> EnumReceiptPresence:
            seen.append(sha)
            return EnumReceiptPresence.ABSENT

        plan = durable_backfill(
            converged_revision=history[-1],
            already_answered=(),
            first_parent_below=below,
            receipt_presence=presence,
            runtime_affecting=lambda _sha: True,
        )
        assert requested == [REEMISSION_LOOKBACK_COMMITS]
        assert len(seen) == REEMISSION_LOOKBACK_COMMITS
        assert plan.examined == REEMISSION_LOOKBACK_COMMITS
        assert plan.backfill == ()

    def test_an_orphan_below_the_oldest_receipt_is_not_answered_for(self) -> None:
        # Nothing on record beneath it inside the lookback: nothing anchors it.
        history = (_Q, _A, _N1, _R)
        plan = durable_backfill(
            converged_revision=_R,
            already_answered=(),
            first_parent_below=_below(history),
            receipt_presence=_presence({_A}),
            runtime_affecting=_runtime,
        )
        assert plan.backfill == ()
        assert "nothing on record beneath them" in plan.reason


class TestEveryUnresolvedCaseAnswersForNothing:
    def test_an_unreadable_listing_is_not_an_absent_receipt(self) -> None:
        def presence(sha: str) -> EnumReceiptPresence:
            return (
                EnumReceiptPresence.UNREADABLE
                if sha == _N1
                else EnumReceiptPresence.ABSENT
            )

        plan = durable_backfill(
            converged_revision=_R,
            already_answered=(),
            first_parent_below=_below(_HISTORY_OLDEST_FIRST),
            receipt_presence=presence,
            runtime_affecting=_runtime,
        )
        assert plan.backfill == ()
        assert "unreadable" in plan.reason

    def test_an_unreadable_history_answers_for_nothing(self) -> None:
        def below(_revision: str, _limit: int) -> Sequence[str]:
            raise RuntimeError("fatal: bad revision")

        plan = durable_backfill(
            converged_revision=_R,
            already_answered=(),
            first_parent_below=below,
            receipt_presence=_presence({_A}),
            runtime_affecting=_runtime,
        )
        assert plan.backfill == ()

    def test_an_unclassifiable_commit_is_named_and_left(self) -> None:
        def runtime(sha: str) -> bool:
            if sha == _Q:
                raise RuntimeError("label read failed")
            return sha in _RUNTIME

        plan = durable_backfill(
            converged_revision=_R,
            already_answered=(),
            first_parent_below=_below(_HISTORY_OLDEST_FIRST),
            receipt_presence=_presence({_A}),
            runtime_affecting=runtime,
        )
        assert plan.backfill == ()
        assert plan.unclassified == (_Q,)

    def test_no_validator_widens_nothing(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(guard, "read_contained_commits", lambda *_a: ())
        code = guard.main(["--plan-reemission", "--converged-revision", _R])
        assert code == 0
        plan: dict[str, Any] = json.loads(capsys.readouterr().out)
        assert plan["durable_backfill"] == []
        assert "--runtime-path-validator" in plan["reason"]
