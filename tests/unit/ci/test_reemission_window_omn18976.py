# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18976 part two — a sha that converges late gets its receipt from a later run.

WHAT PART ONE LEFT
------------------
A merge queued behind another deploy now emits NO receipt instead of a terminal
FAIL, so nothing asserts the lane misbehaved. But the sha still has no receipt,
and the delivery gate's absent branch keeps refusing it. Its own verify run is
over; only a LATER run can answer for it.

THE WINDOW, AND WHY IT IS NOT ARBITRARY ANCESTRY
------------------------------------------------
A verify run reads the lane twice: once when it starts, at revision ``P``, and
once when it converges, at ``X``. Every merge that landed strictly between
those two is a merge that was queued behind this one -- it had its own verify
run, that run found the lane still at ``P`` or earlier, and it exited without a
receipt. That set, and only that set, is what this run may answer for.

``P..X`` is the bound. It is emphatically NOT "every ancestor of X": the lane
has been running for weeks and contains thousands of shas it never individually
converged on, and emitting for those would manufacture receipts for merges this
run observed nothing about.

WHAT A RE-EMITTED RECEIPT CLAIMS
---------------------------------
Exactly what the probes support: the lane came to run an image containing this
sha, and here are the live reads taken against that image. It records
``converged_via`` so the claim is legible -- a reader can tell a receipt written
by the sha's own run from one written by a later convergence that subsumed it,
and the two are different evidence even when both are PASS.

Every assertion here is about a boundary, because the failure mode of this
feature is emitting for a sha nobody watched.
"""

from __future__ import annotations

import pytest

from scripts.ci.check_dev_lane_staleness import reemission_window

pytestmark = pytest.mark.unit

_P = "a" * 40
_S1 = "b" * 40
_S2 = "c" * 40
_X = "d" * 40
_OUTSIDE = "e" * 40


def _contained(_previous: str, _converged: str) -> tuple[str, ...]:
    """The two-sha queue: S1 and S2 landed while the lane sat at P."""
    return (_S1, _S2, _X)


class TestTheWindowIsTheQueueNotTheHistory:
    def test_two_shas_queued_behind_this_one_are_both_in_the_window(self) -> None:
        window = reemission_window(
            previous_revision=_P,
            converged_revision=_X,
            resolve_contained=_contained,
        )
        assert window == (_S1, _S2), (
            "the two merges that landed while the lane sat at the previous "
            "revision are exactly the ones whose own verify runs exited queued "
            "with no receipt"
        )

    def test_the_converged_sha_is_excluded(self) -> None:
        # Its own run is writing its own receipt right now. Emitting twice for
        # it would race two artifacts of the same name from one job.
        window = reemission_window(
            previous_revision=_P,
            converged_revision=_X,
            resolve_contained=_contained,
        )
        assert _X not in window

    def test_a_sha_outside_the_window_gets_nothing(self) -> None:
        # The load-bearing bound. An ancestor the lane happens to contain, but
        # which did not land in this window, was watched by nobody here.
        window = reemission_window(
            previous_revision=_P,
            converged_revision=_X,
            resolve_contained=_contained,
        )
        assert _OUTSIDE not in window

    def test_an_empty_queue_window_yields_nothing(self) -> None:
        # The common case: the lane converged on the very next merge, so the
        # sha's own run wrote its own receipt and there is nothing to answer for.
        window = reemission_window(
            previous_revision=_P,
            converged_revision=_X,
            resolve_contained=lambda _p, _c: (_X,),
        )
        assert window == ()


class TestItRefusesRatherThanGuessing:
    @pytest.mark.parametrize(
        ("previous", "converged"),
        [("", _X), (_P, ""), ("", "")],
        ids=["no-previous", "no-converged", "neither"],
    )
    def test_a_missing_endpoint_yields_nothing(
        self, previous: str, converged: str
    ) -> None:
        # Without both endpoints there is no window, and a half-bounded window
        # is the "arbitrary ancestry" failure this feature must not have.
        assert (
            reemission_window(
                previous_revision=previous,
                converged_revision=converged,
                resolve_contained=_contained,
            )
            == ()
        )

    def test_an_unresolvable_comparison_yields_nothing(self) -> None:
        # Fail closed: emitting nothing leaves the gate refusing those shas,
        # which is the same answer as today. Emitting on a guess would not be.
        def _unresolvable(_previous: str, _converged: str) -> tuple[str, ...]:
            return ()

        assert (
            reemission_window(
                previous_revision=_P,
                converged_revision=_X,
                resolve_contained=_unresolvable,
            )
            == ()
        )

    def test_a_lane_that_never_moved_yields_nothing(self) -> None:
        # P == X. Nothing converged, so nothing was subsumed.
        assert (
            reemission_window(
                previous_revision=_P,
                converged_revision=_P,
                resolve_contained=_contained,
            )
            == ()
        )

    def test_the_window_is_deduplicated_and_ordered(self) -> None:
        # The comparison surface is not this module's to trust blindly; a
        # repeated sha would emit the same artifact name twice from one job.
        window = reemission_window(
            previous_revision=_P,
            converged_revision=_X,
            resolve_contained=lambda _p, _c: (_S2, _S1, _S2, _X),
        )
        assert window == (_S2, _S1)
        assert len(set(window)) == len(window)


class TestConvergedViaIsRecordedAndOptional:
    """The provenance half: a re-emitted receipt says so, an ordinary one does not."""

    @staticmethod
    def _receipt(converged_via: str):
        from datetime import UTC, datetime

        from scripts.ci.lab_pass_receipt import (
            EnumLabLane,
            ModelLabPassCheck,
            build_receipt,
        )

        return build_receipt(
            sha=_S1,
            lane=EnumLabLane.COMPOSE_DEV,
            started_at=datetime(2026, 9, 21, 8, 0, tzinfo=UTC),
            finished_at=datetime(2026, 9, 21, 8, 5, tzinfo=UTC),
            checks=[
                ModelLabPassCheck(
                    name="deployed_revision", ok=True, evidence="lane at X"
                )
            ],
            agent_command_id=None,
            converged_via=converged_via,
        )

    def test_a_reemitted_receipt_records_the_sha_that_answered_for_it(self) -> None:
        receipt = self._receipt(_X)
        assert receipt.converged_via == _X
        assert '"converged_via"' in receipt.to_json()

    def test_an_ordinary_receipt_is_byte_identical_to_before(self) -> None:
        # The compatibility guard. An emitter that did not re-emit must produce
        # exactly the wire shape it produced before this change, or every
        # receipt already in flight stops parsing.
        assert '"converged_via"' not in self._receipt("").to_json()

    def test_it_survives_a_round_trip(self) -> None:
        from scripts.ci.lab_pass_receipt import ModelLabPassReceipt

        original = self._receipt(_X)
        assert ModelLabPassReceipt.from_json(original.to_json()).converged_via == _X

    def test_a_receipt_written_before_this_change_still_parses(self) -> None:
        # Positive control on the optionality: absence means "its own run
        # wrote it", which is a real answer rather than a missing field.
        from scripts.ci.lab_pass_receipt import ModelLabPassReceipt

        body = self._receipt("").to_json()
        assert "converged_via" not in body
        assert ModelLabPassReceipt.from_json(body).converged_via == ""


class TestReemitCopiesEvidenceAndRefusesNonsense:
    """The re-key itself: copied evidence, recorded provenance, two refusals."""

    @staticmethod
    def _converged():
        from datetime import UTC, datetime

        from scripts.ci.lab_pass_receipt import (
            EnumLabLane,
            ModelLabPassCheck,
            build_receipt,
        )

        return build_receipt(
            sha=_X,
            lane=EnumLabLane.COMPOSE_DEV,
            started_at=datetime(2026, 9, 21, 8, 0, tzinfo=UTC),
            finished_at=datetime(2026, 9, 21, 8, 5, tzinfo=UTC),
            checks=[
                ModelLabPassCheck(
                    name="deployed_revision", ok=True, evidence=f"lane at {_X}"
                ),
                ModelLabPassCheck(name="ready_main", ok=True, evidence="200 healthy"),
            ],
            agent_command_id=None,
        )

    def test_the_queued_sha_gets_the_converged_runs_evidence_verbatim(self) -> None:
        from scripts.ci.lab_pass_receipt import reemit_receipt

        source = self._converged()
        out = reemit_receipt(source, _S1)
        assert out.sha == _S1
        assert out.converged_via == _X
        # Copied, not re-taken: the probes were read against the image that
        # contains this sha, which is the only moment they could be.
        assert out.checks == source.checks
        assert out.result == source.result
        assert out.started_at == source.started_at

    def test_a_passing_convergence_yields_a_passing_receipt(self) -> None:
        from scripts.ci.lab_pass_receipt import EnumLabPassResult, reemit_receipt

        assert reemit_receipt(self._converged(), _S1).result is EnumLabPassResult.PASS

    def test_it_refuses_to_rekey_onto_its_own_sha(self) -> None:
        # Would race two artifacts of one name from one job.
        from scripts.ci.lab_pass_receipt import reemit_receipt

        with pytest.raises(ValueError, match="onto itself"):
            reemit_receipt(self._converged(), _X)

    def test_it_refuses_to_chain_provenance(self) -> None:
        # A re-emission of a re-emission has provenance nobody can read.
        from scripts.ci.lab_pass_receipt import reemit_receipt

        once = reemit_receipt(self._converged(), _S1)
        with pytest.raises(ValueError, match="itself a re-emission"):
            reemit_receipt(once, _S2)

    def test_a_failing_convergence_re_emits_a_failure(self) -> None:
        # The direction that must NOT be lost: if the lane converged and the
        # probes failed, every sha in that window inherits the failure. A
        # re-emission that quietly upgraded to PASS would be the worst
        # possible outcome of this whole ticket.
        from datetime import UTC, datetime

        from scripts.ci.lab_pass_receipt import (
            EnumLabLane,
            EnumLabPassResult,
            ModelLabPassCheck,
            build_receipt,
            reemit_receipt,
        )

        failed = build_receipt(
            sha=_X,
            lane=EnumLabLane.COMPOSE_DEV,
            started_at=datetime(2026, 9, 21, 8, 0, tzinfo=UTC),
            finished_at=datetime(2026, 9, 21, 8, 5, tzinfo=UTC),
            checks=[
                ModelLabPassCheck(name="ready_main", ok=False, evidence="503 not ready")
            ],
            agent_command_id=None,
        )
        assert failed.result is EnumLabPassResult.FAIL
        assert reemit_receipt(failed, _S1).result is EnumLabPassResult.FAIL
