# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19233: staging delivery requires the compose-dev receipt of the proof subject.

Operator ruling 2026-09-23 (decision S of the unified verification plan): the
staging delivery gate requires the compose-dev lane's lab-pass receipt; missing,
failed or unreadable compose-dev evidence prevents promotion; another lane's
PASS never substitutes. The ruling's bounded wait is the plan's PS-5, not a wait
inside the 10-minute gate job:

1. no compose-dev receipt for the subject refuses PENDING while the subject's
   rebuild is in flight, and ABSENT otherwise;
2. the compose-dev emitter re-runs the refused gate once it uploads a PASS for
   the same subject (tests/scripts/ci/test_lab_pass_rerun_selector.py);
3. a read more than 14,400 s after the delivery run's first gate read refuses
   TIMED_OUT, even when a PASS has since arrived, and is never a pass.

MEASURED, AND WHY THIS EXISTS. Between 2026-09-22T19:27Z and 2026-09-23T06:20Z
all ten successful staging deliveries passed on the kind-cluster onex-lab boot
receipt alone, while the last compose-dev PASS was over ten hours stale: the
own-sha gate step was any-of across lanes (plan section 0b).

Every refusal here has a flipped sibling that passes on the same surface once
the missing fact is supplied.
"""

from __future__ import annotations

import io
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.ci.lab_pass_receipt import (
    ANY_OF_DEFAULT_LANES,
    DELIVERY_OVERALL_BOUND_SECONDS,
    EnumGateToken,
    EnumLabLane,
    ReceiptLookupError,
    evaluate_gate,
    main,
    resolve_first_gate_read,
)
from tests.scripts.ci._lab_pass_fixtures import (
    GATE_JOB,
    REPO,
    SHA_3E4A,
    SHA_4ACA,
    SHA_DF6F,
    FakeSurface,
    receipt,
    ts,
)

pytestmark = pytest.mark.unit

COMPOSE_DEV = EnumLabLane.COMPOSE_DEV
ONEX_LAB = EnumLabLane.ONEX_LAB


def _gate(
    surface: FakeSurface,
    monkeypatch: pytest.MonkeyPatch,
    *,
    sha: str,
    subject: str | None = None,
    pending: bool = False,
    first_read_at: Any = None,
    now: Any = None,
    verdict_path: Path | None = None,
) -> tuple[int, str]:
    monkeypatch.setattr("scripts.ci.lab_pass_receipt._gh_api", surface)
    read_at = now or ts("2026-09-23T12:00:00Z")
    out = io.StringIO()
    code = evaluate_gate(
        REPO,
        sha,
        list(ANY_OF_DEFAULT_LANES),
        out,
        required=[COMPOSE_DEV],
        required_sha=subject,
        rebuild_pending=lambda s: pending,
        first_read_at=first_read_at,
        overall_bound_seconds=DELIVERY_OVERALL_BOUND_SECONDS,
        now=lambda: read_at,
        verdict_out=verdict_path,
    )
    return code, out.getvalue()


def _verdict(path: Path) -> dict[str, Any]:
    body = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(body, dict)
    return body


# --------------------------------------------------------------------------- #
# AC1 -- an onex-lab PASS with no compose-dev receipt is refused, naming it.    #
# --------------------------------------------------------------------------- #
class TestRequiredLaneSelection:
    def test_missing_lane_onex_lab_pass_alone_is_refused_naming_compose_dev(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        surface = FakeSurface()
        surface.add(receipt(SHA_4ACA, ONEX_LAB))
        verdict = tmp_path / "verdict.json"
        code, output = _gate(surface, monkeypatch, sha=SHA_4ACA, verdict_path=verdict)
        assert code == 1
        assert "REQUIRED lane compose-dev does not pass" in output
        assert "token=ABSENT" in output
        body = _verdict(verdict)
        assert body["token"] == "ABSENT"
        assert body["lanes"] == {"compose-dev": "ABSENT"}
        assert body["subject"] == SHA_4ACA

    def test_missing_lane_positive_control_compose_dev_pass_passes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        surface = FakeSurface()
        surface.add(receipt(SHA_4ACA, ONEX_LAB), receipt(SHA_4ACA, COMPOSE_DEV))
        verdict = tmp_path / "verdict.json"
        code, output = _gate(surface, monkeypatch, sha=SHA_4ACA, verdict_path=verdict)
        assert code == 0, output
        assert _verdict(verdict)["token"] == "PASS"

    # --------------------------------------------------------------------- #
    # AC2 -- a compose-dev FAIL beside another lane's PASS is refused.       #
    # --------------------------------------------------------------------- #
    def test_compose_dev_fail_beside_other_pass_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        surface = FakeSurface()
        surface.add(
            receipt(SHA_4ACA, ONEX_LAB),
            receipt(SHA_4ACA, EnumLabLane.ONEX_LAB_K3S),
            receipt(SHA_4ACA, COMPOSE_DEV, outcome="fail"),
        )
        verdict = tmp_path / "verdict.json"
        code, output = _gate(surface, monkeypatch, sha=SHA_4ACA, verdict_path=verdict)
        assert code == 1
        assert "token=FAIL" in output
        assert _verdict(verdict)["token"] == "FAIL"

    def test_compose_dev_fail_beside_other_pass_indeterminate_only_is_its_own_token(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """PS-3 R2: a receipt whose only non-passing checks are INDETERMINATE
        refuses, under a token the re-run selector may act on."""
        surface = FakeSurface()
        surface.add(
            receipt(SHA_4ACA, ONEX_LAB),
            receipt(SHA_4ACA, COMPOSE_DEV, outcome="indeterminate"),
        )
        verdict = tmp_path / "verdict.json"
        code, _ = _gate(surface, monkeypatch, sha=SHA_4ACA, verdict_path=verdict)
        assert code == 1
        assert _verdict(verdict)["token"] == "INDETERMINATE"

    def test_compose_dev_unreadable_surface_is_refused_unreadable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        surface = FakeSurface(fail_paths=("lab-pass-receipt-compose-dev-",))
        surface.add(receipt(SHA_4ACA, ONEX_LAB))
        verdict = tmp_path / "verdict.json"
        code, _ = _gate(surface, monkeypatch, sha=SHA_4ACA, verdict_path=verdict)
        assert code == 1
        assert _verdict(verdict)["token"] == "UNREADABLE"

    def test_subject_is_named_when_the_delivered_sha_inherits(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """S.2: a delivered sha that is not runtime-affecting is judged on its
        PS-1 subject's receipt, and the refusal names the subject."""
        surface = FakeSurface()
        surface.add(receipt(SHA_DF6F, ONEX_LAB))
        verdict = tmp_path / "verdict.json"
        code, output = _gate(
            surface,
            monkeypatch,
            sha=SHA_DF6F,
            subject=SHA_3E4A,
            verdict_path=verdict,
        )
        assert code == 1
        assert f"does not pass for sha {SHA_3E4A}" in output
        body = _verdict(verdict)
        assert body["sha"] == SHA_DF6F
        assert body["subject"] == SHA_3E4A


# --------------------------------------------------------------------------- #
# AC6 -- a late-arriving receipt: refused PENDING or ABSENT, passed on re-run.  #
# --------------------------------------------------------------------------- #
class TestLateArrivingReceipt:
    def test_late_arriving_receipt_rerun_4aca83d9_pending_then_pass(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The measured 4aca83d9 ordering: the gate read at 19:44:29Z while the
        rebuild was in flight, the compose-dev PASS was written at 19:47:04Z,
        and the re-run read passes."""
        first_read = ts("2026-09-22T19:44:29Z")
        surface = FakeSurface()
        surface.add(receipt(SHA_4ACA, ONEX_LAB))

        first = tmp_path / "attempt1.json"
        code, output = _gate(
            surface,
            monkeypatch,
            sha=SHA_4ACA,
            pending=True,
            now=first_read,
            verdict_path=first,
        )
        assert code == 1
        assert "token=PENDING" in output
        assert _verdict(first)["token"] == "PENDING"
        assert _verdict(first)["first_read_at"] == "2026-09-22T19:44:29Z"

        surface.add(receipt(SHA_4ACA, COMPOSE_DEV))  # written 19:47:04Z

        second = tmp_path / "attempt2.json"
        code, output = _gate(
            surface,
            monkeypatch,
            sha=SHA_4ACA,
            first_read_at=first_read,
            now=ts("2026-09-22T19:50:00Z"),
            verdict_path=second,
        )
        assert code == 0, output
        body = _verdict(second)
        assert body["token"] == "PASS"
        assert body["elapsed_seconds"] == 331

    def test_late_arriving_receipt_rerun_3e4aaded_r1_absent_then_pass(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The 3e4aaded R1 case: attempt 1 of the emitter was queued behind
        other deploys and wrote no receipt, nothing was in flight, so the gate
        (delivering df6ffe02, whose PS-1 subject is 3e4aaded) reads ABSENT.
        Attempt 2's PASS arrives at 06:13:23Z and the re-run read passes."""
        first_read = ts("2026-09-23T03:40:00Z")
        surface = FakeSurface()
        surface.add(receipt(SHA_DF6F, ONEX_LAB))

        first = tmp_path / "attempt1.json"
        code, _ = _gate(
            surface,
            monkeypatch,
            sha=SHA_DF6F,
            subject=SHA_3E4A,
            pending=False,
            now=first_read,
            verdict_path=first,
        )
        assert code == 1
        assert _verdict(first)["token"] == "ABSENT"

        surface.add(receipt(SHA_3E4A, COMPOSE_DEV))  # attempt 2, 06:13:23Z

        second = tmp_path / "attempt2.json"
        code, output = _gate(
            surface,
            monkeypatch,
            sha=SHA_DF6F,
            subject=SHA_3E4A,
            first_read_at=first_read,
            now=ts("2026-09-23T06:15:00Z"),
            verdict_path=second,
        )
        assert code == 0, output
        assert _verdict(second)["token"] == "PASS"

    def test_late_arriving_receipt_rerun_pending_is_distinct_from_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        surface = FakeSurface()
        surface.add(receipt(SHA_4ACA, ONEX_LAB))
        _, pending = _gate(surface, monkeypatch, sha=SHA_4ACA, pending=True)
        _, absent = _gate(surface, monkeypatch, sha=SHA_4ACA, pending=False)
        assert "token=PENDING" in pending
        assert "token=ABSENT" in absent

    def test_late_arriving_receipt_rerun_pending_probe_failure_is_absent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A probe that cannot say whether a rebuild is in flight refuses ABSENT
        with the reason; both are refusals, neither is a pass."""
        surface = FakeSurface()
        surface.add(receipt(SHA_4ACA, ONEX_LAB))
        monkeypatch.setattr("scripts.ci.lab_pass_receipt._gh_api", surface)

        def boom(sha: str) -> bool:
            raise RuntimeError("actions api unreadable")

        out = io.StringIO()
        verdict = tmp_path / "v.json"
        code = evaluate_gate(
            REPO,
            SHA_4ACA,
            list(ANY_OF_DEFAULT_LANES),
            out,
            required=[COMPOSE_DEV],
            rebuild_pending=boom,
            verdict_out=verdict,
        )
        assert code == 1
        assert _verdict(verdict)["token"] == "ABSENT"
        assert "actions api unreadable" in out.getvalue()


# --------------------------------------------------------------------------- #
# AC7 -- the overall bound: TIMED_OUT past 14,400 s, never a pass.              #
# --------------------------------------------------------------------------- #
class TestOverallBound:
    FIRST = ts("2026-09-23T03:00:00Z")

    def _with_pass(self) -> FakeSurface:
        surface = FakeSurface()
        surface.add(receipt(SHA_4ACA, ONEX_LAB), receipt(SHA_4ACA, COMPOSE_DEV))
        return surface

    def test_timed_out_at_overall_bound_refuses_even_with_a_pass(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        verdict = tmp_path / "v.json"
        code, output = _gate(
            self._with_pass(),
            monkeypatch,
            sha=SHA_4ACA,
            first_read_at=self.FIRST,
            now=self.FIRST + timedelta(seconds=14_401),
            verdict_path=verdict,
        )
        assert code == 1
        assert "token=TIMED_OUT" in output
        assert "PASSED" not in output
        # Every lane reads PASS: the refusal must not claim one does not.
        assert "a required lane does not carry a PASS" not in output
        assert _verdict(verdict)["token"] == "TIMED_OUT"

    def test_timed_out_at_overall_bound_a_read_at_14399_with_a_pass_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        code, output = _gate(
            self._with_pass(),
            monkeypatch,
            sha=SHA_4ACA,
            first_read_at=self.FIRST,
            now=self.FIRST + timedelta(seconds=14_399),
        )
        assert code == 0, output

    def test_timed_out_at_overall_bound_is_distinct_from_the_other_tokens(
        self,
    ) -> None:
        tokens = {
            EnumGateToken.TIMED_OUT,
            EnumGateToken.PENDING,
            EnumGateToken.ABSENT,
            EnumGateToken.FAIL,
        }
        assert len({t.value for t in tokens}) == 4
        assert EnumGateToken.TIMED_OUT is not EnumGateToken.PASS

    def test_timed_out_at_overall_bound_the_bound_is_four_hours(self) -> None:
        # 10,963 s (3e4aaded's queue bound) + 2,700 s (one verify run) = 13,663,
        # rounded up to four hours. Changing it is a ruling, not a constant.
        assert DELIVERY_OVERALL_BOUND_SECONDS == 14_400

    def test_timed_out_at_overall_bound_past_bound_pending_is_timed_out(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        surface = FakeSurface()
        surface.add(receipt(SHA_4ACA, ONEX_LAB))
        verdict = tmp_path / "v.json"
        code, _ = _gate(
            surface,
            monkeypatch,
            sha=SHA_4ACA,
            pending=True,
            first_read_at=self.FIRST,
            now=self.FIRST + timedelta(seconds=20_000),
            verdict_path=verdict,
        )
        assert code == 1
        assert _verdict(verdict)["token"] == "TIMED_OUT"


# --------------------------------------------------------------------------- #
# The first gate read of a delivery run, across its attempts.                   #
# --------------------------------------------------------------------------- #
class TestFirstGateRead:
    NOW = ts("2026-09-23T08:00:00Z")

    def test_attempt_one_is_its_own_first_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        surface = FakeSurface()
        monkeypatch.setattr("scripts.ci.lab_pass_receipt._gh_api", surface)
        got = resolve_first_gate_read(REPO, 77, 1, GATE_JOB, now=lambda: self.NOW)
        assert got == self.NOW
        assert surface.reads == []

    def test_a_rerun_reads_the_earliest_prior_gate_job_start(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        surface = FakeSurface(
            jobs={
                (77, 1): [
                    {"name": "Build", "started_at": "2026-09-23T03:00:00Z"},
                    {
                        "name": GATE_JOB,
                        "started_at": "2026-09-23T03:40:00Z",
                        "conclusion": "failure",
                    },
                ],
                (77, 2): [
                    {
                        "name": GATE_JOB,
                        "started_at": "2026-09-23T05:00:00Z",
                        "conclusion": "failure",
                    }
                ],
            }
        )
        monkeypatch.setattr("scripts.ci.lab_pass_receipt._gh_api", surface)
        got = resolve_first_gate_read(REPO, 77, 3, GATE_JOB, now=lambda: self.NOW)
        assert got == ts("2026-09-23T03:40:00Z")

    def test_a_prior_attempt_whose_gate_never_started_does_not_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        surface = FakeSurface(
            jobs={
                (77, 1): [
                    {"name": GATE_JOB, "started_at": None, "conclusion": "skipped"}
                ],
            }
        )
        monkeypatch.setattr("scripts.ci.lab_pass_receipt._gh_api", surface)
        got = resolve_first_gate_read(REPO, 77, 2, GATE_JOB, now=lambda: self.NOW)
        assert got == self.NOW

    def test_an_unreadable_jobs_surface_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        surface = FakeSurface(fail_paths=("/attempts/",))
        monkeypatch.setattr("scripts.ci.lab_pass_receipt._gh_api", surface)
        with pytest.raises(ReceiptLookupError):
            resolve_first_gate_read(REPO, 77, 2, GATE_JOB, now=lambda: self.NOW)

    def test_the_cli_refuses_unreadable_when_the_first_read_cannot_be_resolved(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        surface = FakeSurface(fail_paths=("/attempts/",))
        surface.add(receipt(SHA_4ACA, ONEX_LAB), receipt(SHA_4ACA, COMPOSE_DEV))
        monkeypatch.setattr("scripts.ci.lab_pass_receipt._gh_api", surface)
        verdict = tmp_path / "v.json"
        code = main(
            [
                "gate",
                "--sha",
                SHA_4ACA,
                "--repo",
                REPO,
                "--require-lane",
                "compose-dev",
                "--overall-bound-seconds",
                "14400",
                "--run-id",
                "77",
                "--run-attempt",
                "2",
                "--gate-job-name",
                GATE_JOB,
                "--verdict-out",
                str(verdict),
            ]
        )
        assert code == 1
        assert _verdict(verdict)["token"] == "UNREADABLE"
