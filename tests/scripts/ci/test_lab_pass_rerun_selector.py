# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19233 AC8: the compose-dev emitter re-runs a refused staging delivery.

PS-5 item 2 of the unified verification plan: after the compose-dev emitter
uploads a PASS for a subject, it re-runs the failed gate job of every delivery
run whose sha resolves to that subject and whose gate refused PENDING, ABSENT or
on an INDETERMINATE-only receipt, within the 14,400 s bound. A delivery refused
on a FAIL check is never re-run: the FAIL stands until the subject changes.

One further rule, which is this module's and not the plan's: only the NEWEST
delivery run is ever re-run. Re-running an older run after a newer one has been
created would deliver older code over newer code on staging, and it would share
the newer run's concurrency group, which cancels in progress.

The selector reads each run's own gate verdict (the JSON the gate step writes
and uploads as ``lab-pass-gate-verdict-<run_id>-<attempt>``), so what it acts
on is what the gate itself concluded, not a re-derivation.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timedelta
from typing import Any

import pytest

from scripts.ci.lab_pass_receipt import (
    DELIVERY_OVERALL_BOUND_SECONDS,
    EnumLabLane,
    ModelDeliveryRun,
    rerun_refused_deliveries,
    select_refused_deliveries,
    verdict_artifact_name,
)
from tests.scripts.ci._lab_pass_fixtures import (
    REPO,
    SHA_3E4A,
    SHA_4ACA,
    SHA_DF6F,
    FakeSurface,
    receipt,
    ts,
)

pytestmark = pytest.mark.unit

FIRST_READ = ts("2026-09-22T19:44:29Z")
NOW = ts("2026-09-22T19:50:00Z")
WORKFLOW = "deliver-dev-candidate-to-staging.yml"


def _verdict(
    *,
    run_id: int,
    attempt: int = 1,
    subject: str = SHA_4ACA,
    sha: str = SHA_4ACA,
    refusal: str = "PENDING",
    first_read_at: datetime = FIRST_READ,
) -> dict[str, Any]:
    return {
        "schema": "lab_pass_gate_verdict.v1",
        "sha": sha,
        "subject": subject,
        "token": refusal,
        "lanes": {"compose-dev": refusal},
        "first_read_at": first_read_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": run_id,
        "run_attempt": attempt,
    }


def _run(
    run_id: int,
    *,
    created: str = "2026-09-22T19:30:00Z",
    status: str = "completed",
    conclusion: str | None = "failure",
    attempt: int = 1,
    verdict: dict[str, Any] | None = None,
) -> ModelDeliveryRun:
    return ModelDeliveryRun(
        run_id=run_id,
        created_at=ts(created),
        status=status,
        conclusion=conclusion,
        run_attempt=attempt,
        verdict=verdict,
    )


def _select(
    subject: str, runs: list[ModelDeliveryRun], now: datetime = NOW
) -> list[int]:
    selected, _ = select_refused_deliveries(
        subject, runs, now=now, bound_seconds=DELIVERY_OVERALL_BOUND_SECONDS
    )
    return [r.run_id for r in selected]


class TestSelector:
    @pytest.mark.parametrize("refusal", ["PENDING", "ABSENT", "INDETERMINATE"])
    def test_picks_the_newest_run_refused_on_an_eligible_token(
        self, refusal: str
    ) -> None:
        runs = [_run(900, verdict=_verdict(run_id=900, refusal=refusal))]
        assert _select(SHA_4ACA, runs) == [900]

    @pytest.mark.parametrize(
        "refusal", ["FAIL", "UNREADABLE", "TIMED_OUT", "ANY_OF_UNMET", "PASS"]
    )
    def test_never_picks_a_run_refused_on_a_fail_check_or_other_token(
        self, refusal: str
    ) -> None:
        runs = [_run(900, verdict=_verdict(run_id=900, refusal=refusal))]
        assert _select(SHA_4ACA, runs) == []

    def test_picks_exactly_the_eligible_run_among_several(self) -> None:
        """The falsifier's shape: of several delivery runs only the newest,
        refused PENDING for this subject inside the bound, is picked."""
        runs = [
            _run(
                903,
                created="2026-09-22T19:40:00Z",
                verdict=_verdict(run_id=903, refusal="PENDING"),
            ),
            _run(
                902,
                created="2026-09-22T19:20:00Z",
                verdict=_verdict(run_id=902, refusal="ABSENT"),
            ),
            _run(
                901,
                created="2026-09-22T19:10:00Z",
                verdict=_verdict(run_id=901, refusal="FAIL"),
            ),
            _run(900, created="2026-09-22T19:00:00Z", conclusion="success"),
        ]
        assert _select(SHA_4ACA, runs) == [903]

    def test_another_subject_is_not_picked(self) -> None:
        runs = [_run(900, verdict=_verdict(run_id=900, subject=SHA_3E4A))]
        assert _select(SHA_4ACA, runs) == []

    def test_inside_the_bound_is_picked_and_past_it_is_not(self) -> None:
        runs = [_run(900, verdict=_verdict(run_id=900))]
        assert _select(SHA_4ACA, runs, FIRST_READ + timedelta(seconds=14_399)) == [900]
        assert _select(SHA_4ACA, runs, FIRST_READ + timedelta(seconds=14_401)) == []

    def test_an_older_run_is_never_re_run_over_a_newer_one(self) -> None:
        runs = [
            _run(
                901,
                created="2026-09-22T19:48:00Z",
                status="in_progress",
                conclusion=None,
            ),
            _run(900, verdict=_verdict(run_id=900)),
        ]
        assert _select(SHA_4ACA, runs) == []

    def test_a_newer_successful_run_supersedes_a_refused_older_one(self) -> None:
        runs = [
            _run(901, created="2026-09-22T19:48:00Z", conclusion="success"),
            _run(900, verdict=_verdict(run_id=900)),
        ]
        assert _select(SHA_4ACA, runs) == []

    def test_a_run_with_no_verdict_is_not_picked(self) -> None:
        """The gate step never ran (an earlier step in the job refused), so
        nothing says the refusal was about compose-dev timing."""
        assert _select(SHA_4ACA, [_run(900, verdict=None)]) == []

    def test_a_verdict_from_an_earlier_attempt_is_not_read_as_current(
        self,
    ) -> None:
        runs = [_run(900, attempt=2, verdict=_verdict(run_id=900, attempt=1))]
        assert _select(SHA_4ACA, runs) == []

    def test_every_decision_is_explained(self) -> None:
        runs = [
            _run(
                901,
                created="2026-09-22T19:48:00Z",
                verdict=_verdict(run_id=901, refusal="FAIL"),
            ),
        ]
        selected, reasons = select_refused_deliveries(
            SHA_4ACA, runs, now=NOW, bound_seconds=DELIVERY_OVERALL_BOUND_SECONDS
        )
        assert selected == []
        assert any("901" in r and "FAIL" in r for r in reasons)


class TestRerunEndToEnd:
    """The emitter's subcommand over the fake REST surface."""

    def _surface(self, *, compose_dev_outcome: str | None) -> FakeSurface:
        surface = FakeSurface()
        surface.add(receipt(SHA_DF6F, EnumLabLane.ONEX_LAB))
        if compose_dev_outcome is not None:
            surface.add(receipt(SHA_3E4A, outcome=compose_dev_outcome))
        verdict = _verdict(
            run_id=35812000001,
            sha=SHA_DF6F,
            subject=SHA_3E4A,
            refusal="ABSENT",
            first_read_at=ts("2026-09-23T03:40:00Z"),
        )
        surface.extra_artifacts[verdict_artifact_name(35812000001, 1)] = (
            "verdict.json",
            json.dumps(verdict),
        )
        surface.runs[WORKFLOW] = [
            {
                "id": 35812000001,
                "created_at": "2026-09-23T03:30:00Z",
                "status": "completed",
                "conclusion": "failure",
                "run_attempt": 1,
                "head_branch": "dev",
            }
        ]
        return surface

    def _run_it(
        self, monkeypatch: pytest.MonkeyPatch, surface: FakeSurface
    ) -> tuple[int, list[str], str]:
        posted: list[str] = []
        monkeypatch.setattr("scripts.ci.lab_pass_receipt._gh_api", surface)
        monkeypatch.setattr(
            "scripts.ci.lab_pass_receipt._gh_api_post", lambda p: posted.append(p)
        )
        out = io.StringIO()
        code = rerun_refused_deliveries(
            REPO,
            [SHA_3E4A],
            workflow=WORKFLOW,
            branch="dev",
            bound_seconds=DELIVERY_OVERALL_BOUND_SECONDS,
            out=out,
            now=lambda: ts("2026-09-23T06:14:00Z"),
        )
        return code, posted, out.getvalue()

    def test_a_pass_re_runs_the_refused_3e4aaded_delivery(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        code, posted, output = self._run_it(
            monkeypatch, self._surface(compose_dev_outcome="ok")
        )
        assert code == 0, output
        assert posted == [f"repos/{REPO}/actions/runs/35812000001/rerun-failed-jobs"]
        assert "compose-dev" in output

    def test_a_fail_receipt_re_runs_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        code, posted, output = self._run_it(
            monkeypatch, self._surface(compose_dev_outcome="fail")
        )
        assert code == 0, output
        assert posted == []

    def test_no_receipt_re_runs_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        code, posted, _ = self._run_it(
            monkeypatch, self._surface(compose_dev_outcome=None)
        )
        assert code == 0
        assert posted == []

    def test_a_failed_re_run_request_fails_the_step(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        surface = self._surface(compose_dev_outcome="ok")
        monkeypatch.setattr("scripts.ci.lab_pass_receipt._gh_api", surface)

        def refuse(path: str) -> None:
            msg = "`gh api -X POST` exited 1: HTTP 403 Resource not accessible"
            raise RuntimeError(msg)

        monkeypatch.setattr("scripts.ci.lab_pass_receipt._gh_api_post", refuse)
        out = io.StringIO()
        code = rerun_refused_deliveries(
            REPO,
            [SHA_3E4A],
            workflow=WORKFLOW,
            branch="dev",
            bound_seconds=DELIVERY_OVERALL_BOUND_SECONDS,
            out=out,
            now=lambda: ts("2026-09-23T06:14:00Z"),
        )
        assert code == 1
        assert "HTTP 403" in out.getvalue()

    def test_an_unreadable_run_listing_fails_the_step(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        surface = self._surface(compose_dev_outcome="ok")
        surface.fail_paths = ("/actions/workflows/",)
        code, posted, _ = self._run_it(monkeypatch, surface)
        assert code == 1
        assert posted == []
