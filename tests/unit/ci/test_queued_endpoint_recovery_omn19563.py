# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19563: a queued head is receipted after it eventually deploys.

A queued verify run deliberately emits no receipt. If that head deploys only
after its verify job has ended, the next convergence sees it as the lower
endpoint rather than inside the open re-emission window. The endpoint is safe
to answer for only when the newest staging delivery is waiting for that exact
subject and the live-lane binding has already been established by the caller.
"""

from __future__ import annotations

import contextlib
import io
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts.ci import lab_pass_receipt as receipts

pytestmark = pytest.mark.unit

QUEUED_SHA = "a" * 40
LATER_SHA = "b" * 40
NOW = datetime(2026, 9, 25, 17, 0, tzinfo=UTC)
WORKFLOW = (
    Path(__file__).resolve().parents[3]
    / ".github"
    / "workflows"
    / "runtime-rebuild-trigger.yml"
)


def _delivery(
    *,
    subject: str = QUEUED_SHA,
    verdict_kind: str = "ABSENT",
    first_read_at: datetime | None = None,
    created_at: datetime | None = None,
) -> receipts.ModelDeliveryRun:
    first_read = first_read_at or NOW - timedelta(minutes=20)
    return receipts.ModelDeliveryRun(
        run_id=42,
        created_at=created_at or NOW - timedelta(minutes=20),
        status="completed",
        conclusion="failure",
        run_attempt=1,
        verdict={
            "schema": receipts.VERDICT_SCHEMA,
            "run_id": 42,
            "run_attempt": 1,
            "subject": subject,
            "token": verdict_kind,
            "first_read_at": first_read.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    )


def _eligibility(*runs: receipts.ModelDeliveryRun) -> str:
    return receipts.endpoint_reemit_eligibility(
        QUEUED_SHA,
        repo="OmniNode-ai/omnibase_infra",
        workflow="deliver-dev-candidate-to-staging.yml",
        branch="dev",
        bound_seconds=14_400,
        now=NOW,
        read_runs=lambda _repo, _workflow, _branch: list(runs),
    )


class TestTimeoutThenDeploy:
    def test_the_exact_head_a_delivery_is_waiting_on_becomes_eligible(self) -> None:
        assert _eligibility(_delivery()) == "reemit:absent-endpoint-delivery-waiting"

    def test_the_later_convergence_can_write_a_bound_pass_for_the_queued_head(
        self,
    ) -> None:
        assert _eligibility(_delivery()) == "reemit:absent-endpoint-delivery-waiting"
        source = receipts.build_receipt(
            sha=LATER_SHA,
            lane=receipts.EnumLabLane.COMPOSE_DEV,
            started_at=NOW,
            finished_at=NOW + timedelta(minutes=5),
            checks=[
                receipts.ModelLabPassCheck(
                    name="deployed_revision",
                    ok=True,
                    evidence=f"lane at {LATER_SHA}, containing {QUEUED_SHA}",
                ),
                receipts.ModelLabPassCheck(
                    name="ready_main", ok=True, evidence="GET /ready -> 200"
                ),
            ],
            agent_command_id=None,
        )

        recovered = receipts.reemit_receipt(
            source,
            QUEUED_SHA,
            converged_via=f"{LATER_SHA} via container-revision",
        )

        assert recovered.sha == QUEUED_SHA
        assert recovered.result is receipts.EnumLabPassResult.PASS
        assert recovered.converged_via == f"{LATER_SHA} via container-revision"

    def test_a_later_failed_probe_is_preserved_as_fail_for_the_queued_head(
        self,
    ) -> None:
        assert _eligibility(_delivery()) == "reemit:absent-endpoint-delivery-waiting"
        source = receipts.build_receipt(
            sha=LATER_SHA,
            lane=receipts.EnumLabLane.COMPOSE_DEV,
            started_at=NOW,
            finished_at=NOW + timedelta(minutes=5),
            checks=[
                receipts.ModelLabPassCheck(
                    name="deployed_revision",
                    ok=True,
                    evidence=f"lane at {LATER_SHA}, containing {QUEUED_SHA}",
                ),
                receipts.ModelLabPassCheck(
                    name="ready_main", ok=False, evidence="GET /ready -> 503"
                ),
            ],
            agent_command_id=None,
        )

        recovered = receipts.reemit_receipt(
            source,
            QUEUED_SHA,
            converged_via=f"{LATER_SHA} via container-revision",
        )

        assert recovered.sha == QUEUED_SHA
        assert recovered.result is receipts.EnumLabPassResult.FAIL
        assert [check.name for check in recovered.checks if not check.ok] == [
            "ready_main"
        ]


class TestEndpointRecoveryStaysFailClosed:
    def test_a_delivery_for_another_subject_does_not_open_the_endpoint(self) -> None:
        assert _eligibility(_delivery(subject="c" * 40)).startswith("skip:")

    def test_a_real_health_failure_does_not_open_the_endpoint(self) -> None:
        assert _eligibility(_delivery(verdict_kind="FAIL")).startswith("skip:")

    def test_an_expired_delivery_does_not_open_the_endpoint(self) -> None:
        assert _eligibility(
            _delivery(first_read_at=NOW - timedelta(seconds=14_401))
        ).startswith("skip:")

    def test_a_newer_delivery_for_another_head_keeps_the_old_one_closed(self) -> None:
        old = _delivery(created_at=NOW - timedelta(minutes=20))
        newer = _delivery(
            subject="c" * 40,
            created_at=NOW - timedelta(minutes=5),
        )
        assert _eligibility(old, newer).startswith("skip:")

    def test_an_unreadable_delivery_surface_keeps_the_endpoint_closed(self) -> None:
        def _unreadable(
            _repo: str, _workflow: str, _branch: str
        ) -> list[receipts.ModelDeliveryRun]:
            raise receipts.ReceiptLookupError("delivery API unavailable")

        decision = receipts.endpoint_reemit_eligibility(
            QUEUED_SHA,
            repo="OmniNode-ai/omnibase_infra",
            workflow="deliver-dev-candidate-to-staging.yml",
            branch="dev",
            bound_seconds=14_400,
            now=NOW,
            read_runs=_unreadable,
        )
        assert decision.startswith("skip:endpoint-delivery-unreadable")


def test_the_cli_wires_the_endpoint_subject_and_delivery_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def _decision(
        subject: str,
        *,
        repo: str,
        workflow: str,
        branch: str,
        bound_seconds: float,
        now: datetime,
    ) -> str:
        observed.update(
            subject=subject,
            repo=repo,
            workflow=workflow,
            branch=branch,
            bound_seconds=bound_seconds,
            now=now,
        )
        return "reemit:absent-endpoint-delivery-waiting"

    monkeypatch.setattr(receipts, "endpoint_reemit_eligibility", _decision)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = receipts.main(
            [
                "reemit-eligible",
                "--endpoint",
                "--endpoint-subject",
                QUEUED_SHA,
                "--repo",
                "OmniNode-ai/omnibase_infra",
                "--delivery-workflow",
                "deliver-dev-candidate-to-staging.yml",
                "--delivery-branch",
                "dev",
                "--overall-bound-seconds",
                "14400",
            ]
        )

    assert code == 0
    assert out.getvalue().strip() == "reemit:absent-endpoint-delivery-waiting"
    assert observed == {
        "subject": QUEUED_SHA,
        "repo": "OmniNode-ai/omnibase_infra",
        "workflow": "deliver-dev-candidate-to-staging.yml",
        "branch": "dev",
        "bound_seconds": 14_400,
        "now": observed["now"],
    }
    assert isinstance(observed["now"], datetime)


def test_reemission_runs_after_a_completed_failed_verify_that_wrote_a_receipt() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    job = text.split("  reemit-queued-receipts:", 1)[1].split(
        "\n  rerun-refused-deliveries:", 1
    )[0]
    condition = job.split("    runs-on:", 1)[0]

    assert "always()" in condition
    assert "needs.verify-lane-converged.result != 'skipped'" in condition
    assert "needs.verify-lane-converged.result != 'cancelled'" in condition
    assert "needs.verify-lane-converged.result == 'success'" not in condition
