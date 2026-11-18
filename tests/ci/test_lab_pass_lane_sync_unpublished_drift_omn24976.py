# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-24976 -- a lane census drift the receipt does not publish is a FAILED lane_sync.

``check_lane_sync`` runs ``scripts/lane-census-check.sh`` report-only: it names
no broker container, so the census never publishes a drift event from the
receipt job. Since OMN-20798 the census exits ``8`` (drift detected, event not
published) rather than ``30`` in exactly that case. The check accepted only
``0`` and ``30``, so every real drift reached the receipt as ``lane_sync``
INDETERMINATE. The original receipt checks used for the candidates retain that
exit and the drift findings, alongside the train's recorded decisions.

The fixture labels the later census re-run separately from the original receipt
checks. The regressions derive a census plan from each original check's logged
findings and replay its recorded exit through the check and the train reader.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.ci import release_train as rt

# The train reads receipts through its own import of the receipt module, and its
# reader compares enums by identity, so every model here comes from that import.
_lab = rt.lab_pass_receipt
LANE_SYNC_CHECK = _lab.LANE_SYNC_CHECK
EnumLabLane = _lab.EnumLabLane
EnumLabPassCheckOutcome = _lab.EnumLabPassCheckOutcome
EnumLabPassResult = _lab.EnumLabPassResult
ModelLabPassCheck = _lab.ModelLabPassCheck
build_receipt = _lab.build_receipt
check_lane_sync = _lab.check_lane_sync

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO_ROOT
    / "tests"
    / "ci"
    / "fixtures"
    / "omn24976_lane_sync_unpublished_drift.json"
)
CANDIDATE = "0e5f656d03b70515dfdf418347c98ca0e833d188"
T0 = datetime(2026, 11, 16, 6, 26, 43, tzinfo=UTC)


class _ReplayRunner:
    """Answers each call with the next recorded response, in order."""

    def __init__(self, responses: Sequence[Mapping[str, Any]]) -> None:
        self._responses = list(responses)
        self.envs: list[Mapping[str, str] | None] = []

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        env: Mapping[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        self.envs.append(env)
        answer = self._responses.pop(0)
        return subprocess.CompletedProcess(
            list(argv),
            int(answer["returncode"]),
            str(answer["stdout"]),
            str(answer["stderr"]),
        )


def _recorded() -> dict[str, Any]:
    loaded = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _replay(census_rc: int, plan: Mapping[str, Any]) -> Any:
    responses = [
        {"returncode": 0, "stdout": "lab-host-a\n", "stderr": ""},
        {"returncode": census_rc, "stdout": json.dumps(plan), "stderr": ""},
    ]
    return check_lane_sync(runner=_ReplayRunner(responses))


def _candidate_replay(document: Mapping[str, Any], repo: str) -> Any:
    """Derive the plan from the original receipt's stderr, not the later re-run."""
    recorded = document["candidate_receipts"][repo]
    evidence = recorded["lane_sync"]["evidence"]
    exit_match = re.search(r"census exit=(-?\d+):", evidence)
    assert exit_match is not None, evidence
    assert int(exit_match[1]) == 8, evidence
    host_match = re.search(r"Starting \(lane=ALL, host=([^,]+),", evidence)
    assert host_match is not None, evidence
    findings = [
        dict(
            zip(("severity", "kind", "container", "lane", "detail"), match, strict=True)
        )
        for match in re.findall(
            r"^  \[(\w+)\] (\S+) (\S+) \(([^)]+)\): ([^\n]+)",
            evidence,
            re.MULTILINE,
        )
    ]
    assert len(findings) == 4, evidence
    decision = document["_provenance"]["train_decisions"][repo]
    assert recorded["candidate_sha"] == decision["candidate"]
    assert recorded["sha"] == decision["receipt_sha"]
    assert decision["reason"] == "lab_receipt_indeterminate"
    assert recorded["lane_sync"]["outcome"] == "indeterminate"
    return _replay(
        int(exit_match[1]),
        {
            "host": host_match[1],
            "lanes_checked": sorted({finding["lane"] for finding in findings}),
            "findings": findings,
            "has_drift": bool(findings),
        },
    )


def _train_reason(
    lane_sync: Any,
    *,
    repo: str = "omnibase_infra",
    candidate: str = CANDIDATE,
) -> tuple[Any, str]:
    checks = (
        ModelLabPassCheck(name="ready_main", ok=True, evidence="GET /ready -> 200"),
        ModelLabPassCheck(name="ready_effects", ok=True, evidence="GET /ready -> 200"),
        lane_sync,
    )
    receipt = build_receipt(
        candidate,
        EnumLabLane.COMPOSE_DEV,
        T0,
        T0 + timedelta(minutes=28),
        checks,
        "4a1b5dee-fda7-4af8-843d-7b70ecb57ce4",
    )
    assert (receipt.result is EnumLabPassResult.PASS) is lane_sync.ok
    return rt.classify_lab_receipt(
        repo,
        candidate,
        list_artifacts=lambda repo, name: [
            {"id": 1, "created_at": "2026-11-16T06:58:05Z"}
        ],
        download_receipt=lambda repo, artifact_id: receipt,
        lanes=(EnumLabLane.COMPOSE_DEV,),
    )


def test_the_check_names_no_broker_so_drift_is_never_published(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANE_MEMORY_BROKER_CONTAINER", "ambient")
    runner = _ReplayRunner(_recorded()["responses"])

    def run(argv: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return runner(argv, timeout=kwargs["timeout"], env=kwargs["env"])

    monkeypatch.setattr(_lab.subprocess, "run", run)
    check = check_lane_sync()
    assert check.outcome is EnumLabPassCheckOutcome.FAIL, check.evidence
    census_env = runner.envs[-1]
    assert census_env is not None
    assert census_env["LANE_MEMORY_BROKER_CONTAINER"] == ""
    assert census_env["KAFKA_BOOTSTRAP_SERVERS"] == ""


@pytest.mark.live_contact("tests/ci/fixtures/omn24976_lane_sync_unpublished_drift.json")
@pytest.mark.parametrize("repo", ["omnibase_infra", "omnimarket"])
def test_the_recorded_unpublished_drift_is_a_failed_lane_sync(
    repo: str, recorded_response: dict[str, Any]
) -> None:
    check = _candidate_replay(recorded_response, repo)
    assert check.name == LANE_SYNC_CHECK
    assert check.outcome is EnumLabPassCheckOutcome.FAIL, check.evidence
    assert not check.ok
    plan = json.loads(check.evidence)
    assert plan["has_drift"] is True
    assert len(plan["findings"]) == 4
    assert {f["lane"] for f in plan["findings"]} == {"lane-a"}


@pytest.mark.live_contact("tests/ci/fixtures/omn24976_lane_sync_unpublished_drift.json")
@pytest.mark.parametrize("repo", ["omnibase_infra", "omnimarket"])
def test_the_train_names_lane_sync_as_a_failed_check(
    repo: str, recorded_response: dict[str, Any]
) -> None:
    reason, detail = _train_reason(
        _candidate_replay(recorded_response, repo),
        repo=repo,
        candidate=recorded_response["candidate_receipts"][repo]["sha"],
    )
    assert reason is rt.EnumTrainReason.LAB_RECEIPT_FAIL, detail
    assert "checks not passing: lane_sync" in detail


def test_exit_8_without_findings_stays_indeterminate() -> None:
    """Negative control: an 8 whose plan has no findings disagrees with itself."""
    plan = {
        "host": "lab-host-a",
        "lanes_checked": ["dev"],
        "findings": [],
        "has_drift": False,
    }
    check = _replay(8, plan)
    assert check.outcome is EnumLabPassCheckOutcome.INDETERMINATE
    reason, _detail = _train_reason(check)
    assert reason is rt.EnumTrainReason.LAB_RECEIPT_INDETERMINATE


def test_exit_0_with_findings_stays_indeterminate() -> None:
    plan = _recorded()["responses"][1]["stdout"]
    check = _replay(0, json.loads(plan))
    assert check.outcome is EnumLabPassCheckOutcome.INDETERMINATE


def test_a_published_drift_still_fails_and_a_clean_census_passes() -> None:
    drift = json.loads(_recorded()["responses"][1]["stdout"])
    assert _replay(30, drift).outcome is EnumLabPassCheckOutcome.FAIL
    clean = {
        "host": "lab-host-a",
        "lanes_checked": ["dev"],
        "findings": [],
        "has_drift": False,
    }
    passed = _replay(0, clean)
    assert passed.ok, passed.evidence
    assert _train_reason(passed)[0] is None


@pytest.mark.parametrize("newest_passes", [True, False])
@pytest.mark.parametrize("newest_listed_first", [True, False])
def test_lab_receipt_fail_is_superseded_by_a_newer_pass(
    newest_passes: bool,
    newest_listed_first: bool,
) -> None:
    """The newest artifact decides the same candidate, in either list order."""
    drift = json.loads(_recorded()["responses"][1]["stdout"])
    clean = {**drift, "findings": [], "has_drift": False}

    def receipt(passes: bool) -> Any:
        return build_receipt(
            CANDIDATE,
            EnumLabLane.COMPOSE_DEV,
            T0,
            T0 + timedelta(minutes=28),
            (
                ModelLabPassCheck(name="ready_main", ok=True, evidence="ready"),
                ModelLabPassCheck(name="ready_effects", ok=True, evidence="ready"),
                _replay(0 if passes else 8, clean if passes else drift),
            ),
            "4a1b5dee-fda7-4af8-843d-7b70ecb57ce4",
        )

    receipts = {1: receipt(not newest_passes), 2: receipt(newest_passes)}
    artifacts = [{"id": 1, "created_at": "2026-11-16T06:58:05Z"}]
    downloaded: list[int] = []

    def download(repo: str, artifact_id: int) -> Any:
        downloaded.append(artifact_id)
        return receipts[artifact_id]

    def classify() -> tuple[Any, str]:
        return rt.classify_lab_receipt(
            "omnibase_infra",
            CANDIDATE,
            list_artifacts=lambda repo, name: list(artifacts),
            download_receipt=download,
            lanes=(EnumLabLane.COMPOSE_DEV,),
        )

    assert classify()[0] is (
        rt.EnumTrainReason.LAB_RECEIPT_FAIL if newest_passes else None
    )
    artifacts.append({"id": 2, "created_at": "2026-11-16T07:58:05Z"})
    if newest_listed_first:
        artifacts.reverse()
    reason, detail = classify()
    assert reason is (None if newest_passes else rt.EnumTrainReason.LAB_RECEIPT_FAIL), (
        detail
    )
    assert downloaded == [1, 2]


@pytest.mark.parametrize("census_rc", [1, 2, 3, 4, 5, 6, 7, 31, 124, 127, -9])
def test_every_other_census_exit_stays_indeterminate(census_rc: int) -> None:
    drift = json.loads(_recorded()["responses"][1]["stdout"])
    assert _replay(census_rc, drift).outcome is EnumLabPassCheckOutcome.INDETERMINATE
