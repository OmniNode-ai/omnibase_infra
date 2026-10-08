# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20073/OMN-20074: repo-owned evidence and S6 part 1 enforcement."""

from __future__ import annotations

import re
import shlex
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from scripts.ci.ci_summary_gate import (
    EXIT_FAILURE,
    EXIT_PENDING,
    EXIT_SUCCESS,
    EXPECTED_EXTERNAL_CONTEXTS,
    EXTERNAL_SWEEP_EXCLUSIONS,
    SKIPPABLE_GATE_JOBS,
    STRICT_GATE_JOBS,
    SWEEP_NON_PR_EVENTS,
    check_run_event_index,
    evaluate,
    evaluate_external_contexts,
    evaluate_external_sweep,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CALLER_PATH = REPO_ROOT / ".github" / "workflows" / "call-repo-evidence-gate.yml"

# First release whose wheel ships node_dod_verify occ-difference, omnimarket#3277.
_DIFFERENCE_CLASSIFIER_FLOOR = (0, 4, 294)

_VERIFY = "repo-evidence / verify"
_DOD_VERIFY = "repo-evidence / dod-verify"
_RUN_ID = 424242
_NOW = datetime(2026, 10, 7, 12, 0, 0, tzinfo=UTC)


def _job() -> dict[str, Any]:
    data = yaml.safe_load(CALLER_PATH.read_text(encoding="utf-8"))
    return data["jobs"]["repo-evidence"]


def test_caller_workflow_shape() -> None:
    text = CALLER_PATH.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    # PyYAML 1.1 resolves the bare `on:` key to the boolean True.
    triggers = data.get("on", data.get(True))
    assert isinstance(triggers, dict), "caller must declare a mapping on: block"
    assert "pull_request" in triggers, "caller must run on pull requests"
    # scripts/audit-runner-routing.py bans the base-branch pull request trigger
    # in this repo, so the caller uses plain pull_request.
    assert set(triggers) == {"pull_request"}, "caller must only use pull_request"
    target = triggers["pull_request"]
    assert not {"paths", "paths-ignore"} & set(target), (
        "the registered verdict must report without a paths filter"
    )
    assert target["branches"] == ["dev", "main"], "caller must target dev and main"
    assert target["types"] == [
        "opened",
        "synchronize",
        "reopened",
        "edited",
        "ready_for_review",
    ], "caller must cover the declared PR activity types"
    assert data["permissions"] == {"contents": "read", "pull-requests": "read"}, (
        "caller permissions must be exactly contents: read and pull-requests: read"
    )
    assert set(data["jobs"]) == {"repo-evidence"}, (
        "caller must have one repo-evidence job"
    )
    job = data["jobs"]["repo-evidence"]
    assert re.fullmatch(
        r"OmniNode-ai/omnibase_core/\.github/workflows/receipt-gate\.yml@[0-9a-f]{40}",
        job["uses"],
    ), "receipt-gate reusable must be pinned by an immutable full SHA"
    assert job["with"]["evidence-source"] == "caller", (
        "caller evidence mode is required"
    )
    assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", job["with"]["verifier-version"]), (
        "verifier-version must be a numeric semantic version"
    )
    for key in ("steps", "secrets", "if", "name", "permissions"):
        assert key not in job, f"caller job must not declare {key}"
    assert "secrets: inherit" not in text, "caller must not inherit secrets"


def test_caller_verifier_ships_the_occ_difference_classifier() -> None:
    inputs = _job()["with"]
    version = tuple(int(part) for part in inputs["verifier-version"].split("."))
    assert version >= _DIFFERENCE_CLASSIFIER_FLOOR, (
        "verifier-version must ship node_dod_verify occ-difference "
        f"(>= {'.'.join(map(str, _DIFFERENCE_CLASSIFIER_FLOOR))})"
    )


def test_caller_enforces_after_the_s6_part1_cutover() -> None:
    job = _job()
    assert job["uses"].endswith("@7394003b290a140df6ddf0921a510ca10f642218")
    assert job["with"].get("shadow") == "false"
    assert job["with"].get("compare-with-occ") == "false"


def _caller_rows(*, verify: str | None, dod_verify: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, conclusion in ((_VERIFY, verify), (_DOD_VERIFY, dod_verify)):
        if conclusion is None:
            continue
        rows.append(
            {
                "id": len(rows) + 1,
                "name": name,
                "status": "completed",
                "conclusion": conclusion,
                "started_at": "2026-10-07T11:00:00Z",
                "completed_at": "2026-10-07T11:05:00Z",
                "head_sha": "a" * 40,
                "html_url": (
                    "https://github.com/OmniNode-ai/omnibase_infra/actions/runs/"
                    f"{_RUN_ID}/job/{len(rows) + 1}"
                ),
            }
        )
    return rows


def _sweep(rows: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    failures, _in_flight, swept, _excluded, _provisional = evaluate_external_sweep(
        rows,
        expected=EXPECTED_EXTERNAL_CONTEXTS,
        in_run_names=frozenset(),
        self_name="CI Summary",
        exclusions=EXTERNAL_SWEEP_EXCLUSIONS,
        events=check_run_event_index([{"id": _RUN_ID, "event": "pull_request"}]),
        now=_NOW,
    )
    return failures, swept


def _ci_summary(rows: list[dict[str, Any]]) -> tuple[int, str]:
    jobs = [
        {"name": name, "status": "completed", "conclusion": "success"}
        for name in (*STRICT_GATE_JOBS, *SKIPPABLE_GATE_JOBS)
    ]
    other_contexts = [
        {"name": name, "status": "completed", "conclusion": "success"}
        for name in EXPECTED_EXTERNAL_CONTEXTS
        if name != _DOD_VERIFY
    ]
    return evaluate(
        jobs,
        check_runs=other_contexts + rows,
        external_contexts=EXPECTED_EXTERNAL_CONTEXTS,
        workflow_runs=[{"id": _RUN_ID, "event": "pull_request"}],
        now=_NOW,
    )


def test_ci_summary_accepts_the_registered_verdict_beside_occ() -> None:
    assert "pull_request" not in SWEEP_NON_PR_EVENTS, (
        "the sweep must judge the caller's pull_request rows"
    )
    assert _DOD_VERIFY in EXPECTED_EXTERNAL_CONTEXTS
    assert _VERIFY not in EXPECTED_EXTERNAL_CONTEXTS
    assert "verify / verify" in EXPECTED_EXTERNAL_CONTEXTS
    assert "occ-preflight / eligibility" in STRICT_GATE_JOBS
    assert "OCC Companion Merged Gate (OMN-15214)" in STRICT_GATE_JOBS
    assert not {_VERIFY, _DOD_VERIFY} & set(EXTERNAL_SWEEP_EXCLUSIONS), (
        "caller checks must not be excluded from enforcement"
    )
    rows = _caller_rows(verify="success", dod_verify="success")
    assert evaluate_external_contexts(rows, (_DOD_VERIFY,), now=_NOW) == ([], [])
    failures, swept = _sweep(rows)
    assert failures == [], failures
    assert swept == [_VERIFY], "the registered verdict belongs to layer 4"
    code, report = _ci_summary(rows)
    assert code == EXIT_SUCCESS, report


@pytest.mark.parametrize(
    ("dod_verify", "expected_code"),
    [
        pytest.param("failure", EXIT_FAILURE, id="red-dod-verify"),
        pytest.param("skipped", EXIT_FAILURE, id="skipped-dod-verify"),
        pytest.param(None, EXIT_PENDING, id="absent-dod-verify"),
    ],
)
def test_ci_summary_refuses_a_red_or_absent_registered_verdict(
    dod_verify: str | None, expected_code: int
) -> None:
    assert _DOD_VERIFY in EXPECTED_EXTERNAL_CONTEXTS
    assert _VERIFY not in EXPECTED_EXTERNAL_CONTEXTS
    rows = _caller_rows(verify="success", dod_verify=dod_verify)
    expected_failures = [] if dod_verify is None else [_DOD_VERIFY]
    expected_unresolved = [_DOD_VERIFY] if dod_verify is None else []
    assert evaluate_external_contexts(rows, (_DOD_VERIFY,), now=_NOW) == (
        expected_failures,
        expected_unresolved,
    )
    # Layer 5 excludes the registered verdict; layer 4 must be load-bearing.
    assert _sweep(rows) == ([], [_VERIFY])
    code, report = _ci_summary(rows)
    assert code == expected_code, report
    if dod_verify is None:
        # Absence holds PENDING, then fails closed at the poller's deadline.
        assert f"external contexts missing/pending: {_DOD_VERIFY}" in report
    else:
        assert f"external-context failures: {_DOD_VERIFY}" in report


@pytest.mark.live_contact("tests/ci/fixtures/omn20074_repo_evidence_check_runs.json")
def test_ci_summary_registered_name_matches_the_recorded_admission_window(
    recorded_response: dict[str, object],
) -> None:
    """The 16 admission heads' recorded check-runs satisfy the registered context."""
    registered = tuple(
        name for name in EXPECTED_EXTERNAL_CONTEXTS if name.startswith("repo-evidence")
    )
    assert registered == (_DOD_VERIFY,)
    heads = cast("list[dict[str, Any]]", recorded_response["heads"])
    assert len(heads) == 16
    for head in heads:
        rows = cast("list[dict[str, object]]", head["check_runs"])
        assert {row["name"] for row in rows} == {_DOD_VERIFY, _VERIFY}, head["pr"]
        assert evaluate_external_contexts(rows, registered) == ([], []), head["pr"]


def test_every_repo_contract_binds_every_criterion() -> None:
    assert CALLER_PATH.is_file(), "repo-owned evidence requires the caller workflow"
    contract_paths = sorted((REPO_ROOT / "contracts").glob("OMN-*.yaml"))
    assert (REPO_ROOT / "contracts" / "OMN-20073.yaml") in contract_paths
    checked = 0
    for path in contract_paths:
        contract = yaml.safe_load(path.read_text(encoding="utf-8"))
        items = [
            item for item in contract.get("dod_evidence", []) if "binds_ac" in item
        ]
        if not items:
            continue
        checked += 1
        criteria = {
            ac["id"]
            for requirement in contract.get("requirements", [])
            for ac in requirement.get("acceptance", [])
        }
        bound: set[str] = set()
        for item in items:
            label = f"{path.name}:{item['id']}"
            bound.update(item["binds_ac"])
            assert "ac_bindings" not in item, f"{label}: use binds_ac, not ac_bindings"
            checks = item.get("checks", [])
            assert checks, f"{label}: binds_ac requires at least one check"
            for check in checks:
                if check.get("check_type") != "test_passes" or not check.get(
                    "check_value", ""
                ).startswith("uv run pytest "):
                    continue
                selector = next(
                    (
                        token
                        for token in shlex.split(check["check_value"])
                        if token.endswith((".py", "/tests"))
                    ),
                    "",
                )
                assert (
                    selector
                    and not Path(selector).is_absolute()
                    and (REPO_ROOT / selector).exists()
                    and (REPO_ROOT / selector).resolve().is_relative_to(REPO_ROOT)
                ), (
                    f"{label}: pytest evidence must name an existing relative path inside the repo"
                )
        assert criteria <= bound, (
            f"{path.name}: acceptance criteria missing binds_ac: {sorted(criteria - bound)}"
        )
    assert checked >= 1, "no contract declares binds_ac"
