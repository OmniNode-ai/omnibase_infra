# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20073/OMN-20074: repo-owned evidence caller and its S6 part 1 trigger."""

from __future__ import annotations

import re
import shlex
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from scripts.ci.ci_summary_gate import (
    EXPECTED_EXTERNAL_CONTEXTS,
    EXTERNAL_SWEEP_EXCLUSIONS,
    SWEEP_NON_PR_EVENTS,
    check_run_event_index,
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
    # OMN-20074: the base-branch trigger makes GitHub read this definition from
    # the base branch, so a pull request cannot edit what judges it. The operator
    # ruled (2026-10-08T22:41:24Z) that this one file is the single named
    # exception to the OMN-15699 ban in scripts/audit-runner-routing.py.
    assert set(triggers) == {"pull_request_target"}, (
        "caller must only use the base-branch pull request trigger"
    )
    target = triggers["pull_request_target"]
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
    assert version >= (0, 4, 305), (
        "verifier-version must ship omnimarket#3563 (omnimarket v0.4.305), "
        "the release the receipt-gate pin's contract-home step expects"
    )


def test_caller_pins_the_s6_part1_inputs() -> None:
    job = _job()
    assert job["uses"].endswith("@4e4f5e0404d364e296c5d37db6aa3ab9e07f8ffa")
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


def _sweep(
    rows: list[dict[str, Any]], event: str = "pull_request"
) -> tuple[list[str], list[str], list[str]]:
    """Combine the sweep for verify with the registered dod-verify evaluation."""
    runs = [
        {"id": int(match.group(1)), "event": event}
        for row in rows
        if (match := re.search(r"/runs/(\d+)/job/", str(row["html_url"])))
    ]
    failures, _in_flight, swept, _excluded, _provisional = evaluate_external_sweep(
        rows,
        expected=EXPECTED_EXTERNAL_CONTEXTS,
        in_run_names=frozenset(),
        self_name="CI Summary",
        exclusions=EXTERNAL_SWEEP_EXCLUSIONS,
        events=check_run_event_index(runs),
        now=_NOW,
    )
    # OMN-20074 registers dod-verify, so CI Summary's expected-context layer
    # judges its presence/conclusion and the external sweep only judges verify.
    expected = (_DOD_VERIFY,)
    expected_failures, unresolved = evaluate_external_contexts(
        rows, expected=expected, now=_NOW
    )
    assert unresolved == [], "the caller fixtures must have decided expected contexts"
    judged = [name for name in expected if name not in unresolved]
    return failures + expected_failures, swept, judged


def test_ci_summary_sweep_accepts_the_caller_shape() -> None:
    """Green caller rows pass both layers now that dod-verify is registered."""
    assert {"pull_request", "pull_request_target"}.isdisjoint(SWEEP_NON_PR_EVENTS), (
        "the sweep judges the caller's pull_request and pull_request_target rows"
    )
    assert _DOD_VERIFY in EXPECTED_EXTERNAL_CONTEXTS
    assert _VERIFY not in EXPECTED_EXTERNAL_CONTEXTS
    assert not {_VERIFY, _DOD_VERIFY} & set(EXTERNAL_SWEEP_EXCLUSIONS), (
        "caller checks must pass their evaluation on their own conclusion"
    )
    failures, swept, expected_judged = _sweep(
        _caller_rows(verify="success", dod_verify="success")
    )
    assert failures == [], failures
    assert swept == [_VERIFY], "only the unregistered verify row is swept"
    assert expected_judged == [_DOD_VERIFY]


@pytest.mark.parametrize(
    ("verify", "dod_verify", "refused"),
    [
        pytest.param("success", "success", None, id="green"),
        pytest.param("skipped", "success", _VERIFY, id="skipped-verify"),
        pytest.param("success", "failure", _DOD_VERIFY, id="red-dod-verify"),
    ],
)
def test_ci_summary_sweep_judges_the_base_branch_trigger_rows(
    verify: str, dod_verify: str, refused: str | None
) -> None:
    """Base-branch rows use the sweep for verify and expected contexts for dod-verify."""
    failures, swept, expected_judged = _sweep(
        _caller_rows(verify=verify, dod_verify=dod_verify), "pull_request_target"
    )
    assert swept == [_VERIFY], "only the unregistered verify row is swept"
    assert expected_judged == [_DOD_VERIFY]
    if refused is None:
        assert failures == [], failures
    else:
        assert len(failures) == 1, failures
        assert failures[0].startswith(refused), failures


@pytest.mark.parametrize(
    ("verify", "dod_verify", "refused"),
    [
        pytest.param("skipped", "success", _VERIFY, id="skipped-verify"),
        pytest.param("success", "failure", _DOD_VERIFY, id="red-dod-verify"),
    ],
)
def test_ci_summary_sweep_refuses_a_skipped_verify_and_a_red_dod_verify(
    verify: str, dod_verify: str, refused: str
) -> None:
    """The sweep refuses skipped verify; expected contexts refuse red dod-verify."""
    failures, swept, expected_judged = _sweep(
        _caller_rows(verify=verify, dod_verify=dod_verify)
    )
    assert swept == [_VERIFY]
    assert expected_judged == [_DOD_VERIFY]
    assert len(failures) == 1, failures
    assert failures[0].startswith(refused), failures


@pytest.mark.live_contact("tests/ci/fixtures/omn20074_repo_evidence_check_runs.json")
def test_ci_summary_sweep_accepts_the_recorded_caller_rows_of_merged_heads(
    recorded_response: dict[str, object],
) -> None:
    """All 16 merged heads pass the sweep plus registered dod-verify evaluation."""
    heads = cast("list[dict[str, Any]]", recorded_response["heads"])
    assert len(heads) == 16
    for head in heads:
        rows = cast("list[dict[str, Any]]", head["check_runs"])
        assert {row["name"] for row in rows} == {_VERIFY, _DOD_VERIFY}, head["pr"]
        failures, swept, expected_judged = _sweep(rows)
        assert failures == [], (head["pr"], failures)
        assert swept == [_VERIFY], head["pr"]
        assert expected_judged == [_DOD_VERIFY], head["pr"]


def test_repo_evidence_replaces_the_companion_contract_compliance_job() -> None:
    """OMN-20074: the DoD check runs from this repository's contracts only."""
    from scripts.ci.ci_summary_gate import GATE_JOBS

    assert "Contract Compliance Check" not in GATE_JOBS
    assert _DOD_VERIFY in set(EXPECTED_EXTERNAL_CONTEXTS)


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
