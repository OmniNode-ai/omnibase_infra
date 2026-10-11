# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Persistent apply evidence, exact delivery selection and kind smoke isolation."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
import yaml

from scripts.ci import fetch_lab_overlay_record as overlay
from scripts.ci import lab_pass_receipt as lab
from tests.scripts.ci._lab_pass_fixtures import (
    REPO,
    SHA_3E4A,
    SHA_4ACA,
    FakeSurface,
    receipt,
)
from tests.scripts.ci.test_lab_pass_rerun_selector import NOW, _run, _verdict

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
K3S = lab.EnumLabLane.ONEX_LAB_K3S


def _workflow(name: str) -> dict:
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text())


def test_persistent_apply_receipt_names_its_source_host() -> None:
    url = f"http://persistent-lab:8098/lab-overlay/{SHA_4ACA}"
    checks = overlay.checks_from_record(
        {
            "sha": SHA_4ACA,
            "checks": [
                {
                    "name": "lab_overlay_applied",
                    "ok": True,
                    "evidence": "apply_lab_lane.sh exited 0",
                }
            ],
        },
        sha=SHA_4ACA,
        url=url,
    )
    emitted = lab.build_receipt(
        sha=SHA_4ACA,
        lane=K3S,
        started_at=NOW,
        finished_at=NOW,
        checks=[lab.ModelLabPassCheck(**c) for c in checks],
        agent_command_id=None,
    )
    assert emitted.result is lab.EnumLabPassResult.PASS
    assert any(
        "persistent-lab" in c.evidence and SHA_4ACA in c.evidence
        for c in emitted.checks
    )


def test_descendant_apply_evidence_names_the_same_persistent_host() -> None:
    checks = overlay.resolve_via_latest(
        base_url="http://persistent-lab:8098",
        repo=REPO,
        sha=SHA_4ACA,
        request_timeout_seconds=1,
        fetch=lambda *_: (
            200,
            json.dumps(
                {
                    "sha": SHA_3E4A,
                    "checks": [
                        {
                            "name": "lab_overlay_applied",
                            "ok": True,
                            "evidence": "apply record",
                        }
                    ],
                }
            ),
        ),
        compare=lambda *_: overlay.RELATION_DESCENDANT,
    )
    assert checks and checks[0]["ok"] is True
    assert "persistent-lab" in checks[0]["evidence"]
    assert SHA_3E4A in checks[0]["evidence"]


@pytest.mark.parametrize("state", ["absent", "fail", "wrong-sha", "unreadable", "pass"])
def test_delivery_requires_persistent_apply_for_exact_sha(
    monkeypatch, tmp_path, state
) -> None:
    surface = FakeSurface(receipts=[receipt(SHA_4ACA)])
    if state in {"fail", "pass"}:
        surface.add(receipt(SHA_4ACA, K3S, outcome="fail" if state == "fail" else "ok"))
    elif state == "wrong-sha":
        surface.add(receipt(SHA_3E4A, K3S))
    elif state == "unreadable":
        surface.extra_artifacts[lab.artifact_name(K3S, SHA_4ACA)] = (
            "receipt.json",
            "broken",
        )
    monkeypatch.setattr(lab, "_gh_api", surface)
    verdict = tmp_path / "verdict.json"
    result = lab.evaluate_gate(
        REPO,
        SHA_4ACA,
        [K3S],
        io.StringIO(),
        required=[lab.EnumLabLane.COMPOSE_DEV],
        verdict_out=verdict,
    )
    assert result == (0 if state == "pass" else 1)
    body = json.loads(verdict.read_text())
    assert body["exact_lanes"] == {K3S.value: SHA_4ACA}
    assert (
        body["lanes"][K3S.value]
        == {
            "absent": "ABSENT",
            "fail": "FAIL",
            "wrong-sha": "ABSENT",
            "unreadable": "UNREADABLE",
            "pass": "PASS",
        }[state]
    )


def test_delivery_workflow_selects_persistent_receipt_without_inheriting_it() -> None:
    job = _workflow("deliver-dev-candidate-to-staging.yml")["jobs"]["lab-pass-gate"]
    step = next(s for s in job["steps"] if s.get("id") == "own-sha-gate")
    assert "--lane onex-lab-k3s" in step["run"]
    assert '--sha "${GITHUB_SHA}"' in step["run"]
    assert "--require-lane compose-dev" in step["run"]
    assert "--resolve-runtime-ancestor" in step["run"]
    assert "if" not in step


@pytest.mark.parametrize("persistent_sha", [SHA_3E4A, SHA_4ACA])
def test_compose_subject_inheritance_never_inherits_the_persistent_receipt(
    monkeypatch, persistent_sha
) -> None:
    surface = FakeSurface(receipts=[receipt(SHA_3E4A), receipt(persistent_sha, K3S)])
    monkeypatch.setattr(lab, "_gh_api", surface)
    result = lab.evaluate_gate(
        REPO,
        SHA_4ACA,
        [K3S],
        io.StringIO(),
        required=[lab.EnumLabLane.COMPOSE_DEV],
        required_sha=SHA_3E4A,
    )
    assert result == (0 if persistent_sha == SHA_4ACA else 1)


def test_kind_boot_is_named_smoke_and_cannot_supply_a_lab_pass(monkeypatch) -> None:
    boot = _workflow("deliver-dev-candidate-to-staging.yml")["jobs"][
        "candidate-boot-gate"
    ]
    assert "smoke" in boot["name"].lower()
    emit = next(
        s for s in boot["steps"] if "lab_pass_receipt.py emit" in s.get("run", "")
    )
    assert "--lane candidate-boot" in emit["run"]
    upload = next(
        s
        for s in boot["steps"]
        if s.get("with", {}).get("name", "").startswith("candidate-boot-receipt-")
    )
    smoke = lab.EnumLabLane.CANDIDATE_BOOT
    assert upload["with"]["name"] == f"{lab.artifact_name(smoke, '${{ github.sha }}')}"
    assert smoke not in lab.ANY_OF_DEFAULT_LANES
    assert lab.EnumLabLane.ONEX_LAB not in lab.ANY_OF_DEFAULT_LANES
    monkeypatch.setattr(
        lab, "_gh_api", FakeSurface(receipts=[receipt(SHA_4ACA, smoke)])
    )
    assert lab.evaluate_gate(REPO, SHA_4ACA, [smoke], io.StringIO()) == 1
    assert (
        lab.evaluate_gate(REPO, SHA_4ACA, lab.ANY_OF_DEFAULT_LANES, io.StringIO()) == 1
    )
    historical_kind = lab.EnumLabLane.ONEX_LAB
    monkeypatch.setattr(
        lab, "_gh_api", FakeSurface(receipts=[receipt(SHA_4ACA, historical_kind)])
    )
    assert lab.evaluate_gate(REPO, SHA_4ACA, [historical_kind], io.StringIO()) == 1


def test_candidate_smoke_emit_roundtrip_never_uses_a_lab_artifact_name(
    tmp_path,
) -> None:
    target = tmp_path / "receipt.json"
    assert (
        lab.main(
            [
                "emit",
                "--sha",
                SHA_4ACA,
                "--lane",
                "candidate-boot",
                "--started-at",
                "2026-09-22T19:00:00Z",
                "--finished-at",
                "2026-09-22T19:47:04Z",
                "--check",
                "manifests_render:ok:kind render/wiring smoke",
                "--out",
                str(target),
            ]
        )
        == 0
    )
    parsed = lab.ModelLabPassReceipt.from_json(target.read_text())
    assert parsed.lane is lab.EnumLabLane.CANDIDATE_BOOT
    assert (
        lab.artifact_name(parsed.lane, parsed.sha)
        == f"candidate-boot-receipt-{SHA_4ACA}"
    )


@pytest.mark.parametrize("state", ["absent", "fail", "pass"])
def test_reread_waits_for_persistent_pass(monkeypatch, state) -> None:
    verdict = _verdict(run_id=900, refusal="ABSENT")
    verdict["exact_lanes"] = {K3S.value: SHA_4ACA}
    verdict["lanes"][K3S.value] = "ABSENT"
    surface = FakeSurface(receipts=[receipt(SHA_4ACA)])
    if state != "absent":
        surface.add(receipt(SHA_4ACA, K3S, outcome="fail" if state == "fail" else "ok"))
    monkeypatch.setattr(lab, "_gh_api", surface)
    monkeypatch.setattr(
        lab, "read_delivery_runs", lambda *_: [_run(900, verdict=verdict)]
    )
    posted = []
    monkeypatch.setattr(lab, "_gh_api_post", posted.append)
    assert (
        lab.rerun_refused_deliveries(
            REPO,
            [SHA_4ACA],
            workflow="deliver-dev-candidate-to-staging.yml",
            branch="dev",
            bound_seconds=14400,
            out=io.StringIO(),
            now=lambda: NOW,
        )
        == 0
    )
    assert bool(posted) is (state == "pass")


def test_reread_is_ordered_after_overlay_receipt_upload() -> None:
    job = _workflow("runtime-rebuild-trigger.yml")["jobs"]["rerun-refused-deliveries"]
    assert "verify-lab-overlay-converged" in job["needs"]


@pytest.mark.parametrize("state", ["fail", "unreadable"])
def test_a_persistent_health_or_read_failure_is_never_retried_as_timing(
    monkeypatch, tmp_path, state
) -> None:
    surface = FakeSurface(receipts=[receipt(SHA_4ACA)])
    if state == "fail":
        surface.add(receipt(SHA_4ACA, K3S, outcome="fail"))
    else:
        surface.extra_artifacts[lab.artifact_name(K3S, SHA_4ACA)] = (
            "receipt.json",
            "broken",
        )
    monkeypatch.setattr(lab, "_gh_api", surface)
    verdict = tmp_path / "verdict.json"
    assert (
        lab.evaluate_gate(
            REPO,
            SHA_4ACA,
            [K3S],
            io.StringIO(),
            required=[lab.EnumLabLane.COMPOSE_DEV],
            verdict_out=verdict,
            run_id=900,
            run_attempt=1,
            now=lambda: NOW,
        )
        == 1
    )
    body = json.loads(verdict.read_text())
    monkeypatch.setattr(lab, "read_delivery_runs", lambda *_: [_run(900, verdict=body)])
    monkeypatch.setattr(
        lab,
        "_gh_api",
        FakeSurface(receipts=[receipt(SHA_4ACA), receipt(SHA_4ACA, K3S)]),
    )
    posted = []
    monkeypatch.setattr(lab, "_gh_api_post", posted.append)
    assert (
        lab.rerun_refused_deliveries(
            REPO,
            [SHA_4ACA],
            workflow="deliver-dev-candidate-to-staging.yml",
            branch="dev",
            bound_seconds=14400,
            out=io.StringIO(),
            now=lambda: NOW,
        )
        == 0
    )
    assert posted == []


def test_regression_is_wired_into_ci_and_precommit() -> None:
    name = Path(__file__).name
    assert name in (ROOT / ".pre-commit-config.yaml").read_text()
    assert name in (ROOT / ".github/workflows/ci.yml").read_text()
