# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18488: lab checks preserve author bindings and criterion pins."""

from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.ci import lab_pass_receipt as lab
from tests.scripts.ci._lab_pass_fixtures import REPO, SHA_4ACA, FakeSurface, receipt
from tests.unit.nodes.node_evidence_autoclose_sweep_effect import (
    test_omn_18330_criterion_hash as closer,
)

pytestmark = pytest.mark.unit


def test_a_check_round_trips_its_binding_and_pin() -> None:
    payload = {
        "name": "ready_main",
        "ok": True,
        "evidence": "GET /ready -> 200",
        "binds_ac": ["AC1"],
        "ac_binding_hashes": {"AC1": "a" * 64},
    }
    try:
        check = lab.ModelLabPassCheck.from_dict(payload)
    except ValueError as exc:
        pytest.fail(f"the receipt reader lost a declared criterion binding: {exc}")
    assert check.binds_ac == ("AC1",)
    assert check.to_dict() == payload


def test_a_receipt_refuses_a_label_absent_from_its_ticket() -> None:
    bound = lab.ModelLabPassCheck(
        "ready_main",
        True,
        "GET /ready -> 200",
        binds_ac=("AC2",),
        ac_binding_hashes=(("AC2", "a" * 64),),
    )
    with pytest.raises(ValueError, match="AC2"):
        replace(
            receipt(SHA_4ACA),
            checks=(bound,),
            ticket_id="OMN-18488",
            criterion_labels=("AC1",),
        )
    valid = replace(
        receipt(SHA_4ACA),
        checks=(bound,),
        ticket_id="OMN-18488",
        criterion_labels=("AC2",),
    )
    assert lab.ModelLabPassReceipt.from_json(valid.to_json()) == valid


def test_an_unbound_receipt_preserves_its_wire_form() -> None:
    original = receipt(SHA_4ACA)
    payload = json.loads(original.to_json())
    assert "ticket_id" not in payload
    assert "criterion_labels" not in payload
    assert "binds_ac" not in payload["checks"][0]
    assert lab.ModelLabPassReceipt.from_json(original.to_json()) == original


def _bound_receipt(
    *,
    passing: bool = True,
    labels: tuple[str, ...] = ("AC1",),
    pins: tuple[tuple[str, str], ...] | None = None,
    ticket: str = "OMN-0000",
) -> lab.ModelLabPassReceipt:
    if pins is None:
        pins = (("AC1", closer._ACCEPTED_PIN),) if labels else ()
    check = lab.ModelLabPassCheck(
        "ready_main",
        passing,
        "GET /ready -> 200" if passing else "GET /ready -> 503",
        binds_ac=labels,
        ac_binding_hashes=pins,
    )
    return replace(
        receipt(SHA_4ACA),
        checks=(check,),
        result=lab.EnumLabPassResult.PASS if passing else lab.EnumLabPassResult.FAIL,
        ticket_id=ticket,
        criterion_labels=("AC1",),
    )


@pytest.mark.parametrize(
    "lane", [lab.EnumLabLane.COMPOSE_DEV, lab.EnumLabLane.ONEX_LAB_K3S]
)
def test_both_emitters_copy_authored_bindings_at_the_cited_commit(
    monkeypatch: pytest.MonkeyPatch, lane: lab.EnumLabLane
) -> None:
    reads = []
    contract = {
        "ticket_id": "OMN-18488",
        "requirements": [
            {"acceptance": [{"id": "AC1", "statement": "AC1: the lane is ready."}]}
        ],
        "dod_evidence": [
            {
                "id": f"lab-pass-{lane.value}-ready_main",
                "binds_ac": ["AC1"],
                "ac_bindings": [
                    {
                        "label": "AC1",
                        "criterion_hash": "a" * 64,
                        "accepted_by": "author",
                        "accepted_at": "2026-10-07T19:30:00Z",
                    }
                ],
            }
        ],
    }

    def git_read(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        reads.append(args)
        output = (
            json.dumps(contract)
            if args[-1].endswith(".yaml")
            else "fix(OMN-18488): ready"
        )
        return subprocess.CompletedProcess(args, 0, output, "")

    monkeypatch.setattr(lab.subprocess, "run", git_read)
    checks, ticket, labels = lab.bind_commit_checks(
        SHA_4ACA, lane, receipt(SHA_4ACA).checks, Path()
    )
    result = lab.build_receipt(
        SHA_4ACA,
        lane,
        receipt(SHA_4ACA).started_at,
        receipt(SHA_4ACA).finished_at,
        checks,
        None,
        ticket_id=ticket,
        criterion_labels=labels,
    )
    assert result.checks[0].binds_ac == ("AC1",)
    assert result.checks[0].ac_binding_hashes == (("AC1", "a" * 64),)
    assert reads[-1][-1] == f"{SHA_4ACA}:contracts/OMN-18488.yaml"


def test_a_squash_body_citing_other_tickets_binds_the_subject_ticket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lane = lab.EnumLabLane.COMPOSE_DEV
    reads = []
    contract = {
        "ticket_id": "OMN-18488",
        "requirements": [
            {"acceptance": [{"id": "AC1", "statement": "AC1: the lane is ready."}]}
        ],
        "dod_evidence": [
            {
                "id": f"lab-pass-{lane.value}-ready_main",
                "binds_ac": ["AC1"],
                "ac_bindings": [
                    {
                        "label": "AC1",
                        "criterion_hash": "a" * 64,
                        "accepted_by": "author",
                        "accepted_at": "2026-10-07T19:30:00Z",
                    }
                ],
            }
        ],
    }

    def git_read(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        reads.append(args)
        if args[-1].endswith(".yaml"):
            output = json.dumps(contract)
        elif "--format=%s" in args:
            output = "refactor(OMN-18488): move payloads (#4687)"
        elif "--format=%B" in args:
            output = (
                "refactor(OMN-18488): move payloads (#4687)\n\n"
                "* ci(OMN-18488): x\nLane ticket OMN-17427.\n"
                "CI Evidence Policy (OMN-18247)"
            )
        else:
            pytest.fail(f"unexpected git read: {args}")
        return subprocess.CompletedProcess(args, 0, output, "")

    monkeypatch.setattr(lab.subprocess, "run", git_read)
    checks, ticket, labels = lab.bind_commit_checks(
        SHA_4ACA, lane, receipt(SHA_4ACA).checks, Path()
    )
    assert ticket == "OMN-18488"
    assert checks[0].binds_ac == ("AC1",)
    assert labels == ("AC1",)
    assert reads[-1][-1] == f"{SHA_4ACA}:contracts/OMN-18488.yaml"


def test_a_subject_citing_two_tickets_is_still_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def git_read(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, "fix(OMN-1, OMN-2): both", "")

    monkeypatch.setattr(lab.subprocess, "run", git_read)
    with pytest.raises(ValueError, match="ambiguous"):
        lab.bind_commit_checks(
            SHA_4ACA, lab.EnumLabLane.COMPOSE_DEV, receipt(SHA_4ACA).checks, Path()
        )


def test_an_uncited_commit_emits_unbound_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def git_read(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args, 0, "ordinary commit without a ticket", ""
        )

    monkeypatch.setattr(lab.subprocess, "run", git_read)
    checks, ticket, labels = lab.bind_commit_checks(
        SHA_4ACA, lab.EnumLabLane.COMPOSE_DEV, receipt(SHA_4ACA).checks, Path()
    )
    assert checks == list(receipt(SHA_4ACA).checks)
    assert (ticket, labels) == ("", ())


@pytest.mark.parametrize("passing", [True, False])
def test_artifact_checks_preserve_pins_and_the_whole_receipt_verdict(
    monkeypatch: pytest.MonkeyPatch, passing: bool
) -> None:
    surface = FakeSurface()
    surface.add(_bound_receipt(passing=passing))
    monkeypatch.setattr(lab, "_gh_api", surface)
    checks = lab.receipt_evidence_for_ticket(REPO, SHA_4ACA, "OMN-0000")
    assert len(checks) == 1
    assert checks[0]["status"] == ("verified" if passing else "failed")
    assert checks[0]["ac_binding_hashes"] == {"AC1": closer._ACCEPTED_PIN}
    assert lab.receipt_evidence_for_ticket(REPO, SHA_4ACA, "OMN-18488") == []
    assert lab.receipt_evidence_for_ticket(REPO, "b" * 40, "OMN-0000") == []


def test_a_locally_green_check_in_a_fail_receipt_discharges_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _bound_receipt()
    failed = replace(
        original,
        result=lab.EnumLabPassResult.FAIL,
        checks=(
            *original.checks,
            lab.ModelLabPassCheck("other_probe", False, "probe failed"),
        ),
    )
    surface = FakeSurface()
    surface.add(failed)
    monkeypatch.setattr(lab, "_gh_api", surface)
    checks = lab.receipt_evidence_for_ticket(REPO, SHA_4ACA, "OMN-0000")
    assert checks[0]["status"] == "failed"


@pytest.mark.parametrize(
    "pins", [(("AC1", ""),), (("AC1", "a" * 63),), (("AC2", "a" * 64),)]
)
def test_malformed_or_undeclared_pins_are_refused_at_the_check(pins) -> None:
    with pytest.raises(ValueError, match="ac_binding_hashes"):
        lab.ModelLabPassCheck(
            "ready_main",
            True,
            "GET /ready -> 200",
            binds_ac=("AC1",),
            ac_binding_hashes=pins,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["pass", "fail", "unbound", "stale", "missing-pin"])
async def test_the_full_closer_uses_receipt_bindings_and_keeps_its_pin_gate(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    criterion = (
        closer._REWRITTEN_CRITERION if kind == "stale" else closer._ACCEPTED_CRITERION
    )
    linear = closer.FakeLinear(criterion)
    surface = FakeSurface()
    surface.add(
        _bound_receipt(
            passing=kind != "fail",
            labels=() if kind == "unbound" else ("AC1",),
            pins=() if kind == "missing-pin" else None,
        )
    )
    monkeypatch.setattr(lab, "_gh_api", surface)
    original_gh = closer._gh_fake()

    async def gh_read(args: list[str], timeout: float):
        payload, error = await original_gh(args, timeout)
        if isinstance(payload, dict) and "/pulls/" in args[2]:
            payload["merge_commit_sha"] = SHA_4ACA
        return payload, error

    reads = []

    async def lab_read(repo: str, sha: str, ticket: str, cwd: str, timeout: float):
        reads.append((repo, sha, ticket))
        return lab.receipt_evidence_for_ticket(repo, sha, ticket)

    verdict = closer._receipt(None)
    verdict["result"]["checks"][0]["binds_ac"] = []
    handler = closer.HandlerEvidenceAutocloseSweep(
        linear_client=linear,
        autoclose_disabled=False,
        run_gh_command=gh_read,
        run_dod_verify_command=closer._dod_fake(verdict),
    )
    monkeypatch.setattr(handler, "_run_lab_pass_checks", lab_read, raising=False)
    await handler.handle(closer._request())
    result = await handler.handle(closer._request())
    if kind == "pass":
        assert result.tickets_flipped == 1
        assert linear.state_updates == [("issue-1", "state-done")]
    else:
        assert result.tickets_flipped == 0
        assert linear.state_updates == []
        assert result.outcomes[0].reason
        if kind == "stale":
            assert "changed" in result.outcomes[0].reason.lower()
        if kind == "missing-pin":
            assert "NO readable `criterion_hash`" in result.outcomes[0].reason
    assert reads == [(REPO, SHA_4ACA, "OMN-0000")] * 2


def test_rekeying_a_receipt_never_transfers_another_commits_bindings() -> None:
    source = _bound_receipt()
    rekeyed = lab.reemit_receipt(source, "b" * 40)
    assert rekeyed.ticket_id == ""
    assert rekeyed.checks[0].binds_ac == ()
    assert rekeyed.checks[0].ac_binding_hashes == ()


def test_workflows_and_gates_are_wired() -> None:
    import yaml

    root = Path(__file__).resolve().parents[3]
    for filename, job, directory in (
        ("runtime-rebuild-trigger.yml", "verify-lane-converged", "."),
        (
            "deliver-dev-candidate-to-staging.yml",
            "candidate-boot-gate",
            "omnibase_infra",
        ),
    ):
        workflow = yaml.safe_load((root / ".github/workflows" / filename).read_text())
        steps = workflow["jobs"][job]["steps"]
        emitters = [
            step for step in steps if "lab_pass_receipt.py emit" in step.get("run", "")
        ]
        assert len(emitters) == 1
        assert f"--bind-from-commit {directory}" in emitters[0]["run"]
    assert (
        "test_lab_pass_criterion_bindings_omn18488.py"
        in (root / ".github/workflows/ci.yml").read_text()
    )
    assert (
        "test_lab_pass_criterion_bindings_omn18488.py"
        in (root / ".pre-commit-config.yaml").read_text()
    )


@pytest.mark.asyncio
async def test_the_real_reader_preserves_the_declared_cwd_and_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class Process:
        returncode = 0

        async def communicate(self):
            return b"[]", b""

    async def create(*args: str, **kwargs: object):
        calls.append((args, kwargs))
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    handler = closer.HandlerEvidenceAutocloseSweep(autoclose_disabled=False)
    assert (
        await handler._run_lab_pass_checks_real(REPO, SHA_4ACA, "OMN-18488", "", 30)
        == []
    )
    assert calls[0][1]["cwd"] is None
    assert calls[0][0][-3:] == (REPO, SHA_4ACA, "OMN-18488")
