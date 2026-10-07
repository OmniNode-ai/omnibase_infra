# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18488 — a lab-pass receipt's binding reaches the closer's receipt JSON.

The unit suite asserts the flip decision on the in-memory outcome. This test
drives the whole closer with a bound lab-pass receipt read through the
emitter's own artifact reader, then serialises the result exactly as receipt
mode does, ``model_dump(mode="json")``, and asks the document which check
discharged the criterion. A change that dropped the lab row on the way out
would flip a ticket whose receipt never names the probe that proved it.

It also asserts the reverse trip: the receipt must parse back into the
result model, because ``extra="forbid"`` turns any unserialisable lab field
into a hard failure for every consumer.
"""

from __future__ import annotations

import json

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_result import (
    ModelEvidenceAutocloseSweepResult,
)
from scripts.ci import lab_pass_receipt as lab
from tests.scripts.ci._lab_pass_fixtures import REPO, SHA_4ACA, FakeSurface
from tests.scripts.ci.test_lab_pass_criterion_bindings_omn18488 import (
    _bound_receipt,
)
from tests.unit.nodes.node_evidence_autoclose_sweep_effect import (
    test_omn_18330_criterion_hash as closer,
)

pytestmark = pytest.mark.integration

_LAB_CHECK = f"lab-pass-compose-dev-ready_main@{SHA_4ACA}"


async def _sweep(
    monkeypatch: pytest.MonkeyPatch, *, passing: bool
) -> tuple[ModelEvidenceAutocloseSweepResult, closer.FakeLinear]:
    surface = FakeSurface()
    surface.add(_bound_receipt(passing=passing))
    monkeypatch.setattr(lab, "_gh_api", surface)
    original_gh = closer._gh_fake()

    async def gh_read(args: list[str], timeout: float):
        payload, error = await original_gh(args, timeout)
        if isinstance(payload, dict) and "/pulls/" in args[2]:
            payload["merge_commit_sha"] = SHA_4ACA
        return payload, error

    async def lab_read(repo: str, sha: str, ticket: str, cwd: str, timeout: float):
        assert (repo, sha) == (REPO, SHA_4ACA)
        return lab.receipt_evidence_for_ticket(repo, sha, ticket)

    # The verifier's own check binds nothing: only the lab receipt can
    # discharge AC1, so a flip is proof the lab binding was carried.
    verdict = closer._receipt(None)
    verdict["result"]["checks"][0]["binds_ac"] = []
    linear = closer.FakeLinear(closer._ACCEPTED_CRITERION)
    handler = closer.HandlerEvidenceAutocloseSweep(
        linear_client=linear,
        autoclose_disabled=False,
        run_gh_command=gh_read,
        run_dod_verify_command=closer._dod_fake(verdict),
    )
    monkeypatch.setattr(handler, "_run_lab_pass_checks", lab_read)
    await handler.handle(closer._request())
    return await handler.handle(closer._request()), linear


@pytest.mark.asyncio
async def test_the_receipt_json_names_the_lab_check_that_bound_the_criterion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, linear = await _sweep(monkeypatch, passing=True)

    document = json.dumps(result.model_dump(mode="json"))
    assert _LAB_CHECK in document
    (outcome,) = json.loads(document)["outcomes"]
    assert outcome["decision"] == "flipped"
    (row,) = [r for r in outcome["ac_binding_rows"] if r["label"] == "AC1"]
    assert (row["evidence_check"], row["status"], row["bound"]) == (
        _LAB_CHECK,
        "verified",
        True,
    )
    assert linear.state_updates == [("issue-1", "state-done")]


@pytest.mark.asyncio
async def test_a_fail_receipt_reaches_the_json_as_failed_and_flips_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, linear = await _sweep(monkeypatch, passing=False)

    (outcome,) = result.model_dump(mode="json")["outcomes"]
    assert outcome["decision"] != "flipped"
    (lab_row,) = [
        r for r in outcome["check_results"] if r["evidence_check"] == _LAB_CHECK
    ]
    assert lab_row["status"] == "failed"
    assert linear.state_updates == []


@pytest.mark.asyncio
async def test_the_receipt_round_trips_with_the_lab_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = await _sweep(monkeypatch, passing=True)

    reparsed = ModelEvidenceAutocloseSweepResult.model_validate(
        json.loads(json.dumps(result.model_dump(mode="json")))
    )
    assert reparsed.outcomes[0].check_results == result.outcomes[0].check_results
    assert _LAB_CHECK in {
        row.evidence_check for row in reparsed.outcomes[0].check_results
    }
