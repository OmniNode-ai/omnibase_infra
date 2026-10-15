# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise local offer discovery through the closer's serialized receipt.

Real Git refs and watcher JSON supply the candidate and its changed files.
Only the downstream Linear and verifier boundaries are doubles, so no remote
service is needed to check the offer's provenance and binding safety.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_decision import (
    EnumEvidenceAutocloseDecision,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_request import (
    ModelEvidenceAutocloseSweepRequest,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_result import (
    ModelEvidenceAutocloseSweepResult,
)
from tests.unit.nodes.node_evidence_autoclose_sweep_effect.test_omn_16106_offer_selection import (
    _RecordingLinear,
)
from tests.unit.nodes.node_evidence_autoclose_sweep_effect.test_omn_21146_offer_local_companion import (
    _Gateway,
    _Gh,
    _git,
    _handler,
    _land,
    _make_clone,
    _two_ticks,
    _write_watcher,
)

pytestmark = pytest.mark.integration

_TICKET = "OMN-21146"
_OCC_REPO = "OmniNode-ai/onex_change_control"


@pytest.mark.parametrize("source", ["companion", "product_contract"])
async def test_local_offer_provenance_survives_the_receipt(
    tmp_path: Path, source: str
) -> None:
    clones = tmp_path / "clones"
    occ = _make_clone(clones, "onex_change_control")
    watcher = tmp_path / "watcher.json"
    merged_at = datetime.now(tz=UTC) - timedelta(days=12)
    if source == "companion":
        subject = f"evidence({_TICKET}): OCC batch window (#12455)"
        merged_sha = _land(occ, subject, [f"contracts/{_TICKET}.yaml"], when=merged_at)
        _write_watcher(
            watcher,
            "onex_change_control",
            12455,
            subject,
            merged_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        # A newer, unmerged companion in the working tree must not win.
        _git(occ, "checkout", "-q", "-b", "unmerged-companion")
        _land(
            occ,
            f"evidence({_TICKET}): pending companion (#12456)",
            [f"contracts/{_TICKET}.yaml"],
        )
        _git(occ, "update-ref", "refs/remotes/origin/dev", merged_sha)
        expected_number = 12455
        expected_url = f"https://github.com/{_OCC_REPO}/pull/12455"
    else:
        product = _make_clone(clones, "omnibase_infra")
        merged_sha = _land(
            product,
            f"feat({_TICKET}): product evidence (#77)",
            [f"contracts/{_TICKET}.yaml"],
            when=merged_at,
        )
        expected_number = 0
        expected_url = (
            "https://github.com/OmniNode-ai/omnibase_infra/"
            f"blob/{merged_sha}/contracts/{_TICKET}.yaml"
        )

    # Include the request serialization boundary used by runtime dispatch.
    request = ModelEvidenceAutocloseSweepRequest.model_validate_json(
        ModelEvidenceAutocloseSweepRequest(
            correlation_id=uuid4(),
            occ_repo=_OCC_REPO,
            apply=True,
            offer_tickets=(_TICKET,),
            clone_root=str(clones),
            pr_watcher_state_path=str(watcher),
        ).model_dump_json()
    )
    linear = _RecordingLinear()
    gh = _Gh()
    gateway = _Gateway()
    handler, verified = _handler(linear, gh, gateway)

    result = await _two_ticks(handler, request)
    receipt_json = result.model_dump_json()
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(receipt_json)
    parsed = ModelEvidenceAutocloseSweepResult.model_validate_json(
        receipt_path.read_text()
    )

    assert parsed.correlation_id == request.correlation_id
    assert parsed.tickets_flipped == 1
    assert len(parsed.outcomes) == 1
    outcome = parsed.outcomes[0]
    assert outcome.ticket_id == _TICKET
    assert outcome.decision is EnumEvidenceAutocloseDecision.FLIPPED
    assert outcome.enumeration_arm.value == "offer"
    assert outcome.companion_pr_number == expected_number
    assert outcome.companion_pr_url == expected_url
    assert verified == [_TICKET, _TICKET]
    assert linear.state_updates == [(f"issue-{_TICKET}", "state-done-id")]
    assert gh.paths == []
    assert [call[2] for call in gateway.calls] == (
        [] if source == "companion" else ["pr-heads", "pr-heads"]
    )


async def test_local_changed_file_conflict_is_refused_in_the_receipt(
    tmp_path: Path,
) -> None:
    clones = tmp_path / "clones"
    occ = _make_clone(clones, "onex_change_control")
    _land(
        occ,
        f"evidence({_TICKET}): mismatched contract (#12455)",
        ["contracts/OMN-21147.yaml"],
    )
    request = ModelEvidenceAutocloseSweepRequest(
        correlation_id=uuid4(),
        occ_repo=_OCC_REPO,
        apply=True,
        offer_tickets=(_TICKET,),
        clone_root=str(clones),
        pr_watcher_state_path=str(tmp_path / "absent-watcher.json"),
    )
    linear = _RecordingLinear()
    gh = _Gh()
    gateway = _Gateway()
    handler, verified = _handler(linear, gh, gateway)

    result = await handler.handle(request)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(result.model_dump_json())
    parsed = ModelEvidenceAutocloseSweepResult.model_validate_json(
        receipt_path.read_text()
    )

    assert parsed.tickets_flipped == 0
    assert len(parsed.outcomes) == 1
    assert parsed.outcomes[0].ticket_id == _TICKET
    assert (
        parsed.outcomes[0].decision
        is EnumEvidenceAutocloseDecision.SKIPPED_NO_OFFER_COMPANION
    )
    assert "(ambiguous)" in parsed.outcomes[0].reason
    assert verified == []
    assert linear.reads == [_TICKET]
    assert linear.state_updates == []
    assert linear.comments == []
    assert gh.paths == []
    assert gateway.calls == []
