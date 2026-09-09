# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18106 — the prior-revert fence baselines AFTER the evidence already moved.

The measured defect, on OMN-15542:

* the ticket was moved Done -> In Progress at 2026-09-02T07:46:30Z;
* real remediation landed in the gap that followed — AC1/AC2/AC4/AC5 bound to
  verified checks, AC6 demoted out of the criterion set on 2026-09-09T22:05Z;
* the closer's FIRST post-revert look happened at 2026-09-09T23:16:09Z, run
  34415223971, and recorded fingerprint ``ca128f676aa74a44`` as the post-revert
  baseline — reading the ALREADY-IMPROVED state as the thing change would have
  to be measured against.

dod_verify is deterministic over the same ticket state, so every later tick
recomputes that identical fingerprint, ``fingerprint not in baselines`` can
never fire again, and the ticket is held forever. "Hold until something changed
since the revert" silently became "hold forever whenever the closer's first
look lands after the remediation" — which is the COMMON case, because work
continues after a revert and the closer only ticks every half hour.

This suite pins the ordering the fence was always about: evidence that
POSTDATES the revert is positive evidence of change, whoever recorded what
fingerprint first. The control below pins the other half — evidence that
entirely predates the revert still holds, which is the case the fence exists
for (the OMN-17298 shape: companion merged, flip, revert, nothing new since).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    HandlerEvidenceAutocloseSweep,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_decision import (
    EnumEvidenceAutocloseDecision,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_request import (
    ModelEvidenceAutocloseSweepRequest,
)

pytestmark = pytest.mark.unit

_OCC_REPO = "OmniNode-ai/onex_change_control"
_TICKET = "OMN-9999"
_ISSUE_ID = "issue-uuid-1"
_HUMAN_ACTOR = "7a850ce1-f95e-431f-b4e3-62f7449f04c0"
_DOD_VERIFY_STATE_MODEL = (
    "omnimarket.nodes.node_dod_verify.models.model_dod_verify_state.ModelDodVerifyState"
)

# Two labelled criteria, both bound below. No checkbox, no `Gate:` line, no PR
# citation — so every other conjunct releases and only the revert fence can
# hold this fixture.
_DESCRIPTION = """\
Terminal-event isolation for concurrent RuntimeLocal invocations.

## Acceptance criteria

- **AC1** RED-first: a test drives two concurrent invocations and fails on the
  current code.
- **AC2** GREEN: each invocation accepts only its own correlation-scoped
  terminal event.
"""


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _companion(number: int, merged_at: str) -> dict[str, object]:
    return {
        "number": number,
        "html_url": f"https://github.com/{_OCC_REPO}/pull/{number}",
        "title": f"evidence({_TICKET}): OCC companion",
        "updated_at": merged_at,
        "merged_at": merged_at,
    }


def _check(evidence_id: str, status: str, proof_class: str, binds: tuple[str, ...]):
    return {
        "evidence_id": evidence_id,
        "description": evidence_id,
        "status": status,
        "message": "OK (1ms)",
        "proof_class": proof_class,
        "binds_ac": list(binds),
    }


def _bound_checks() -> list[dict[str, object]]:
    return [
        _check("dod-tests", "verified", "behavior", ("AC1", "AC2")),
        _check("dod-pr-files", "verified", "merge-state", ("AC1",)),
    ]


def _skill_result(checks: list[dict[str, object]]) -> dict[str, object]:
    """A green ``onex skill dod_verify`` receipt whose counters derive from ``checks``."""
    verified = sum(1 for c in checks if c["status"] == "verified")
    non_probative = sum(1 for c in checks if c["status"] == "non_probative")
    behavior = sum(
        1
        for c in checks
        if c["status"] == "verified" and c["proof_class"] == "behavior"
    )
    verdict: dict[str, object] = {
        "correlation_id": str(uuid4()),
        "ticket_id": _TICKET,
        "status": "verified",
        "dry_run": False,
        "checks": checks,
        "total_checks": len(checks),
        "verified_count": verified,
        "failed_count": 0,
        "skipped_count": 0,
        "superseded_count": 0,
        "non_probative_count": non_probative,
        "behavior_proving_count": behavior,
        "error_message": None,
    }
    return {
        "skill_name": "dod_verify",
        "node_name": "node_dod_verify",
        "status": "success",
        "correlation_id": str(uuid4()),
        "run_id": str(uuid4()),
        "exit_code": 0,
        "duration_ms": 1,
        "result": verdict,
        "result_model": _DOD_VERIFY_STATE_MODEL,
    }


class _RevertedLinear:
    """A ticket reverted out of Done at ``revert_at`` by a real person.

    The flip that the revert undid is a HUMAN one (both entries carry an actor
    id), so ``_closer_flip_was_reverted`` — the negative branch — cannot fire
    and the POSITIVE fence is the only thing adjudicating this fixture. That is
    the OMN-15542 shape: nothing in its history is a fingerprint this closer
    wrote.
    """

    def __init__(self, revert_at: datetime) -> None:
        self._revert_at = revert_at
        self.state_updates: list[tuple[str, str]] = []
        self.comments: list[tuple[str, str]] = []

    async def fetch_issue(self, ticket_id: str) -> dict[str, object]:
        return {
            "id": _ISSUE_ID,
            "identifier": _TICKET,
            "state": {"id": "s1", "name": "In Progress", "type": "started"},
            "labels": {"nodes": []},
            "children": {"nodes": []},
            "team": {"id": "team-1"},
            "description": _DESCRIPTION,
        }

    async def fetch_done_state_id(self, team_id: str) -> str:
        return "state-done-id"

    async def update_issue_state(self, issue_id: str, state_id: str) -> bool:
        self.state_updates.append((issue_id, state_id))
        return True

    async def fetch_issue_history(
        self, issue_id: str, page_size: int, max_pages: int
    ) -> tuple[list[dict[str, object]] | None, str]:
        written = [
            {
                "id": f"entry-sweep-{index}",
                "createdAt": _iso(_now()),
                "actorId": None,
                "fromState": {"type": "started"},
                "toState": {"type": "completed"},
            }
            for index, (target, _state_id) in enumerate(
                reversed(self.state_updates), start=1
            )
            if target == issue_id
        ]
        return [
            *written,
            {
                "id": "entry-revert",
                "createdAt": _iso(self._revert_at),
                "actorId": _HUMAN_ACTOR,
                "fromState": {"type": "completed"},
                "toState": {"type": "started"},
            },
            {
                "id": "entry-flip",
                "createdAt": _iso(self._revert_at - timedelta(minutes=10)),
                "actorId": _HUMAN_ACTOR,
                "fromState": {"type": "started"},
                "toState": {"type": "completed"},
            },
        ], ""

    async def create_comment(self, issue_id: str, body: str) -> bool:
        self.comments.append((issue_id, body))
        return True

    async def fetch_comment_bodies(self, issue_id: str) -> tuple[str, ...] | None:
        return tuple(body for target, body in self.comments if target == issue_id)


def _handler(
    linear: _RevertedLinear, companion_merged_at: str
) -> HandlerEvidenceAutocloseSweep:
    async def fake_gh(args: list[str], timeout: float):
        path = args[2]
        if "/files" in path:
            return [{"filename": f"contracts/{_TICKET}.yaml"}], ""
        page = int(path.rsplit("page=", 1)[1])
        if page == 1:
            return [_companion(7000, companion_merged_at)], ""
        return [], ""

    async def fake_dod_verify(ticket_id: str, cwd: str, timeout: float):
        return _skill_result(_bound_checks()), 0, ""

    return HandlerEvidenceAutocloseSweep(
        linear_client=linear,  # type: ignore[arg-type]
        autoclose_disabled=False,
        run_gh_command=fake_gh,
        run_dod_verify_command=fake_dod_verify,
    )


def _request() -> ModelEvidenceAutocloseSweepRequest:
    return ModelEvidenceAutocloseSweepRequest(
        correlation_id=uuid4(),
        occ_repo=_OCC_REPO,
        lookback_hours=24,
        apply=True,
    )


async def _tick(
    linear: _RevertedLinear, companion_merged_at: str
) -> EnumEvidenceAutocloseDecision:
    result = await _handler(linear, companion_merged_at).handle(_request())
    return result.outcomes[0].decision


async def test_evidence_that_postdates_the_revert_releases_the_fence() -> None:
    """THE RECORDED DEFECT — the OMN-15542 timeline, T0 < T1 < T2.

    T0  the human revert.
    T1  the OCC evidence companion merges, carrying the bindings that were the
        remediation.
    T2  the closer looks for the FIRST time.

    Under the recorded-baseline mechanism T2's verdict becomes the baseline, so
    the ticket can never produce a fingerprint that differs from it and is held
    for good. The fence must instead read the ordering it was always about:
    this ticket's evidence moved AFTER the disagreement, so the disagreement is
    not being overruled by re-asserting an unchanged verdict.

    Released means released to the closer's ORDINARY predicate, not flipped on
    the spot: the re-draw still costs one tick, which is why the flip is
    asserted on the second observation and not the first.
    """
    revert_at = _now() - timedelta(days=7)
    evidence_merged_at = _iso(_now() - timedelta(hours=1))
    linear = _RevertedLinear(revert_at)

    first = await _tick(linear, evidence_merged_at)
    assert first is not EnumEvidenceAutocloseDecision.SKIPPED_PRIOR_REVERT
    assert first is EnumEvidenceAutocloseDecision.SKIPPED_REDRAW_PENDING

    second = await _tick(linear, evidence_merged_at)
    assert second is EnumEvidenceAutocloseDecision.FLIPPED
    assert linear.state_updates == [(_ISSUE_ID, "state-done-id")]


async def test_the_release_is_stated_in_the_flip_reason() -> None:
    """A release nobody can read is a release nobody can audit.

    The flip's own reason has to say WHICH evidence postdated the revert and by
    how much, because "the closer decided the fence did not apply" is a claim
    and "companion #7000 merged after the 2026-09-02 revert" is a fact.
    """
    revert_at = _now() - timedelta(days=7)
    evidence_merged_at = _iso(_now() - timedelta(hours=1))
    linear = _RevertedLinear(revert_at)

    await _tick(linear, evidence_merged_at)
    flip = (await _handler(linear, evidence_merged_at).handle(_request())).outcomes[0]

    assert flip.decision is EnumEvidenceAutocloseDecision.FLIPPED
    assert "released_post_revert_evidence" in flip.reason
    assert flip.post_revert_evidence_release
    assert _iso(revert_at) in flip.post_revert_evidence_release
    assert "7000" in flip.post_revert_evidence_release
    body = linear.comments[-1][1]
    assert "released_post_revert_evidence" in body


async def test_evidence_entirely_before_the_revert_still_holds() -> None:
    """THE CONTROL — the OMN-17298 shape, and the case the fence exists for.

    Companion merged, somebody flipped, somebody took it back. Nothing has
    merged since. The verdict is identical to the one that was overruled, so
    re-applying it is the cron tick overruling a person, and it must hold on
    every tick — not just the first.
    """
    linear = _RevertedLinear(_now() - timedelta(minutes=30))
    evidence_merged_at = _iso(_now() - timedelta(hours=1))

    first = await _tick(linear, evidence_merged_at)
    assert first is EnumEvidenceAutocloseDecision.SKIPPED_PRIOR_REVERT

    # Second tick: the OMN-16808 idempotency layer recognises the statement has
    # already been made, so the decision reads SKIPPED_DUPLICATE_COMMENT. The
    # hold is unchanged and nothing is written, which is what is asserted.
    second = await _tick(linear, evidence_merged_at)
    assert second is EnumEvidenceAutocloseDecision.SKIPPED_DUPLICATE_COMMENT

    third = await _tick(linear, evidence_merged_at)
    assert third is EnumEvidenceAutocloseDecision.SKIPPED_DUPLICATE_COMMENT
    assert linear.state_updates == []


async def test_an_unreadable_revert_time_holds_rather_than_releases() -> None:
    """Fail-closed. "I cannot tell when the revert happened" is not "it is old".

    A history entry whose ``createdAt`` cannot be read leaves the ordering
    unresolvable, and an unresolvable ordering must take the hold — the same
    direction of conservatism every other fence in this module takes.
    """

    class _NoTimestampLinear(_RevertedLinear):
        async def fetch_issue_history(
            self, issue_id: str, page_size: int, max_pages: int
        ) -> tuple[list[dict[str, object]] | None, str]:
            history, error = await super().fetch_issue_history(
                issue_id, page_size, max_pages
            )
            assert history is not None
            for entry in history:
                if entry.get("id") == "entry-revert":
                    entry["createdAt"] = ""
            return history, error

    linear = _NoTimestampLinear(_now() - timedelta(days=7))
    evidence_merged_at = _iso(_now() - timedelta(hours=1))

    outcome = (await _handler(linear, evidence_merged_at).handle(_request())).outcomes[
        0
    ]
    assert outcome.decision is EnumEvidenceAutocloseDecision.SKIPPED_PRIOR_REVERT
    assert "could not be read" in outcome.reason
    assert linear.state_updates == []
