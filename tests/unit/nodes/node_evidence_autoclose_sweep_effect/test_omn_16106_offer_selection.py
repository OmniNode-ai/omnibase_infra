# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16106 Item 1 — a RESTRICTIVE typed offer selector for the sweep.

The sweep's two enumeration arms are both WINDOWS over merged OCC companions.
``lookback_hours`` is a freshness window that sees a companion once and never
again; ``backfill_lookback_hours`` re-offers older companions on a rotating
five-wide slice bounded by a 200-item pool. Neither can be pointed at a named
ticket. ``exclude_tickets`` only subtracts, so the only way to act on one aged
ticket was to wait for its slice to come round — measured at ~34 ticks (~17h)
for one pass of the pool — or to widen the window past the run budget.

``offer_tickets`` is the selector, and RESTRICTIVE is the whole design:

* **Empty is the default and changes nothing.** Discovery, candidate order and
  per-candidate I/O are byte-identical to the pre-field run.
* **Non-empty REPLACES discovery.** It does not prepend to it. A run that
  nominates one ticket touches that ticket and nothing else — no forward
  window, no backfill slice, no pool enumeration. A selector that ADDED to
  discovery would be a way to smuggle a five-wide unscoped applying run in
  behind a one-ticket request, which is precisely what the bounded pilot must
  not be able to do by accident.
* **Direct resolution, not filtering.** Each nominated ticket's newest merged
  companion is resolved by search, so a companion outside the freshness window,
  outside the 200-item pool, and outside the current slice is still reachable.
  Filtering the slice could never have reached it.
* **Nothing about what counts as proven moves.** The resolved companion goes
  through the same changed-file binding check and the same ``_process_ticket``
  as any other candidate, so the verifier, AC-coverage, behaviour, children,
  cited-PR, prior-revert, label, disarm, flip-budget, comment-dedup and
  readback guards all still gate the write.
* **Caller exclusion wins first.** An offered ticket that is also excluded is
  refused before the companion is even searched for — zero GitHub I/O, zero
  Linear I/O.

The four groups below are the four the plan requires, in order.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
import yaml

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    HandlerEvidenceAutocloseSweep,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_arm import (
    EnumEvidenceAutocloseArm,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_decision import (
    EnumEvidenceAutocloseDecision,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_request import (
    ModelEvidenceAutocloseSweepRequest,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SWEEP_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "evidence-autoclose-sweep.yml"
SKILL_MAPPING = REPO_ROOT / "src" / "omnibase_infra" / "cli" / "skill_mapping.yaml"

_OCC_REPO = "OmniNode-ai/onex_change_control"
# The AGED ticket. Its companion merged 40 days ago: outside the 6h forward
# window, and — with the pool deliberately sized to 1 in the fixture below —
# outside whatever slice the current rotation tick would take.
_AGED = "OMN-16831"
_AGED_PR = 7001
# The FRESH, otherwise fully eligible ticket. It is in the forward window and
# its verdict clears the flip predicate outright, so if a non-empty offer list
# were additive rather than restrictive it would flip. It must not be touched.
_UNLISTED = "OMN-17397"
_UNLISTED_PR = 8501

_DOD_VERIFY_STATE_MODEL = (
    "omnimarket.nodes.node_dod_verify.models.model_dod_verify_state.ModelDodVerifyState"
)


def _iso(delta: timedelta) -> str:
    return (datetime.now(tz=UTC) - delta).strftime("%Y-%m-%dT%H:%M:%SZ")


def _search_item(ticket: str, number: int, merged_delta: timedelta) -> dict[str, Any]:
    """One `search/issues` item, in the shape the GitHub search API returns.

    `merged_at` lives under `pull_request`, not at the top level — that is the
    search endpoint's own shape, and a double that flattened it would let a
    resolver that reads the wrong key pass here and return nothing live.
    """
    merged_at = _iso(merged_delta)
    return {
        "number": number,
        "html_url": f"https://github.com/{_OCC_REPO}/pull/{number}",
        "title": f"evidence({ticket}): OCC companion",
        "updated_at": merged_at,
        "pull_request": {
            "url": f"https://api.github.com/repos/{_OCC_REPO}/pulls/{number}",
            "merged_at": merged_at,
        },
    }


def _merged_pr(ticket: str, number: int, merged_delta: timedelta) -> dict[str, Any]:
    """One `pulls` list entry, the shape the forward/backfill arms enumerate."""
    stamp = _iso(merged_delta)
    return {
        "number": number,
        "html_url": f"https://github.com/{_OCC_REPO}/pull/{number}",
        "title": f"evidence({ticket}): OCC companion",
        "updated_at": stamp,
        "merged_at": stamp,
    }


def _all_green(ticket: str) -> dict[str, Any]:
    """A verdict that clears the flip predicate outright."""
    terminal: dict[str, Any] = {
        "correlation_id": str(uuid4()),
        "ticket_id": ticket,
        "status": "verified",
        "dry_run": False,
        "checks": [
            {
                "evidence_id": "dod-pr-state",
                "description": "dod-pr-state",
                "status": "verified",
                "message": "OK (1ms)",
                "proof_class": "merge-state",
            },
            {
                "evidence_id": "dod-tests",
                "description": "dod-tests",
                "status": "verified",
                "message": "OK (1ms)",
                "proof_class": "behavior",
            },
        ],
        "total_checks": 2,
        "verified_count": 2,
        "failed_count": 0,
        "skipped_count": 0,
        "superseded_count": 0,
        "behavior_proving_count": 1,
        "error_message": None,
    }
    return {
        "skill_name": "dod_verify",
        "node_name": "node_dod_verify",
        "status": "success",
        "exit_code": 0,
        "result": terminal,
        "result_model": _DOD_VERIFY_STATE_MODEL,
    }


def _failed_dod(ticket: str) -> dict[str, Any]:
    """A verdict with a genuinely FAILED check — the gap path."""
    verdict = _all_green(ticket)
    result: dict[str, Any] = dict(verdict["result"])
    checks = [dict(check) for check in result["checks"]]
    checks[1]["status"] = "failed"
    checks[1]["message"] = "pytest exited 1"
    result["checks"] = checks
    result["status"] = "failed"
    result["verified_count"] = 1
    result["failed_count"] = 1
    result["behavior_proving_count"] = 0
    verdict["result"] = result
    return verdict


class _RecordingLinear:
    """Records reads as well as writes.

    ``reads`` is what proves group 2: an unlisted ticket must not appear in it
    at all. A double that only logged writes could not tell "never selected"
    from "selected, read, then refused".
    """

    def __init__(
        self,
        *,
        children: tuple[dict[str, Any], ...] = (),
        description: str | None = None,
    ) -> None:
        self.reads: list[str] = []
        self.state_updates: list[tuple[str, str]] = []
        self.comments: list[tuple[str, str]] = []
        self._children = children
        self._description = description

    async def fetch_issue(self, ticket_id: str) -> dict[str, Any]:
        self.reads.append(ticket_id)
        return {
            "id": f"issue-{ticket_id}",
            "identifier": ticket_id,
            "state": {"id": "s1", "name": "In Progress", "type": "started"},
            "labels": {"nodes": []},
            "children": {"nodes": list(self._children)},
            "team": {"id": "team-1"},
            "description": self._description,
        }

    async def fetch_done_state_id(self, team_id: str) -> str:
        return "state-done-id"

    async def update_issue_state(self, issue_id: str, state_id: str) -> bool:
        self.state_updates.append((issue_id, state_id))
        return True

    async def fetch_issue_history(
        self, issue_id: str, page_size: int, max_pages: int
    ) -> tuple[list[dict[str, Any]] | None, str]:
        return [
            {
                "id": f"entry-{index}",
                "createdAt": f"2026-09-07T00:00:{index:02d}Z",
                "actorId": None,
                "fromState": {"type": "started"},
                "toState": {"type": "completed"},
            }
            for index, (target, _state_id) in enumerate(self.state_updates, start=1)
            if target == issue_id
        ], ""

    async def create_comment(self, issue_id: str, body: str) -> bool:
        self.comments.append((issue_id, body))
        return True

    async def fetch_comment_bodies(self, issue_id: str) -> tuple[str, ...] | None:
        return tuple(body for target, body in self.comments if target == issue_id)


class _GhRecorder:
    """A `gh api` double that records every path it is asked for.

    The recorded paths are the evidence for the restrictive claim: in offer
    mode there must be NO `pulls?state=closed` enumeration at all, because a
    single such call is the forward/backfill arm running.
    """

    def __init__(
        self,
        *,
        cited_pr_merged: bool | None = None,
        search_items: dict[str, list[dict[str, Any]]] | None = None,
        search_error: str = "",
    ) -> None:
        self.paths: list[str] = []
        self._cited_pr_merged = cited_pr_merged
        self._search_items = (
            search_items
            if search_items is not None
            else {
                _AGED: [_search_item(_AGED, _AGED_PR, timedelta(days=40))],
                _UNLISTED: [_search_item(_UNLISTED, _UNLISTED_PR, timedelta(hours=1))],
            }
        )
        self._search_error = search_error

    async def __call__(self, args: list[str], timeout: float) -> tuple[Any, str]:
        path = args[2]
        self.paths.append(path)
        if path.startswith("search/issues"):
            if self._search_error:
                return None, self._search_error
            for ticket, items in self._search_items.items():
                if ticket in path:
                    return {"total_count": len(items), "items": items}, ""
            return {"total_count": 0, "items": []}, ""
        if "/files" in path:
            number = int(path.split("/pulls/", 1)[1].split("/", 1)[0])
            ticket = _AGED if number == _AGED_PR else _UNLISTED
            return [{"filename": f"contracts/{ticket}.yaml"}], ""
        if "/pulls?" in path:
            page = int(path.rsplit("page=", 1)[1])
            if page != 1:
                return [], ""
            return [_merged_pr(_UNLISTED, _UNLISTED_PR, timedelta(hours=1))], ""
        if "/pulls/" in path:
            # A cited product PR read (OMN-16106 D1).
            return {
                "merged_at": _iso(timedelta(hours=2)) if self._cited_pr_merged else None
            }, ""
        raise AssertionError(f"unexpected gh api path: {path}")


def _handler(
    linear: _RecordingLinear,
    gh: _GhRecorder,
    *,
    verdicts: dict[str, dict[str, Any]] | None = None,
) -> tuple[HandlerEvidenceAutocloseSweep, list[str]]:
    verified: list[str] = []

    async def fake_dod_verify(
        ticket_id: str, cwd: str, timeout: float
    ) -> tuple[dict[str, Any], int, str]:
        verified.append(ticket_id)
        table = verdicts or {}
        return table.get(ticket_id, _all_green(ticket_id)), 0, ""

    handler = HandlerEvidenceAutocloseSweep(
        linear_client=linear,  # type: ignore[arg-type]
        run_gh_command=gh,
        run_dod_verify_command=fake_dod_verify,
    )
    return handler, verified


def _request(**overrides: Any) -> ModelEvidenceAutocloseSweepRequest:
    defaults: dict[str, Any] = {
        "correlation_id": uuid4(),
        "occ_repo": _OCC_REPO,
        "lookback_hours": 6,
        # The pool is deliberately tiny so the aged companion is provably
        # outside the current rotation slice as well as outside the window.
        "backfill_lookback_hours": 168,
        "backfill_max_candidates": 5,
        "apply": True,
    }
    defaults.update(overrides)
    return ModelEvidenceAutocloseSweepRequest(**defaults)


# =====================================================================
# GROUP 1 — an aged nominated ticket is direct-resolved, bound, evaluated
#           and reported as an offer.
# =====================================================================


async def test_an_aged_nominated_ticket_is_resolved_outside_every_window() -> None:
    """The blocker, executed.

    ``_AGED``'s companion merged 40 days ago. It is outside the 6h forward
    window and, because the backfill arm is not enumerated at all in offer
    mode, outside the current slice too. It is still resolved, bound from its
    changed-file listing, verified and flipped.
    """
    linear = _RecordingLinear()
    gh = _GhRecorder()
    handler, verified = _handler(linear, gh)

    result = await handler.handle(_request(offer_tickets=(_AGED,)))

    assert result.tickets_flipped == 1
    outcome = result.outcomes[0]
    assert outcome.ticket_id == _AGED
    assert outcome.decision is EnumEvidenceAutocloseDecision.FLIPPED
    assert outcome.companion_pr_number == _AGED_PR
    # The DISTINCT ARM. A run that reported this as FORWARD or BACKFILL would
    # make the receipt claim a coverage window the run never enumerated.
    assert outcome.enumeration_arm is EnumEvidenceAutocloseArm.OFFER
    assert verified == [_AGED]
    assert linear.state_updates == [(f"issue-{_AGED}", "state-done-id")]
    # Resolution went through the SEARCH endpoint, not a window enumeration.
    assert any(path.startswith("search/issues") for path in gh.paths)


async def test_the_newest_merged_companion_wins_when_a_ticket_has_several() -> None:
    """ "Newest merged", resolved on merge time and not on list order.

    The search endpoint sorts by update time, which is not merge time: a stale
    companion touched yesterday sorts above the real one. Picking the first
    item would bind the wrong evidence.
    """
    older = _search_item(_AGED, 6001, timedelta(days=90))
    newer = _search_item(_AGED, 7002, timedelta(days=40))
    linear = _RecordingLinear()
    gh = _GhRecorder(search_items={_AGED: [older, newer]})
    handler, _verified = _handler(linear, gh)

    result = await handler.handle(_request(offer_tickets=(_AGED,)))

    assert result.outcomes[0].companion_pr_number == 7002


async def test_a_nominated_ticket_with_no_merged_companion_gets_a_named_outcome() -> (
    None
):
    """The named no-companion refusal.

    Reporting nothing at all would be indistinguishable from a run that never
    considered the ticket, and reporting SKIPPED_NO_BINDING would claim a
    companion was read and found unbindable. Neither happened.
    """
    linear = _RecordingLinear()
    gh = _GhRecorder(search_items={})
    handler, verified = _handler(linear, gh)

    result = await handler.handle(_request(offer_tickets=(_AGED,)))

    outcome = result.outcomes[0]
    assert outcome.ticket_id == _AGED
    assert outcome.decision is EnumEvidenceAutocloseDecision.SKIPPED_NO_OFFER_COMPANION
    assert outcome.enumeration_arm is EnumEvidenceAutocloseArm.OFFER
    assert result.tickets_skipped == 1
    assert verified == []
    assert linear.reads == []


async def test_a_fuzzy_search_hit_for_another_ticket_is_never_accepted() -> None:
    """Found live, on the first real dispatch of this path (run 34157024050).

    ``gh api search/issues?q=repo:...+is:pr+is:merged+OMN-99999`` is a FULL-TEXT
    query, not an exact-id lookup. GitHub tokenises ``OMN-99999`` and returns
    loosely related pull requests, so a dispatch offering a ticket with no
    companion at all came back with ``OMN-16007``'s companion (OCC#6448) — and
    the run then adjudicated ``OMN-16007``, a ticket nobody nominated, with
    ``enumeration_arm: offer`` on the outcome. The second dispatch did the same
    thing with ``OMN-15669``.

    It was harmless only because both were already ``completed`` and the run was
    a dry run. Under apply, a restrictive selector that reaches a ticket outside
    its own nomination is the worst failure this field can have: it is
    unbounded in exactly the direction the design claims to bound.

    The fix does not depend on search semantics, because search semantics are
    not ours to pin. The resolved companion must BIND the nominated ticket
    through the same ``_extract_ticket_binding`` the window arms use, and a
    companion that binds anything else is discarded — leaving the nominated
    ticket with the honest "no companion" outcome it should have had.
    """
    # The search returns a perfectly real, perfectly merged companion — for a
    # ticket the caller never named.
    linear = _RecordingLinear()
    gh = _GhRecorder(
        search_items={
            _AGED: [_search_item(_UNLISTED, _UNLISTED_PR, timedelta(days=40))]
        }
    )
    handler, verified = _handler(linear, gh)

    result = await handler.handle(_request(offer_tickets=(_AGED,)))

    assert [o.ticket_id for o in result.outcomes] == [_AGED], (
        "the run adjudicated a ticket that was not offered — a restrictive "
        "selector reached outside its own nomination"
    )
    assert (
        result.outcomes[0].decision
        is EnumEvidenceAutocloseDecision.SKIPPED_NO_OFFER_COMPANION
    )
    assert result.tickets_flipped == 0
    assert linear.reads == []
    assert verified == []


async def test_a_search_hit_binding_a_second_ticket_is_discarded_and_the_right_one_kept() -> (
    None
):
    """The filter discards the impostor without discarding the real answer."""
    linear = _RecordingLinear()
    gh = _GhRecorder(
        search_items={
            _AGED: [
                _search_item(_UNLISTED, _UNLISTED_PR, timedelta(days=1)),
                _search_item(_AGED, _AGED_PR, timedelta(days=40)),
            ]
        }
    )
    handler, _verified = _handler(linear, gh)

    result = await handler.handle(_request(offer_tickets=(_AGED,)))

    assert [o.ticket_id for o in result.outcomes] == [_AGED]
    assert result.outcomes[0].companion_pr_number == _AGED_PR
    assert result.outcomes[0].decision is EnumEvidenceAutocloseDecision.FLIPPED


async def test_a_companion_whose_files_bind_a_different_ticket_is_refused() -> None:
    """Defence in depth, past the title filter.

    The title filter runs on the search payload; the changed-file listing is
    fetched afterwards and is the stronger binding signal. If the two disagree
    for an offered candidate, the run must refuse rather than adjudicate
    whichever one it happens to resolve to.
    """
    linear = _RecordingLinear()
    gh = _GhRecorder()
    # The companion's title binds the aged ticket; its contract file does not.
    original = gh.__call__

    async def mismatched(args: list[str], timeout: float) -> tuple[Any, str]:
        if "/files" in args[2]:
            gh.paths.append(args[2])
            return [{"filename": f"contracts/{_UNLISTED}.yaml"}], ""
        return await original(args, timeout)

    handler, verified = _handler(linear, mismatched)  # type: ignore[arg-type]

    result = await handler.handle(_request(offer_tickets=(_AGED,)))

    assert result.tickets_flipped == 0
    assert [o.ticket_id for o in result.outcomes] == [_AGED]
    assert result.outcomes[0].decision in (
        EnumEvidenceAutocloseDecision.SKIPPED_NO_OFFER_COMPANION,
        EnumEvidenceAutocloseDecision.SKIPPED_AMBIGUOUS_BINDING,
    )
    assert linear.state_updates == []
    assert verified == []


async def test_a_failed_offer_resolution_is_a_github_error_not_an_absence() -> None:
    """ "I could not look" must never resolve to "there is nothing there"."""
    linear = _RecordingLinear()
    gh = _GhRecorder(search_error="gh api: 502 Bad Gateway")
    handler, _verified = _handler(linear, gh)

    result = await handler.handle(_request(offer_tickets=(_AGED,)))

    outcome = result.outcomes[0]
    assert outcome.decision is EnumEvidenceAutocloseDecision.ERROR_GITHUB_API
    assert result.tickets_errored == 1
    assert linear.reads == []


# =====================================================================
# GROUP 2 — an otherwise fully eligible UNLISTED ticket is not touched.
# =====================================================================


async def test_an_eligible_unlisted_ticket_gets_zero_reads_and_zero_writes() -> None:
    """RESTRICTIVE, measured on I/O rather than on the outcome list.

    ``_UNLISTED`` sits in the forward window with an all-green verdict, so
    under an ADDITIVE selector it would flip. Under a restrictive one it must
    cost nothing at all: no issue read, no changed-file read, no verifier call,
    no comment, no state mutation — and no forward enumeration to find it with.
    """
    linear = _RecordingLinear()
    gh = _GhRecorder()
    handler, verified = _handler(linear, gh)

    result = await handler.handle(_request(offer_tickets=(_AGED,)))

    assert [o.ticket_id for o in result.outcomes] == [_AGED]
    assert _UNLISTED not in linear.reads
    assert verified == [_AGED]
    # Scoped to the unlisted ticket, not to "no comments at all": the OFFERED
    # ticket flipping does post its audit comment, and that is the run working.
    assert [issue for issue, _body in linear.comments] == [f"issue-{_AGED}"]
    assert [issue for issue, _state in linear.state_updates] == [f"issue-{_AGED}"]
    # No window enumeration ran at all, so the unlisted ticket was never even
    # a candidate. This is the assertion that separates "restrictive" from
    # "discovered then filtered".
    assert [path for path in gh.paths if "/pulls?" in path] == []
    # And no file listing for the unlisted companion.
    assert f"/pulls/{_UNLISTED_PR}/files" not in " ".join(gh.paths)


async def test_offer_replaces_discovery_rather_than_prepending_to_it() -> None:
    """The receipt must not claim a window this run never enumerated."""
    linear = _RecordingLinear()
    gh = _GhRecorder()
    handler, _verified = _handler(linear, gh)

    result = await handler.handle(_request(offer_tickets=(_AGED,)))

    assert result.companions_scanned == 1
    assert result.backfill_pool_size == 0
    assert result.backfill_candidates_selected == 0


async def test_caller_exclusion_wins_before_any_io_about_an_offered_ticket() -> None:
    """Exclusion beats nomination, and beats it before the search runs.

    A ticket named in both lists is one a caller has asserted is mid-write by
    another lane. The refusal has to be terminal, which means it has to happen
    before anything can reclassify it — before the companion search, before the
    file listing, before the first Linear read.
    """
    linear = _RecordingLinear()
    gh = _GhRecorder()
    handler, verified = _handler(linear, gh)

    result = await handler.handle(
        _request(offer_tickets=(_AGED,), exclude_tickets=("  omn-16831 ",))
    )

    outcome = result.outcomes[0]
    assert outcome.decision is EnumEvidenceAutocloseDecision.SKIPPED_EXCLUDED
    assert outcome.enumeration_arm is EnumEvidenceAutocloseArm.OFFER
    assert linear.reads == []
    assert linear.state_updates == []
    assert verified == []
    assert gh.paths == []


async def test_the_kill_switch_still_dominates_an_offer_scoped_run() -> None:
    """An offer list must never be a way back INTO a halted run."""
    linear = _RecordingLinear()
    gh = _GhRecorder()
    handler, _verified = _handler(linear, gh)
    handler._autoclose_disabled_ctor = True

    result = await handler.handle(_request(offer_tickets=(_AGED,)))

    assert result.kill_switch_engaged is True
    assert result.outcomes == ()
    assert gh.paths == []
    assert linear.reads == []


# =====================================================================
# GROUP 3 — a selected offer still refuses on every guard.
# =====================================================================


async def test_a_selected_offer_still_refuses_on_a_failed_dod_check() -> None:
    linear = _RecordingLinear()
    gh = _GhRecorder()
    handler, _verified = _handler(linear, gh, verdicts={_AGED: _failed_dod(_AGED)})

    result = await handler.handle(_request(offer_tickets=(_AGED,)))

    assert result.tickets_flipped == 0
    assert result.outcomes[0].decision is EnumEvidenceAutocloseDecision.GAP_POSTED
    assert linear.state_updates == []


async def test_a_selected_offer_still_refuses_on_an_unmerged_cited_pr() -> None:
    """OMN-16106 D1's cited-PR merge conjunct is not bypassed by nomination."""
    linear = _RecordingLinear(
        description="AC-5 is proved by OmniNode-ai/omnimarket#4242."
    )
    gh = _GhRecorder(cited_pr_merged=False)
    handler, _verified = _handler(linear, gh)

    result = await handler.handle(_request(offer_tickets=(_AGED,)))

    assert result.tickets_flipped == 0
    assert (
        result.outcomes[0].decision
        is EnumEvidenceAutocloseDecision.SKIPPED_REFERENCED_PR_UNMERGED
    )
    assert linear.state_updates == []


async def test_a_selected_offer_still_refuses_on_an_open_child() -> None:
    linear = _RecordingLinear(
        children=({"identifier": "OMN-99001", "state": {"type": "started"}},)
    )
    gh = _GhRecorder()
    handler, _verified = _handler(linear, gh)

    result = await handler.handle(_request(offer_tickets=(_AGED,)))

    assert result.tickets_flipped == 0
    assert (
        result.outcomes[0].decision
        is EnumEvidenceAutocloseDecision.SKIPPED_HAS_CHILDREN
    )
    assert linear.state_updates == []


async def test_a_selected_offer_still_refuses_on_an_exhausted_flip_budget() -> None:
    """``max_flips_per_run=0`` is fail-closed, not "unbounded"."""
    linear = _RecordingLinear()
    gh = _GhRecorder()
    handler, _verified = _handler(linear, gh)

    result = await handler.handle(_request(offer_tickets=(_AGED,), max_flips_per_run=0))

    assert result.tickets_flipped == 0
    assert (
        result.outcomes[0].decision
        is EnumEvidenceAutocloseDecision.SKIPPED_FLIP_BUDGET_EXHAUSTED
    )
    assert linear.state_updates == []


async def test_a_selected_offer_still_refuses_when_the_run_is_disarmed() -> None:
    linear = _RecordingLinear()
    gh = _GhRecorder()
    handler, verified = _handler(linear, gh)

    result = await handler.handle(
        _request(offer_tickets=(_AGED,), disarmed_by_ticket="OMN-17556")
    )

    assert result.outcomes[0].decision is EnumEvidenceAutocloseDecision.SKIPPED_DISARMED
    assert verified == []
    assert linear.state_updates == []


async def test_the_all_green_positive_case_still_flips() -> None:
    """The control for the four refusals above.

    Without it, "not flipped" proves only that something withheld the write,
    not that the guard under test is what did.
    """
    linear = _RecordingLinear()
    gh = _GhRecorder()
    handler, _verified = _handler(linear, gh)

    result = await handler.handle(_request(offer_tickets=(_AGED,)))

    assert result.tickets_flipped == 1
    assert result.outcomes[0].decision is EnumEvidenceAutocloseDecision.FLIPPED
    assert linear.state_updates == [(f"issue-{_AGED}", "state-done-id")]


# =====================================================================
# GROUP 4 — empty offers preserve today's behaviour; dry-run writes nothing;
#           model / CLI / workflow-input plumbing.
# =====================================================================


async def test_empty_offers_preserve_candidate_order_and_lookups() -> None:
    """The default path is byte-identical to the pre-field run.

    The forward window is enumerated, the unlisted ticket IS the candidate,
    and the aged ticket is not reached — exactly as before the field existed.
    """
    linear = _RecordingLinear()
    gh = _GhRecorder()
    handler, verified = _handler(linear, gh)

    result = await handler.handle(_request())

    assert [o.ticket_id for o in result.outcomes] == [_UNLISTED]
    assert result.outcomes[0].enumeration_arm is EnumEvidenceAutocloseArm.FORWARD
    assert verified == [_UNLISTED]
    assert linear.reads == [_UNLISTED]
    # The window arms ran and the search endpoint did not.
    assert any("/pulls?" in path for path in gh.paths)
    assert [path for path in gh.paths if path.startswith("search/issues")] == []


async def test_an_all_green_offered_ticket_in_dry_run_writes_nothing() -> None:
    """A rehearsal of the offer path reaches the decision and writes none of it."""
    linear = _RecordingLinear()
    gh = _GhRecorder()
    handler, verified = _handler(linear, gh)

    result = await handler.handle(_request(offer_tickets=(_AGED,), apply=False))

    assert result.dry_run is True
    outcome = result.outcomes[0]
    assert outcome.decision is EnumEvidenceAutocloseDecision.FLIPPED
    assert outcome.applied is False
    assert outcome.linear_comment_posted is False
    assert verified == [_AGED]
    assert linear.state_updates == []
    assert linear.comments == []


def test_the_request_model_declares_offer_tickets_empty_by_default() -> None:
    """Additive on a frozen, ``extra="forbid"`` model."""
    request = ModelEvidenceAutocloseSweepRequest(correlation_id=uuid4())
    assert request.offer_tickets == ()
    assert (
        ModelEvidenceAutocloseSweepRequest.model_fields["offer_tickets"].default == ()
    )


def test_offer_matching_ignores_case_and_surrounding_whitespace() -> None:
    """Operator-typed text, spliced from a script or a dispatch box."""
    request = ModelEvidenceAutocloseSweepRequest(
        correlation_id=uuid4(), offer_tickets=("  omn-16831 ",)
    )
    assert request.offer_tickets == ("  omn-16831 ",)


async def test_a_lowercase_offer_still_resolves() -> None:
    linear = _RecordingLinear()
    gh = _GhRecorder()
    handler, _verified = _handler(linear, gh)

    result = await handler.handle(_request(offer_tickets=("  omn-16831 ",)))

    assert result.outcomes[0].ticket_id == _AGED
    assert result.outcomes[0].decision is EnumEvidenceAutocloseDecision.FLIPPED


def test_the_cli_exposes_offer_as_a_typed_comma_list() -> None:
    mapping = yaml.safe_load(SKILL_MAPPING.read_text(encoding="utf-8"))
    entry = next(
        skill
        for skill in mapping["skills"]
        if skill["skill_name"] == "evidence_autoclose_sweep"
    )
    args = {arg["name"]: arg for arg in entry["args"]}
    assert "offer" in args, (
        "skill_mapping.yaml does not expose `--offer` for "
        "evidence_autoclose_sweep — the workflow could not pass it"
    )
    assert args["offer"]["payload_field"] == "offer_tickets"
    assert args["offer"]["arg_type"] == "string_list", (
        "the offer selector must be a typed comma-list, the same wire form as "
        "`--exclude`, not a single string the node has to re-parse"
    )


def test_the_workflow_exposes_a_manual_offer_input_and_no_standing_variable() -> None:
    """A dispatch input, deliberately — and NOT a repo variable.

    The standing fence ``ONEX_AUTOCLOSE_EXCLUDE`` is a repo variable precisely
    because a fence must reach the unattended runs. An OFFER is the opposite
    kind of thing: it SELECTS what a run acts on, so a standing one would
    silently narrow every scheduled tick to a fixed list and stop the closer
    adjudicating the board — a restriction nobody typed, on every run nobody is
    watching. It is per-dispatch or it does not exist.
    """
    workflow: dict[Any, Any] = yaml.safe_load(
        SWEEP_WORKFLOW.read_text(encoding="utf-8")
    )
    triggers: dict[str, Any] = {}
    for key in (True, "on"):
        candidate = workflow.get(key)
        if isinstance(candidate, dict):
            triggers = candidate
            break
    inputs = (triggers.get("workflow_dispatch") or {}).get("inputs") or {}
    assert "offer" in inputs, (
        "evidence-autoclose-sweep.yml declares no `offer` dispatch input"
    )
    assert inputs["offer"].get("default") == "", (
        "the offer input must default to the empty string — empty offers are "
        "the discovery-preserving path and every scheduled tick takes it"
    )

    source = SWEEP_WORKFLOW.read_text(encoding="utf-8")
    assert "ONEX_AUTOCLOSE_OFFER" not in source, (
        "a standing repo variable for offers would narrow every unattended run "
        "to a fixed list. The offer selector is per-dispatch only."
    )
    # Passed as a value through the step environment, never interpolated into
    # a `run:` body where an operator-typed string becomes shell syntax.
    assert "OFFER_TICKETS: ${{ github.event.inputs.offer || '' }}" in source
    assert "${{ github.event.inputs.offer }}" not in source.replace(
        "OFFER_TICKETS: ${{ github.event.inputs.offer || '' }}", ""
    )
