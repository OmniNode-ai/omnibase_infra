# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17658 / OMN-17934 — the fences the applying closer was armed without.

The evidence closer began writing to Linear on its own 30-minute schedule at
``e9f178165`` (omnibase_infra#3195, merged 2026-09-05T00:03:31Z). Its arming row
— plan P1-8, OMN-17658 — was still Backlog, and every fence that row makes a
precondition of arming was absent: no children conjunct, no
``scheduled_apply`` arming authority, no ``max_flips_per_run``, no bound
readback, no auto-disarm.

The first applying run (33932169358) flipped four tickets and one was wrong:
OMN-17292 went Done on ``onex_change_control#8224``, the evidence companion of
``omnibase_infra#3192`` — ``chore(OMN-17292): advance omnimarket contract pin to
55c8f2642214``, a routine emission of the standing 6-hourly refresh bot. That
ticket had already been flipped Done once (2026-08-31) and reopened
(2026-09-03). It re-cleared the predicate because nothing in the predicate is
about WHO produced the evidence or whether anybody had already disagreed with a
close.

Every test here asserts a NARROWING. Not one of them changes what counts as
proof: the OMN-16821 denominator equality, the OMN-15911 behaviour conjunct and
the OMN-16736 AC-coverage re-read are untouched, and a candidate that clears
them all still flips unless one of the fences below refuses it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    HandlerEvidenceAutocloseSweep,
    _ac_coverage_gap,
    _is_recurring_bot_product_pr,
    _product_pr_ref,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_decision import (
    EnumEvidenceAutocloseDecision,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_mode import (
    EnumEvidenceAutocloseMode,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_trigger import (
    EnumEvidenceAutocloseTrigger,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_request import (
    ModelEvidenceAutocloseSweepRequest,
)
from tests.unit.nodes.node_evidence_autoclose_sweep_effect._ac_binding_support import (
    BOUND_AC_DESCRIPTION,
    bound_ac_checks,
    redraw_marker_comment,
)

pytestmark = pytest.mark.unit

_OCC_REPO = "OmniNode-ai/onex_change_control"
_DOD_VERIFY_STATE_MODEL = (
    "omnimarket.nodes.node_dod_verify.models.model_dod_verify_state.ModelDodVerifyState"
)
# The real product PR the OMN-17292 mis-flip was bound through.
_PIN_BUMP_TITLE = "chore(OMN-17292): advance omnimarket contract pin to 55c8f2642214"


# ---------------------------------------------------------------- doubles ---


def _merged_companion(
    number: int, ticket: str, product: str = "OmniNode-ai/omnibase_infra#3194"
) -> dict[str, object]:
    recent = (datetime.now(tz=UTC) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "number": number,
        "html_url": f"https://github.com/{_OCC_REPO}/pull/{number}",
        "title": f"evidence({ticket}): OCC companion for {product}",
        "updated_at": recent,
        "merged_at": recent,
    }


def _flip_clearing_receipt() -> dict[str, object]:
    """A verdict that clears every pre-existing flip conjunct outright."""
    verdict: dict[str, object] = {
        "correlation_id": str(uuid4()),
        "ticket_id": "OMN-0000",
        "status": "verified",
        "dry_run": False,
        # OMN-18056: the corpus DECLARES which criterion it proves. Without a
        # binding the AC-binding gate holds every fixture in this file before
        # the fence under test runs.
        "checks": bound_ac_checks(),
        "total_checks": 2,
        "verified_count": 2,
        "failed_count": 0,
        "skipped_count": 0,
        "superseded_count": 0,
        "non_probative_count": 0,
        "behavior_proving_count": 1,
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


def _issue(
    *,
    issue_id: str = "issue-1",
    identifier: str = "OMN-0000",
    state_type: str = "started",
    children: tuple[tuple[str, str], ...] = (),
    description: str | None = BOUND_AC_DESCRIPTION,
) -> dict[str, object]:
    """A Linear issue payload in the shape ``_ISSUE_QUERY`` actually returns.

    OMN-18056: ``description`` defaults to a body carrying ONE labelled,
    parseable acceptance criterion. It used to default to ``None``, which the
    AC-binding gate now holds -- a closer that cannot read a ticket's criteria
    can say nothing about them -- so every fence in this file would be
    unreachable behind a guard that fires first.
    """
    return {
        "id": issue_id,
        "identifier": identifier,
        "state": {"id": "s1", "name": "In Progress", "type": state_type},
        "labels": {"nodes": []},
        "team": {"id": "team-1"},
        "description": description,
        "children": {
            "nodes": [
                {"id": f"child-{ident}", "identifier": ident, "state": {"type": stype}}
                for ident, stype in children
            ]
        },
    }


def _history(*entries: tuple[str, str | None, str | None, str | None]):
    """``(entry_id, from_type, to_type, actor_id)`` tuples -> history nodes."""
    nodes: list[dict[str, object]] = []
    base = datetime.now(tz=UTC) - timedelta(days=10)
    for index, (entry_id, from_type, to_type, actor_id) in enumerate(entries):
        nodes.append(
            {
                "id": entry_id,
                "createdAt": (base + timedelta(hours=index)).isoformat(),
                "actorId": actor_id,
                "botActor": None,
                "fromState": None if from_type is None else {"type": from_type},
                "toState": None if to_type is None else {"type": to_type},
            }
        )
    # Linear returns this connection newest-first; the handler must not depend
    # on the order it happens to arrive in.
    return list(reversed(nodes))


class FakeLinear:
    def __init__(
        self,
        issues: dict[str, dict[str, object]],
        histories: dict[str, list[dict[str, object]] | None] | None = None,
        post_flip_histories: dict[str, list[dict[str, object]]] | None = None,
    ) -> None:
        self._issues = issues
        self._histories = histories or {}
        self._post_flip = post_flip_histories or {}
        self.state_updates: list[tuple[str, str]] = []
        self.comments: list[tuple[str, str]] = []
        self.history_calls: list[str] = []

    async def fetch_issue(self, ticket_id: str) -> dict[str, object] | None:
        return self._issues.get(ticket_id)

    async def fetch_done_state_id(self, team_id: str) -> str | None:
        return "state-done"

    async def update_issue_state(self, issue_id: str, state_id: str) -> bool:
        self.state_updates.append((issue_id, state_id))
        return True

    async def create_comment(self, issue_id: str, body: str) -> bool:
        self.comments.append((issue_id, body))
        return True

    async def fetch_comment_bodies(self, issue_id: str) -> tuple[str, ...] | None:
        return tuple(body for target, body in self.comments if target == issue_id)

    async def fetch_issue_history(
        self, issue_id: str, page_size: int, max_pages: int
    ) -> tuple[list[dict[str, object]] | None, str]:
        self.history_calls.append(issue_id)
        # OMN-18056: the history moves when the WRITE happens, not on the
        # second read. The re-draw makes every flip a two-tick sequence, so a
        # read-counting double would serve the post-flip history to tick two's
        # PRE-write read and the bound readback would then find no segment the
        # pre-write read did not already have -- turning every legitimate flip
        # in this file into ERROR_READBACK_UNCONFIRMED.
        if (
            any(target == issue_id for target, _state in self.state_updates)
            and issue_id in self._post_flip
        ):
            return self._post_flip[issue_id], ""
        history = self._histories.get(issue_id, [])
        if history is None:
            return None, "history unreadable"
        return history, ""


def _gh_fake(
    companions: list[dict[str, object]],
    files_by_pr: dict[int, list[str]],
    product_prs: dict[str, dict[str, object]] | None = None,
):
    product_prs = product_prs or {}

    async def run_gh(args: list[str], timeout: float):
        path = args[2]
        if "/files" in path:
            number = int(path.split("/pulls/")[1].split("/files")[0])
            return [{"filename": f} for f in files_by_pr.get(number, [])], ""
        if "/pulls/" in path and "state=closed" not in path:
            key = path.split("repos/", 1)[1]
            if key in product_prs:
                return product_prs[key], ""
            return None, f"no such PR: {key}"
        page = int(path.rsplit("page=", 1)[1])
        return (companions, "") if page == 1 else ([], "")

    return run_gh


def _dod_fake(receipt: dict[str, object]):
    async def run_dod(ticket_id: str, cwd: str, timeout: int):
        return receipt, 0, ""

    return run_dod


def _request(**overrides: object) -> ModelEvidenceAutocloseSweepRequest:
    payload: dict[str, object] = {
        "correlation_id": uuid4(),
        "occ_repo": _OCC_REPO,
        "lookback_hours": 24,
        "apply": True,
    }
    payload.update(overrides)
    return ModelEvidenceAutocloseSweepRequest(**payload)


def _handler(linear: FakeLinear, gh, dod) -> HandlerEvidenceAutocloseSweep:
    return HandlerEvidenceAutocloseSweep(
        linear_client=linear,  # type: ignore[arg-type]
        autoclose_disabled=False,
        run_gh_command=gh,
        run_dod_verify_command=dod,
    )


def _clean_product_prs() -> dict[str, dict[str, object]]:
    return {
        "OmniNode-ai/omnibase_infra/pulls/3194": {
            "title": "fix(OMN-17872): the disk guard halts on free space",
            "user": {"login": "jonahgabriel", "type": "User"},
        }
    }


# ------------------------------------------------- (a) children conjunct ---


@pytest.mark.asyncio
class TestChildrenConjunct:
    """OMN-17658 F-R5-7. A parent is not Done while its decomposition is open."""

    async def test_a_parent_with_open_children_is_refused_as_skipped_has_children(
        self,
    ) -> None:
        linear = FakeLinear(
            issues={
                "OMN-8525": _issue(
                    identifier="OMN-8525",
                    children=(("OMN-8526", "started"), ("OMN-8527", "completed")),
                )
            },
            histories={"issue-1": []},
        )
        handler = _handler(
            linear,
            _gh_fake(
                [_merged_companion(9001, "OMN-8525")],
                {9001: ["contracts/OMN-8525.yaml"]},
                _clean_product_prs(),
            ),
            _dod_fake(_flip_clearing_receipt()),
        )

        result = await handler.handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.SKIPPED_HAS_CHILDREN
        ]
        assert result.tickets_flipped == 0
        # The refusal names the open child, not merely the count: a receipt
        # that says "has children" is not actionable by whoever reads it.
        assert "OMN-8526" in result.outcomes[0].reason
        # And it is a CONJUNCT, not a replacement for the verifier: the closed
        # child must not appear as a reason to refuse.
        assert "OMN-8527" not in result.outcomes[0].reason
        assert linear.state_updates == []

    async def test_a_parent_whose_children_are_all_done_still_flips(self) -> None:
        """The narrowing must not swallow the ordinary case (OMN-17934 AC3)."""
        linear = FakeLinear(
            issues={
                "OMN-8525": _issue(
                    identifier="OMN-8525",
                    children=(("OMN-8526", "completed"), ("OMN-8527", "canceled")),
                )
            },
            histories={"issue-1": []},
            post_flip_histories={
                "issue-1": _history(("e-flip", "started", "completed", "actor-bot"))
            },
        )
        handler = _handler(
            linear,
            _gh_fake(
                [_merged_companion(9001, "OMN-8525")],
                {9001: ["contracts/OMN-8525.yaml"]},
                _clean_product_prs(),
            ),
            _dod_fake(_flip_clearing_receipt()),
        )

        # OMN-18056: the first eligible observation arms the re-draw; the
        # flip is the second tick on the same fingerprint.
        await handler.handle(_request())
        result = await handler.handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.FLIPPED
        ]
        assert result.tickets_flipped == 1

    async def test_an_unreadable_children_connection_fails_closed(self) -> None:
        """A payload the guard cannot interpret is an error, never 'no children'."""
        issue = _issue(identifier="OMN-8525")
        del issue["children"]
        linear = FakeLinear(issues={"OMN-8525": issue}, histories={"issue-1": []})
        handler = _handler(
            linear,
            _gh_fake(
                [_merged_companion(9001, "OMN-8525")],
                {9001: ["contracts/OMN-8525.yaml"]},
                _clean_product_prs(),
            ),
            _dod_fake(_flip_clearing_receipt()),
        )

        result = await handler.handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.ERROR_LINEAR_API
        ]
        assert linear.state_updates == []


# ------------------------------------------------- (b) recurrence fence ---


class TestRecurringProductPrDiscriminator:
    """The discriminator is derived from the real PRs, and it is a conjunction."""

    def test_the_companion_title_yields_its_product_pr(self) -> None:
        assert _product_pr_ref(
            "evidence(OMN-17292): OCC companion for OmniNode-ai/omnibase_infra#3192"
        ) == ("OmniNode-ai/omnibase_infra", 3192)

    def test_a_companion_naming_no_product_pr_yields_none(self) -> None:
        # OCC observation appends carry no product PR at all; the fence must
        # decline to classify rather than guess.
        assert (
            _product_pr_ref(
                "evidence(OMN-14888): OCC observation append (782f70b__v1__run1.yaml)"
            )
            is None
        )

    def test_the_measured_pin_bump_pr_matches(self) -> None:
        assert _is_recurring_bot_product_pr(
            {
                "title": _PIN_BUMP_TITLE,
                "user": {"login": "onexbot-occ-writer[bot]", "type": "Bot"},
            }
        )

    def test_a_human_pr_never_matches_even_on_the_title(self) -> None:
        # The conjunction is what stops a human PR that happens to say
        # "advance ... contract pin" from being fenced off.
        assert not _is_recurring_bot_product_pr(
            {
                "title": "fix(OMN-1): advance the omnimarket contract pin by hand",
                "user": {"login": "jonahgabriel", "type": "User"},
            }
        )

    def test_a_bot_pr_of_another_shape_never_matches(self) -> None:
        assert not _is_recurring_bot_product_pr(
            {
                "title": "chore(deps): bump kafka-python from 2.3.1 to 2.3.2",
                "user": {"login": "dependabot[bot]", "type": "Bot"},
            }
        )


@pytest.mark.asyncio
class TestRecurrenceFence:
    async def test_the_omn_17292_shape_is_refused_not_flipped(self) -> None:
        """OMN-17934 AC2, replayed against the real titles and author identity."""
        linear = FakeLinear(
            issues={"OMN-17292": _issue(identifier="OMN-17292")},
            histories={"issue-1": []},
        )
        handler = _handler(
            linear,
            _gh_fake(
                [
                    _merged_companion(
                        8224, "OMN-17292", "OmniNode-ai/omnibase_infra#3192"
                    )
                ],
                {8224: ["contracts/OMN-17292.yaml"]},
                {
                    "OmniNode-ai/omnibase_infra/pulls/3192": {
                        "title": _PIN_BUMP_TITLE,
                        "user": {"login": "onexbot-occ-writer[bot]", "type": "Bot"},
                    }
                },
            ),
            _dod_fake(_flip_clearing_receipt()),
        )

        result = await handler.handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.SKIPPED_RECURRING_COMPANION
        ]
        assert result.tickets_flipped == 0
        assert linear.state_updates == []
        assert "3192" in result.outcomes[0].reason

    async def test_a_ticket_previously_done_and_reopened_is_refused(self) -> None:
        """The second shape: somebody already disagreed with a close."""
        linear = FakeLinear(
            issues={"OMN-17292": _issue(identifier="OMN-17292")},
            histories={
                "issue-1": _history(
                    ("h1", "started", "completed", None),
                    ("h2", "completed", "started", "human-actor-uuid"),
                )
            },
        )
        handler = _handler(
            linear,
            _gh_fake(
                [_merged_companion(8300, "OMN-17292")],
                {8300: ["contracts/OMN-17292.yaml"]},
                _clean_product_prs(),
            ),
            _dod_fake(_flip_clearing_receipt()),
        )

        result = await handler.handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.SKIPPED_PRIOR_REVERT
        ]
        assert linear.state_updates == []

    async def test_an_unreadable_history_fails_closed(self) -> None:
        linear = FakeLinear(
            issues={"OMN-1": _issue(identifier="OMN-1")},
            histories={"issue-1": None},
        )
        handler = _handler(
            linear,
            _gh_fake(
                [_merged_companion(8300, "OMN-1")],
                {8300: ["contracts/OMN-1.yaml"]},
                _clean_product_prs(),
            ),
            _dod_fake(_flip_clearing_receipt()),
        )

        result = await handler.handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.ERROR_LINEAR_API
        ]
        assert linear.state_updates == []


# ------------------------------------------------------ (c) auto-disarm ---


@pytest.mark.asyncio
class TestAutoDisarm:
    async def test_a_prior_revert_holds_that_ticket_and_disarms_nothing(self) -> None:
        """OMN-16106 D3. The fence WORKING costs the run nothing.

        This assertion is the inverse of the one it replaces. A prior-revert
        skip is the closer declining to re-assert a verdict a person reversed;
        the mechanism was not overruled, it deferred. Treating that as grounds
        to stop the whole run conflated "the closer overrode a human" with "the
        closer correctly refused to override a human", and only the first of
        those costs the mechanism its standing.

        Measured cost of the conflation: run 34008532058 flipped OMN-17556, a
        person adjudicated it back at 2026-09-06T03:22:47Z, and from 03:36Z
        every scheduled tick recomputed the identical refusal (fingerprint
        dd4745aa23025542) from unchanged evidence and disarmed the fleet with
        it. The disarm is in-run state recomputed per tick, not a persisted
        variable, so nothing decayed it: runs 34009400454 and 34010596242 both
        reported `tickets_flipped: 0` with every other candidate
        `skipped_disarmed`.
        """
        linear = FakeLinear(
            issues={
                "OMN-17292": _issue(issue_id="issue-a", identifier="OMN-17292"),
                "OMN-17872": _issue(issue_id="issue-b", identifier="OMN-17872"),
            },
            histories={
                "issue-a": _history(
                    ("h1", "started", "completed", None),
                    ("h2", "completed", "started", "human-actor-uuid"),
                ),
                "issue-b": [],
            },
            post_flip_histories={
                "issue-b": _history(("e-b", "started", "completed", "bot"))
            },
        )
        # Companions are offered newest-first by the enumerator; 8300 (the
        # reverted ticket) is scanned first and must NOT affect 8301.
        handler = _handler(
            linear,
            _gh_fake(
                [
                    _merged_companion(8300, "OMN-17292"),
                    _merged_companion(8301, "OMN-17872"),
                ],
                {
                    8300: ["contracts/OMN-17292.yaml"],
                    8301: ["contracts/OMN-17872.yaml"],
                },
                _clean_product_prs(),
            ),
            _dod_fake(_flip_clearing_receipt()),
        )

        # OMN-18056: the first eligible observation arms the re-draw; the
        # flip is the second tick on the same fingerprint.
        await handler.handle(_request())
        result = await handler.handle(_request())

        assert result.mode is EnumEvidenceAutocloseMode.APPLY_DISPATCHED
        assert result.disarm_triggered_by == ""
        assert result.disarm_reason == ""
        decisions = [o.decision for o in result.outcomes]
        # The refused ticket is held, and ONLY it. This fixture is refused by
        # the integration-authored-Done branch, which sits ahead of the
        # OMN-18056 positive fence, so the decision and the reason are both
        # unchanged by this ticket -- what changed is only that the run under
        # test is now the second tick.
        assert decisions[0] is EnumEvidenceAutocloseDecision.SKIPPED_PRIOR_REVERT
        assert result.outcomes[0].ticket_id == "OMN-17292"
        assert not result.outcomes[0].flip_reverted_during_run
        # Every other candidate reaches the decision it would have reached had
        # OMN-17292 never been enumerated -- the positive control on the
        # narrowing, without which "nothing disarmed" is also satisfied by
        # "nothing was adjudicated".
        assert decisions[1] is EnumEvidenceAutocloseDecision.FLIPPED
        assert EnumEvidenceAutocloseDecision.SKIPPED_DISARMED not in decisions
        assert linear.state_updates == [("issue-b", "state-done")]
        assert result.tickets_flipped == 1

    async def test_a_flip_reverted_inside_its_own_readback_window_disarms(
        self,
    ) -> None:
        """OMN-16106 D3. The ONE shape that still earns a fleet disarm.

        This run wrote a Done and its own readback then saw the ticket moved
        back out of a completed state. The fences did not hold -- the closer
        overrode a person and was overruled back while the run was still
        going -- so the remaining candidates get no further writes.
        """
        linear = FakeLinear(
            issues={
                "OMN-17292": _issue(issue_id="issue-a", identifier="OMN-17292"),
                "OMN-17872": _issue(issue_id="issue-b", identifier="OMN-17872"),
            },
            histories={"issue-a": [], "issue-b": []},
            post_flip_histories={
                # The flip's own segment (`e-a`), and a reversal NEWER than it.
                "issue-a": _history(
                    ("e-a", "started", "completed", "bot"),
                    ("e-a-undo", "completed", "started", "human-actor-uuid"),
                ),
                "issue-b": _history(("e-b", "started", "completed", "bot")),
            },
        )
        handler = _handler(
            linear,
            _gh_fake(
                [
                    _merged_companion(8300, "OMN-17292"),
                    _merged_companion(8301, "OMN-17872"),
                ],
                {
                    8300: ["contracts/OMN-17292.yaml"],
                    8301: ["contracts/OMN-17872.yaml"],
                },
                _clean_product_prs(),
            ),
            _dod_fake(_flip_clearing_receipt()),
        )

        # OMN-18056: the first eligible observation arms the re-draw; the
        # flip is the second tick on the same fingerprint.
        await handler.handle(_request())
        result = await handler.handle(_request())

        decisions = [o.decision for o in result.outcomes]
        # The first candidate genuinely flipped -- the write happened and the
        # readback confirmed it -- and was then reverted under this run.
        assert decisions[0] is EnumEvidenceAutocloseDecision.FLIPPED
        assert result.outcomes[0].flip_reverted_during_run
        assert result.mode is EnumEvidenceAutocloseMode.DISARMED
        assert result.disarm_triggered_by == "OMN-17292"
        assert "readback" in result.disarm_reason
        # And the rest of the run writes nothing.
        assert decisions[1] is EnumEvidenceAutocloseDecision.SKIPPED_DISARMED
        assert linear.state_updates == [("issue-a", "state-done")]

    async def test_an_older_reversal_is_not_a_during_run_revert(self) -> None:
        """`_revert_entry_since` stops at the flip's own segment.

        A ticket that was reopened at some point in its past and has since been
        re-proven is the ordinary case the prior-revert fence already
        adjudicates. Reading its history as "reverted during this run" would
        reinstate the same over-broad attribution from the other direction.
        """
        linear = FakeLinear(
            issues={"OMN-17872": _issue(issue_id="issue-b", identifier="OMN-17872")},
            histories={"issue-b": []},
            post_flip_histories={
                "issue-b": _history(
                    # An OLD reversal, then this run's flip on top of it.
                    ("old-undo", "completed", "started", "human-actor-uuid"),
                    ("e-b", "started", "completed", "bot"),
                )
            },
        )
        handler = _handler(
            linear,
            _gh_fake(
                [_merged_companion(8301, "OMN-17872")],
                {8301: ["contracts/OMN-17872.yaml"]},
                _clean_product_prs(),
            ),
            _dod_fake(_flip_clearing_receipt()),
        )

        # OMN-18056: the first eligible observation arms the re-draw; the
        # flip is the second tick on the same fingerprint.
        await handler.handle(_request())
        result = await handler.handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.FLIPPED
        ]
        assert not result.outcomes[0].flip_reverted_during_run
        assert result.disarm_triggered_by == ""

    async def test_the_persisted_marker_disarms_before_the_first_candidate(
        self,
    ) -> None:
        """The next scheduled run refuses to apply until an operator re-arms."""
        linear = FakeLinear(
            issues={"OMN-17872": _issue(identifier="OMN-17872")},
            histories={"issue-1": []},
        )
        handler = _handler(
            linear,
            _gh_fake(
                [_merged_companion(8301, "OMN-17872")],
                {8301: ["contracts/OMN-17872.yaml"]},
                _clean_product_prs(),
            ),
            _dod_fake(_flip_clearing_receipt()),
        )

        result = await handler.handle(_request(disarmed_by_ticket="OMN-17292"))

        assert result.mode is EnumEvidenceAutocloseMode.DISARMED
        assert result.disarm_triggered_by == "OMN-17292"
        assert linear.state_updates == []
        # A disarmed run is NOT a halted run: it still reaches every decision,
        # which is the evidence an operator needs to decide whether to re-arm.
        assert result.companions_scanned == 1
        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.SKIPPED_DISARMED
        ]


# ------------------ (c2) the measured verdict shapes, OMN-16106 D3 ---


def _receipt(
    *,
    total: int,
    verified: int,
    non_probative: int,
    behavior: int = 1,
    binds_ac: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """A dod_verify receipt with the counters spelled out by the caller.

    OMN-18056: ``binds_ac`` replaces the default single-`AC1` binding for the
    fixtures whose bodies label several criteria. Several criteria binding to
    ONE verified check is honest and common -- `verified_count` counts checks,
    not criteria -- and it is the shape the counting rules still have to
    refute, which is why they are not made redundant here.
    """
    receipt = _flip_clearing_receipt()
    verdict = receipt["result"]
    assert isinstance(verdict, dict)
    if binds_ac is not None:
        verdict["checks"] = [
            {
                "evidence_id": "omn18056-bound-check",
                "status": "verified",
                "proof_class": "behavior",
                "binds_ac": list(binds_ac),
            }
        ]
    verdict.update(
        {
            "total_checks": total,
            "verified_count": verified,
            "failed_count": 0,
            "non_probative_count": non_probative,
            "behavior_proving_count": behavior,
        }
    )
    return receipt


# The two descriptions, transcribed to their load-bearing shape: the heading
# spelling, the bullet style, and the item count. OMN-17556 writes four PROSE
# bullets under `## Acceptance`; OMN-17976 writes four NUMBERED items under
# `## Acceptance criteria`.
_OMN_17556_DESCRIPTION = """## Target shape

Each binding declares a store path in place of an env-var slot.

## Acceptance

* `grep -rn '_DB_URL'` returns zero env-var materializations of a resolved DSN.
* All four bindings connect on onex-dev with `current_user` attestation passing.
* A CI gate rejects reintroduction of a materialized `*_DB_URL` env.
* The `.201` compose lanes carry parity with the k8s resolver change.

## Out of scope

Credential rotation policy.
"""

_OMN_17976_DESCRIPTION = """## The cause

The materialise step exports `GIT_ASKPASS` only within that step.

## Acceptance criteria

1. AC1: The sweep step and the diagnose step both export a git credential path.
2. AC2: Proven by execution: a diagnostic run reaches a real verdict.
3. AC3: A test pins that the sweep step carries the credential wiring.
4. AC4: A scheduled tick reaches `behavior_proving_count > 0` unattended.
"""
# OMN-18056: the four items above carry `AC<n>` labels the measured body did
# not. They are the POSITIVE control, and after this ticket a criterion no
# contract can point at is held by the binding gate before any counting rule
# runs -- so an unlabelled control would assert a hold from the wrong
# conjunct and stop being a control at all. The item COUNT, the heading
# spelling and the bullet style -- the three things the counting rules read --
# are unchanged.


@pytest.mark.asyncio
class TestTheMeasuredVerdictShapes:
    """The two tickets the same sweep judged minutes apart on 2026-09-06.

    Neither shape has an unchecked checkbox, so rule 1 of the AC-coverage
    guard cannot fire on either. Under the old `total_checks` denominator the
    guard's second rule could not fire on OMN-17556 either (`4 > 22` is false),
    and it flipped -- a person reverted it two minutes later. The ONLY
    difference between the two tickets that the old predicate could see was
    markdown formatting.
    """

    async def test_the_omn_17556_shape_does_not_flip(self) -> None:
        """RED. 4/22 verified, 18 non-probative, prose bullets."""
        linear = FakeLinear(
            issues={
                "OMN-17556": _issue(
                    identifier="OMN-17556", description=_OMN_17556_DESCRIPTION
                )
            },
            histories={"issue-1": []},
        )
        handler = _handler(
            linear,
            _gh_fake(
                [_merged_companion(8327, "OMN-17556")],
                {8327: ["contracts/OMN-17556.yaml"]},
                _clean_product_prs(),
            ),
            _dod_fake(_receipt(total=22, verified=4, non_probative=18)),
        )

        result = await handler.handle(_request())

        # OMN-18056 changed WHICH conjunct holds it, and that change is the
        # measurement rather than a regression. The four prose bullets carry
        # no `AC<n>` label and the contract behind this verdict declares no
        # `binds_ac` -- true of all 8709 contracts in the corpus -- so the
        # binding gate reaches it first and names the criteria nothing proves
        # instead of reporting a ratio. It does not flip, which is the claim
        # this case exists to make.
        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.GAP_AC_UNBOUND
        ]
        assert result.tickets_flipped == 0
        assert linear.state_updates == []
        # The four prose bullets are named, so the receipt says WHAT was
        # unproven rather than only that something was.
        assert len(result.outcomes[0].uncovered_acceptance_criteria) == 4

        # CONTROL: the counting rule that used to hold this body still refutes
        # it on its own terms, and still states the arithmetic. The binding
        # gate has replaced no coverage; it runs ahead of it.
        coverage_reason, uncovered = _ac_coverage_gap(_OMN_17556_DESCRIPTION, 4, 18)
        assert "18" in coverage_reason
        assert len(uncovered) == 4

    async def test_the_omn_17976_shape_still_flips(self) -> None:
        """Positive control. 4/6 verified, 2 non-probative, all covered.

        Without this the RED above is also satisfied by a guard that holds
        everything, which would be a worse failure than the one it fixes.
        """
        linear = FakeLinear(
            issues={
                "OMN-17976": _issue(
                    identifier="OMN-17976", description=_OMN_17976_DESCRIPTION
                )
            },
            histories={"issue-1": []},
            post_flip_histories={
                "issue-1": _history(("e-1", "started", "completed", "bot"))
            },
        )
        handler = _handler(
            linear,
            _gh_fake(
                [_merged_companion(8328, "OMN-17976")],
                {8328: ["contracts/OMN-17976.yaml"]},
                _clean_product_prs(),
            ),
            _dod_fake(
                _receipt(
                    total=6,
                    verified=4,
                    non_probative=2,
                    binds_ac=("AC1", "AC2", "AC3", "AC4"),
                )
            ),
        )

        # OMN-18056: the first eligible observation arms the re-draw; the
        # flip is the second tick on the same fingerprint.
        await handler.handle(_request())
        result = await handler.handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.FLIPPED
        ]
        assert result.tickets_flipped == 1
        assert linear.state_updates == [("issue-1", "state-done")]


# -------------------------------- (d) flip budget + the bound readback ---


@pytest.mark.asyncio
class TestFlipBudgetAndReadback:
    async def test_max_flips_per_run_truncates_and_says_so(self) -> None:
        issues = {
            f"OMN-90{n}": _issue(issue_id=f"issue-{n}", identifier=f"OMN-90{n}")
            for n in range(3)
        }
        linear = FakeLinear(
            issues=issues,
            histories={f"issue-{n}": [] for n in range(3)},
            post_flip_histories={
                f"issue-{n}": _history(("e", "started", "completed", "bot"))
                for n in range(3)
            },
        )
        companions = [_merged_companion(8400 + n, f"OMN-90{n}") for n in range(3)]
        handler = _handler(
            linear,
            _gh_fake(
                companions,
                {8400 + n: [f"contracts/OMN-90{n}.yaml"] for n in range(3)},
                _clean_product_prs(),
            ),
            _dod_fake(_flip_clearing_receipt()),
        )

        # OMN-18056: the first eligible observation arms the re-draw; the
        # flip is the second tick on the same fingerprint.
        await handler.handle(_request(max_flips_per_run=2))
        result = await handler.handle(_request(max_flips_per_run=2))

        decisions = [o.decision for o in result.outcomes]
        assert decisions.count(EnumEvidenceAutocloseDecision.FLIPPED) == 2
        assert (
            decisions.count(EnumEvidenceAutocloseDecision.SKIPPED_FLIP_BUDGET_EXHAUSTED)
            == 1
        )
        assert result.tickets_flipped == 2
        assert result.flip_budget_remaining == 0
        assert len(linear.state_updates) == 2

    async def test_a_flip_carries_a_bound_readback(self) -> None:
        linear = FakeLinear(
            issues={"OMN-9000": _issue(identifier="OMN-9000")},
            histories={"issue-1": _history(("head-before", "backlog", "started", "u"))},
            post_flip_histories={
                "issue-1": _history(
                    ("head-before", "backlog", "started", "u"),
                    ("entry-after", "started", "completed", "bot"),
                )
            },
        )
        handler = _handler(
            linear,
            _gh_fake(
                [_merged_companion(8500, "OMN-9000")],
                {8500: ["contracts/OMN-9000.yaml"]},
                _clean_product_prs(),
            ),
            _dod_fake(_flip_clearing_receipt()),
        )

        # OMN-18056: the first eligible observation arms the re-draw; the
        # flip is the second tick on the same fingerprint.
        await handler.handle(_request())
        result = await handler.handle(_request())

        outcome = result.outcomes[0]
        assert outcome.decision is EnumEvidenceAutocloseDecision.FLIPPED
        assert outcome.pre_write_head_entry_id == "head-before"
        assert outcome.readback_entry_id == "entry-after"
        assert outcome.readback_entry_id != outcome.pre_write_head_entry_id
        assert outcome.verdict_fingerprint
        assert outcome.applied is True

    async def test_a_write_that_cannot_be_read_back_is_not_a_flip(self) -> None:
        """`issueUpdate: success` is the API agreeing, not the board changing."""
        linear = FakeLinear(
            issues={"OMN-9000": _issue(identifier="OMN-9000")},
            histories={"issue-1": _history(("head-before", "backlog", "started", "u"))},
            # No new completed segment after the write.
            post_flip_histories={
                "issue-1": _history(("head-before", "backlog", "started", "u"))
            },
        )
        handler = _handler(
            linear,
            _gh_fake(
                [_merged_companion(8500, "OMN-9000")],
                {8500: ["contracts/OMN-9000.yaml"]},
                _clean_product_prs(),
            ),
            _dod_fake(_flip_clearing_receipt()),
        )

        # OMN-18056: the first eligible observation arms the re-draw; the
        # flip is the second tick on the same fingerprint.
        await handler.handle(_request())
        result = await handler.handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.ERROR_READBACK_UNCONFIRMED
        ]
        assert result.tickets_flipped == 0

    async def test_a_dry_run_needs_no_readback_and_writes_nothing(self) -> None:
        linear = FakeLinear(
            issues={"OMN-9000": _issue(identifier="OMN-9000")},
            histories={"issue-1": []},
        )
        handler = _handler(
            linear,
            _gh_fake(
                [_merged_companion(8500, "OMN-9000")],
                {8500: ["contracts/OMN-9000.yaml"]},
                _clean_product_prs(),
            ),
            _dod_fake(_flip_clearing_receipt()),
        )

        # OMN-18056: a DRY-RUN writes nothing, so it can never arm its own
        # re-draw. The marker an APPLY tick would have left is seeded, so the
        # rehearsal previews the flip rather than a hold forever.
        linear.comments.append(
            (
                "issue-1",
                redraw_marker_comment(
                    total_checks=2, verified_count=2, behavior_proving_count=1
                ),
            )
        )
        result = await handler.handle(
            _request(apply=False, trigger=EnumEvidenceAutocloseTrigger.DISPATCH)
        )

        assert result.mode is EnumEvidenceAutocloseMode.DRY_RUN
        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.FLIPPED
        ]
        assert result.outcomes[0].applied is False
        assert linear.state_updates == []


# ------------------------------------ (e) scheduled_apply as the authority ---


@pytest.mark.asyncio
class TestScheduledApplyIsTheArmingAuthority:
    def _handler_for(self, linear: FakeLinear) -> HandlerEvidenceAutocloseSweep:
        return _handler(
            linear,
            _gh_fake(
                [_merged_companion(8600, "OMN-9000")],
                {8600: ["contracts/OMN-9000.yaml"]},
                _clean_product_prs(),
            ),
            _dod_fake(_flip_clearing_receipt()),
        )

    def _linear(self) -> FakeLinear:
        return FakeLinear(
            issues={"OMN-9000": _issue(identifier="OMN-9000")},
            histories={"issue-1": []},
            post_flip_histories={
                "issue-1": _history(("e", "started", "completed", "bot"))
            },
        )

    async def test_a_scheduled_run_under_a_disarmed_contract_reports_dry_run(
        self,
    ) -> None:
        """The authority is load-bearing: flipping it off stops the write."""
        linear = self._linear()
        result = await self._handler_for(linear).handle(
            _request(
                apply=False,
                trigger=EnumEvidenceAutocloseTrigger.SCHEDULE,
                scheduled_apply=False,
            )
        )
        assert result.mode is EnumEvidenceAutocloseMode.DRY_RUN
        assert result.dry_run is True
        assert linear.state_updates == []

    async def test_a_scheduled_run_under_an_armed_contract_applies(self) -> None:
        linear = self._linear()
        handler = self._handler_for(linear)
        request = _request(
            apply=False,
            trigger=EnumEvidenceAutocloseTrigger.SCHEDULE,
            scheduled_apply=True,
        )
        # OMN-18056: the first eligible observation arms the re-draw; the
        # flip is the second tick on the same fingerprint.
        await handler.handle(request)
        result = await handler.handle(request)
        assert result.mode is EnumEvidenceAutocloseMode.APPLY_SCHEDULED
        assert result.dry_run is False
        assert len(linear.state_updates) == 1

    async def test_a_dispatch_stays_a_dry_run_even_while_the_schedule_is_armed(
        self,
    ) -> None:
        """The rehearsal surface has to survive arming (OMN-17658 §3b)."""
        linear = self._linear()
        result = await self._handler_for(linear).handle(
            _request(
                apply=False,
                trigger=EnumEvidenceAutocloseTrigger.DISPATCH,
                scheduled_apply=True,
            )
        )
        assert result.mode is EnumEvidenceAutocloseMode.DRY_RUN
        assert linear.state_updates == []

    async def test_a_dispatch_with_apply_writes_regardless_of_the_contract(
        self,
    ) -> None:
        linear = self._linear()
        handler = self._handler_for(linear)
        request = _request(
            apply=True,
            trigger=EnumEvidenceAutocloseTrigger.DISPATCH,
            scheduled_apply=False,
        )
        # OMN-18056: the first eligible observation arms the re-draw; the
        # flip is the second tick on the same fingerprint.
        await handler.handle(request)
        result = await handler.handle(request)
        assert result.mode is EnumEvidenceAutocloseMode.APPLY_DISPATCHED
        assert len(linear.state_updates) == 1

    async def test_an_unnamed_trigger_is_not_the_schedule(self) -> None:
        """Fail-closed default: omission must never inherit unattended arming."""
        assert (
            ModelEvidenceAutocloseSweepRequest(correlation_id=uuid4()).trigger
            is EnumEvidenceAutocloseTrigger.DISPATCH
        )

    async def test_the_kill_switch_still_reports_halted_and_does_zero_io(self) -> None:
        linear = self._linear()
        handler = HandlerEvidenceAutocloseSweep(
            linear_client=linear,  # type: ignore[arg-type]
            autoclose_disabled=True,
            run_gh_command=_gh_fake([], {}, {}),
            run_dod_verify_command=_dod_fake(_flip_clearing_receipt()),
        )
        result = await handler.handle(
            _request(
                trigger=EnumEvidenceAutocloseTrigger.SCHEDULE, scheduled_apply=True
            )
        )
        assert result.kill_switch_engaged is True
        assert result.mode is EnumEvidenceAutocloseMode.HALTED
        assert result.companions_scanned == 0
        assert linear.state_updates == []
