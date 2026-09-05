# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16106 — the two live-gate defects the closer flipped OMN-17957 through.

Measured 2026-09-05, twice on one ticket:

  * D1. OMN-17957's AC-5 cites ``OmniNode-ai/knowledge-base-internal#125``,
    which was OPEN. The OMN-13856 done-flip guard reads exactly that citation
    and refused the interactive flip at 17:21:23Z with ``pr_not_merged``. The
    closer flipped it anyway at 17:35:59Z and again at 19:36:02Z (run
    33987388489, ``decision=flipped``), because the guard lives in an omniclaude
    ``PreToolUse`` hook on the Linear MCP tool and the closer writes through the
    Linear HTTP API from an Actions runner — it never crosses that seam.

  * D2. ``SKIPPED_PRIOR_REVERT`` exists (OMN-17934 shape 2) and did not fire on
    the 19:28:17Z audit revert. Its discriminator is that the Done being
    reverted was written with a NULL ``actorId`` ("an integration wrote it —
    which is what this sweep's LINEAR_API_KEY mutation is"). That premise is
    false. Read live from Linear at 2026-09-05T20:33Z, EVERY state-history
    entry on OMN-17957 — including the sweep's own 19:36:02.430Z flip — carries
    ``actorId 7a850ce1-f95e-431f-b4e3-62f7449f04c0``: the API key is a personal
    key and Linear attributes its writes to that user. So the fence's null-actor
    half can never match a flip THIS closer writes, and the fence is
    structurally dead for the population it exists to protect.

Every test here asserts a NARROWING. Not one changes what counts as proof: the
OMN-16821 denominator equality, the OMN-15911 behaviour conjunct and the
OMN-16736 AC-coverage re-read are untouched, and a candidate that clears them
all still flips unless one of the two conjuncts below refuses it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    _FLIP_COMMENT_CLASS_MARKER,
    HandlerEvidenceAutocloseSweep,
    _cited_product_pr_refs,
    _verdict_fingerprint,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_decision import (
    EnumEvidenceAutocloseDecision,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_request import (
    ModelEvidenceAutocloseSweepRequest,
)

pytestmark = pytest.mark.unit

_OCC_REPO = "OmniNode-ai/onex_change_control"
_DOD_VERIFY_STATE_MODEL = (
    "omnimarket.nodes.node_dod_verify.models.model_dod_verify_state.ModelDodVerifyState"
)
# The real citation OMN-17957's AC-5 carries.
_CITED_PR_URL = "https://github.com/OmniNode-ai/knowledge-base-internal/pull/125"
_CITED_PR_KEY = "OmniNode-ai/knowledge-base-internal/pulls/125"

# The fingerprint the flip-clearing receipt below produces. It is the SAME
# statement across two runs — which is the whole point of D2: the evidence did
# not change between the reverted flip and the re-flip.
_FLIP_FINGERPRINT = _verdict_fingerprint(
    total_checks=2,
    verified_count=2,
    failed_count=0,
    non_probative_count=0,
    behavior_proving_count=1,
)


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
    verdict: dict[str, object] = {
        "correlation_id": str(uuid4()),
        "ticket_id": "OMN-0000",
        "status": "verified",
        "dry_run": False,
        "checks": [],
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
    description: str | None = None,
    attachment_urls: tuple[str, ...] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": issue_id,
        "identifier": identifier,
        "state": {"id": "s1", "name": "In Progress", "type": "started"},
        "labels": {"nodes": []},
        "team": {"id": "team-1"},
        "description": description,
        "children": {"nodes": []},
    }
    if attachment_urls is not None:
        payload["attachments"] = {"nodes": [{"url": u} for u in attachment_urls]}
    return payload


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
                "fromState": None if from_type is None else {"type": from_type},
                "toState": None if to_type is None else {"type": to_type},
            }
        )
    return list(reversed(nodes))


# The OMN-17957 shape, with the actor ids AS MEASURED: the closer's own flip
# carries a real actor id, exactly like the human revert that followed it.
_HUMAN_ACTOR = "7a850ce1-f95e-431f-b4e3-62f7449f04c0"


def _reverted_closer_flip_history():
    return _history(
        ("e-start", "backlog", "started", _HUMAN_ACTOR),
        ("e-closer-flip", "started", "completed", _HUMAN_ACTOR),
        ("e-audit-revert", "completed", "started", _HUMAN_ACTOR),
    )


def _flip_comment(fingerprint: str) -> str:
    return (
        f"{_FLIP_COMMENT_CLASS_MARKER}\n"
        "Automatic Done flip (OMN-16106 evidence autoclose sweep).\n\n"
        "Merged evidence companion: https://github.com/OmniNode-ai/"
        "onex_change_control/pull/8281\n"
        f"Verdict fingerprint {fingerprint}."
    )


class FakeLinear:
    def __init__(
        self,
        issues: dict[str, dict[str, object]],
        histories: dict[str, list[dict[str, object]] | None] | None = None,
        post_flip_histories: dict[str, list[dict[str, object]]] | None = None,
        preexisting_comments: dict[str, tuple[str, ...] | None] | None = None,
    ) -> None:
        self._issues = issues
        self._histories = histories or {}
        self._post_flip = post_flip_histories or {}
        self._preexisting = preexisting_comments or {}
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
        if issue_id in self._preexisting:
            prior = self._preexisting[issue_id]
            if prior is None:
                return None
        else:
            prior = ()
        return tuple(prior) + tuple(
            body for target, body in self.comments if target == issue_id
        )

    async def fetch_issue_history(
        self, issue_id: str, page_size: int, max_pages: int
    ) -> tuple[list[dict[str, object]] | None, str]:
        self.history_calls.append(issue_id)
        if self.history_calls.count(issue_id) > 1 and issue_id in self._post_flip:
            return self._post_flip[issue_id], ""
        history = self._histories.get(issue_id, [])
        if history is None:
            return None, "history unreadable"
        return history, ""


def _gh_fake(
    companions: list[dict[str, object]],
    files_by_pr: dict[int, list[str]],
    product_prs: dict[str, dict[str, object]] | None = None,
    failing_paths: frozenset[str] = frozenset(),
):
    product_prs = product_prs or {}

    async def run_gh(args: list[str], timeout: float):
        path = args[2]
        if "/files" in path:
            number = int(path.split("/pulls/")[1].split("/files")[0])
            return [{"filename": f} for f in files_by_pr.get(number, [])], ""
        if "/pulls/" in path and "state=closed" not in path:
            key = path.split("repos/", 1)[1]
            if key in failing_paths:
                return None, "gh api: 502 Bad Gateway"
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


def _bound_product_pr() -> dict[str, object]:
    return {
        "OmniNode-ai/omnibase_infra/pulls/3194": {
            "title": "fix(OMN-17957): the guard refuses an uncited rotation",
            "user": {"login": "jonahgabriel", "type": "User"},
        }
    }


def _pr(state: str, merged: bool) -> dict[str, object]:
    return {
        "number": 125,
        "state": state,
        "merged": merged,
        "merged_at": "2026-09-05T20:15:40Z" if merged else None,
        "html_url": _CITED_PR_URL,
        "title": "docs(OMN-17926): beta PRD v0.5",
        "user": {"login": "jonahgabriel", "type": "User"},
    }


# ---------------------------------------------- (D1) cited-PR merge check ---


@pytest.mark.asyncio
class TestCitedProductPrMustBeMerged:
    """The OMN-13856 guard's `pr_not_merged` refusal, replicated in the closer."""

    async def test_a_cited_open_pr_holds_the_flip_with_the_gates_reason(self) -> None:
        linear = FakeLinear(
            issues={
                "OMN-17957": _issue(
                    identifier="OMN-17957",
                    description=f"AC-5 — the INV lands on {_CITED_PR_URL}.",
                )
            },
            histories={"issue-1": []},
        )
        product = _bound_product_pr()
        product[_CITED_PR_KEY] = _pr("open", merged=False)
        handler = _handler(
            linear,
            _gh_fake(
                [_merged_companion(8281, "OMN-17957")],
                {8281: ["contracts/OMN-17957.yaml"]},
                product,
            ),
            _dod_fake(_flip_clearing_receipt()),
        )

        result = await handler.handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.SKIPPED_REFERENCED_PR_UNMERGED
        ]
        assert result.tickets_flipped == 0
        # The gate's own reason word, so the two surfaces are greppable together.
        assert "pr_not_merged" in result.outcomes[0].reason
        # And the refusal NAMES the citation, or nobody can act on it.
        assert "knowledge-base-internal#125" in result.outcomes[0].reason
        assert linear.state_updates == []

    async def test_the_same_ticket_flips_once_the_cited_pr_is_merged(self) -> None:
        """The positive control: the conjunct clears the moment the PR lands."""
        linear = FakeLinear(
            issues={
                "OMN-17957": _issue(
                    identifier="OMN-17957",
                    description=f"AC-5 — the INV lands on {_CITED_PR_URL}.",
                )
            },
            histories={"issue-1": []},
            post_flip_histories={
                "issue-1": _history(("e-flip", "started", "completed", _HUMAN_ACTOR))
            },
        )
        product = _bound_product_pr()
        product[_CITED_PR_KEY] = _pr("closed", merged=True)
        handler = _handler(
            linear,
            _gh_fake(
                [_merged_companion(8281, "OMN-17957")],
                {8281: ["contracts/OMN-17957.yaml"]},
                product,
            ),
            _dod_fake(_flip_clearing_receipt()),
        )

        result = await handler.handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.FLIPPED
        ]
        assert result.tickets_flipped == 1

    async def test_an_open_occ_evidence_companion_reference_never_blocks(self) -> None:
        """OMN-14641: a receipt companion is not the shipped work, either way."""
        linear = FakeLinear(
            issues={
                "OMN-17957": _issue(
                    identifier="OMN-17957",
                    description=(
                        "Evidence rides on https://github.com/OmniNode-ai/"
                        "onex_change_control/pull/9999."
                    ),
                )
            },
            histories={"issue-1": []},
            post_flip_histories={
                "issue-1": _history(("e-flip", "started", "completed", _HUMAN_ACTOR))
            },
        )
        handler = _handler(
            linear,
            _gh_fake(
                [_merged_companion(8281, "OMN-17957")],
                {8281: ["contracts/OMN-17957.yaml"]},
                _bound_product_pr(),
            ),
            _dod_fake(_flip_clearing_receipt()),
        )

        result = await handler.handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.FLIPPED
        ]

    async def test_an_unmerged_pr_reachable_only_as_a_linear_attachment_holds(
        self,
    ) -> None:
        """The OMN-14582 shape: the linked PR is never cited in the body."""
        linear = FakeLinear(
            issues={
                "OMN-17957": _issue(
                    identifier="OMN-17957",
                    description="No PR is named anywhere in this body.",
                    attachment_urls=(_CITED_PR_URL,),
                )
            },
            histories={"issue-1": []},
        )
        product = _bound_product_pr()
        product[_CITED_PR_KEY] = _pr("open", merged=False)
        handler = _handler(
            linear,
            _gh_fake(
                [_merged_companion(8281, "OMN-17957")],
                {8281: ["contracts/OMN-17957.yaml"]},
                product,
            ),
            _dod_fake(_flip_clearing_receipt()),
        )

        result = await handler.handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.SKIPPED_REFERENCED_PR_UNMERGED
        ]
        assert linear.state_updates == []

    async def test_a_bare_pr_number_with_no_resolvable_repo_holds(self) -> None:
        """Unresolvable is not merged. The gate blocks on it; so does this."""
        linear = FakeLinear(
            issues={
                "OMN-17957": _issue(
                    identifier="OMN-17957",
                    description="Shipped in PR #4242.",
                )
            },
            histories={"issue-1": []},
        )
        handler = _handler(
            linear,
            _gh_fake(
                [_merged_companion(8281, "OMN-17957")],
                {8281: ["contracts/OMN-17957.yaml"]},
                _bound_product_pr(),
            ),
            _dod_fake(_flip_clearing_receipt()),
        )

        result = await handler.handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.SKIPPED_REFERENCED_PR_UNMERGED
        ]
        assert "4242" in result.outcomes[0].reason
        assert linear.state_updates == []

    async def test_a_github_read_failure_fails_closed_as_a_github_error(self) -> None:
        """ "I could not check" must never resolve to "so I will flip it"."""
        linear = FakeLinear(
            issues={
                "OMN-17957": _issue(
                    identifier="OMN-17957",
                    description=f"AC-5 — the INV lands on {_CITED_PR_URL}.",
                )
            },
            histories={"issue-1": []},
        )
        handler = _handler(
            linear,
            _gh_fake(
                [_merged_companion(8281, "OMN-17957")],
                {8281: ["contracts/OMN-17957.yaml"]},
                _bound_product_pr(),
                failing_paths=frozenset({_CITED_PR_KEY}),
            ),
            _dod_fake(_flip_clearing_receipt()),
        )

        result = await handler.handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.ERROR_GITHUB_API
        ]
        assert linear.state_updates == []

    async def test_a_ticket_citing_nothing_is_unaffected(self) -> None:
        linear = FakeLinear(
            issues={
                "OMN-17957": _issue(
                    identifier="OMN-17957",
                    description="A decision-only ticket. No PR is cited.",
                )
            },
            histories={"issue-1": []},
            post_flip_histories={
                "issue-1": _history(("e-flip", "started", "completed", _HUMAN_ACTOR))
            },
        )
        handler = _handler(
            linear,
            _gh_fake(
                [_merged_companion(8281, "OMN-17957")],
                {8281: ["contracts/OMN-17957.yaml"]},
                _bound_product_pr(),
            ),
            _dod_fake(_flip_clearing_receipt()),
        )

        result = await handler.handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.FLIPPED
        ]


class TestCitedRefParsing:
    """The parser is pure, so its corpus is pinned directly."""

    def test_it_reads_urls_linear_rewritten_tags_and_bare_numbers(self) -> None:
        refs = _cited_product_pr_refs(
            "Landed in https://github.com/OmniNode-ai/omniclaude/pull/2107, "
            "with OmniNode-ai/knowledge-base-internal#125 still open, "
            "receipt OmniNode-ai/onex_change_control#8281, and PR #4242.",
            (),
        )
        assert ("OmniNode-ai/omniclaude", 2107) in refs
        assert ("OmniNode-ai/knowledge-base-internal", 125) in refs
        assert (None, 4242) in refs
        # The evidence companion is filtered, in BOTH spellings.
        assert all(repo != "OmniNode-ai/onex_change_control" for repo, _ in refs), refs

    def test_a_bare_hash_in_prose_is_not_a_pr_reference(self) -> None:
        """OMN-15025: `Rule #4` is prose, and a false hold is still a defect."""
        assert _cited_product_pr_refs("CLAUDE.md Rule #4 and issue #17", ()) == ()


# --------------------------------------------- (D2) prior-revert, working ---


@pytest.mark.asyncio
class TestPriorRevertFenceReadsTheMarkerNotTheActorId:
    """The fence must fire on a revert of a flip THIS closer wrote."""

    async def test_a_reverted_closer_flip_is_held_even_though_it_has_an_actor_id(
        self,
    ) -> None:
        linear = FakeLinear(
            issues={"OMN-17957": _issue(identifier="OMN-17957", description="Body.")},
            histories={"issue-1": _reverted_closer_flip_history()},
            preexisting_comments={"issue-1": (_flip_comment(_FLIP_FINGERPRINT),)},
        )
        handler = _handler(
            linear,
            _gh_fake(
                [_merged_companion(8281, "OMN-17957")],
                {8281: ["contracts/OMN-17957.yaml"]},
                _bound_product_pr(),
            ),
            _dod_fake(_flip_clearing_receipt()),
        )

        result = await handler.handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.SKIPPED_PRIOR_REVERT
        ]
        assert result.tickets_flipped == 0
        assert linear.state_updates == []
        # The refusal names the fingerprint it matched — the evidence did not
        # change, which is the whole reason the re-flip is refused.
        assert _FLIP_FINGERPRINT in result.outcomes[0].reason
        # And one such candidate disarms the rest of the run (OMN-17658).
        assert result.disarm_triggered_by == "OMN-17957"

    async def test_a_revert_with_no_closer_flip_comment_is_not_this_fences_business(
        self,
    ) -> None:
        """A human Done a human reopened is an ordinary ticket back in flight."""
        linear = FakeLinear(
            issues={"OMN-17957": _issue(identifier="OMN-17957", description="Body.")},
            histories={"issue-1": _reverted_closer_flip_history()},
            preexisting_comments={"issue-1": ("An ordinary human comment.",)},
            post_flip_histories={
                "issue-1": _history(("e-flip", "started", "completed", _HUMAN_ACTOR))
            },
        )
        handler = _handler(
            linear,
            _gh_fake(
                [_merged_companion(8281, "OMN-17957")],
                {8281: ["contracts/OMN-17957.yaml"]},
                _bound_product_pr(),
            ),
            _dod_fake(_flip_clearing_receipt()),
        )

        result = await handler.handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.FLIPPED
        ]

    async def test_a_changed_verdict_after_a_revert_may_close_again(self) -> None:
        """The hold is on RE-ASSERTING a verdict, not on the ticket forever."""
        linear = FakeLinear(
            issues={"OMN-17957": _issue(identifier="OMN-17957", description="Body.")},
            histories={"issue-1": _reverted_closer_flip_history()},
            preexisting_comments={"issue-1": (_flip_comment("0" * 16),)},
            post_flip_histories={
                "issue-1": _history(("e-flip", "started", "completed", _HUMAN_ACTOR))
            },
        )
        handler = _handler(
            linear,
            _gh_fake(
                [_merged_companion(8281, "OMN-17957")],
                {8281: ["contracts/OMN-17957.yaml"]},
                _bound_product_pr(),
            ),
            _dod_fake(_flip_clearing_receipt()),
        )

        result = await handler.handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.FLIPPED
        ]

    async def test_an_unreadable_comment_history_fails_closed(self) -> None:
        linear = FakeLinear(
            issues={"OMN-17957": _issue(identifier="OMN-17957", description="Body.")},
            histories={"issue-1": _reverted_closer_flip_history()},
            preexisting_comments={"issue-1": None},
        )
        handler = _handler(
            linear,
            _gh_fake(
                [_merged_companion(8281, "OMN-17957")],
                {8281: ["contracts/OMN-17957.yaml"]},
                _bound_product_pr(),
            ),
            _dod_fake(_flip_clearing_receipt()),
        )

        result = await handler.handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.ERROR_LINEAR_API
        ]
        assert linear.state_updates == []

    async def test_a_ticket_with_no_revert_at_all_never_reads_its_comments(
        self,
    ) -> None:
        """The extra read is scoped to candidates that HAVE been reverted."""
        linear = FakeLinear(
            issues={"OMN-17957": _issue(identifier="OMN-17957", description="Body.")},
            histories={
                "issue-1": _history(
                    ("e-start", "backlog", "started", _HUMAN_ACTOR),
                )
            },
            preexisting_comments={"issue-1": None},
            post_flip_histories={
                "issue-1": _history(("e-flip", "started", "completed", _HUMAN_ACTOR))
            },
        )
        handler = _handler(
            linear,
            _gh_fake(
                [_merged_companion(8281, "OMN-17957")],
                {8281: ["contracts/OMN-17957.yaml"]},
                _bound_product_pr(),
            ),
            _dod_fake(_flip_clearing_receipt()),
        )

        result = await handler.handle(_request())

        # An unreadable comment history did NOT fail this candidate closed,
        # which is only possible if the read never happened.
        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.FLIPPED
        ]
