# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18749 — a closed-unmerged citation and the successor the ticket declares.

A cited pull request that is CLOSED and never merged can never become merged,
so the cited-PR conjunct's hold was permanent and its advice was impossible.
Measured on OMN-18172: `omnibase_infra#3615` and `#3626` are both closed with a
null merge time, the work landed as `#3627`, and every tick refused the
candidate with "merging the cited PR is all this candidate needs".

The OMN-18233 predicate does not reach this shape. It proves supersession for a
dependency-cascade bump by reading a pin, and neither of those two is a bump —
clause 1 does not resolve, so they keep blocking. What this file pins is the
other half: a supersession the TICKET declares, accepted only when the
successor's own probe says merged, and refused, with a remedy, otherwise.

Every test here asserts a NARROWING or a refusal. Nothing in the file relaxes
what counts as proof for an OPEN citation, an unreadable probe, or a closed one
with nothing declared for it.
"""

from __future__ import annotations

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.declared_supersession import (
    parse_declared_supersessions,
    resolve_declared_supersession,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_decision import (
    EnumEvidenceAutocloseDecision,
)
from tests.unit.nodes.node_evidence_autoclose_sweep_effect.test_omn_16106_closer_flip_conjuncts import (
    _HUMAN_ACTOR,
    FakeLinear,
    _bound_product_pr,
    _dod_fake,
    _flip_clearing_receipt,
    _gh_fake,
    _handler,
    _history,
    _issue,
    _merged_companion,
    _request,
)

pytestmark = pytest.mark.unit

_REPO = "OmniNode-ai/omnibase_infra"
_CLOSED_A = 3615
_CLOSED_B = 3626
_SUCCESSOR = 3627

#: The line a closer would add to OMN-18172 to state where the work went.
_DECLARATION = (
    f"{_REPO}#{_CLOSED_A} and {_REPO}#{_CLOSED_B} are closed unmerged; "
    f"superseded by {_REPO}#{_SUCCESSOR}."
)

#: OMN-18172's shape: the citations reach the closer as Linear attachments, and
#: the body says nothing about any pull request.
_ATTACHMENTS = (
    f"https://github.com/{_REPO}/pull/{_CLOSED_A}",
    f"https://github.com/{_REPO}/pull/{_CLOSED_B}",
    f"https://github.com/{_REPO}/pull/{_SUCCESSOR}",
)


def _key(number: int) -> str:
    return f"{_REPO}/pulls/{number}"


def _pr_payload(number: int, state: str, merged_at: str | None) -> dict[str, object]:
    return {
        "number": number,
        "state": state,
        "merged": merged_at is not None,
        "merged_at": merged_at,
        "html_url": f"https://github.com/{_REPO}/pull/{number}",
        "title": "test(OMN-18172): AC4 attribution query proof",
        # No cascade provenance block, so the OMN-18233 predicate stops at
        # clause 1 without issuing a single call — which is the whole reason
        # this second predicate exists.
        "body": "## Summary\n\nThree integration tests prove the query works.",
        "user": {"login": "jonahgabriel", "type": "User"},
    }


def _product(
    *,
    successor_merged: bool = True,
    successor_state: str = "closed",
    successor_readable: bool = True,
) -> dict[str, dict[str, object]]:
    product = _bound_product_pr()
    product[_key(_CLOSED_A)] = _pr_payload(_CLOSED_A, "closed", None)
    product[_key(_CLOSED_B)] = _pr_payload(_CLOSED_B, "closed", None)
    if successor_readable:
        product[_key(_SUCCESSOR)] = _pr_payload(
            _SUCCESSOR,
            successor_state,
            "2026-09-16T16:56:24Z" if successor_merged else None,
        )
    return product


def _linear(description: str) -> FakeLinear:
    return FakeLinear(
        issues={
            "OMN-18172": _issue(
                identifier="OMN-18172",
                description=description,
                attachment_urls=_ATTACHMENTS,
            )
        },
        histories={"issue-1": []},
        post_flip_histories={
            "issue-1": _history(("e-flip", "started", "completed", _HUMAN_ACTOR))
        },
    )


def _built(linear: FakeLinear, product: dict[str, dict[str, object]]):
    return _handler(
        linear,
        _gh_fake(
            [_merged_companion(9974, "OMN-18172")],
            {9974: ["contracts/OMN-18172.yaml"]},
            product,
        ),
        _dod_fake(_flip_clearing_receipt()),
    )


class TestDeclarationParsing:
    """The pure half. It proves nothing; it reads what the ticket claims."""

    def test_one_line_binds_every_closed_reference_to_the_successor(self) -> None:
        declarations = parse_declared_supersessions(_DECLARATION)
        assert set(declarations) == {(_REPO, _CLOSED_A), (_REPO, _CLOSED_B)}
        for declaration in declarations.values():
            assert declaration.successor_repo == _REPO
            assert declaration.successor_number == _SUCCESSOR

    def test_a_full_url_on_either_side_binds(self) -> None:
        line = (
            f"https://github.com/{_REPO}/pull/{_CLOSED_A} superseded by "
            f"https://github.com/{_REPO}/pull/{_SUCCESSOR}"
        )
        declarations = parse_declared_supersessions(line)
        assert declarations[(_REPO, _CLOSED_A)].successor_number == _SUCCESSOR

    def test_the_phrase_set_is_directional(self) -> None:
        """A symmetric verb would record the MERGED one as superseded."""
        assert (
            parse_declared_supersessions(
                f"{_REPO}#{_SUCCESSOR} replaces {_REPO}#{_CLOSED_A}"
            )
            == {}
        )

    def test_a_right_side_naming_nothing_is_not_a_declaration(self) -> None:
        assert (
            parse_declared_supersessions(
                f"{_REPO}#{_CLOSED_A} was superseded by later work"
            )
            == {}
        )

    def test_a_right_side_that_gestures_but_resolves_to_nothing_is_recorded(
        self,
    ) -> None:
        """Recorded, not ignored — it holds, and the hold says why."""
        declarations = parse_declared_supersessions(
            f"{_REPO}#{_CLOSED_A} superseded by #40404"
        )
        assert declarations[(_REPO, _CLOSED_A)].successor_repo == ""

    def test_an_unqualified_left_side_binds_nothing(self) -> None:
        assert (
            parse_declared_supersessions(
                f"PR #{_CLOSED_A} superseded by {_REPO}#{_SUCCESSOR}"
            )
            == {}
        )


@pytest.mark.asyncio
class TestClosedCitationsOnTheCloser:
    """The conjunct itself, on OMN-18172's measured shape."""

    async def test_the_omn_18172_shape_is_held_when_nothing_is_declared(
        self,
    ) -> None:
        linear = _linear("No code has been changed for this issue.")
        result = await _built(linear, _product()).handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.SKIPPED_REFERENCED_PR_UNMERGED
        ]
        assert result.tickets_flipped == 0
        assert linear.state_updates == []
        reason = result.outcomes[0].reason
        assert f"{_REPO}#{_CLOSED_A}" in reason
        assert f"{_REPO}#{_CLOSED_B}" in reason

    async def test_the_hold_names_a_remedy_that_is_possible(self) -> None:
        """The refusal used to recommend merging something already closed."""
        linear = _linear("No code has been changed for this issue.")
        result = await _built(linear, _product()).handle(_request())
        reason = result.outcomes[0].reason

        assert "merging the cited PR is all this candidate needs" not in reason
        assert "can never merge" in reason
        assert "superseded by" in reason

    async def test_one_declaration_line_clears_both_closed_citations(self) -> None:
        linear = _linear("No code has been changed for this issue.\n\n" + _DECLARATION)
        handler = _built(linear, _product())

        armed = await handler.handle(_request())
        assert [o.decision for o in armed.outcomes] == [
            EnumEvidenceAutocloseDecision.SKIPPED_REDRAW_PENDING
        ]
        result = await handler.handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.FLIPPED
        ]
        assert result.tickets_flipped == 1

    async def test_a_declared_successor_that_is_open_does_not_satisfy(self) -> None:
        linear = _linear("No code has been changed for this issue.\n\n" + _DECLARATION)
        product = _product(successor_merged=False, successor_state="open")
        result = await _built(linear, product).handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.SKIPPED_REFERENCED_PR_UNMERGED
        ]
        assert "merged_at=null" in result.outcomes[0].reason

    async def test_an_unreadable_declared_successor_holds_as_a_github_error(
        self,
    ) -> None:
        """Unresolved, not refuted — and it holds either way.

        Naming a successor in the body makes it a CITATION as well, so the
        conjunct reads it before this predicate does and an unreadable one
        fails closed there, as a GitHub error. That is the same direction and
        a stricter decision, so the test asserts what actually happens rather
        than the outcome this predicate would have produced on its own. The
        predicate's own unreadable branch is pinned directly below.
        """
        unread = 3628
        linear = _linear(
            "No code has been changed for this issue.\n\n"
            f"{_REPO}#{_CLOSED_A} and {_REPO}#{_CLOSED_B} superseded by "
            f"{_REPO}#{unread}."
        )
        result = await _built(linear, _product()).handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.ERROR_GITHUB_API
        ]
        assert result.tickets_flipped == 0
        assert linear.state_updates == []

    async def test_the_predicate_itself_holds_on_an_unreadable_successor(
        self,
    ) -> None:
        async def _unreadable(args: list[str], timeout: float):
            return None, "gh api: 502 Bad Gateway"

        verdict = await resolve_declared_supersession(
            repo=_REPO,
            closed_pr_number=_CLOSED_A,
            description=_DECLARATION,
            run_gh_command=_unreadable,
            gh_timeout_seconds=30,
        )

        assert verdict.superseded is False
        assert "could not be read" in verdict.detail

    async def test_the_predicate_holds_when_the_ticket_declares_nothing(
        self,
    ) -> None:
        async def _never_called(args: list[str], timeout: float):
            raise AssertionError("no probe is issued without a declaration")

        verdict = await resolve_declared_supersession(
            repo=_REPO,
            closed_pr_number=_CLOSED_A,
            description="No code has been changed for this issue.",
            run_gh_command=_never_called,
            gh_timeout_seconds=30,
        )

        assert verdict.superseded is False
        assert "declares no successor" in verdict.detail
        assert "superseded by" in verdict.detail

    async def test_a_declared_successor_resolving_to_no_repo_holds(self) -> None:
        linear = _linear(
            "No code has been changed for this issue.\n\n"
            f"{_REPO}#{_CLOSED_A} and {_REPO}#{_CLOSED_B} superseded by #40404."
        )
        result = await _built(linear, _product()).handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.SKIPPED_REFERENCED_PR_UNMERGED
        ]
        assert "resolves to no repository" in result.outcomes[0].reason

    async def test_an_open_citation_is_never_superseded_by_a_declaration(
        self,
    ) -> None:
        """Supersession is reachable only from CLOSED. An open PR can merge."""
        linear = _linear("No code has been changed for this issue.\n\n" + _DECLARATION)
        product = _product()
        product[_key(_CLOSED_A)] = _pr_payload(_CLOSED_A, "open", None)
        result = await _built(linear, product).handle(_request())

        assert [o.decision for o in result.outcomes] == [
            EnumEvidenceAutocloseDecision.SKIPPED_REFERENCED_PR_UNMERGED
        ]
        reason = result.outcomes[0].reason
        assert f"{_REPO}#{_CLOSED_A}: state=OPEN" in reason
        # With an open citation still holding, the old sentence is the right
        # one: merging it really is all that candidate needs.
        assert "the next tick re-offers it" in reason
