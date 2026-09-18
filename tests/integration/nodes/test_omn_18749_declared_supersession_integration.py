# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration contract for the OMN-18749 declared-supersession predicate.

The unit suite proves the decisions. What this file pins is the seam between
the predicate and GitHub: the exact `gh` invocation it issues, the real payload
shape that invocation returns, and the fact that it issues NO call at all when
the ticket declares nothing.

The payloads below are the live REST bodies for the pull requests in the
measured case, read from the GitHub API on 2026-09-18 and reduced to the fields
the predicate touches: `omnibase_infra#3615` is CLOSED with `merged_at: null`,
`#3627` is CLOSED with `merged: true` and a merge time. A closed-and-merged
pull request reports `state: "closed"` exactly as an abandoned one does, so
reading `state` alone would mark the successor unmerged and hold forever —
which is why the predicate reads `merged_at` and why that is asserted here
against the real shape rather than against a hand-written one.

Subprocess is not spawned, for the reason the auto-merge effect's integration
test gives: hitting the real API from CI needs credentials. The argument shape
is the contract, and it is asserted exactly.
"""

from __future__ import annotations

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.declared_supersession import (
    resolve_declared_supersession,
)

pytestmark = pytest.mark.integration

_REPO = "OmniNode-ai/omnibase_infra"
_CLOSED = 3615
_SUCCESSOR = 3627

_DECLARATION = (
    f"{_REPO}#{_CLOSED} and {_REPO}#3626 are closed unmerged; "
    f"superseded by {_REPO}#{_SUCCESSOR}."
)

#: Live REST body for the merged successor, read 2026-09-18.
_MERGED_PAYLOAD: dict[str, object] = {
    "number": 3627,
    "state": "closed",
    "merged": True,
    "merged_at": "2026-09-16T16:56:24Z",
    "html_url": f"https://github.com/{_REPO}/pull/3627",
    "title": "test(OMN-18172): AC4 attribution query proof",
    "draft": False,
    "base": {"ref": "dev"},
}

#: Live REST body for the abandoned pull request, read the same day. Note that
#: `state` is identical to the merged one above.
_ABANDONED_PAYLOAD: dict[str, object] = {
    "number": 3615,
    "state": "closed",
    "merged": False,
    "merged_at": None,
    "html_url": f"https://github.com/{_REPO}/pull/3615",
    "title": "test(OMN-18172): AC4 attribution query proof",
    "draft": False,
    "base": {"ref": "dev"},
}


class RecordingGh:
    """A gh double that records argv and serves live payloads by path."""

    def __init__(self, payloads: dict[str, dict[str, object]]) -> None:
        self._payloads = payloads
        self.calls: list[list[str]] = []

    async def __call__(
        self, args: list[str], timeout: float
    ) -> tuple[object | None, str]:
        self.calls.append(list(args))
        path = args[2]
        if path in self._payloads:
            return self._payloads[path], ""
        return None, f"gh api: 404 Not Found ({path})"


@pytest.mark.asyncio
class TestGithubSeam:
    async def test_the_predicate_issues_exactly_one_call_for_the_successor(
        self,
    ) -> None:
        gh = RecordingGh({f"repos/{_REPO}/pulls/{_SUCCESSOR}": _MERGED_PAYLOAD})

        verdict = await resolve_declared_supersession(
            repo=_REPO,
            closed_pr_number=_CLOSED,
            description=_DECLARATION,
            run_gh_command=gh,
            gh_timeout_seconds=30,
        )

        assert verdict.superseded is True
        assert verdict.replacement_pr == _SUCCESSOR
        assert "2026-09-16T16:56:24Z" in verdict.detail
        # The seam, exactly: one read, of the successor, by the REST path.
        assert gh.calls == [["gh", "api", f"repos/{_REPO}/pulls/{_SUCCESSOR}"]]

    async def test_a_closed_state_alone_never_stands_in_for_a_merge_time(
        self,
    ) -> None:
        """The real payloads differ only in `merged_at`, and that is the field."""
        assert _MERGED_PAYLOAD["state"] == _ABANDONED_PAYLOAD["state"] == "closed"
        gh = RecordingGh({f"repos/{_REPO}/pulls/{_SUCCESSOR}": _ABANDONED_PAYLOAD})

        verdict = await resolve_declared_supersession(
            repo=_REPO,
            closed_pr_number=_CLOSED,
            description=_DECLARATION,
            run_gh_command=gh,
            gh_timeout_seconds=30,
        )

        assert verdict.superseded is False
        assert "merged_at=null" in verdict.detail

    async def test_no_call_is_issued_when_the_ticket_declares_nothing(self) -> None:
        """An undeclared citation costs no API budget and holds regardless."""
        gh = RecordingGh({f"repos/{_REPO}/pulls/{_SUCCESSOR}": _MERGED_PAYLOAD})

        verdict = await resolve_declared_supersession(
            repo=_REPO,
            closed_pr_number=_CLOSED,
            description="No code has been changed for this issue.",
            run_gh_command=gh,
            gh_timeout_seconds=30,
        )

        assert verdict.superseded is False
        assert gh.calls == []
        assert "declares no successor" in verdict.detail

    async def test_a_404_on_the_successor_holds_as_unresolved(self) -> None:
        gh = RecordingGh({})

        verdict = await resolve_declared_supersession(
            repo=_REPO,
            closed_pr_number=_CLOSED,
            description=_DECLARATION,
            run_gh_command=gh,
            gh_timeout_seconds=30,
        )

        assert verdict.superseded is False
        assert "could not be read" in verdict.detail
        assert "404" in verdict.detail
