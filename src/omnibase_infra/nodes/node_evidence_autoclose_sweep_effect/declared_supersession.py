# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18749 — the ticket-declared supersession predicate.

A cited pull request that is CLOSED and never merged can never become merged.
The cited-PR conjunct held such a ticket forever and told its reader to merge
the citation, which is impossible: OMN-18172 carried two closed proof pull
requests whose work landed in a third, and was refused on both on every tick.

The OMN-18233 predicate already resolves this shape for dependency-cascade
bumps, by reading a declared source package and comparing the delivered pin as
a version. It is right to be narrow and it does not reach a closed citation
that is not a bump — clause one does not resolve, and the ticket keeps
blocking. This predicate covers the rest, with the only statement available for
them: the one the TICKET makes.

    <owner/repo#N> superseded by <owner/repo#M>

Both sides must be spelled in a resolvable form, the phrase set is directional,
and the successor must be verified MERGED by its own probe. Every other outcome
holds. Two things are deliberately NOT accepted as the declaration, because
both are written by the party that wants the close and neither is read by the
guard this conjunct replicates: a Linear comment, and the successor pull
request's own body.

Directionality is load-bearing. A symmetric verb such as "replaces" would bind
the pair backwards and record the MERGED pull request as superseded by the
closed one, which is the reading that would wave real abandoned work through.

This module sits at the node ROOT for the reason ``cascade_supersession``
states: it is not a handler, it is routed by no contract, and everything under
``handlers/`` must appear in the contract's ``handler_routing``.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_declared_supersession import (
    ModelDeclaredSupersession,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_supersession_verdict import (
    ModelSupersessionVerdict,
)

#: `(args, timeout) -> (payload | None, error)`, the node's injected gh runner.
GhRunner = Callable[[list[str], float], Awaitable[tuple[object | None, str]]]

#: Directional only: every phrase reads `<closed> <phrase> <successor>`.
_SUPERSESSION_PHRASE_RE = re.compile(
    r"\b(?:super[sc]eded\s+by|replaced\s+by|re-?landed\s+as|landed\s+as|"
    r"closed\s+in\s+favou?r\s+of)\b",
    re.IGNORECASE,
)

#: Only QUALIFIED citations bind on a declaration line. A bare `#N` there is
#: refused rather than guessed: the line is the one place an author is telling
#: the gate which pull requests it is talking about, and the refusal that asks
#: for the line prints the qualified form.
_QUALIFIED_URL_RE = re.compile(
    r"https?://github\.com/([\w.-]+)/([\w.-]+)/pull/(\d+)", re.IGNORECASE
)
_QUALIFIED_HASH_RE = re.compile(r"\b(OmniNode-ai/[\w.-]+)#(\d+)\b", re.IGNORECASE)


def _qualified_refs(fragment: str) -> list[tuple[str, int]]:
    """Every `owner/repo#N` and pull URL in `fragment`, in order, deduplicated."""
    found: list[tuple[str, int]] = []
    for match in _QUALIFIED_URL_RE.finditer(fragment):
        ref = (f"{match.group(1)}/{match.group(2)}", int(match.group(3)))
        if ref not in found:
            found.append(ref)
    for match in _QUALIFIED_HASH_RE.finditer(fragment):
        ref = (match.group(1), int(match.group(2)))
        if ref not in found:
            found.append(ref)
    return found


def parse_declared_supersessions(
    description: str,
) -> dict[tuple[str, int], ModelDeclaredSupersession]:
    """Read every declaration line, keyed by the CLOSED citation it binds.

    Several closed references may share one line — the two-proof-pull-requests,
    one-successor shape — and each binds to the same successor. A line whose
    right side names nothing at all is not a declaration and is ignored:
    "superseded by later work" gives the gate nothing to check. A line whose
    right side names something unresolvable IS a declaration, recorded with an
    empty successor so the hold can say the successor could not be resolved
    rather than that none was offered.

    Pure function.
    """
    declarations: dict[tuple[str, int], ModelDeclaredSupersession] = {}
    for line in description.splitlines():
        phrase = _SUPERSESSION_PHRASE_RE.search(line)
        if phrase is None:
            continue
        left = _qualified_refs(line[: phrase.start()])
        right = _qualified_refs(line[phrase.end() :])
        if not left:
            continue
        successor_repo, successor_number = right[0] if right else ("", 0)
        if not right and not _looks_like_a_named_successor(line[phrase.end() :]):
            continue
        for closed_repo, closed_number in left:
            declarations[(closed_repo, closed_number)] = ModelDeclaredSupersession(
                closed_repo=closed_repo,
                closed_number=closed_number,
                successor_repo=successor_repo,
                successor_number=successor_number,
                line=line.strip(),
            )
    return declarations


def _looks_like_a_named_successor(fragment: str) -> bool:
    """True when the right side gestures at a pull request but names none.

    "superseded by #4040" and "superseded by PR 4040" are attempts at a
    citation and must be REFUSED with a reason, not silently ignored as prose.
    "superseded by later work" names nothing and is prose.
    """
    return bool(re.search(r"#\s*\d+|\bpull\b|\bPR\b\s*\d", fragment))


async def resolve_declared_supersession(
    *,
    repo: str,
    closed_pr_number: int,
    description: str,
    run_gh_command: GhRunner,
    gh_timeout_seconds: int,
) -> ModelSupersessionVerdict:
    """Verify the ticket's declaration for one closed citation.

    ``superseded`` is True only when the ticket declared a successor for THIS
    citation, that successor resolved to a repository, and its own probe came
    back with a merge time. An unreadable probe holds, for the reason every
    fence in this node holds on one: "I could not check" is not "so I will
    ignore it".
    """
    declarations = parse_declared_supersessions(description)
    declaration = declarations.get((repo, closed_pr_number))
    if declaration is None:
        return ModelSupersessionVerdict(
            superseded=False,
            detail=(
                f"the ticket declares no successor for {repo}#{closed_pr_number}. "
                f"It is closed and can never merge, so merging it is not the "
                f"remedy: if the work landed elsewhere, add one line to the "
                f"ticket description reading `{repo}#{closed_pr_number} "
                f"superseded by OmniNode-ai/<repo>#<N>`, and that successor "
                f"must itself be merged. If nothing replaced it, this is "
                f"abandoned work and the citation belongs off the ticket"
            ),
        )
    if not declaration.successor_repo:
        return ModelSupersessionVerdict(
            superseded=False,
            detail=(
                f"the successor declared for {repo}#{closed_pr_number} resolves "
                f"to no repository — spell it `OmniNode-ai/<repo>#<N>` or as a "
                f"full GitHub URL on that line: {declaration.line!r}"
            ),
        )
    successor = f"{declaration.successor_repo}#{declaration.successor_number}"
    payload, error = await run_gh_command(
        [
            "gh",
            "api",
            f"repos/{declaration.successor_repo}/pulls/{declaration.successor_number}",
        ],
        gh_timeout_seconds,
    )
    if not isinstance(payload, dict):
        return ModelSupersessionVerdict(
            superseded=False,
            detail=(
                f"the declared successor {successor} could not be read, so the "
                f"supersession is unresolved rather than refuted "
                f"({error or 'no payload'})"
            ),
            replacement_pr=declaration.successor_number,
        )
    merged_at = str(payload.get("merged_at") or "")
    if not merged_at:
        state = str(payload.get("state") or "unknown").upper()
        return ModelSupersessionVerdict(
            superseded=False,
            detail=(
                f"the declared successor {successor} is state={state}, "
                f"merged_at=null — a chain of unmerged pull requests is still "
                f"unlanded work"
            ),
            replacement_pr=declaration.successor_number,
        )
    return ModelSupersessionVerdict(
        superseded=True,
        detail=(
            f"the ticket declares {repo}#{closed_pr_number} superseded by "
            f"{successor}, which merged at {merged_at}"
        ),
        replacement_pr=declaration.successor_number,
    )


__all__ = [
    "ModelDeclaredSupersession",
    "parse_declared_supersessions",
    "resolve_declared_supersession",
]
