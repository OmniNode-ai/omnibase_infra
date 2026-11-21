# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Open ledger CLAIMs from the reader node, for scripts that used to grep the ledger file (OMN-20547).

WHY. A ledger on the event bus with one writer and many readers of a file copy is
two ledgers (operator ruling 2026-11-05, "no more logging to files"). Two scripts of
this repository read the file to see who holds work: ``scripts/git-maintenance.sh``
(an open CLAIM keeps a worktree and a remote branch) and
``scripts/reconcile_verify_movement.py`` (the lane that owns a refused clone's dirty
paths). Both now ask ``omnimarket.nodes.node_work_ledger_reader_effect`` and never
open the file.

WHAT. One read is two reader calls over the same window, ``open-claims`` first and
``rows --kind CLAIM`` second. The reader node decides which claims are open (closed by a
TERMINAL, by a closes-CLAIM token or by a later lane TERMINAL), so the caller carries no
closure logic. An ``OPEN-CLAIM`` line names the lane, the ticket and the host but not the
claim's text, and a branch name or a path lives only in the text, so each open claim is
joined by (stamp, lane) to its CLAIM row. A claim that falls off the rows window's edge
in the seconds between the two calls keeps its ``OPEN-CLAIM`` line as its text, which
still names its lane and ticket, so it is never dropped.

REFUSES. A reader that exits 3 (not in the overlays' ``moved_readers``), 1 (stale), 69
(unreadable) or that cannot be run is a refusal: nothing is read from the file in its
place and the caller must treat it as an unread ledger. The command that runs the
reader (interpreter and module) is named by the caller (no default), so this file names no
other node's module.

Command line: ``python -m omnibase_infra.nodes.node_ledger_open_claims_effect.runtime_ledger_open_claims``
(``snapshot``, ``match`` and ``owner``; see that module).
"""

from __future__ import annotations

import re
import shlex
import subprocess
from collections.abc import Sequence

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.model_ledger_open_claim import ModelLedgerOpenClaim
from omnibase_infra.models.model_ledger_open_claims_request import (
    ModelLedgerOpenClaimsRequest,
)
from omnibase_infra.models.model_ledger_open_claims_result import (
    ModelLedgerOpenClaimsResult,
)
from omnibase_infra.models.operational_timeouts import TIMEOUTS

_CLAIM_ROW = re.compile(r"^(?P<stamp>\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) \| CLAIM \|")
_OPEN_CLAIM = re.compile(r"^OPEN-CLAIM lane=(?P<lane>\S+) claim=(?P<stamp>\S+) ")
_LANE = re.compile(r"(?<![\w-])lane=([^\s|]+)")
_TOKEN_EDGE = "[A-Za-z0-9_.-]"


class LedgerReadRefusedError(Exception):
    """The ledger was not read; the message names why."""


def _call_reader(
    request: ModelLedgerOpenClaimsRequest, query: str, extra: tuple[str, ...]
) -> list[str]:
    """One reader call; the rows it printed, or a refusal."""
    argv = [
        *shlex.split(request.reader_command),
        "--reader",
        request.reader,
        "--query",
        query,
        "--window-hours",
        str(request.window_hours),
        *extra,
    ]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=TIMEOUTS.ledger_reader_seconds,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LedgerReadRefusedError(f"ledger reader {query} not run: {exc}") from exc
    if proc.returncode != 0:
        summary = (proc.stderr or "").strip().splitlines()[-1:] or ["no output"]
        raise LedgerReadRefusedError(
            f"ledger reader exit {proc.returncode} ({query}): {summary[0]}"
        )
    return (proc.stdout or "").splitlines()


def read_open_claims(
    request: ModelLedgerOpenClaimsRequest,
) -> ModelLedgerOpenClaimsResult:
    """Every claim open in the window, each with its CLAIM row, oldest first."""
    opened = _call_reader(request, "open-claims", ())
    rows = _call_reader(request, "rows", ("--kind", "CLAIM"))
    texts: dict[tuple[str, str], str] = {}
    for line in rows:
        stamp = _CLAIM_ROW.match(line)
        lane = _LANE.search(line)
        if stamp and lane:
            texts[(stamp["stamp"], lane[1])] = line
    claims: list[ModelLedgerOpenClaim] = []
    for line in opened:
        hit = _OPEN_CLAIM.match(line)
        if hit is None:
            continue
        key = (hit["stamp"], hit["lane"])
        claims.append(
            ModelLedgerOpenClaim(stamp=key[0], lane=key[1], row=texts.get(key, line))
        )
    return ModelLedgerOpenClaimsResult(
        claims=tuple(sorted(claims, key=lambda c: c.stamp))
    )


class HandlerLedgerOpenClaims:
    """Fail closed: a read that is not whole is a refusal, never a partial answer."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def handle(
        self, request: ModelLedgerOpenClaimsRequest
    ) -> ModelLedgerOpenClaimsResult:
        """Read the open claims of the request's window, or raise LedgerReadRefusedError."""
        return read_open_claims(request)


def matching(
    claims: Sequence[ModelLedgerOpenClaim], keys: Sequence[str]
) -> list[ModelLedgerOpenClaim]:
    """The claims whose row holds a key as a whole token (a name never ends inside a longer one)."""
    patterns = [
        re.compile(f"(?<!{_TOKEN_EDGE}){re.escape(key)}(?!{_TOKEN_EDGE})")
        for key in keys
    ]
    return [c for c in claims if any(p.search(c.row) for p in patterns)]


def owner_of(
    claims: Sequence[ModelLedgerOpenClaim], repo: str, paths: Sequence[str]
) -> str | None:
    """The lane of the newest claim naming the repo AND one of the paths, so a path such
    as ``README.md`` alone never claims every clone's staged README."""
    for claim in sorted(claims, key=lambda c: c.stamp, reverse=True):
        if repo in claim.row and any(path in claim.row for path in paths):
            return claim.lane
    return None


__all__ = [
    "HandlerLedgerOpenClaims",
    "LedgerReadRefusedError",
    "matching",
    "owner_of",
    "read_open_claims",
]
