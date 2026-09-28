# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Resolve who is issuing an ``onex delegate`` run (OMN-19860).

The caller's LANE is the token the rolling work ledger writes in its ``lane=``
cell, so a delegation row and a ledger row name the same lane in the same
vocabulary. It is resolved most explicit first:

1. the ``--caller-lane`` flag -- a malformed value is a usage error, never
   dropped, because a lane the caller believes they named and silently lost
   is a run that attributes to nobody;
2. the lane environment variables, in the order the gh shim
   (``omniclaude/scripts/user-bin/gh``) and the PR-ownership guard already
   read them, so one lane resolves to one name across all three -- a
   malformed value is skipped and named in the source, never guessed;
3. the lane registered for the ``omni_worktrees/<ticket>/<dir>`` worktree the
   command runs from, read from the registry omniclaude's ``lane_identity.py``
   writes (``$ONEX_LANE_REGISTRY_ROOT`` or ``$OMNI_HOME/.onex_state``, then
   ``lane_identity/<sha256(worktree)[:32]>.json``);
4. otherwise none.

The caller's SESSION is the Claude Code session id, else the session the
worktree registration recorded. It is sent only as a UUID: the delegate-skill
terminal projection types ``session_id`` as one, and free text would be
dropped there anyway.

Attribution, not authority: a lane states its own name and nothing here
proves it, the same limit ``lane_identity.py`` records about commit trailers.
What this removes is the silent case, a delegation row that names nobody.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Mapping
from pathlib import Path

from omnibase_infra.cli.model_delegate_caller import ModelDelegateCaller

__all__ = [
    "CALLER_LANE_ENV_VARS",
    "DELEGATE_CALLER_LANE_METADATA_KEY",
    "resolve_delegate_caller",
]

#: The request ``metadata`` key naming the caller's lane. omnimarket's
#: delegate-skill handler and projection read the same key (OMN-19860).
DELEGATE_CALLER_LANE_METADATA_KEY = "caller_lane"

#: A ledger lane token; the same shape omnimarket's projection accepts.
_CALLER_LANE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

#: Lane environment variables, most explicit first: the gh shim's order.
CALLER_LANE_ENV_VARS: tuple[str, ...] = (
    "ONEX_LANE",
    "ONEX_LANE_ID",
    "ONEX_AGENT_NAME",
    "CLAUDE_AGENT_NAME",
    "CLAUDE_SUBAGENT_NAME",
)

_SESSION_ENV_VAR = "CLAUDE_CODE_SESSION_ID"
_REGISTRY_ROOT_ENV_VAR = "ONEX_LANE_REGISTRY_ROOT"
_WORKSPACE_ENV_VAR = "OMNI_HOME"

#: The worktree root of a per-ticket worktree path (Operating Rule 9).
_WORKTREE_ROOT_PATTERN = re.compile(r"^(.*/omni_worktrees/[^/]+/[^/]+)(?:/|$)")


def _is_lane(value: str) -> bool:
    return _CALLER_LANE_PATTERN.fullmatch(value) is not None


def _as_session(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return str(uuid.UUID(value.strip()))
    except ValueError:
        return None


def _registry_record(cwd: Path, environ: Mapping[str, str]) -> dict[str, object]:
    """The lane registration for the worktree ``cwd`` is in, or an empty map."""
    root_text = environ.get(_REGISTRY_ROOT_ENV_VAR) or (
        str(Path(environ[_WORKSPACE_ENV_VAR]) / ".onex_state")
        if environ.get(_WORKSPACE_ENV_VAR)
        else ""
    )
    if not root_text:
        return {}
    try:
        resolved = cwd.resolve()
    except OSError:
        return {}
    found = _WORKTREE_ROOT_PATTERN.match(resolved.as_posix())
    if found is None:
        return {}
    worktree = found.group(1)
    key = hashlib.sha256(worktree.encode("utf-8")).hexdigest()[:32]
    record_path = Path(root_text) / "lane_identity" / f"{key}.json"
    try:
        loaded = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _with_skipped(source: str, skipped: list[str]) -> str:
    return f"{source}; skipped {', '.join(skipped)}" if skipped else source


def resolve_delegate_caller(
    caller_lane: str | None,
    *,
    cwd: Path,
    environ: Mapping[str, str],
) -> ModelDelegateCaller:
    """Resolve the caller's lane and session, and say how each was chosen."""
    record = _registry_record(cwd, environ)

    lane: str | None = None
    lane_source = "none"
    skipped_lanes: list[str] = []
    if caller_lane is not None:
        named = caller_lane.strip()
        if not _is_lane(named):
            raise ValueError(
                f"--caller-lane {caller_lane!r} is not a lane token: expected a "
                "letter or digit, then letters, digits, '.', '_', ':' or '-', at "
                "most 128 characters, as the ledger writes lane=<token>."
            )
        lane, lane_source = named, "explicit"
    else:
        for name in CALLER_LANE_ENV_VARS:
            value = environ.get(name, "").strip()
            if not value:
                continue
            if _is_lane(value):
                lane, lane_source = value, f"env {name}"
                break
            skipped_lanes.append(name)
        if lane is None:
            registered = record.get("lane")
            if isinstance(registered, str) and _is_lane(registered):
                lane, lane_source = registered, "registered worktree"
    lane_source = _with_skipped(lane_source, skipped_lanes)

    session: str | None = None
    session_source = "none"
    skipped_sessions: list[str] = []
    harness_session = environ.get(_SESSION_ENV_VAR, "")
    if harness_session.strip():
        session = _as_session(harness_session)
        if session is not None:
            session_source = f"env {_SESSION_ENV_VAR}"
        else:
            skipped_sessions.append(_SESSION_ENV_VAR)
    if session is None:
        registered_session = _as_session(record.get("session_id"))
        if registered_session is not None:
            session, session_source = registered_session, "registered worktree"
    session_source = _with_skipped(session_source, skipped_sessions)

    return ModelDelegateCaller(
        lane=lane,
        lane_source=lane_source,
        session_id=session,
        session_source=session_source,
    )
