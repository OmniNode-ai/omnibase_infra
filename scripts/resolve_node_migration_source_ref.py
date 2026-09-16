#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Resolve the omnimarket ref for node-migration vendor sync CI."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Final

_REPO_ROOT: Final = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.ci.check_pin_reachability import (
    _GITHUB_API,
    _ORG,
    Resolution,
    Verdict,
    _api_get,
    _Resolver,
)
from scripts.ci.pr_trailers import parse_trailer

DEFAULT_REF = "dev"
FIELD_NAMES = ("Omnimarket-Source-Ref", "Node-Migration-Source-Ref")
SOURCE_PR_FIELD_NAMES = ("Node-Migration-Source-PR",)
SOURCE_SHA_FIELD_NAMES = ("Node-Migration-Source-SHA", "Omnimarket-Source-SHA")
REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")
SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
SOURCE_PR_RE = re.compile(
    r"^(?:(?:OmniNode-ai/)?omnimarket)?#?(?P<number>[1-9][0-9]*)$"
)


def _body_from_event(path: str | None) -> str:
    if not path:
        return ""
    event_path = Path(path)
    if not event_path.is_file():
        return ""
    payload = json.loads(event_path.read_text(encoding="utf-8"))
    pull_request = payload.get("pull_request")
    if not isinstance(pull_request, dict):
        return ""
    body = pull_request.get("body")
    return body if isinstance(body, str) else ""


def _parse_ref(body: str) -> str:
    """The declared omnimarket ref, or :data:`DEFAULT_REF` when none is.

    Trailer recognition is delegated to :func:`scripts.ci.pr_trailers.parse_trailer`,
    which honours only column-0 declarations outside fenced code blocks and
    inline code spans (OMN-17294). Before that, this matched any line whose
    stripped text began with the field name, so a trailer merely QUOTED in the
    body -- a runbook excerpt, a pasted log, an example -- selected the
    omnimarket tree the OMN-15361 grant-derivation job runs against, and beat
    the author's real trailer to it by appearing first.

    Raises:
        ValueError: the declared ref is unsafe, or two different refs are
            declared (``TrailerConflictError`` is a ``ValueError``).
    """
    candidate = parse_trailer(body, FIELD_NAMES)
    if candidate is None:
        return DEFAULT_REF
    return _validate(candidate)


def _parse_source_pr(body: str) -> int | None:
    candidate = parse_trailer(body, SOURCE_PR_FIELD_NAMES)
    if candidate is None:
        return None
    match = SOURCE_PR_RE.fullmatch(candidate)
    if match is None:
        raise ValueError(f"invalid node-migration source PR: {candidate!r}")
    return int(match["number"])


def _parse_source_sha(body: str) -> str | None:
    candidate = parse_trailer(body, SOURCE_SHA_FIELD_NAMES)
    if candidate is None:
        return None
    candidate = candidate.strip().lower()
    if not SHA40_RE.fullmatch(candidate):
        raise ValueError(f"invalid node-migration source SHA: {candidate!r}")
    return candidate


def _validate(candidate: str) -> str:
    if (
        not REF_RE.fullmatch(candidate)
        or ".." in candidate
        or candidate.endswith(("/", ".lock"))
        or "//" in candidate
    ):
        raise ValueError(f"invalid omnimarket source ref: {candidate!r}")
    return candidate


def _resolve_ref_from_dev(ref: str) -> Resolution:
    """Resolve ``ref`` against the protected omnimarket ``dev`` history.

    A source-ref trailer is an input to the grants derivation gate, not a
    permission to checkout an arbitrary feature branch. Reuse the existing
    OMN-15538 GitHub-compare oracle so ``behind`` and ``identical`` are the
    only passing outcomes; unmerged, missing, and unavailable refs stay
    fail-closed.
    """
    return _Resolver(("dev",)).resolve("omnimarket", ref)


def _require_ref_reachable_from_dev(ref: str) -> str:
    """Return a durably reachable ref or raise a fail-closed error."""
    resolution = _resolve_ref_from_dev(ref)
    if resolution.verdict is Verdict.REACHABLE:
        return ref
    if resolution.verdict is Verdict.UNREACHABLE:
        raise ValueError(
            "declared omnimarket source ref is not reachable from "
            f"omnimarket/dev: {ref!r}; {resolution.detail}"
        )
    raise ValueError(
        "could not prove declared omnimarket source ref reachable from "
        f"omnimarket/dev (fail-closed): {ref!r}; {resolution.detail}"
    )


def _paired_pr_url(number: int) -> str:
    return f"{_GITHUB_API}/repos/{_ORG}/omnimarket/pulls/{number}"


def _require_paired_source_ref(ref: str, pr_number: int, sha: str) -> str:
    """Return the exact paired PR head SHA when the declared source is open.

    Node-migration vendor PRs are the one PR-CI surface that sometimes must
    compare against a still-unmerged omnimarket migration before the product PR
    can land. This is not a durable pin: it is only the CI source tree for the
    infra PR. Keep it constrained to an open omnimarket PR whose current head
    branch and current head SHA match explicit trailers in the infra PR body.
    """
    status, body, detail = _api_get(_paired_pr_url(pr_number))
    if status != 200 or body is None:
        raise ValueError(
            "could not prove node-migration source PR is open "
            f"(fail-closed): omnimarket#{pr_number}; {detail}"
        )
    if body.get("state") != "open":
        raise ValueError(
            f"node-migration source PR is not open: omnimarket#{pr_number}"
        )
    if body.get("draft") is True:
        raise ValueError(f"node-migration source PR is draft: omnimarket#{pr_number}")
    base = body.get("base")
    head = body.get("head")
    if not isinstance(base, dict) or base.get("ref") != "dev":
        raise ValueError(
            f"node-migration source PR must target omnimarket/dev: omnimarket#{pr_number}"
        )
    if not isinstance(head, dict):
        raise ValueError(
            f"node-migration source PR has no head: omnimarket#{pr_number}"
        )
    head_repo = head.get("repo")
    if not isinstance(head_repo, dict) or head_repo.get("full_name") != (
        f"{_ORG}/omnimarket"
    ):
        raise ValueError(
            "node-migration source PR head repo must be "
            f"{_ORG}/omnimarket: omnimarket#{pr_number}"
        )
    if head.get("ref") != ref:
        raise ValueError(
            "node-migration source PR head ref does not match declared source: "
            f"omnimarket#{pr_number} head={head.get('ref')!r} declared={ref!r}"
        )
    if head.get("sha") != sha:
        raise ValueError(
            "node-migration source PR head SHA does not match declared source SHA: "
            f"omnimarket#{pr_number} head={head.get('sha')!r} declared={sha!r}"
        )
    return sha


def _resolve_declared_ref(body: str) -> str:
    ref = _parse_ref(body)
    source_pr = _parse_source_pr(body)
    source_sha = _parse_source_sha(body)
    if source_pr is not None or source_sha is not None:
        if source_pr is None or source_sha is None:
            raise ValueError(
                "node-migration source refs with pair metadata require both "
                "Node-Migration-Source-PR and Node-Migration-Source-SHA trailers"
            )
        return _require_paired_source_ref(ref, source_pr, source_sha)
    return _require_ref_reachable_from_dev(ref)


def main() -> int:
    try:
        ref = _resolve_declared_ref(
            _body_from_event(os.environ.get("GITHUB_EVENT_PATH"))
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as output:
            output.write(f"ref={ref}\n")
    print(ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
