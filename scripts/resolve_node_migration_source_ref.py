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

from scripts.ci.pr_trailers import parse_trailer

DEFAULT_REF = "dev"
FIELD_NAMES = ("Omnimarket-Source-Ref", "Node-Migration-Source-Ref")
REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")


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


def _validate(candidate: str) -> str:
    if (
        not REF_RE.fullmatch(candidate)
        or ".." in candidate
        or candidate.endswith(("/", ".lock"))
        or "//" in candidate
    ):
        raise ValueError(f"invalid omnimarket source ref: {candidate!r}")
    return candidate


def main() -> int:
    try:
        ref = _parse_ref(_body_from_event(os.environ.get("GITHUB_EVENT_PATH")))
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
