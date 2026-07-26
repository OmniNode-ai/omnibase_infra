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
    for raw_line in body.splitlines():
        line = raw_line.strip()
        for field_name in FIELD_NAMES:
            prefix = f"{field_name}:"
            if line.lower().startswith(prefix.lower()):
                candidate = line[len(prefix) :].strip()
                return _validate(candidate)
    return DEFAULT_REF


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
