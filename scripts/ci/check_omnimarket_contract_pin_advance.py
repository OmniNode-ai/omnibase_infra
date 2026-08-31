#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Forward-only guard for the omnimarket contract pin (OMN-17292 / OMN-15701).

OMN-15701 was not "a pin existed". It was a **stale** pin used as the input to a
``--write`` regeneration: ``generate_application_database_table_grants.py`` owns
the TABLE-grant subset outright (it drops every existing TABLE grant and re-emits
from the contract set it is handed), so deriving from an older contract set
silently deletes grants. That is how omnibase_infra#2632 reverted #2634's
``tenant_projection_writer`` grants for 8 house-tenant relations, with a green
build. The pin in ci.yml at the time was ``4637e625`` -- which GitHub reports as
**61 commits behind and diverged** from the ``54356a83`` that replaced it.

Pinning the derivation input (OMN-17292) is therefore only safe if the pin can
never move backwards. This guard is that assertion.

It is deliberately a **pure function of bytes**: it reads a GitHub compare
payload for ``<base_pin>...<head_pin>`` rather than calling the API or shelling
out to git. That is what makes it replayable against the captured response from
the real incident (``tests/fixtures/omn17292/``, registry case
``omn15701-stale-pin-backwards-move``) instead of against a synthetic git graph
that cannot exhibit the failure -- the exact defect OMN-15547 exists to stop.

Verdicts, from GitHub's own ``status``:

* ``identical`` -- accept; the pin did not move.
* ``ahead`` -- accept; head is a strict descendant of base, a forward advance.
* ``behind`` -- reject; head is an ANCESTOR of base. This is the OMN-15701
  direction, the one that deletes grants.
* ``diverged`` -- reject; neither commit is reachable from the other, so the
  advance cannot be proven forward. This is the shape the OMN-15701 pin had.
* anything else -- reject; fail closed rather than guess.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# GitHub's compare `status` values that represent a provable forward move.
FORWARD_STATUSES = frozenset({"ahead", "identical"})
BACKWARD_STATUS = "behind"
DIVERGED_STATUS = "diverged"


def _load_compare(compare_json: Path) -> dict[str, Any]:
    payload = json.loads(compare_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"{compare_json} is not a GitHub compare object (got {type(payload).__name__})"
        )
    return payload


def check_advance(compare_json: Path, base_pin: str, head_pin: str) -> tuple[bool, str]:
    """Return ``(ok, message)`` for a proposed pin move.

    ``compare_json`` must be the response for ``compare/<base_pin>...<head_pin>``.
    The pair is re-derived from the payload's own ``url`` so a capture for some
    other pair cannot be passed off as proof for this one.
    """
    payload = _load_compare(compare_json)

    url = payload.get("url")
    if not isinstance(url, str) or "/compare/" not in url:
        return (
            False,
            f"{compare_json} has no usable `url`; cannot bind it to a pin pair",
        )
    described = url.rsplit("/compare/", 1)[1]
    expected = f"{base_pin}...{head_pin}"
    if described != expected:
        return False, (
            f"{compare_json} describes {described!r}, not the claimed advance "
            f"{expected!r}. Refusing to accept a comparison of a different pair "
            "as proof of this one."
        )

    status = payload.get("status")
    behind_by = payload.get("behind_by")
    ahead_by = payload.get("ahead_by")

    if status in FORWARD_STATUSES:
        return True, (
            f"pin advances forward ({status}, +{ahead_by}): {base_pin} -> {head_pin}"
        )
    if status == BACKWARD_STATUS:
        return False, (
            f"pin moves BACKWARDS ({status}, {behind_by} commit(s) behind): "
            f"{base_pin} -> {head_pin}. The proposed pin is an ANCESTOR of the "
            "current one, so regenerating from it would delete TABLE grants "
            "derived from contracts that already merged upstream -- this is "
            "exactly the OMN-15701 silent grant reversion. Refused."
        )
    if status == DIVERGED_STATUS:
        return False, (
            f"pin moves to DIVERGED history ({behind_by} commit(s) behind, "
            f"+{ahead_by} ahead): {base_pin} -> {head_pin}. Neither commit is "
            "reachable from the other, so the advance cannot be proven forward. "
            "This is the shape of the OMN-15701 pin (61 behind, diverged). "
            "Refused (fail closed)."
        )
    return False, (
        f"unrecognised compare status {status!r} for {base_pin} -> {head_pin}; "
        "refusing rather than guessing (fail closed)."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare-json",
        type=Path,
        required=True,
        help="GitHub compare payload for <base-pin>...<head-pin>.",
    )
    parser.add_argument("--base-pin", required=True, help="Currently committed pin.")
    parser.add_argument("--head-pin", required=True, help="Proposed replacement pin.")
    args = parser.parse_args(argv)

    try:
        ok, message = check_advance(args.compare_json, args.base_pin, args.head_pin)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1

    if ok:
        print(f"omnimarket contract pin advance OK: {message}")
        return 0
    print(f"::error::{message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
