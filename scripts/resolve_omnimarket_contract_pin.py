#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Resolve the pinned omnimarket contract SHA for the grants derivation gate.

OMN-17292. This replaces the OMN-15703 "live-resolve omnimarket dev HEAD" step
in ``Application Database Domain Enforcement (OMN-15361)``.

The point of this script is what it *cannot* do: it performs no network I/O and
reads no event payload, so the omnimarket ref the gate derives from is a pure
function of committed repository state. An omnimarket merge therefore cannot
change the verdict of an already-green omnibase_infra PR -- which, because infra
``dev`` enforces via the single required ``CI Summary`` umbrella (OMN-4497), is
what let one upstream merge red every open PR in the repo at once.

Freshness has not been dropped, only moved off the per-PR path: the pin is
advanced by ``.github/workflows/omnimarket-contract-pin-refresh.yml``, which
regenerates the grants in the same commit and opens a PR.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

PIN_PATH = (
    Path(__file__).resolve().parents[1] / ".github" / "omnimarket-contract-pin.yaml"
)
EXPECTED_REPOSITORY = "OmniNode-ai/omnimarket"
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# STDLIB ONLY -- deliberately, and load-bearing. In .github/workflows/ci.yml this
# script runs as bare `python3` inside `Application Database Domain Enforcement
# (OMN-15361)` roughly seventy lines BEFORE that job's `Setup Python and uv`
# step, because the omnimarket ref it emits is what the very next step checks
# out. The runners' ambient python3 has no PyYAML -- proven in this same
# workflow, whose contract-compliance job carries an explicit
# `python3 -m pip install --quiet --user pyyaml` step for exactly that reason.
# A `import yaml` here would therefore ImportError on every PR and take the
# required check down repo-wide: precisely the org-wide red OMN-17292 exists to
# remove, self-inflicted. The pin file's two fields are a fixed `key: value`
# shape this module owns end to end, so a strict line parse is sufficient and
# is checked by tests/ci/test_omnimarket_contract_pin.py.
_FIELD_RE = re.compile(r"^(?P<key>[a-z_]+):[ \t]*(?P<value>\S+)[ \t]*$")


def _parse_pin_fields(text: str) -> dict[str, str]:
    """Parse the pin file's ``key: value`` lines, ignoring comments and blanks."""
    fields: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _FIELD_RE.fullmatch(line)
        if match is None:
            raise ValueError(
                f"unparseable line in the omnimarket contract pin: {raw_line!r}. "
                "The pin file is a flat `key: value` document by design (this "
                "resolver is stdlib-only, see the module comment)."
            )
        fields[match.group("key")] = match.group("value")
    return fields


def resolve_pin(pin_path: Path = PIN_PATH) -> str:
    """Return the pinned omnimarket commit SHA, or raise ``ValueError``.

    Fails closed on anything that is not a full 40-hex commit SHA. A branch or
    tag here would silently reintroduce the moving-input coupling this pin
    exists to remove, so it is rejected rather than resolved.
    """
    if not pin_path.is_file():
        raise ValueError(f"missing omnimarket contract pin file: {pin_path}")

    fields = _parse_pin_fields(pin_path.read_text(encoding="utf-8"))

    repository = fields.get("repository")
    if repository != EXPECTED_REPOSITORY:
        raise ValueError(
            f"{pin_path} declares repository {repository!r}; expected {EXPECTED_REPOSITORY!r}"
        )

    ref = fields.get("omnimarket_contract_ref")
    if ref is None or not _SHA_RE.fullmatch(ref):
        raise ValueError(
            f"omnimarket_contract_ref must be a full 40-hex commit sha, got {ref!r}. "
            "A branch or tag would make the grants derivation depend on a moving "
            "foreign branch tip again (OMN-17292)."
        )
    return ref


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve the omnimarket contract pin.")
    parser.add_argument(
        "--pin-file",
        type=Path,
        default=PIN_PATH,
        help=(
            "Pin file to read (default: the committed one). CI passes the base "
            "revision's copy here to read the pin a PR is moving away from, so "
            "both sides of a forward-only proof use this same validated parser."
        ),
    )
    args = parser.parse_args(argv)

    try:
        ref = resolve_pin(args.pin_file)
    except (OSError, ValueError) as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1

    # Only the committed pin is a step OUTPUT. Resolving some other file (the
    # base revision's copy, during the forward-only proof) must not also publish
    # a `sha=` output, or the step that reads a base pin would advertise it as
    # though it were the ref this run derives from.
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output and args.pin_file == PIN_PATH:
        with open(github_output, "a", encoding="utf-8") as output:
            output.write(f"sha={ref}\n")
    print(ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
