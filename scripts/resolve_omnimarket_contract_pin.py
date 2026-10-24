#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Resolve the pinned omnimarket contract release tag for the grants derivation gate.

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
follows omnimarket's newest release tag once a day, after the overnight release
train (``release-train-nightly.yml``), regenerates the grants in the same commit
and opens a PR. The pin is a release tag and never a commit sha: a release is
created only by the train (or by a one-off grant), so the pin moves once per
release and no pull request is opened per omnimarket merge.
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
# An omnimarket RELEASE tag, written by auto-tag-on-merge when a release PR
# merges. The per-merge ``dev-<UTC>-<sha12>`` tags do not match: they move with
# every merge, which is the coupling this pin removes.
_RELEASE_TAG_RE = re.compile(r"^v[0-9]+\.[0-9]+\.[0-9]+$")
# The pin a PR moves AWAY from was committed under the base's own rules, and
# before OMN-17292 those rules wrote a commit sha. The forward-only proof only
# needs the commit that predecessor names, so it may read a sha there; the pin
# this run derives from is still a release tag and nothing else.
_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# STDLIB ONLY -- deliberately, and load-bearing. In .github/workflows/ci.yml this
# script runs as bare `python3` inside `Application Database Domain Enforcement
# (OMN-15361)` roughly seventy lines BEFORE that job's `Setup Python and uv`
# step, because the omnimarket ref it emits is what the very next step checks
# out. The runners' ambient python3 cannot be assumed to have PyYAML.
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


def resolve_pin(pin_path: Path = PIN_PATH, *, predecessor: bool = False) -> str:
    """Return the pinned omnimarket release tag, or raise ``ValueError``.

    Fails closed on anything that is not a ``vX.Y.Z`` release tag. A branch, a
    per-merge dev tag or a commit sha here would either reintroduce the
    moving-input coupling this pin exists to remove or be a pin written by hand
    or by a bot, so it is rejected rather than resolved.

    With ``predecessor=True`` (the base revision's pin, read only to prove the
    advance is forward-only) a 40-hex commit sha is also returned: that value
    was committed under the base's rules and names an immutable commit.
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
    if ref is not None and predecessor and _COMMIT_SHA_RE.fullmatch(ref):
        return ref
    if ref is None or not _RELEASE_TAG_RE.fullmatch(ref):
        raise ValueError(
            f"omnimarket_contract_ref must be an omnimarket release tag (vX.Y.Z), "
            f"got {ref!r}. A branch would make the grants derivation depend on a "
            "moving foreign branch tip again, and a commit sha is a pin written "
            "by hand or by a bot rather than the release train's tag (OMN-17292)."
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
            "both sides of a forward-only proof use this same validated parser. "
            "Any file other than the committed pin is read as that predecessor."
        ),
    )
    args = parser.parse_args(argv)

    try:
        ref = resolve_pin(args.pin_file, predecessor=args.pin_file != PIN_PATH)
    except (OSError, ValueError) as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1

    # Only the committed pin is a step OUTPUT. Resolving some other file (the
    # base revision's copy, during the forward-only proof) must not also publish
    # a `ref=` output, or the step that reads a base pin would advertise it as
    # though it were the ref this run derives from.
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output and args.pin_file == PIN_PATH:
        with open(github_output, "a", encoding="utf-8") as output:
            output.write(f"ref={ref}\n")
    print(ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
