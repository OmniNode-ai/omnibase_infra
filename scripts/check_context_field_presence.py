#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Fresh-deploy fitness gate: context-ROI claim field presence (OMN-13412).

A node contract that advertises a context-ROI claim (i.e. asserts that injecting
a curated context pack improved a downstream outcome — utilization, savings,
quality) MUST also carry a ``context_pack_hash`` that pins the exact context pack
the claim is measured against. Without the hash, the ROI number is unfalsifiable:
a fresh deploy cannot replay which pack produced the gain, and the savings/ROI
projection renders a value with no provenance — the same no-provenance gap this
wave hardens for build identity.

What it scans
-------------
Every ``contract.yaml`` under ``src/``. A contract is considered to make a
context-ROI claim if it carries any of these keys (top-level or nested under
``metadata``/``context``):

  * ``context_roi``
  * ``context_roi_claim``
  * ``context_pack_roi``

When a claim is present, the contract must ALSO carry a non-empty
``context_pack_hash`` (top-level, under ``metadata``, under ``context``, or
inside the claim block). An absent or empty hash fails the gate.

This is a presence ratchet: contracts that make no ROI claim are unaffected. A
contract that adds an ROI claim without the pinning hash fails CI before merge.

Usage::

    uv run python scripts/check_context_field_presence.py
    uv run python scripts/check_context_field_presence.py --root src
    uv run python scripts/check_context_field_presence.py path/to/contract.yaml ...

Exit codes:
    0 — no contract makes an unpinned context-ROI claim
    1 — one or more contracts claim context-ROI without context_pack_hash
    2 — configuration error (root missing) or unparseable contract
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ROOT = _REPO_ROOT / "src"

# Keys that signal a context-ROI claim.
_CLAIM_KEYS = ("context_roi", "context_roi_claim", "context_pack_roi")
# The pinning field every claim must carry.
_HASH_KEY = "context_pack_hash"
# Containers under which a claim or the hash may be nested.
_NESTING_KEYS = ("metadata", "context")


def _find_claim_block(data: dict[str, Any]) -> dict[str, Any] | Any | None:
    """Return the claim value if a context-ROI claim is present, else None."""
    for key in _CLAIM_KEYS:
        if key in data and data[key] is not None:
            return data[key]
    for container in _NESTING_KEYS:
        nested = data.get(container)
        if isinstance(nested, dict):
            for key in _CLAIM_KEYS:
                if key in nested and nested[key] is not None:
                    return nested[key]
    return None


def _has_pinning_hash(data: dict[str, Any], claim: Any) -> bool:
    """Return True if a non-empty context_pack_hash is reachable for the claim."""

    def _nonempty(value: Any) -> bool:
        return isinstance(value, str) and value.strip() != ""

    # Inside the claim block itself.
    if isinstance(claim, dict) and _nonempty(claim.get(_HASH_KEY)):
        return True
    # Top-level.
    if _nonempty(data.get(_HASH_KEY)):
        return True
    # Under known nesting containers.
    for container in _NESTING_KEYS:
        nested = data.get(container)
        if isinstance(nested, dict) and _nonempty(nested.get(_HASH_KEY)):
            return True
    return False


def check_contract(path: Path) -> str | None:
    """Return a violation message if the contract claims ROI without a hash."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"unparseable contract {path}: {exc}") from exc
    if not isinstance(data, dict):
        return None
    claim = _find_claim_block(data)
    if claim is None:
        return None
    if _has_pinning_hash(data, claim):
        return None
    return (
        f"contract declares a context-ROI claim but carries no non-empty "
        f"'{_HASH_KEY}' to pin the measured context pack"
    )


def _iter_targets(root: Path, explicit: list[Path]) -> list[Path]:
    if explicit:
        return [p for p in explicit if p.name == "contract.yaml" and p.is_file()]
    return sorted(root.rglob("contract.yaml"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Explicit contract.yaml files to scan (pre-commit passes staged "
        "files). If omitted, scans --root recursively.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=_DEFAULT_ROOT,
        help="Directory to scan when no explicit files are given (default: src/).",
    )
    args = parser.parse_args(argv)

    if not args.files and not args.root.exists():
        print(f"ERROR: scan root not found: {args.root}", file=sys.stderr)
        return 2

    targets = _iter_targets(args.root, args.files)
    violations: list[tuple[Path, str]] = []
    try:
        for path in targets:
            msg = check_contract(path)
            if msg is not None:
                violations.append((path, msg))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if violations:
        print(
            "FAIL: context-ROI claim without context_pack_hash "
            "(OMN-13412 context-field-presence gate).",
            file=sys.stderr,
        )
        for path, msg in violations:
            rel = (
                path.relative_to(_REPO_ROOT)
                if path.is_relative_to(_REPO_ROOT)
                else path
            )
            print(f"  {rel}: {msg}", file=sys.stderr)
        return 1

    print(f"OK: no unpinned context-ROI claim in {len(targets)} contract(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
