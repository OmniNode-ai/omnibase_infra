#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Reject seed-prefixed pricing manifest keys."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_SEED_KEY_RE = re.compile(
    r"""^\s*(?P<quote>["']?)seed_[A-Za-z0-9_-]*(?P=quote)\s*:""",
    re.MULTILINE,
)


def lint_pricing_manifest(path: Path) -> list[str]:
    """Return lint errors for seed-prefixed manifest keys."""
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    for match in _SEED_KEY_RE.finditer(text):
        line_no = text.count("\n", 0, match.start()) + 1
        key = match.group(0).strip().removesuffix(":")
        errors.append(f"{path}:{line_no}: seed pricing key is forbidden: {key}")
    return errors


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("src/omnibase_infra/configs/pricing_manifest.yaml")],
        help="Pricing manifest paths to lint.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    errors: list[str] = []
    for path in args.paths:
        errors.extend(lint_pricing_manifest(path))

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
