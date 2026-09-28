# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Validate the immutable source revisions declared for a sim runtime image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_SOURCE_NAMES = frozenset({"omnibase_core", "omnibase_infra"})


def verify(path: Path) -> dict[str, str]:
    """Return the two exact source revisions or fail before lifecycle work."""
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(raw, dict)
        or raw.get("profile") != "sim-preflight-runtime-source-pins-v1"
    ):
        raise ValueError("invalid sim preflight runtime source pin profile")
    sources = raw.get("sources")
    if not isinstance(sources, dict) or set(sources) != _SOURCE_NAMES:
        raise ValueError("sim runtime source pins must name core and infra exactly")
    if not all(
        isinstance(revision, str)
        and len(revision) == 40
        and all(character in "0123456789abcdef" for character in revision)
        for revision in sources.values()
    ):
        raise ValueError("sim runtime source pins must be lowercase Git revisions")
    return {name: sources[name] for name in sorted(_SOURCE_NAMES)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"verified": verify(args.profile)}, sort_keys=True))


if __name__ == "__main__":
    main()
