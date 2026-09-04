# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""disk_watermark_thresholds.py — Read the guard's own threshold declaration (OMN-17872).

The disk guard's thresholds are declared in ``disk-watermark-thresholds.json``
next to it, never in an environment variable. This reader validates the
declaration and prints the three numbers on one line for the shell to consume:

    crit_free_gb warn_free_gb warn_pct

It is fail-fast by construction (Rule 8). A missing file, malformed JSON, a
missing key, a non-integer, or a warn floor at or below the critical floor is a
non-zero exit with a message on stderr — never a built-in default. A silently
defaulted admission threshold is exactly the failure OMN-17872 removes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REQUIRED_KEYS = ("crit_free_gb", "warn_free_gb", "warn_pct")


def load_thresholds(path: Path) -> tuple[int, int, int]:
    """Return (crit_free_gb, warn_free_gb, warn_pct) or raise ValueError."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"threshold declaration not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"threshold declaration is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError("threshold declaration must be a JSON object")

    values: list[int] = []
    for key in _REQUIRED_KEYS:
        if key not in raw:
            raise ValueError(f"threshold declaration is missing {key!r}")
        value = raw[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{key} must be a positive integer, got {value!r}")
        values.append(value)

    crit_free_gb, warn_free_gb, warn_pct = values
    if warn_free_gb <= crit_free_gb:
        raise ValueError(
            f"warn_free_gb ({warn_free_gb}) must exceed crit_free_gb "
            f"({crit_free_gb}) — the warning has to arrive before the halt"
        )
    if warn_pct > 100:
        raise ValueError(f"warn_pct must be <= 100, got {warn_pct}")

    return crit_free_gb, warn_free_gb, warn_pct


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <thresholds.json>", file=sys.stderr)
        return 2
    try:
        crit_free_gb, warn_free_gb, warn_pct = load_thresholds(Path(argv[1]))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"{crit_free_gb} {warn_free_gb} {warn_pct}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
