# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Hold OMN-19655's published >=FLOOR,<NEXT_MINOR shape; core's dev_tag_lineage
validator, run by CI, owns the tag, override and lock rules.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_REQUIREMENT = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(.*?)\s*$")
_RANGE = re.compile(
    r"^>=\s*([0-9]+)\.([0-9]+)\.([0-9]+)\s*,\s*<\s*([0-9]+)\.([0-9]+)\.([0-9]+)$"
)


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _specs(requirements: Any, package: str) -> list[str]:
    if not isinstance(requirements, list):
        return []
    found: list[str] = []
    for raw in requirements:
        match = _REQUIREMENT.match(str(raw).split(";", 1)[0])
        if match and _normalize(match.group(1)) == package:
            found.append(match.group(2))
    return found


def violations(data: dict[str, Any], package: str) -> list[str]:
    """Return violations of the one published compatible-range requirement."""
    package = _normalize(package)
    published = _specs(data.get("project", {}).get("dependencies"), package)
    if len(published) != 1:
        return [
            f"[project.dependencies] must carry exactly one {package} requirement; "
            f"found {len(published)}"
        ]
    spec = published[0]
    if spec.startswith("=="):
        return [
            f"[project.dependencies] publishes {package}{spec}, an exact pin. "
            f"Publish >=FLOOR,<NEXT_MINOR (OMN-19655)."
        ]
    ranged = _RANGE.match(spec)
    if ranged is None or tuple(map(int, ranged.group(4, 5, 6))) != (
        int(ranged.group(1)),
        int(ranged.group(2)) + 1,
        0,
    ):
        return [
            f"[project.dependencies] publishes {package}{spec!r}; the ruled shape is "
            f">=FLOOR,<NEXT_MINOR with NEXT_MINOR = FLOOR's minor + 1 and .0 "
            f"(OMN-19655)."
        ]
    return []


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--package", default="omnibase-core")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        data = tomllib.loads(args.pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        print(
            f"ERROR: cannot read {args.pyproject}: {exc}",
            file=sys.stderr,
        )
        return 2
    found = violations(data, args.package)
    for item in found:
        print(f"ERROR: {args.pyproject}: {item}", file=sys.stderr)
    if not found:
        print(
            f"{args.pyproject}: {args.package} publishes a compatible floor/next-minor range"
        )
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
