# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Hold a sibling's published compatible range and its exact override together (OMN-19655).

Operator ruling 2026-09-25: omnibase_infra publishes a compatible range for
omnibase-core in ``[project.dependencies]`` and keeps the exact version it
builds, tests and images against only in ``[tool.uv] override-dependencies``.
A downstream core floor raise inside the minor then resolves against the
published infra instead of waiting for an infra release.

uv's override REPLACES the project requirement during resolution, so uv never
notices when the two disagree. A dependency cascade that moves the override to
0.48.0 while the range still says ``<0.48.0`` would lock, test and image 0.48.0
and then publish metadata that forbids it. This check refuses that, and every
other shape but the ruled one:

* ``[project.dependencies]`` carries exactly one requirement on the package,
  ``>=FLOOR,<NEXT_MINOR`` where NEXT_MINOR is FLOOR's minor plus one (``.0``);
* ``[tool.uv] override-dependencies`` carries an exact ``==V`` pin for it;
* the range admits V.

Standard library only. Exit codes: ``0`` the shape holds, ``1`` it does not
(each violation printed), ``2`` the pyproject could not be read.
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
_EXACT = re.compile(r"^==\s*([0-9]+(?:\.[0-9]+)*)$")
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


def _key(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def violations(data: dict[str, Any], package: str) -> list[str]:
    """Return every way ``data`` departs from the ruled range/override shape."""
    package = _normalize(package)
    published = _specs(data.get("project", {}).get("dependencies"), package)
    overrides = _specs(
        data.get("tool", {}).get("uv", {}).get("override-dependencies"), package
    )
    if len(published) != 1:
        return [
            f"[project.dependencies] must carry exactly one {package} requirement; "
            f"found {len(published)}"
        ]
    spec = published[0]
    if _EXACT.match(spec):
        return [
            f"[project.dependencies] publishes {package}{spec}, an exact pin. "
            f"Publish >=FLOOR,<NEXT_MINOR and keep the exact pin only in "
            f"[tool.uv] override-dependencies (OMN-19655)."
        ]
    ranged = _RANGE.match(spec)
    if ranged is None or (
        int(ranged.group(4)),
        int(ranged.group(5)),
        int(ranged.group(6)),
    ) != (int(ranged.group(1)), int(ranged.group(2)) + 1, 0):
        return [
            f"[project.dependencies] publishes {package}{spec!r}; the ruled shape is "
            f">=FLOOR,<NEXT_MINOR with NEXT_MINOR = FLOOR's minor + 1 and .0 "
            f"(OMN-19655)."
        ]
    exact = [m.group(1) for s in overrides if (m := _EXACT.match(s))]
    if len(overrides) != 1 or len(exact) != 1:
        return [
            f"[tool.uv] override-dependencies must carry exactly one exact "
            f"{package}==V pin, the version uv.lock and the image resolve; found "
            f"{overrides!r}"
        ]
    floor = _key(".".join(ranged.group(1, 2, 3)))
    ceiling = _key(".".join(ranged.group(4, 5, 6)))
    version = _key(exact[0])
    width = max(len(floor), len(version))
    padded = version + (0,) * (width - len(version))
    if (
        not floor + (0,) * (width - len(floor))
        <= padded
        < ceiling + (0,) * (width - len(ceiling))
    ):
        return [
            f"the published range {package}{spec} does not admit the override "
            f"{package}=={exact[0]} that uv.lock and the image resolve. Move the "
            f"range to cover it (a new minor is a deliberate range change), or "
            f"move the override back inside it."
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
        print(f"ERROR: cannot read {args.pyproject}: {exc}", file=sys.stderr)
        return 2
    found = violations(data, args.package)
    for item in found:
        print(f"ERROR: {args.pyproject}: {item}", file=sys.stderr)
    if not found:
        print(f"{args.pyproject}: {args.package} compatible range and override agree")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
