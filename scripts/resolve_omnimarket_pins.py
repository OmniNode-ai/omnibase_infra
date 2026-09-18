# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read co-install pin versions out of a pyproject.toml, by distribution name.

OMN-18675. ``scripts/install-node-skill-package.sh`` co-installs a small set of
omni-internal packages beside omnimarket with ``--no-deps``. Their versions used
to be shell literals in that script, written once on 2026-07-02 and never
revisited; because ``--no-deps`` suppresses resolution and ``==`` is an exact
instruction, those stale literals silently DOWNGRADED a healthy shared venv and
broke the ``onex`` CLI host-wide.

The single source of truth for what omnimarket needs is omnimarket's own
``[project].dependencies``, at the exact ref being installed. This module reads
it and prints one PEP 508 requirement per line, in the order the names were
asked for. A name the ref does not declare is reported on stderr and omitted
rather than guessed.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_UNREADABLE = 3

# PEP 508: the distribution name runs until the first extras/specifier/marker
# character. That prefix is all this module needs to match on.
_REQUIREMENT_NAME = re.compile(r"^\s*(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)")


def normalize_name(name: str) -> str:
    """Normalize a distribution name per PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(requirement: str) -> str | None:
    """Return the normalized distribution name a requirement string names."""
    match = _REQUIREMENT_NAME.match(requirement)
    if match is None:
        return None
    return normalize_name(match.group("name"))


def select_requirements(
    dependencies: list[str], wanted: list[str]
) -> tuple[list[str], list[str]]:
    """Pick the declared requirements for ``wanted``, preserving ask order.

    Returns ``(found_requirements, missing_names)``.
    """
    by_name: dict[str, str] = {}
    for dependency in dependencies:
        name = requirement_name(dependency)
        if name is not None and name not in by_name:
            by_name[name] = dependency.strip()

    found: list[str] = []
    missing: list[str] = []
    for raw in wanted:
        name = normalize_name(raw)
        if name in by_name:
            found.append(by_name[name])
        else:
            missing.append(raw)
    return found, missing


def read_dependencies(pyproject_path: Path) -> list[str]:
    """Return ``[project].dependencies`` from a pyproject.toml."""
    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)
    project = data.get("project")
    if not isinstance(project, dict):
        raise ValueError(f"{pyproject_path} has no [project] table")
    dependencies = project.get("dependencies", [])
    if not isinstance(dependencies, list):
        raise ValueError(f"{pyproject_path} has a non-list [project].dependencies")
    return [str(item) for item in dependencies]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Print the requirement strings a pyproject.toml declares for the "
            "given distribution names, one per line."
        )
    )
    parser.add_argument(
        "--pyproject", required=True, help="path to the pyproject.toml to read"
    )
    parser.add_argument("names", nargs="*", help="distribution names to resolve")
    args = parser.parse_args(argv)

    if not args.names:
        print("ERROR: no distribution names given.", file=sys.stderr)
        return EXIT_USAGE

    path = Path(args.pyproject)
    try:
        dependencies = read_dependencies(path)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"ERROR: cannot read dependencies from {path}: {exc}", file=sys.stderr)
        return EXIT_UNREADABLE

    found, missing = select_requirements(dependencies, list(args.names))
    for name in missing:
        print(
            f"NOTE: {path} does not declare {name!r} — it will not be co-installed.",
            file=sys.stderr,
        )
    for requirement in found:
        print(requirement)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
