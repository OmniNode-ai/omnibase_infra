#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Refuse Coding Plan endpoints in git-tracked configuration (OMN-20173).

Coding Plan quota is reserved for Claude Code; system callers must not ship
these endpoints. YAML is parsed so comments never become findings.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

_ENDPOINT = re.compile(r"https?://[^\s\"']*/api/coding(/|$)|api\.z\.ai/api/coding")
_ROOT = Path(__file__).resolve().parents[1]


def _is_config(path: Path) -> bool:
    if {"tests", "docs"} & set(path.parts) or path.suffix.lower() == ".md":
        return False
    name = path.name.lower()
    return (
        path.suffix.lower() in {".yaml", ".yml", ".json", ".toml", ".env", ".template"}
        or name == ".env"
        or ".env." in name
        or name.startswith((".env.", "compose.", "docker-compose"))
        or name == "compose"
    )


def _yaml_values(node: Node, seen: set[int]) -> Iterator[ScalarNode]:
    """Walk scalar values, excluding mapping keys and recursive aliases."""
    if id(node) in seen:
        return
    seen.add(id(node))
    if isinstance(node, ScalarNode):
        yield node
    elif isinstance(node, MappingNode):
        for _, value in node.value:
            yield from _yaml_values(value, seen)
    elif isinstance(node, SequenceNode):
        for value in node.value:
            yield from _yaml_values(value, seen)


def check_file(path: Path) -> list[int]:
    """Return offending line numbers without exposing configured values."""
    source = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        lines: set[int] = set()
        for document in yaml.compose_all(source, Loader=yaml.SafeLoader):
            if document is not None:
                for scalar in _yaml_values(document, set()):
                    if _ENDPOINT.search(scalar.value):
                        lines.add(scalar.start_mark.line + 1)
        return sorted(lines)
    return [
        number
        for number, line in enumerate(source.splitlines(), start=1)
        if not line.lstrip().startswith(("#", "//", ";")) and _ENDPOINT.search(line)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_ROOT)
    root = parser.parse_args(argv).root.resolve()
    try:
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            capture_output=True,
            text=True,
            check=True,
            env=scrub_git_location_env(os.environ),
        ).stdout.split("\0")
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Cannot list tracked configuration: {exc}", file=sys.stderr)
        return 1

    findings: list[str] = []
    for name in tracked:
        relative = Path(name)
        path = root / relative
        if not name or not _is_config(relative) or not path.is_file():
            continue
        try:
            findings.extend(f"{relative}:{line}" for line in check_file(path))
        except (OSError, UnicodeError, yaml.YAMLError):
            findings.append(f"{relative}: cannot read or parse configuration")
    if findings:
        print("Coding Plan endpoint check failed (OMN-20173):", file=sys.stderr)
        for finding in findings:
            print(f"  {finding}", file=sys.stderr)
        return 1
    print("Coding Plan endpoint check passed (OMN-20173)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
