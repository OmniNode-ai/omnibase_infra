# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""EFFECT handler for the sibling-clone pin resolution node.

Reads the package's ``pyproject.toml`` and runs ``git`` to resolve the actual
pin (version + git SHA). When the pyproject leaves its version to git (dynamic
versioning), the version is derived from ``git describe`` using the pyproject's
own ``tool.hatch.version.raw-options.git_describe_command``.

Ticket: OMN-23947
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any

from omnibase_infra.models.sibling_pin.model_sibling_clone_pin_request import (
    ModelSiblingClonePinRequest,
)
from omnibase_infra.models.sibling_pin.model_sibling_clone_pin_result import (
    ModelSiblingClonePinResult,
)

# Regex to parse `git describe --long` output: tag-distance-gsha[-dirty]
_GIT_DESCRIBE_RE: re.Pattern[str] = re.compile(
    r"^(?P<tag>v?[0-9]+(?:\.[0-9]+)*)-(?P<distance>[0-9]+)-g(?P<sha>[0-9a-fA-F]{7,40})(?P<dirty>-dirty)?$"
)


def derive_git_version(repo_root: Path, project_data: dict[str, Any]) -> str:
    """Derive a version from git using the pyproject's git_describe_command.

    Runs the pyproject's own ``tool.hatch.version.raw-options.git_describe_command``
    (a list such as ``["git", "describe", "--dirty", "--tags", "--long",
    "--match", "v[0-9]*"]``). There is NO fallback command: if the pyproject
    names no ``git_describe_command``, a ``ValueError`` is raised naming the
    pyproject. A leading ``"git"`` element is dropped, then the command is run
    with ``safe.directory`` and ``-C`` flags.

    The output is parsed with ``_GIT_DESCRIBE_RE``. No match raises
    ``ValueError``. A leading ``"v"`` is stripped from the tag. If distance is
    0 and not dirty, the tag is returned. Otherwise the last dot-separated
    integer of the tag is bumped by one and the result is
    ``f"{bumped}.dev{distance}+g{sha}"`` with ``".dirty"`` appended when dirty.

    Args:
        repo_root: Filesystem path to the repository root.
        project_data: The parsed ``pyproject.toml`` data (full dict).

    Returns:
        The derived version string.

    Raises:
        ValueError: If no ``git_describe_command`` is found, or the output
            cannot be parsed.
    """
    raw_options = (
        project_data.get("tool", {})
        .get("hatch", {})
        .get("version", {})
        .get("raw-options", {})
    )
    git_describe_command = raw_options.get("git_describe_command")
    if not git_describe_command:
        raise ValueError(f"no git_describe_command in pyproject for {repo_root}")

    # Drop a leading "git" element if present.
    args: list[str] = list(git_describe_command)
    if args and args[0] == "git":
        args = args[1:]

    cmd: list[str] = [
        "git",
        "-c",
        f"safe.directory={repo_root}",
        "-C",
        str(repo_root),
        *args,
    ]
    out = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    m = _GIT_DESCRIBE_RE.match(out)
    if not m:
        raise ValueError(f"cannot parse git describe output {out!r} for {repo_root}")

    tag = m.group("tag")
    distance = int(m.group("distance"))
    sha = m.group("sha")
    dirty = m.group("dirty") is not None

    # Strip a leading "v" from the tag.
    if tag.startswith("v"):
        tag = tag[1:]

    if distance == 0 and not dirty:
        return tag

    # Bump the last dot-separated integer of the tag by one.
    parts = tag.split(".")
    parts[-1] = str(int(parts[-1]) + 1)
    bumped = ".".join(parts)

    result = f"{bumped}.dev{distance}+g{sha}"
    if dirty:
        result += ".dirty"
    return result


def resolve_clone_pin(
    request: ModelSiblingClonePinRequest,
) -> ModelSiblingClonePinResult:
    """Resolve the actual pin for the given package and repo root.

    Reads the package's ``pyproject.toml``, resolves the version (``project.version``,
    or derived from git when the pyproject lists it as dynamic) and runs
    ``git rev-parse HEAD`` for the commit.

    Raises:
        FileNotFoundError: If ``pyproject.toml`` is missing.
        ValueError: If no version can be resolved, or git HEAD cannot be resolved.
    """
    package: str = request.package
    repo_root: Path = request.repo_root

    pyproject_path = repo_root / "pyproject.toml"
    if not pyproject_path.is_file():
        raise FileNotFoundError(
            f"missing pyproject.toml for {package}: {pyproject_path}"
        )
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data.get("project", {})
    version = project.get("version")
    version_derived = False
    if not version and "version" in project.get("dynamic", []):
        version = derive_git_version(repo_root, data)
        version_derived = True
    if not version:
        raise ValueError(f"no project.version in {pyproject_path}")

    head_sha = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo_root}",
            "-C",
            str(repo_root),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not head_sha:
        raise ValueError(f"could not resolve git HEAD for {repo_root}")

    return ModelSiblingClonePinResult(
        package=package,
        version=version,
        version_derived=version_derived,
        git_sha=head_sha,
    )


class HandlerSiblingClonePin:
    """EFFECT handler: answers the pin (version and HEAD commit) of a sibling clone.

    Reads the clone's ``pyproject.toml`` and runs ``git`` in the clone; writes nothing.
    """

    def handle(
        self, request: ModelSiblingClonePinRequest
    ) -> ModelSiblingClonePinResult:
        """Resolve the pin of the clone the request names."""
        return resolve_clone_pin(request)


__all__: list[str] = [
    "HandlerSiblingClonePin",
    "derive_git_version",
    "resolve_clone_pin",
]
