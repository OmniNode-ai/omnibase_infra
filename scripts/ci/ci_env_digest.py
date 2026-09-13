# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Compute the immutable key for the shared omnibase_infra CI environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tomllib
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

# Inputs hashed as raw bytes. ``pyproject.toml`` is deliberately NOT here: it
# participates through ``pyproject_dependency_projection`` instead (OMN-18351).
DEFAULT_ENV_INPUTS = (
    "uv.lock",
    ".github/actions/setup-python-uv/action.yml",
    "scripts/ci/ci_env_digest.py",
    "scripts/ci/ensure_ci_env.sh",
)

PYPROJECT_RELATIVE = "pyproject.toml"

# The parts of ``pyproject.toml`` that shape what the shared env and the runner
# image actually contain. The bake is
# ``uv sync --frozen --all-extras --all-groups --no-install-project``, so the
# installed set is: the resolved lock, the dependency tables below, the
# interpreter floor, and uv's own resolution settings. Nothing else in the file
# reaches the environment -- under ``--no-install-project`` the project is never
# installed, so its version, its entry points and its tool configuration are
# absent from the image by construction.
#
# OMN-18351. This is a NARROWING of the binding, not a relaxation of the gate.
# Hashing the whole file bound bytes that cannot change the image, which made
# every unrelated PR re-commit docker/runners/runner-image.lock.json -- and,
# because CI evaluates the lock against the MERGE tree, made the correct value
# unknowable from any single branch. A merge that combines dev's pyproject edit
# with the PR's produces a manifest neither parent hashed, so whichever side of
# the lock file the merge resolves to is wrong. Binding parsed values instead of
# file layout makes whole-table ordering irrelevant for independent edits; list
# ordering inside dependency tables still remains part of the binding.
DEPENDENCY_PATHS: tuple[tuple[str, ...], ...] = (
    ("build-system",),
    ("project", "requires-python"),
    ("project", "dependencies"),
    ("project", "optional-dependencies"),
    ("dependency-groups",),
    ("tool", "uv"),
)


def _json_ready(value: Any, path: tuple[str, ...]) -> Any:
    """Return ``value`` in canonical JSON form, or fail closed with context."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list):
        return [_json_ready(item, path) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_ready(item, (*path, str(key)))
            for key, item in value.items()
        }
    dotted = ".".join(path)
    if isinstance(value, datetime | date | time):
        raise TypeError(
            f"pyproject.toml value at {dotted} is not JSON-canonical: {value!r}"
        )
    raise TypeError(
        f"pyproject.toml value at {dotted} is not JSON-canonical: "
        f"{type(value).__name__}"
    )


def _lookup(document: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    cursor: Any = document
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            return {"present": False}
        cursor = cursor[key]
    return {"present": True, "value": _json_ready(cursor, path)}


def _table_names(document: dict[str, Any]) -> list[str]:
    """Return every TOML table name, sorted.

    This is the tripwire that keeps the allowlist above honest. A future table
    that shapes the environment would otherwise be dropped in silence; carrying
    the names means its arrival moves the digest and forces someone to decide
    whether it belongs in ``DEPENDENCY_PATHS``. Values inside existing tables
    stay governed by the path allowlist; this tripwire is only for new table
    surfaces, including nested installer configuration.
    """
    names: list[str] = []

    def visit(table: dict[str, Any], prefix: tuple[str, ...] = ()) -> None:
        for key, value in table.items():
            current = (*prefix, key)
            if not isinstance(value, dict):
                continue
            names.append(".".join(current))
            visit(value, current)

    visit(document)
    return sorted(names)


def pyproject_dependency_projection(repo_root: Path) -> bytes:
    """Return the canonical dependency projection of ``pyproject.toml``.

    Fails closed: an absent or unparseable manifest raises rather than
    projecting an empty dependency set, because hashing a default would make
    every broken tree agree with every other one -- silent under-binding, which
    is strictly worse than the churn this narrowing removes.
    """
    path = repo_root / PYPROJECT_RELATIVE
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    if not isinstance(document, dict):
        raise TypeError(f"pyproject.toml must parse to a table: {path}")

    projection = {
        "schema": 1,
        "tables": _table_names(document),
        "values": {
            ".".join(path_parts): _lookup(document, path_parts)
            for path_parts in DEPENDENCY_PATHS
        },
    }
    return json.dumps(projection, sort_keys=True, separators=(",", ":")).encode("utf-8")


def compute_digest(
    repo_root: Path,
    *,
    python_version: str,
    uv_version: str,
    install_args: str,
    extra: str = "",
    platform_id: str | None = None,
) -> str:
    """Return a short deterministic digest for the CI dependency environment."""
    root = repo_root.resolve()
    resolved_platform = platform_id or f"{platform.system()}-{platform.machine()}"
    payload = {
        "schema": 1,
        "repo": "omnibase_infra",
        "python_version": python_version,
        "uv_version": uv_version,
        "install_args": install_args,
        "platform": resolved_platform,
        "extra": extra,
    }

    digest = hashlib.sha256()
    digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    digest.update(b"\0")

    digest.update(PYPROJECT_RELATIVE.encode("utf-8"))
    digest.update(b"\0")
    digest.update(pyproject_dependency_projection(root))
    digest.update(b"\0")

    for relative in DEFAULT_ENV_INPUTS:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        if path.exists():
            digest.update(path.read_bytes())
        else:
            digest.update(b"<missing>")
        digest.update(b"\0")

    return digest.hexdigest()[:24]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute the shared omnibase_infra CI environment digest."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--uv-version", required=True)
    parser.add_argument("--install-args", required=True)
    parser.add_argument("--extra", default="")
    parser.add_argument("--platform-id")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    digest = compute_digest(
        args.repo_root,
        python_version=args.python_version,
        uv_version=args.uv_version,
        install_args=args.install_args,
        extra=args.extra,
        platform_id=args.platform_id,
    )
    if args.json:
        print(json.dumps({"digest": digest}, sort_keys=True))
    else:
        print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
