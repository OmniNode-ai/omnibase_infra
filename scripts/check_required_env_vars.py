# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Check that docker-compose required env vars are declared in the in-repo manifest.

Parses the compose file for ``${VARNAME:?...}`` required-var patterns and diffs that set
against a checked-in, reviewable manifest of declared names (``docker/required-env-vars.manifest.txt``
by default). This validates a property of the two committed files — never the invoking
host's process environment or any local env file — so the result is identical on every
build host and in CI (OMN-15537).

Exit codes:
  0 — the compose file's required-var set matches the manifest exactly
  1 — the compose file and the manifest have diverged (additions and/or removals)
  2 — the compose file or the manifest file is missing

This hook intentionally does NOT check whether required vars have real values on the
invoking machine. That is a host-provisioning/bootstrap concern — see
``scripts/bootstrap-infisical.sh`` — not something a commit-time diff gate can honestly
assert (see OMN-15537 for why the prior host-environment-reading shape was a defect).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Matches ${VARNAME:?error-message} — docker-compose required-var syntax only.
# The :? form causes docker-compose to abort with an error if the variable is
# unset or empty, which is the class of failure this guard prevents.
_VAR_PATTERN = re.compile(r"\$\{([A-Z][A-Z0-9_]+):\?")


def _parse_compose_vars(compose_path: Path) -> set[str]:
    """Return the set of variable names referenced in *compose_path*."""
    content = compose_path.read_text(encoding="utf-8")
    return set(_VAR_PATTERN.findall(content))


def _parse_manifest_vars(manifest_path: Path) -> set[str]:
    """Return the set of variable names declared in *manifest_path*.

    One name per line; blank lines and ``#``-prefixed comment lines are ignored.
    """
    declared: set[str] = set()
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        declared.add(line)
    return declared


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that docker-compose required env vars (${VAR:?...}) are declared in "
            "the checked-in required-env-vars manifest — never against the invoking "
            "host's environment."
        ),
    )
    parser.add_argument(
        "--compose-file",
        default="docker/docker-compose.infra.yml",
        help="Path to the docker-compose file to inspect (default: docker/docker-compose.infra.yml)",
    )
    parser.add_argument(
        "--manifest-file",
        default="docker/required-env-vars.manifest.txt",
        help="Path to the declared-var manifest (default: docker/required-env-vars.manifest.txt)",
    )
    args = parser.parse_args(argv)

    compose_path = Path(args.compose_file)
    manifest_path = Path(args.manifest_file)

    if not compose_path.exists():
        print(f"ERROR: compose file not found: {compose_path}", file=sys.stderr)
        return 2
    if not manifest_path.exists():
        print(f"ERROR: manifest file not found: {manifest_path}", file=sys.stderr)
        return 2

    required_vars = _parse_compose_vars(compose_path)
    declared_vars = _parse_manifest_vars(manifest_path)

    undeclared = sorted(required_vars - declared_vars)
    stale = sorted(declared_vars - required_vars)

    if not undeclared and not stale:
        print(
            f"OK: all {len(required_vars)} env vars referenced in {compose_path} are "
            f"declared in {manifest_path}"
        )
        return 0

    print(
        f"ERROR: {compose_path} and {manifest_path} have diverged:",
        file=sys.stderr,
    )
    if undeclared:
        print(
            f"  {len(undeclared)} var(s) required by {compose_path} but not declared in "
            f"{manifest_path}:",
            file=sys.stderr,
        )
        for var in undeclared:
            print(f"    + {var}", file=sys.stderr)
    if stale:
        print(
            f"  {len(stale)} var(s) declared in {manifest_path} but no longer required by "
            f"{compose_path}:",
            file=sys.stderr,
        )
        for var in stale:
            print(f"    - {var}", file=sys.stderr)
    print(file=sys.stderr)
    print(
        f"Remediation — edit {manifest_path} to match the ${{VAR:?}} names in "
        f"{compose_path} (add new names, remove stale ones). This manifest declares "
        "names only, never values — it never touches ~/.omnibase/.env or any other "
        "runtime env file.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
