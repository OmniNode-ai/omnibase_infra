# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Check that required compose env vars are set in configured env files.

Parses the compose file for ${VARNAME:?...} required-var patterns and validates that each
var is present with a non-empty value in at least one configured env file.

Exit codes:
  0 — all required vars are set
  1 — one or more required vars are missing
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


def _parse_env_file(env_path: Path) -> set[str]:
    """Return variable names that are set (non-empty) in *env_path*."""
    if not env_path.exists():
        return set()
    set_vars: set[str] = set()
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key and value:
            set_vars.add(key)
    return set_vars


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify that docker-compose required env vars are set in configured env files",
    )
    parser.add_argument(
        "--compose-file",
        default="docker/docker-compose.infra.yml",
        help="Path to the docker-compose file to inspect (default: docker/docker-compose.infra.yml)",
    )
    parser.add_argument(
        "--env-file",
        action="append",
        default=None,
        help=(
            "Path to an env file to validate against. Repeatable. Defaults to "
            "docker/runtime-policy.env and ~/.omnibase/.env."
        ),
    )
    args = parser.parse_args(argv)

    compose_path = Path(args.compose_file)
    env_paths = (
        [Path("docker/runtime-policy.env"), Path("~/.omnibase/.env")]
        if args.env_file is None
        else [Path(path) for path in args.env_file]
    )
    env_paths = [path.expanduser() for path in env_paths]

    if not compose_path.exists():
        print(f"ERROR: compose file not found: {compose_path}", file=sys.stderr)
        return 1

    required_vars = _parse_compose_vars(compose_path)
    set_vars: set[str] = set()
    for env_path in env_paths:
        set_vars.update(_parse_env_file(env_path))

    missing = sorted(required_vars - set_vars)

    if not missing:
        env_path_text = ", ".join(str(path) for path in env_paths)
        print(
            f"OK: all {len(required_vars)} env vars referenced in {compose_path} are set in {env_path_text}"
        )
        return 0

    env_path_text = ", ".join(str(path) for path in env_paths)
    print(
        f"ERROR: {len(missing)} env var(s) referenced in {compose_path} are missing from {env_path_text}:",
        file=sys.stderr,
    )
    for var in missing:
        print(f"  {var}", file=sys.stderr)
    print(file=sys.stderr)
    print("Remediation — add the missing vars to your env file:", file=sys.stderr)
    env_path = env_paths[-1]
    for var in missing:
        print(f"  echo '{var}=<set_me>' >> {env_path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
