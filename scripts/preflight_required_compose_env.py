# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Report EVERY unset required compose variable in one message, before validation.

OMN-17530. ``docker compose config`` reports the FIRST unset ``${VAR:?}`` it
reaches and stops. A lane whose compose files require N variables that the host
does not carry therefore surfaces as N consecutive failed deploys, each one
naming exactly one variable, with no way to see the size of the hole until the
last one is fixed. That is not hypothetical: two dev-lane deploy commands died
~14 minutes apart on 2026-09-08 on two different variables added by the same PR,
and a mechanical enumeration afterwards put the real count at ten.

This preflight runs BEFORE compose validation, parses the ``${VAR:?}`` set out
of every compose file the lane loads, checks each name against the environment
this process was handed, and prints the WHOLE missing set in a single message —
each name with the store file that should carry it.

WHY THIS IS NOT ``scripts/check_required_env_vars.py``
-----------------------------------------------------
That script is a commit-time gate. It deliberately never reads the invoking
host's environment: it diffs two committed files (a compose file and its
declared-name manifest) so the result is identical on every build host and in CI
(OMN-15537). This script is the deploy-time half and does the opposite — it
reads the environment and answers "does THIS host carry values". The two are
complementary and neither substitutes for the other:

    check_required_env_vars.py   is the NAME declared?      (committed files)
    preflight_required_compose_env.py  does the host carry a VALUE?  (this host)

Values are never printed, never logged and never compared — only presence and
emptiness. A variable set to the empty string counts as missing, because that is
exactly how ``${VAR:?}`` treats it.

Exit codes:
  0 — every required variable is set and non-empty in this environment
  1 — at least one is missing (all of them are named in the message)
  2 — a compose file could not be read
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# docker-compose required-var syntax: ${VARNAME:?message}. The :? form aborts
# compose when the variable is unset OR empty. Same pattern as
# scripts/check_required_env_vars.py — deliberately duplicated rather than
# imported, so that script and this one cannot weaken each other by accident.
_REQUIRED_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):\?")

_DEFAULT_RUNTIME_POLICY_ENV = "docker/runtime-policy.env"


def _required_names(compose_path: Path) -> set[str]:
    """Return the ``${VAR:?}`` names referenced in *compose_path*."""
    return set(_REQUIRED_VAR_PATTERN.findall(compose_path.read_text(encoding="utf-8")))


def _dotenv_keys(path: Path) -> set[str]:
    """Return the KEY names a dotenv-style file declares. Values are not read."""
    keys: set[str] = set()
    if not path.is_file():
        return keys
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if key:
            keys.add(key)
    return keys


def _operator_env_display_path() -> str:
    """The operator env store path, as a portable display string.

    Never interpolated into a committed default — an absolute home path in
    source is a portability bug (Operating Rule 6). This resolves the same knob
    deploy-runtime.sh and refresh_dev_lane.sh resolve.
    """
    explicit = os.environ.get("OMNIBASE_OPERATOR_ENV_FILE")
    if explicit:
        return explicit
    return "${OMNIBASE_OPERATOR_ENV_FILE:-$HOME/.omnibase/.env}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Report every unset required (${VAR:?}) compose variable in one "
            "message, before compose validation reports only the first."
        ),
    )
    parser.add_argument(
        "--compose-file",
        action="append",
        required=True,
        dest="compose_files",
        metavar="PATH",
        help="A compose file the lane loads. Repeat for each file, in load order.",
    )
    parser.add_argument(
        "--runtime-policy-env",
        default=_DEFAULT_RUNTIME_POLICY_ENV,
        help=(
            "Path to the contract-rendered runtime policy env file, used ONLY to "
            "decide which store a missing name should have come from "
            f"(default: {_DEFAULT_RUNTIME_POLICY_ENV})."
        ),
    )
    parser.add_argument(
        "--lane",
        default="dev",
        help="Lane name, for the message only (default: dev).",
    )
    args = parser.parse_args(argv)

    compose_paths = [Path(p) for p in args.compose_files]
    missing_files = [p for p in compose_paths if not p.is_file()]
    if missing_files:
        for path in missing_files:
            print(f"ERROR: compose file not found: {path}", file=sys.stderr)
        return 2

    required: set[str] = set()
    source_of: dict[str, list[str]] = {}
    for path in compose_paths:
        for name in _required_names(path):
            required.add(name)
            source_of.setdefault(name, []).append(path.name)

    policy_keys = _dotenv_keys(Path(args.runtime_policy_env))

    missing = sorted(name for name in required if not os.environ.get(name, "").strip())

    if not missing:
        print(
            f"OK: all {len(required)} required compose variables for lane "
            f"'{args.lane}' are set in this environment "
            f"({', '.join(p.name for p in compose_paths)})."
        )
        return 0

    operator_env = _operator_env_display_path()
    lines = [
        f"ERROR: REQUIRED_COMPOSE_ENV_MISSING — {len(missing)} of "
        f"{len(required)} required compose variables are unset or empty for lane "
        f"'{args.lane}'.",
        "",
        "  Reported as a SET, deliberately. `docker compose config` stops at the",
        "  first one, so fixing them one deploy at a time costs one failed deploy",
        "  per variable (OMN-17530). Every name below must be set before the next",
        "  deploy attempt — supplying only the first will not get further.",
        "",
    ]
    for name in missing:
        if name in policy_keys:
            store = (
                f"{args.runtime_policy_env} (contract-rendered — re-render the "
                "runtime policy contract; do not hand-edit)"
            )
        else:
            store = operator_env
        referenced_by = ", ".join(sorted(set(source_of[name])))
        lines.append(f"    {name}")
        lines.append(f"        required by : {referenced_by}")
        lines.append(f"        store file  : {store}")
    lines.append("")
    lines.append(
        "  Every name above is declared in this repo — docker/required-env-vars."
        "manifest.txt for the base compose, docker/dev-lane-required-env.manifest.txt"
    )
    lines.append(
        "  for the dev-lane overlay — so this list is the complete set for these "
        "files, not the first wall of an unknown number."
    )
    lines.append(
        "  No values are read or printed by this check; only whether each name is "
        "set and non-empty."
    )
    print("\n".join(lines), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
