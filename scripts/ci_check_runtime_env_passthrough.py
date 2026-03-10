#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Verify that x-runtime-env in docker-compose.infra.yml contains all required keys.

Runtime containers inherit environment variables from the ``x-runtime-env`` YAML
anchor in ``docker/docker-compose.infra.yml``.  If a required key is missing from
the anchor, the container silently gets an empty value and the feature fails at
runtime with no obvious error.

This script parses the compose file with PyYAML (not regex -- regex-parsing YAML
is a future incident) and checks that every key in REQUIRED_KEYS appears in the
``x-runtime-env`` mapping.

Usage::

    # From the repo root
    uv run python scripts/ci_check_runtime_env_passthrough.py
    uv run python scripts/ci_check_runtime_env_passthrough.py --compose docker/docker-compose.infra.yml

Exit codes:
    0 -- all required keys present
    1 -- one or more required keys missing (stderr has details)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Required keys
# ---------------------------------------------------------------------------
# Maintain this list when a new env var becomes required for runtime
# containers.  Adding a new required key is a one-line change.

REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "INFISICAL_ADDR",
        "KAFKA_BOOTSTRAP_SERVERS",
        "OMNIBASE_INFRA_DB_URL",
        "ONEX_CONTRACTS_DIR",
        "POSTGRES_PASSWORD",
        "USE_EVENT_ROUTING",
    }
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _extract_runtime_env_keys(compose_path: Path) -> set[str] | None:
    """Parse *compose_path* and return the set of keys in ``x-runtime-env``.

    Returns ``None`` on any extraction or parsing error (with the error already
    printed to stderr).  Returns an empty ``set`` when ``x-runtime-env`` is a
    valid but empty mapping — callers must distinguish between the two cases.
    """
    with compose_path.open() as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        print(
            f"ERROR: {compose_path} did not parse as a YAML mapping",
            file=sys.stderr,
        )
        return None

    runtime_env = data.get("x-runtime-env")
    if runtime_env is None:
        print(
            f"ERROR: {compose_path} has no x-runtime-env anchor",
            file=sys.stderr,
        )
        return None

    if not isinstance(runtime_env, dict):
        print(
            f"ERROR: x-runtime-env is not a mapping (got {type(runtime_env).__name__})",
            file=sys.stderr,
        )
        return None

    return set(runtime_env.keys())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compose",
        default="docker/docker-compose.infra.yml",
        help="Path to docker-compose.infra.yml (default: docker/docker-compose.infra.yml)",
    )
    args = parser.parse_args(argv)

    compose_path = Path(args.compose)
    if not compose_path.exists():
        print(f"ERROR: compose file not found: {compose_path}", file=sys.stderr)
        return 1

    actual_keys = _extract_runtime_env_keys(compose_path)
    if actual_keys is None:
        # Extraction error already printed to stderr
        return 1

    missing = REQUIRED_KEYS - actual_keys
    if missing:
        print(
            "x-runtime-env completeness check FAILED:\n",
            file=sys.stderr,
        )
        for key in sorted(missing):
            print(
                f"  MISSING: {key} -- not listed in x-runtime-env",
                file=sys.stderr,
            )
        print(
            "\nAdd the missing key(s) to the x-runtime-env anchor in "
            "docker/docker-compose.infra.yml.\n"
            "See MEMORY.md: 'Docker container env passthrough' for details.",
            file=sys.stderr,
        )
        return 1

    print(
        f"x-runtime-env completeness check passed "
        f"({len(REQUIRED_KEYS)} required keys verified, "
        f"{len(actual_keys)} total keys in anchor)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
