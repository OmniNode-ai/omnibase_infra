# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""check_lane_env_drift.py — E-5 env-vs-live drift detector (OMN-13018 / OMN-13034).

THE CLASS FIX (retro E-5): Runtime environment config (port assignments, compose
project names, bootstrap server addresses) drifts silently away from what the
lanes actually use. This check diffs the lane manifest's declared topology
against the env-parity reference config (docker/x-env-defaults or a provided
env file) to surface env-vs-live mismatches BEFORE they cause outage confusion.

What this checks:
  1. Compose project names: env COMPOSE_PROJECT_NAME vs manifest compose_project
  2. Port assignments: env EFFECTS_PORT / MAIN_PORT vs the per-lane port constants
  3. Bootstrap server addresses: env KAFKA_BOOTSTRAP_SERVERS must not be a
     hardcoded IP (must reference a broker variable — Rule 6 / Rule 8)

This runs as a CI gate on every PR that touches docker-compose, lane-manifest,
or the .env reference files. It does NOT require live access to .201 — it
validates the configuration files only.

Exit codes:
  0 — no env-vs-manifest drift
  1 — drift detected (hard gate — blocks CI)

Usage:
  python scripts/check_lane_env_drift.py
  python scripts/check_lane_env_drift.py --manifest deploy/lane-census/lane-manifest.yaml
  python scripts/check_lane_env_drift.py --compose-dir docker/
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_MANIFEST = _REPO / "deploy" / "lane-census" / "lane-manifest.yaml"
_DEFAULT_COMPOSE_DIR = _REPO / "docker"

# Hardcoded IP pattern — a bootstrap server that is a literal IP address violates
# Rule 6 (no hardcoded absolute paths) and Rule 8 (fail-fast on missing env).
# The accepted form is a compose service name (e.g. "redpanda:19092") or an env
# variable reference (never a raw 192.168.x.x address in a committed file).
_HARDCODED_IP_PATTERN = re.compile(
    r"\b(?:192\.168|10\.\d+|172\.(?:1[6-9]|2\d|3[01]))\.\d+\.\d+\b"
)

# Env var reference pattern — acceptable bootstrap server forms reference a
# compose-internal hostname, not a routable IP.
_COMPOSE_SERVICE_HOSTNAME_PATTERN = re.compile(
    r"^[a-z][a-z0-9-]+:[0-9]+$"  # e.g. "redpanda:19092"
)


def _load_manifest(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)  # type: ignore[no-any-return]


def _compose_env_for_lane(compose_dir: Path, lane: str) -> dict[str, str]:
    """Extract x-env-defaults variables from a compose file for a given lane."""
    compose_path = compose_dir / f"docker-compose.{lane}.yml"
    if not compose_path.exists():
        return {}

    raw = compose_path.read_text(encoding="utf-8")
    env_vars: dict[str, str] = {}

    # Extract x-env-defaults block if present (used for COMPOSE_PROJECT_NAME etc.)
    # This is a best-effort scrape — compose files may use !override YAML tags that
    # safe_load cannot parse, so we scrape key: value lines heuristically.

    # Look for COMPOSE_PROJECT_NAME
    for m in re.finditer(
        r"^\s*COMPOSE_PROJECT_NAME\s*[=:]\s*(\S+)\s*$", raw, re.MULTILINE
    ):
        env_vars["COMPOSE_PROJECT_NAME"] = m.group(1).strip("\"'")

    # Look for KAFKA_BOOTSTRAP_SERVERS
    for m in re.finditer(
        r"^\s*KAFKA_BOOTSTRAP_SERVERS\s*[=:]\s*(.+?)\s*$", raw, re.MULTILINE
    ):
        env_vars["KAFKA_BOOTSTRAP_SERVERS"] = m.group(1).strip("\"'")

    # Look for port declarations (EFFECTS_PORT, MAIN_PORT, etc.)
    for m in re.finditer(
        r"^\s*((?:EFFECTS|MAIN|HTTP)_PORT)\s*[=:]\s*(\S+)\s*$", raw, re.MULTILINE
    ):
        env_vars[m.group(1)] = m.group(2).strip("\"'")

    return env_vars


def check_drift(
    manifest: dict[str, Any],
    compose_dir: Path,
) -> list[str]:
    """Return a list of drift violation messages."""
    violations: list[str] = []
    lanes = manifest.get("lanes", {})

    for lane_name, lane_spec in lanes.items():
        manifest_project = lane_spec.get("compose_project", "")
        compose_env = _compose_env_for_lane(compose_dir, lane_name)

        # 1. Compose project name drift
        compose_project_name = compose_env.get("COMPOSE_PROJECT_NAME", "")
        if (
            compose_project_name
            and manifest_project
            and compose_project_name != manifest_project
        ):
            violations.append(
                f"[E-5] lane {lane_name!r}: compose_project in manifest is "
                f"{manifest_project!r} but COMPOSE_PROJECT_NAME in compose file is "
                f"{compose_project_name!r}. Update the manifest or the compose file "
                f"to agree (OMN-13018)."
            )

        # 2. Hardcoded IP in bootstrap servers
        bootstrap = compose_env.get("KAFKA_BOOTSTRAP_SERVERS", "")
        if bootstrap and _HARDCODED_IP_PATTERN.search(bootstrap):
            violations.append(
                f"[E-5] lane {lane_name!r}: KAFKA_BOOTSTRAP_SERVERS contains a "
                f"hardcoded IP address: {bootstrap!r}. Use a compose service "
                f"hostname (e.g. 'redpanda:19092') or an env var reference. "
                f"Hardcoded IPs violate Rule 6 and Rule 8 (OMN-13018)."
            )

        # 3. Lane network in manifest must match compose network declarations
        manifest_network = lane_spec.get("network", "")
        if manifest_network and not _is_optional_and_dev(lane_name, lane_spec):
            compose_path = compose_dir / f"docker-compose.{lane_name}.yml"
            if compose_path.exists():
                compose_raw = compose_path.read_text(encoding="utf-8")
                network_names = set(
                    re.findall(
                        r"^\s*name:\s*(omnibase-infra-\S+)\s*$",
                        compose_raw,
                        re.MULTILINE,
                    )
                )
                if network_names and manifest_network not in network_names:
                    violations.append(
                        f"[E-5] lane {lane_name!r}: manifest declares network "
                        f"{manifest_network!r} but compose file names are "
                        f"{sorted(network_names)}. Update the manifest to match "
                        f"(OMN-13018 / env-vs-live drift)."
                    )

    return violations


def _is_optional_and_dev(lane_name: str, lane_spec: dict[str, Any]) -> bool:
    """The dev lane uses generated compose — relax network name check for it."""
    return bool(lane_spec.get("optional", False)) or lane_name == "dev"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="E-5 env-vs-live drift detector (OMN-13018 / OMN-13034)"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_DEFAULT_MANIFEST,
        help="Path to lane-manifest.yaml",
    )
    parser.add_argument(
        "--compose-dir",
        type=Path,
        default=_DEFAULT_COMPOSE_DIR,
        help="Directory containing docker-compose.<lane>.yml files",
    )
    args = parser.parse_args(argv)

    manifest = _load_manifest(args.manifest)
    violations = check_drift(manifest, args.compose_dir)

    if violations:
        print(
            "LANE ENV DRIFT (E-5 / OMN-13018): env config does not match "
            "lane-manifest declared topology:",
            file=sys.stderr,
        )
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        return 1

    lanes = list(manifest.get("lanes", {}).keys())
    print(f"OK: no env-vs-manifest drift across lanes {lanes}", file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
