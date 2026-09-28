# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Resolve the arm64 verify runner proof's legs (OMN-19894, OMN-19895).

The sibling of .github/actions/resolve-lab-lane: it turns overlay data into
what a job runs on, and refuses rather than defaults.

The proof has one leg per arm64 verify host, so each leg is a statement about
one machine. The legs used to be a literal host-label matrix in the workflow,
which pinned the check definition to named machines. They now come from the
repository variable ARM64_VERIFY_PROOF_LEGS_JSON, which is data:

    [{"name": "<leg name>",
      "runs_on": ["self-hosted", "omnibase-verify", "arch-arm64", "<host label>"],
      "lab_credentials": true}, ...]

``lab_credentials`` says whether that runner is expected to carry the lab
kubeconfig (a lane-pinned runner does not, by design).

This script checks the variable against the declared inventory before any leg
runs. The inventory is config/runner_fleet.yaml, plus the compose file of each
host that realises it. The host labels in the legs must be exactly the host
labels that the arm64 verify hosts' runners register. A missing leg would leave
a declared host unproven. An extra leg would wait on a machine that never
answers. Both are RED, and so is an unset or malformed variable. Nothing
defaults.

Writes ``legs=<json list>`` to $GITHUB_OUTPUT when it is set, and prints it.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
FLEET_CONFIG = REPO_ROOT / "config" / "runner_fleet.yaml"
PRIMARY_COMPOSE = REPO_ROOT / "docker" / "docker-compose.runners.yml"
VARIABLE = "ARM64_VERIFY_PROOF_LEGS_JSON"
REQUIRED_LABELS = ("self-hosted", "omnibase-verify", "arch-arm64")
HOST_LABEL = re.compile(r"host-[A-Za-z0-9]+")


class LegsError(ValueError):
    """The legs variable disagrees with itself or with the inventory."""


def declared_host_labels(
    fleet_config: Path = FLEET_CONFIG, primary_compose: Path = PRIMARY_COMPOSE
) -> set[str]:
    """The host labels that the arm64 verify hosts' runners register."""
    fleet = yaml.safe_load(fleet_config.read_text(encoding="utf-8"))
    primary_prefix = fleet["runner_name_prefix"]
    labels: set[str] = set()
    for host in fleet.get("hosts") or []:
        if host.get("arch") != "arm64" or "verify" not in (host.get("classes") or []):
            continue
        prefix = host["runner_name_prefix"]
        compose = (
            primary_compose
            if prefix == primary_prefix
            else primary_compose.parent / f"docker-compose.runners-{prefix}.yml"
        )
        services = yaml.safe_load(compose.read_text(encoding="utf-8"))["services"]
        for name, definition in services.items():
            if not re.fullmatch(rf"{re.escape(prefix)}-\d+", name):
                continue
            raw = str((definition.get("environment") or {}).get("RUNNER_LABELS", ""))
            labels.update(x for x in raw.split(",") if HOST_LABEL.fullmatch(x))
    return labels


def parse_legs(raw: str | None) -> list[dict[str, object]]:
    if raw is None or not raw.strip():
        raise LegsError(f"{VARIABLE} is unset or empty; the proof has no legs")
    try:
        legs = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LegsError(f"{VARIABLE} is not JSON: {exc}") from exc
    if not isinstance(legs, list) or not legs:
        raise LegsError(f"{VARIABLE} must be a non-empty JSON list of legs")
    for index, leg in enumerate(legs):
        if not isinstance(leg, Mapping):
            raise LegsError(f"leg {index} is not an object: {leg!r}")
        name = leg.get("name")
        runs_on = leg.get("runs_on")
        creds = leg.get("lab_credentials")
        if not isinstance(name, str) or not name:
            raise LegsError(f"leg {index} has no name")
        if not isinstance(runs_on, Sequence) or isinstance(runs_on, str):
            raise LegsError(f"leg {name!r}: runs_on must be a list of labels")
        if not all(isinstance(label, str) for label in runs_on):
            raise LegsError(f"leg {name!r}: every runs_on label must be a string")
        missing = [label for label in REQUIRED_LABELS if label not in runs_on]
        if missing:
            raise LegsError(f"leg {name!r}: runs_on lacks {missing}")
        hosts = [label for label in runs_on if HOST_LABEL.fullmatch(label)]
        if len(hosts) != 1:
            raise LegsError(
                f"leg {name!r}: runs_on must name exactly one host label, got {hosts}"
            )
        if not isinstance(creds, bool):
            raise LegsError(f"leg {name!r}: lab_credentials must be true or false")
    return [dict(leg) for leg in legs]


def check_against_inventory(legs: list[dict[str, object]], declared: set[str]) -> None:
    if not declared:
        raise LegsError("the inventory declares no arm64 verify host")
    seen: list[str] = []
    for leg in legs:
        runs_on = leg["runs_on"]
        assert isinstance(runs_on, list)
        seen.extend(label for label in runs_on if HOST_LABEL.fullmatch(label))
    duplicates = sorted({label for label in seen if seen.count(label) > 1})
    if duplicates:
        raise LegsError(f"more than one leg proves {duplicates}")
    missing = sorted(declared - set(seen))
    stale = sorted(set(seen) - declared)
    problems = []
    if missing:
        problems.append(
            f"declared arm64 verify hosts with no leg: {missing} (add a leg per host)"
        )
    if stale:
        problems.append(
            f"legs for hosts no compose file registers: {stale} "
            "(a leg for an absent host waits on a machine that never answers)"
        )
    if problems:
        raise LegsError("; ".join(problems))


def main(environ: Mapping[str, str] | None = None) -> int:
    env = os.environ if environ is None else environ
    try:
        legs = parse_legs(env.get(VARIABLE))
        check_against_inventory(legs, declared_host_labels())
    except LegsError as exc:
        sys.stdout.write(f"::error::{exc}\n")
        return 1
    rendered = json.dumps(legs, separators=(",", ":"))
    sys.stdout.write(f"legs={rendered}\n")
    output = env.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"legs={rendered}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
