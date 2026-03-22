# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Verify running Docker containers against catalog bundle manifests.

Resolves expected containers from catalog bundles and verifies them
against ``docker ps -a``. Used by both integration-sweep (CONTAINER_HEALTH
probe) and redeploy (VERIFY phase).

Exit codes:
    0 — all expected containers running (or recovered via restart-once)
    1 — container manifest mismatch (missing or non-running containers)
    2 — infrastructure error (Docker unavailable, ambiguous manifest)
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess

logger = logging.getLogger(__name__)
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ContainerVerifyResult:
    """Result of container manifest verification."""

    exit_code: int = 0
    failures: list[str] = field(default_factory=list)
    recovered: list[str] = field(default_factory=list)


def resolve_expected_from_catalog(catalog_dir: str, bundles: list[str]) -> list[str]:
    """Resolve expected container names from catalog bundle definitions.

    Args:
        catalog_dir: Path to the catalog directory containing bundles.yaml
            and services/ subdirectory.
        bundles: List of bundle names to resolve.

    Returns:
        Sorted list of expected container names.

    Raises:
        ValueError: If a bundle name is unknown or a service YAML is missing.
    """
    bundles_path = Path(catalog_dir) / "bundles.yaml"
    services_dir = Path(catalog_dir) / "services"

    with open(bundles_path) as f:
        bundle_defs = yaml.safe_load(f)

    # Resolve bundles transitively
    resolved_services: set[str] = set()
    visited_bundles: set[str] = set()

    def _resolve_bundle(name: str) -> None:
        if name in visited_bundles:
            return
        visited_bundles.add(name)

        if name not in bundle_defs:
            msg = f"Unknown bundle: {name!r}. Available: {sorted(bundle_defs.keys())}"
            raise ValueError(msg)

        bundle = bundle_defs[name]
        for svc in bundle.get("services", []):
            resolved_services.add(svc)
        for inc in bundle.get("includes", []):
            _resolve_bundle(inc)

    for b in bundles:
        _resolve_bundle(b)

    # Resolve container names from service YAMLs
    container_names: list[str] = []
    for svc in sorted(resolved_services):
        # Try exact name match first, then with common prefixes
        svc_path = services_dir / f"{svc}.yaml"
        if not svc_path.exists():
            msg = (
                f"Service YAML not found for {svc!r} at {svc_path}. "
                f"Manifest resolution is ambiguous."
            )
            raise ValueError(msg)

        with open(svc_path) as f:
            svc_def = yaml.safe_load(f)

        cname = svc_def.get("container_name")
        if cname:
            container_names.append(cname)

    return sorted(container_names)


def verify_containers(
    expected: list[str],
    restart_once: bool = False,
) -> ContainerVerifyResult:
    """Verify expected containers are running.

    Args:
        expected: List of expected container names.
        restart_once: If True, attempt one restart for non-running containers.

    Returns:
        ContainerVerifyResult with exit_code, failures, and recovered lists.
    """
    result = ContainerVerifyResult()

    # Get current container states
    try:
        ps_output = subprocess.run(
            [
                "docker",
                "ps",
                "-a",
                "--format",
                '{"Names":"{{.Names}}","State":"{{.State}}","Status":"{{.Status}}"}',
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        result.exit_code = 2
        result.failures.append("Docker command not found")
        return result

    if ps_output.returncode != 0:
        result.exit_code = 2
        result.failures.append(f"docker ps failed: {ps_output.stderr.strip()}")
        return result

    # Parse container states
    containers: dict[str, str] = {}
    for line in ps_output.stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            containers[data["Names"]] = data["State"]
        except (json.JSONDecodeError, KeyError):
            continue

    # Check each expected container
    non_running: list[str] = []
    for name in expected:
        if name not in containers:
            result.failures.append(f"{name}: not found in docker ps -a")
            result.exit_code = 1
        elif containers[name] != "running":
            non_running.append(name)

    # Attempt restart-once for non-running containers
    if non_running and restart_once:
        for name in non_running:
            try:
                subprocess.run(
                    ["docker", "restart", name],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except FileNotFoundError:
                logger.debug("Docker CLI not found during restart of %s", name)

        # Recheck after restart
        try:
            ps_recheck = subprocess.run(
                [
                    "docker",
                    "ps",
                    "-a",
                    "--format",
                    '{"Names":"{{.Names}}","State":"{{.State}}","Status":"{{.Status}}"}',
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            result.exit_code = 2
            result.failures.append("Docker command not found during recheck")
            return result

        rechecked: dict[str, str] = {}
        for line in ps_recheck.stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                rechecked[data["Names"]] = data["State"]
            except (json.JSONDecodeError, KeyError):
                continue

        for name in non_running:
            if rechecked.get(name) == "running":
                result.recovered.append(name)
            else:
                state = rechecked.get(name, "unknown")
                result.failures.append(
                    f"{name}: {state} (restart-once attempted, still not running)"
                )
                result.exit_code = 1
    elif non_running:
        for name in non_running:
            state = containers.get(name, "unknown")
            result.failures.append(f"{name}: {state}")
            result.exit_code = 1

    return result


def main() -> None:
    """CLI entry point for container manifest verification."""
    parser = argparse.ArgumentParser(
        description="Verify Docker containers against catalog bundle manifests."
    )
    parser.add_argument(
        "--catalog-dir",
        required=True,
        help="Path to catalog directory containing bundles.yaml and services/",
    )
    parser.add_argument(
        "--bundles",
        required=True,
        help="Comma-separated list of bundle names to verify",
    )
    parser.add_argument(
        "--restart-once",
        action="store_true",
        help="Attempt one restart for non-running containers",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    bundle_list = [b.strip() for b in args.bundles.split(",")]

    try:
        expected = resolve_expected_from_catalog(args.catalog_dir, bundle_list)
    except ValueError as e:
        if args.json_output:
            print(
                json.dumps(
                    {"exit_code": 2, "error": str(e), "failures": [], "recovered": []}
                )
            )
        else:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    result = verify_containers(expected, restart_once=args.restart_once)

    if args.json_output:
        print(
            json.dumps(
                {
                    "exit_code": result.exit_code,
                    "expected": expected,
                    "failures": result.failures,
                    "recovered": result.recovered,
                }
            )
        )
    else:
        if result.failures:
            print("CONTAINER MANIFEST FAILURES:")
            for f in result.failures:
                print(f"  - {f}")
        if result.recovered:
            print("RECOVERED (restart-once):")
            for r in result.recovered:
                print(f"  - {r}")
        if not result.failures and not result.recovered:
            print(f"All {len(expected)} expected containers running.")

    sys.exit(result.exit_code)


if __name__ == "__main__":
    main()
