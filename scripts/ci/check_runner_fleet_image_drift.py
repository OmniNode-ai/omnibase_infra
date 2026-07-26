# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Detect self-hosted runner containers running a stale image (OMN-15104).

OMN-12567/OMN-13946 bind the runner image identity in
``docker/runners/runner-image.lock.json`` and verify it at build time. Nothing
previously verified that the *live* fleet ever converges to a bound identity
that has landed on ``main``: OMN-13946 (add ``libatomic1`` to the image,
bumping ``image_version`` 5 -> 6) merged 2026-07-09 and was marked Done, but
every one of the 64 running ``omninode-runner-*`` containers was still on
``image_version: 5`` sixteen days later (2026-07-25), deterministically
failing every Pyright job fleet-wide. ``runner-monitor.sh`` only checks
Docker-healthy + GitHub-registration status, so the drift was silent.

This script closes that gap: it compares the ``image_version`` baked into
each running container's ``/etc/omni/runner-image.lock.json`` (written by the
Dockerfile at ``docker/runners/Dockerfile``, see ``COPY
runner-image.lock.json /etc/omni/runner-image.lock.json``) against the
``image_version`` in the *caller-supplied* expected lock (normally the
checked-in ``docker/runners/runner-image.lock.json`` on ``main``) and reports
every container running a version older than expected.

Deliberately NOT threaded into ``runner-monitor.sh``'s existing state machine
(alert-transition tracking, auto-bounce mutex, Slack dedup) — that script is
already complex and cron-critical against a live 64-runner fleet; adding a new
failure class to its alerting logic is a bigger, riskier change than the
defect this ticket exists to fix. This runs standalone, on its own cron
cadence, alongside it.

Modes:
* ``report`` (default) — print a table of stale/unknown containers; exit 1 if
  any are found, else 0. Suitable for a cron entry piping to a Slack webhook.
* ``discover`` — print the live docker-exec-derived per-container versions as
  JSON (debugging aid).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

DEFAULT_LOCK_FILE = Path("docker/runners/runner-image.lock.json")
CONTAINER_NAME_FILTER = "name=omninode-runner-"
BAKED_LOCK_PATH = "/etc/omni/runner-image.lock.json"


class ModelDriftFinding:
    """One stale-or-unknown container finding."""

    def __init__(
        self, container: str, expected_version: int, observed_version: int | None
    ) -> None:
        self.container = container
        self.expected_version = expected_version
        self.observed_version = observed_version

    def as_line(self) -> str:
        observed = (
            "UNKNOWN (baked lock unreadable)"
            if self.observed_version is None
            else str(self.observed_version)
        )
        return (
            f"{self.container}: running image_version={observed}, "
            f"expected >= {self.expected_version}"
        )


def find_stale_containers(
    expected_image_version: int,
    observed_versions: dict[str, int | None],
) -> list[ModelDriftFinding]:
    """Return findings for every container running a stale or unreadable image.

    A container is stale if its baked ``image_version`` is strictly less than
    ``expected_image_version``. A container whose baked lock could not be
    read (``None``) is reported too — fail-closed, not silently skipped, per
    the "optional input means the check does not exist" lesson: a container
    we cannot verify is treated as unverified, not as passing.
    """
    findings: list[ModelDriftFinding] = []
    for container in sorted(observed_versions):
        version = observed_versions[container]
        if version is None or version < expected_image_version:
            findings.append(
                ModelDriftFinding(container, expected_image_version, version)
            )
    return findings


def _load_expected_version(lock_path: Path) -> int:
    data = json.loads(lock_path.read_text(encoding="utf-8"))
    version = data["image_version"]
    if not isinstance(version, int):
        raise TypeError(
            f"image_version in {lock_path} must be an int, got {type(version)!r}"
        )
    return version


def _discover_container_names() -> list[str]:
    result = subprocess.run(
        ["docker", "ps", "--filter", CONTAINER_NAME_FILTER, "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return sorted(name for name in result.stdout.splitlines() if name.strip())


def _read_baked_version(container: str) -> int | None:
    """Return the ``image_version`` baked into ``container``, or ``None``.

    ``None`` covers every failure mode uniformly: the container is gone, the
    exec fails, the file is missing, or the JSON is malformed. All of these
    mean "cannot verify this container is fixed" and must be reported, not
    swallowed.
    """
    try:
        result = subprocess.run(
            ["docker", "exec", container, "cat", BAKED_LOCK_PATH],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    try:
        data = json.loads(result.stdout)
        version = data["image_version"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    return version if isinstance(version, int) else None


def discover_observed_versions(container_names: list[str]) -> dict[str, int | None]:
    """Return ``{container_name: baked image_version | None}`` for each name."""
    return {name: _read_baked_version(name) for name in container_names}


def render_report(
    expected_image_version: int, findings: list[ModelDriftFinding]
) -> str:
    if not findings:
        return (
            f"runner fleet image drift: OK — all containers running "
            f"image_version >= {expected_image_version}"
        )
    lines = [
        f"runner fleet image drift: {len(findings)} container(s) stale "
        f"(expected image_version >= {expected_image_version}):"
    ]
    lines.extend(f"  - {finding.as_line()}" for finding in findings)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Detect self-hosted runner containers running a stale image (OMN-15104)."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK_FILE)
    parser.add_argument("--mode", choices=("report", "discover"), default="report")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    lock_path = (
        args.lock_file if args.lock_file.is_absolute() else repo_root / args.lock_file
    )
    expected_version = _load_expected_version(lock_path)
    container_names = _discover_container_names()
    observed = discover_observed_versions(container_names)

    if args.mode == "discover":
        print(json.dumps(observed, sort_keys=True, indent=2))
        return 0

    findings = find_stale_containers(expected_version, observed)
    print(render_report(expected_version, findings))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
