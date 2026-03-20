#!/usr/bin/env -S uv run python
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Post-release version verification for deployed runtime containers [OMN-5608].

Verifies that the package versions installed inside running Docker containers
match the expected release versions.  Designed to be called by the redeploy
skill's VERIFY phase immediately after container restart.

Usage:
    # Verify specific versions (from redeploy --versions flag)
    uv run python scripts/verify_deployed_versions.py \
        --versions "omniintelligence=0.16.0,omninode-claude=0.4.0"

    # Verify with explicit container name
    uv run python scripts/verify_deployed_versions.py \
        --versions "omniintelligence=0.16.0" \
        --container omninode-runtime

    # JSON output for machine consumption
    uv run python scripts/verify_deployed_versions.py \
        --versions "omniintelligence=0.16.0" --json

Exit codes:
    0 - All package versions match expectations
    1 - One or more version mismatches detected
    2 - Could not connect to container or run version check
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field

# ─── Data structures ───────────────────────────────────────────────────────


@dataclass
class VersionCheckResult:
    """Result of a single package version check."""

    package: str
    expected: str
    actual: str | None
    match: bool
    error: str | None = None


@dataclass
class VerificationReport:
    """Aggregate report of all version checks."""

    container: str
    results: list[VersionCheckResult] = field(default_factory=list)

    @property
    def all_match(self) -> bool:
        """Return True only if every check passed with a matching version."""
        return bool(self.results) and all(r.match for r in self.results)

    @property
    def mismatches(self) -> list[VersionCheckResult]:
        """Return only the results that failed."""
        return [r for r in self.results if not r.match]

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-friendly dict."""
        return {
            "container": self.container,
            "all_match": self.all_match,
            "results": [
                {
                    "package": r.package,
                    "expected": r.expected,
                    "actual": r.actual,
                    "match": r.match,
                    "error": r.error,
                }
                for r in self.results
            ],
        }


# ─── Core verification logic ──────────────────────────────────────────────


def get_installed_version(
    container: str,
    package: str,
    *,
    runner: object | None = None,
) -> tuple[str | None, str | None]:
    """Query the installed version of *package* inside *container*.

    Args:
        container: Docker container name or ID.
        package: Python package name (pip-style, e.g. ``omniintelligence``).
        runner: Optional callable ``(cmd: list[str]) -> subprocess.CompletedProcess``
            for testing.  Defaults to ``subprocess.run``.

    Returns:
        ``(version_string, None)`` on success, or ``(None, error_message)`` on
        failure.
    """
    cmd = [
        "docker",
        "exec",
        container,
        "uv",
        "pip",
        "show",
        package,
    ]
    run_fn = runner or subprocess.run
    try:
        result = run_fn(  # type: ignore[operator]
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return None, f"Timed out querying {package} in {container}"
    except FileNotFoundError:
        return None, "docker command not found"

    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else ""
        return None, f"docker exec failed (exit {result.returncode}): {stderr}"

    # Parse "Version: X.Y.Z" from pip show output
    for line in result.stdout.splitlines():
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip(), None

    return None, f"No 'Version:' line in output for {package}"


def verify_versions(
    container: str,
    expected_versions: dict[str, str],
    *,
    runner: object | None = None,
) -> VerificationReport:
    """Verify that all *expected_versions* match what is deployed in *container*.

    Args:
        container: Docker container name or ID.
        expected_versions: Mapping of ``{package_name: expected_version}``.
        runner: Optional subprocess runner for testing.

    Returns:
        A :class:`VerificationReport` with per-package results.
    """
    report = VerificationReport(container=container)

    for package, expected in sorted(expected_versions.items()):
        actual, error = get_installed_version(container, package, runner=runner)

        if error is not None:
            report.results.append(
                VersionCheckResult(
                    package=package,
                    expected=expected,
                    actual=None,
                    match=False,
                    error=error,
                )
            )
        else:
            match = actual == expected
            report.results.append(
                VersionCheckResult(
                    package=package,
                    expected=expected,
                    actual=actual,
                    match=match,
                )
            )

    return report


# ─── CLI ───────────────────────────────────────────────────────────────────


def parse_versions_string(raw: str) -> dict[str, str]:
    """Parse ``"pkg1=1.0.0,pkg2=2.0.0"`` into a dict.

    Raises:
        ValueError: If any token does not contain exactly one ``=``.
    """
    versions: dict[str, str] = {}
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            msg = f"Invalid version token (expected pkg=version): {token!r}"
            raise ValueError(msg)
        name, ver = token.split("=", 1)
        versions[name.strip()] = ver.strip()
    return versions


def format_report_text(report: VerificationReport) -> str:
    """Format a human-readable verification report."""
    lines: list[str] = []
    lines.append(f"Version verification for container: {report.container}")
    lines.append("")

    for r in report.results:
        if r.match:
            lines.append(f"  OK   {r.package}: {r.actual} (expected {r.expected})")
        elif r.error:
            lines.append(
                f"  FAIL {r.package}: ERROR - {r.error} (expected {r.expected})"
            )
        else:
            lines.append(f"  FAIL {r.package}: {r.actual} (expected {r.expected})")

    lines.append("")
    if report.all_match:
        lines.append("Result: ALL VERSIONS MATCH")
    else:
        mismatch_count = len(report.mismatches)
        lines.append(
            f"Result: {mismatch_count} MISMATCH(ES) DETECTED -- "
            f"deployed versions do not match expected release"
        )

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns exit code (0=ok, 1=mismatch, 2=infra error)."""
    parser = argparse.ArgumentParser(
        description="Verify deployed package versions in Docker containers.",
    )
    parser.add_argument(
        "--versions",
        required=True,
        help='Comma-separated pkg=version pairs, e.g. "omniintelligence=0.16.0,omninode-claude=0.4.0"',
    )
    parser.add_argument(
        "--container",
        default="omninode-runtime",
        help="Docker container name (default: omninode-runtime)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON",
    )

    args = parser.parse_args(argv)

    try:
        expected = parse_versions_string(args.versions)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if not expected:
        print("ERROR: No versions specified", file=sys.stderr)
        return 2

    report = verify_versions(args.container, expected)

    if args.json_output:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report_text(report))

    if report.all_match:
        return 0

    # Distinguish between infra errors (couldn't check) and actual mismatches
    has_infra_error = any(r.error is not None for r in report.mismatches)
    has_version_mismatch = any(r.error is None and not r.match for r in report.results)

    if has_version_mismatch:
        return 1
    if has_infra_error:
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
