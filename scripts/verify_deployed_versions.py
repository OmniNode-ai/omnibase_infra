#!/usr/bin/env -S uv run python
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Post-deploy readback for running runtime containers [OMN-5608 / RT-6 OMN-14469].

The deploy readback (RT-6, `docs/plans/2026-07-12-mechanical-release-trains.md` §4)
reads a fact only the freshly-built image could carry off the RUNNING container and
asserts it equals the intended ref. Two facts are checked, either or both:

1. **Image revision** (``--expected-revision``): the ``org.opencontainers.image.revision``
   label stamped from ``VCS_REF`` at build time (Dockerfile.runtime, OMN-12965/13412)
   must equal the intended git SHA. This is the load-bearing fast-lane fact — it
   distinguishes every build even when the package version is unchanged, so a
   **stale / mis-targeted container (deployed code != intended ref) goes RED**
   instead of silently passing.
2. **Package versions** (``--versions``): the versions installed inside the running
   container (``uv pip show``) must equal the expected release versions — the
   Train-2 (release) fact.

Called as the deploy's TERMINAL step by ``scripts/deploy-runtime.sh``
(``readback_deployed_ref``), not an optional flag.

Usage:
    # Fast-lane readback: assert the running container is the intended git SHA
    uv run python scripts/verify_deployed_versions.py \
        --container omninode-runtime \
        --expected-revision "$(git rev-parse --short=12 HEAD)"

    # Release-train readback: assert installed package versions
    uv run python scripts/verify_deployed_versions.py \
        --versions "omniintelligence=0.16.0,omninode-claude=0.4.0"

    # Both facts at once (the deploy terminal step)
    uv run python scripts/verify_deployed_versions.py \
        --container omninode-runtime \
        --expected-revision "<git_sha>" \
        --versions "omnibase-infra=0.38.4"

    # JSON output for machine consumption
    uv run python scripts/verify_deployed_versions.py \
        --versions "omniintelligence=0.16.0" --json

At least one of ``--versions`` / ``--expected-revision`` is required.

Exit codes:
    0 - All requested readback checks matched expectations
    1 - One or more mismatches detected (revision or version) -- deployed != intended
    2 - Could not connect to container or run the readback (infra error / bad args)
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


# ─── Image-revision readback (RT-6 OMN-14469) ─────────────────────────────

# OCI label stamped from VCS_REF/GIT_SHA at build time (Dockerfile.runtime).
_REVISION_LABEL = "org.opencontainers.image.revision"


@dataclass
class RevisionCheckResult:
    """Result of the image-revision readback for a running container."""

    container: str
    expected: str
    actual: str | None
    match: bool
    error: str | None = None


def get_image_revision(
    container: str,
    *,
    runner: object | None = None,
) -> tuple[str | None, str | None]:
    """Read the ``org.opencontainers.image.revision`` label off *container*.

    The label is stamped from ``VCS_REF`` at build time, so it identifies the
    exact source commit the running image was built from -- a fact only the
    intended code could carry. A stale container carries the OLD revision.

    Args:
        container: Docker container name or ID (the running lane runtime).
        runner: Optional subprocess runner for testing.

    Returns:
        ``(revision_string, None)`` on success, or ``(None, error_message)`` on
        failure (container down, missing label, docker unavailable).
    """
    cmd = [
        "docker",
        "inspect",
        container,
        "--format",
        f'{{{{index .Config.Labels "{_REVISION_LABEL}"}}}}',
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
        return None, f"Timed out inspecting {container}"
    except FileNotFoundError:
        return None, "docker command not found"

    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else ""
        return None, f"docker inspect failed (exit {result.returncode}): {stderr}"

    revision = (result.stdout or "").strip()
    # Go templates print "<no value>" (or empty) when the label key is absent.
    if not revision or revision == "<no value>":
        return None, (
            f"no {_REVISION_LABEL} label on {container} "
            "(image identity missing -- cannot prove the running code)"
        )
    return revision, None


def _revisions_match(actual: str, expected: str) -> bool:
    """Compare two git revisions tolerating short/full SHA length differences.

    The build stamps a ``--short=12`` SHA while a caller may pass a full 40-char
    SHA (or vice versa), so a strict equality would false-negative. We accept a
    prefix relationship on lowercased hex; a genuinely stale container carries an
    entirely different SHA (not a prefix), so this stays fail-closed against the
    exists-but-WRONG case.
    """
    a = actual.strip().lower()
    b = expected.strip().lower()
    if not a or not b:
        return False
    return a == b or a.startswith(b) or b.startswith(a)


def verify_revision(
    container: str,
    expected_revision: str,
    *,
    runner: object | None = None,
) -> RevisionCheckResult:
    """Assert *container*'s image revision label equals *expected_revision*.

    Fail-closed: a missing/unreadable label or a mismatched revision both yield
    ``match=False`` (the caller maps an ``error`` to an infra exit and a plain
    mismatch to a deployed!=intended exit).
    """
    actual, error = get_image_revision(container, runner=runner)
    if error is not None:
        return RevisionCheckResult(
            container=container,
            expected=expected_revision,
            actual=None,
            match=False,
            error=error,
        )
    return RevisionCheckResult(
        container=container,
        expected=expected_revision,
        actual=actual,
        match=_revisions_match(actual or "", expected_revision),
    )


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


def format_revision_text(result: RevisionCheckResult) -> str:
    """Format a human-readable image-revision readback line."""
    if result.match:
        return (
            f"Image revision readback for {result.container}:\n"
            f"  OK   revision {result.actual} matches intended {result.expected}"
        )
    if result.error:
        return (
            f"Image revision readback for {result.container}:\n"
            f"  FAIL ERROR - {result.error} (intended {result.expected})"
        )
    return (
        f"Image revision readback for {result.container}:\n"
        f"  FAIL revision {result.actual} != intended {result.expected} "
        "-- deployed code is NOT the intended ref (stale/mis-targeted image)"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns exit code (0=ok, 1=mismatch, 2=infra error)."""
    parser = argparse.ArgumentParser(
        description=(
            "Post-deploy readback: assert a running container's image revision "
            "and/or installed package versions match the intended ref."
        ),
    )
    parser.add_argument(
        "--versions",
        default=None,
        help='Comma-separated pkg=version pairs, e.g. "omniintelligence=0.16.0,omninode-claude=0.4.0"',
    )
    parser.add_argument(
        "--expected-revision",
        dest="expected_revision",
        default=None,
        help=(
            "Intended git SHA. Asserts the running container's "
            "org.opencontainers.image.revision label equals it (RT-6 readback)."
        ),
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

    # At least one readback dimension must be requested. A no-arg invocation that
    # asserted nothing would be a green producer that produces no proof -- exactly
    # the silent-pass failure RT-6 exists to kill.
    if args.versions is None and args.expected_revision is None:
        print(
            "ERROR: nothing to read back -- pass --versions and/or --expected-revision",
            file=sys.stderr,
        )
        return 2

    version_report: VerificationReport | None = None
    if args.versions is not None:
        try:
            expected = parse_versions_string(args.versions)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        if not expected:
            print("ERROR: --versions was empty", file=sys.stderr)
            return 2
        version_report = verify_versions(args.container, expected)

    revision_result: RevisionCheckResult | None = None
    if args.expected_revision is not None:
        revision_result = verify_revision(args.container, args.expected_revision)

    # ── Output ──────────────────────────────────────────────────────────────
    if args.json_output:
        payload: dict[str, object] = {}
        if version_report is not None:
            payload["versions"] = version_report.to_dict()
        if revision_result is not None:
            payload["revision"] = {
                "container": revision_result.container,
                "expected": revision_result.expected,
                "actual": revision_result.actual,
                "match": revision_result.match,
                "error": revision_result.error,
            }
        print(json.dumps(payload, indent=2))
    else:
        if version_report is not None:
            print(format_report_text(version_report))
        if revision_result is not None:
            print(format_revision_text(revision_result))

    # ── Exit code: a genuine mismatch (1) outranks an infra error (2) ────────
    version_mismatch = version_report is not None and any(
        r.error is None and not r.match for r in version_report.results
    )
    version_infra_error = version_report is not None and (
        # all_match is False AND every failure is an error (nothing genuinely
        # mismatched) -> couldn't check rather than a real drift.
        not version_report.all_match and not version_mismatch
    )
    revision_mismatch = (
        revision_result is not None
        and not revision_result.match
        and revision_result.error is None
    )
    revision_infra_error = (
        revision_result is not None and revision_result.error is not None
    )

    if version_mismatch or revision_mismatch:
        return 1
    if version_infra_error or revision_infra_error:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
