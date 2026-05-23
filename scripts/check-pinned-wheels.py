#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Detect drift between pinned wheel refs in Dockerfile.runtime / pyproject.toml
and the latest published GitHub releases.

Checks three classes of pin:

1. Dockerfile ARG git-ref pins (OMNIBASE_COMPAT_REF, ONEX_CHANGE_CONTROL_REF,
   OMNIMARKET_REF) — compared against the latest release tag on GitHub.
2. pyproject.toml [tool.uv.sources] git rev/tag pins (omnibase-core,
   omnibase-spi, onex-change-control) — compared against latest release tag.
3. Dockerfile ``uv pip install --no-deps`` PyPI range pins (omninode-claude,
   omninode-memory, omninode-intelligence) — compared against the latest
   published PyPI version (covered by check_dockerfile_pins.py; included here
   for the structured JSON output consumed by merge-sweep integration).

Exit codes:
    0  — no drift detected
    1  — drift detected or hard error
    2  — one or more checks could not run (network / auth failure)

Requires ``gh`` CLI authenticated with read access to OmniNode-ai org.

Usage::

    uv run python scripts/check-pinned-wheels.py
    uv run python scripts/check-pinned-wheels.py --dockerfile docker/Dockerfile.runtime
    uv run python scripts/check-pinned-wheels.py --json          # machine-readable
    uv run python scripts/check-pinned-wheels.py --threshold 1   # warn on any minor behind
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Repo slug → GitHub org/repo mapping
# ---------------------------------------------------------------------------

_GITHUB_REPOS: dict[str, str] = {
    "omnibase_compat": "OmniNode-ai/omnibase_compat",
    "omnibase-compat": "OmniNode-ai/omnibase_compat",
    "onex_change_control": "OmniNode-ai/onex_change_control",
    "onex-change-control": "OmniNode-ai/onex_change_control",
    "omnimarket": "OmniNode-ai/omnimarket",
    "omnibase_core": "OmniNode-ai/omnibase_core",
    "omnibase-core": "OmniNode-ai/omnibase_core",
    "omnibase_spi": "OmniNode-ai/omnibase_spi",
    "omnibase-spi": "OmniNode-ai/omnibase_spi",
    "omninode-claude": "OmniNode-ai/omniclaude",
    "omninode-memory": "OmniNode-ai/omnimemory",
    "omninode-intelligence": "OmniNode-ai/omniintelligence",
}

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PinResult:
    source: str  # "dockerfile_arg" | "pyproject_uv_source" | "pypi_range"
    name: str
    pinned: str  # what the repo currently pins (commit SHA, tag, range)
    latest: str  # latest release tag / version on GitHub/PyPI
    drifted: bool
    minor_behind: int  # 0 if not semver-comparable
    message: str
    check_failed: bool = False  # True if we could not reach the API


@dataclass
class DriftReport:
    dockerfile: str
    pyproject: str
    results: list[PinResult] = field(default_factory=list)

    def has_drift(self) -> bool:
        return any(r.drifted for r in self.results)

    def has_failures(self) -> bool:
        return any(r.check_failed for r in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dockerfile": self.dockerfile,
            "pyproject": self.pyproject,
            "drifted": self.has_drift(),
            "check_failures": self.has_failures(),
            "results": [asdict(r) for r in self.results],
        }


# ---------------------------------------------------------------------------
# GitHub helpers — use gh CLI to avoid rate-limit / auth complexity
# ---------------------------------------------------------------------------


def _gh_latest_release(repo: str) -> str | None:
    """Return the tag name of the latest published (non-draft, non-prerelease) release."""
    try:
        result = subprocess.run(
            ["gh", "api", f"repos/{repo}/releases/latest", "--jq", ".tag_name"],
            capture_output=True,
            text=True,
            check=False,
        )
        tag = result.stdout.strip()
        if result.returncode != 0 or not tag:
            return None
        return tag
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# PyPI helper
# ---------------------------------------------------------------------------


def _pypi_latest(package: str) -> str | None:
    url = f"https://pypi.org/pypi/{package}/json"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
            data = json.load(resp)
        return data["info"]["version"]
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Version arithmetic
# ---------------------------------------------------------------------------


def _parse_version(v: str) -> tuple[int, ...]:
    """Strip leading 'v' and parse dotted ints; returns () on failure."""
    v = v.lstrip("v")
    try:
        return tuple(int(x) for x in v.split(".")[:3])
    except ValueError:
        return ()


def _minor_behind(pinned_tag: str, latest_tag: str) -> int:
    """Return how many minor versions pinned is behind latest (0 if not comparable)."""
    pinned = _parse_version(pinned_tag)
    latest = _parse_version(latest_tag)
    if not pinned or not latest or len(pinned) < 2 or len(latest) < 2:
        return 0
    if pinned[0] != latest[0]:
        # Major version difference — report as 999 to always trigger
        return 999
    return max(0, latest[1] - pinned[1])


# ---------------------------------------------------------------------------
# Dockerfile ARG parsing
# ---------------------------------------------------------------------------

_ARG_RE = re.compile(r'^ARG\s+(\w+)="?([^"\s]+)"?', re.MULTILINE)

# Maps Dockerfile ARG name → (canonical package name, pin type)
_DOCKERFILE_ARG_PINS: dict[str, tuple[str, str]] = {
    "OMNIBASE_COMPAT_REF": ("omnibase_compat", "commit_sha"),
    "ONEX_CHANGE_CONTROL_REF": ("onex_change_control", "branch_or_sha"),
    "OMNIMARKET_REF": ("omnimarket", "branch_or_sha"),
}


def _parse_dockerfile_args(path: Path) -> dict[str, str]:
    """Return {ARG_NAME: value} for all ARG declarations in the Dockerfile."""
    text = path.read_text()
    return {m.group(1): m.group(2) for m in _ARG_RE.finditer(text)}


# ---------------------------------------------------------------------------
# pyproject.toml [tool.uv.sources] parsing (minimal TOML — no stdlib tomllib
# dep needed because we parse just the uv.sources block with regex)
# ---------------------------------------------------------------------------

_UVS_BLOCK_RE = re.compile(
    r"\[tool\.uv\.sources\](.*?)(?=^\[|\Z)",
    re.MULTILINE | re.DOTALL,
)
_UVS_ENTRY_RE = re.compile(
    r"^(\S+)\s*=\s*\{([^}]+)\}",
    re.MULTILINE,
)
_KV_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')


def _parse_uv_sources(path: Path) -> dict[str, dict[str, str]]:
    """Return {package_name: {key: value}} from [tool.uv.sources]."""
    text = path.read_text()
    block_m = _UVS_BLOCK_RE.search(text)
    if not block_m:
        return {}
    block = block_m.group(1)
    sources: dict[str, dict[str, str]] = {}
    for entry_m in _UVS_ENTRY_RE.finditer(block):
        pkg = entry_m.group(1)
        kv_str = entry_m.group(2)
        sources[pkg] = dict(_KV_RE.findall(kv_str))
    return sources


# ---------------------------------------------------------------------------
# Check: Dockerfile ARG pins
# ---------------------------------------------------------------------------


def _check_dockerfile_args(
    args: dict[str, str],
    threshold: int,
) -> list[PinResult]:
    results: list[PinResult] = []

    for arg_name, (pkg, pin_type) in _DOCKERFILE_ARG_PINS.items():
        pinned_val = args.get(arg_name)
        if pinned_val is None:
            continue

        repo_slug = _GITHUB_REPOS.get(pkg)
        if repo_slug is None:
            continue

        latest_release = _gh_latest_release(repo_slug)

        if pinned_val == "main":
            # "main" is an unpinned floating ref — always drifted by definition.
            # Distinguish: if there's a latest release, this is a real drift gap.
            if latest_release:
                results.append(
                    PinResult(
                        source="dockerfile_arg",
                        name=arg_name,
                        pinned="main",
                        latest=latest_release,
                        drifted=True,
                        minor_behind=0,
                        message=(
                            f"{arg_name} is pinned to 'main' (floating ref). "
                            f"Latest release is {latest_release}. "
                            f"Pin to a specific commit SHA to eliminate silent drift."
                        ),
                    )
                )
            else:
                results.append(
                    PinResult(
                        source="dockerfile_arg",
                        name=arg_name,
                        pinned="main",
                        latest="unknown",
                        drifted=True,
                        minor_behind=0,
                        message=(
                            f"{arg_name} is pinned to 'main' (floating ref). "
                            "Could not fetch latest release from GitHub."
                        ),
                        check_failed=True,
                    )
                )
            continue

        # Pinned to a commit SHA — check whether HEAD has advanced beyond it
        if latest_release is None:
            results.append(
                PinResult(
                    source="dockerfile_arg",
                    name=arg_name,
                    pinned=pinned_val,
                    latest="unknown",
                    drifted=False,
                    minor_behind=0,
                    message=f"Could not fetch latest release for {repo_slug}.",
                    check_failed=True,
                )
            )
            continue

        behind = _minor_behind(pinned_val, latest_release)
        # For commit SHAs: pinned_val is a SHA, not a semver tag.
        # We compare to the latest release tag instead.
        # If a release exists and the pinned value is a SHA (not a tag), always
        # flag — a newer tagged release means we're behind by definition.
        is_sha = re.fullmatch(r"[0-9a-f]{7,40}", pinned_val) is not None
        drifted = is_sha and latest_release != pinned_val

        results.append(
            PinResult(
                source="dockerfile_arg",
                name=arg_name,
                pinned=pinned_val,
                latest=latest_release,
                drifted=drifted,
                minor_behind=behind,
                message=(
                    f"{arg_name} pinned to commit {pinned_val[:12]}; "
                    f"latest release is {latest_release}."
                )
                if drifted
                else f"{arg_name} is current ({pinned_val[:12]!r} / latest {latest_release}).",
            )
        )

    return results


# ---------------------------------------------------------------------------
# Check: pyproject.toml [tool.uv.sources] pins
# ---------------------------------------------------------------------------


def _check_uv_sources(
    sources: dict[str, dict[str, str]],
    threshold: int,
) -> list[PinResult]:
    results: list[PinResult] = []

    for pkg_name, attrs in sources.items():
        repo_slug = _GITHUB_REPOS.get(pkg_name)
        if repo_slug is None:
            continue

        rev = attrs.get("rev")
        tag = attrs.get("tag")
        branch = attrs.get("branch")

        if branch and not rev and not tag:
            # Floating branch ref
            latest_release = _gh_latest_release(repo_slug)
            results.append(
                PinResult(
                    source="pyproject_uv_source",
                    name=pkg_name,
                    pinned=f"branch={branch}",
                    latest=latest_release or "unknown",
                    drifted=True,
                    minor_behind=0,
                    message=(
                        f"{pkg_name} tracks branch '{branch}' (floating). "
                        f"Latest release: {latest_release or 'unknown'}. "
                        "Pin to a commit SHA for reproducibility."
                    ),
                    check_failed=latest_release is None,
                )
            )
            continue

        if tag:
            latest_release = _gh_latest_release(repo_slug)
            if latest_release is None:
                results.append(
                    PinResult(
                        source="pyproject_uv_source",
                        name=pkg_name,
                        pinned=tag,
                        latest="unknown",
                        drifted=False,
                        minor_behind=0,
                        message=f"Could not fetch latest release for {repo_slug}.",
                        check_failed=True,
                    )
                )
                continue

            behind = _minor_behind(tag, latest_release)
            drifted = behind > threshold
            results.append(
                PinResult(
                    source="pyproject_uv_source",
                    name=pkg_name,
                    pinned=tag,
                    latest=latest_release,
                    drifted=drifted,
                    minor_behind=behind,
                    message=(
                        f"{pkg_name} pinned to {tag}; latest is {latest_release} "
                        f"({behind} minor version(s) behind)."
                    )
                    if drifted
                    else f"{pkg_name} tag {tag!r} is current (latest {latest_release}).",
                )
            )
            continue

        if rev:
            latest_release = _gh_latest_release(repo_slug)
            if latest_release is None:
                results.append(
                    PinResult(
                        source="pyproject_uv_source",
                        name=pkg_name,
                        pinned=rev,
                        latest="unknown",
                        drifted=False,
                        minor_behind=0,
                        message=f"Could not fetch latest release for {repo_slug}.",
                        check_failed=True,
                    )
                )
                continue

            # Commit SHA vs latest release tag: flag as drifted since a release exists
            results.append(
                PinResult(
                    source="pyproject_uv_source",
                    name=pkg_name,
                    pinned=rev,
                    latest=latest_release,
                    drifted=True,
                    minor_behind=0,
                    message=(
                        f"{pkg_name} pinned to pre-release commit {rev[:12]}; "
                        f"latest tagged release is {latest_release}. "
                        "Consider upgrading to the tagged release."
                    ),
                )
            )

    return results


# ---------------------------------------------------------------------------
# Check: Dockerfile --no-deps PyPI range pins
# ---------------------------------------------------------------------------

_NODEPS_INSTALL_RE = re.compile(
    r"uv pip install --no-deps\s+((?:\"[^\"]+\"\s*)+)",
    re.MULTILINE,
)
_PKG_SPEC_RE = re.compile(r'"([A-Za-z0-9_\-]+)([>=<!,.\d*]+)"')

_ONEX_PYPI_PACKAGES = {
    "omninode-claude",
    "omninode-memory",
    "omninode-intelligence",
}


def _check_pypi_range_pins(
    dockerfile_path: Path,
    threshold: int,
) -> list[PinResult]:
    # Normalize backslash-newline continuations so multi-line RUN blocks match
    text = dockerfile_path.read_text().replace("\\\n", " ")
    results: list[PinResult] = []

    for block_m in _NODEPS_INSTALL_RE.finditer(text):
        block = block_m.group(1)
        for pkg_m in _PKG_SPEC_RE.finditer(block):
            pkg = pkg_m.group(1)
            if pkg not in _ONEX_PYPI_PACKAGES:
                continue
            specifier = pkg_m.group(2)

            latest = _pypi_latest(pkg)
            if latest is None:
                results.append(
                    PinResult(
                        source="pypi_range",
                        name=pkg,
                        pinned=specifier,
                        latest="unknown",
                        drifted=False,
                        minor_behind=0,
                        message=f"Could not fetch PyPI data for {pkg}.",
                        check_failed=True,
                    )
                )
                continue

            # Extract lower bound from specifier for "how far behind" calc
            lower_m = re.search(r">=([0-9.]+)", specifier)
            lower_tag = f"v{lower_m.group(1)}" if lower_m else "v0.0.0"
            behind = _minor_behind(lower_tag, f"v{latest}")
            drifted = behind > threshold

            results.append(
                PinResult(
                    source="pypi_range",
                    name=pkg,
                    pinned=specifier,
                    latest=latest,
                    drifted=drifted,
                    minor_behind=behind,
                    message=(
                        f"{pkg}{specifier} lower bound is {behind} minor "
                        f"version(s) behind PyPI latest ({latest})."
                    )
                    if drifted
                    else f"{pkg}{specifier} covers latest PyPI release ({latest}).",
                )
            )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_report(
    dockerfile_path: Path,
    pyproject_path: Path,
    threshold: int,
) -> DriftReport:
    report = DriftReport(
        dockerfile=str(dockerfile_path),
        pyproject=str(pyproject_path),
    )

    # 1. Dockerfile ARG pins
    df_args = _parse_dockerfile_args(dockerfile_path)
    report.results.extend(_check_dockerfile_args(df_args, threshold))

    # 2. pyproject.toml [tool.uv.sources] pins
    uv_sources = _parse_uv_sources(pyproject_path)
    report.results.extend(_check_uv_sources(uv_sources, threshold))

    # 3. Dockerfile --no-deps PyPI range pins (ONEX packages only)
    report.results.extend(_check_pypi_range_pins(dockerfile_path, threshold))

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dockerfile",
        default="docker/Dockerfile.runtime",
        help="Path to Dockerfile.runtime (default: docker/Dockerfile.runtime)",
    )
    parser.add_argument(
        "--pyproject",
        default="pyproject.toml",
        help="Path to pyproject.toml (default: pyproject.toml)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Report drift only when a pin is more than N minor versions behind "
            "(default: 1 — flag anything >1 minor behind)"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Emit a structured JSON report to stdout",
    )
    args = parser.parse_args(argv)

    dockerfile_path = Path(args.dockerfile)
    pyproject_path = Path(args.pyproject)

    if not dockerfile_path.exists():
        print(f"ERROR: Dockerfile not found: {dockerfile_path}", file=sys.stderr)
        return 1
    if not pyproject_path.exists():
        print(f"ERROR: pyproject.toml not found: {pyproject_path}", file=sys.stderr)
        return 1

    report = _build_report(dockerfile_path, pyproject_path, args.threshold)

    if args.output_json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        for r in report.results:
            prefix = "DRIFT" if r.drifted else ("FAIL" if r.check_failed else "OK  ")
            print(f"[{prefix}] {r.message}")

    if report.has_failures():
        if not args.output_json:
            print(
                "\nWARNING: some checks could not reach GitHub/PyPI — results may be incomplete.",
                file=sys.stderr,
            )
        if not report.has_drift():
            return 2

    if report.has_drift():
        if not args.output_json:
            print(
                "\nDrift detected. Update the pinned refs in Dockerfile.runtime / pyproject.toml.",
                file=sys.stderr,
            )
        return 1

    if not args.output_json:
        print(f"\nAll {len(report.results)} pin checks passed — no drift detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
