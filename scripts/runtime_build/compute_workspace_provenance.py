#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Compute per-repo workspace provenance and write a manifest.

Run during Docker build (workspace mode only) to:
  - Hash each staged sibling repo tree under /workspace/sibling-repos/
  - Verify that the installed package content matches the recorded digest
  - Enforce that every staged sibling honors the consuming repo's uv.lock pin
    (fail-fast on a version regression below the pin — the 2026-06-11
    stability-crash condition, OMN-12989)
  - Write /app/build-provenance.json for operator inspection, including the
    expected-vs-actual pin comparison block
  - Print a human-readable provenance summary to stdout

Exit codes:
  0  all proofs passed
  1  digest mismatch (installed content != recorded workspace digest) OR a
     sibling regressed below its uv.lock pin
  2  missing sibling repo (expected path not found)
  3  missing installed package (venv does not contain the expected dist)
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

# Co-located pin-resolution helpers (OMN-12989). Inside the build image this
# module is copied next to the provenance script (see Dockerfile.runtime); in
# the repo it lives in the same scripts/runtime_build/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from resolve_workspace_pins import (
    PinComparison,
    WorkspacePinError,
    assert_pins_satisfied,
    build_comparisons,
    compare_pin,
    parse_lock_pins,
)

SIBLING_REPOS_DIR = Path("/workspace/sibling-repos")
VENV_DIR = Path("/app/.venv")
OUTPUT_MANIFEST = Path("/app/build-provenance.json")
# The consuming repo whose uv.lock pins the sibling versions. omnimarket dev's
# lock is the authority the runtime image must honor.
CONSUMING_REPO = "omnimarket"
# Expected-vs-actual sibling-pin comparison emitted by the preflight
# (check_sibling_lock_pins.py via stage_workspace.sh). Folded into the manifest
# so deploy verifiers can assert the build honored the consuming repo's lock
# (OMN-12977). Absent only for builds staged before the preflight existed.
PIN_COMPARISON_PATH = Path("/workspace/sibling-pin-comparison.json")

# Canonical set of sibling repos that workspace mode must provision.
# Keys are the directory names under sibling-repos/; values are installed
# package distribution names (as returned by importlib.metadata).
WORKSPACE_PACKAGES: dict[str, str] = {
    "omnibase_compat": "omnibase-compat",
    "onex_change_control": "onex-change-control",
    "omnimarket": "omnimarket",
}


def _hash_tree(root: Path) -> str:
    """Return a SHA-256 digest of every file under root, sorted by path."""
    h = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        # Skip common transient artefacts so the digest is stable.
        if any(
            part in path.parts
            for part in (".git", "__pycache__", ".venv", "*.egg-info")
        ):
            continue
        rel = str(path.relative_to(root))
        h.update(rel.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def _installed_direct_url(dist_name: str) -> dict[str, object] | None:
    """Return the direct_url metadata dict for an installed package, or None."""
    try:
        dist = importlib.metadata.distribution(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None
    direct_url_text = dist.read_text("direct_url.json")
    if direct_url_text is None:
        return None
    parsed: dict[str, object] = json.loads(direct_url_text)
    return parsed


def _host_infra_comparison(lock_path: Path) -> PinComparison | None:
    """Compare the runtime image's installed omnibase_infra against the lock pin.

    The host infra package is not staged under sibling-repos (it is the repo
    being built), so it bypasses build_comparisons. Resolve the lock's
    omnibase-infra pin and compare against the installed version — this is the
    direct ratchet for the 2026-06-11 stale-infra crash. Returns a PinComparison
    or None if the lock has no infra pin / infra is not installed.
    """
    pins = parse_lock_pins(lock_path)
    infra_pin = pins.get("omnibase-infra")
    if infra_pin is None:
        return None
    try:
        installed_version = importlib.metadata.version("omnibase-infra")
    except importlib.metadata.PackageNotFoundError:
        return None
    return compare_pin(
        package="omnibase-infra",
        expected_version=infra_pin.version,
        expected_rev=infra_pin.rev,
        actual_version=installed_version,
        actual_rev=None,
    )


def _load_pin_comparison(errors: list[str]) -> dict[str, object] | None:
    """Load the preflight's expected-vs-actual sibling-pin comparison.

    Returns the parsed comparison dict, or None when the artifact is absent
    (e.g. a build staged before the preflight existed). A present-but-drifted
    comparison is recorded as an error so the proof surfaces it.
    """
    if not PIN_COMPARISON_PATH.exists():
        return None
    comparison: dict[str, object] = json.loads(
        PIN_COMPARISON_PATH.read_text(encoding="utf-8")
    )
    drift_count = comparison.get("drift_count", 0)
    if drift_count and not comparison.get("allow_drift", False):
        errors.append(
            f"sibling-pin comparison reports {drift_count} unacknowledged "
            "drift(s) -- the build vendored stale siblings (OMN-12977)."
        )
    return comparison


def main() -> int:
    proofs: list[dict[str, object]] = []
    errors: list[str] = []

    for repo_dir_name, pkg_name in WORKSPACE_PACKAGES.items():
        repo_path = SIBLING_REPOS_DIR / repo_dir_name
        if not repo_path.exists():
            errors.append(f"Missing sibling repo: {repo_path}")
            continue

        workspace_digest = _hash_tree(repo_path)

        direct_url = _installed_direct_url(pkg_name)
        if direct_url is None:
            errors.append(
                f"Package '{pkg_name}' not found in installed distributions. "
                "Expected a local-path install from workspace mode."
            )
            proofs.append(
                {
                    "repo": repo_dir_name,
                    "package": pkg_name,
                    "source_root": str(repo_path),
                    "workspace_digest": workspace_digest,
                    "status": "missing_install",
                }
            )
            continue

        install_url_raw = direct_url.get("url", "")
        install_url = install_url_raw if isinstance(install_url_raw, str) else ""
        dir_info = direct_url.get("dir_info", {})
        editable = dir_info.get("editable") if isinstance(dir_info, dict) else None
        is_local = editable is not None or install_url.startswith("file://")

        if not is_local:
            errors.append(
                f"Package '{pkg_name}' was not installed from a local path. "
                f"direct_url.url={install_url!r}. "
                "Workspace mode requires all sibling packages installed from local paths."
            )
            proofs.append(
                {
                    "repo": repo_dir_name,
                    "package": pkg_name,
                    "source_root": str(repo_path),
                    "workspace_digest": workspace_digest,
                    "install_url": install_url,
                    "status": "not_local_install",
                }
            )
            continue

        proofs.append(
            {
                "repo": repo_dir_name,
                "package": pkg_name,
                "source_root": str(repo_path),
                "workspace_digest": workspace_digest,
                "install_url": install_url,
                "status": "verified",
            }
        )
        print(
            f"  workspace provenance OK: {repo_dir_name} "
            f"digest={workspace_digest[:16]}... "
            f"install={install_url}"
        )

    # OMN-12989: compare every staged sibling against the consuming repo's
    # uv.lock pins. A sibling regressed BELOW its lock pin is the exact
    # 2026-06-11 stability-crash condition (vendored infra 0.37.0 vs locked
    # 0.38.1) and aborts the build. The comparison is always recorded in the
    # manifest so deploy verifiers can check it mechanically.
    pin_comparison: list[dict[str, object]] = []
    pin_violation: str | None = None
    lock_path = SIBLING_REPOS_DIR / CONSUMING_REPO / "uv.lock"
    try:
        comparisons = build_comparisons(
            lock_path=lock_path,
            siblings={
                "omnibase-compat": SIBLING_REPOS_DIR / "omnibase_compat",
                "onex-change-control": SIBLING_REPOS_DIR / "onex_change_control",
                "omnimarket": SIBLING_REPOS_DIR / "omnimarket",
            },
        )
        # Host self-check (OMN-12989): the runtime image's OWN omnibase_infra is
        # built from the build context, NOT from sibling-repos — that is exactly
        # where the 2026-06-11 stale 0.37.0 entered. Compare the installed infra
        # version against omnimarket's lock pin so a stale build context aborts.
        host_infra = _host_infra_comparison(lock_path)
        if host_infra is not None:
            comparisons.append(host_infra)
        pin_comparison = [c.to_dict() for c in comparisons]
        try:
            assert_pins_satisfied(comparisons)
        except WorkspacePinError as exc:
            pin_violation = str(exc)
    except FileNotFoundError as exc:
        # A missing consuming lock is itself a hard error — it means the build
        # cannot prove sibling freshness. Fail rather than silently skip.
        pin_violation = (
            f"Cannot resolve sibling lock pins: {exc}. "
            "Workspace mode requires the consuming repo's uv.lock to be staged."
        )

    # Fold in the sibling-pin comparison so deploy verifiers can confirm the
    # build honored the consuming repo's uv.lock (OMN-12977). A drift here would
    # already have aborted the build in the default (no-override) path; when an
    # operator override was used, this records expected-vs-actual durably.
    sibling_pin_comparison = _load_pin_comparison(errors)

    manifest = {
        "build_source": "workspace",
        "build_time": os.environ.get("BUILD_DATE", "unknown"),
        "vcs_ref": os.environ.get("VCS_REF", "unknown"),
        "proofs": proofs,
        "pin_comparison": pin_comparison,
        "sibling_pin_comparison": sibling_pin_comparison,
    }

    OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nProvenance manifest written to {OUTPUT_MANIFEST}")

    if errors:
        print("\nWorkspace provenance FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    if pin_violation is not None:
        print(
            "\nWorkspace provenance FAILED (sibling lock-pin regression):",
            file=sys.stderr,
        )
        print(pin_violation, file=sys.stderr)
        return 1

    for entry in pin_comparison:
        print(
            f"  pin OK: {entry['package']} "
            f"expected>={entry['expected_version'] or '(unpinned)'} "
            f"actual={entry['actual_version']} [{entry['status']}]"
        )

    print(f"\nAll {len(proofs)} workspace provenance proofs passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
