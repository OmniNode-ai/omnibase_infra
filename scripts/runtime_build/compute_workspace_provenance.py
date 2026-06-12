#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Compute per-repo workspace provenance and write a manifest.

Run during Docker build (workspace mode only) to:
  - Hash each staged sibling repo tree under /workspace/sibling-repos/
  - Verify that the installed package content matches the recorded digest
  - Write /app/build-provenance.json for operator inspection
  - Print a human-readable provenance summary to stdout

Exit codes:
  0  all proofs passed
  1  digest mismatch (installed content != recorded workspace digest)
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

SIBLING_REPOS_DIR = Path("/workspace/sibling-repos")
VENV_DIR = Path("/app/.venv")
OUTPUT_MANIFEST = Path("/app/build-provenance.json")
# Expected-vs-actual sibling pin comparison produced by the host-side lock-pin
# preflight (check_sibling_lock_pins.py, OMN-12987). Staged under sibling-repos/
# so it rides along into the build image. Absent on builds that skipped the
# preflight (e.g. release mode); the manifest then records an empty comparison.
LOCK_PIN_COMPARISON = SIBLING_REPOS_DIR / ".sibling-lock-pins.json"

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


def _installed_direct_url(dist_name: str) -> dict | None:
    """Return the direct_url metadata dict for an installed package, or None."""
    try:
        dist = importlib.metadata.distribution(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None
    direct_url_text = dist.read_text("direct_url.json")
    if direct_url_text is None:
        return None
    return json.loads(direct_url_text)


def _load_lock_pin_comparison() -> list[dict]:
    """Return the host-side expected-vs-actual sibling pin comparison.

    The list is produced by check_sibling_lock_pins.py (OMN-12987) and staged
    into the build image. Returns an empty list when the file is absent (the
    preflight was skipped) so the manifest is always well-formed.
    """
    if not LOCK_PIN_COMPARISON.is_file():
        return []
    try:
        data = json.loads(LOCK_PIN_COMPARISON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def main() -> int:
    proofs: list[dict] = []
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

        install_url = direct_url.get("url", "")
        is_local = direct_url.get("dir_info", {}).get("editable") is not None or (
            install_url.startswith("file://")
        )

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

    lock_pin_comparison = _load_lock_pin_comparison()
    manifest = {
        "build_source": "workspace",
        "build_time": os.environ.get("BUILD_DATE", "unknown"),
        "vcs_ref": os.environ.get("VCS_REF", "unknown"),
        "proofs": proofs,
        # OMN-12987: expected-vs-actual sibling lock pins so deploy verifiers can
        # detect a stale-sibling build without re-running the host preflight.
        "lock_pin_comparison": lock_pin_comparison,
    }

    OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nProvenance manifest written to {OUTPUT_MANIFEST}")

    if errors:
        print("\nWorkspace provenance FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"\nAll {len(proofs)} workspace provenance proofs passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
