# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Validate an immutable sim image context with the workspace pin authority."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from scripts.runtime_build import compute_workspace_provenance as provenance
from scripts.runtime_build.check_sibling_lock_pins import check_pins

VENDORED = ("omnibase_core", "omnibase_compat", "omnimarket")
PREFLIGHT = {
    "omnibase-infra": "omnibase_infra",
    "omnibase-core": "omnibase_core",
    "omnibase-spi": "omnibase_spi",
    "omnibase-compat": "omnibase_compat",
    "omnimarket": "omnimarket",
}


def tree_digest(root: Path, *, exclude_workspace: bool = False) -> str:
    """Hash paths and bytes; a source byte change must change the image identity."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if exclude_workspace and relative.parts[0] == "workspace":
            continue
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git(source: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source), *args],
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(os.environ),
    ).stdout.rstrip("\n")


def prepare(
    context: Path,
    source_home: Path,
    infra: Path,
    market: Path,
    core: Path,
    heads: dict[str, str],
) -> dict[str, str]:
    """Emit canonical pin comparisons and verified stage_workspace VCS shape."""
    staged = context / "workspace" / "sibling-repos"
    sources = {
        "omnibase_infra": infra,
        "omnibase_core": core,
        "omnibase_spi": source_home / "omnibase_spi",
        "omnibase_compat": source_home / "omnibase_compat",
        "omnimarket": market,
    }
    for name in VENDORED:
        source = sources[name]
        expected = heads[name]
        if _git(source, "rev-parse", "HEAD") != expected:
            raise ValueError(f"{name} HEAD changed during context capture")
        dirty = _git(source, "status", "--porcelain", "--untracked-files=all")
        unstaged_runtime = [
            line[3:] for line in dirty.splitlines() if not line[3:].startswith("tests/")
        ]
        if unstaged_runtime:
            raise ValueError(
                f"{name} has uncommitted non-test files omitted by git archive: "
                f"{unstaged_runtime}"
            )
        archived = staged / name
        if (archived / "pyproject.toml").read_bytes() != (
            source / "pyproject.toml"
        ).read_bytes():
            raise ValueError(f"{name} pyproject differs from archived source")
        (archived / ".build-sha").write_text(expected + "\n", encoding="utf-8")

    # This is the same checker stage_workspace.sh calls. Its Pydantic model and
    # drift rules produce the artifact consumed by compute_workspace_provenance.
    roots = {package: sources[name] for package, name in PREFLIGHT.items()}
    comparison = context / "workspace" / "sibling-pin-comparison.json"
    with (
        patch.dict(os.environ, scrub_git_location_env(os.environ), clear=True),
        contextlib.redirect_stdout(sys.stderr),
    ):
        status = check_pins(
            staged / "omnimarket" / "uv.lock",
            roots,
            comparison,
            workspace_mode=True,
        )
    if status != 0:
        raise ValueError(f"sibling pin preflight failed (exit {status})")
    actual = json.loads(comparison.read_text(encoding="utf-8"))
    for row in actual["comparisons"]:
        name = PREFLIGHT[row["package"]]
        if row["actual_git_sha"] != heads[name]:
            raise ValueError(f"{name} pin comparison resolved a different HEAD")
    if (context / "pyproject.toml").read_bytes() != (
        infra / "pyproject.toml"
    ).read_bytes():
        raise ValueError("infra pyproject changed during context capture")

    siblings: dict[str, dict[str, str | bool]] = {}
    for name in VENDORED:
        source = sources[name]
        # Archive contains only committed bytes. The source may have unrelated
        # dirty tests, while the image's vendored content is exactly this SHA.
        siblings[name] = {
            "vcs_ref": heads[name],
            "vcs_dirty": False,
            "vcs_branch": _git(source, "rev-parse", "--abbrev-ref", "HEAD"),
        }
    vcs = context / "workspace" / "sibling-vcs-provenance.json"
    vcs.write_text(
        json.dumps({"siblings": siblings}, indent=2) + "\n", encoding="utf-8"
    )
    # Dockerfile.runtime installs this module in the image and consumes these
    # exact two artifacts. Exercise its readers before Docker sees the context.
    errors: list[str] = []
    with (
        patch.object(provenance, "PIN_COMPARISON_PATH", comparison),
        patch.object(provenance, "VCS_PROVENANCE_PATH", vcs),
    ):
        consumed_pins = provenance._load_pin_comparison(errors)
        consumed_vcs = provenance._load_vcs_provenance(errors)
    if errors or consumed_pins != actual or consumed_vcs != {"siblings": siblings}:
        raise ValueError(
            f"Dockerfile workspace provenance rejected artifacts: {errors}"
        )
    if set(provenance.WORKSPACE_PACKAGES) != set(VENDORED):
        raise ValueError(
            "staged siblings differ from Dockerfile provenance package set"
        )
    return {
        "infra_snapshot_sha256": tree_digest(context, exclude_workspace=True),
        "core_snapshot_sha256": tree_digest(staged / "omnibase_core"),
        "market_snapshot_sha256": tree_digest(staged / "omnimarket"),
        "compat_snapshot_sha256": tree_digest(staged / "omnibase_compat"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--source-home", type=Path, required=True)
    parser.add_argument("--infra", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--market", type=Path, required=True)
    parser.add_argument("--head", action="append", required=True)
    args = parser.parse_args()
    heads = dict(item.split("=", 1) for item in args.head)
    if set(heads) != set(PREFLIGHT.values()):
        parser.error("--head must name all five preflight repositories")
    print(
        json.dumps(
            prepare(
                args.context,
                args.source_home,
                args.infra,
                args.market,
                args.core,
                heads,
            )
        )
    )


if __name__ == "__main__":
    main()
