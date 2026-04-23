# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime build provenance helpers for workspace and release modes."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any, cast

SIBLING_DEPENDENCIES: dict[str, dict[str, str]] = {
    "omnibase_compat": {
        "distribution": "omnibase-compat",
        "package": "omnibase_compat",
        "source_subdir": "src/omnibase_compat",
    },
    "onex_change_control": {
        "distribution": "onex-change-control",
        "package": "onex_change_control",
        "source_subdir": "src/onex_change_control",
    },
    "omnimarket": {
        "distribution": "omnimarket",
        "package": "omnimarket",
        "source_subdir": "src/omnimarket",
    },
}

IGNORED_SUFFIXES = {".pyc", ".pyo"}
IGNORED_DIRS = {"__pycache__", ".git"}


def _iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix not in IGNORED_SUFFIXES:
            files.append(path)
    return files


def digest_tree(root: Path) -> str:
    """Return deterministic sha256 digest for the selected source tree."""
    if not root.exists() or not root.is_dir():
        msg = f"Cannot digest missing tree: {root}"
        raise FileNotFoundError(msg)

    digest = hashlib.sha256()
    for path in _iter_files(root):
        rel = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(rel)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def workspace_source_roots(workspace_root: Path) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for repo, metadata in SIBLING_DEPENDENCIES.items():
        source_root = workspace_root / repo / metadata["source_subdir"]
        if not source_root.is_dir():
            msg = f"Workspace source root missing for {repo}: {source_root}"
            raise FileNotFoundError(msg)
        roots[repo] = source_root
    return roots


def build_workspace_manifest(workspace_root: Path) -> dict[str, object]:
    repos = []
    for repo, source_root in workspace_source_roots(workspace_root).items():
        metadata = SIBLING_DEPENDENCIES[repo]
        repos.append(
            {
                "repo": repo,
                "distribution": metadata["distribution"],
                "package": metadata["package"],
                "source_root": str(source_root),
                "source_digest": digest_tree(source_root),
            }
        )
    return {
        "schema_version": "1.0.0",
        "build_source": "workspace",
        "repos": repos,
    }


def load_manifest(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = f"Manifest must be a JSON object: {path}"
        raise ValueError(msg)
    return data


def load_release_manifest(path: Path) -> dict[str, object]:
    data = load_manifest(path)
    deps = data.get("dependencies")
    if not isinstance(deps, dict) or not deps:
        msg = f"Release manifest must define non-empty dependencies: {path}"
        raise ValueError(msg)
    for repo, metadata in SIBLING_DEPENDENCIES.items():
        if repo not in deps:
            msg = f"Release manifest missing dependency for {repo}"
            raise ValueError(msg)
        dep = deps[repo]
        if not isinstance(dep, dict):
            msg = f"Release manifest dependency for {repo} must be an object"
            raise ValueError(msg)
        if dep.get("distribution") != metadata["distribution"]:
            msg = (
                f"Release manifest distribution mismatch for {repo}: "
                f"{dep.get('distribution')} != {metadata['distribution']}"
            )
            raise ValueError(msg)
        if not dep.get("version"):
            msg = f"Release manifest version missing for {repo}"
            raise ValueError(msg)
    return data


def installed_package_root(package_name: str) -> Path:
    module = importlib.import_module(package_name)
    module_file = getattr(module, "__file__", None)
    if not module_file:
        msg = f"Installed package root not found for {package_name}"
        raise RuntimeError(msg)
    return Path(module_file).resolve().parent


def verify_workspace_installations(manifest: dict[str, object]) -> None:
    repos = manifest.get("repos", [])
    if not isinstance(repos, list):
        msg = "Workspace manifest must contain a 'repos' list"
        raise ValueError(msg)

    mismatches: list[str] = []
    for entry in repos:
        if not isinstance(entry, dict):
            mismatches.append("Invalid repo entry: expected object")
            continue
        package_name = str(entry["package"])
        expected_digest = str(entry["source_digest"])
        installed_root = installed_package_root(package_name)
        installed_digest = digest_tree(installed_root)
        if installed_digest != expected_digest:
            mismatches.append(
                f"{package_name}: installed digest {installed_digest} != {expected_digest}"
            )

    if mismatches:
        msg = "Workspace install verification failed:\n" + "\n".join(mismatches)
        raise RuntimeError(msg)


def verify_release_installations(manifest: dict[str, object]) -> None:
    deps = manifest.get("dependencies", {})
    if not isinstance(deps, dict):
        msg = "Release manifest must contain a dependencies object"
        raise ValueError(msg)

    failures: list[str] = []
    for repo, expected in deps.items():
        if not isinstance(expected, dict):
            failures.append(f"{repo}: invalid dependency entry")
            continue
        distribution = str(expected["distribution"])
        version = str(expected["version"])
        installed_version = importlib.metadata.version(distribution)
        if installed_version != version:
            failures.append(
                f"{distribution}: installed {installed_version} != manifest {version}"
            )

        dist = importlib.metadata.distribution(distribution)
        try:
            direct_url = dist.read_text("direct_url.json")
        except FileNotFoundError:
            direct_url = None
        if direct_url:
            failures.append(
                f"{distribution}: direct_url.json present, release mode forbids VCS/local/editable installs"
            )

    if failures:
        msg = "Release install verification failed:\n" + "\n".join(failures)
        raise RuntimeError(msg)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    workspace = subparsers.add_parser("write-workspace-manifest")
    workspace.add_argument("--workspace-root", required=True)
    workspace.add_argument("--output", required=True)

    verify_workspace = subparsers.add_parser("verify-workspace-install")
    verify_workspace.add_argument("--manifest", required=True)

    verify_release = subparsers.add_parser("verify-release-install")
    verify_release.add_argument("--manifest", required=True)

    release_requirement = subparsers.add_parser("release-requirement")
    release_requirement.add_argument("--manifest", required=True)
    release_requirement.add_argument("--repo", required=True)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "write-workspace-manifest":
        manifest = build_workspace_manifest(Path(args.workspace_root))
        Path(args.output).write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return

    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    build_source = str(manifest.get("build_source", ""))
    if args.command == "verify-release-install":
        if build_source != "release":
            msg = f"Expected release manifest, got build_source={build_source!r}"
            raise ValueError(msg)
        manifest = load_release_manifest(manifest_path)
        verify_release_installations(manifest)
        return

    if args.command == "release-requirement":
        manifest = load_release_manifest(manifest_path)
        deps = cast(dict[str, Any], manifest["dependencies"])
        repo = args.repo
        if repo not in deps:
            msg = f"Release manifest missing repo: {repo}"
            raise ValueError(msg)
        dep = cast(dict[str, str], deps[repo])
        if not isinstance(dep, dict):
            msg = f"Invalid dependency entry for repo: {repo}"
            raise ValueError(msg)
        sys.stdout.write(f"{dep['distribution']}=={dep['version']}\n")
        return

    if args.command == "verify-workspace-install":
        if build_source != "workspace":
            msg = f"Expected workspace manifest, got build_source={build_source!r}"
            raise ValueError(msg)
        verify_workspace_installations(manifest)
        return


if __name__ == "__main__":
    main()
