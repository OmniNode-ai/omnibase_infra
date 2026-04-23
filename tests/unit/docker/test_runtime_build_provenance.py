# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for runtime build provenance helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnibase_infra.docker.runtime_build import (
    build_workspace_manifest,
    digest_tree,
    load_release_manifest,
)


@pytest.mark.unit
def test_digest_tree_ignores_git_and_pyc_files(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    package = root / "src" / "sample"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "__init__.cpython-312.pyc").write_bytes(b"junk")
    (root / ".git").mkdir()
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    digest_one = digest_tree(root)
    (package / "__pycache__" / "other.pyc").write_bytes(b"still ignored")
    digest_two = digest_tree(root)

    assert digest_one == digest_two


@pytest.mark.unit
def test_build_workspace_manifest_records_repo_roots_and_digests(
    tmp_path: Path,
) -> None:
    for repo, package in (
        ("omnibase_compat", "omnibase_compat"),
        ("onex_change_control", "onex_change_control"),
        ("omnimarket", "omnimarket"),
    ):
        source_root = tmp_path / repo / "src" / package
        source_root.mkdir(parents=True)
        (source_root / "__init__.py").write_text(f"NAME = '{repo}'\n", encoding="utf-8")

    manifest = build_workspace_manifest(tmp_path)
    repos = {entry["repo"]: entry for entry in manifest["repos"]}

    assert manifest["build_source"] == "workspace"
    assert set(repos) == {"omnibase_compat", "onex_change_control", "omnimarket"}
    for repo, entry in repos.items():
        assert entry["source_root"].endswith(
            {
                "omnibase_compat": "omnibase_compat/src/omnibase_compat",
                "onex_change_control": "onex_change_control/src/onex_change_control",
                "omnimarket": "omnimarket/src/omnimarket",
            }[repo]
        )
        assert str(entry["source_digest"]).startswith("sha256:")


@pytest.mark.unit
def test_release_manifest_pins_all_runtime_siblings() -> None:
    manifest_path = Path("docker/runtime-release-manifest.json")
    manifest = load_release_manifest(manifest_path)

    assert manifest["build_source"] == "release"
    assert manifest["schema_version"] == "1.0.0"
    assert manifest["dependencies"] == {
        "omnibase_compat": {
            "distribution": "omnibase-compat",
            "version": "0.3.1",
        },
        "onex_change_control": {
            "distribution": "onex-change-control",
            "version": "0.5.0",
        },
        "omnimarket": {
            "distribution": "omnimarket",
            "version": "0.2.0",
        },
    }


@pytest.mark.unit
def test_release_manifest_is_strict_about_missing_repos(tmp_path: Path) -> None:
    manifest_path = tmp_path / "runtime-release-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "build_source": "release",
                "dependencies": {
                    "omnibase_compat": {
                        "distribution": "omnibase-compat",
                        "version": "0.3.1",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="onex_change_control"):
        load_release_manifest(manifest_path)
