# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for runtime build provenance summaries and taint handling."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from deploy_agent.events import BuildSource
from deploy_agent.executor import DeployExecutor


def test_release_provenance_summary_reports_manifest_and_versions(
    tmp_path, monkeypatch
) -> None:
    repo_root = tmp_path / "omnibase_infra"
    manifest_path = repo_root / "docker" / "runtime-release-manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "build_source": "release",
                "dependencies": {
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
                },
            }
        ),
        encoding="utf-8",
    )
    taint_marker = tmp_path / "runtime-taint.json"

    monkeypatch.setattr("deploy_agent.executor.REPO_DIR", str(repo_root))
    monkeypatch.setattr("deploy_agent.executor.TAINT_MARKER", taint_marker)

    summary = DeployExecutor()._provenance_summary(BuildSource.RELEASE)

    assert summary["build_source"] == "release"
    assert summary["taint"] == {"status": "clean"}
    assert summary["manifest_path"] == str(manifest_path)
    repos = {entry["repo"]: entry for entry in summary["repos"]}
    assert repos["omnimarket"]["version"] == "0.2.0"


def test_workspace_provenance_summary_reports_dirty_state_and_legacy_worktrees(
    tmp_path, monkeypatch
) -> None:
    omni_home = tmp_path / "omni_home"
    expected_roots = {
        "omnibase_compat": omni_home / "omnibase_compat" / "src" / "omnibase_compat",
        "onex_change_control": omni_home
        / "onex_change_control"
        / "src"
        / "onex_change_control",
        "omnimarket": omni_home / "omnimarket" / "src" / "omnimarket",
    }
    for root in expected_roots.values():
        root.mkdir(parents=True)
        (root / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (omni_home / "worktrees").symlink_to(omni_home / "omni_worktrees")

    monkeypatch.setattr("deploy_agent.executor.OMNI_HOME", str(omni_home))
    monkeypatch.setattr(
        "deploy_agent.executor.TAINT_MARKER",
        tmp_path / "runtime-taint.json",
    )

    with patch("deploy_agent.executor._repo_dirty", side_effect=[True, False, False]):
        summary = DeployExecutor()._provenance_summary(BuildSource.WORKSPACE)

    repos = {entry["repo"]: entry for entry in summary["repos"]}
    assert summary["legacy_worktrees"]["kind"] == "symlink"
    assert repos["omnibase_compat"]["dirty"] is True
    assert str(repos["omnimarket"]["digest"]).startswith("sha256:")


def test_mark_runtime_tainted_and_clear(tmp_path, monkeypatch) -> None:
    taint_marker = tmp_path / "runtime-taint.json"
    monkeypatch.setattr("deploy_agent.executor.TAINT_MARKER", taint_marker)

    executor = DeployExecutor()
    executor.mark_runtime_tainted(reason="manual_patch", source="debug")

    assert taint_marker.is_file()
    data = json.loads(taint_marker.read_text(encoding="utf-8"))
    assert data["reason"] == "manual_patch"

    executor._clear_runtime_taint()
    assert not taint_marker.exists()
