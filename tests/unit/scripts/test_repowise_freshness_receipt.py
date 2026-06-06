# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    module_path = (
        Path(__file__).parents[3] / "scripts" / "emit_repowise_freshness_receipt.py"
    )
    spec = importlib.util.spec_from_file_location(
        "emit_repowise_freshness_receipt", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_emit_repowise_freshness_receipt_reports_stale_and_no_index(
    tmp_path, monkeypatch
):
    module = _load_module()

    (tmp_path / "repo_a").mkdir()
    (tmp_path / "repo_b").mkdir()
    (tmp_path / ".repowise-workspace.yaml").write_text(
        "\n".join(
            [
                "repos:",
                "  - path: repo_a",
                "    alias: alpha",
                "    indexed_at: '2026-05-29T00:00:00+00:00'",
                "    last_commit_at_index: oldsha",
                "  - path: repo_b",
                "    alias: beta",
                "    docs_mode: skipped",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "_git_branch", lambda repo_dir: "main")
    monkeypatch.setattr(module, "_git_head", lambda repo_dir: "newsha")

    out_path = tmp_path / "freshness.json"
    receipt = module.emit(tmp_path, out_path)
    saved = json.loads(out_path.read_text(encoding="utf-8"))

    assert receipt == saved
    assert saved["summary"]["total"] == 2
    assert saved["summary"]["indexed"] == 1
    assert saved["summary"]["stale"] == 1
    assert saved["summary"]["no_index"] == 1
    assert saved["summary"]["failures"] == 2
    assert saved["repos"][0]["alias"] == "alpha"
    assert saved["repos"][0]["head_sha"] == "newsha"
    assert saved["repos"][0]["index_head_sha"] == "oldsha"
    assert saved["repos"][0]["stale"] is True
    assert saved["repos"][1]["docs_mode"] == "skipped"
    assert saved["repos"][1]["no_index"] is True
    assert (tmp_path / "latest-freshness.json").is_symlink()


def test_order_tolerant_yaml_parsing(tmp_path, monkeypatch):
    """Parser must handle list items where path is not the first key."""
    module = _load_module()

    (tmp_path / "repo_c").mkdir()
    # alias comes before path — order should not matter.
    (tmp_path / ".repowise-workspace.yaml").write_text(
        "\n".join(
            [
                "repos:",
                "  - alias: gamma",
                "    path: repo_c",
                "    indexed_at: '2026-05-30T00:00:00+00:00'",
                "    last_commit_at_index: abc123",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "_git_branch", lambda repo_dir: "main")
    monkeypatch.setattr(module, "_git_head", lambda repo_dir: "abc123")

    out_path = tmp_path / "freshness.json"
    receipt = module.emit(tmp_path, out_path)

    assert receipt["summary"]["total"] == 1
    assert receipt["repos"][0]["alias"] == "gamma"
    assert receipt["repos"][0]["path"] == "repo_c"
    # Same SHA → not stale.
    assert receipt["repos"][0]["stale"] is False
    assert receipt["repos"][0]["failure"] is None


def test_head_unreadable_flagged_as_failure(tmp_path, monkeypatch):
    """A repo that exists and is indexed but has an unreadable HEAD is a failure."""
    module = _load_module()

    (tmp_path / "repo_d").mkdir()
    (tmp_path / ".repowise-workspace.yaml").write_text(
        "\n".join(
            [
                "repos:",
                "  - path: repo_d",
                "    alias: delta",
                "    indexed_at: '2026-05-30T00:00:00+00:00'",
                "    last_commit_at_index: abc123",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "_git_branch", lambda repo_dir: None)
    monkeypatch.setattr(module, "_git_head", lambda repo_dir: None)

    out_path = tmp_path / "freshness.json"
    receipt = module.emit(tmp_path, out_path)

    assert receipt["summary"]["failures"] == 1
    repo = receipt["repos"][0]
    assert repo["failure"] is not None
    assert "unreadable" in repo["failure"]
    assert repo["stale"] is False  # can't determine staleness without HEAD


def test_latest_freshness_falls_back_to_copy_when_symlink_unavailable(
    tmp_path, monkeypatch
):
    module = _load_module()

    (tmp_path / "repo_e").mkdir()
    (tmp_path / ".repowise-workspace.yaml").write_text(
        "\n".join(
            [
                "repos:",
                "  - path: repo_e",
                "    alias: epsilon",
                "    indexed_at: '2026-05-30T00:00:00+00:00'",
                "    last_commit_at_index: abc123",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "_git_branch", lambda repo_dir: "main")
    monkeypatch.setattr(module, "_git_head", lambda repo_dir: "abc123")

    def reject_symlink(self, target):
        raise OSError("symlinks unavailable")

    monkeypatch.setattr(module.Path, "symlink_to", reject_symlink)

    out_path = tmp_path / "freshness.json"
    module.emit(tmp_path, out_path)

    latest_path = tmp_path / "latest-freshness.json"
    assert not latest_path.is_symlink()
    assert json.loads(latest_path.read_text(encoding="utf-8")) == json.loads(
        out_path.read_text(encoding="utf-8")
    )
