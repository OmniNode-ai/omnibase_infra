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
