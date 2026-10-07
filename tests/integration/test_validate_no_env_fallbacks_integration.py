# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration coverage for the env-fallback node over this repository's runtime roots."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_core.nodes.node_no_env_fallbacks_check_compute.runtime_no_env_fallbacks_check import (
    main,
)

pytestmark = pytest.mark.integration


def test_env_fallback_node_is_clean_for_runtime_roots(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_root = Path(__file__).parents[2]
    monkeypatch.chdir(repo_root)
    files = [
        str(path.relative_to(repo_root))
        for root in (repo_root / "src", repo_root / "scripts")
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".sh", ".bash"}
    ]
    assert files, "runtime roots must contain files to scan"
    assert main(files) == 0
    assert "PASS" in capsys.readouterr().out
