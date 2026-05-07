# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration coverage for file-backed scope overlay loading (OMN-9905)."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.scope.loader import ScopeCache, load_scope


@pytest.mark.integration
def test_load_scope_applies_file_backed_overlay(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        """
        overlays:
          - selector:
              hook_glob: "pre_tool_use_*"
            set:
              enforcement:
                default: warn
          - selector:
              hook: pre_tool_use_bash_guard
            set:
              enforcement:
                default: block
        """,
        encoding="utf-8",
    )

    result = load_scope(
        "pre_tool_use_bash_guard",
        overlay_path=overlay,
        cache=ScopeCache(),
    )

    assert result.enforcement.default == "block"
