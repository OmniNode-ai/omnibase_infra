# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for the OMN-10822 type-ignore budget gate."""

from __future__ import annotations

from pathlib import Path

from omnibase_infra.validators.type_ignore_budget import main


def test_type_ignore_budget_gate_runs_against_fixture_tree(tmp_path: Path) -> None:
    module = tmp_path / "module_fixture.py"
    module.write_text("value = 1\n", encoding="utf-8")

    assert main(["--max-violations", "0", str(tmp_path)]) == 0
