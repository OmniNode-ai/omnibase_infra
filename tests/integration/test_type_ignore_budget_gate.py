# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Root integration coverage for the type-ignore budget gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.validators.type_ignore_budget import main


@pytest.mark.integration
def test_type_ignore_budget_gate_accepts_clean_fixture(tmp_path: Path) -> None:
    fixture = tmp_path / "module_fixture.py"
    fixture.write_text("value = 1\n", encoding="utf-8")

    assert main(["--max-violations", "0", str(tmp_path)]) == 0
