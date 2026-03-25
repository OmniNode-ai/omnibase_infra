# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the catalog CLI."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.mark.unit
def test_cli_generate_produces_compose_file(tmp_path: Path) -> None:
    """onex generate core must produce a valid compose YAML."""
    output = str(tmp_path / "compose.yml")
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "omnibase_infra.docker.catalog.cli",
            "generate",
            "core",
            "--output",
            output,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Generate failed: {result.stderr}"
    assert Path(output).exists()
