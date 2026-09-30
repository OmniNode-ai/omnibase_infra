# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The real tracked tree is free of the GLM Coding Plan endpoint (OMN-20173)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

pytestmark = pytest.mark.integration
_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "scripts" / "check_no_coding_plan_endpoint.py"


def test_real_tracked_tree_addresses_no_coding_plan_endpoint() -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--root", str(_ROOT)],
        capture_output=True,
        text=True,
        check=False,
        env=scrub_git_location_env(os.environ),
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "tracked",
    ["docker/catalog/model_registry.yaml", "docker/docker-compose.judge.yml"],
)
def test_guarded_config_files_exist_in_the_scanned_tree(tracked: str) -> None:
    assert (_ROOT / tracked).is_file()
