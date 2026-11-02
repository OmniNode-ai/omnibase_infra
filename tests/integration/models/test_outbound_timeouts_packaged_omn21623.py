# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Load timeout settings from a built wheel outside the source tree."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_built_wheel_loads_declared_timeouts(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    built = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path / "dist")],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    (wheel,) = (tmp_path / "dist").glob("*.whl")
    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(installed)
    program = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from omnibase_infra.models.outbound_timeouts import load_outbound_timeouts
import omnibase_infra.models.outbound_timeouts.model_outbound_timeouts as module
assert Path(module.__file__).is_relative_to(sys.argv[1]), module.__file__
timeouts = load_outbound_timeouts()
assert timeouts.llm_coder_request_seconds == 30
assert timeouts.llm_documentation_request_seconds == 150
assert timeouts.llm_coder_request_seconds <= timeouts.llm_coder_max_seconds
assert timeouts.llm_documentation_request_seconds <= timeouts.llm_long_form_max_seconds
print("wheel timeout readback: coder=30/120s documentation=150/180s")
"""
    readback = subprocess.run(
        [sys.executable, "-I", "-c", program, str(installed)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert readback.returncode == 0, readback.stdout + readback.stderr
    assert "wheel timeout readback" in readback.stdout
