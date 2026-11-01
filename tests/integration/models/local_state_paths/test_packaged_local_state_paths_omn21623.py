# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Built wheel must ship the local state paths declaration (OMN-21623)."""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from omnibase_infra.models.local_state_paths.model_local_state_paths import (
    load_local_state_paths,
)

pytestmark = pytest.mark.integration

_YAML_MEMBER = "omnibase_infra/models/local_state_paths/local_state_paths.yaml"
_ROOT = Path(__file__).resolve().parents[4]


def test_packaged_declaration_loads_from_installed_package() -> None:
    paths = load_local_state_paths()
    assert paths.runtime_socket_path
    assert paths.scope_manifest_path


def test_built_wheel_ships_declaration(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv unavailable")
    subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=_ROOT,
        check=False,
        capture_output=True,
    )
    wheels = list(tmp_path.glob("*.whl"))
    if not wheels:
        pytest.skip("wheel build backend unavailable in this environment")
    with zipfile.ZipFile(wheels[0]) as wheel:
        assert _YAML_MEMBER in wheel.namelist()
