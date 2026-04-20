# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Smoke tests for relocated topic-parity scripts (OMN-9286).

Verifies path resolution and argparse surface for both scripts after relocation
from omni_home/scripts/ to omnibase_infra/scripts/. Full functional coverage
lives at the CI workflow level (.github/workflows/topic-parity.yml in
omni_home) where the scripts run against a real omni_home checkout.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECK_SCRIPT = REPO_ROOT / "scripts" / "check-topic-parity.py"
SYNC_SCRIPT = REPO_ROOT / "scripts" / "sync-topic-registry.py"


@pytest.mark.unit
def test_check_topic_parity_has_expected_shebang_and_spdx() -> None:
    content = CHECK_SCRIPT.read_text().splitlines()
    assert content[0] == "#!/usr/bin/env python3"
    assert "SPDX-FileCopyrightText" in content[1]
    assert "SPDX-License-Identifier: MIT" in content[2]


@pytest.mark.unit
def test_sync_topic_registry_has_expected_shebang_and_spdx() -> None:
    content = SYNC_SCRIPT.read_text().splitlines()
    assert content[0] == "#!/usr/bin/env python3"
    assert "SPDX-FileCopyrightText" in content[1]
    assert "SPDX-License-Identifier: MIT" in content[2]


@pytest.mark.unit
def test_check_topic_parity_help() -> None:
    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--check" in result.stdout
    assert "--omni-home" in result.stdout
    assert "--registry" in result.stdout


@pytest.mark.unit
def test_sync_topic_registry_help() -> None:
    result = subprocess.run(
        [sys.executable, str(SYNC_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--check" in result.stdout
    assert "--write" in result.stdout
    assert "--dry-run" in result.stdout


@pytest.mark.unit
def test_check_topic_parity_omni_home_env_var_resolution(tmp_path: Path) -> None:
    """--check with a bogus OMNI_HOME fails fast, proving env resolution is wired."""
    bogus_home = tmp_path / "nonexistent-omni-home"
    bogus_home.mkdir()
    result = subprocess.run(
        [sys.executable, str(CHECK_SCRIPT), "--check"],
        env={"OMNI_HOME": str(bogus_home), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Registry not found" in result.stderr
    assert str(bogus_home) in result.stderr


@pytest.mark.unit
def test_sync_topic_registry_omni_home_env_var_resolution(tmp_path: Path) -> None:
    """--check with a bogus OMNI_HOME fails fast, proving env resolution is wired."""
    bogus_home = tmp_path / "nonexistent-omni-home"
    bogus_home.mkdir()
    result = subprocess.run(
        [sys.executable, str(SYNC_SCRIPT), "--check"],
        env={"OMNI_HOME": str(bogus_home), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Registry not found" in result.stderr
    assert str(bogus_home) in result.stderr
