# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Guard delegation capability cells migration declarations and file presence."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker/migrations/forward"
_VALIDATOR = _ROOT / "scripts/validation/validate_application_migration_manifest.py"
_NODE = Path("nodes/node_projection_delegation_capability_map")
_FILENAMES = (
    "0000_create_delegation_capability_cells.sql",
    "0001_grant_tenant_projection_writer_delegation_capability_cells.sql",
    "0002_force_rls_delegation_capability_cells.sql",
)


def _validate(migrations_dir: Path | None = None) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(_VALIDATOR)]
    if migrations_dir is not None:
        command.extend(
            [
                "--migrations-dir",
                str(migrations_dir),
                "--declarations",
                str(migrations_dir / "_ledger/application-migrations.tsv"),
            ]
        )
    return subprocess.run(
        command, cwd=_ROOT, capture_output=True, text=True, check=False
    )


def test_delegation_capability_cells_migrations_real_tree_validates() -> None:
    result = _validate()
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("filename", _FILENAMES)
def test_delegation_capability_cells_migrations_missing_file_fails(
    tmp_path: Path, filename: str
) -> None:
    forward = shutil.copytree(_FORWARD, tmp_path / "forward")
    missing = forward / _NODE / filename
    missing.unlink()
    result = _validate(forward)
    assert result.returncode == 1, result.stderr
    assert result.stderr.strip() == f"FAIL: declared artifact is missing: {missing}"


def test_delegation_capability_cells_migrations_ledger_and_disk_are_in_sync() -> None:
    ledger = _FORWARD / "_ledger/application-migrations.tsv"
    artifacts = {
        line.split("\t")[0] for line in ledger.read_text(encoding="utf-8").splitlines()
    }
    for filename in _FILENAMES:
        artifact = _NODE / filename
        assert artifact.as_posix() in artifacts, f"Missing ledger row: {artifact}"
        assert (_FORWARD / artifact).is_file(), f"Missing migration file: {artifact}"
