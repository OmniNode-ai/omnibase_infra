# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Rebuilt PostgreSQL 16 integration proof for the generated ACL matrix."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[2]
_COMPOSE = _ROOT / "docker" / "application-acl-proof" / "compose.yml"
_HAS_DOCKER = shutil.which("docker") is not None


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DOCKER, reason="Requires a Docker engine")
def test_rebuilt_postgres16_acl_and_rollback_proof() -> None:
    project_name = f"application-acl-proof-{os.getpid()}"
    command = [
        "docker",
        "compose",
        "-p",
        project_name,
        "-f",
        str(_COMPOSE),
    ]
    result = subprocess.run(
        [
            *command,
            "up",
            "--build",
            "--abort-on-container-exit",
            "--exit-code-from",
            "proof",
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=1_500,
        check=False,
    )
    try:
        assert result.returncode == 0, result.stdout + result.stderr
        output = result.stdout + result.stderr
        assert "acl_phase=scaffold_wrong_database_guard status=PASS" in output
        assert "acl_phase=scaffold_fresh_additive_round_trip status=PASS" in output
        assert "acl_phase=rollback_atomic_failure" in output
        assert "acl_status=PASS postgres_major=16" in output
    finally:
        subprocess.run(
            [*command, "down", "--volumes", "--remove-orphans"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
