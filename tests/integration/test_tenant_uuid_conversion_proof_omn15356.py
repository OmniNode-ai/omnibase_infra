# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Rebuilt PostgreSQL 16 integration proof for the OMN-15356 canonical UUID
tenant identity conversion (capability_scores.tenant_id TEXT -> UUID).

Mirrors ``test_application_acl_postgres16_proof.py``: builds and runs a
disposable Docker Compose fixture, asserts on its stdout proof markers, and
tears it down unconditionally. No live, shared, or deployed database is
touched -- ``docker/tenant-uuid-conversion-proof/compose.yml`` builds its own
tmpfs-backed PostgreSQL 16 container from scratch every run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[2]
_COMPOSE = _ROOT / "docker" / "tenant-uuid-conversion-proof" / "compose.yml"
_HAS_DOCKER = shutil.which("docker") is not None


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DOCKER, reason="Requires a Docker engine")
def test_capability_scores_tenant_uuid_conversion_proof() -> None:
    project_name = f"tenant-uuid-conversion-proof-{os.getpid()}"
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
        timeout=600,
        check=False,
    )
    try:
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "tenant_uuid_conversion_phase=fail_closed status=PASS" in output
        assert "tenant_uuid_conversion_phase=total_mapping status=PASS" in output
        assert "tenant_uuid_conversion_phase=continuity status=PASS" in output
        assert "tenant_uuid_conversion_status=PASS" in output
    finally:
        subprocess.run(
            [*command, "down", "--volumes", "--remove-orphans"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
