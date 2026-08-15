# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Rebuilt PostgreSQL 16 application-domain enforcement proof."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[2]
_COMPOSE = _ROOT / "docker" / "application-domain-enforcement" / "compose.yml"
_HAS_DOCKER = shutil.which("docker") is not None


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_DOCKER, reason="Requires a Docker engine")
def test_rebuilt_postgres16_application_domain_enforcement() -> None:
    project_name = f"application-domain-enforcement-{os.getpid()}"
    command = ["docker", "compose", "-p", project_name, "-f", str(_COMPOSE)]
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
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert output.count("domain_control=") == 46
        assert "domain_control=identity-root-runtime-login status=PASS" in output
        assert "domain_control=identity-root-unproven-enumeration status=PASS" in output
        assert "domain_control=identity-root-runtime-membership status=PASS" in output
        assert "domain_control=identity-root-runtime-set-role status=PASS" in output
        assert "domain_control=canonical-policy-unrelated-role status=PASS" in output
        assert "domain_control=tenant-text-key status=PASS" in output
        assert "domain_control=unsafe-security-definer status=PASS" in output
        assert "domain_control=security-definer-volatility-drift status=PASS" in output
        assert "domain_control=old-application-database status=PASS" in output
        assert "domain_control=public-catalog-leak status=PASS" in output
        assert (
            "domain_control=source-tenant-partial-unique-predicate status=PASS"
            in output
        )
        assert (
            "domain_control=source-tenant-generated-unique-alias status=PASS" in output
        )
        assert (
            "domain_control=source-tenant-transitive-whole-row-helper status=PASS"
            in output
        )
        assert (
            "domain_control=source-tenant-named-whole-row-helper status=PASS" in output
        )
        assert "domain_control=source-tenant-check-constraint status=PASS" in output
        assert "domain_control=source-tenant-trigger-body status=PASS" in output
        assert "domain_control=source-tenant-dependent-view status=PASS" in output
        assert (
            "application_domain_enforcement_status=PASS postgres_major=16 "
            "relations=6 catalog_objects=6 pools=4 red_controls=46"
        ) in output
    finally:
        subprocess.run(
            [*command, "down", "--volumes", "--remove-orphans"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
