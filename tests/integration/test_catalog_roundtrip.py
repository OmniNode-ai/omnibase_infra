# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for catalog generate -> start -> health -> stop."""

from __future__ import annotations

import subprocess

import pytest

REPO_ROOT = "/Volumes/PRO-G40/Code/omni_home/omnibase_infra"


@pytest.mark.integration
@pytest.mark.slow
def test_catalog_generates_and_starts_core_bundle() -> None:
    """Resolve core bundle, generate compose, start, health check, stop."""
    # Generate
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
            "docker/docker-compose.generated.yml",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, f"Generate failed: {result.stderr}"

    # Start
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            "docker/docker-compose.generated.yml",
            "up",
            "-d",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, f"Start failed: {result.stderr}"

    try:
        # Health check postgres
        result = subprocess.run(
            [
                "docker",
                "exec",
                "omnibase-infra-postgres",
                "pg_isready",
                "-U",
                "postgres",
                "-d",
                "omnibase_infra",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0

        # Health check redpanda
        result = subprocess.run(
            [
                "docker",
                "exec",
                "omnibase-infra-redpanda",
                "rpk",
                "cluster",
                "health",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0
    finally:
        subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                "docker/docker-compose.generated.yml",
                "down",
            ],
            capture_output=True,
            cwd=REPO_ROOT,
            check=False,
        )


@pytest.mark.integration
@pytest.mark.slow
def test_catalog_validator_rejects_missing_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validator must fail before starting if required vars are missing."""
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "omnibase_infra.docker.catalog.cli",
            "validate",
            "runtime",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode != 0
    assert "POSTGRES_PASSWORD" in result.stderr
