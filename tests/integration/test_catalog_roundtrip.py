# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for catalog generate -> start -> health -> stop."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from omnibase_infra.docker.catalog.resolver import CatalogResolver

REPO_ROOT = str(Path(__file__).parent.parent.parent)
CATALOG_DIR = str(Path(REPO_ROOT) / "docker" / "catalog")

_HAS_DOCKER = shutil.which("docker") is not None
_HAS_POSTGRES_PASSWORD = bool(os.environ.get("POSTGRES_PASSWORD"))


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not _HAS_DOCKER or not _HAS_POSTGRES_PASSWORD,
    reason="Requires Docker daemon and POSTGRES_PASSWORD env var",
)
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
def test_catalog_validator_rejects_missing_env() -> None:
    """Validator must fail before starting if required vars are missing."""
    # Build env dict without POSTGRES_PASSWORD — must pass explicit env to
    # subprocess since monkeypatch.delenv only affects the current process.
    env = {k: v for k, v in os.environ.items() if k != "POSTGRES_PASSWORD"}
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
        env=env,
    )
    assert result.returncode != 0
    assert "POSTGRES_PASSWORD" in result.stderr


@pytest.mark.integration
def test_catalog_runtime_memgraph_bundle_injects_env() -> None:
    """runtime+memgraph bundle injects OMNIMEMORY_* vars and includes memgraph service."""
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    stack = resolver.resolve(["runtime", "memgraph"])

    # Memgraph service must be present
    assert "omnibase-infra-memgraph" in stack.service_names

    # Core infrastructure services pulled in transitively via runtime→core
    assert "postgres" in stack.service_names
    assert "redpanda" in stack.service_names

    # OMNIMEMORY env vars injected by memgraph bundle
    assert stack.injected_env.get("OMNIMEMORY_ENABLED") == "true"
    assert (
        stack.injected_env.get("OMNIMEMORY_MEMGRAPH_HOST") == "omnibase-infra-memgraph"
    )
    assert stack.injected_env.get("OMNIMEMORY_MEMGRAPH_PORT") == "7687"


@pytest.mark.integration
def test_catalog_runtime_without_memgraph_excludes_memory_env() -> None:
    """runtime bundle alone must not inject any OMNIMEMORY_* vars or include memgraph."""
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    stack = resolver.resolve(["runtime"])

    # Memgraph service must NOT be present
    assert "omnibase-infra-memgraph" not in stack.service_names

    # OMNIMEMORY vars must not be injected
    for key in stack.injected_env:
        assert not key.startswith("OMNIMEMORY_"), (
            f"Unexpected OMNIMEMORY var '{key}' in runtime-only stack"
        )


@pytest.mark.integration
def test_catalog_tracing_bundle_injects_otel_env() -> None:
    """tracing bundle injects OTEL vars and pulls in phoenix transitively."""
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    stack = resolver.resolve(["tracing"])

    # Phoenix must be included (tracing→observability→phoenix)
    assert "phoenix" in stack.service_names

    # OTEL env vars injected by tracing bundle
    assert (
        stack.injected_env.get("OTEL_EXPORTER_OTLP_ENDPOINT") == "http://phoenix:6006"
    )
    assert stack.injected_env.get("OTEL_TRACES_EXPORTER") == "otlp"

    # OMNIMEMORY vars must NOT leak in from tracing bundle
    for key in stack.injected_env:
        assert not key.startswith("OMNIMEMORY_"), (
            f"Unexpected OMNIMEMORY var '{key}' leaked into tracing-only stack"
        )
