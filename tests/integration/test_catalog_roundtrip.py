# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for catalog generate -> start -> health -> stop."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
import yaml

from omnibase_infra.docker.catalog.generator import generate_compose
from omnibase_infra.docker.catalog.resolver import CatalogResolver
from tests.helpers.compose_isolation import (
    assert_compose_isolated,
    isolated_project_name,
    without_host_ports,
)

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
def test_catalog_generates_and_starts_core_bundle(tmp_path: Path) -> None:
    """Resolve core bundle, generate compose, start, health check, stop.

    OMN-17427: the stack runs under a compose project of its own, with every
    container, volume and network scoped to that project and no host port
    published. The earlier form ran the default ``omnibase-infra`` project with
    the dev lane's container names, and on the .201 lab host its ``up``/``down``
    removed the dev lane's postgres, redpanda, valkey, keycloak and infisical.
    """
    # The CLI still renders the core bundle; its output goes to a scratch path,
    # never into the checkout's docker/ directory.
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
            str(tmp_path / "cli-generated.yml"),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, f"Generate failed: {result.stderr}"

    project = isolated_project_name("catalog-roundtrip")
    resolved = CatalogResolver(catalog_dir=CATALOG_DIR).resolve(["core"])
    resolved.project = project
    compose = without_host_ports(generate_compose(resolved, environment=os.environ))
    assert_compose_isolated(compose)
    services = compose["services"]
    assert isinstance(services, dict)
    postgres = str(services["postgres"]["container_name"])
    redpanda = str(services["redpanda"]["container_name"])

    compose_file = tmp_path / "compose.yml"
    compose_file.write_text(
        yaml.safe_dump(compose, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    command = ["docker", "compose", "-p", project, "-f", str(compose_file)]

    # Start -- wrap in try/finally immediately to ensure cleanup
    result = subprocess.run(
        [*command, "up", "-d"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )

    try:
        assert result.returncode == 0, f"Start failed: {result.stderr}"
        assert _eventually(
            [
                "docker",
                "exec",
                postgres,
                "pg_isready",
                "-U",
                "postgres",
                "-d",
                "omnibase_infra",
            ]
        ), f"{postgres} never reported ready"
        assert _eventually(["docker", "exec", redpanda, "rpk", "cluster", "health"]), (
            f"{redpanda} never reported healthy"
        )
    finally:
        subprocess.run(
            [*command, "down", "--volumes", "--remove-orphans"],
            capture_output=True,
            cwd=REPO_ROOT,
            check=False,
        )


def _eventually(argv: list[str], attempts: int = 30, delay_s: float = 2.0) -> bool:
    """True once ``argv`` exits 0; a freshly started container needs a moment."""
    for _ in range(attempts):
        done = subprocess.run(
            argv, capture_output=True, text=True, timeout=30, check=False
        )
        if done.returncode == 0:
            return True
        time.sleep(delay_s)
    return False


@pytest.mark.integration
@pytest.mark.slow
def test_catalog_validator_rejects_missing_env(tmp_path: Path) -> None:
    """Validator must fail before starting if required vars are missing.

    The CLI auto-loads ``$HOME/.omnibase/.env`` at startup, so removing the
    var from the subprocess env is not sufficient on operator machines that
    have a populated .env. Redirect HOME to an empty tmpdir so the autoload
    finds nothing.
    """
    env = {k: v for k, v in os.environ.items() if k != "POSTGRES_PASSWORD"}
    env["HOME"] = str(tmp_path)
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
def test_catalog_runtime_renders_local_ingress_tmpfs() -> None:
    """Generated runtime compose must preserve the runtime-owned socket tmpfs."""
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    stack = resolver.resolve(["runtime"])
    compose = generate_compose(stack)
    services = compose["services"]
    assert isinstance(services, dict)

    runtime_service = services["omninode-runtime"]
    assert runtime_service["tmpfs"] == ["/run/onex-runtime:uid=1000,gid=1000,mode=0770"]


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
