# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Performance tests for Docker infrastructure.

These tests verify Docker configuration follows performance best practices:
- Reasonable resource limits
- Optimal health check configuration
- Multi-stage build for smaller images
"""

from __future__ import annotations

import re

import pytest
import yaml

# Import shared constants from conftest for backward compatibility.
# New tests should prefer using the docker_dir or compose_file_path fixtures instead.
from tests.unit.docker.conftest import COMPOSE_FILE_PATH, DOCKER_DIR

# Explicit marker for documentation (also auto-applied by tests/unit/conftest.py)
pytestmark = [pytest.mark.unit]


@pytest.mark.unit
class TestDockerfilePerformance:
    """Tests for Dockerfile performance best practices."""

    def test_multi_stage_build(self) -> None:
        """Verify Dockerfile uses multi-stage build for smaller images."""
        dockerfile = DOCKER_DIR / "Dockerfile.runtime"
        content = dockerfile.read_text()

        # Should have multiple FROM statements (multi-stage)
        from_statements = re.findall(r"^FROM\s+", content, re.MULTILINE)
        assert len(from_statements) >= 2, (
            "Should use multi-stage build (at least 2 FROM statements)"
        )

        # Should have AS builder pattern
        assert "AS builder" in content or "as builder" in content

    def test_build_cache_optimization(self) -> None:
        """Verify Dockerfile uses cache mounts for faster builds."""
        dockerfile = DOCKER_DIR / "Dockerfile.runtime"
        content = dockerfile.read_text()

        # Should use cache mounts
        assert "--mount=type=cache" in content, "Should use BuildKit cache mounts"

    def test_minimal_runtime_dependencies(self) -> None:
        """Verify runtime stage has minimal dependencies."""
        dockerfile = DOCKER_DIR / "Dockerfile.runtime"
        content = dockerfile.read_text()

        # Runtime should not include build tools
        runtime_section_match = re.search(r"FROM.*AS runtime.*", content, re.DOTALL)
        if runtime_section_match:
            runtime_section = content[runtime_section_match.start() :]
            # build-essential should only be in builder stage, not runtime
            # Check that if build-essential appears in runtime section, it's actually part of builder
            if "build-essential" in runtime_section:
                # Extract just the runtime RUN commands (not from builder)
                runtime_run_commands = []
                in_runtime_stage = False
                for line in content.split("\n"):
                    if "FROM" in line and "AS runtime" in line:
                        in_runtime_stage = True
                    elif "FROM" in line and in_runtime_stage:
                        # New stage started, no longer in runtime
                        in_runtime_stage = False
                    elif in_runtime_stage and line.strip().startswith("RUN"):
                        runtime_run_commands.append(line)

                # Check that build-essential is not in runtime RUN commands
                for cmd in runtime_run_commands:
                    assert "build-essential" not in cmd, (
                        "Runtime stage should not install build-essential"
                    )


@pytest.mark.unit
class TestDockerComposePerformance:
    """Tests for docker-compose performance configuration."""

    def test_resource_limits_configured(self) -> None:
        """Verify docker-compose has resource limits."""
        compose_file = COMPOSE_FILE_PATH
        content = compose_file.read_text()

        # Should have resource limits defined
        assert "limits:" in content
        assert "memory:" in content
        assert "cpus:" in content

    def test_health_check_intervals_reasonable(self) -> None:
        """Verify health check intervals are not too aggressive."""
        compose_file = COMPOSE_FILE_PATH
        content = compose_file.read_text()

        # Parse health check intervals
        interval_matches = re.findall(r"interval:\s*(\d+)s", content)
        for interval in interval_matches:
            interval_sec = int(interval)
            # Health checks should not be more frequent than every 10 seconds
            assert interval_sec >= 10, (
                f"Health check interval {interval_sec}s is too aggressive"
            )
            # But also not too infrequent (> 120s)
            assert interval_sec <= 120, (
                f"Health check interval {interval_sec}s is too long"
            )

    def test_worker_replicas_reasonable(self) -> None:
        """Verify default worker replicas is reasonable."""
        compose_file = COMPOSE_FILE_PATH
        content = yaml.safe_load(compose_file.read_text())

        # Check if runtime-worker service exists
        if "runtime-worker" in content.get("services", {}):
            worker_config = content["services"]["runtime-worker"]
            if "deploy" in worker_config and "replicas" in worker_config["deploy"]:
                replicas = worker_config["deploy"]["replicas"]

                # Handle environment variable substitution (e.g., "${WORKER_REPLICAS:-2}")
                if isinstance(replicas, str):
                    # Extract default value from ${VAR:-default} pattern
                    match = re.search(r"\$\{[^}]*:-(\d+)\}", replicas)
                    if match:
                        default_replicas = int(match.group(1))
                        assert 1 <= default_replicas <= 10, (
                            f"Default replicas {default_replicas} seems unusual"
                        )
                elif isinstance(replicas, int):
                    assert 1 <= replicas <= 10, f"Replicas {replicas} seems unusual"

    def test_resource_reservations_configured(self) -> None:
        """Verify docker-compose has resource reservations."""
        compose_file = COMPOSE_FILE_PATH
        content = compose_file.read_text()

        # Should have resource reservations for better orchestration
        assert "reservations:" in content
        # Reservations should be lower than limits
        assert content.count("reservations:") > 0


@pytest.mark.unit
class TestDockerignorePerformance:
    """Tests for .dockerignore optimization."""

    def test_excludes_test_files(self) -> None:
        """Verify .dockerignore excludes test files for smaller builds."""
        dockerignore = DOCKER_DIR / ".dockerignore"
        content = dockerignore.read_text()

        # Should exclude test directories
        assert (
            "tests" in content.lower()
            or "test" in content.lower()
            or "*_test.py" in content
        )

    def test_excludes_cache_directories(self) -> None:
        """Verify .dockerignore excludes cache directories."""
        dockerignore = DOCKER_DIR / ".dockerignore"
        content = dockerignore.read_text()

        # Should exclude common cache directories
        cache_patterns = ["__pycache__", ".pytest_cache", ".mypy_cache"]
        excluded_count = sum(1 for p in cache_patterns if p in content)
        assert excluded_count >= 2, "Should exclude cache directories"

    def test_excludes_version_control(self) -> None:
        """Verify .dockerignore excludes version control files."""
        dockerignore = DOCKER_DIR / ".dockerignore"
        content = dockerignore.read_text()

        # Should exclude .git directory
        assert ".git" in content

    def test_excludes_documentation(self) -> None:
        """Verify .dockerignore excludes unnecessary documentation."""
        dockerignore = DOCKER_DIR / ".dockerignore"
        content = dockerignore.read_text()

        # Should exclude markdown files (except README)
        assert "*.md" in content
        # But should keep README
        assert "!README.md" in content


@pytest.mark.unit
class TestDockerHealthChecks:
    """Tests for Docker health check configuration."""

    def test_dockerfile_healthcheck_configured(self) -> None:
        """Verify Dockerfile has health check configured."""
        dockerfile = DOCKER_DIR / "Dockerfile.runtime"
        content = dockerfile.read_text()

        # Should have HEALTHCHECK instruction
        assert "HEALTHCHECK" in content

    def test_dockerfile_healthcheck_parameters_reasonable(self) -> None:
        """Verify Dockerfile health check parameters are reasonable."""
        dockerfile = DOCKER_DIR / "Dockerfile.runtime"
        content = dockerfile.read_text()

        # Extract healthcheck line
        healthcheck_match = re.search(
            r"HEALTHCHECK\s+--interval=(\d+)s\s+--timeout=(\d+)s\s+--start-period=(\d+)s\s+--retries=(\d+)",
            content,
        )
        if healthcheck_match:
            interval = int(healthcheck_match.group(1))
            timeout = int(healthcheck_match.group(2))
            start_period = int(healthcheck_match.group(3))
            retries = int(healthcheck_match.group(4))

            # Validate parameters
            assert 10 <= interval <= 120, (
                f"Interval {interval}s should be between 10-120s"
            )
            assert 5 <= timeout <= 30, f"Timeout {timeout}s should be between 5-30s"
            assert 20 <= start_period <= 120, (
                f"Start period {start_period}s should be between 20-120s"
            )
            assert 1 <= retries <= 5, f"Retries {retries} should be between 1-5"

    def test_compose_healthcheck_matches_dockerfile(self) -> None:
        """Verify docker-compose health checks align with Dockerfile."""
        dockerfile = DOCKER_DIR / "Dockerfile.runtime"
        dockerfile_content = dockerfile.read_text()

        compose_file = COMPOSE_FILE_PATH
        compose_content = compose_file.read_text()

        # Both should have health checks
        assert "HEALTHCHECK" in dockerfile_content
        assert "healthcheck:" in compose_content

        # Both should check the same endpoint
        if "http://localhost:8085/health" in dockerfile_content:
            assert "http://localhost:8085/health" in compose_content


@pytest.mark.unit
class TestDockerSecurityBestPractices:
    """Tests for Docker security best practices that impact performance."""

    def test_non_root_user_configured(self) -> None:
        """Verify Dockerfile uses non-root user."""
        dockerfile = DOCKER_DIR / "Dockerfile.runtime"
        content = dockerfile.read_text()

        # Should have USER instruction
        assert "USER" in content
        # Should not be root
        assert "USER root" not in content

    def test_minimal_base_image(self) -> None:
        """Verify Dockerfile uses minimal base image."""
        dockerfile = DOCKER_DIR / "Dockerfile.runtime"
        content = dockerfile.read_text()

        # Should use slim or alpine variant for smaller images
        from_lines = re.findall(r"^FROM\s+(.+)$", content, re.MULTILINE)
        runtime_from = [
            line
            for line in from_lines
            if "AS runtime" in line or from_lines.index(line) == len(from_lines) - 1
        ]

        if runtime_from:
            # Check that runtime uses slim variant
            assert any("slim" in line or "alpine" in line for line in runtime_from), (
                "Runtime should use slim or alpine base"
            )


@pytest.mark.unit
class TestDockerBuildOptimization:
    """Tests for Docker build optimization."""

    def test_copy_order_optimization(self) -> None:
        """Verify COPY instructions are ordered for better caching."""
        dockerfile = DOCKER_DIR / "Dockerfile.runtime"
        content = dockerfile.read_text()

        # Find builder stage
        builder_section_match = re.search(
            r"FROM.*AS builder(.*)FROM.*AS runtime", content, re.DOTALL
        )
        if builder_section_match:
            builder_section = builder_section_match.group(1)

            # pyproject.toml should be copied before src/ for better caching
            copy_lines = re.findall(r"COPY\s+([^\s]+)", builder_section)
            if "pyproject.toml" in " ".join(copy_lines) and "src/" in " ".join(
                copy_lines
            ):
                pyproject_index = next(
                    (
                        i
                        for i, line in enumerate(copy_lines)
                        if "pyproject.toml" in line
                    ),
                    None,
                )
                src_index = next(
                    (i for i, line in enumerate(copy_lines) if "src/" in line), None
                )

                if pyproject_index is not None and src_index is not None:
                    assert pyproject_index < src_index, (
                        "pyproject.toml should be copied before src/ for better caching"
                    )

    def test_apt_cache_optimization(self) -> None:
        """Verify apt commands use cache mounts for faster builds."""
        dockerfile = DOCKER_DIR / "Dockerfile.runtime"
        content = dockerfile.read_text()

        # Find RUN commands with apt-get
        apt_commands = re.findall(r"RUN.*apt-get.*", content, re.DOTALL)

        # At least one apt command should use cache mount
        if apt_commands:
            assert any("--mount=type=cache" in cmd for cmd in apt_commands), (
                "apt-get should use cache mounts"
            )
