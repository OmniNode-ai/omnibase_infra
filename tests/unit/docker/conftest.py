# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pytest fixtures for Docker unit tests.

This module provides shared fixtures and constants for Docker unit tests.
These tests validate Docker configuration files without requiring Docker daemon.

Module-Level Constants
----------------------
Shared path constants available via fixtures:
    - docker_dir: Path to docker/ directory
    - dockerfile_path: Path to Dockerfile.runtime
    - compose_file_path: Path to docker-compose.runtime.yml
    - env_example_path: Path to .env.example
    - dockerignore_path: Path to .dockerignore
"""

from __future__ import annotations

from pathlib import Path

import pytest

# =============================================================================
# Path Constants (computed once at module load)
# =============================================================================

# Navigate from this file's location to project root.
# Path traversal: tests/unit/docker/conftest.py
#   .parent -> tests/unit/docker/
#   .parent -> tests/unit/
#   .parent -> tests/
#   .parent -> project_root (omnibase_infra2/)
# This is fragile if the file moves - consider using a marker file like pyproject.toml
# to find project root dynamically if this becomes a maintenance burden.
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_DOCKER_DIR = _PROJECT_ROOT / "docker"

# Docker configuration file paths
_DOCKERFILE_PATH = _DOCKER_DIR / "Dockerfile.runtime"
_COMPOSE_FILE_PATH = _DOCKER_DIR / "docker-compose.runtime.yml"
_ENV_EXAMPLE_PATH = _DOCKER_DIR / ".env.example"
_DOCKERIGNORE_PATH = _DOCKER_DIR / ".dockerignore"


# =============================================================================
# Path Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return project root directory path.

    Returns:
        Path: Absolute path to project root.
    """
    return _PROJECT_ROOT


@pytest.fixture(scope="session")
def docker_dir() -> Path:
    """Return Docker configuration directory path.

    Returns:
        Path: Absolute path to docker/ directory.
    """
    return _DOCKER_DIR


@pytest.fixture(scope="session")
def dockerfile_path() -> Path:
    """Return Dockerfile path.

    Returns:
        Path: Absolute path to Dockerfile.runtime.
    """
    return _DOCKERFILE_PATH


@pytest.fixture(scope="session")
def compose_file_path() -> Path:
    """Return docker-compose file path.

    Returns:
        Path: Absolute path to docker-compose.runtime.yml.
    """
    return _COMPOSE_FILE_PATH


@pytest.fixture(scope="session")
def env_example_path() -> Path:
    """Return .env.example file path.

    Returns:
        Path: Absolute path to .env.example.
    """
    return _ENV_EXAMPLE_PATH


@pytest.fixture(scope="session")
def dockerignore_path() -> Path:
    """Return .dockerignore file path.

    Returns:
        Path: Absolute path to .dockerignore.
    """
    return _DOCKERIGNORE_PATH


# =============================================================================
# Content Fixtures (cached file reads)
# =============================================================================


@pytest.fixture(scope="session")
def dockerfile_content(dockerfile_path: Path) -> str:
    """Return Dockerfile content.

    Cached at session scope to avoid repeated file reads.

    Args:
        dockerfile_path: Path fixture for Dockerfile.

    Returns:
        str: Content of Dockerfile.runtime.
    """
    return dockerfile_path.read_text()


@pytest.fixture(scope="session")
def compose_file_content(compose_file_path: Path) -> str:
    """Return docker-compose file content.

    Cached at session scope to avoid repeated file reads.

    Args:
        compose_file_path: Path fixture for docker-compose file.

    Returns:
        str: Content of docker-compose.runtime.yml.
    """
    return compose_file_path.read_text()


@pytest.fixture(scope="session")
def env_example_content(env_example_path: Path) -> str:
    """Return .env.example file content.

    Cached at session scope to avoid repeated file reads.

    Args:
        env_example_path: Path fixture for .env.example file.

    Returns:
        str: Content of .env.example.
    """
    return env_example_path.read_text()


@pytest.fixture(scope="session")
def dockerignore_content(dockerignore_path: Path) -> str:
    """Return .dockerignore file content.

    Cached at session scope to avoid repeated file reads.

    Args:
        dockerignore_path: Path fixture for .dockerignore file.

    Returns:
        str: Content of .dockerignore.
    """
    return dockerignore_path.read_text()


# =============================================================================
# Backward Compatibility Exports
# =============================================================================

# Export constants for direct import in test files that need module-level access.
# Prefer using fixtures where possible for better test isolation.
DOCKER_DIR = _DOCKER_DIR
PROJECT_ROOT = _PROJECT_ROOT
