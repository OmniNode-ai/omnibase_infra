# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pytest fixtures for Docker integration tests.

This module provides shared fixtures for Docker integration tests including:
- Docker client initialization with availability detection
- Image build and cleanup helpers
- Container lifecycle management
- Port allocation utilities

All fixtures handle proper cleanup to prevent orphan containers/images.

Fixture Scoping Strategy
------------------------
Fixtures are organized by scope to optimize test execution:

Session-scoped (run once per test session):
    - docker_available: Docker daemon availability check
    - buildkit_available: BuildKit support detection
    - project_root, docker_dir: Path fixtures (immutable)
    - dockerfile_path, compose_file_path: File path fixtures

Module-scoped (shared within test module):
    - test_image_name: Unique image name per module
    - built_test_image: Expensive image build, reused across module tests

Function-scoped (per-test isolation):
    - available_port: Dynamic port allocation (prevents conflicts)
    - container_runner: Container lifecycle context manager
    - wait_for_healthy_fixture: Health check polling
    - wait_for_log_message_fixture: Log message detection

Module-Level Utilities
----------------------
Helper functions available for direct import:
    - run_container(): Context manager for container lifecycle
    - wait_for_healthy(): Poll container health status
    - wait_for_log_message(): Wait for log output
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from collections.abc import Callable, Generator, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

import pytest

# Project root and Docker directory paths.
# Navigate from this file's location to project root:
# Path traversal: tests/integration/docker/conftest.py
#   .parent -> tests/integration/docker/
#   .parent -> tests/integration/
#   .parent -> tests/
#   .parent -> project_root (omnibase_infra2/)
# This is fragile if the file moves - consider using a marker file like pyproject.toml
# to find project root dynamically if this becomes a maintenance burden.
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DOCKER_DIR = PROJECT_ROOT / "docker"
DOCKERFILE_PATH = DOCKER_DIR / "Dockerfile.runtime"
COMPOSE_FILE_PATH = DOCKER_DIR / "docker-compose.runtime.yml"


def _is_docker_available() -> bool:
    """Check if Docker daemon is available and running.

    Returns:
        bool: True if Docker is available, False otherwise.
    """
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _is_buildkit_available() -> bool:
    """Check if Docker BuildKit is available.

    Returns:
        bool: True if BuildKit is available, False otherwise.
    """
    try:
        result = subprocess.run(
            ["docker", "buildx", "version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _find_available_port(start: int = 18000, end: int = 19000) -> int:
    """Find an available port in the specified range.

    Args:
        start: Start of port range (inclusive).
        end: End of port range (exclusive).

    Returns:
        int: An available port number.

    Raises:
        RuntimeError: If no available port found in range.
    """
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    msg = f"No available port found in range {start}-{end}"
    raise RuntimeError(msg)


@pytest.fixture(scope="session")
def docker_available() -> bool:
    """Session-scoped fixture indicating Docker availability.

    Returns:
        bool: True if Docker daemon is available.
    """
    return _is_docker_available()


@pytest.fixture(scope="session")
def buildkit_available(docker_available: bool) -> bool:
    """Session-scoped fixture indicating BuildKit availability.

    Args:
        docker_available: Whether Docker is available.

    Returns:
        bool: True if BuildKit is available.
    """
    if not docker_available:
        return False
    return _is_buildkit_available()


@pytest.fixture(scope="session")
def skip_if_no_docker(docker_available: bool) -> None:
    """Skip test if Docker is not available.

    Args:
        docker_available: Whether Docker is available.
    """
    if not docker_available:
        pytest.skip("Docker daemon not available")


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return project root directory path.

    Returns:
        Path: Absolute path to project root.
    """
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def docker_dir() -> Path:
    """Return Docker configuration directory path.

    Returns:
        Path: Absolute path to Docker directory.
    """
    return DOCKER_DIR


@pytest.fixture(scope="session")
def dockerfile_path() -> Path:
    """Return Dockerfile path.

    Returns:
        Path: Absolute path to Dockerfile.runtime.
    """
    return DOCKERFILE_PATH


@pytest.fixture(scope="session")
def compose_file_path() -> Path:
    """Return docker-compose file path.

    Returns:
        Path: Absolute path to docker-compose.runtime.yml.
    """
    return COMPOSE_FILE_PATH


@pytest.fixture
def available_port() -> int:
    """Find an available port for test containers.

    Returns:
        int: Available port number.
    """
    return _find_available_port()


@pytest.fixture(scope="module")
def test_image_name() -> str:
    """Generate unique test image name.

    Returns:
        str: Unique image name for testing.
    """
    return f"omnibase-infra-test:{os.getpid()}"


@pytest.fixture(scope="module")
def built_test_image(
    docker_available: bool,
    project_root: Path,
    test_image_name: str,
) -> Generator[str, None, None]:
    """Build test Docker image with cleanup.

    This fixture builds the Docker image once per test module and
    cleans up after all tests in the module complete.

    Args:
        docker_available: Whether Docker is available.
        project_root: Project root directory.
        test_image_name: Name for the test image.

    Yields:
        str: Name of the built image.
    """
    if not docker_available:
        pytest.skip("Docker daemon not available")

    # Build image without GitHub token (tests public dependencies only)
    build_cmd = [
        "docker",
        "build",
        "-f",
        str(DOCKERFILE_PATH),
        "-t",
        test_image_name,
        "--build-arg",
        "RUNTIME_VERSION=test",
        str(project_root),
    ]

    env = os.environ.copy()
    env["DOCKER_BUILDKIT"] = "1"

    result = subprocess.run(
        build_cmd,
        capture_output=True,
        text=True,
        timeout=600,  # 10 minute timeout for build
        env=env,
        check=False,
    )

    if result.returncode != 0:
        pytest.fail(f"Docker build failed: {result.stderr}")

    yield test_image_name

    # Cleanup: remove test image
    subprocess.run(
        ["docker", "rmi", "-f", test_image_name],
        capture_output=True,
        timeout=60,
        check=False,
    )


@contextmanager
def run_container(
    image: str,
    name: str,
    *,
    ports: dict[str, int] | None = None,
    environment: dict[str, str] | None = None,
    command: list[str] | None = None,
    detach: bool = True,
    remove_on_exit: bool = True,
) -> Iterator[str]:
    """Context manager for running a container with automatic cleanup.

    Args:
        image: Docker image name/tag.
        name: Container name.
        ports: Port mappings (container_port: host_port).
        environment: Environment variables.
        command: Override container command.
        detach: Run in detached mode.
        remove_on_exit: Remove container on exit.

    Yields:
        str: Container ID.
    """
    container_id = None
    run_cmd = ["docker", "run"]

    if detach:
        run_cmd.append("-d")

    run_cmd.extend(["--name", name])

    if ports:
        for container_port, host_port in ports.items():
            run_cmd.extend(["-p", f"{host_port}:{container_port}"])

    if environment:
        for key, value in environment.items():
            run_cmd.extend(["-e", f"{key}={value}"])

    run_cmd.append(image)

    if command:
        run_cmd.extend(command)

    try:
        result = subprocess.run(
            run_cmd,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

        if result.returncode != 0:
            msg = f"Failed to start container: {result.stderr}"
            raise RuntimeError(msg)

        container_id = result.stdout.strip()
        yield container_id

    finally:
        if container_id or name:
            # Stop container
            subprocess.run(
                ["docker", "stop", name],
                capture_output=True,
                timeout=30,
                check=False,
            )
            if remove_on_exit:
                # Remove container
                subprocess.run(
                    ["docker", "rm", "-f", name],
                    capture_output=True,
                    timeout=30,
                    check=False,
                )


@pytest.fixture
def container_runner() -> Callable[..., AbstractContextManager[str]]:
    """Provide the run_container context manager.

    Returns:
        The run_container context manager function.
    """
    return run_container


def wait_for_healthy(
    container_name: str,
    timeout: int = 60,
    interval: int = 2,
) -> bool:
    """Wait for container to become healthy.

    Args:
        container_name: Name of the container.
        timeout: Maximum wait time in seconds.
        interval: Check interval in seconds.

    Returns:
        bool: True if container became healthy, False if timeout.
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{.State.Health.Status}}",
                container_name,
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        if result.returncode == 0:
            status = result.stdout.strip()
            if status == "healthy":
                return True
            if status == "unhealthy":
                return False

        time.sleep(interval)

    return False


def wait_for_log_message(
    container_name: str,
    message: str,
    timeout: int = 30,
) -> bool:
    """Wait for a specific message in container logs.

    Args:
        container_name: Name of the container.
        message: Message to search for in logs.
        timeout: Maximum wait time in seconds.

    Returns:
        bool: True if message found, False if timeout.
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        result = subprocess.run(
            ["docker", "logs", container_name],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        if message in result.stdout or message in result.stderr:
            return True

        time.sleep(1)

    return False


@pytest.fixture
def wait_for_healthy_fixture() -> Callable[[str, int, int], bool]:
    """Provide the wait_for_healthy function.

    Returns:
        The wait_for_healthy function.
    """
    return wait_for_healthy


@pytest.fixture
def wait_for_log_message_fixture() -> Callable[[str, str, int], bool]:
    """Provide the wait_for_log_message function.

    Returns:
        The wait_for_log_message function.
    """
    return wait_for_log_message
