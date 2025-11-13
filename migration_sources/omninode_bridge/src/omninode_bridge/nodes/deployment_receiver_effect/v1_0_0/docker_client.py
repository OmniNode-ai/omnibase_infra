#!/usr/bin/env python3
"""
Docker client wrapper with circuit breaker for deployment receiver effect node.
Provides resilient Docker operations with error handling and retry logic.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    from docker.errors import APIError, DockerException, ImageNotFound, NotFound
    from docker.models.containers import Container
    from docker.models.images import Image

    import docker
    from docker import DockerClient
except ImportError:
    docker = None  # Will raise error at runtime if used
    DockerClient = None
    Container = None
    Image = None
    DockerException = Exception
    APIError = Exception
    ImageNotFound = Exception
    NotFound = Exception

from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event

from .models.model_deployment import (
    EnumHealthStatus,
    ModelDeploymentConfig,
    ModelHealthCheckResult,
)


class DockerClientWrapper:
    """
    Docker client wrapper with circuit breaker and resilience patterns.

    Features:
    - Docker SDK integration
    - Circuit breaker for failure handling
    - Async-friendly operations (runs in thread pool)
    - Health monitoring
    - Container lifecycle management
    """

    def __init__(
        self, docker_host: str = "unix:///var/run/docker.sock", timeout: int = 60
    ) -> None:
        """
        Initialize Docker client wrapper.

        Args:
            docker_host: Docker daemon socket URL
            timeout: Operation timeout in seconds
        """
        if docker is None:
            raise ImportError(
                "docker library not installed. Install with: pip install docker"
            )

        self.docker_host = docker_host
        self.timeout = timeout
        self.client: Optional[DockerClient] = None

        # Circuit breaker state
        self.failure_count = 0
        self.max_failures = 5
        self.circuit_open = False
        self.last_failure_time: Optional[datetime] = None
        self.circuit_reset_timeout = 60  # seconds

        self._connect()

    def _connect(self) -> None:
        """Establish connection to Docker daemon."""
        try:
            self.client = docker.DockerClient(
                base_url=self.docker_host, timeout=self.timeout
            )

            # Test connection
            self.client.ping()

            emit_log_event(
                LogLevel.INFO,
                "Docker client connected",
                {"docker_host": self.docker_host, "version": self.client.version()},
            )

            # Reset circuit breaker on successful connection
            self.failure_count = 0
            self.circuit_open = False

        except Exception as e:
            emit_log_event(
                LogLevel.ERROR,
                f"Docker client connection failed: {e!s}",
                {"docker_host": self.docker_host, "error": str(e)},
            )
            self._record_failure()
            raise

    def _record_failure(self) -> None:
        """Record failure and potentially open circuit breaker."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.max_failures:
            self.circuit_open = True
            emit_log_event(
                LogLevel.ERROR,
                "Circuit breaker opened",
                {
                    "failure_count": self.failure_count,
                    "max_failures": self.max_failures,
                },
            )

    def _check_circuit_breaker(self) -> None:
        """Check circuit breaker state and reset if timeout elapsed."""
        if not self.circuit_open:
            return

        if self.last_failure_time:
            elapsed = (datetime.now() - self.last_failure_time).total_seconds()
            if elapsed >= self.circuit_reset_timeout:
                emit_log_event(
                    LogLevel.INFO,
                    "Circuit breaker reset timeout elapsed, attempting reconnect",
                    {"elapsed_seconds": elapsed},
                )
                self.circuit_open = False
                self.failure_count = 0
                self._connect()

        if self.circuit_open:
            raise RuntimeError(
                "Circuit breaker is open - Docker operations unavailable"
            )

    async def load_image(
        self, image_tar_path: str
    ) -> tuple[bool, Optional[str], Optional[str], str]:
        """
        Load Docker image from tar file.

        Args:
            image_tar_path: Path to Docker image tar file

        Returns:
            Tuple of (success, image_id, image_name, error_message)
        """
        self._check_circuit_breaker()

        path = Path(image_tar_path)
        if not path.exists():
            return False, None, None, f"Image tar file not found: {image_tar_path}"

        try:
            # Run blocking Docker operation in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._load_image_sync, image_tar_path
            )

            emit_log_event(
                LogLevel.INFO,
                "Docker image loaded successfully",
                {
                    "image_path": image_tar_path,
                    "image_id": result[1],
                    "image_name": result[2],
                },
            )

            return result

        except Exception as e:
            error_msg = f"Image load failed: {e!s}"
            emit_log_event(
                LogLevel.ERROR,
                error_msg,
                {"image_path": image_tar_path, "error": str(e)},
            )
            self._record_failure()
            return False, None, None, error_msg

    def _load_image_sync(
        self, image_tar_path: str
    ) -> tuple[bool, Optional[str], Optional[str], str]:
        """Synchronous image load operation."""
        with open(image_tar_path, "rb") as f:
            images = self.client.images.load(f.read())

            if images:
                image = images[0]
                image_id = image.id
                image_name = image.tags[0] if image.tags else None
                return True, image_id, image_name, ""
            else:
                return False, None, None, "No image loaded from tar file"

    async def deploy_container(
        self, config: ModelDeploymentConfig
    ) -> tuple[bool, Optional[str], Optional[str], Optional[str], str]:
        """
        Deploy Docker container with provided configuration.

        Args:
            config: Container deployment configuration

        Returns:
            Tuple of (success, container_id, container_short_id, container_url, error_message)
        """
        self._check_circuit_breaker()

        try:
            # Run blocking Docker operation in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._deploy_container_sync, config
            )

            if result[0]:  # success
                emit_log_event(
                    LogLevel.INFO,
                    "Container deployed successfully",
                    {
                        "container_id": result[1],
                        "container_name": config.container_name,
                        "image": config.image_name,
                    },
                )

            return result

        except Exception as e:
            error_msg = f"Container deployment failed: {e!s}"
            emit_log_event(
                LogLevel.ERROR,
                error_msg,
                {
                    "container_name": config.container_name,
                    "image": config.image_name,
                    "error": str(e),
                },
            )
            self._record_failure()
            return False, None, None, None, error_msg

    def _deploy_container_sync(
        self, config: ModelDeploymentConfig
    ) -> tuple[bool, Optional[str], Optional[str], Optional[str], str]:
        """Synchronous container deployment operation."""
        # Build container configuration
        container_config: dict[str, Any] = {
            "image": config.image_name,
            "name": config.container_name,
            "detach": True,
            "restart_policy": {"Name": config.restart_policy.value},
        }

        # Add port mappings
        if config.ports:
            container_config["ports"] = {
                f"{container_port}/tcp": host_port
                for container_port, host_port in config.ports.items()
            }

        # Add environment variables
        if config.environment_vars:
            container_config["environment"] = config.environment_vars

        # Add volume mounts
        if config.volumes:
            container_config["volumes"] = {
                vol.host_path: {"bind": vol.container_path, "mode": vol.mode}
                for vol in config.volumes
            }

        # Add resource limits
        if config.resource_limits:
            limits = config.resource_limits
            if limits.cpu_limit:
                container_config["nano_cpus"] = int(
                    float(limits.cpu_limit) * 1_000_000_000
                )
            if limits.memory_limit:
                # Parse memory limit (e.g., "512m", "1g")
                memory = limits.memory_limit.lower()
                if memory.endswith("k"):
                    container_config["mem_limit"] = int(memory[:-1]) * 1024
                elif memory.endswith("m"):
                    container_config["mem_limit"] = int(memory[:-1]) * 1024 * 1024
                elif memory.endswith("g"):
                    container_config["mem_limit"] = (
                        int(memory[:-1]) * 1024 * 1024 * 1024
                    )

        # Create and start container
        container: Container = self.client.containers.run(**container_config)

        container_id = container.id
        container_short_id = container.short_id

        # Build container URL if ports are mapped
        container_url = None
        if config.ports:
            # Use first mapped port
            first_port = list(config.ports.values())[0]
            container_url = f"http://localhost:{first_port}"

        return True, container_id, container_short_id, container_url, ""

    async def health_check(self, container_id: str) -> ModelHealthCheckResult:
        """
        Check container health status.

        Args:
            container_id: Docker container ID

        Returns:
            Health check result
        """
        self._check_circuit_breaker()

        try:
            # Run blocking Docker operation in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._health_check_sync, container_id
            )

            return result

        except Exception as e:
            emit_log_event(
                LogLevel.ERROR,
                f"Health check failed: {e!s}",
                {"container_id": container_id, "error": str(e)},
            )
            return ModelHealthCheckResult(
                is_healthy=False,
                status=EnumHealthStatus.UNHEALTHY,
                error_message=str(e),
            )

    def _health_check_sync(self, container_id: str) -> ModelHealthCheckResult:
        """Synchronous health check operation."""
        try:
            container: Container = self.client.containers.get(container_id)

            # Check container status
            container.reload()  # Refresh container state
            status = container.status

            if status == "running":
                # Container is running - consider healthy
                return ModelHealthCheckResult(
                    is_healthy=True,
                    status=EnumHealthStatus.HEALTHY,
                    checks_passed=1,
                    last_check_time=datetime.now().isoformat(),
                )
            elif status == "created":
                return ModelHealthCheckResult(
                    is_healthy=False,
                    status=EnumHealthStatus.NOT_STARTED,
                    error_message="Container created but not started",
                )
            elif status in ["restarting", "paused"]:
                return ModelHealthCheckResult(
                    is_healthy=False,
                    status=EnumHealthStatus.STARTING,
                    error_message=f"Container status: {status}",
                )
            else:
                return ModelHealthCheckResult(
                    is_healthy=False,
                    status=EnumHealthStatus.UNHEALTHY,
                    error_message=f"Container unhealthy status: {status}",
                )

        except NotFound:
            return ModelHealthCheckResult(
                is_healthy=False,
                status=EnumHealthStatus.NOT_STARTED,
                error_message=f"Container not found: {container_id}",
            )

    def close(self) -> None:
        """Close Docker client connection."""
        if self.client:
            try:
                self.client.close()
                emit_log_event(LogLevel.INFO, "Docker client closed", {})
            except Exception as e:
                emit_log_event(
                    LogLevel.WARNING,
                    f"Error closing Docker client: {e!s}",
                    {"error": str(e)},
                )


__all__ = ["DockerClientWrapper"]
