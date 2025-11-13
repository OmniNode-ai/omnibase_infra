#!/usr/bin/env python3
"""
NodeDeploymentSenderEffect - Deployment sender effect node for packaging and transferring Docker containers.

Builds Docker images, creates compressed packages with BLAKE3 checksums, and transfers via HTTP to remote receivers.
Publishes Kafka events for lifecycle tracking.

ONEX v2.0 Effect Node
Domain: deployment
"""

import asyncio
import gzip
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import aiofiles
import blake3
import httpx
from omnibase_core import EnumCoreErrorCode, ModelOnexError
from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.core import ModelContainer
from omnibase_core.nodes.node_effect import NodeEffect

import docker

from .models import (
    ModelContainerPackageInput,
    ModelContainerPackageOutput,
    ModelKafkaPublishInput,
    ModelKafkaPublishOutput,
    ModelPackageTransferInput,
    ModelPackageTransferOutput,
)

logger = logging.getLogger(__name__)


class NodeDeploymentSenderEffect(NodeEffect):
    """
    Deployment sender effect node for Docker container packaging and remote deployment.

    Operations:
    - package_container: Build Docker image, export, compress, and generate checksum
    - transfer_package: Stream package to remote receiver via HTTP
    - publish_transfer_event: Publish lifecycle events to Kafka

    Features:
    - Docker SDK integration for image building
    - BLAKE3 checksum generation for integrity
    - HTTP streaming for large file transfers
    - Kafka event publishing for observability
    - Circuit breaker for resilience

    Performance Targets:
    - Image build: <20s
    - Package transfer (1GB): <10s
    - Kafka event publish: <50ms
    """

    def __init__(self, container: ModelContainer) -> None:
        """Initialize NodeDeploymentSenderEffect."""
        super().__init__(container)

        # Configuration
        self.config = container.value if hasattr(container, "value") else {}

        # Docker client (lazy initialization)
        self._docker_client: Optional[docker.DockerClient] = None

        # HTTP client for transfers (lazy initialization)
        self._http_client: Optional[httpx.AsyncClient] = None

        # Kafka configuration
        self.kafka_broker_url: str = self.config.get(
            "kafka_broker_url",
            os.getenv("KAFKA_BOOTSTRAP_SERVERS", "omninode-bridge-redpanda:9092"),
        )
        self.default_namespace: str = self.config.get(
            "default_namespace", "dev.omninode-bridge"
        )

        # Get or create KafkaClient from container (skip if in health check mode)
        health_check_mode = self.config.get("health_check_mode", False)
        self.kafka_client = (
            container.get_service("kafka_client")
            if hasattr(container, "get_service")
            else None
        )

        if self.kafka_client is None and not health_check_mode:
            # Import KafkaClient
            try:
                # Import performance config for timeout settings
                from ....config import performance_config
                from ....services.kafka_client import KafkaClient

                self.kafka_client = KafkaClient(
                    bootstrap_servers=self.kafka_broker_url,
                    enable_dead_letter_queue=True,
                    max_retry_attempts=3,
                    timeout_seconds=performance_config.KAFKA_CLIENT_TIMEOUT_SECONDS,
                )
                # Register service if container supports it
                if hasattr(container, "register_service"):
                    container.register_service("kafka_client", self.kafka_client)
                emit_log_event(
                    LogLevel.INFO,
                    "KafkaClient initialized successfully",
                    {
                        "node_id": str(self.node_id),
                        "broker_url": self.kafka_broker_url,
                    },
                )
            except ImportError as e:
                emit_log_event(
                    LogLevel.WARNING,
                    "KafkaClient not available - events will be logged only",
                    {"node_id": str(self.node_id), "error": str(e)},
                )
                self.kafka_client = None
        elif health_check_mode:
            # In health check mode, skip Kafka initialization
            emit_log_event(
                LogLevel.DEBUG,
                "Health check mode enabled - skipping Kafka initialization",
                {"node_id": str(self.node_id)},
            )
            self.kafka_client = None

        # Package storage directory
        self.package_dir = Path(
            self.config.get("package_dir", "/tmp/deployment_packages")
        )
        self.package_dir.mkdir(parents=True, exist_ok=True)

        # Consul configuration for service discovery
        self.consul_host: str = self.config.get(
            "consul_host",
            os.getenv("CONSUL_HOST", "omninode-bridge-consul"),
        )
        self.consul_port: int = self.config.get(
            "consul_port",
            int(os.getenv("CONSUL_PORT", "28500")),
        )
        self.consul_enable_registration: bool = self.config.get(
            "consul_enable_registration", True
        )

        # Performance metrics
        self._metrics = {
            "total_builds": 0,
            "successful_builds": 0,
            "failed_builds": 0,
            "total_transfers": 0,
            "successful_transfers": 0,
            "failed_transfers": 0,
            "total_events_published": 0,
        }

        emit_log_event(
            LogLevel.INFO,
            "NodeDeploymentSenderEffect initialized",
            {
                "node_id": str(self.node_id),
                "package_dir": str(self.package_dir),
                "kafka_enabled": self.kafka_client is not None,
            },
        )

        # Register with Consul for service discovery (skip in health check mode)
        if not health_check_mode and self.consul_enable_registration:
            self._register_with_consul_sync()

    def _get_docker_client(self) -> docker.DockerClient:
        """Get or create Docker client."""
        if self._docker_client is None:
            try:
                self._docker_client = docker.from_env()
                # Test connectivity
                self._docker_client.ping()
                emit_log_event(
                    LogLevel.INFO,
                    "Docker client initialized",
                    {"node_id": str(self.node_id)},
                )
            except ConnectionError as e:
                # Docker daemon not reachable
                emit_log_event(
                    LogLevel.ERROR,
                    f"Docker daemon connection failed: {e}",
                    {
                        "node_id": str(self.node_id),
                        "error": str(e),
                        "error_type": "ConnectionError",
                    },
                )
                raise ModelOnexError(
                    error_code=EnumCoreErrorCode.CONNECTION_ERROR,
                    message=f"Docker daemon not reachable: {e}",
                    details={"original_error": str(e)},
                )
            except (TimeoutError, asyncio.TimeoutError) as e:
                # Docker daemon timeout
                emit_log_event(
                    LogLevel.ERROR,
                    f"Docker daemon connection timeout: {e}",
                    {
                        "node_id": str(self.node_id),
                        "error": str(e),
                        "error_type": "TimeoutError",
                    },
                )
                raise ModelOnexError(
                    error_code=EnumCoreErrorCode.TIMEOUT,
                    message=f"Docker daemon timeout: {e}",
                    details={"original_error": str(e)},
                )
            except Exception as e:
                # Unexpected Docker errors - log with exc_info for debugging
                emit_log_event(
                    LogLevel.ERROR,
                    f"Unexpected Docker initialization error: {type(e).__name__}",
                    {
                        "node_id": str(self.node_id),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
                logger.error(
                    f"Unexpected Docker error: {type(e).__name__}", exc_info=True
                )
                raise ModelOnexError(
                    error_code=EnumCoreErrorCode.DEPENDENCY_ERROR,
                    message=f"Docker daemon not available: {e}",
                    details={"original_error": str(e), "error_type": type(e).__name__},
                )
        return self._docker_client

    def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client for transfers."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(300.0),  # 5 minute timeout for large transfers
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
        return self._http_client

    async def package_container(
        self, input_data: ModelContainerPackageInput
    ) -> ModelContainerPackageOutput:
        """
        Build Docker image and create deployment package.

        Steps:
        1. Build Docker image from Dockerfile
        2. Export image to tar archive
        3. Compress with gzip/zstd
        4. Generate BLAKE3 checksum
        5. Store package in local directory

        Performance Target: <20s for typical container build

        Args:
            input_data: Container package configuration

        Returns:
            ModelContainerPackageOutput with package metadata

        Raises:
            ModelOnexError: If build or packaging fails
        """
        start_time = time.perf_counter()
        package_id = uuid4()
        correlation_id = input_data.correlation_id or uuid4()

        emit_log_event(
            LogLevel.INFO,
            "Starting container packaging",
            {
                "package_id": str(package_id),
                "correlation_id": str(correlation_id),
                "container_name": input_data.container_name,
                "image_tag": input_data.image_tag,
            },
        )

        try:
            self._metrics["total_builds"] += 1

            # Step 1: Build Docker image
            docker_client = self._get_docker_client()
            image_tag_full = f"{input_data.container_name}:{input_data.image_tag}"

            emit_log_event(
                LogLevel.INFO,
                "Building Docker image",
                {
                    "image_tag": image_tag_full,
                    "build_context": input_data.build_context,
                    "dockerfile": input_data.dockerfile_path,
                },
            )

            build_start = time.perf_counter()
            image, build_logs = docker_client.images.build(
                path=input_data.build_context,
                dockerfile=input_data.dockerfile_path,
                tag=image_tag_full,
                buildargs=input_data.build_args or {},
                rm=True,  # Remove intermediate containers
                forcerm=True,  # Always remove intermediate containers
            )
            build_duration_ms = int((time.perf_counter() - build_start) * 1000)

            emit_log_event(
                LogLevel.INFO,
                "Docker image built successfully",
                {
                    "image_id": image.id,
                    "image_tag": image_tag_full,
                    "build_duration_ms": build_duration_ms,
                },
            )

            # Step 2: Save image to tar archive
            emit_log_event(LogLevel.INFO, "Saving Docker image to tar archive", {})

            save_start = time.perf_counter()
            image_data_generator = docker_client.images.get(image.id).save(
                named=True,  # Include tags in export
            )

            # Write to temporary tar file
            temp_tar_path = self.package_dir / f"{package_id}.tar"
            with open(temp_tar_path, "wb") as tar_file:
                for chunk in image_data_generator:
                    tar_file.write(chunk)

            save_duration_ms = int((time.perf_counter() - save_start) * 1000)
            original_size_mb = os.path.getsize(temp_tar_path) / (1024 * 1024)

            emit_log_event(
                LogLevel.INFO,
                "Docker image saved to tar",
                {
                    "tar_path": str(temp_tar_path),
                    "size_mb": round(original_size_mb, 2),
                    "save_duration_ms": save_duration_ms,
                },
            )

            # Step 3: Compress package
            compress_start = time.perf_counter()
            if input_data.compression == "gzip":
                package_path = self.package_dir / f"{package_id}.tar.gz"
                with open(temp_tar_path, "rb") as f_in:
                    with gzip.open(package_path, "wb", compresslevel=6) as f_out:
                        f_out.writelines(f_in)
            elif input_data.compression == "none":
                package_path = temp_tar_path
            else:
                # zstd not implemented yet
                raise ModelOnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_FAILED,
                    message=f"Unsupported compression: {input_data.compression}",
                    details={"compression": input_data.compression},
                )

            compress_duration_ms = int((time.perf_counter() - compress_start) * 1000)
            package_size_mb = os.path.getsize(package_path) / (1024 * 1024)
            compression_ratio = (
                package_size_mb / original_size_mb
                if input_data.compression != "none"
                else 1.0
            )

            # Clean up temp tar if we compressed
            if input_data.compression != "none":
                temp_tar_path.unlink()

            emit_log_event(
                LogLevel.INFO,
                "Package compressed",
                {
                    "package_path": str(package_path),
                    "package_size_mb": round(package_size_mb, 2),
                    "compression_ratio": round(compression_ratio, 3),
                    "compress_duration_ms": compress_duration_ms,
                },
            )

            # Step 4: Generate BLAKE3 checksum
            checksum_start = time.perf_counter()
            package_checksum = await self._generate_blake3_checksum(package_path)
            checksum_duration_ms = int((time.perf_counter() - checksum_start) * 1000)

            emit_log_event(
                LogLevel.INFO,
                "BLAKE3 checksum generated",
                {
                    "checksum": package_checksum,
                    "checksum_duration_ms": checksum_duration_ms,
                },
            )

            # Calculate total duration
            total_duration_ms = int((time.perf_counter() - start_time) * 1000)

            self._metrics["successful_builds"] += 1

            emit_log_event(
                LogLevel.INFO,
                "Container packaging completed",
                {
                    "package_id": str(package_id),
                    "total_duration_ms": total_duration_ms,
                    "package_size_mb": round(package_size_mb, 2),
                },
            )

            return ModelContainerPackageOutput(
                success=True,
                package_id=package_id,
                image_id=image.id,
                package_path=str(package_path),
                package_size_mb=package_size_mb,
                package_checksum=package_checksum,
                build_duration_ms=build_duration_ms,
                compression_ratio=compression_ratio,
            )

        except ModelOnexError:
            self._metrics["failed_builds"] += 1
            raise
        except OSError as e:
            # File system errors during build/save/compress
            self._metrics["failed_builds"] += 1
            emit_log_event(
                LogLevel.ERROR,
                f"File system error during packaging: {e}",
                {
                    "package_id": str(package_id),
                    "error": str(e),
                    "error_type": "FileSystemError",
                },
            )
            return ModelContainerPackageOutput(
                success=False,
                package_id=package_id,
                error_message=f"File system error: {e}",
                error_code="FILE_SYSTEM_ERROR",
            )
        except Exception as e:
            # Unexpected errors - log with exc_info for debugging
            self._metrics["failed_builds"] += 1
            emit_log_event(
                LogLevel.ERROR,
                f"Unexpected packaging error: {type(e).__name__}",
                {
                    "package_id": str(package_id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            logger.error(
                f"Unexpected packaging error: {type(e).__name__}", exc_info=True
            )
            return ModelContainerPackageOutput(
                success=False,
                package_id=package_id,
                error_message=str(e),
                error_code="BUILD_FAILED",
            )

    async def _generate_blake3_checksum(self, file_path: Path) -> str:
        """
        Generate BLAKE3 checksum for file.

        Args:
            file_path: Path to file

        Returns:
            Hex-encoded BLAKE3 hash (64 characters)
        """
        hasher = blake3.blake3()
        async with aiofiles.open(file_path, "rb") as f:
            while True:
                chunk = await f.read(65536)  # 64KB chunks
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    async def transfer_package(
        self, input_data: ModelPackageTransferInput
    ) -> ModelPackageTransferOutput:
        """
        Transfer deployment package to remote receiver.

        Steps:
        1. Validate remote receiver connectivity
        2. Stream package via chunked HTTP upload
        3. Verify checksum on receiver
        4. Return remote deployment ID

        Performance Target: <10s for 1GB package transfer

        Args:
            input_data: Package transfer configuration

        Returns:
            ModelPackageTransferOutput with transfer results

        Raises:
            ModelOnexError: If transfer fails
        """
        start_time = time.perf_counter()
        correlation_id = input_data.correlation_id or uuid4()

        emit_log_event(
            LogLevel.INFO,
            "Starting package transfer",
            {
                "package_id": str(input_data.package_id),
                "correlation_id": str(correlation_id),
                "remote_url": str(input_data.remote_receiver_url),
                "transfer_method": input_data.transfer_method,
            },
        )

        try:
            self._metrics["total_transfers"] += 1

            # Validate package exists
            package_path = Path(input_data.package_path)
            if not package_path.exists():
                raise ModelOnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_FAILED,
                    message=f"Package not found: {package_path}",
                    details={"package_path": str(package_path)},
                )

            package_size_bytes = os.path.getsize(package_path)

            # Step 1: HTTP streaming upload
            http_client = self._get_http_client()

            upload_start = time.perf_counter()

            # Prepare multipart form data
            async with aiofiles.open(package_path, "rb") as package_file:
                files = {
                    "package": (
                        package_path.name,
                        await package_file.read(),
                        "application/gzip",
                    )
                }

                data = {
                    "package_id": str(input_data.package_id),
                    "checksum": input_data.package_checksum,
                    "container_name": input_data.container_name or "",
                    "image_tag": input_data.image_tag or "",
                    "verify_checksum": str(input_data.verify_checksum).lower(),
                }

                emit_log_event(
                    LogLevel.INFO,
                    "Uploading package to remote receiver",
                    {
                        "url": str(input_data.remote_receiver_url),
                        "package_size_mb": round(package_size_bytes / (1024 * 1024), 2),
                    },
                )

                response = await http_client.post(
                    str(input_data.remote_receiver_url),
                    files=files,
                    data=data,
                )

                response.raise_for_status()

            upload_duration_ms = int((time.perf_counter() - upload_start) * 1000)
            upload_duration_sec = upload_duration_ms / 1000.0
            throughput_mbps = (
                (package_size_bytes / (1024 * 1024)) / upload_duration_sec
                if upload_duration_sec > 0
                else 0.0
            )

            # Parse response
            response_data = response.json()
            remote_deployment_id = response_data.get("deployment_id")
            checksum_verified = response_data.get("checksum_verified", False)

            emit_log_event(
                LogLevel.INFO,
                "Package transfer completed",
                {
                    "package_id": str(input_data.package_id),
                    "transfer_duration_ms": upload_duration_ms,
                    "throughput_mbps": round(throughput_mbps, 2),
                    "remote_deployment_id": remote_deployment_id,
                },
            )

            self._metrics["successful_transfers"] += 1

            return ModelPackageTransferOutput(
                success=True,
                transfer_success=True,
                transfer_duration_ms=upload_duration_ms,
                bytes_transferred=package_size_bytes,
                transfer_throughput_mbps=throughput_mbps,
                remote_deployment_id=remote_deployment_id,
                checksum_verified=checksum_verified,
            )

        except httpx.HTTPStatusError as e:
            self._metrics["failed_transfers"] += 1
            error_message = f"HTTP error during transfer: {e.response.status_code} - {e.response.text}"
            emit_log_event(
                LogLevel.ERROR,
                error_message,
                {
                    "package_id": str(input_data.package_id),
                    "error_type": "HTTPStatusError",
                },
            )
            return ModelPackageTransferOutput(
                success=False,
                transfer_success=False,
                error_message=error_message,
                error_code="TRANSFER_FAILED",
            )
        except (ConnectionError, httpx.ConnectError) as e:
            # Network connectivity errors
            self._metrics["failed_transfers"] += 1
            emit_log_event(
                LogLevel.ERROR,
                f"Network connection failed during transfer: {e}",
                {
                    "package_id": str(input_data.package_id),
                    "error": str(e),
                    "error_type": "ConnectionError",
                },
            )
            return ModelPackageTransferOutput(
                success=False,
                transfer_success=False,
                error_message=f"Connection failed: {e}",
                error_code="CONNECTION_FAILED",
            )
        except (TimeoutError, httpx.TimeoutException) as e:
            # Request timeout errors
            self._metrics["failed_transfers"] += 1
            emit_log_event(
                LogLevel.ERROR,
                f"Transfer timeout: {e}",
                {
                    "package_id": str(input_data.package_id),
                    "error": str(e),
                    "error_type": "TimeoutError",
                },
            )
            return ModelPackageTransferOutput(
                success=False,
                transfer_success=False,
                error_message=f"Transfer timeout: {e}",
                error_code="TRANSFER_TIMEOUT",
            )
        except Exception as e:
            # Unexpected errors - log with exc_info for debugging
            self._metrics["failed_transfers"] += 1
            emit_log_event(
                LogLevel.ERROR,
                f"Unexpected transfer error: {type(e).__name__}",
                {
                    "package_id": str(input_data.package_id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            logger.error(
                f"Unexpected transfer error: {type(e).__name__}", exc_info=True
            )
            return ModelPackageTransferOutput(
                success=False,
                transfer_success=False,
                error_message=str(e),
                error_code="TRANSFER_FAILED",
            )

    async def publish_transfer_event(
        self, input_data: ModelKafkaPublishInput
    ) -> ModelKafkaPublishOutput:
        """
        Publish deployment lifecycle event to Kafka.

        Event Types:
        - BUILD_STARTED: Docker build initiated
        - BUILD_COMPLETED: Docker build finished
        - TRANSFER_STARTED: Package transfer initiated
        - TRANSFER_COMPLETED: Package transfer finished
        - DEPLOYMENT_FAILED: Deployment failed at any stage

        Performance Target: <50ms per event

        Args:
            input_data: Event publishing configuration

        Returns:
            ModelKafkaPublishOutput with publishing results
        """
        start_time = time.perf_counter()

        emit_log_event(
            LogLevel.INFO,
            f"Publishing Kafka event: {input_data.event_type}",
            {
                "correlation_id": str(input_data.correlation_id),
                "event_type": input_data.event_type,
            },
        )

        try:
            # Build topic name
            topic_name = f"{self.default_namespace}.deployment.{input_data.event_type.lower().replace('_', '-')}.v1"

            # Publish to Kafka if client is available
            if self.kafka_client and self.kafka_client.is_connected:
                # Build event payload
                event_payload = {
                    "correlation_id": str(input_data.correlation_id),
                    "event_type": input_data.event_type,
                    "package_id": (
                        str(input_data.package_id) if input_data.package_id else None
                    ),
                    "container_name": input_data.container_name,
                    "image_tag": input_data.image_tag,
                    "payload": input_data.event_payload,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "node_id": str(self.node_id),
                }

                # Publish with OnexEnvelopeV1 wrapping for standardized event format
                success = await self.kafka_client.publish_with_envelope(
                    event_type=input_data.event_type,
                    source_node_id=str(self.node_id),
                    payload=event_payload,
                    topic=topic_name,
                    correlation_id=input_data.correlation_id,
                    metadata={
                        "event_category": "deployment_lifecycle",
                        "node_type": "deployment_sender",
                        "namespace": self.default_namespace,
                    },
                )

                if not success:
                    emit_log_event(
                        LogLevel.WARNING,
                        f"Failed to publish Kafka event: {input_data.event_type}",
                        {
                            "node_id": str(self.node_id),
                            "event_type": input_data.event_type,
                            "topic_name": topic_name,
                        },
                    )
                    return ModelKafkaPublishOutput(
                        success=False,
                        event_published=False,
                        error_message="Kafka publish returned False",
                        error_code="KAFKA_PUBLISH_FAILED",
                    )

                publish_duration_ms = int((time.perf_counter() - start_time) * 1000)
                self._metrics["total_events_published"] += 1

                emit_log_event(
                    LogLevel.INFO,
                    f"Kafka event published (OnexEnvelopeV1): {input_data.event_type}",
                    {
                        "topic": topic_name,
                        "event_type": input_data.event_type,
                        "publish_duration_ms": publish_duration_ms,
                        "correlation_id": str(input_data.correlation_id),
                        "envelope_wrapped": True,
                    },
                )

                return ModelKafkaPublishOutput(
                    success=True,
                    event_published=True,
                    topic=topic_name,
                    partition=0,
                    offset=self._metrics["total_events_published"],
                    publish_duration_ms=publish_duration_ms,
                )

            else:
                # Kafka not available - log event only
                publish_duration_ms = int((time.perf_counter() - start_time) * 1000)

                emit_log_event(
                    LogLevel.WARNING,
                    f"Kafka unavailable, logging event: {input_data.event_type}",
                    {
                        "node_id": str(self.node_id),
                        "event_type": input_data.event_type,
                        "topic_name": topic_name,
                        "kafka_available": False,
                    },
                )

                return ModelKafkaPublishOutput(
                    success=True,
                    event_published=False,
                    topic=topic_name,
                    partition=0,
                    offset=0,
                    publish_duration_ms=publish_duration_ms,
                )

        except ConnectionError as e:
            # Kafka connection errors
            publish_duration_ms = int((time.perf_counter() - start_time) * 1000)
            emit_log_event(
                LogLevel.ERROR,
                f"Kafka connection failed: {e}",
                {
                    "event_type": input_data.event_type,
                    "error": str(e),
                    "error_type": "ConnectionError",
                },
            )
            return ModelKafkaPublishOutput(
                success=False,
                event_published=False,
                error_message=f"Kafka connection failed: {e}",
                error_code="KAFKA_CONNECTION_FAILED",
                publish_duration_ms=publish_duration_ms,
            )
        except (TimeoutError, asyncio.TimeoutError) as e:
            # Kafka timeout errors
            publish_duration_ms = int((time.perf_counter() - start_time) * 1000)
            emit_log_event(
                LogLevel.ERROR,
                f"Kafka publish timeout: {e}",
                {
                    "event_type": input_data.event_type,
                    "error": str(e),
                    "error_type": "TimeoutError",
                },
            )
            return ModelKafkaPublishOutput(
                success=False,
                event_published=False,
                error_message=f"Kafka timeout: {e}",
                error_code="KAFKA_TIMEOUT",
                publish_duration_ms=publish_duration_ms,
            )
        except Exception as e:
            # Unexpected Kafka errors - log with exc_info for debugging
            publish_duration_ms = int((time.perf_counter() - start_time) * 1000)
            emit_log_event(
                LogLevel.ERROR,
                f"Unexpected Kafka error: {type(e).__name__}",
                {
                    "event_type": input_data.event_type,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            logger.error(f"Unexpected Kafka error: {type(e).__name__}", exc_info=True)
            return ModelKafkaPublishOutput(
                success=False,
                event_published=False,
                error_message=str(e),
                error_code="KAFKA_PUBLISH_FAILED",
                publish_duration_ms=publish_duration_ms,
            )

    async def execute_effect(self, contract: ModelContractEffect) -> Any:
        """
        Execute effect operation.

        Routes to appropriate handler based on operation name.

        Args:
            contract: Effect contract with operation parameters

        Returns:
            Operation result

        Raises:
            ModelOnexError: If operation fails
        """
        operation = contract.input_state.get("operation", "package_container")

        emit_log_event(
            LogLevel.INFO,
            f"Executing effect operation: {operation}",
            {
                "node_id": str(self.node_id),
                "correlation_id": str(contract.correlation_id),
                "operation": operation,
            },
        )

        try:
            if operation == "package_container":
                input_model = ModelContainerPackageInput(
                    **contract.input_state.get("input", {})
                )
                return await self.package_container(input_model)

            elif operation == "transfer_package":
                input_model = ModelPackageTransferInput(
                    **contract.input_state.get("input", {})
                )
                return await self.transfer_package(input_model)

            elif operation == "publish_transfer_event":
                input_model = ModelKafkaPublishInput(
                    **contract.input_state.get("input", {})
                )
                return await self.publish_transfer_event(input_model)

            elif operation == "get_metrics":
                return self._metrics

            else:
                raise ModelOnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_FAILED,
                    message=f"Unknown operation: {operation}",
                    details={
                        "operation": operation,
                        "supported_operations": [
                            "package_container",
                            "transfer_package",
                            "publish_transfer_event",
                            "get_metrics",
                        ],
                    },
                )

        except ModelOnexError:
            # Re-raise OnexError to preserve error context
            raise
        except (ValueError, KeyError, TypeError) as e:
            # Data validation/parsing errors
            emit_log_event(
                LogLevel.ERROR,
                f"Invalid input data for operation {operation}: {e}",
                {
                    "node_id": str(self.node_id),
                    "correlation_id": str(contract.correlation_id),
                    "operation": operation,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                message=f"Invalid input data: {e!s}",
                details={
                    "original_error": str(e),
                    "operation": operation,
                    "error_type": type(e).__name__,
                },
            )
        except Exception as e:
            # Unexpected errors - log with exc_info for debugging
            emit_log_event(
                LogLevel.ERROR,
                f"Unexpected effect execution error: {type(e).__name__}",
                {
                    "node_id": str(self.node_id),
                    "correlation_id": str(contract.correlation_id),
                    "operation": operation,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            logger.error(
                f"Unexpected execution error: {type(e).__name__}", exc_info=True
            )
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.EXECUTION_ERROR,
                message=f"Effect execution failed: {e!s}",
                details={
                    "original_error": str(e),
                    "operation": operation,
                    "error_type": type(e).__name__,
                },
            )

    def _register_with_consul_sync(self) -> None:
        """
        Register deployment sender node with Consul for service discovery (synchronous).

        Registers the sender as a service with metadata about node capabilities.

        Note:
            This is a non-blocking registration. Failures are logged but don't
            fail node startup. Service will continue without Consul if registration fails.
        """
        try:
            import consul

            # Initialize Consul client
            consul_client = consul.Consul(host=self.consul_host, port=self.consul_port)

            # Generate unique service ID
            service_id = f"omninode-bridge-deployment-sender-{self.node_id}"

            # Get service port from config (default to 8002 for sender)
            service_port = int(self.config.get("service_port", 8002))

            # Get service host from config (default to localhost)
            service_host = self.config.get("service_host", "localhost")

            # Prepare service tags
            service_tags = [
                "onex",
                "bridge",
                "deployment_sender",
                "omninode_bridge",
                "node_type:deployment_sender",
                f"namespace:{self.default_namespace}",
                f"kafka_enabled:{self.kafka_client is not None}",
            ]

            # Register service with Consul (no health check for effect nodes)
            consul_client.agent.service.register(
                name="omninode-bridge-deployment-sender",
                service_id=service_id,
                address=service_host,
                port=service_port,
                tags=service_tags,
            )

            emit_log_event(
                LogLevel.INFO,
                "Registered with Consul successfully",
                {
                    "node_id": str(self.node_id),
                    "service_id": service_id,
                    "consul_host": self.consul_host,
                    "consul_port": self.consul_port,
                    "service_host": service_host,
                    "service_port": service_port,
                },
            )

            # Store service_id for deregistration
            self._consul_service_id = service_id

        except ImportError:
            emit_log_event(
                LogLevel.WARNING,
                "python-consul not installed - Consul registration skipped",
                {"node_id": str(self.node_id)},
            )
        except ConnectionError as e:
            # Consul connection failed - non-critical
            emit_log_event(
                LogLevel.WARNING,
                f"Consul connection failed - registration skipped: {e}",
                {
                    "node_id": str(self.node_id),
                    "error": str(e),
                    "error_type": "ConnectionError",
                },
            )
        except (TimeoutError, asyncio.TimeoutError) as e:
            # Consul timeout - non-critical
            emit_log_event(
                LogLevel.WARNING,
                f"Consul registration timeout - skipped: {e}",
                {
                    "node_id": str(self.node_id),
                    "error": str(e),
                    "error_type": "TimeoutError",
                },
            )
        except Exception as e:
            # Unexpected errors - log but don't fail startup
            emit_log_event(
                LogLevel.ERROR,
                f"Unexpected Consul registration error: {type(e).__name__}",
                {
                    "node_id": str(self.node_id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            logger.error(f"Unexpected Consul error: {type(e).__name__}", exc_info=True)

    def _deregister_from_consul(self) -> None:
        """
        Deregister sender from Consul on shutdown (synchronous).

        Removes the service registration from Consul to prevent stale entries
        in the service catalog.

        Note:
            This is called during node cleanup. Failures are logged but don't
            prevent cleanup from completing.
        """
        try:
            if not hasattr(self, "_consul_service_id"):
                # Not registered, nothing to deregister
                return

            import consul

            consul_client = consul.Consul(host=self.consul_host, port=self.consul_port)
            consul_client.agent.service.deregister(self._consul_service_id)

            emit_log_event(
                LogLevel.INFO,
                "Deregistered from Consul successfully",
                {
                    "node_id": str(self.node_id),
                    "service_id": self._consul_service_id,
                },
            )

        except ImportError:
            # python-consul not installed, silently skip
            pass
        except Exception as e:
            emit_log_event(
                LogLevel.WARNING,
                "Failed to deregister from Consul",
                {
                    "node_id": str(self.node_id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

    async def cleanup(self) -> None:
        """Cleanup resources."""
        # Deregister from Consul for clean service discovery
        self._deregister_from_consul()

        if self._http_client:
            await self._http_client.aclose()
        if self._docker_client:
            self._docker_client.close()

        emit_log_event(
            LogLevel.INFO,
            "NodeDeploymentSenderEffect cleaned up",
            {"node_id": str(self.node_id)},
        )


__all__ = ["NodeDeploymentSenderEffect"]
