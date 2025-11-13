#!/usr/bin/env python3
"""
NodeDeploymentReceiverEffect - Receives and deploys Docker containers on remote systems.

ONEX v2.0 Effect Node
Domain: deployment_automation
Generated: 2025-10-25

Features:
- Docker image package reception with HMAC authentication
- BLAKE3 checksum validation
- IP whitelisting
- Docker SDK integration for image loading
- Container deployment with full configuration
- Health check monitoring
- Kafka event publishing for deployment lifecycle

Operations:
- receive_package: Receive and validate Docker image package
- load_image: Load Docker image into daemon
- deploy_container: Deploy container with configuration
- health_check: Verify container health
- publish_deployment_event: Publish Kafka deployment events
- full_deployment: Execute all steps in sequence
"""

import asyncio
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4

from omnibase_core import EnumCoreErrorCode, ModelOnexError
from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event
from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.nodes.node_effect import NodeEffect

# Import EventBus for event-driven coordination
from ....services.event_bus import EventBusService
from .docker_client import DockerClientWrapper
from .models import (
    ModelContainerDeployInput,
    ModelContainerDeployOutput,
    ModelDeploymentEventInput,
    ModelDeploymentEventOutput,
    ModelFullDeploymentInput,
    ModelFullDeploymentOutput,
    ModelHealthCheckInput,
    ModelHealthCheckOutput,
    ModelImageLoadInput,
    ModelImageLoadOutput,
    ModelPackageReceiveInput,
    ModelPackageReceiveOutput,
)
from .security_validator import SecurityValidator

logger = logging.getLogger(__name__)


class NodeDeploymentReceiverEffect(NodeEffect):
    """
    Deployment Receiver Effect Node.

    Receives Docker image packages with security validation,
    loads images into Docker daemon, deploys containers with configuration,
    monitors health, and publishes deployment events to Kafka.

    Security Features:
    - HMAC authentication with SHA256
    - BLAKE3 checksum validation
    - IP whitelisting with CIDR support
    - Sandbox execution with resource limits

    Performance Targets:
    - Image load: <3s
    - Container start: <2s
    - Health check: <1s
    - Total deployment: <8s

    Kafka Events Published:
    - DEPLOYMENT_STARTED
    - IMAGE_LOADED
    - CONTAINER_STARTED
    - HEALTH_CHECK_PASSED
    - DEPLOYMENT_COMPLETED
    - DEPLOYMENT_FAILED
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """
        Initialize deployment receiver effect node.

        Args:
            container: ModelONEXContainer for dependency injection
        """
        super().__init__(container)

        # Configuration from container
        self.config = container.value if hasattr(container, "value") else {}

        # Get configuration from environment
        docker_host = os.getenv("DOCKER_HOST", "unix:///var/run/docker.sock")

        # AUTH_SECRET_KEY is required for HMAC signature validation
        auth_secret_key = os.getenv("AUTH_SECRET_KEY")
        if not auth_secret_key:
            raise ValueError(
                "AUTH_SECRET_KEY environment variable must be set. "
                "Generate a secure key with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
            )

        allowed_ip_ranges = os.getenv(
            "ALLOWED_IP_RANGES", "192.168.86.0/24,10.0.0.0/8"
        ).split(",")

        # Initialize Docker client
        self.docker_client = DockerClientWrapper(docker_host=docker_host, timeout=60)

        # Initialize security validator
        self.security_validator = SecurityValidator(
            secret_key=auth_secret_key, allowed_ip_ranges=allowed_ip_ranges
        )

        # Get or create EventBus from container for event-driven coordination
        self.event_bus = container.get_service("event_bus")
        if self.event_bus is None:
            # Get or create KafkaClient from container
            kafka_client = container.get_service("kafka_client")
            if kafka_client:
                # Initialize EventBus service
                default_namespace = os.getenv("NAMESPACE", "dev")
                self.event_bus = EventBusService(
                    kafka_client=kafka_client,
                    node_id=str(self.node_id),
                    namespace=default_namespace,
                )
                container.register_service("event_bus", self.event_bus)
                emit_log_event(
                    LogLevel.INFO,
                    "EventBus service initialized successfully",
                    {
                        "node_id": str(self.node_id),
                        "namespace": default_namespace,
                    },
                )
            else:
                emit_log_event(
                    LogLevel.WARNING,
                    "EventBus not available - Kafka client required for event-driven coordination",
                    {"node_id": str(self.node_id)},
                )
                self.event_bus = None

        # Package storage directory
        self.package_dir = Path(os.getenv("PACKAGE_DIR", "/tmp/deployment_packages"))
        self.package_dir.mkdir(parents=True, exist_ok=True)

        # Consul configuration for service discovery
        self.consul_host: str = os.getenv("CONSUL_HOST", "omninode-bridge-consul")
        self.consul_port: int = int(os.getenv("CONSUL_PORT", "28500"))
        self.consul_enable_registration: bool = self.config.get(
            "consul_enable_registration", True
        )

        emit_log_event(
            LogLevel.INFO,
            "NodeDeploymentReceiverEffect initialized",
            {
                "node_id": str(self.node_id),
                "docker_host": docker_host,
                "package_dir": str(self.package_dir),
                "allowed_ip_ranges": allowed_ip_ranges,
            },
        )

        # Register with Consul for service discovery
        if self.consul_enable_registration:
            self._register_with_consul_sync()

    async def execute_effect(self, contract: ModelContractEffect) -> Any:
        """
        Execute effect operation.

        Args:
            contract: Effect contract with operation parameters

        Returns:
            Operation result based on operation type

        Raises:
            ModelOnexError: If operation fails
        """
        correlation_id = contract.correlation_id or uuid4()

        emit_log_event(
            LogLevel.INFO,
            "Executing deployment receiver effect",
            {
                "node_id": str(self.node_id),
                "correlation_id": str(correlation_id),
                "operation_type": contract.input_state.get("operation_type"),
            },
        )

        try:
            # Route to appropriate operation handler
            operation_type = contract.input_state.get("operation_type")

            if operation_type == "receive_package":
                return await self._handle_receive_package(contract, correlation_id)
            elif operation_type == "load_image":
                return await self._handle_load_image(contract, correlation_id)
            elif operation_type == "deploy_container":
                return await self._handle_deploy_container(contract, correlation_id)
            elif operation_type == "health_check":
                return await self._handle_health_check(contract, correlation_id)
            elif operation_type == "publish_deployment_event":
                return await self._handle_publish_event(contract, correlation_id)
            elif operation_type == "full_deployment":
                return await self._handle_full_deployment(contract, correlation_id)
            else:
                raise ModelOnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message=f"Unknown operation type: {operation_type}",
                    details={"operation_type": operation_type},
                )

        except ModelOnexError:
            # Re-raise OnexError to preserve error context
            raise
        except (ValueError, KeyError, TypeError) as e:
            # Data validation/parsing errors
            emit_log_event(
                LogLevel.ERROR,
                f"Invalid input data: {e}",
                {
                    "node_id": str(self.node_id),
                    "correlation_id": str(correlation_id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                message=f"Invalid input data: {e!s}",
                details={
                    "original_error": str(e),
                    "correlation_id": str(correlation_id),
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
                    "correlation_id": str(correlation_id),
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
                    "correlation_id": str(correlation_id),
                    "error_type": type(e).__name__,
                },
            )

    async def _handle_receive_package(
        self, contract: ModelContractEffect, correlation_id: UUID
    ) -> ModelPackageReceiveOutput:
        """Handle package receive operation."""
        start_time = time.time()

        try:
            # Parse input
            input_data = ModelPackageReceiveInput(**contract.input_state)

            # Validate authentication
            message = f"{input_data.package_data.image_tar_path}{input_data.package_data.checksum}".encode()
            auth_result = self.security_validator.validate_hmac_signature(
                input_data.sender_auth, message
            )

            if not auth_result.is_valid:
                return ModelPackageReceiveOutput(
                    success=False,
                    checksum_valid=False,
                    auth_valid=False,
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    error_message=auth_result.error_message or "Authentication failed",
                )

            # Validate IP whitelist
            if input_data.sender_auth.sender_ip:
                ip_result = self.security_validator.validate_ip_whitelist(
                    input_data.sender_auth.sender_ip
                )
                if not ip_result.is_allowed:
                    return ModelPackageReceiveOutput(
                        success=False,
                        checksum_valid=False,
                        auth_valid=True,
                        execution_time_ms=int((time.time() - start_time) * 1000),
                        error_message=ip_result.error_message or "IP not in whitelist",
                    )

            # Validate checksum
            checksum_result = self.security_validator.validate_checksum(
                input_data.package_data.image_tar_path, input_data.package_data.checksum
            )

            if not checksum_result.is_valid:
                return ModelPackageReceiveOutput(
                    success=False,
                    checksum_valid=False,
                    auth_valid=True,
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    error_message=checksum_result.error_message
                    or "Checksum validation failed",
                )

            # All validations passed
            execution_time_ms = int((time.time() - start_time) * 1000)
            return ModelPackageReceiveOutput(
                success=True,
                package_path=input_data.package_data.image_tar_path,
                checksum_valid=True,
                auth_valid=True,
                execution_time_ms=execution_time_ms,
            )

        except (ValueError, KeyError) as e:
            # Data validation errors
            emit_log_event(
                LogLevel.ERROR,
                f"Invalid package data: {e}",
                {
                    "correlation_id": str(correlation_id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            return ModelPackageReceiveOutput(
                success=False,
                checksum_valid=False,
                auth_valid=False,
                execution_time_ms=int((time.time() - start_time) * 1000),
                error_message=f"Invalid package data: {e}",
            )
        except Exception as e:
            # Unexpected errors - log with exc_info for debugging
            emit_log_event(
                LogLevel.ERROR,
                f"Unexpected package receive error: {type(e).__name__}",
                {
                    "correlation_id": str(correlation_id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            logger.error(
                f"Unexpected package receive error: {type(e).__name__}", exc_info=True
            )
            return ModelPackageReceiveOutput(
                success=False,
                checksum_valid=False,
                auth_valid=False,
                execution_time_ms=int((time.time() - start_time) * 1000),
                error_message=str(e),
            )

    async def _handle_load_image(
        self, contract: ModelContractEffect, correlation_id: UUID
    ) -> ModelImageLoadOutput:
        """Handle Docker image load operation."""
        start_time = time.time()

        try:
            input_data = ModelImageLoadInput(**contract.input_state)

            # Load image using Docker client
            success, image_id, image_name, error_msg = (
                await self.docker_client.load_image(input_data.image_tar_path)
            )

            execution_time_ms = int((time.time() - start_time) * 1000)

            return ModelImageLoadOutput(
                success=success,
                image_id=image_id,
                image_name=image_name,
                execution_time_ms=execution_time_ms,
                error_message=error_msg if not success else None,
            )

        except (OSError, FileNotFoundError) as e:
            # File system errors
            emit_log_event(
                LogLevel.ERROR,
                f"Docker image file error: {e}",
                {
                    "correlation_id": str(correlation_id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            return ModelImageLoadOutput(
                success=False,
                execution_time_ms=int((time.time() - start_time) * 1000),
                error_message=f"Image file error: {e}",
            )
        except Exception as e:
            # Unexpected Docker errors - log with exc_info for debugging
            emit_log_event(
                LogLevel.ERROR,
                f"Unexpected Docker load error: {type(e).__name__}",
                {
                    "correlation_id": str(correlation_id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            logger.error(
                f"Unexpected Docker load error: {type(e).__name__}", exc_info=True
            )
            return ModelImageLoadOutput(
                success=False,
                execution_time_ms=int((time.time() - start_time) * 1000),
                error_message=str(e),
            )

    async def _handle_deploy_container(
        self, contract: ModelContractEffect, correlation_id: UUID
    ) -> ModelContainerDeployOutput:
        """Handle container deployment operation."""
        start_time = time.time()

        try:
            input_data = ModelContainerDeployInput(**contract.input_state)

            # Deploy container using Docker client
            success, container_id, container_short_id, container_url, error_msg = (
                await self.docker_client.deploy_container(input_data.deployment_config)
            )

            execution_time_ms = int((time.time() - start_time) * 1000)

            return ModelContainerDeployOutput(
                success=success,
                container_id=container_id,
                container_short_id=container_short_id,
                container_url=container_url,
                execution_time_ms=execution_time_ms,
                error_message=error_msg if not success else None,
            )

        except ConnectionError as e:
            # Docker daemon connection error
            emit_log_event(
                LogLevel.ERROR,
                f"Docker daemon connection failed: {e}",
                {
                    "correlation_id": str(correlation_id),
                    "error": str(e),
                    "error_type": "ConnectionError",
                },
            )
            return ModelContainerDeployOutput(
                success=False,
                execution_time_ms=int((time.time() - start_time) * 1000),
                error_message=f"Docker connection failed: {e}",
            )
        except Exception as e:
            # Unexpected Docker deployment errors - log with exc_info for debugging
            emit_log_event(
                LogLevel.ERROR,
                f"Unexpected Docker deployment error: {type(e).__name__}",
                {
                    "correlation_id": str(correlation_id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            logger.error(
                f"Unexpected Docker deployment error: {type(e).__name__}", exc_info=True
            )
            return ModelContainerDeployOutput(
                success=False,
                execution_time_ms=int((time.time() - start_time) * 1000),
                error_message=str(e),
            )

    async def _handle_health_check(
        self, contract: ModelContractEffect, correlation_id: UUID
    ) -> ModelHealthCheckOutput:
        """Handle container health check operation."""
        start_time = time.time()

        try:
            input_data = ModelHealthCheckInput(**contract.input_state)

            # Perform health check using Docker client
            health_status = await self.docker_client.health_check(
                input_data.container_id
            )

            execution_time_ms = int((time.time() - start_time) * 1000)

            return ModelHealthCheckOutput(
                success=health_status.is_healthy,
                health_status=health_status,
                execution_time_ms=execution_time_ms,
                error_message=(
                    health_status.error_message
                    if not health_status.is_healthy
                    else None
                ),
            )

        except ConnectionError as e:
            # Docker daemon connection error
            emit_log_event(
                LogLevel.WARNING,
                f"Docker health check connection failed: {e}",
                {
                    "correlation_id": str(correlation_id),
                    "error": str(e),
                    "error_type": "ConnectionError",
                },
            )
            return ModelHealthCheckOutput(
                success=False,
                execution_time_ms=int((time.time() - start_time) * 1000),
                error_message=f"Health check connection failed: {e}",
            )
        except (TimeoutError, asyncio.TimeoutError) as e:
            # Health check timeout
            emit_log_event(
                LogLevel.WARNING,
                f"Docker health check timeout: {e}",
                {
                    "correlation_id": str(correlation_id),
                    "error": str(e),
                    "error_type": "TimeoutError",
                },
            )
            return ModelHealthCheckOutput(
                success=False,
                execution_time_ms=int((time.time() - start_time) * 1000),
                error_message=f"Health check timeout: {e}",
            )
        except Exception as e:
            # Unexpected health check errors - log with exc_info for debugging
            emit_log_event(
                LogLevel.ERROR,
                f"Unexpected health check error: {type(e).__name__}",
                {
                    "correlation_id": str(correlation_id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            logger.error(
                f"Unexpected health check error: {type(e).__name__}", exc_info=True
            )
            return ModelHealthCheckOutput(
                success=False,
                execution_time_ms=int((time.time() - start_time) * 1000),
                error_message=str(e),
            )

    async def _handle_publish_event(
        self, contract: ModelContractEffect, correlation_id: UUID
    ) -> ModelDeploymentEventOutput:
        """Handle deployment event publishing operation via EventBus."""
        start_time = time.time()

        try:
            input_data = ModelDeploymentEventInput(**contract.input_state)

            # Publish deployment event via EventBus
            success = False
            if self.event_bus and self.event_bus.is_initialized:
                event_payload = {
                    "event_type": input_data.event_type,
                    "container_name": input_data.deployment_config.container_name,
                    "deployment_config": input_data.deployment_config.model_dump(),
                    "timestamp": datetime.now(UTC).isoformat(),
                }

                success = await self.event_bus.publish_action_event(
                    correlation_id=correlation_id,
                    action_type=f"DEPLOYMENT_{input_data.event_type}",
                    payload=event_payload,
                )

                emit_log_event(
                    LogLevel.INFO,
                    f"Published deployment event via EventBus: {input_data.event_type}",
                    {
                        "event_type": input_data.event_type,
                        "container_name": input_data.deployment_config.container_name,
                        "correlation_id": str(correlation_id),
                        "success": success,
                    },
                )
            else:
                # Fallback: Just log the event if EventBus unavailable
                emit_log_event(
                    LogLevel.WARNING,
                    f"EventBus unavailable, logging deployment event: {input_data.event_type}",
                    {
                        "event_type": input_data.event_type,
                        "container_name": input_data.deployment_config.container_name,
                        "correlation_id": str(correlation_id),
                    },
                )

            execution_time_ms = int((time.time() - start_time) * 1000)

            # Generate topic name based on event type
            topic = f"dev.omninode-bridge.deployment.{input_data.event_type.lower().replace('_', '-')}.v1"

            return ModelDeploymentEventOutput(
                success=(
                    success if self.event_bus else True
                ),  # Success if logged in fallback mode
                event_id=str(uuid4()),
                topic=topic,
                execution_time_ms=execution_time_ms,
            )

        except ConnectionError as e:
            # Kafka connection error
            emit_log_event(
                LogLevel.ERROR,
                f"Kafka connection failed: {e}",
                {
                    "correlation_id": str(correlation_id),
                    "error": str(e),
                    "error_type": "ConnectionError",
                },
            )
            return ModelDeploymentEventOutput(
                success=False,
                execution_time_ms=int((time.time() - start_time) * 1000),
                error_message=f"Kafka connection failed: {e}",
            )
        except (TimeoutError, asyncio.TimeoutError) as e:
            # Kafka timeout
            emit_log_event(
                LogLevel.ERROR,
                f"Kafka publish timeout: {e}",
                {
                    "correlation_id": str(correlation_id),
                    "error": str(e),
                    "error_type": "TimeoutError",
                },
            )
            return ModelDeploymentEventOutput(
                success=False,
                execution_time_ms=int((time.time() - start_time) * 1000),
                error_message=f"Kafka timeout: {e}",
            )
        except Exception as e:
            # Unexpected Kafka errors - log with exc_info for debugging
            emit_log_event(
                LogLevel.ERROR,
                f"Unexpected Kafka publish error: {type(e).__name__}",
                {
                    "correlation_id": str(correlation_id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            logger.error(
                f"Unexpected Kafka publish error: {type(e).__name__}", exc_info=True
            )
            return ModelDeploymentEventOutput(
                success=False,
                execution_time_ms=int((time.time() - start_time) * 1000),
                error_message=str(e),
            )

    async def _handle_full_deployment(
        self, contract: ModelContractEffect, correlation_id: UUID
    ) -> ModelFullDeploymentOutput:
        """
        Handle full deployment operation (all steps).

        Executes:
        1. Receive and validate package
        2. Load Docker image
        3. Deploy container
        4. Health check
        5. Publish deployment events
        """
        start_time = time.time()
        events_published = []

        try:
            input_data = ModelFullDeploymentInput(**contract.input_state)

            # Step 1: Publish DEPLOYMENT_STARTED event
            await self._publish_event_internal(
                "DEPLOYMENT_STARTED", input_data, correlation_id
            )
            events_published.append("DEPLOYMENT_STARTED")

            # Step 2: Validate package
            receive_result = await self._handle_receive_package(
                ModelContractEffect(
                    correlation_id=correlation_id,
                    input_state={
                        "operation_type": "receive_package",
                        "package_data": input_data.package_data.model_dump(),
                        "sender_auth": input_data.sender_auth.model_dump(),
                        "correlation_id": str(correlation_id),
                    },
                ),
                correlation_id,
            )

            if not receive_result.success:
                await self._publish_event_internal(
                    "DEPLOYMENT_FAILED", input_data, correlation_id
                )
                events_published.append("DEPLOYMENT_FAILED")
                return ModelFullDeploymentOutput(
                    success=False,
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    kafka_events_published=events_published,
                    error_message=receive_result.error_message,
                )

            # Step 3: Load image
            load_result = await self._handle_load_image(
                ModelContractEffect(
                    correlation_id=correlation_id,
                    input_state={
                        "operation_type": "load_image",
                        "image_tar_path": input_data.package_data.image_tar_path,
                        "correlation_id": str(correlation_id),
                    },
                ),
                correlation_id,
            )

            if not load_result.success:
                await self._publish_event_internal(
                    "DEPLOYMENT_FAILED", input_data, correlation_id
                )
                events_published.append("DEPLOYMENT_FAILED")
                return ModelFullDeploymentOutput(
                    success=False,
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    kafka_events_published=events_published,
                    error_message=load_result.error_message,
                )

            await self._publish_event_internal(
                "IMAGE_LOADED", input_data, correlation_id
            )
            events_published.append("IMAGE_LOADED")

            # Step 4: Deploy container
            deploy_result = await self._handle_deploy_container(
                ModelContractEffect(
                    correlation_id=correlation_id,
                    input_state={
                        "operation_type": "deploy_container",
                        "deployment_config": input_data.deployment_config.model_dump(),
                        "correlation_id": str(correlation_id),
                    },
                ),
                correlation_id,
            )

            if not deploy_result.success:
                await self._publish_event_internal(
                    "DEPLOYMENT_FAILED", input_data, correlation_id
                )
                events_published.append("DEPLOYMENT_FAILED")
                return ModelFullDeploymentOutput(
                    success=False,
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    kafka_events_published=events_published,
                    error_message=deploy_result.error_message,
                )

            await self._publish_event_internal(
                "CONTAINER_STARTED",
                input_data,
                correlation_id,
                deploy_result.container_id,
            )
            events_published.append("CONTAINER_STARTED")

            # Step 5: Health check
            health_result = await self._handle_health_check(
                ModelContractEffect(
                    correlation_id=correlation_id,
                    input_state={
                        "operation_type": "health_check",
                        "container_id": deploy_result.container_id,
                        "correlation_id": str(correlation_id),
                    },
                ),
                correlation_id,
            )

            if health_result.success and health_result.health_status:
                await self._publish_event_internal(
                    "HEALTH_CHECK_PASSED",
                    input_data,
                    correlation_id,
                    deploy_result.container_id,
                )
                events_published.append("HEALTH_CHECK_PASSED")

            # Step 6: Publish deployment completed
            await self._publish_event_internal(
                "DEPLOYMENT_COMPLETED",
                input_data,
                correlation_id,
                deploy_result.container_id,
            )
            events_published.append("DEPLOYMENT_COMPLETED")

            execution_time_ms = int((time.time() - start_time) * 1000)

            return ModelFullDeploymentOutput(
                success=True,
                deployment_success=True,
                container_id=deploy_result.container_id,
                container_short_id=deploy_result.container_short_id,
                container_url=deploy_result.container_url,
                health_status=health_result.health_status,
                image_loaded=load_result.success,
                image_id=load_result.image_id,
                kafka_events_published=events_published,
                execution_time_ms=execution_time_ms,
            )

        except (ValueError, KeyError, TypeError) as e:
            # Data validation errors
            emit_log_event(
                LogLevel.ERROR,
                f"Invalid deployment data: {e}",
                {
                    "correlation_id": str(correlation_id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

            # Publish failure event
            try:
                await self._publish_event_internal(
                    "DEPLOYMENT_FAILED", input_data, correlation_id
                )
                events_published.append("DEPLOYMENT_FAILED")
            except Exception:
                pass

            return ModelFullDeploymentOutput(
                success=False,
                execution_time_ms=int((time.time() - start_time) * 1000),
                kafka_events_published=events_published,
                error_message=f"Invalid deployment data: {e}",
                error_details={
                    "original_error": str(e),
                    "error_type": type(e).__name__,
                },
            )
        except Exception as e:
            # Unexpected deployment errors - log with exc_info for debugging
            emit_log_event(
                LogLevel.ERROR,
                f"Unexpected full deployment error: {type(e).__name__}",
                {
                    "correlation_id": str(correlation_id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            logger.error(
                f"Unexpected full deployment error: {type(e).__name__}", exc_info=True
            )

            # Publish failure event
            try:
                await self._publish_event_internal(
                    "DEPLOYMENT_FAILED", input_data, correlation_id
                )
                events_published.append("DEPLOYMENT_FAILED")
            except Exception:
                pass

            return ModelFullDeploymentOutput(
                success=False,
                execution_time_ms=int((time.time() - start_time) * 1000),
                kafka_events_published=events_published,
                error_message=str(e),
                error_details={
                    "original_error": str(e),
                    "error_type": type(e).__name__,
                },
            )

    async def _publish_event_internal(
        self,
        event_type: str,
        input_data: ModelFullDeploymentInput,
        correlation_id: UUID,
        container_id: Optional[str] = None,
    ) -> None:
        """Internal helper to publish deployment events via EventBus."""
        emit_log_event(
            LogLevel.INFO,
            f"Publishing deployment event: {event_type}",
            {
                "event_type": event_type,
                "container_name": input_data.deployment_config.container_name,
                "correlation_id": str(correlation_id),
                "container_id": container_id,
            },
        )

        # Publish deployment event via EventBus
        if self.event_bus and self.event_bus.is_initialized:
            event_payload = {
                "event_type": event_type,
                "container_name": input_data.deployment_config.container_name,
                "container_id": container_id,
                "deployment_config": input_data.deployment_config.model_dump(),
                "package_data": input_data.package_data.model_dump(),
                "timestamp": datetime.now(UTC).isoformat(),
            }

            try:
                success = await self.event_bus.publish_action_event(
                    correlation_id=correlation_id,
                    action_type=f"DEPLOYMENT_{event_type}",
                    payload=event_payload,
                )

                if success:
                    emit_log_event(
                        LogLevel.DEBUG,
                        f"Published deployment event via EventBus: {event_type}",
                        {
                            "event_type": event_type,
                            "correlation_id": str(correlation_id),
                            "container_id": container_id,
                        },
                    )
                else:
                    emit_log_event(
                        LogLevel.WARNING,
                        f"Failed to publish deployment event via EventBus: {event_type}",
                        {
                            "event_type": event_type,
                            "correlation_id": str(correlation_id),
                        },
                    )
            except ConnectionError as e:
                # EventBus connection error - non-critical
                emit_log_event(
                    LogLevel.WARNING,
                    f"EventBus connection failed: {e}",
                    {
                        "event_type": event_type,
                        "correlation_id": str(correlation_id),
                        "error": str(e),
                        "error_type": "ConnectionError",
                    },
                )
            except (TimeoutError, asyncio.TimeoutError) as e:
                # EventBus timeout - non-critical
                emit_log_event(
                    LogLevel.WARNING,
                    f"EventBus publish timeout: {e}",
                    {
                        "event_type": event_type,
                        "correlation_id": str(correlation_id),
                        "error": str(e),
                        "error_type": "TimeoutError",
                    },
                )
            except Exception as e:
                # Unexpected EventBus errors - log with exc_info but don't fail
                emit_log_event(
                    LogLevel.WARNING,
                    f"Unexpected EventBus error: {type(e).__name__}",
                    {
                        "event_type": event_type,
                        "correlation_id": str(correlation_id),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
                logger.error(
                    f"Unexpected EventBus error: {type(e).__name__}", exc_info=True
                )
        else:
            # Fallback: EventBus unavailable, event logged above
            emit_log_event(
                LogLevel.DEBUG,
                f"EventBus unavailable, event logged only: {event_type}",
                {
                    "event_type": event_type,
                    "correlation_id": str(correlation_id),
                },
            )

    def _register_with_consul_sync(self) -> None:
        """
        Register deployment receiver node with Consul for service discovery (synchronous).

        Registers the receiver as a service with health check endpoint.
        Includes metadata about node capabilities.

        Note:
            This is a non-blocking registration. Failures are logged but don't
            fail node startup. Service will continue without Consul if registration fails.
        """
        try:
            import consul

            # Initialize Consul client
            consul_client = consul.Consul(host=self.consul_host, port=self.consul_port)

            # Generate unique service ID
            service_id = f"omninode-bridge-deployment-receiver-{self.node_id}"

            # Get service port from environment (default to 8001 for receiver)
            service_port = int(os.getenv("SERVICE_PORT", "8001"))

            # Get service host from environment (default to localhost)
            service_host = os.getenv("SERVICE_HOST", "localhost")

            # Prepare service tags
            service_tags = [
                "onex",
                "bridge",
                "deployment_receiver",
                "omninode_bridge",
                "node_type:deployment_receiver",
                f"namespace:{os.getenv('NAMESPACE', 'dev')}",
            ]

            # Health check URL (assumes FastAPI health endpoint is available)
            health_check_url = f"http://{service_host}:{service_port}/health"

            # Register service with Consul
            consul_client.agent.service.register(
                name="omninode-bridge-deployment-receiver",
                service_id=service_id,
                address=service_host,
                port=service_port,
                tags=service_tags,
                http=health_check_url,
                interval="30s",
                timeout="5s",
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
        Deregister receiver from Consul on shutdown (synchronous).

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
        except ConnectionError as e:
            # Consul connection failed - non-critical during shutdown
            emit_log_event(
                LogLevel.WARNING,
                f"Consul connection failed during deregistration: {e}",
                {
                    "node_id": str(self.node_id),
                    "error": str(e),
                    "error_type": "ConnectionError",
                },
            )
        except Exception as e:
            # Unexpected errors - log but don't fail shutdown
            emit_log_event(
                LogLevel.WARNING,
                f"Unexpected Consul deregistration error: {type(e).__name__}",
                {
                    "node_id": str(self.node_id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            logger.error(
                f"Unexpected Consul deregistration error: {type(e).__name__}",
                exc_info=True,
            )

    def __del__(self):
        """Cleanup Docker client on node destruction."""
        # Deregister from Consul for clean service discovery
        if hasattr(self, "_deregister_from_consul"):
            try:
                self._deregister_from_consul()
            except Exception:
                pass

        if hasattr(self, "docker_client"):
            try:
                self.docker_client.close()
            except Exception:
                pass


__all__ = ["NodeDeploymentReceiverEffect"]
