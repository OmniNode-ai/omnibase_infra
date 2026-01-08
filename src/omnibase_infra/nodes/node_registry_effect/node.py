# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Registry Effect - Declarative effect node for dual-backend registration.

This node follows the ONEX declarative pattern:
    - DECLARATIVE effect driven by contract.yaml
    - Operation routing logic dispatches to handlers based on contract configuration
    - Lightweight shell that delegates to handlers via container resolution
    - Pattern: "Contract-driven, handlers wired externally"

Extends NodeEffect from omnibase_core for infrastructure I/O operations.
Operation routing is 100% driven by contract.yaml's operation_routing section.

Operation Pattern:
    1. Receive registration request (input_model in contract)
    2. Route to appropriate handler based on operation type (operation_routing)
    3. Execute infrastructure I/O (Consul registration, PostgreSQL upsert)
    4. Return structured response (output_model in contract)

Design Decisions:
    - Contract-Driven Routing: All operation routing in YAML, not Python
    - Parallel Execution: Dual-backend operations execute concurrently
    - Declarative Routing: Operations defined in contract's operation_routing
    - Handler Resolution: Handlers resolved via container dependency injection

Node Responsibilities:
    - Dual-backend registration (Consul + PostgreSQL)
    - Partial failure handling with per-backend results
    - Idempotency tracking for retry safety
    - Error sanitization for security

Coroutine Safety:
    This node is async-safe. Multiple operations can be processed concurrently
    when using appropriate backend adapters with connection pooling.

Related Modules:
    - contract.yaml: Operation routing and I/O model definitions
    - handlers/: Operation-specific handlers
    - models/: Node-specific input/output models
    - nodes.effects.registry_effect: Legacy implementation (for reference)

Migration Notes (OMN-1103):
    This declarative node replaces the imperative implementation at
    omnibase_infra.nodes.effects.registry_effect.NodeRegistryEffect.
    The legacy implementation (507 lines) is being refactored into:
    - This declarative node shell
    - Handler classes for specific operations
    - Contract-defined operation routing
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import yaml  # type: ignore[import-untyped]
from omnibase_core.nodes.node_effect import NodeEffect

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.nodes.node_registry_effect.handlers import (
    HandlerConsulDeregister,
    HandlerConsulRegister,
    HandlerPartialRetry,
    HandlerPostgresDeactivate,
    HandlerPostgresUpsert,
)
from omnibase_infra.nodes.node_registry_effect.models import (
    ModelBackendResult,
    ModelRegistryRequest,
    ModelRegistryResponse,
)
from omnibase_infra.utils import sanitize_error_message

if TYPE_CHECKING:
    from uuid import UUID

    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

    from omnibase_infra.nodes.effects.protocol_consul_client import ProtocolConsulClient
    from omnibase_infra.nodes.effects.protocol_postgres_adapter import (
        ProtocolPostgresAdapter,
    )

# Type alias for operation names
OperationType = Literal["register_node", "deregister_node", "retry_partial_failure"]

_logger = logging.getLogger(__name__)


class NodeRegistryEffect(NodeEffect):
    """Declarative effect node for dual-backend node registration.

    Operation routing is defined in contract.yaml - this class implements the
    dispatch logic that routes requests to handlers based on the contract.

    The node coordinates registration against:
        - Consul: Service discovery registration
        - PostgreSQL: Registration record persistence

    Features:
        - Parallel execution of dual-backend operations
        - Partial failure handling (one backend can fail while other succeeds)
        - Contract-driven operation routing
        - Error sanitization to prevent credential exposure
        - Per-backend timing and result tracking

    Supported Operations (from contract.yaml):
        - register_node: Register to both Consul and PostgreSQL in parallel
        - deregister_node: Deregister from both backends in parallel
        - retry_partial_failure: Retry a specific backend after partial failure

    Usage:
        ```python
        from omnibase_core.models.container import ModelONEXContainer
        from omnibase_infra.nodes.node_registry_effect import NodeRegistryEffect

        # Create via container injection
        container = ModelONEXContainer()
        effect = NodeRegistryEffect(container)

        # Execute a registration operation
        response = await effect.execute_operation(request, "register_node")

        # Check for partial failures
        if response.is_partial_failure():
            failed_backends = response.get_failed_backends()
            # Retry failed backends...
        ```
    """

    # Handler class mappings for each backend
    _HANDLER_CLASSES: dict[str, dict[str, type]] = {
        "register_node": {
            "consul": HandlerConsulRegister,
            "postgres": HandlerPostgresUpsert,
        },
        "deregister_node": {
            "consul": HandlerConsulDeregister,
            "postgres": HandlerPostgresDeactivate,
        },
        "retry_partial_failure": {
            "dynamic": HandlerPartialRetry,
        },
    }

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the effect node.

        Args:
            container: ONEX dependency injection container providing:
                - Backend adapters (Consul client, PostgreSQL adapter)
                - Idempotency store
                - Handler instances
                - Configuration
        """
        super().__init__(container)

        # Cached contract configuration
        self._operation_routing: dict[str, object] | None = None

        # Backend adapters (resolved lazily from container)
        self._consul_client: ProtocolConsulClient | None = None
        self._postgres_adapter: ProtocolPostgresAdapter | None = None

    def _load_operation_routing(self) -> dict[str, object]:
        """Load and cache operation routing configuration from contract.yaml.

        Returns:
            The operation_routing section from contract.yaml.

        Raises:
            FileNotFoundError: If contract.yaml does not exist.
            yaml.YAMLError: If contract.yaml is malformed.
        """
        if self._operation_routing is not None:
            return self._operation_routing

        contract_path = Path(__file__).parent / "contract.yaml"
        with open(contract_path) as f:
            contract = yaml.safe_load(f)

        self._operation_routing = contract.get("operation_routing", {})
        return self._operation_routing

    def _get_handlers_for_operation(
        self, operation: OperationType
    ) -> list[dict[str, object]]:
        """Get handler configurations for an operation from contract.yaml.

        Args:
            operation: The operation type (register_node, deregister_node, etc.)

        Returns:
            List of handler configurations with handler class, module, and backend.
        """
        routing = self._load_operation_routing()
        operations = cast(list[dict[str, object]], routing.get("operations", []))

        matching_handlers: list[dict[str, object]] = []
        for op_config in operations:
            if op_config.get("operation") == operation:
                matching_handlers.append(op_config)

        return matching_handlers

    def set_consul_client(self, client: ProtocolConsulClient) -> None:
        """Set the Consul client for service registration.

        Args:
            client: Protocol-compliant Consul client implementation.
        """
        self._consul_client = client

    def set_postgres_adapter(self, adapter: ProtocolPostgresAdapter) -> None:
        """Set the PostgreSQL adapter for registration persistence.

        Args:
            adapter: Protocol-compliant PostgreSQL adapter implementation.
        """
        self._postgres_adapter = adapter

    def _get_consul_client(self) -> ProtocolConsulClient:
        """Get the Consul client, resolving from container if not set.

        Returns:
            The Consul client instance.

        Raises:
            ProtocolConfigurationError: If Consul client is not configured.
        """
        if self._consul_client is not None:
            return self._consul_client

        # Try to resolve from container
        if hasattr(self._container, "resolve"):
            try:
                from omnibase_infra.nodes.effects.protocol_consul_client import (
                    ProtocolConsulClient,
                )

                self._consul_client = self._container.resolve(ProtocolConsulClient)
                return self._consul_client
            except Exception:
                pass

        raise ProtocolConfigurationError(
            "Consul client not configured. Call set_consul_client() or configure "
            "the container to provide ProtocolConsulClient."
        )

    def _get_postgres_adapter(self) -> ProtocolPostgresAdapter:
        """Get the PostgreSQL adapter, resolving from container if not set.

        Returns:
            The PostgreSQL adapter instance.

        Raises:
            ProtocolConfigurationError: If PostgreSQL adapter is not configured.
        """
        if self._postgres_adapter is not None:
            return self._postgres_adapter

        # Try to resolve from container
        if hasattr(self._container, "resolve"):
            try:
                from omnibase_infra.nodes.effects.protocol_postgres_adapter import (
                    ProtocolPostgresAdapter,
                )

                self._postgres_adapter = self._container.resolve(
                    ProtocolPostgresAdapter
                )
                return self._postgres_adapter
            except Exception:
                pass

        raise ProtocolConfigurationError(
            "PostgreSQL adapter not configured. Call set_postgres_adapter() or "
            "configure the container to provide ProtocolPostgresAdapter."
        )

    def _create_handler(
        self, handler_config: dict[str, object]
    ) -> (
        HandlerConsulRegister
        | HandlerConsulDeregister
        | HandlerPostgresUpsert
        | HandlerPostgresDeactivate
        | HandlerPartialRetry
    ):
        """Create a handler instance from configuration.

        Args:
            handler_config: Handler configuration from contract.yaml containing:
                - handler.name: Handler class name
                - handler.module: Module path for the handler
                - backend: Backend type (consul, postgres, dynamic)

        Returns:
            Instantiated handler with required dependencies.

        Raises:
            RuntimeError: If handler cannot be instantiated.
        """
        backend = cast(str, handler_config.get("backend", ""))
        handler_info = cast(dict[str, object], handler_config.get("handler", {}))
        handler_name = cast(str, handler_info.get("name", ""))

        # Instantiate based on handler class and backend
        if handler_name == "HandlerConsulRegister":
            return HandlerConsulRegister(self._get_consul_client())

        elif handler_name == "HandlerConsulDeregister":
            return HandlerConsulDeregister(self._get_consul_client())

        elif handler_name == "HandlerPostgresUpsert":
            return HandlerPostgresUpsert(self._get_postgres_adapter())

        elif handler_name == "HandlerPostgresDeactivate":
            return HandlerPostgresDeactivate(self._get_postgres_adapter())

        elif handler_name == "HandlerPartialRetry":
            return HandlerPartialRetry(
                self._get_consul_client(),
                self._get_postgres_adapter(),
            )

        raise RuntimeError(f"Unknown handler: {handler_name}")

    async def execute_operation(
        self,
        request: ModelRegistryRequest,
        operation: OperationType = "register_node",
        target_backend: str | None = None,
    ) -> ModelRegistryResponse:
        """Execute a registration operation against configured backends.

        Routes the request to appropriate handlers based on the operation type
        and contract.yaml configuration. For dual-backend operations (register_node,
        deregister_node), handlers execute in parallel using asyncio.gather().

        Args:
            request: The registration request containing node details.
            operation: The operation to perform:
                - "register_node": Register to Consul and PostgreSQL
                - "deregister_node": Deregister from both backends
                - "retry_partial_failure": Retry a specific backend
            target_backend: For retry_partial_failure, the backend to target.
                Must be "consul" or "postgres".

        Returns:
            ModelRegistryResponse with:
                - status: "success" (both succeeded), "partial" (one failed),
                  or "failed" (both failed)
                - consul_result: Result from Consul operation
                - postgres_result: Result from PostgreSQL operation
                - processing_time_ms: Total operation time
                - error_summary: Aggregated error messages if any failed

        Raises:
            ProtocolConfigurationError: If backend adapters are not configured.
            ValueError: If operation is invalid or target_backend is missing
                for retry_partial_failure.

        Example:
            >>> response = await effect.execute_operation(request, "register_node")
            >>> if response.status == "partial":
            ...     # Handle partial failure
            ...     for backend in response.get_failed_backends():
            ...         await effect.execute_operation(
            ...             request, "retry_partial_failure", target_backend=backend
            ...         )
        """
        start_time = time.perf_counter()
        correlation_id = request.correlation_id

        # Get handler configurations from contract
        handler_configs = self._get_handlers_for_operation(operation)
        if not handler_configs:
            raise ValueError(f"Unknown operation: {operation}")

        # Check for parallel execution mode from contract
        routing = self._load_operation_routing()
        execution_mode = routing.get("execution_mode", "parallel")
        partial_failure_handling = routing.get("partial_failure_handling", True)

        # Initialize results with failure defaults
        consul_result = ModelBackendResult(
            success=False,
            error="Handler not executed",
            error_code="HANDLER_NOT_EXECUTED",
            duration_ms=0.0,
            backend_id="consul",
            correlation_id=correlation_id,
        )
        postgres_result = ModelBackendResult(
            success=False,
            error="Handler not executed",
            error_code="HANDLER_NOT_EXECUTED",
            duration_ms=0.0,
            backend_id="postgres",
            correlation_id=correlation_id,
        )

        # Handle retry_partial_failure operation (single backend)
        if operation == "retry_partial_failure":
            if not target_backend:
                raise ValueError(
                    "target_backend is required for retry_partial_failure operation"
                )

            # Create partial retry handler - must be HandlerPartialRetry
            handler_config = handler_configs[0]  # There's only one for retry
            retry_handler = HandlerPartialRetry(
                self._get_consul_client(),
                self._get_postgres_adapter(),
            )

            # Create a protocol-compliant request for the retry handler
            # The handler expects a ProtocolPartialRetryRequest
            from omnibase_infra.nodes.node_registry_effect.models.model_partial_retry_request import (
                ModelPartialRetryRequest,
            )

            retry_request = ModelPartialRetryRequest(
                node_id=request.node_id,
                node_type=request.node_type,
                node_version=request.node_version,
                target_backend=target_backend,
                idempotency_key=None,  # Can be extended to support idempotency
                service_name=request.service_name,
                tags=list(request.tags),
                health_check_config=request.health_check_config,
                endpoints=dict(request.endpoints),
                metadata=dict(request.metadata),
            )

            result = await retry_handler.handle(retry_request, correlation_id)

            # Map result to appropriate backend
            if target_backend == "consul":
                consul_result = result
            elif target_backend == "postgres":
                postgres_result = result

            processing_time_ms = (time.perf_counter() - start_time) * 1000

            return ModelRegistryResponse.from_backend_results(
                node_id=request.node_id,
                correlation_id=correlation_id,
                consul_result=consul_result,
                postgres_result=postgres_result,
                timestamp=datetime.now(UTC),
            )

        # Create handlers for the specific operation
        # For register_node/deregister_node, we need both consul and postgres handlers
        consul_handler_instance: (
            HandlerConsulRegister | HandlerConsulDeregister | None
        ) = None
        postgres_handler_instance: (
            HandlerPostgresUpsert | HandlerPostgresDeactivate | None
        ) = None

        for config in handler_configs:
            backend = cast(str, config.get("backend", ""))
            handler_info = cast(dict[str, object], config.get("handler", {}))
            handler_name = cast(str, handler_info.get("name", ""))

            if backend == "consul":
                if handler_name == "HandlerConsulRegister":
                    consul_handler_instance = HandlerConsulRegister(
                        self._get_consul_client()
                    )
                elif handler_name == "HandlerConsulDeregister":
                    consul_handler_instance = HandlerConsulDeregister(
                        self._get_consul_client()
                    )
            elif backend == "postgres":
                if handler_name == "HandlerPostgresUpsert":
                    postgres_handler_instance = HandlerPostgresUpsert(
                        self._get_postgres_adapter()
                    )
                elif handler_name == "HandlerPostgresDeactivate":
                    postgres_handler_instance = HandlerPostgresDeactivate(
                        self._get_postgres_adapter()
                    )

        # Execute handlers based on execution mode
        if (
            execution_mode == "parallel"
            and consul_handler_instance
            and postgres_handler_instance
        ):
            # Parallel execution using asyncio.gather
            # return_exceptions=True allows partial failure handling
            results = await asyncio.gather(
                consul_handler_instance.handle(request, correlation_id),
                postgres_handler_instance.handle(request, correlation_id),
                return_exceptions=True,
            )

            # Process results - results is tuple[ModelBackendResult | BaseException, ...]
            consul_raw = results[0]
            postgres_raw = results[1]

            if isinstance(consul_raw, BaseException):
                # Cast to Exception for type checker - BaseException from gather
                exc = (
                    consul_raw
                    if isinstance(consul_raw, Exception)
                    else Exception(str(consul_raw))
                )
                consul_result = ModelBackendResult(
                    success=False,
                    error=sanitize_error_message(exc),
                    error_code="CONSUL_EXCEPTION",
                    duration_ms=0.0,
                    backend_id="consul",
                    correlation_id=correlation_id,
                )
            else:
                consul_result = consul_raw

            if isinstance(postgres_raw, BaseException):
                # Cast to Exception for type checker - BaseException from gather
                exc = (
                    postgres_raw
                    if isinstance(postgres_raw, Exception)
                    else Exception(str(postgres_raw))
                )
                postgres_result = ModelBackendResult(
                    success=False,
                    error=sanitize_error_message(exc),
                    error_code="POSTGRES_EXCEPTION",
                    duration_ms=0.0,
                    backend_id="postgres",
                    correlation_id=correlation_id,
                )
            else:
                postgres_result = postgres_raw

        else:
            # Sequential execution
            if consul_handler_instance:
                try:
                    consul_result = await consul_handler_instance.handle(
                        request, correlation_id
                    )
                except Exception as e:
                    consul_result = ModelBackendResult(
                        success=False,
                        error=sanitize_error_message(e),
                        error_code="CONSUL_EXCEPTION",
                        duration_ms=0.0,
                        backend_id="consul",
                        correlation_id=correlation_id,
                    )

            if postgres_handler_instance:
                try:
                    postgres_result = await postgres_handler_instance.handle(
                        request, correlation_id
                    )
                except Exception as e:
                    postgres_result = ModelBackendResult(
                        success=False,
                        error=sanitize_error_message(e),
                        error_code="POSTGRES_EXCEPTION",
                        duration_ms=0.0,
                        backend_id="postgres",
                        correlation_id=correlation_id,
                    )

        # Build and return response
        processing_time_ms = (time.perf_counter() - start_time) * 1000

        return ModelRegistryResponse.from_backend_results(
            node_id=request.node_id,
            correlation_id=correlation_id,
            consul_result=consul_result,
            postgres_result=postgres_result,
            timestamp=datetime.now(UTC),
        )


__all__ = ["NodeRegistryEffect"]
