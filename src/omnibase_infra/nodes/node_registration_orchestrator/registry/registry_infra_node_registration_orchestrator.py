# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry for NodeRegistrationOrchestrator handler wiring.

This registry provides a static factory method for creating handler instances
used by the NodeRegistrationOrchestrator. It follows the ONEX registry pattern
and provides a declarative mapping between event models and their handlers.

Handler Wiring (from contract.yaml):
    - ModelNodeIntrospectionEvent -> HandlerNodeIntrospected
    - ModelRuntimeTick -> HandlerRuntimeTick
    - ModelNodeRegistrationAcked -> HandlerNodeRegistrationAcked
    - ModelNodeHeartbeatEvent -> HandlerNodeHeartbeat

Handler Implementation:
    All handlers implement ProtocolMessageHandler directly with:
    - handler_id, category, message_types, node_kind properties
    - handle(envelope) -> ModelHandlerOutput signature

    Handlers are registered directly with ServiceHandlerRegistry without
    adapter classes.

Handler Dependencies:
    All handlers require ProjectionReaderRegistration for state queries.
    Some handlers optionally accept:
    - ProjectorShell: For projection persistence
    - HandlerConsul: For Consul service registration (dual registration)

Maintenance Note:
    When adding a new handler to contract.yaml, you MUST also update the
    ``handler_dependencies`` dict in ``create_registry()`` with the handler's
    constructor arguments. Failure to do so will raise ProtocolConfigurationError
    at runtime with the message: "No dependency configuration found for handler..."

Usage:
    ```python
    from omnibase_infra.nodes.node_registration_orchestrator.registry import (
        RegistryInfraNodeRegistrationOrchestrator,
    )

    registry = RegistryInfraNodeRegistrationOrchestrator.create_registry(
        projection_reader=reader,
        projector=projector,
        consul_handler=consul_handler,
    )
    # registry is frozen and thread-safe

    # Get handler by ID
    handler = registry.get_handler_by_id("handler-node-introspected")
    result = await handler.handle(envelope)
    ```

Related Tickets:
    - OMN-1102: Make NodeRegistrationOrchestrator fully declarative
    - OMN-888 (C1): Registration Orchestrator
    - OMN-1006: Node Heartbeat for Liveness Tracking
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from omnibase_core.services.service_handler_registry import ServiceHandlerRegistry

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.runtime.contract_loaders import (
    load_handler_class_info_from_contract,
)

logger = logging.getLogger(__name__)

# Security: Namespace allowlist for dynamic handler imports
# Per CLAUDE.md Handler Plugin Loader security patterns, only trusted namespaces
# are allowed for dynamic imports to prevent arbitrary code execution.
# Error code: NAMESPACE_NOT_ALLOWED (HANDLER_LOADER_013)
ALLOWED_NAMESPACES: tuple[str, ...] = (
    "omnibase_infra.",
    "omnibase_core.",
)


def _validate_handler_protocol(handler: object) -> tuple[bool, list[str]]:
    """Validate handler implements ProtocolMessageHandler via duck typing.

    Uses duck typing to verify the handler has the required properties and
    methods for ProtocolMessageHandler compliance. Per ONEX conventions,
    protocol compliance is verified via structural typing rather than
    isinstance checks.

    Protocol Requirements (ProtocolMessageHandler):
        - handler_id (property): Unique identifier string
        - category (property): EnumMessageCategory value
        - message_types (property): set[str] of message type names
        - node_kind (property): EnumNodeKind value
        - handle (method): async def handle(envelope) -> ModelHandlerOutput

    Args:
        handler: The handler instance to validate.

    Returns:
        A tuple of (is_valid, missing_members) where:
        - is_valid: True if handler implements all required members
        - missing_members: List of member names that are missing.
          Empty list if all members are present.
    """
    missing_members: list[str] = []

    # Required properties
    if not hasattr(handler, "handler_id"):
        missing_members.append("handler_id")
    if not hasattr(handler, "category"):
        missing_members.append("category")
    if not hasattr(handler, "message_types"):
        missing_members.append("message_types")
    if not hasattr(handler, "node_kind"):
        missing_members.append("node_kind")

    # Required method - handle()
    if not callable(getattr(handler, "handle", None)):
        missing_members.append("handle")

    return (len(missing_members) == 0, missing_members)


def _load_handler_class(class_name: str, module_path: str) -> type:
    """Dynamically load a handler class from a module.

    Security: This function validates the module_path against ALLOWED_NAMESPACES
    before importing. Per CLAUDE.md Handler Plugin Loader security patterns,
    dynamic imports are restricted to trusted namespaces to prevent arbitrary
    code execution via malicious contract.yaml configurations.

    Args:
        class_name: The name of the handler class to load.
        module_path: The fully qualified module path.

    Returns:
        The handler class type.

    Raises:
        ProtocolConfigurationError: If the module namespace is not allowed
            (NAMESPACE_NOT_ALLOWED / HANDLER_LOADER_013).
        ProtocolConfigurationError: If the module or class cannot be loaded.
    """
    # Security: Validate namespace before import
    # Error code: NAMESPACE_NOT_ALLOWED (HANDLER_LOADER_013)
    if not any(module_path.startswith(ns) for ns in ALLOWED_NAMESPACES):
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="load_handler_class",
            target_name=f"{module_path}.{class_name}",
        )
        raise ProtocolConfigurationError(
            f"Handler module namespace not allowed: {module_path}. "
            f"Allowed namespaces: {', '.join(ALLOWED_NAMESPACES)}. "
            "Error code: NAMESPACE_NOT_ALLOWED (HANDLER_LOADER_013)",
            context=ctx,
        )

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="load_handler_class",
            target_name=f"{module_path}.{class_name}",
        )
        raise ProtocolConfigurationError(
            f"Handler module not found: {module_path}. "
            f"Verify the module path is correct and the package is installed. "
            f"Error code: MODULE_NOT_FOUND (HANDLER_LOADER_010)",
            context=ctx,
        ) from e
    except ImportError as e:
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="load_handler_class",
            target_name=f"{module_path}.{class_name}",
        )
        raise ProtocolConfigurationError(
            f"Failed to import handler module: {module_path}. "
            f"Check for syntax errors or missing dependencies. "
            f"Error code: IMPORT_ERROR (HANDLER_LOADER_012)",
            context=ctx,
        ) from e

    if not hasattr(module, class_name):
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="load_handler_class",
            target_name=f"{module_path}.{class_name}",
        )
        raise ProtocolConfigurationError(
            f"Handler class '{class_name}' not found in module '{module_path}'. "
            f"Verify the class name matches the contract.yaml handler.name field. "
            f"Error code: CLASS_NOT_FOUND (HANDLER_LOADER_011)",
            context=ctx,
        )

    handler_class: type = getattr(module, class_name)
    return handler_class


if TYPE_CHECKING:
    from omnibase_infra.handlers import HandlerConsul
    from omnibase_infra.projectors import ProjectionReaderRegistration
    from omnibase_infra.runtime import ProjectorShell


class RegistryInfraNodeRegistrationOrchestrator:
    """Handler registry for NodeRegistrationOrchestrator.

    This registry provides a static factory method for creating handler registries
    used by the NodeRegistrationOrchestrator. It follows the ONEX registry pattern
    with the naming convention ``RegistryInfra<NodeName>``.

    Why a class instead of a function?
        ONEX registry pattern (CLAUDE.md) requires registry classes. This enables:

        - **Centralized wiring**: All handler creation logic in one place
        - **Contract alignment**: Maps event models to handlers per contract.yaml
        - **Testability**: Mock dependencies for unit testing
        - **Extensibility**: Subclassing for specialized registries

    Usage:
        ```python
        registry = RegistryInfraNodeRegistrationOrchestrator.create_registry(
            projection_reader=reader,
            projector=projector,
            consul_handler=consul_handler,
        )
        handler = registry.get_handler_by_id("handler-node-introspected")
        result = await handler.handle(envelope)
        ```
    """

    @staticmethod
    def create_registry(
        projection_reader: ProjectionReaderRegistration,
        projector: ProjectorShell | None = None,
        consul_handler: HandlerConsul | None = None,
        *,
        require_heartbeat_handler: bool = True,
    ) -> ServiceHandlerRegistry:
        """Create a frozen ServiceHandlerRegistry with all handlers wired.

        This is the preferred method for creating handler registries. It returns
        a thread-safe, frozen registry that can be used by the orchestrator.

        Contract-Driven Loading:
            Handlers are loaded dynamically from contract.yaml using the Handler
            Plugin Loader pattern. The contract.yaml handler_routing section defines
            handler classes and modules that are imported at runtime.

        Handler Registration:
            The contract.yaml defines 4 handlers:
            - ModelNodeIntrospectionEvent -> HandlerNodeIntrospected (always registered)
            - ModelRuntimeTick -> HandlerRuntimeTick (always registered)
            - ModelNodeRegistrationAcked -> HandlerNodeRegistrationAcked (always registered)
            - ModelNodeHeartbeatEvent -> HandlerNodeHeartbeat (requires projector)

        Fail-Fast Behavior:
            By default (require_heartbeat_handler=True), this method raises
            ProtocolConfigurationError if projector is None, because the contract
            defines heartbeat routing which requires a projector for persistence.

            This fail-fast approach prevents silent failures where heartbeat events
            would be silently dropped at runtime due to missing handler registration.

        Args:
            projection_reader: Projection reader for state queries.
            projector: Projector for state persistence. Required for
                HandlerNodeHeartbeat to persist heartbeat timestamps.
            consul_handler: Optional Consul handler for service registration.
            require_heartbeat_handler: If True (default), raises ProtocolConfigurationError
                when projector is None. Set to False only for testing scenarios where
                heartbeat functionality is intentionally disabled. This creates a
                contract.yaml mismatch (4 handlers defined, only 3 registered).

        Returns:
            Frozen ServiceHandlerRegistry with handlers registered:
            - 4 handlers when projector is provided
            - 3 handlers when projector is None and require_heartbeat_handler=False

        Raises:
            ProtocolConfigurationError: If projector is None and
                require_heartbeat_handler is True (default).
            ProtocolConfigurationError: If contract.yaml is missing or invalid.
            ProtocolConfigurationError: If a handler class cannot be loaded.

        Example:
            ```python
            # Production usage - projector required
            registry = RegistryInfraNodeRegistrationOrchestrator.create_registry(
                projection_reader=reader,
                projector=projector,
                consul_handler=consul_handler,
            )

            # Testing without heartbeat support (explicit opt-in)
            registry = RegistryInfraNodeRegistrationOrchestrator.create_registry(
                projection_reader=reader,
                projector=None,
                require_heartbeat_handler=False,  # Explicitly disable
            )

            # Get handler by ID
            handler = registry.get_handler_by_id("handler-node-introspected")

            # Or iterate all handlers
            for handler in registry.get_handlers():
                print(f"{handler.handler_id}: {handler.message_types}")
            ```
        """
        # Fail-fast: contract.yaml defines heartbeat routing which requires projector
        if projector is None and require_heartbeat_handler:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="create_registry",
                target_name="RegistryInfraNodeRegistrationOrchestrator",
            )
            raise ProtocolConfigurationError(
                "Heartbeat handler requires projector but none was provided. "
                "The contract.yaml defines ModelNodeHeartbeatEvent routing which "
                "requires a ProjectorShell instance to persist heartbeat updates. "
                "Either provide a projector or set require_heartbeat_handler=False "
                "to explicitly disable heartbeat support (testing only).",
                context=ctx,
            )

        # Load handler routing configuration from contract.yaml
        # Uses shared loader from omnibase_infra.runtime.contract_loaders (OMN-1316)
        contract_path = Path(__file__).parent.parent / "contract.yaml"
        handler_configs = load_handler_class_info_from_contract(contract_path)

        # Map of handler dependencies by handler class name.
        # Each handler class has specific dependencies based on its constructor.
        #
        # MAINTENANCE: When adding a new handler to contract.yaml, you MUST add
        # a corresponding entry here with the handler's constructor arguments.
        # The keys match handler class names from contract.yaml handler_routing.handlers[].handler.name.
        # See module docstring "Maintenance Note" for details.
        handler_dependencies: dict[str, dict[str, object]] = {
            "HandlerNodeIntrospected": {
                "projection_reader": projection_reader,
                "projector": projector,
                "consul_handler": consul_handler,
            },
            "HandlerRuntimeTick": {
                "projection_reader": projection_reader,
            },
            "HandlerNodeRegistrationAcked": {
                "projection_reader": projection_reader,
            },
            "HandlerNodeHeartbeat": {
                "projection_reader": projection_reader,
                "projector": projector,
            },
        }

        registry = ServiceHandlerRegistry()

        # Load and instantiate handlers from contract configuration
        for handler_config in handler_configs:
            handler_class_name = handler_config["handler_class"]
            handler_module = handler_config["handler_module"]

            # Special handling for HandlerNodeHeartbeat - requires projector
            if handler_class_name == "HandlerNodeHeartbeat":
                if projector is None:
                    # Skip heartbeat handler if no projector (require_heartbeat_handler=False)
                    logger.warning(
                        "HandlerNodeHeartbeat NOT registered: require_heartbeat_handler=False. "
                        "This creates a contract.yaml mismatch (4 handlers defined, only 3 registered). "
                        "Heartbeat events (ModelNodeHeartbeatEvent) will NOT be handled. "
                        "This configuration is intended for testing only."
                    )
                    continue

            # Load handler class dynamically
            handler_cls = _load_handler_class(handler_class_name, handler_module)

            # Get dependencies for this handler
            deps = handler_dependencies.get(handler_class_name, {})
            if not deps:
                ctx = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="create_registry",
                    target_name="RegistryInfraNodeRegistrationOrchestrator",
                )
                raise ProtocolConfigurationError(
                    f"No dependency configuration found for handler {handler_class_name}. "
                    "Update handler_dependencies map with required constructor arguments.",
                    context=ctx,
                )

            # Filter dependencies for handler instantiation.
            #
            # WHY projection_reader is ALWAYS included (even if None):
            #   - projection_reader is a REQUIRED dependency for all handlers
            #   - Even if the value is None, we pass it so handlers can perform
            #     their own validation and raise clear errors if missing
            #   - This is intentional: handlers should fail-fast with clear messages
            #     rather than silently receiving no projection_reader parameter
            #
            # WHY projector and consul_handler are only included when not None:
            #   - These are OPTIONAL dependencies used by specific handlers
            #   - projector: Only handlers that persist state changes need this
            #     (e.g., HandlerNodeHeartbeat for updating heartbeat timestamps)
            #   - consul_handler: Only handlers that interact with service discovery
            #     (e.g., HandlerNodeIntrospected for dual Consul registration)
            #   - Passing None would override handler defaults or cause TypeErrors
            #
            # Summary:
            #   projection_reader: Required by all handlers -> always pass
            #   projector: Optional -> pass only if provided
            #   consul_handler: Optional -> pass only if provided
            filtered_deps = {
                k: v
                for k, v in deps.items()
                if v is not None or k == "projection_reader"
            }

            # Instantiate handler with dependencies
            try:
                handler_instance = handler_cls(**filtered_deps)
            except TypeError as e:
                ctx = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="create_registry",
                    target_name=handler_class_name,
                )
                raise ProtocolConfigurationError(
                    f"Failed to instantiate handler {handler_class_name}: {e}. "
                    "Check that handler_dependencies map matches handler constructor.",
                    context=ctx,
                ) from e

            # Validate handler implements ProtocolMessageHandler
            is_valid, missing = _validate_handler_protocol(handler_instance)
            if not is_valid:
                ctx = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="create_registry",
                    target_name=handler_class_name,
                )
                raise ProtocolConfigurationError(
                    f"Handler '{handler_class_name}' does not implement ProtocolMessageHandler. "
                    f"Missing required members: {', '.join(missing)}. "
                    f"Handlers must have: handler_id, category, message_types, node_kind properties "
                    f"and handle(envelope) method. "
                    f"Error code: PROTOCOL_NOT_IMPLEMENTED (HANDLER_LOADER_006)",
                    context=ctx,
                )

            # Register handler
            registry.register_handler(handler_instance)
            logger.debug(
                "Registered handler from contract: %s",
                handler_class_name,
            )

        # Freeze registry to make it thread-safe
        registry.freeze()

        return registry


__all__ = ["RegistryInfraNodeRegistrationOrchestrator"]
