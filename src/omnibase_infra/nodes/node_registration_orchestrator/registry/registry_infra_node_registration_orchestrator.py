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
    All handlers implement ProtocolContainerAware directly with:
    - handler_id, category, message_types, node_kind properties
    - handle(envelope) -> ModelHandlerOutput signature

    Handlers are registered directly with ServiceHandlerRegistry without
    adapter classes.

Handler Dependencies:
    All handlers require ProjectionReaderRegistration for state queries.
    Some handlers optionally accept:
    - ProjectorShell: For projection persistence
    - HandlerConsul: For Consul service registration (dual registration)

Handler Dependency Map - Design Trade-off:
    The ``handler_dependencies`` dict in ``create_registry()`` requires manual
    updates when adding handlers to contract.yaml. This is an INTENTIONAL design
    trade-off that prioritizes type safety, testability, and security over
    convenience.

    **Why NOT Auto-Discovery:**

    1. **Type Safety**: Explicit dependency declarations are validated at startup.
       Auto-discovery via reflection (e.g., inspect.signature()) would defer
       errors to runtime when handlers are instantiated.

    2. **Testability**: Explicit dependencies can be easily mocked in tests.
       The dependency map serves as documentation of what each handler needs,
       making test setup straightforward.

    3. **Security**: No reflection-based injection means no attack surface for
       dependency injection exploits. We know exactly what gets passed to each
       handler constructor.

    4. **Clarity**: The dependency map is a clear, auditable record of handler
       wiring. New developers can see at a glance what each handler receives.

    **Maintenance Requirement:**

    When adding a new handler to contract.yaml, you MUST also update the
    ``handler_dependencies`` dict with the handler's constructor arguments.
    Failure to do so raises ProtocolConfigurationError at startup with a clear
    message explaining how to fix it.

    This fail-fast behavior catches configuration errors immediately rather than
    at runtime when the handler is first invoked.

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
from typing import TYPE_CHECKING, cast

from omnibase_core.services.service_handler_registry import ServiceHandlerRegistry
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.protocols import ProtocolContainerAware
from omnibase_infra.runtime.constants_security import (
    TRUSTED_HANDLER_NAMESPACE_PREFIXES,
)
from omnibase_infra.runtime.contract_loaders import (
    load_handler_class_info_from_contract,
)

logger = logging.getLogger(__name__)

# Security: Use centralized namespace allowlist for dynamic handler imports
# Per CLAUDE.md Handler Plugin Loader security patterns, only trusted namespaces
# are allowed for dynamic imports to prevent arbitrary code execution.
# Error code: NAMESPACE_NOT_ALLOWED (HANDLER_LOADER_013)
# NOTE: Aliased for backwards compatibility - prefer importing directly from
# constants_security for new code.
ALLOWED_NAMESPACES: tuple[str, ...] = TRUSTED_HANDLER_NAMESPACE_PREFIXES


def _validate_handler_protocol(handler: object) -> tuple[bool, list[str]]:
    """Validate handler implements ProtocolContainerAware via duck typing.

    Uses duck typing to verify the handler has the required properties and
    methods for ProtocolContainerAware compliance. Per ONEX conventions,
    protocol compliance is verified via structural typing rather than
    isinstance checks.

    Protocol Requirements (ProtocolContainerAware):
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


def _load_handler_class(class_name: str, module_path: str) -> type[object]:
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
        ctx = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.RUNTIME,
            operation="load_handler_class",
            target_name=f"{module_path}.{class_name}",
        )
        raise ProtocolConfigurationError(
            f"Handler module namespace not allowed: {module_path}. "
            f"Allowed namespaces: {', '.join(ALLOWED_NAMESPACES)}. "
            f"Use a handler module from an allowed namespace or update ALLOWED_NAMESPACES. "
            f"Error code: NAMESPACE_NOT_ALLOWED (HANDLER_LOADER_013)",
            context=ctx,
        )

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        ctx = ModelInfraErrorContext.with_correlation(
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
        ctx = ModelInfraErrorContext.with_correlation(
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
        ctx = ModelInfraErrorContext.with_correlation(
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

    handler_class: type[object] = getattr(module, class_name)
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
            ctx = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="create_registry",
                target_name="RegistryInfraNodeRegistrationOrchestrator",
            )
            raise ProtocolConfigurationError(
                "Heartbeat handler requires projector but none was provided. "
                "The contract.yaml defines ModelNodeHeartbeatEvent routing which "
                "requires a ProjectorShell instance to persist heartbeat updates. "
                "Either provide a projector or set require_heartbeat_handler=False "
                "to explicitly disable heartbeat support (testing only). "
                "Error code: PROJECTOR_REQUIRED (HANDLER_LOADER_060)",
                context=ctx,
            )

        # Load handler routing configuration from contract.yaml
        # Uses shared loader from omnibase_infra.runtime.contract_loaders (OMN-1316)
        contract_path = Path(__file__).parent.parent / "contract.yaml"
        handler_configs = load_handler_class_info_from_contract(contract_path)

        # =====================================================================
        # HANDLER DEPENDENCY MAP - Explicit Wiring (Intentional Design)
        # =====================================================================
        #
        # This map explicitly declares constructor arguments for each handler.
        # This is an INTENTIONAL design trade-off over auto-discovery.
        #
        # WHY EXPLICIT OVER AUTO-DISCOVERY:
        #   - Type Safety: Validated at startup, not at runtime invocation
        #   - Testability: Dependencies are easily mocked without reflection
        #   - Security: No reflection-based injection attack surface
        #   - Auditability: Clear record of what each handler receives
        #
        # MAINTENANCE REQUIREMENT:
        #   When adding a handler to contract.yaml, add an entry here.
        #   Keys must match handler_routing.handlers[].handler.name in contract.yaml.
        #   Missing entries cause ProtocolConfigurationError at startup (fail-fast).
        #
        # See module docstring "Handler Dependency Map - Design Trade-off" for details.
        # =====================================================================
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

            # Get dependencies for this handler from the explicit dependency map.
            # Missing entries indicate a contract.yaml/registry mismatch that must be fixed.
            deps = handler_dependencies.get(handler_class_name, {})
            if not deps:
                ctx = ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="create_registry",
                    target_name="RegistryInfraNodeRegistrationOrchestrator",
                )
                raise ProtocolConfigurationError(
                    f"No dependency configuration found for handler '{handler_class_name}'. "
                    f"This handler is defined in contract.yaml but missing from the "
                    f"handler_dependencies map in create_registry(). "
                    f"\n\nTo fix: Add an entry to handler_dependencies:\n"
                    f"    '{handler_class_name}': {{\n"
                    f"        'projection_reader': projection_reader,\n"
                    f"        # Add other constructor args as needed\n"
                    f"    }},\n\n"
                    f"This explicit wiring is intentional for type safety and testability. "
                    f"See module docstring 'Handler Dependency Map - Design Trade-off'. "
                    f"Error code: MISSING_DEPENDENCY_CONFIG (HANDLER_LOADER_061)",
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
            #   projection_reader: Always pass (even if None) -> handlers validate and fail-fast
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
                ctx = ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="create_registry",
                    target_name=handler_class_name,
                )
                raise ProtocolConfigurationError(
                    f"Failed to instantiate handler {handler_class_name}: {e}. "
                    f"Check that handler_dependencies map matches handler constructor. "
                    f"Error code: HANDLER_INSTANTIATION_FAILED (HANDLER_LOADER_062)",
                    context=ctx,
                ) from e

            # Validate handler implements ProtocolContainerAware
            is_valid, missing = _validate_handler_protocol(handler_instance)
            if not is_valid:
                ctx = ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="create_registry",
                    target_name=handler_class_name,
                )
                raise ProtocolConfigurationError(
                    f"Handler '{handler_class_name}' does not implement ProtocolContainerAware. "
                    f"Missing required members: {', '.join(missing)}. "
                    f"Handlers must have: handler_id, category, message_types, node_kind properties "
                    f"and handle(envelope) method. "
                    f"Error code: PROTOCOL_NOT_IMPLEMENTED (HANDLER_LOADER_006)",
                    context=ctx,
                )

            # Register handler
            registry.register_handler(handler_instance)  # type: ignore[arg-type]
            logger.debug(
                "Registered handler from contract: %s",
                handler_class_name,
            )

        # Freeze registry to make it thread-safe
        registry.freeze()

        return registry


__all__ = ["RegistryInfraNodeRegistrationOrchestrator"]
