# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract dependency resolver for ONEX nodes.

This module provides ContractDependencyResolver, which reads protocol
dependencies from a node's contract.yaml and resolves them from the
container's service_registry.

Part of OMN-1732: Runtime dependency injection for zero-code nodes.

Architecture:
    - Container: service provider (owns protocol instances)
    - Runtime (this resolver): wiring authority (resolves + validates)
    - Node: consumer (receives resolved deps via constructor)
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any, cast

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    ModelInfraErrorContext,
    ProtocolDependencyResolutionError,
)
from omnibase_infra.models.runtime.model_resolved_dependencies import (
    ModelResolvedDependencies,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer
    from omnibase_core.models.contracts import ModelContractBase

logger = logging.getLogger(__name__)


class ContractDependencyResolver:
    """Resolves protocol dependencies from contracts using container.

    Reads the `dependencies` field from a node's contract, filters for
    protocol-type dependencies, imports the protocol classes, and resolves
    instances from the container's service_registry.

    This resolver implements the "fail-fast" principle: if any required
    protocol cannot be resolved, it raises ProtocolDependencyResolutionError
    immediately rather than allowing node creation with missing dependencies.

    Example:
        >>> from omnibase_core.container import ModelONEXContainer
        >>>
        >>> container = ModelONEXContainer()
        >>> # ... register protocols in container ...
        >>>
        >>> resolver = ContractDependencyResolver(container)
        >>> resolved = await resolver.resolve(node_contract)
        >>>
        >>> # Use resolved dependencies for node creation
        >>> node = NodeRegistry.create(container, dependencies=resolved)

    .. versionadded:: 0.x.x
        Part of OMN-1732 runtime dependency injection.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the resolver with a container.

        Args:
            container: ONEX container with service_registry for protocol resolution.
        """
        self._container = container

    async def resolve(
        self,
        contract: ModelContractBase,
        *,
        allow_missing: bool = False,
    ) -> ModelResolvedDependencies:
        """Resolve protocol dependencies from a contract.

        Reads contract.dependencies, filters for protocol-type entries,
        and resolves each from the container's service_registry.

        Args:
            contract: The node contract containing dependency declarations.
            allow_missing: If True, skip missing protocols instead of raising.
                          Default False (fail-fast behavior).

        Returns:
            ModelResolvedDependencies containing resolved protocol instances.

        Raises:
            ProtocolDependencyResolutionError: If any required protocol cannot
                be resolved and allow_missing is False.

        Example:
            >>> resolved = await resolver.resolve(contract)
            >>> adapter = resolved.get("ProtocolPostgresAdapter")
        """
        # Extract protocol dependencies from contract
        protocol_deps = self._extract_protocol_dependencies(contract)

        if not protocol_deps:
            logger.debug(
                "No protocol dependencies declared in contract",
                extra={"contract_name": getattr(contract, "name", "unknown")},
            )
            return ModelResolvedDependencies()

        # Resolve each protocol from container
        # ONEX_EXCLUDE: any_type - dict holds heterogeneous protocol instances resolved at runtime
        resolved: dict[str, Any] = {}
        missing: list[str] = []
        resolution_errors: dict[str, str] = {}

        for dep in protocol_deps:
            class_name = dep.get("class_name")
            module_path = dep.get("module")

            if not class_name:
                logger.warning(
                    "Protocol dependency missing class_name, skipping",
                    extra={"dependency": dep},
                )
                continue

            try:
                # Import the protocol class
                protocol_class = self._import_protocol_class(class_name, module_path)

                # Resolve from container
                instance = await self._resolve_from_container(protocol_class)
                resolved[class_name] = instance

                logger.debug(
                    "Resolved protocol dependency",
                    extra={
                        "protocol": class_name,
                        "module_path": module_path,
                    },
                )

            except Exception as e:
                error_msg = str(e)
                resolution_errors[class_name] = error_msg
                missing.append(class_name)

                logger.warning(
                    "Failed to resolve protocol dependency",
                    extra={
                        "protocol": class_name,
                        "module_path": module_path,
                        "error": error_msg,
                    },
                )

        # Fail-fast if any protocols are missing
        if missing and not allow_missing:
            contract_name = getattr(contract, "name", "unknown")
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="resolve_dependencies",
                target_name=contract_name,
            )

            error_details = "\n".join(
                f"  - {proto}: {resolution_errors.get(proto, 'unknown error')}"
                for proto in missing
            )

            raise ProtocolDependencyResolutionError(
                f"Cannot create node '{contract_name}': missing required protocols.\n"
                f"The following protocols are declared in contract.yaml but could not "
                f"be resolved from the container:\n{error_details}\n\n"
                f"Ensure these protocols are registered in the container before "
                f"node creation via container.service_registry.register_instance().",
                context=context,
                missing_protocols=missing,
                node_name=contract_name,
            )

        return ModelResolvedDependencies(protocols=resolved)

    def _extract_protocol_dependencies(
        self,
        contract: ModelContractBase,
    ) -> list[dict[str, str]]:
        """Extract protocol-type dependencies from contract.

        Supports multiple detection patterns:
        1. `type` field = "protocol" (YAML contract style)
        2. `is_protocol()` method returns True
        3. `dependency_type` attribute with value "PROTOCOL"

        Args:
            contract: The contract to extract dependencies from.

        Returns:
            List of dependency dicts with keys: name, type, class_name, module
        """
        dependencies: list[dict[str, str]] = []

        # Get dependencies attribute if present
        contract_deps = getattr(contract, "dependencies", None)
        if not contract_deps:
            return dependencies

        for dep in contract_deps:
            is_protocol = self._is_protocol_dependency(dep)

            if is_protocol:
                dep_dict: dict[str, str] = {}

                # Extract fields - handle both attribute and dict access
                if hasattr(dep, "name"):
                    dep_dict["name"] = str(dep.name) if dep.name else ""
                if hasattr(dep, "class_name"):
                    dep_dict["class_name"] = (
                        str(dep.class_name) if dep.class_name else ""
                    )
                if hasattr(dep, "module"):
                    dep_dict["module"] = str(dep.module) if dep.module else ""

                # Only add if we have a class_name
                if dep_dict.get("class_name"):
                    dependencies.append(dep_dict)

        return dependencies

    # ONEX_EXCLUDE: any_type - dep can be ModelContractDependency or dict from various sources
    def _is_protocol_dependency(self, dep: Any) -> bool:
        """Check if a dependency is a protocol dependency.

        Supports multiple detection patterns:
        1. `type` field = "protocol" (YAML contract style)
        2. `is_protocol()` method returns True
        3. `dependency_type` attribute with value "PROTOCOL"

        Args:
            dep: The dependency object to check.

        Returns:
            True if this is a protocol dependency, False otherwise.
        """
        # Check is_protocol() method first (most explicit)
        if hasattr(dep, "is_protocol") and callable(dep.is_protocol):
            if dep.is_protocol():
                return True

        # Check dependency_type attribute (enum or string)
        if hasattr(dep, "dependency_type"):
            dep_type_val = dep.dependency_type
            if hasattr(dep_type_val, "value"):
                # Enum with .value
                if str(dep_type_val.value).upper() == "PROTOCOL":
                    return True
            elif str(dep_type_val).upper() == "PROTOCOL":
                return True

        # Check type field (YAML contract style: type: "protocol")
        if hasattr(dep, "type"):
            type_val = getattr(dep, "type", "")
            if str(type_val).lower() == "protocol":
                return True

        return False

    def _import_protocol_class(
        self,
        class_name: str,
        module_path: str | None,
    ) -> type:
        """Import a protocol class by name and module.

        Args:
            class_name: The class name to import
            module_path: The module path (required for import)

        Returns:
            The imported class type.

        Raises:
            ImportError: If the class cannot be imported.
        """
        if not module_path:
            raise ImportError(
                f"Protocol '{class_name}' has no module path specified in contract. "
                f"Cannot import without module path."
            )

        try:
            module = importlib.import_module(module_path)
            protocol_class = getattr(module, class_name)
            return cast("type", protocol_class)
        except ModuleNotFoundError as e:
            raise ImportError(
                f"Module '{module_path}' not found for protocol '{class_name}': {e}"
            ) from e
        except AttributeError as e:
            raise ImportError(
                f"Class '{class_name}' not found in module '{module_path}': {e}"
            ) from e

    # ONEX_EXCLUDE: any_type - returns protocol instance, type varies by resolved protocol class
    async def _resolve_from_container(self, protocol_class: type) -> Any:
        """Resolve a protocol instance from the container.

        Args:
            protocol_class: The protocol class type to resolve.

        Returns:
            The resolved protocol instance.

        Raises:
            RuntimeError: If container.service_registry is None.
            Exception: If resolution fails (propagated from container).
        """
        if self._container.service_registry is None:
            raise RuntimeError(
                "Container service_registry is None. "
                "Ensure container is properly initialized with service_registry enabled."
            )

        return await self._container.service_registry.resolve_service(protocol_class)


__all__ = ["ContractDependencyResolver"]
