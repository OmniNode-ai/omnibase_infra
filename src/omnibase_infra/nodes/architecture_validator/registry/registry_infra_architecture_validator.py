# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry for NodeArchitectureValidator.

Provides dependency injection registration for the architecture validator node.
This registry follows the ONEX pattern of container-managed service registration.

Usage:
    ```python
    from omnibase_core.models.container import ModelONEXContainer
    from omnibase_infra.nodes.architecture_validator.registry import (
        RegistryInfraArchitectureValidator,
    )

    container = ModelONEXContainer()
    RegistryInfraArchitectureValidator.register(container)

    # Later, resolve the validator
    validator = container.get_service(NodeArchitectureValidator)
    ```

Related:
    - Ticket: OMN-1099 (Architecture Validator)
    - NodeArchitectureValidator: The COMPUTE node this registry registers
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

from omnibase_infra.nodes.architecture_validator.node import NodeArchitectureValidator


class RegistryInfraArchitectureValidator:
    """Registry for architecture validator components.

    This registry provides a static method to register the NodeArchitectureValidator
    with the ONEX dependency injection container. The validator is registered as
    an instance after being created with the container reference.

    Thread Safety:
        Registration is typically done at startup before the container is frozen.
        After registration, the container provides thread-safe resolution.

    Example:
        ```python
        from omnibase_core.models.container import ModelONEXContainer
        from omnibase_infra.nodes.architecture_validator.registry import (
            RegistryInfraArchitectureValidator,
        )

        container = ModelONEXContainer()
        await RegistryInfraArchitectureValidator.register(container)
        ```
    """

    @staticmethod
    async def register(container: ModelONEXContainer) -> None:
        """Register architecture validator with container.

        Registers the NodeArchitectureValidator as a service in the ONEX
        dependency injection container. The validator is created immediately
        and registered as a singleton instance.

        Args:
            container: The DI container to register with.

        Note:
            If the container's service_registry is not available (e.g., in
            minimal container configurations), this method will log a warning
            but not raise an exception. This allows for graceful degradation
            in environments where full DI is not configured.

        Example:
            ```python
            container = ModelONEXContainer()
            await RegistryInfraArchitectureValidator.register(container)

            # Resolve when needed
            validator = await container.service_registry.resolve_service(
                NodeArchitectureValidator
            )
            result = await validator.compute(request)
            ```
        """
        from omnibase_core.enums import EnumInjectionScope

        # Check if service_registry is available
        if container.service_registry is None:
            # Container doesn't have full DI support - skip registration
            # This allows the code to work with minimal container configurations
            return

        # Create instance and register (factory registration not supported in v1.0)
        validator_instance = NodeArchitectureValidator(container)
        await container.service_registry.register_instance(
            interface=NodeArchitectureValidator,
            instance=validator_instance,
            scope=EnumInjectionScope.GLOBAL,
        )


__all__ = ["RegistryInfraArchitectureValidator"]
