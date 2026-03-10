# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Registry for Architecture Validator node.

The infrastructure registry for the
Architecture Validator node, following the naming convention:
    RegistryInfraArchitectureValidator

Usage:
    ```python
    from omnibase_core.models.container import ModelONEXContainer
    from omnibase_infra.nodes.node_architecture_validator.registry import (
        RegistryInfraArchitectureValidator,
    )

    container = ModelONEXContainer()
    RegistryInfraArchitectureValidator.register(container)
    ```

Related:
    - Ticket: OMN-1099 (Architecture Validator)
"""

from omnibase_infra.nodes.node_architecture_validator.registry.registry_infra_architecture_validator import (
    RegistryInfraArchitectureValidator,
)

__all__: list[str] = ["RegistryInfraArchitectureValidator"]
