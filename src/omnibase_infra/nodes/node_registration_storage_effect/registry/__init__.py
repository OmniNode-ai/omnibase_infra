# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry for Registration Storage Effect Node.

This module exports the dependency injection registry for
NodeRegistrationStorageEffect.

Available Exports:
    - RegistryInfraRegistrationStorage: Registry for node dependencies

Usage:
    >>> from omnibase_core.models.container import ModelONEXContainer
    >>> from omnibase_infra.nodes.node_registration_storage_effect.registry import (
    ...     RegistryInfraRegistrationStorage,
    ... )
    >>>
    >>> container = ModelONEXContainer()
    >>> RegistryInfraRegistrationStorage.register(container)
"""

from .registry_infra_registration_storage import RegistryInfraRegistrationStorage

__all__ = ["RegistryInfraRegistrationStorage"]
