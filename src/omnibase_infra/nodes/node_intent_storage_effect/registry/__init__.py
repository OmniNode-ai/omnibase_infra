# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Registry for Intent Storage Effect Node.

This module exports the RegistryInfraIntentStorage for dependency registration.

Usage:
    >>> from omnibase_infra.nodes.node_intent_storage_effect.registry import (
    ...     RegistryInfraIntentStorage,
    ... )
    >>> RegistryInfraIntentStorage.register(container)
"""

from .registry_infra_intent_storage import RegistryInfraIntentStorage

__all__ = ["RegistryInfraIntentStorage"]
