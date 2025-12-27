# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry Models Module.

This module provides Pydantic models for message type registry functionality
in the ONEX runtime dispatch infrastructure.

Models:
    - ModelDomainConstraint: Domain ownership rules and cross-domain constraints
    - ModelMessageTypeEntry: Registry entry for message type to handler mappings

Related:
    - OMN-937: Central Message Type Registry implementation
    - runtime.registry: Registry implementation using these models

.. versionadded:: 0.5.0
"""

from omnibase_infra.models.registry.model_domain_constraint import (
    ModelDomainConstraint,
)
from omnibase_infra.models.registry.model_message_type_entry import (
    ModelMessageTypeEntry,
)

__all__: list[str] = [
    "ModelDomainConstraint",
    "ModelMessageTypeEntry",
]
