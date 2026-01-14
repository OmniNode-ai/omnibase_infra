# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Routing models for declarative orchestrator configuration.

These models define the structure for routing configuration
loaded from contract.yaml files. They enable declarative mapping of
event models to handler implementations.

Note:
    These models are local to omnibase_infra as they support infrastructure-
    specific routing patterns. They may be promoted to omnibase_core
    in the future if they become part of the standard ONEX contract schema.
"""

from omnibase_infra.models.routing.model_routing_entry import (
    ModelRoutingEntry,
)
from omnibase_infra.models.routing.model_routing_subcontract import (
    ModelRoutingSubcontract,
)

__all__ = [
    "ModelRoutingEntry",
    "ModelRoutingSubcontract",
]
