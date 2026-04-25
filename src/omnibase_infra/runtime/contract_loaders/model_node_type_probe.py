# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Probe model for extracting node_type from raw YAML contract dicts.

Used internally by handler_routing_loader._dispatch_contract_model() to validate
the node_type field without requiring the full contract schema. Kept in its own
file to satisfy the one-model-per-file architecture invariant.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.enums.enum_node_type import EnumNodeType
from omnibase_core.mixins.mixin_node_type_validator import MixinNodeTypeValidator


class ModelNodeTypeProbe(MixinNodeTypeValidator, BaseModel):
    """Minimal probe model for extracting node_type from raw YAML contract dicts.

    Inherits MixinNodeTypeValidator to accept lowercase aliases ("effect",
    "compute", "reducer", "orchestrator") and full string enum values
    ("EFFECT_GENERIC", etc.). Uses extra='ignore' to accept any contract YAML
    structure without schema enforcement.

    Used by _dispatch_contract_model() in handler_routing_loader.py.
    """

    model_config = ConfigDict(extra="ignore")

    node_type: EnumNodeType = Field(..., description="Node type from contract YAML")


__all__ = ["ModelNodeTypeProbe"]
