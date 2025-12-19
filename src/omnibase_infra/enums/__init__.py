# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Enumerations Module.

Provides infrastructure-specific enumerations for transport types,
protocol identification, policy classification, execution shape
validation, and other infrastructure concerns.

Exports:
    EnumInfraTransportType: Infrastructure transport type enumeration
    EnumPolicyType: Policy type enumeration for PolicyRegistry plugins
    EnumHandlerType: Handler type enumeration for ONEX 4-node architecture
    EnumMessageCategory: Message category enumeration for event-driven architecture
    EnumExecutionShapeViolation: Execution shape violation types for validation
"""

from omnibase_infra.enums.enum_execution_shape_violation import (
    EnumExecutionShapeViolation,
)
from omnibase_infra.enums.enum_handler_type import EnumHandlerType
from omnibase_infra.enums.enum_infra_transport_type import EnumInfraTransportType
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.enums.enum_policy_type import EnumPolicyType

__all__ = [
    "EnumExecutionShapeViolation",
    "EnumHandlerType",
    "EnumInfraTransportType",
    "EnumMessageCategory",
    "EnumPolicyType",
]
