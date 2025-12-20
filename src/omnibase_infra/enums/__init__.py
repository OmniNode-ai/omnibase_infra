# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Enumerations Module.

Provides infrastructure-specific enumerations for transport types,
protocol identification, policy classification, dispatch status,
message categories, topic types, topic standards, execution shape
validation, and other infrastructure concerns.

Exports:
    EnumDispatchStatus: Dispatch operation status enumeration
    EnumExecutionShapeViolation: Execution shape violation types for validation
    EnumHandlerType: Handler type enumeration for ONEX 4-node architecture
    EnumInfraTransportType: Infrastructure transport type enumeration
    EnumMessageCategory: Message category enumeration (EVENT, COMMAND, INTENT, PROJECTION)
    EnumPolicyType: Policy type enumeration for PolicyRegistry plugins
    EnumTopicStandard: Topic standard enumeration (ONEX_KAFKA, ENVIRONMENT_AWARE)
    EnumTopicType: Topic type enumeration (EVENTS, COMMANDS, INTENTS, SNAPSHOTS)
"""

from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus
from omnibase_infra.enums.enum_execution_shape_violation import (
    EnumExecutionShapeViolation,
)
from omnibase_infra.enums.enum_handler_type import EnumHandlerType
from omnibase_infra.enums.enum_infra_transport_type import EnumInfraTransportType
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.enums.enum_policy_type import EnumPolicyType
from omnibase_infra.enums.enum_topic_standard import EnumTopicStandard
from omnibase_infra.enums.enum_topic_type import EnumTopicType

__all__ = [
    "EnumDispatchStatus",
    "EnumExecutionShapeViolation",
    "EnumHandlerType",
    "EnumInfraTransportType",
    "EnumMessageCategory",
    "EnumPolicyType",
    "EnumTopicStandard",
    "EnumTopicType",
]
