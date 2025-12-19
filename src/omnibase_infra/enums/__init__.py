# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Enumerations Module.

Provides infrastructure-specific enumerations for transport types,
protocol identification, policy classification, dispatch status,
message categories, topic types, topic standards, and other infrastructure concerns.

Exports:
    EnumDispatchStatus: Dispatch operation status enumeration
    EnumInfraTransportType: Infrastructure transport type enumeration
    EnumMessageCategory: Message category enumeration (EVENT, COMMAND, INTENT)
    EnumPolicyType: Policy type enumeration for PolicyRegistry plugins
    EnumTopicStandard: Topic standard enumeration (ONEX_KAFKA, ENVIRONMENT_AWARE)
    EnumTopicType: Topic type enumeration (EVENTS, COMMANDS, INTENTS, SNAPSHOTS)
"""

from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus
from omnibase_infra.enums.enum_infra_transport_type import EnumInfraTransportType
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.enums.enum_policy_type import EnumPolicyType
from omnibase_infra.enums.enum_topic_standard import EnumTopicStandard
from omnibase_infra.enums.enum_topic_type import EnumTopicType

__all__ = [
    "EnumDispatchStatus",
    "EnumInfraTransportType",
    "EnumMessageCategory",
    "EnumPolicyType",
    "EnumTopicStandard",
    "EnumTopicType",
]
