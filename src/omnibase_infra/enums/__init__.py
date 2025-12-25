# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Enumerations Module.

Provides infrastructure-specific enumerations for transport types,
protocol identification, policy classification, dispatch status,
message categories, topic types, topic standards, chain validation,
registration states, handler types, and other infrastructure concerns.

Exports:
    EnumChainViolationType: Chain violation type enumeration for correlation/causation validation
    EnumDispatchStatus: Dispatch operation status enumeration
    EnumHandlerType: Handler type enumeration for ONEX 4-node architecture (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR)
    EnumInfraTransportType: Infrastructure transport type enumeration
    EnumMessageCategory: Message category enumeration (EVENT, COMMAND, INTENT)
    EnumNodeOutputType: Node output type enumeration for execution shape validation
    EnumNonRetryableErrorCategory: Non-retryable error categories for DLQ and retry logic
    EnumPolicyType: Policy type enumeration for PolicyRegistry plugins
    EnumRegistrationState: Registration FSM state enumeration for two-way registration workflow
    EnumTopicStandard: Topic standard enumeration (ONEX_KAFKA, ENVIRONMENT_AWARE)
    EnumTopicType: Topic type enumeration (EVENTS, COMMANDS, INTENTS, SNAPSHOTS)
"""

from omnibase_core.enums import EnumTopicType

from omnibase_infra.enums.enum_chain_violation_type import EnumChainViolationType
from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus
from omnibase_infra.enums.enum_handler_type import EnumHandlerType
from omnibase_infra.enums.enum_infra_transport_type import EnumInfraTransportType
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.enums.enum_node_output_type import EnumNodeOutputType
from omnibase_infra.enums.enum_non_retryable_error_category import (
    EnumNonRetryableErrorCategory,
)
from omnibase_infra.enums.enum_policy_type import EnumPolicyType
from omnibase_infra.enums.enum_registration_state import EnumRegistrationState
from omnibase_infra.enums.enum_topic_standard import EnumTopicStandard

__all__ = [
    "EnumChainViolationType",
    "EnumDispatchStatus",
    "EnumHandlerType",
    "EnumInfraTransportType",
    "EnumMessageCategory",
    "EnumNodeOutputType",
    "EnumNonRetryableErrorCategory",
    "EnumPolicyType",
    "EnumRegistrationState",
    "EnumTopicStandard",
    "EnumTopicType",
]
