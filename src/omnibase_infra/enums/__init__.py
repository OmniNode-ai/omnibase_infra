# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Enumerations Module.

Provides infrastructure-specific enumerations for transport types,
protocol identification, policy classification, dispatch status,
message categories, topic types, topic standards, chain validation,
registration states, handler types, handler error types, handler source types,
node archetypes, introspection reasons, contract types, circuit breaker states, retry error categories,
and other infrastructure concerns.

Exports:
    EnumChainViolationType: Chain violation types for correlation/causation validation
    EnumCircuitState: Circuit breaker states (CLOSED, OPEN, HALF_OPEN)
    EnumContractType: Contract types for ONEX nodes (effect, compute, reducer, orchestrator)
    EnumDispatchStatus: Dispatch operation status enumeration
    EnumHandlerErrorType: Handler error types for validation and lifecycle
    EnumHandlerSourceType: Handler validation error source types
    EnumHandlerType: Handler architectural roles (INFRA_HANDLER, NODE_HANDLER)
    EnumHandlerTypeCategory: Behavioral classification (COMPUTE, EFFECT)
    EnumInfraTransportType: Infrastructure transport type enumeration
    EnumIntrospectionReason: Introspection event reasons (STARTUP, SHUTDOWN, etc.)
    EnumMessageCategory: Message categories (EVENT, COMMAND, INTENT)
    EnumNodeArchetype: 4-node architecture (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR)
    EnumNodeOutputType: Node output types for execution shape validation
    EnumNonRetryableErrorCategory: Non-retryable error categories for DLQ
    EnumPolicyType: Policy types for PolicyRegistry plugins
    EnumRegistrationState: Registration FSM states for two-way registration
    EnumRetryErrorCategory: Error categories for retry decision making
    EnumTopicStandard: Topic standards (ONEX_KAFKA, ENVIRONMENT_AWARE)
    EnumTopicType: Topic types (EVENTS, COMMANDS, INTENTS, SNAPSHOTS)
"""

from omnibase_core.enums import EnumTopicType

from omnibase_infra.enums.enum_chain_violation_type import EnumChainViolationType
from omnibase_infra.enums.enum_circuit_state import EnumCircuitState
from omnibase_infra.enums.enum_contract_type import EnumContractType
from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus
from omnibase_infra.enums.enum_handler_error_type import EnumHandlerErrorType
from omnibase_infra.enums.enum_handler_source_type import EnumHandlerSourceType
from omnibase_infra.enums.enum_handler_type import EnumHandlerType
from omnibase_infra.enums.enum_handler_type_category import EnumHandlerTypeCategory
from omnibase_infra.enums.enum_infra_transport_type import EnumInfraTransportType
from omnibase_infra.enums.enum_introspection_reason import EnumIntrospectionReason
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.enums.enum_node_archetype import EnumNodeArchetype
from omnibase_infra.enums.enum_node_output_type import EnumNodeOutputType
from omnibase_infra.enums.enum_non_retryable_error_category import (
    EnumNonRetryableErrorCategory,
)
from omnibase_infra.enums.enum_policy_type import EnumPolicyType
from omnibase_infra.enums.enum_registration_state import EnumRegistrationState
from omnibase_infra.enums.enum_retry_error_category import EnumRetryErrorCategory
from omnibase_infra.enums.enum_topic_standard import EnumTopicStandard

__all__: list[str] = [
    "EnumChainViolationType",
    "EnumCircuitState",
    "EnumContractType",
    "EnumDispatchStatus",
    "EnumHandlerErrorType",
    "EnumHandlerSourceType",
    "EnumHandlerType",
    "EnumHandlerTypeCategory",
    "EnumInfraTransportType",
    "EnumIntrospectionReason",
    "EnumMessageCategory",
    "EnumNodeArchetype",
    "EnumNodeOutputType",
    "EnumNonRetryableErrorCategory",
    "EnumPolicyType",
    "EnumRegistrationState",
    "EnumRetryErrorCategory",
    "EnumTopicStandard",
    "EnumTopicType",
]
