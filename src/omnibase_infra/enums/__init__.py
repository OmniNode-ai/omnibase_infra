# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Enumerations Module.

Provides infrastructure-specific enumerations for transport types,
protocol identification, policy classification, dispatch status,
message categories, topic types, topic standards, chain validation,
registration states, handler types, handler error types, handler source types,
node archetypes, introspection reasons, contract types, circuit breaker states, retry error categories,
any type violations, security validation, validation severity, selection strategies, backend types,
registration status, confirmation events, and other infrastructure concerns.

Exports:
    EnumAnyTypeViolation: Any type violation categories for strong typing validation
    EnumBackendType: Infrastructure backend types (CONSUL, POSTGRES)
    EnumChainViolationType: Chain violation types for correlation/causation validation
    EnumCircuitState: Circuit breaker states (CLOSED, OPEN, HALF_OPEN)
    EnumConfirmationEventType: Registration confirmation event types
    EnumConsumerGroupPurpose: Consumer group purpose (CONSUME, INTROSPECTION, REPLAY, AUDIT, BACKFILL)
    EnumContractType: Contract types for ONEX nodes (effect, compute, reducer, orchestrator)
    EnumDispatchStatus: Dispatch operation status enumeration
    EnumEnvironment: Deployment environment classification (DEVELOPMENT, STAGING, PRODUCTION, CI)
    EnumExecutionShapeViolation: Specific execution shape violation types
    EnumHandlerErrorType: Handler error types for validation and lifecycle
    EnumHandlerLoaderError: Handler loader error codes for plugin loading
    EnumResponseStatus: Handler response status (SUCCESS, ERROR)
    EnumHandlerSourceMode: Handler source modes for loading strategy (BOOTSTRAP, CONTRACT, HYBRID)
    EnumHandlerSourceType: Handler validation error source types
    EnumHandlerType: Handler architectural roles (INFRA_HANDLER, NODE_HANDLER)
    EnumHandlerTypeCategory: Behavioral classification (COMPUTE, EFFECT)
    EnumInfraTransportType: Infrastructure transport type enumeration
    EnumIntrospectionReason: Introspection event reasons (STARTUP, SHUTDOWN, etc.)
    EnumKafkaAcks: Kafka producer acknowledgment policy (ALL, NONE, LEADER, ALL_REPLICAS)
    EnumMessageCategory: Message categories (EVENT, COMMAND, INTENT)
    EnumNodeArchetype: 4-node architecture (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR)
    EnumNodeOutputType: Node output types for execution shape validation
    EnumNonRetryableErrorCategory: Non-retryable error categories for DLQ
    EnumPolicyType: Policy types for RegistryPolicy plugins
    EnumRegistrationState: Registration FSM states for two-way registration
    EnumRegistrationStatus: Registration workflow status (IDLE, PENDING, PARTIAL, COMPLETE, FAILED)
    EnumRegistryResponseStatus: Registry operation response status (SUCCESS, PARTIAL, FAILED)
    EnumRetryErrorCategory: Error categories for retry decision making
    EnumSecurityRuleId: Security validation rule identifiers for OMN-1098
    EnumSelectionStrategy: Selection strategies for capability-based discovery (FIRST, RANDOM, ROUND_ROBIN, LEAST_LOADED)
    EnumTopicStandard: Topic standards (ONEX_KAFKA, ENVIRONMENT_AWARE)
    EnumTopicType: Topic types (EVENTS, COMMANDS, INTENTS, SNAPSHOTS)
    EnumValidationSeverity: Validation error severity levels (ERROR, CRITICAL, WARNING)
"""

from omnibase_core.enums import EnumTopicType
from omnibase_infra.enums.enum_any_type_violation import EnumAnyTypeViolation
from omnibase_infra.enums.enum_backend_type import EnumBackendType
from omnibase_infra.enums.enum_capture_outcome import EnumCaptureOutcome
from omnibase_infra.enums.enum_capture_state import EnumCaptureState
from omnibase_infra.enums.enum_chain_violation_type import EnumChainViolationType
from omnibase_infra.enums.enum_circuit_state import EnumCircuitState
from omnibase_infra.enums.enum_confirmation_event_type import EnumConfirmationEventType
from omnibase_infra.enums.enum_consumer_group_purpose import EnumConsumerGroupPurpose
from omnibase_infra.enums.enum_contract_type import EnumContractType
from omnibase_infra.enums.enum_dedupe_strategy import EnumDedupeStrategy
from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus
from omnibase_infra.enums.enum_environment import EnumEnvironment
from omnibase_infra.enums.enum_execution_shape_violation import (
    EnumExecutionShapeViolation,
)
from omnibase_infra.enums.enum_handler_error_type import EnumHandlerErrorType
from omnibase_infra.enums.enum_handler_loader_error import EnumHandlerLoaderError
from omnibase_infra.enums.enum_handler_source_mode import EnumHandlerSourceMode
from omnibase_infra.enums.enum_handler_source_type import EnumHandlerSourceType
from omnibase_infra.enums.enum_handler_type import EnumHandlerType
from omnibase_infra.enums.enum_handler_type_category import EnumHandlerTypeCategory
from omnibase_infra.enums.enum_infra_transport_type import EnumInfraTransportType
from omnibase_infra.enums.enum_introspection_reason import EnumIntrospectionReason
from omnibase_infra.enums.enum_kafka_acks import EnumKafkaAcks
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.enums.enum_node_archetype import EnumNodeArchetype
from omnibase_infra.enums.enum_node_output_type import EnumNodeOutputType
from omnibase_infra.enums.enum_non_retryable_error_category import (
    EnumNonRetryableErrorCategory,
)
from omnibase_infra.enums.enum_policy_type import EnumPolicyType
from omnibase_infra.enums.enum_registration_state import EnumRegistrationState
from omnibase_infra.enums.enum_registration_status import EnumRegistrationStatus
from omnibase_infra.enums.enum_registry_response_status import (
    EnumRegistryResponseStatus,
)
from omnibase_infra.enums.enum_response_status import EnumResponseStatus
from omnibase_infra.enums.enum_retry_error_category import EnumRetryErrorCategory
from omnibase_infra.enums.enum_security_rule_id import EnumSecurityRuleId
from omnibase_infra.enums.enum_selection_strategy import EnumSelectionStrategy
from omnibase_infra.enums.enum_topic_standard import EnumTopicStandard
from omnibase_infra.enums.enum_validation_severity import EnumValidationSeverity

__all__: list[str] = [
    "EnumAnyTypeViolation",
    "EnumBackendType",
    "EnumCaptureOutcome",
    "EnumCaptureState",
    "EnumChainViolationType",
    "EnumCircuitState",
    "EnumConfirmationEventType",
    "EnumConsumerGroupPurpose",
    "EnumContractType",
    "EnumDedupeStrategy",
    "EnumDispatchStatus",
    "EnumEnvironment",
    "EnumExecutionShapeViolation",
    "EnumHandlerErrorType",
    "EnumHandlerLoaderError",
    "EnumHandlerSourceMode",
    "EnumHandlerSourceType",
    "EnumHandlerType",
    "EnumHandlerTypeCategory",
    "EnumInfraTransportType",
    "EnumIntrospectionReason",
    "EnumKafkaAcks",
    "EnumMessageCategory",
    "EnumNodeArchetype",
    "EnumNodeOutputType",
    "EnumNonRetryableErrorCategory",
    "EnumPolicyType",
    "EnumRegistrationState",
    "EnumRegistrationStatus",
    "EnumRegistryResponseStatus",
    "EnumResponseStatus",
    "EnumRetryErrorCategory",
    "EnumSecurityRuleId",
    "EnumSelectionStrategy",
    "EnumTopicStandard",
    "EnumTopicType",
    "EnumValidationSeverity",
]
