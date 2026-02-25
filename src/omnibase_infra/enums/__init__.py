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
    EnumAdjudicatorState: Validation adjudicator FSM states (COLLECTING, ADJUDICATING, VERDICT_EMITTED)
    EnumAnyTypeViolation: Any type violation categories for strong typing validation
    EnumAuthDecision: Authorization decision outcomes (ALLOW, DENY, SOFT_DENY)
    EnumAuthSource: Authorization source classification (PLAN_APPROVAL, TICKET_PIPELINE, etc.)
    EnumBackendType: Infrastructure backend types (CONSUL, POSTGRES)
    EnumCheckSeverity: Validation check severity (REQUIRED, RECOMMENDED, INFORMATIONAL)
    EnumChainViolationType: Chain violation types for correlation/causation validation
    EnumCircuitState: Circuit breaker states (CLOSED, OPEN, HALF_OPEN)
    EnumConfirmationEventType: Registration confirmation event types
    EnumConsumerGroupPurpose: Consumer group purpose (CONSUME, INTROSPECTION, REPLAY, AUDIT, BACKFILL)
    EnumContextSectionCategory: Semantic categories for static context sections (CONFIG, RULES, TOPOLOGY, etc.)
    EnumContractType: Contract types for ONEX nodes (effect, compute, reducer, orchestrator)
    EnumCostTier: Cost tier for LLM backend routing (LOW, MID, HIGH)
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
    EnumKafkaEnvironment: Kafka environment identifiers for topic prefixes (DEV, STAGING, PROD, LOCAL)
    EnumLlmFinishReason: LLM finish reasons (STOP, LENGTH, ERROR, CONTENT_FILTER, TOOL_CALLS, UNKNOWN)
    EnumLlmOperationType: LLM operation types (CHAT_COMPLETION, COMPLETION, EMBEDDING)
    EnumMessageCategory: Message categories (EVENT, COMMAND, INTENT)
    EnumNodeArchetype: 4-node architecture (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR)
    EnumNodeOutputType: Node output types for execution shape validation
    EnumNonRetryableErrorCategory: Non-retryable error categories for DLQ
    EnumPolicyType: Policy types for RegistryPolicy plugins
    EnumPostgresErrorCode: PostgreSQL error codes for contract persistence operations
    EnumRegistrationState: Registration FSM states for two-way registration
    EnumRegistrationStatus: Registration workflow status (IDLE, PENDING, PARTIAL, COMPLETE, FAILED)
    EnumRegistryResponseStatus: Registry operation response status (SUCCESS, PARTIAL, FAILED)
    EnumRetryErrorCategory: Error categories for retry decision making
    EnumRunVariant: A/B run variant (BASELINE, CANDIDATE) for baseline comparison
    EnumSecurityRuleId: Security validation rule identifiers for OMN-1098
    EnumSelectionStrategy: Selection strategies for capability-based discovery (FIRST, RANDOM, ROUND_ROBIN, LEAST_LOADED)
    EnumTopicStandard: Topic standards (ONEX_KAFKA, ENVIRONMENT_AWARE)
    EnumTopicType: Topic types (EVENTS, COMMANDS, INTENTS, SNAPSHOTS)
    EnumValidationSeverity: Validation error severity levels (ERROR, CRITICAL, WARNING)
    EnumValidationVerdict: Validation pipeline verdict (PASS, FAIL, QUARANTINE)
    EnumLifecycleTier: Pattern lifecycle promotion tiers (OBSERVED through DEFAULT, SUPPRESSED)
"""

from omnibase_core.enums import EnumTopicType
from omnibase_infra.enums.enum_adjudicator_state import EnumAdjudicatorState
from omnibase_infra.enums.enum_any_type_violation import EnumAnyTypeViolation
from omnibase_infra.enums.enum_auth_decision import EnumAuthDecision
from omnibase_infra.enums.enum_auth_source import EnumAuthSource
from omnibase_infra.enums.enum_backend_type import EnumBackendType
from omnibase_infra.enums.enum_capture_outcome import EnumCaptureOutcome
from omnibase_infra.enums.enum_capture_state import EnumCaptureState
from omnibase_infra.enums.enum_chain_violation_type import EnumChainViolationType
from omnibase_infra.enums.enum_check_severity import EnumCheckSeverity
from omnibase_infra.enums.enum_checkpoint_phase import EnumCheckpointPhase
from omnibase_infra.enums.enum_circuit_state import EnumCircuitState
from omnibase_infra.enums.enum_confirmation_event_type import EnumConfirmationEventType
from omnibase_infra.enums.enum_consumer_group_purpose import EnumConsumerGroupPurpose
from omnibase_infra.enums.enum_context_section_category import (
    EnumContextSectionCategory,
)
from omnibase_infra.enums.enum_contract_type import EnumContractType
from omnibase_infra.enums.enum_cost_tier import EnumCostTier
from omnibase_infra.enums.enum_declarative_node_violation import (
    EnumDeclarativeNodeViolation,
)
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
from omnibase_infra.enums.enum_infra_resource_type import EnumInfraResourceType
from omnibase_infra.enums.enum_infra_transport_type import EnumInfraTransportType
from omnibase_infra.enums.enum_introspection_reason import EnumIntrospectionReason
from omnibase_infra.enums.enum_kafka_acks import EnumKafkaAcks
from omnibase_infra.enums.enum_kafka_environment import EnumKafkaEnvironment
from omnibase_infra.enums.enum_ledger_sink_drop_policy import EnumLedgerSinkDropPolicy
from omnibase_infra.enums.enum_lifecycle_tier import EnumLifecycleTier
from omnibase_infra.enums.enum_llm_finish_reason import EnumLlmFinishReason
from omnibase_infra.enums.enum_llm_operation_type import EnumLlmOperationType
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.enums.enum_node_archetype import EnumNodeArchetype
from omnibase_infra.enums.enum_node_output_type import EnumNodeOutputType
from omnibase_infra.enums.enum_non_retryable_error_category import (
    EnumNonRetryableErrorCategory,
)
from omnibase_infra.enums.enum_policy_type import EnumPolicyType
from omnibase_infra.enums.enum_postgres_error_code import EnumPostgresErrorCode
from omnibase_infra.enums.enum_registration_state import EnumRegistrationState
from omnibase_infra.enums.enum_registration_status import EnumRegistrationStatus
from omnibase_infra.enums.enum_registry_response_status import (
    EnumRegistryResponseStatus,
)
from omnibase_infra.enums.enum_response_status import EnumResponseStatus
from omnibase_infra.enums.enum_retry_error_category import EnumRetryErrorCategory
from omnibase_infra.enums.enum_run_variant import EnumRunVariant
from omnibase_infra.enums.enum_security_rule_id import EnumSecurityRuleId
from omnibase_infra.enums.enum_selection_strategy import EnumSelectionStrategy
from omnibase_infra.enums.enum_session_lifecycle_state import (
    EnumSessionLifecycleState,
)
from omnibase_infra.enums.enum_topic_standard import EnumTopicStandard
from omnibase_infra.enums.enum_validation_severity import EnumValidationSeverity
from omnibase_infra.enums.enum_validation_verdict import EnumValidationVerdict

__all__: list[str] = [
    "EnumAdjudicatorState",
    "EnumAnyTypeViolation",
    "EnumAuthDecision",
    "EnumAuthSource",
    "EnumBackendType",
    "EnumCostTier",
    "EnumCaptureOutcome",
    "EnumCaptureState",
    "EnumChainViolationType",
    "EnumCheckSeverity",
    "EnumCheckpointPhase",
    "EnumCircuitState",
    "EnumConfirmationEventType",
    "EnumConsumerGroupPurpose",
    "EnumContextSectionCategory",
    "EnumContractType",
    "EnumDeclarativeNodeViolation",
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
    "EnumInfraResourceType",
    "EnumInfraTransportType",
    "EnumIntrospectionReason",
    "EnumKafkaAcks",
    "EnumKafkaEnvironment",
    "EnumLedgerSinkDropPolicy",
    "EnumLifecycleTier",
    "EnumLlmFinishReason",
    "EnumLlmOperationType",
    "EnumMessageCategory",
    "EnumNodeArchetype",
    "EnumNodeOutputType",
    "EnumNonRetryableErrorCategory",
    "EnumPolicyType",
    "EnumPostgresErrorCode",
    "EnumRegistrationState",
    "EnumRegistrationStatus",
    "EnumRegistryResponseStatus",
    "EnumResponseStatus",
    "EnumRetryErrorCategory",
    "EnumRunVariant",
    "EnumSecurityRuleId",
    "EnumSelectionStrategy",
    "EnumSessionLifecycleState",
    "EnumTopicStandard",
    "EnumTopicType",
    "EnumValidationSeverity",
    "EnumValidationVerdict",
]
