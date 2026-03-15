# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Parametrized roundtrip serialization tests for boundary models.

Verifies that model_dump(mode="json") -> json.dumps -> json.loads -> model_validate
produces an equivalent model instance for all boundary models in event_bus.models
and runtime.models packages.

Uses curated per-model factory functions (NOT schema-driven automatic construction)
per OMN-4920 specification.

Discovery: 66 total models across event_bus.models (6) and runtime.models (60).
Coverage: 18 factories, 0 skipped, 0 normalizing = 18/66 (27%).
Stop rule applied: 18 models exceeds the ~15 target.
Remaining 48 models are dispositioned below in UNCOVERED_MODELS.
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import pkgutil
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from pydantic import BaseModel

# -- Event bus models --
from omnibase_infra.event_bus.models.model_dlq_event import ModelDlqEvent
from omnibase_infra.event_bus.models.model_dlq_metrics import ModelDlqMetrics
from omnibase_infra.event_bus.models.model_event_bus_readiness import (
    ModelEventBusReadiness,
)
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage

# -- Runtime models --
from omnibase_infra.runtime.models.model_batch_publisher_config import (
    ModelBatchPublisherConfig,
)
from omnibase_infra.runtime.models.model_batch_publisher_metrics import (
    ModelBatchPublisherMetrics,
)
from omnibase_infra.runtime.models.model_component_health import ModelComponentHealth
from omnibase_infra.runtime.models.model_detailed_health_response import (
    ModelDetailedHealthResponse,
)
from omnibase_infra.runtime.models.model_duplicate_response import (
    ModelDuplicateResponse,
)
from omnibase_infra.runtime.models.model_failed_component import ModelFailedComponent
from omnibase_infra.runtime.models.model_lifecycle_result import ModelLifecycleResult
from omnibase_infra.runtime.models.model_logging_config import ModelLoggingConfig
from omnibase_infra.runtime.models.model_optional_string import ModelOptionalString
from omnibase_infra.runtime.models.model_retry_policy import ModelRetryPolicy
from omnibase_infra.runtime.models.model_runtime_scheduler_config import (
    ModelRuntimeSchedulerConfig,
)
from omnibase_infra.runtime.models.model_runtime_tick import ModelRuntimeTick
from omnibase_infra.runtime.models.model_shutdown_config import ModelShutdownConfig

logger = logging.getLogger(__name__)

# ============================================================================
# SKIP_MODELS: Models that cannot be constructed cleanly in test context.
# Each entry requires a reason string.
# ============================================================================
SKIP_MODELS: dict[type[BaseModel], str] = {
    # No models skipped -- all selected models are constructable with curated factories.
}

# ============================================================================
# NORMALIZING_MODELS: Models that don't satisfy strict == after roundtrip
# (e.g., tuple->list coercion). Use weaker invariant instead:
#   roundtripped.model_dump(mode="json") == dumped
# Each entry requires a reason string.
#
# NOTE: With strict=False on model_validate(), all currently tested models
# satisfy strict == equality. This dict is kept for future models that may
# require normalization (e.g., models with custom __eq__ or non-invertible
# serialization).
# ============================================================================
NORMALIZING_MODELS: dict[type[BaseModel], str] = {
    # No normalizing models needed -- strict=False handles all type coercions.
}

# ============================================================================
# UNCOVERED_MODELS: Models not covered by factories, with disposition reason.
# These are the 47 models discovered but not tested. Grouped by category.
# ============================================================================
UNCOVERED_MODELS: dict[str, str] = {
    # -- Event bus config models --
    "ModelKafkaEventBusConfig": "Config model with complex nested dependencies (Kafka-specific)",
    # -- Runtime batch/lifecycle models --
    "ModelBatchLifecycleResult": "Strict model with nested ModelLifecycleResult list dependencies",
    # -- Runtime config models (deep dependency chains) --
    "ModelBindingConfig": "Requires EnumInfraTransportType enum + complex field interactions",
    "ModelBindingConfigCacheStats": "Low-priority metrics model, simple defaults",
    "ModelBindingConfigResolverConfig": "Config with nested model dependencies",
    "ModelCachedSecret": "Requires datetime + secret value construction",
    "ModelComputeKey": "Requires EnumInfraTransportType enum dependency",
    "ModelComputeRegistration": "Requires callable/protocol references",
    "ModelConfigCacheEntry": "Requires nested config model dependencies",
    "ModelConfigRef": "Requires transport/path parsing logic",
    "ModelConfigRefParseResult": "Requires ModelConfigRef dependency",
    "ModelContractLoadResult": "Requires contract object references",
    "ModelContractRegistryConfig": "Config with path dependencies",
    "ModelDomainPluginConfig": "Plugin config with complex validation",
    "ModelDomainPluginResult": "Result model with plugin-specific fields",
    "ModelEnabledProtocolsConfig": "Config with protocol enum dependencies",
    "ModelEventBusConfig": "Config with nested Kafka/transport config",
    "ModelHandshakeCheckResult": "Result model with validation chain",
    "ModelHandshakeResult": "Result model with complex nested validation",
    "ModelHealthCheckResult": "Uses JsonType dict with recursive type alias",
    "ModelHealthCheckResponse": "Uses JsonType dict + Literal status type",
    "ModelHttpClientConfig": "Config with URL/timeout validation",
    "ModelIntentExecutionSummary": "Complex execution tracking model",
    "ModelKafkaProducerConfig": "Kafka-specific config with broker validation",
    "ModelMaterializedResources": "Resource tracking with complex state",
    "ModelMaterializerConfig": "Config with path/resource dependencies",
    "ModelOptionalCorrelationId": "Thin wrapper, similar to ModelOptionalString",
    "ModelOptionalUUID": "Thin wrapper, similar to ModelOptionalString",
    "ModelPolicyContext": "Requires complex policy chain construction",
    "ModelPolicyKey": "Requires EnumPolicyType + version normalization",
    "ModelPolicyRegistration": "Requires callable protocol references",
    "ModelPolicyResult": "Uses extra='allow' with dynamic fields",
    "ModelPolicyTypeFilter": "Requires EnumPolicyType enum dependency",
    "ModelPostgresPoolConfig": "Config with connection pool validation",
    "ModelProjectionResultLocal": "Projection result with complex state",
    "ModelProjectorNotificationConfig": "Config with notification channel deps",
    "ModelProjectorPluginLoaderConfig": "Config with plugin path dependencies",
    "ModelProtocolRegistrationConfig": "Config with protocol enum dependencies",
    "ModelRuntimeConfig": "Top-level config aggregating many nested configs",
    "ModelRuntimeContractConfig": "Config with contract path dependencies",
    "ModelRuntimeSchedulerMetrics": "Metrics model, simple but low priority",
    "ModelSecretCacheStats": "Simple metrics, low priority",
    "ModelSecretMapping": "Secret mapping with source/target validation",
    "ModelSecretResolverConfig": "Config with secret source dependencies",
    "ModelSecretResolverMetrics": "Metrics model, simple but low priority",
    "ModelSecretSourceInfo": "Source info with enum dependencies",
    "ModelSecretSourceSpec": "Source spec with validation chain",
    "ModelSecurityConfig": "Config with security policy dependencies",
    "ModelShutdownBatchResult": "Batch result with nested lifecycle models",
    "ModelTransitionNotificationOutboxConfig": "Config with outbox dependencies",
    "ModelTransitionNotificationOutboxMetrics": "Metrics model, low priority",
    "ModelTransitionNotificationPublisherMetrics": "Metrics model, low priority",
}


# ============================================================================
# Curated per-model factory functions
# ============================================================================


def _make_event_headers() -> ModelEventHeaders:
    return ModelEventHeaders(
        source="test-service",
        event_type="test.event.created",
        timestamp=datetime.now(UTC),
    )


def _make_dlq_event() -> ModelDlqEvent:
    return ModelDlqEvent(
        original_topic="orders.created",
        dlq_topic="dlq.orders",
        correlation_id=uuid4(),
        error_type="ValueError",
        error_message="Invalid order payload",
        retry_count=3,
        message_offset="42",
        message_partition=2,
        success=True,
        timestamp=datetime.now(UTC),
        environment="test",
        consumer_group="order-processor",
    )


def _make_dlq_metrics() -> ModelDlqMetrics:
    return ModelDlqMetrics(
        total_publishes=100,
        successful_publishes=95,
        failed_publishes=5,
    )


def _make_event_bus_readiness() -> ModelEventBusReadiness:
    return ModelEventBusReadiness(
        is_ready=True,
        consumers_started=True,
        assignments={"orders.created": [0, 1, 2]},
        consume_tasks_alive={"orders.created": True},
        required_topics=("orders.created", "payments.processed"),
        required_topics_ready=True,
    )


def _make_event_message() -> ModelEventMessage:
    return ModelEventMessage(
        topic="onex.orders.created",
        key=b"customer-123",
        value=b'{"order_id": "ORD-123", "amount": 99.99}',
        headers=ModelEventHeaders(
            source="order-service",
            event_type="order.created",
            timestamp=datetime.now(UTC),
        ),
        offset="100",
        partition=0,
    )


def _make_component_health() -> ModelComponentHealth:
    return ModelComponentHealth(
        name="kafka",
        status="healthy",
        latency_ms=5.2,
    )


def _make_detailed_health_response() -> ModelDetailedHealthResponse:
    return ModelDetailedHealthResponse(
        status="healthy",
        version="1.0.0",
        components={
            "kafka": ModelComponentHealth(
                name="kafka",
                status="healthy",
                latency_ms=5.2,
            ),
        },
    )


def _make_duplicate_response() -> ModelDuplicateResponse:
    return ModelDuplicateResponse(
        message_id=uuid4(),
        correlation_id=uuid4(),
    )


def _make_batch_publisher_config() -> ModelBatchPublisherConfig:
    return ModelBatchPublisherConfig()


def _make_runtime_scheduler_config() -> ModelRuntimeSchedulerConfig:
    return ModelRuntimeSchedulerConfig()


def _make_batch_publisher_metrics() -> ModelBatchPublisherMetrics:
    return ModelBatchPublisherMetrics(
        total_enqueued=50,
        total_published=45,
        total_failed=5,
        total_batches_flushed=10,
        total_timeout_flushes=2,
        total_size_flushes=8,
    )


def _make_retry_policy() -> ModelRetryPolicy:
    return ModelRetryPolicy(
        max_retries=5,
        backoff_strategy="exponential",
        base_delay_ms=200,
        max_delay_ms=10000,
    )


def _make_lifecycle_result() -> ModelLifecycleResult:
    return ModelLifecycleResult.succeeded("kafka")


def _make_failed_component() -> ModelFailedComponent:
    return ModelFailedComponent(
        component_name="EventBusKafka",
        error_message="Connection timeout during shutdown",
    )


def _make_logging_config() -> ModelLoggingConfig:
    return ModelLoggingConfig(
        level="WARNING",
        format="%(asctime)s %(message)s",
    )


def _make_shutdown_config() -> ModelShutdownConfig:
    return ModelShutdownConfig(
        grace_period_seconds=60,
        handler_shutdown_timeout_seconds=15.0,
    )


def _make_optional_string() -> ModelOptionalString:
    return ModelOptionalString(value="test-correlation-context")


def _make_runtime_tick() -> ModelRuntimeTick:
    now = datetime.now(UTC)
    return ModelRuntimeTick(
        now=now,
        tick_id=uuid4(),
        sequence_number=42,
        scheduled_at=now,
        correlation_id=uuid4(),
        scheduler_id="runtime-instance-001",
        tick_interval_ms=1000,
    )


# ============================================================================
# Factory registry: maps model class -> factory callable
# ============================================================================
MODEL_FACTORIES: dict[type[BaseModel], Any] = {
    # Event bus models (5/6 covered)
    ModelEventHeaders: _make_event_headers,
    ModelDlqEvent: _make_dlq_event,
    ModelDlqMetrics: _make_dlq_metrics,
    ModelEventBusReadiness: _make_event_bus_readiness,
    ModelEventMessage: _make_event_message,
    # Runtime models (13/59 covered)
    ModelBatchPublisherConfig: _make_batch_publisher_config,
    ModelBatchPublisherMetrics: _make_batch_publisher_metrics,
    ModelComponentHealth: _make_component_health,
    ModelDetailedHealthResponse: _make_detailed_health_response,
    ModelDuplicateResponse: _make_duplicate_response,
    ModelFailedComponent: _make_failed_component,
    ModelLifecycleResult: _make_lifecycle_result,
    ModelLoggingConfig: _make_logging_config,
    ModelOptionalString: _make_optional_string,
    ModelRetryPolicy: _make_retry_policy,
    ModelRuntimeSchedulerConfig: _make_runtime_scheduler_config,
    ModelRuntimeTick: _make_runtime_tick,
    ModelShutdownConfig: _make_shutdown_config,
}


def discover_boundary_models() -> list[type[BaseModel]]:
    """Discover all Pydantic BaseModel subclasses in boundary model packages.

    Scans omnibase_infra.event_bus.models and omnibase_infra.runtime.models
    for classes that directly subclass BaseModel (defined in those modules).

    Returns:
        Sorted list of model classes found, for stable test ordering.
    """
    discovered: list[type[BaseModel]] = []
    for pkg_name in [
        "omnibase_infra.event_bus.models",
        "omnibase_infra.runtime.models",
    ]:
        pkg = importlib.import_module(pkg_name)
        for _importer, modname, ispkg in pkgutil.walk_packages(
            pkg.__path__, prefix=pkg.__name__ + "."
        ):
            if ispkg:
                continue
            try:
                mod = importlib.import_module(modname)
            except Exception:
                logger.warning("Failed to import %s", modname, exc_info=True)
                continue
            for _name, obj in inspect.getmembers(mod, inspect.isclass):
                if (
                    issubclass(obj, BaseModel)
                    and obj is not BaseModel
                    and obj.__module__ == mod.__name__
                ):
                    discovered.append(obj)
    return sorted(discovered, key=lambda c: c.__name__)


def _get_testable_models() -> list[type[BaseModel]]:
    """Return models that have factories and are not in SKIP_MODELS."""
    return [cls for cls in MODEL_FACTORIES if cls not in SKIP_MODELS]


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_cls",
    _get_testable_models(),
    ids=lambda m: m.__name__,
)
def test_model_json_roundtrip(model_cls: type[BaseModel]) -> None:
    """Verify model_dump(mode='json') -> json.dumps -> json.loads -> model_validate roundtrip.

    For models in NORMALIZING_MODELS, uses the weaker invariant that the
    re-dumped JSON matches the original dump (accounting for type coercions
    like tuple -> list).
    """
    factory = MODEL_FACTORIES[model_cls]
    instance = factory()

    # Step 1: Dump with mode="json" (produces JSON-safe types)
    dumped = instance.model_dump(mode="json")

    # Step 2: Serialize to JSON string (would fail with TypeError on non-safe types)
    serialized = json.dumps(dumped)

    # Step 3: Deserialize back
    deserialized = json.loads(serialized)

    # Step 4: Reconstruct model
    # Use strict=False for model_validate to handle models with strict=True
    # config that reject coerced JSON types (e.g., str UUIDs, int from JSON)
    roundtripped = model_cls.model_validate(deserialized, strict=False)

    # Step 5: Assert equivalence
    if model_cls in NORMALIZING_MODELS:
        # Weaker invariant: JSON representations match
        # (accounts for tuple->list coercion, strict type differences)
        assert roundtripped.model_dump(mode="json") == dumped, (
            f"{model_cls.__name__} (normalizing): JSON dump mismatch after roundtrip. "
            f"Reason: {NORMALIZING_MODELS[model_cls]}"
        )
    else:
        # Strict invariant: model instances are equal
        assert roundtripped == instance, (
            f"{model_cls.__name__}: model instance mismatch after roundtrip"
        )


@pytest.mark.unit
def test_skip_models_have_reasons() -> None:
    """Verify all SKIP_MODELS entries have non-empty reason strings."""
    for model_cls, reason in SKIP_MODELS.items():
        assert reason.strip(), f"SKIP_MODELS[{model_cls.__name__}] has empty reason"


@pytest.mark.unit
def test_normalizing_models_have_reasons() -> None:
    """Verify all NORMALIZING_MODELS entries have non-empty reason strings."""
    for model_cls, reason in NORMALIZING_MODELS.items():
        assert reason.strip(), (
            f"NORMALIZING_MODELS[{model_cls.__name__}] has empty reason"
        )


@pytest.mark.unit
def test_coverage_report() -> None:
    """Report coverage statistics and verify minimum threshold.

    Discovers all boundary models and reports how many are covered by
    factories, skipped, normalizing, or uncovered.
    """
    all_models = discover_boundary_models()
    factory_models = set(MODEL_FACTORIES.keys())
    skip_models = set(SKIP_MODELS.keys())
    normalizing_models = set(NORMALIZING_MODELS.keys())
    testable = factory_models - skip_models

    # Log coverage report
    logger.info("=== Roundtrip Serialization Coverage Report ===")
    logger.info("Total discovered models: %d", len(all_models))
    logger.info("Models with factories: %d", len(factory_models))
    logger.info("Models skipped: %d", len(skip_models))
    logger.info("Models normalizing: %d", len(normalizing_models))
    logger.info("Models testable: %d", len(testable))
    logger.info(
        "Coverage: %.1f%%",
        (len(factory_models) / len(all_models) * 100) if all_models else 0,
    )

    uncovered = [
        m for m in all_models if m not in factory_models and m not in skip_models
    ]
    if uncovered:
        logger.info("Uncovered models (%d):", len(uncovered))
        for m in uncovered:
            disposition = UNCOVERED_MODELS.get(m.__name__, "No disposition recorded")
            logger.info("  - %s: %s", m.__name__, disposition)

    # Minimum threshold: at least 15 testable models (stop rule target)
    assert len(testable) >= 15, (
        f"Only {len(testable)} testable models; target is >= 15 per stop rule"
    )


@pytest.mark.unit
def test_uncovered_models_have_dispositions() -> None:
    """Verify all discovered-but-uncovered models have a disposition in UNCOVERED_MODELS."""
    all_models = discover_boundary_models()
    factory_models = set(MODEL_FACTORIES.keys())
    skip_models = set(SKIP_MODELS.keys())

    uncovered = [
        m for m in all_models if m not in factory_models and m not in skip_models
    ]
    missing_dispositions = [
        m.__name__ for m in uncovered if m.__name__ not in UNCOVERED_MODELS
    ]
    assert not missing_dispositions, (
        f"Models missing disposition in UNCOVERED_MODELS: {missing_dispositions}"
    )
