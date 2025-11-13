"""Configuration management for OmniNode Bridge."""

# Legacy environment configuration
# Bridge node configuration (YAML-based)
from .config_loader import (
    ConfigurationError,
    get_config_info,
    get_orchestrator_config,
    get_reducer_config,
    load_node_config,
    reload_config,
    validate_config_files,
)
from .environment_config import EnvironmentConfig, get_config

# Pydantic settings models
from .settings import (
    AggregationConfig,
    CacheNodeConfig,
    CircuitBreakerNodeConfig,
    DatabaseNodeConfig,
    EventsAggregationConfig,
    KafkaConsumerConfig,
    KafkaNodeConfig,
    KafkaProducerConfig,
    KafkaTopicsConfig,
    LoggingNodeConfig,
    MetricsAggregationConfig,
    MonitoringNodeConfig,
    NodeConfig,
    OrchestratorConfig,
    OrchestratorSettings,
    ReducerConfig,
    ReducerSettings,
    ServiceEndpointConfig,
    ServicesConfig,
    SessionWindowConfig,
    SlidingWindowConfig,
    TumblingWindowConfig,
    WindowingConfig,
    WorkflowStateAggregationConfig,
)

# Validation utilities
from .validation import ConfigValidationError, validate_config

__all__ = [
    # Legacy configuration
    "ConfigValidationError",
    "EnvironmentConfig",
    "get_config",
    "validate_config",
    # Bridge node configuration loaders
    "ConfigurationError",
    "get_config_info",
    "get_orchestrator_config",
    "get_reducer_config",
    "load_node_config",
    "reload_config",
    "validate_config_files",
    # Settings models
    "AggregationConfig",
    "CacheNodeConfig",
    "CircuitBreakerNodeConfig",
    "DatabaseNodeConfig",
    "EventsAggregationConfig",
    "KafkaConsumerConfig",
    "KafkaNodeConfig",
    "KafkaProducerConfig",
    "KafkaTopicsConfig",
    "LoggingNodeConfig",
    "MetricsAggregationConfig",
    "MonitoringNodeConfig",
    "NodeConfig",
    "OrchestratorConfig",
    "OrchestratorSettings",
    "ReducerConfig",
    "ReducerSettings",
    "ServiceEndpointConfig",
    "ServicesConfig",
    "SessionWindowConfig",
    "SlidingWindowConfig",
    "TumblingWindowConfig",
    "WindowingConfig",
    "WorkflowStateAggregationConfig",
]
