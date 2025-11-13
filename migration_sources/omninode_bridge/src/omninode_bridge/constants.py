"""
Constants and Enums for OmniNode Bridge - Workflow Coordinator
============================================================

Centralized constants to eliminate magic strings throughout the codebase.
"""

# =============================================================================
# Configuration Keys
# =============================================================================


class ConfigKeys:
    """Configuration section keys."""

    KAFKA = "kafka"
    POSTGRES = "postgres"
    SMART_RESPONDER = "smart_responder"
    SERVICE = "service"
    INFRASTRUCTURE_HEALTH = "infrastructure_health"


class KafkaConfigKeys:
    """Kafka configuration keys."""

    BOOTSTRAP_SERVERS = "bootstrap_servers"
    WORKFLOW_TOPIC = "workflow_topic"
    TASK_EVENTS_TOPIC = "task_events_topic"


class PostgresConfigKeys:
    """PostgreSQL configuration keys."""

    HOST = "host"
    PORT = "port"
    DATABASE = "database"
    USER = "user"
    PASSWORD = "password"


# =============================================================================
# Infrastructure Services
# =============================================================================


class ServiceNames:
    """Infrastructure service names."""

    KAFKA = "omninode-bridge-redpanda"
    POSTGRES = "omninode-bridge-postgres"
    CONSUL = "omninode-bridge-consul"
    HOOK_RECEIVER = "omninode-bridge-hook-receiver"
    MODEL_METRICS = "omninode-bridge-model-metrics"
    WORKFLOW_COORDINATOR = "omninode-bridge-workflow-coordinator"


class ServiceStatus:
    """Service health status values."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    UNKNOWN = "unknown"


# =============================================================================
# Kafka Topics
# =============================================================================


class KafkaTopics:
    """Kafka topic names for ONEX event streaming."""

    # Workflow Topics
    WORKFLOWS = "dev.omninode_bridge.onex.workflows.v1"
    TASK_EVENTS = "dev.omninode_bridge.onex.task-events.v1"
    WORKFLOW_COMMANDS = "dev.omninode_bridge.onex.workflow-commands.v1"
    WORKFLOW_RESULTS = "dev.omninode_bridge.onex.workflow-results.v1"

    # Event-Driven Workflow Submission
    WORKFLOW_EXECUTION_REQUESTS = "dev.omninode_bridge.onex.cmd.workflow-execute.v1"
    WORKFLOW_EXECUTION_RESPONSES = "dev.omninode_bridge.onex.rsp.workflow-execute.v1"
    WORKFLOW_STATUS_REQUESTS = "dev.omninode_bridge.onex.cmd.workflow-status.v1"
    WORKFLOW_STATUS_RESPONSES = "dev.omninode_bridge.onex.rsp.workflow-status.v1"

    # Hook Receiver Topics
    HOOK_RECEIVED = "dev.omninode_bridge.onex.evt.hook-received.v1"
    SESSION_CREATED = "dev.omninode_bridge.onex.evt.session-created.v1"
    TOOL_EXECUTED = "dev.omninode_bridge.onex.evt.tool-executed.v1"

    # Service Discovery Topics
    SERVICE_REGISTERED = "dev.omninode_bridge.onex.evt.service-registered.v1"
    SERVICE_DEREGISTERED = "dev.omninode_bridge.onex.evt.service-deregistered.v1"


# =============================================================================
# AI Lab Configuration
# =============================================================================


class AILabNodes:
    """AI lab node identifiers."""

    MAC_STUDIO = "mac_studio"
    MAC_MINI = "mac_mini"
    AI_PC = "ai_pc"
    MACBOOK_AIR = "macbook_air"


class AILabHosts:
    """AI lab node IP addresses."""

    MAC_STUDIO = "192.168.86.200"
    MAC_MINI = "192.168.86.101"
    AI_PC = "192.168.86.201"
    MACBOOK_AIR = "192.168.86.105"


class CommonModels:
    """Common AI model names across the lab."""

    LLAMA31_8B_INSTRUCT_Q6K = "llama3.1:8b-instruct-q6_k"
    LLAMA32_LATEST = "llama3.2:latest"
    LLAMA31_LATEST = "llama3.1:latest"
    MIXTRAL_8X7B = "mixtral:8x7b-instruct-v0.1-q4_K_M"
    CODESTRAL_22B = "codestral:22b-v0.1-q4_K_M"
    DEEPSEEK_CODER_67B = "deepseek-coder:6.7b-instruct-q5_0"
    PHI3_LATEST = "phi3:latest"
    MISTRAL_LATEST = "mistral:latest"
    GPT_OSS_20B = "gpt-oss:20b"


# =============================================================================
# Task Configuration
# =============================================================================


class TaskConfigKeys:
    """Task configuration parameter keys."""

    PROMPT = "prompt"
    COMPLEXITY = "complexity"
    AI_TASK_TYPE = "ai_task_type"
    CONTEXT_SIZE = "context_size"
    PREFERRED_MODEL = "preferred_model"
    TIMEOUT = "timeout"
    RETRY_LIMIT = "retry_limit"
    DEPENDENCIES = "dependencies"


class ComplexityLevels:
    """Task complexity levels."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    CRITICAL = "critical"


# =============================================================================
# Response Data Keys
# =============================================================================


class ResponseKeys:
    """Keys for AI model response data."""

    SUCCESS = "success"
    RESPONSE = "response"
    MODEL = "model"
    NODE = "node"
    ERROR = "error"
    USAGE = "usage"
    TIMING = "timing"
    AI_RESPONSE = "ai_response"
    MODEL_USED = "model_used"
    NODE_LOCATION = "node_location"
    EXECUTION_METRICS = "execution_metrics"


class ExecutionMetricsKeys:
    """Keys for execution metrics."""

    LATENCY_MS = "latency_ms"
    TOKENS_PER_SECOND = "tokens_per_second"
    QUALITY_SCORE = "quality_score"
    MODEL_TIER = "model_tier"
    EXECUTION_ID = "execution_id"
    CONTEXT_SIZE = "context_size"
    RETRY_COUNT = "retry_count"


# =============================================================================
# Database Tables and Columns
# =============================================================================


class DatabaseTables:
    """Database table names."""

    SERVICE_SESSIONS = "service_sessions"
    HOOK_EVENTS = "hook_events"
    EVENT_METRICS = "event_metrics"
    WORKFLOWS = "workflows"
    WORKFLOW_TASKS = "workflow_tasks"


class DatabaseColumns:
    """Common database column names."""

    ID = "id"
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"
    STATUS = "status"
    METADATA = "metadata"
    ERROR_MESSAGE = "error_message"


# =============================================================================
# HTTP and API Constants
# =============================================================================


class HTTPHeaders:
    """HTTP header names."""

    CONTENT_TYPE = "Content-Type"
    AUTHORIZATION = "Authorization"
    USER_AGENT = "User-Agent"


class ContentTypes:
    """HTTP content type values."""

    JSON = "application/json"
    TEXT = "text/plain"
    HTML = "text/html"


# =============================================================================
# Logging and Monitoring
# =============================================================================


class LogEvents:
    """Structured logging event names."""

    WORKFLOW_STARTED = "workflow_coordinator.workflow.started"
    WORKFLOW_COMPLETED = "workflow_coordinator.workflow.completed"
    WORKFLOW_FAILED = "workflow_coordinator.workflow.failed"

    TASK_STARTED = "workflow_coordinator.task.started"
    TASK_COMPLETED = "workflow_coordinator.task.completed"
    TASK_FAILED = "workflow_coordinator.task.failed"

    PERSISTENCE_SUCCESS = "workflow_coordinator.persistence.completed"
    PERSISTENCE_FAILED = "workflow_coordinator.persistence.failed"
    TASK_UPSERT_FAILED = "workflow_coordinator.persistence.task_upsert_failed"

    KAFKA_EVENT_SENT = "workflow_coordinator.kafka.workflow_event_sent"
    KAFKA_EVENT_FAILED = "workflow_coordinator.kafka.task_event_failed"

    AI_EXECUTION_STARTED = "workflow_coordinator.ai.execution_started"
    AI_EXECUTION_COMPLETED = "workflow_coordinator.ai.execution_completed"
    AI_EXECUTION_FAILED = "workflow_coordinator.ai.execution_failed"


class MetricNames:
    """Performance metric names."""

    EXECUTION_DURATION = "execution_duration_ms"
    KAFKA_PUBLISH_DURATION = "kafka_publish_duration_ms"
    DATABASE_QUERY_DURATION = "database_query_duration_ms"
    AI_RESPONSE_DURATION = "ai_response_duration_ms"


# =============================================================================
# Error Messages
# =============================================================================


class ErrorMessages:
    """Standardized error messages."""

    # Infrastructure Errors
    KAFKA_NOT_CONNECTED = "Kafka client not connected"
    POSTGRES_NOT_CONNECTED = "PostgreSQL client not connected"
    SMART_RESPONDER_UNAVAILABLE = "Smart Responder Chain unavailable"

    # Task Execution Errors
    TASK_TYPE_UNSUPPORTED = "Unsupported task type"
    TASK_TIMEOUT = "Task execution timeout"
    TASK_DEPENDENCY_FAILED = "Task dependency failed"

    # Workflow Errors
    WORKFLOW_NOT_FOUND = "Workflow not found"
    WORKFLOW_ALREADY_RUNNING = "Workflow already running"
    WORKFLOW_INVALID_STATE = "Invalid workflow state"

    # Data Validation Errors
    INVALID_UUID = "Invalid UUID format"
    MISSING_REQUIRED_PARAMETER = "Missing required parameter"
    INVALID_PARAMETER_TYPE = "Invalid parameter type"

    # Task Validation Errors
    OUTPUT_VALIDATION_FAILED = "Task output failed validation contract"
    DEFINITION_OF_DONE_NOT_MET = "Task definition of done criteria not met"
    EMPTY_AI_RESPONSE = "AI task returned empty or invalid response"
    MODEL_FALLBACK_EXHAUSTED = "All model fallback options exhausted"


# =============================================================================
# Task Validation Configuration
# =============================================================================


class ValidationTypes:
    """Task output validation types."""

    NONE = "none"
    SCHEMA = "schema"
    CONTENT_LENGTH = "content_length"
    REQUIRED_FIELDS = "required_fields"
    REGEX_PATTERN = "regex_pattern"
    CUSTOM_FUNCTION = "custom_function"


class ValidationCriteria:
    """Definition of done validation criteria."""

    MIN_CONTENT_LENGTH = "min_content_length"
    MAX_CONTENT_LENGTH = "max_content_length"
    REQUIRED_KEYWORDS = "required_keywords"
    FORBIDDEN_KEYWORDS = "forbidden_keywords"
    OUTPUT_FORMAT = "output_format"
    QUALITY_THRESHOLD = "quality_threshold"


class ModelFallbackStrategy:
    """AI model fallback strategies."""

    NONE = "none"
    NEXT_TIER = "next_tier"
    DIFFERENT_NODE = "different_node"
    RETRY_WITH_MODIFICATION = "retry_with_modification"
    ESCALATE_COMPLEXITY = "escalate_complexity"


# =============================================================================
# Database Configuration Constants
# =============================================================================


class DatabaseDefaults:
    """PostgreSQL database configuration defaults."""

    # Connection Pool Configuration
    MIN_CONNECTIONS = 1
    MAX_CONNECTIONS = 200
    DEFAULT_MIN_CONNECTIONS = 5
    DEFAULT_MAX_CONNECTIONS = 50

    # Connection Timeouts (in seconds)
    CONNECTION_TIMEOUT = 30
    QUERY_TIMEOUT = 30
    HEALTH_CHECK_TIMEOUT = 10
    STARTUP_TIMEOUT = 90

    # Retry Configuration
    CONNECTION_RETRY_ATTEMPTS = 3
    HEALTH_CHECK_RETRY_ATTEMPTS = 5
    RETRY_BASE_DELAY = 1
    RETRY_MAX_DELAY = 60
    RETRY_BACKOFF_MULTIPLIER = 2

    # Query Configuration
    DEFAULT_FETCH_SIZE = 1000
    MAX_QUERY_TIMEOUT = 300

    # Batch Processing Configuration (DEPRECATED - use BatchSizeConfig instead)
    BATCH_SIZE_LIMIT = (
        50  # This is deprecated, use config.batch_sizes.database_batch_size
    )
    BATCH_TIMEOUT_MS = 100
    PREPARED_STATEMENT_CACHE_SIZE = 100
    MAX_PREPARED_STATEMENT_CACHE_SIZE = 200

    # Connection Optimization
    MAX_CACHED_STATEMENT_LIFETIME = 300  # 5 minutes
    CONNECTION_TEST_TIMEOUT = 10
    GRACEFUL_SHUTDOWN_TIMEOUT = 10
    FORCE_TERMINATE_TIMEOUT = 5

    # Connection Pool Settings
    MAX_WARMUP_CONNECTIONS = 3
    MIN_BATCH_FOR_EXECUTEMANY = 3

    # Data Retention (in days)
    HOOK_EVENTS_RETENTION_DAYS = 30
    EVENT_METRICS_RETENTION_DAYS = 90
    CONNECTION_METRICS_RETENTION_DAYS = 30

    # TCP Keep-Alive Configuration
    TCP_KEEPALIVES_IDLE = 300  # 5 minutes
    TCP_KEEPALIVES_INTERVAL = 30
    TCP_KEEPALIVES_COUNT = 3

    # Validation Constraints
    MAX_RETRY_COUNT = 10
    MAX_PROCESSING_TIME_MS = 300000  # 5 minutes
    MAX_CACHE_HIT_RATE = 100

    # Connection Acquisition
    CONNECTION_ACQUIRE_TIMEOUT = 5


# =============================================================================
# API Rate Limiting Constants
# =============================================================================


class RateLimits:
    """API rate limiting configuration."""

    # Workflow Operations (requests per minute per IP)
    WORKFLOW_EXECUTE = "10/minute"
    WORKFLOW_STATUS = "30/minute"
    WORKFLOW_PAUSE = "20/minute"
    WORKFLOW_RESUME = "20/minute"
    WORKFLOW_CANCEL = "15/minute"
    WORKFLOW_LIST = "60/minute"

    # System Operations
    HEALTH_CHECK = "100/minute"

    # Test Operations
    HIGH_THROUGHPUT_MESSAGE_COUNT = 100
    HIGH_THROUGHPUT_TIMEOUT = 30


# =============================================================================
# Container and Test Configuration Constants
# =============================================================================


class ContainerDefaults:
    """Container and integration test configuration."""

    # Container Startup Timeouts (in seconds)
    POSTGRES_STARTUP_TIMEOUT = 30
    KAFKA_STARTUP_TIMEOUT = 90
    REDIS_STARTUP_TIMEOUT = 15
    GENERAL_STARTUP_TIMEOUT = 60

    # Container Resource Limits
    MAX_CONCURRENT_POSTGRES = 3
    MAX_CONCURRENT_KAFKA = 2
    MAX_CONCURRENT_REDIS = 2
    STARTUP_QUEUE_SIZE = 5

    # Container Configuration
    POSTGRES_MAX_CONNECTIONS = 50
    POSTGRES_SHARED_BUFFERS_MB = 128
    REDIS_MAX_MEMORY_MB = 64

    # Cleanup Timeouts (in seconds)
    CONTAINER_STOP_TIMEOUT = 10
    CONTAINER_CLEANUP_TIMEOUT = 30
    FORCE_CLEANUP_TIMEOUT = 5

    # Retry Configuration
    CONTAINER_CREATE_ATTEMPTS = 3
    CONTAINER_RETRY_BASE_DELAY = 2
    KAFKA_RETRY_ADDITIONAL_DELAY = 5


# =============================================================================
# Kafka Configuration Constants
# =============================================================================


class KafkaDefaults:
    """Kafka client configuration defaults."""

    # Connection Configuration
    CONNECTION_TIMEOUT = 30
    HEALTH_CHECK_TIMEOUT = 10
    CONSUMER_TIMEOUT = 5000  # milliseconds

    # Producer Configuration (DEPRECATED - use BatchSizeConfig instead)
    BATCH_SIZE = (
        16384  # This is deprecated, use config.batch_sizes.kafka_producer_batch_size
    )
    LINGER_MS = 5  # milliseconds
    BUFFER_MEMORY = 33554432  # 32MB in bytes

    # Consumer Configuration
    MAX_POLL_RECORDS = 500
    SESSION_TIMEOUT_MS = 30000  # 30 seconds
    HEARTBEAT_INTERVAL_MS = 3000  # 3 seconds

    # Retry Configuration
    MAX_RETRY_ATTEMPTS = 3
    RETRY_BACKOFF_BASE = 0.1  # 100ms
    ENABLE_DEAD_LETTER_QUEUE = True

    # Topic Configuration
    DEFAULT_NUM_PARTITIONS = 3
    DEFAULT_REPLICATION_FACTOR = 1

    # Cleanup Configuration
    LOG_RETENTION_HOURS = 1
    LOG_RETENTION_BYTES = 1048576  # 1MB
    LOG_SEGMENT_BYTES = 1048576  # 1MB
    LOG_CLEANUP_INTERVAL_MS = 10000  # 10 seconds


# =============================================================================
# Default Values
# =============================================================================


class Defaults:
    """Default configuration values."""

    # Timeouts (in seconds)
    TASK_TIMEOUT = 300
    WORKFLOW_TIMEOUT = 1800
    HTTP_TIMEOUT = 30
    DATABASE_TIMEOUT = 10

    # Retry Configuration
    MAX_RETRIES = 3
    RETRY_DELAY = 5
    EXPONENTIAL_BACKOFF_MULTIPLIER = 2

    # Performance
    MAX_CONCURRENT_TASKS = 10
    BATCH_SIZE = 50

    # AI Configuration
    DEFAULT_COMPLEXITY = ComplexityLevels.MODERATE
    DEFAULT_CONTEXT_SIZE = 8192
    DEFAULT_MAX_TOKENS = 1000

    # Validation Configuration
    DEFAULT_VALIDATION_TYPE = ValidationTypes.CONTENT_LENGTH
    MIN_AI_RESPONSE_LENGTH = 10
    MAX_AI_RESPONSE_LENGTH = 50000
    DEFAULT_QUALITY_THRESHOLD = 0.7
    MAX_MODEL_FALLBACK_ATTEMPTS = 3


# =============================================================================
# Environment Variable Names
# =============================================================================


class EnvVars:
    """Environment variable names."""

    # Service Configuration
    WORKFLOW_COORDINATOR_HOST = "WORKFLOW_COORDINATOR_HOST"
    WORKFLOW_COORDINATOR_PORT = "WORKFLOW_COORDINATOR_PORT"
    WORKFLOW_COORDINATOR_WORKERS = "WORKFLOW_COORDINATOR_WORKERS"
    LOG_LEVEL = "LOG_LEVEL"

    # Kafka Configuration
    KAFKA_BOOTSTRAP_SERVERS = "KAFKA_BOOTSTRAP_SERVERS"
    KAFKA_WORKFLOW_TOPIC = "KAFKA_WORKFLOW_TOPIC"
    KAFKA_TASK_EVENTS_TOPIC = "KAFKA_TASK_EVENTS_TOPIC"

    # PostgreSQL Configuration
    POSTGRES_HOST = "POSTGRES_HOST"
    POSTGRES_PORT = "POSTGRES_PORT"
    POSTGRES_DATABASE = "POSTGRES_DATABASE"
    POSTGRES_USER = "POSTGRES_USER"
    POSTGRES_PASSWORD = "POSTGRES_PASSWORD"

    # Service Endpoints
    HOOK_RECEIVER_URL = "HOOK_RECEIVER_URL"
    MODEL_METRICS_URL = "MODEL_METRICS_URL"

    # Application Metadata
    ENVIRONMENT = "ENVIRONMENT"
    SERVICE_VERSION = "SERVICE_VERSION"
