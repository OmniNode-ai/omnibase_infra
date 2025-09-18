"""
Test Configuration for Enhanced PostgreSQL Adapter Testing.

Provides centralized configuration for all test types including:
- Integration tests with RedPanda
- Performance testing parameters
- Circuit breaker test settings
- Load testing configurations
- Security validation settings
"""

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class RedPandaTestConfig:
    """Configuration for RedPanda integration testing."""

    # Container settings
    image_version: str = "redpandadata/redpanda:v24.2.7"
    container_name: str = "test-redpanda"
    kafka_port: int = 9092
    admin_port: int = 9644
    proxy_port: int = 8082

    # Topic settings
    default_partitions: int = 3
    default_replication_factor: int = 1
    topic_creation_timeout: int = 30

    # Producer/Consumer settings
    producer_timeout: int = 10
    consumer_timeout: int = 5
    acks: str = "all"
    retries: int = 3
    retry_backoff_ms: int = 100

    # Test-specific topics
    test_topics: list[str] = None

    def __post_init__(self):
        if self.test_topics is None:
            self.test_topics = [
                "dev.omnibase.onex.evt.postgres-query-completed.v1",
                "dev.omnibase.onex.evt.postgres-query-failed.v1",
                "dev.omnibase.onex.qrs.postgres-health-response.v1",
                "test.postgres.events",
            ]


@dataclass
class PostgresTestConfig:
    """Configuration for PostgreSQL testing."""

    # Container settings
    image_version: str = "postgres:15"
    container_name: str = "test-postgres"
    port: int = 5432

    # Database settings
    database: str = "test_omnibase_infra"
    user: str = "postgres"
    password: str = "test_password"

    # Connection pool settings
    min_connections: int = 1
    max_connections: int = 10
    connection_timeout: int = 10

    # Test schema
    test_schema: str = "integration_test"


@dataclass
class PerformanceTestConfig:
    """Configuration for performance testing."""

    # Test execution settings
    measurement_runs: int = 10
    warmup_runs: int = 3
    statistical_significance_threshold: float = 0.05

    # Performance thresholds (milliseconds)
    max_avg_execution_time: float = 100.0
    max_single_execution_time: float = 500.0
    max_event_overhead: float = 50.0
    expected_db_baseline: float = 5.0

    # Load testing parameters
    concurrent_operations_count: int = 50
    concurrent_success_rate_threshold: float = 0.95
    concurrent_max_failures: float = 0.05
    concurrent_avg_time_threshold: float = 100.0
    concurrent_max_time_threshold: float = 1000.0

    # Memory and resource monitoring
    memory_leak_detection_enabled: bool = True
    resource_monitoring_interval: float = 1.0
    gc_force_interval: int = 5  # Force garbage collection every N operations


@dataclass
class CircuitBreakerTestConfig:
    """Configuration for circuit breaker testing."""

    # Circuit breaker parameters (test-optimized)
    failure_threshold: int = 3
    timeout_seconds: int = 2
    half_open_max_calls: int = 2

    # Test scenario settings
    failure_injection_count: int = 5
    recovery_wait_buffer: float = 0.1  # Extra wait time beyond timeout
    concurrent_test_operations: int = 10

    # State validation
    expected_states_sequence: list[str] = None

    def __post_init__(self):
        if self.expected_states_sequence is None:
            self.expected_states_sequence = ["closed", "open", "half_open", "closed"]


@dataclass
class SecurityTestConfig:
    """Configuration for security validation testing."""

    # SQL injection test patterns
    sql_injection_patterns: list[str] = None

    # Sensitive data patterns
    sensitive_patterns: list[str] = None

    # Parameter validation limits
    max_query_size: int = 1024 * 10  # 10KB
    max_parameter_count: int = 100
    max_parameter_size: int = 1024 * 1024  # 1MB
    max_timeout_seconds: float = 30.0

    # Query complexity limits
    max_complexity_score: int = 50
    complexity_weights: dict[str, int] = None

    # Sanitization settings
    enable_error_sanitization: bool = True
    enable_sql_injection_detection: bool = True
    enable_query_complexity_validation: bool = True

    def __post_init__(self):
        if self.sql_injection_patterns is None:
            self.sql_injection_patterns = [
                "'; DROP TABLE",
                "; DELETE FROM",
                "UNION SELECT password",
                "' OR '1'='1",
                "'; TRUNCATE",
                "/**/",
                "--",
                "xp_cmdshell",
            ]

        if self.sensitive_patterns is None:
            self.sensitive_patterns = [
                "password=",
                "token=",
                "secret=",
                "api_key=",
                "bearer ",
                "auth_token=",
            ]

        if self.complexity_weights is None:
            self.complexity_weights = {
                "join": 5,
                "subquery": 3,
                "union": 8,
                "leading_wildcard": 10,
                "regex": 15,
                "expensive_function": 12,
                "order_without_limit": 7,
            }


@dataclass
class LoadTestConfig:
    """Configuration for load testing."""

    # Locust settings
    host: str = "http://localhost:8085"
    users: int = 25
    spawn_rate: float = 5.0
    run_time: str = "300s"

    # Test scenario distribution (weights)
    simple_query_weight: int = 10
    parameterized_query_weight: int = 5
    analytical_query_weight: int = 2
    json_query_weight: int = 3
    health_check_weight: int = 1
    error_scenario_weight: int = 2

    # Performance expectations per load type
    execution_time_thresholds: dict[str, float] = None

    # Stress test settings
    stress_test_users: int = 100
    stress_spawn_rate: float = 10.0
    stress_run_time: str = "600s"

    def __post_init__(self):
        if self.execution_time_thresholds is None:
            self.execution_time_thresholds = {
                "low": 100.0,  # Simple queries under 100ms
                "medium": 500.0,  # Medium queries under 500ms
                "high": 2000.0,  # Complex queries under 2s
                "health": 200.0,  # Health checks under 200ms
            }


@dataclass
class IntegrationTestConfig:
    """Master configuration for all integration tests."""

    redpanda: RedPandaTestConfig = None
    postgres: PostgresTestConfig = None
    performance: PerformanceTestConfig = None
    circuit_breaker: CircuitBreakerTestConfig = None
    security: SecurityTestConfig = None
    load_test: LoadTestConfig = None

    # General test settings
    test_timeout: int = 300  # 5 minutes default timeout
    cleanup_on_failure: bool = True
    verbose_logging: bool = True
    parallel_execution: bool = False

    # Environment-specific overrides
    environment: str = "test"

    def __post_init__(self):
        self.redpanda = self.redpanda or RedPandaTestConfig()
        self.postgres = self.postgres or PostgresTestConfig()
        self.performance = self.performance or PerformanceTestConfig()
        self.circuit_breaker = self.circuit_breaker or CircuitBreakerTestConfig()
        self.security = self.security or SecurityTestConfig()
        self.load_test = self.load_test or LoadTestConfig()

    @classmethod
    def from_environment(cls) -> "IntegrationTestConfig":
        """Create configuration from environment variables."""
        config = cls()

        # Override with environment variables if present
        config.redpanda.kafka_port = int(
            os.getenv("TEST_REDPANDA_PORT", config.redpanda.kafka_port),
        )
        config.postgres.port = int(
            os.getenv("TEST_POSTGRES_PORT", config.postgres.port),
        )
        config.postgres.password = os.getenv(
            "TEST_POSTGRES_PASSWORD", config.postgres.password,
        )

        config.performance.measurement_runs = int(
            os.getenv("PERF_MEASUREMENT_RUNS", config.performance.measurement_runs),
        )
        config.performance.concurrent_operations_count = int(
            os.getenv(
                "PERF_CONCURRENT_OPS", config.performance.concurrent_operations_count,
            ),
        )

        config.load_test.users = int(
            os.getenv("LOAD_TEST_USERS", config.load_test.users),
        )
        config.load_test.host = os.getenv("LOAD_TEST_HOST", config.load_test.host)

        config.test_timeout = int(os.getenv("TEST_TIMEOUT", config.test_timeout))
        config.verbose_logging = os.getenv("VERBOSE_LOGGING", "true").lower() == "true"

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for logging/serialization."""
        return {
            "redpanda": {
                "image_version": self.redpanda.image_version,
                "kafka_port": self.redpanda.kafka_port,
                "topics": len(self.redpanda.test_topics),
            },
            "postgres": {
                "image_version": self.postgres.image_version,
                "database": self.postgres.database,
                "max_connections": self.postgres.max_connections,
            },
            "performance": {
                "measurement_runs": self.performance.measurement_runs,
                "max_avg_execution_time": self.performance.max_avg_execution_time,
                "concurrent_operations": self.performance.concurrent_operations_count,
            },
            "circuit_breaker": {
                "failure_threshold": self.circuit_breaker.failure_threshold,
                "timeout_seconds": self.circuit_breaker.timeout_seconds,
            },
            "security": {
                "max_query_size": self.security.max_query_size,
                "enable_injection_detection": self.security.enable_sql_injection_detection,
            },
            "load_test": {
                "users": self.load_test.users,
                "spawn_rate": self.load_test.spawn_rate,
                "run_time": self.load_test.run_time,
            },
            "environment": self.environment,
            "test_timeout": self.test_timeout,
        }


# Global test configuration instance
TEST_CONFIG = IntegrationTestConfig.from_environment()


# Test utility functions
def get_test_config() -> IntegrationTestConfig:
    """Get the global test configuration instance."""
    return TEST_CONFIG


def override_test_config(**kwargs) -> IntegrationTestConfig:
    """Create test configuration with specific overrides."""
    config = IntegrationTestConfig.from_environment()

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            # Try to set on sub-configs
            for sub_config_name in [
                "redpanda",
                "postgres",
                "performance",
                "circuit_breaker",
                "security",
                "load_test",
            ]:
                sub_config = getattr(config, sub_config_name)
                if hasattr(sub_config, key):
                    setattr(sub_config, key, value)
                    break

    return config


def validate_test_environment() -> bool:
    """Validate that test environment is properly configured."""
    issues = []

    # Check for required dependencies
    try:
        import kafka
        import locust
        import testcontainers
    except ImportError as e:
        issues.append(f"Missing test dependency: {e}")

    # Check Docker availability
    try:
        import docker

        client = docker.from_env()
        client.ping()
    except Exception as e:
        issues.append(f"Docker not available: {e}")

    # Check environment variables
    required_env_vars = []  # Add any required env vars here
    for env_var in required_env_vars:
        if not os.getenv(env_var):
            issues.append(f"Missing required environment variable: {env_var}")

    if issues:
        print("Test environment validation issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    return True


if __name__ == "__main__":
    # Print current test configuration
    config = get_test_config()

    print("Enhanced PostgreSQL Adapter Test Configuration")
    print("=" * 50)

    import json

    print(json.dumps(config.to_dict(), indent=2))

    print("\nEnvironment Validation:")
    print("=" * 30)
    if validate_test_environment():
        print("✅ Test environment is properly configured")
    else:
        print("❌ Test environment has issues (see above)")
        exit(1)
