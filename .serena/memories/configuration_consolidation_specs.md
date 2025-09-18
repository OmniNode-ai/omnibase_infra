# Configuration Consolidation Specifications

## Target: agent-contract-validator

### 1. Centralized Config Validation at Startup

**Files to modify:**
- `src/omnibase_infra/infrastructure/postgres_connection_manager.py`
- `src/omnibase_infra/infrastructure/event_bus_circuit_breaker.py`
- `src/omnibase_infra/infrastructure/kafka_producer_pool.py`

**Implementation Pattern:**
```python
from typing import List, Dict, Any
from omnibase_core.core.errors.onex_error import OnexError
from omnibase_core.core.core_error_codes import CoreErrorCode

class CentralizedConfigValidator:
    """Centralized configuration validation for infrastructure components."""

    @classmethod
    def validate_startup_configuration(cls) -> Dict[str, Any]:
        """Validate all infrastructure configuration at startup."""
        validation_results = {
            'database_config': cls._validate_database_config(),
            'event_bus_config': cls._validate_event_bus_config(),
            'circuit_breaker_config': cls._validate_circuit_breaker_config(),
            'kafka_producer_config': cls._validate_kafka_producer_config()
        }

        # Aggregate validation errors
        errors = [result for result in validation_results.values() if 'error' in result]
        if errors:
            raise OnexError(
                code=CoreErrorCode.CONFIGURATION_ERROR,
                message=f"Configuration validation failed: {errors}"
            )

        return validation_results
```

### 2. Event Bus Connectivity Validation During Initialization

**File:** `src/omnibase_infra/infrastructure/event_bus_circuit_breaker.py`
**Enhancement needed:**

```python
async def validate_event_bus_connectivity(self) -> Dict[str, Any]:
    """Validate event bus connectivity during initialization."""
    validation_result = {
        'connectivity_status': 'unknown',
        'response_time_ms': None,
        'broker_endpoints': [],
        'topic_accessibility': {},
        'authentication_status': 'unknown',
        'ssl_verification': 'unknown'
    }

    try:
        # Test basic connectivity
        start_time = time.time()
        # ... connectivity test implementation ...
        response_time = (time.time() - start_time) * 1000

        validation_result.update({
            'connectivity_status': 'connected',
            'response_time_ms': response_time,
            'authentication_status': 'authenticated',
            'ssl_verification': 'verified'
        })

        # Validate topic accessibility
        validation_result['topic_accessibility'] = await self._validate_topic_access()

    except Exception as e:
        validation_result['connectivity_status'] = 'failed'
        validation_result['error'] = str(e)
        raise OnexError(
            code=CoreErrorCode.EXTERNAL_SERVICE_ERROR,
            message=f"Event bus connectivity validation failed: {e}"
        ) from e

    return validation_result
```

### 3. Database Connectivity Validation During Initialization

**File:** `src/omnibase_infra/infrastructure/postgres_connection_manager.py`
**Enhancement needed:**

```python
async def validate_database_connectivity(self) -> Dict[str, Any]:
    """Validate database connectivity during initialization."""
    validation_result = {
        'connectivity_status': 'unknown',
        'connection_pool_status': 'unknown',
        'schema_validation': 'unknown',
        'permissions_check': 'unknown',
        'performance_baseline': {}
    }

    try:
        # Test basic connectivity
        async with self.acquire_connection() as conn:
            # Validate database version compatibility
            db_version = await conn.fetchval("SELECT version()")
            validation_result['database_version'] = db_version

            # Validate schema accessibility
            schema_tables = await conn.fetch(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = $1",
                self.config.schema
            )
            validation_result['schema_tables'] = [row['table_name'] for row in schema_tables]

            # Test basic operations permissions
            await conn.execute("SELECT 1")  # Read permission
            validation_result['permissions_check'] = 'validated'

            # Establish performance baseline
            start_time = time.time()
            await conn.execute("SELECT pg_sleep(0.001)")  # 1ms sleep test
            baseline_latency = (time.time() - start_time) * 1000

            validation_result.update({
                'connectivity_status': 'connected',
                'connection_pool_status': 'healthy',
                'schema_validation': 'validated',
                'performance_baseline': {
                    'baseline_latency_ms': baseline_latency,
                    'connection_acquire_time_ms': 0.0  # To be measured
                }
            })

    except Exception as e:
        validation_result['connectivity_status'] = 'failed'
        validation_result['error'] = str(e)
        raise OnexError(
            code=CoreErrorCode.DATABASE_CONNECTION_ERROR,
            message=f"Database connectivity validation failed: {e}"
        ) from e

    return validation_result
```

### 4. Circuit Breaker Thresholds Validation During Initialization

**File:** `src/omnibase_infra/infrastructure/event_bus_circuit_breaker.py`
**Enhancement needed:**

```python
def validate_circuit_breaker_thresholds(self) -> Dict[str, Any]:
    """Validate circuit breaker threshold configuration."""
    validation_result = {
        'threshold_validation': 'unknown',
        'configuration_consistency': 'unknown',
        'operational_parameters': {}
    }

    try:
        # Validate failure threshold
        if self.config.failure_threshold <= 0:
            raise ValueError("Failure threshold must be positive")

        # Validate recovery timeout
        if self.config.recovery_timeout <= 0:
            raise ValueError("Recovery timeout must be positive")

        # Validate success threshold for half-open state
        if self.config.success_threshold <= 0:
            raise ValueError("Success threshold must be positive")

        # Validate timeout consistency
        if self.config.timeout_seconds >= self.config.recovery_timeout:
            raise ValueError("Operation timeout should be less than recovery timeout")

        # Validate queue size
        if self.config.max_queue_size <= 0:
            raise ValueError("Max queue size must be positive")

        validation_result.update({
            'threshold_validation': 'validated',
            'configuration_consistency': 'consistent',
            'operational_parameters': {
                'failure_threshold': self.config.failure_threshold,
                'recovery_timeout': self.config.recovery_timeout,
                'success_threshold': self.config.success_threshold,
                'max_queue_size': self.config.max_queue_size,
                'estimated_recovery_cycles': self._calculate_recovery_cycles()
            }
        })

    except Exception as e:
        validation_result['threshold_validation'] = 'failed'
        validation_result['error'] = str(e)
        raise OnexError(
            code=CoreErrorCode.CONFIGURATION_ERROR,
            message=f"Circuit breaker threshold validation failed: {e}"
        ) from e

    return validation_result
```

### 5. Centralized Configuration Validation for All Infrastructure Components

**New File:** `src/omnibase_infra/infrastructure/infrastructure_config_validator.py`
**Purpose:** Central coordinator for all infrastructure configuration validation

```python
"""Centralized Infrastructure Configuration Validator.

Provides centralized validation for all infrastructure components during startup.
Ensures configuration consistency, connectivity validation, and operational readiness.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import asyncio
import time
from datetime import datetime

from omnibase_core.core.errors.onex_error import OnexError
from omnibase_core.core.core_error_codes import CoreErrorCode

@dataclass
class InfrastructureValidationResult:
    """Results from infrastructure configuration validation."""
    component_name: str
    validation_status: str  # 'success', 'warning', 'error'
    validation_time_ms: float
    configuration_details: Dict[str, Any]
    connectivity_status: Dict[str, Any]
    performance_baseline: Dict[str, Any]
    error_details: Optional[str] = None

class InfrastructureConfigValidator:
    """Centralized validator for all infrastructure components."""

    def __init__(self):
        self.validation_results: List[InfrastructureValidationResult] = []
        self.overall_status: str = 'unknown'
        self.validation_start_time: Optional[datetime] = None
        self.validation_duration_ms: float = 0.0

    async def validate_all_infrastructure(self) -> Dict[str, Any]:
        """Validate all infrastructure components."""
        self.validation_start_time = datetime.now()
        start_time = time.time()

        try:
            # Run all validations in parallel for efficiency
            validation_tasks = [
                self._validate_postgres_infrastructure(),
                self._validate_kafka_infrastructure(),
                self._validate_circuit_breaker_infrastructure(),
                self._validate_observability_infrastructure()
            ]

            # Wait for all validations to complete
            results = await asyncio.gather(*validation_tasks, return_exceptions=True)

            # Process results and handle any exceptions
            for result in results:
                if isinstance(result, Exception):
                    self.validation_results.append(
                        InfrastructureValidationResult(
                            component_name='unknown',
                            validation_status='error',
                            validation_time_ms=0.0,
                            configuration_details={},
                            connectivity_status={},
                            performance_baseline={},
                            error_details=str(result)
                        )
                    )
                else:
                    self.validation_results.extend(result)

            # Determine overall status
            self._determine_overall_status()

            self.validation_duration_ms = (time.time() - start_time) * 1000

            return self._generate_validation_report()

        except Exception as e:
            self.overall_status = 'critical_error'
            raise OnexError(
                code=CoreErrorCode.INFRASTRUCTURE_ERROR,
                message=f"Infrastructure validation failed: {e}"
            ) from e
```

## Configuration Consolidation Success Metrics

- All infrastructure components validate configuration at startup
- Zero configuration-related runtime errors
- Centralized configuration validation completes in <2 seconds
- All connectivity validation passes before component initialization
- Configuration errors provide actionable error messages
- Validation results available for monitoring and alerting

## Implementation Requirements

### ONEX Contract Compliance
- All configuration validation must follow contract patterns
- Use Pydantic models for configuration validation
- Strong typing for all configuration parameters
- OnexError chaining for all validation failures

### Performance Requirements
- Configuration validation must complete quickly (<2 seconds total)
- Parallel validation for independent components
- Minimal impact on startup time
- Efficient validation algorithms

### Monitoring Integration
- All validation results must be observable
- Integration with existing monitoring infrastructure
- Alerting for configuration validation failures
- Historical tracking of validation performance
