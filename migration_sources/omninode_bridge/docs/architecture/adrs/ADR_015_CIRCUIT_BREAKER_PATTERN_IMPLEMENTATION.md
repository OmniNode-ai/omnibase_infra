# ADR-015: Circuit Breaker Pattern Implementation

**Status**: Accepted
**Date**: 2024-09-25
**Deciders**: OmniNode Bridge Architecture Team
**Technical Story**: Implementation of circuit breaker resilience pattern for external service dependencies

## Context

The multi-service architecture relies heavily on external dependencies including AI model services, databases, and third-party APIs. Without proper resilience patterns, failures in external services can cause:

- Cascading failures across service boundaries
- Resource exhaustion from repeated failed requests
- Poor user experience during service degradation
- Inability to provide graceful degradation functionality
- Extended recovery times when services become available

Traditional retry mechanisms without circuit breakers can amplify problems by continuing to hammer failing services, preventing them from recovering and consuming valuable resources.

## Decision

We implement a comprehensive Circuit Breaker pattern with graceful degradation capabilities through the `GracefulDegradationService` with the following architecture:

### Circuit Breaker States
1. **Closed**: Service is healthy, requests pass through normally
2. **Open**: Service is failing, requests fail immediately or use fallbacks
3. **Half-Open**: Testing service recovery with limited requests

### Degradation Modes
- **FAIL_FAST**: Immediate failure when service is unavailable
- **GRACEFUL**: Use fallback functionality when service fails
- **CACHED**: Return cached responses during service outages
- **OFFLINE**: Continue operation without external service
- **CIRCUIT_BREAKER**: Full circuit breaker pattern with state management

### Service Configuration
```python
ServiceConfig(
    name="service_name",
    health_check_url="http://service/health",
    health_check_timeout=5.0,
    health_check_interval=30.0,
    degradation_mode=DegradationMode.GRACEFUL,
    circuit_breaker_threshold=5,     # Failures before opening circuit
    circuit_breaker_timeout=60.0,    # Time before trying to close circuit
    cache_ttl=300.0,                 # Cache TTL in seconds
    max_retries=3,
    retry_delay=1.0,
    fallback_enabled=True
)
```

### Implementation Components

#### 1. Health Monitoring
- Periodic health checks with configurable intervals
- Response time tracking and failure counting
- Automatic circuit state transitions based on failure thresholds

#### 2. Circuit State Management
```python
@property
def should_use_circuit_breaker(self) -> bool:
    """Check if circuit breaker should be used."""
    return (self.circuit_breaker_open and
            self.circuit_breaker_until and
            datetime.now(UTC) < self.circuit_breaker_until)
```

#### 3. Fallback Execution
```python
async def execute_with_fallback(
    self,
    service_name: str,
    operation: str,
    primary_func: Callable,
    fallback_func: Optional[Callable] = None,
    cache_key: Optional[str] = None,
    **kwargs,
) -> Any:
    # Execute primary function or fallback based on circuit state
```

#### 4. Response Caching
- Automatic caching of successful responses
- TTL-based cache expiration
- Intelligent cache usage during service degradation

### Service Integration Examples

#### AI Model Services
```python
degradation_service.register_service(
    "ai_model",
    ServiceConfig(
        name="ai_model",
        degradation_mode=DegradationMode.GRACEFUL,
        circuit_breaker_threshold=3,
        circuit_breaker_timeout=120.0,
        fallback_enabled=True,
    ),
    fallback_func=ai_model_fallback,
)
```

#### Smart Responder Chain
```python
degradation_service.register_service(
    "smart_responder",
    ServiceConfig(
        name="smart_responder",
        health_check_url="http://localhost:8001/health",
        degradation_mode=DegradationMode.CACHED,
        cache_ttl=600.0,  # 10 minutes
        circuit_breaker_threshold=5,
        circuit_breaker_timeout=300.0,
        fallback_enabled=True,
    ),
    fallback_func=smart_responder_fallback,
)
```

#### External API Services
```python
degradation_service.register_service(
    "external_api",
    ServiceConfig(
        name="external_api",
        degradation_mode=DegradationMode.CACHED,
        cache_ttl=1800.0,  # 30 minutes
        circuit_breaker_threshold=10,
        circuit_breaker_timeout=600.0,
        max_retries=5,
    ),
)
```

## Consequences

### Positive Consequences

- **Failure Isolation**: Circuit breakers prevent cascading failures across service boundaries
- **Resource Protection**: Failing services are protected from additional load during outages
- **Graceful Degradation**: System continues operating with reduced functionality rather than failing completely
- **Fast Recovery**: Circuit breakers automatically detect service recovery and resume normal operation
- **User Experience**: Fallback responses provide immediate feedback instead of timeouts
- **Observability**: Comprehensive metrics and health status for all external dependencies
- **Configurable Resilience**: Different degradation modes for different service criticality levels
- **Automatic Management**: No manual intervention required for circuit state transitions

### Negative Consequences

- **Increased Complexity**: Additional components and configuration required
- **False Positives**: Healthy services may be temporarily marked as failing due to network issues
- **Stale Data**: Cached responses may not reflect current state during outages
- **Resource Overhead**: Health check tasks and monitoring consume system resources
- **Configuration Complexity**: Multiple parameters need tuning for optimal performance
- **Testing Challenges**: Circuit breaker states are difficult to test comprehensively

## Implementation Details

### Health Check Loop
```python
async def _health_check_loop(self, service_name: str, health_check_func: Callable) -> None:
    """Run periodic health checks for a service."""
    while not self.shutdown_event.is_set():
        try:
            # Skip health check if circuit breaker is open and not expired
            if health.should_use_circuit_breaker:
                await asyncio.sleep(config.health_check_interval)
                continue

            is_healthy = await health_check_func(service_name)

            if is_healthy:
                # Reset circuit breaker on successful health check
                health.consecutive_failures = 0
                health.circuit_breaker_open = False
                health.status = ServiceStatus.HEALTHY
            else:
                # Increment failure count and check threshold
                health.consecutive_failures += 1
                if health.consecutive_failures >= config.circuit_breaker_threshold:
                    # Open circuit breaker
                    health.circuit_breaker_open = True
                    health.circuit_breaker_until = datetime.now(UTC) + timedelta(
                        seconds=config.circuit_breaker_timeout
                    )
                    health.status = ServiceStatus.UNAVAILABLE
```

### Context Manager Integration
```python
async with degradation_service.with_service("ai_model", "inference") as ctx:
    if ctx["should_use_service"]:
        # Use primary service
        result = await primary_ai_service.infer(request)
    elif ctx["should_use_fallback"]:
        # Use fallback
        result = await fallback_ai_service.infer(request)
    elif ctx["should_use_cache"]:
        # Use cached response
        result = degradation_service.get_cached_response("ai_model", cache_key)
```

### Metrics and Monitoring
- **total_requests**: Total number of service requests
- **successful_requests**: Successfully completed requests
- **failed_requests**: Failed primary service requests
- **degraded_requests**: Requests completed in degraded mode
- **cached_responses**: Responses served from cache
- **fallback_responses**: Responses from fallback functions

### Service Status API
```python
# Get individual service health
health = degradation_service.get_service_health("smart_responder")

# Get all service status
all_status = degradation_service.get_all_health_status()

# Get service statistics
stats = degradation_service.get_stats()
```

## Compliance

This implementation aligns with ONEX standards by:

- **Resilience First**: Circuit breakers are a core resilience pattern
- **Observability**: Comprehensive metrics and health status monitoring
- **Graceful Degradation**: Multiple degradation strategies based on service criticality
- **Automatic Recovery**: Self-healing capabilities without manual intervention
- **Configuration Management**: Service-specific configuration with sensible defaults
- **Error Handling**: Structured error handling with fallback mechanisms

## Related Decisions

- ADR-013: Multi-Service Architecture Pattern
- ADR-014: Event-Driven Architecture with Kafka
- ADR-016: Database Strategy for Multi-Service Architecture
- ADR-017: Authentication and Authorization Strategy

## References

- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)
- [Release It! - Michael T. Nygard](https://pragprog.com/titles/mnee2/release-it-second-edition/)
- [Building Resilient Systems](https://www.oreilly.com/library/view/building-microservices/9781491950340/)
- [Netflix Hystrix Documentation](https://github.com/Netflix/Hystrix/wiki)
- [Microsoft Azure Circuit Breaker Pattern](https://docs.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker)
