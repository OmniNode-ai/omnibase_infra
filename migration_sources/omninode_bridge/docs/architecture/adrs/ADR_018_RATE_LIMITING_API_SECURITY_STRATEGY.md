# ADR-018: Rate Limiting and API Security Strategy

**Status**: Accepted
**Date**: 2024-09-25
**Deciders**: OmniNode Bridge Architecture Team
**Technical Story**: Implementation of comprehensive rate limiting and API security controls for multi-service architecture

## Context

The multi-service architecture (HookReceiver, ModelMetrics API, WorkflowCoordinator) requires robust rate limiting and API security controls that can handle:

- Protection against denial of service (DoS) and distributed denial of service (DDoS) attacks
- Fair resource allocation across multiple clients and API consumers
- Abuse prevention while maintaining legitimate user experience
- Different rate limiting strategies for different endpoint types and criticality levels
- API security headers and content validation
- Request size limits and payload validation
- Integration with authentication and authorization systems
- Monitoring and alerting for suspicious activity patterns
- Graceful degradation under high load conditions

Traditional rate limiting approaches often use simple per-IP limits that don't account for legitimate high-volume users, authenticated vs unauthenticated traffic, or different endpoint criticality levels.

## Decision

We adopt a **Multi-Tier Adaptive Rate Limiting Strategy** with comprehensive API security controls using the following architecture:

### Rate Limiting Architecture

#### 1. Multi-Tier Rate Limiting Levels
```python
# Per-IP rate limiting (baseline protection)
@limiter.limit("100/minute")  # Basic per-IP protection

# Per-API-Key rate limiting (authenticated users)
@limiter.limit("1000/minute", key_func=get_api_key)  # Higher limits for authenticated users

# Endpoint-specific rate limiting (criticality-based)
@limiter.limit("10/minute")   # Expensive operations (workflows, bulk operations)
@limiter.limit("300/minute")  # Standard operations (status checks, simple queries)
@limiter.limit("1000/minute") # Health checks and monitoring endpoints
```

#### 2. Service-Specific Rate Limit Configuration
```python
# HookReceiver Service - High throughput event processing
HOOK_RECEIVER_LIMITS = {
    "webhook_events": "1000/minute",      # High volume webhook processing
    "health_check": "100/minute",         # Standard health monitoring
    "metrics": "500/minute",              # Moderate metrics collection
}

# ModelMetrics API Service - Analytics and reporting
MODEL_METRICS_LIMITS = {
    "model_inference": "500/minute",      # Model execution requests
    "metrics_query": "200/minute",        # Complex analytics queries
    "batch_operations": "20/minute",      # Resource-intensive batch jobs
    "health_check": "100/minute",         # Standard health monitoring
}

# WorkflowCoordinator Service - Orchestration and management
WORKFLOW_COORDINATOR_LIMITS = {
    "workflow_execute": "50/minute",      # Workflow execution (expensive)
    "workflow_status": "300/minute",      # Status monitoring (frequent)
    "workflow_pause": "100/minute",       # Control operations
    "workflow_resume": "100/minute",      # Control operations
    "workflow_cancel": "100/minute",      # Control operations
    "workflow_list": "200/minute",        # List operations
    "health_check": "100/minute",         # Standard health monitoring
}
```

### Adaptive Rate Limiting Features

#### 1. User Reputation-Based Limits
```python
class AdaptiveRateLimiter:
    """Dynamic rate limiting based on user behavior and reputation."""

    def calculate_dynamic_limit(self, api_key: str, endpoint: str) -> int:
        """Calculate adaptive rate limit based on user reputation."""
        base_limit = self.get_base_limit(endpoint)
        reputation_score = self.get_user_reputation(api_key)

        # Reputation-based multipliers
        if reputation_score > 0.9:
            return int(base_limit * 2.0)      # Trusted users: 2x limit
        elif reputation_score > 0.7:
            return int(base_limit * 1.5)      # Good users: 1.5x limit
        elif reputation_score < 0.3:
            return int(base_limit * 0.5)      # Suspicious users: 0.5x limit
        elif reputation_score < 0.1:
            return int(base_limit * 0.1)      # Problematic users: 0.1x limit

        return base_limit  # Standard limit

    def update_reputation(self, api_key: str, event_type: str) -> None:
        """Update user reputation based on behavior."""
        current_score = self.get_user_reputation(api_key)

        reputation_adjustments = {
            "successful_request": +0.001,     # Slight positive for normal usage
            "rate_limit_hit": -0.05,          # Moderate penalty for hitting limits
            "invalid_request": -0.02,         # Small penalty for bad requests
            "security_violation": -0.2,       # Large penalty for security issues
            "suspicious_activity": -0.1,      # Penalty for suspicious patterns
            "long_term_good_behavior": +0.01, # Bonus for consistent good behavior
        }

        adjustment = reputation_adjustments.get(event_type, 0)
        new_score = max(0.0, min(1.0, current_score + adjustment))
        self.set_user_reputation(api_key, new_score)
```

#### 2. Time-Based Dynamic Limits
```python
class TimeBasedRateLimiter:
    """Adjust limits based on time patterns and system load."""

    def get_time_adjusted_limit(self, base_limit: int) -> int:
        """Adjust rate limits based on time of day and system load."""
        current_hour = datetime.now().hour
        system_load = self.get_system_load_factor()

        # Time-of-day adjustments (assume peak hours 9-17 UTC)
        if 9 <= current_hour <= 17:
            time_multiplier = 0.8  # Reduce limits during peak hours
        elif 22 <= current_hour <= 6:
            time_multiplier = 1.5  # Higher limits during off-peak hours
        else:
            time_multiplier = 1.0  # Standard limits

        # System load adjustments
        if system_load > 0.9:
            load_multiplier = 0.5  # Drastically reduce during high load
        elif system_load > 0.7:
            load_multiplier = 0.7  # Moderately reduce during elevated load
        elif system_load < 0.3:
            load_multiplier = 1.3  # Increase during low load
        else:
            load_multiplier = 1.0  # Standard limits

        adjusted_limit = int(base_limit * time_multiplier * load_multiplier)
        return max(1, adjusted_limit)  # Ensure at least 1 request allowed
```

### API Security Controls

#### 1. Request Size and Content Validation
```python
class APISecurityMiddleware(BaseHTTPMiddleware):
    """Comprehensive API security middleware."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Request size validation
        await self._validate_request_size(request)

        # Content type validation
        await self._validate_content_type(request)

        # Payload validation and sanitization
        await self._validate_and_sanitize_payload(request)

        # Suspicious pattern detection
        await self._detect_suspicious_patterns(request)

        # Security headers enforcement
        response = await call_next(request)
        self._add_security_headers(response)

        return response

    async def _validate_request_size(self, request: Request) -> None:
        """Enforce maximum request size limits."""
        content_length = request.headers.get("content-length")
        if content_length:
            size = int(content_length)
            max_sizes = {
                "/api/v1/hooks/receive": 10 * 1024 * 1024,    # 10MB for webhook payloads
                "/api/v1/workflows/execute": 5 * 1024 * 1024,  # 5MB for workflow definitions
                "default": 1 * 1024 * 1024,                   # 1MB default limit
            }

            endpoint = request.url.path
            max_size = max_sizes.get(endpoint, max_sizes["default"])

            if size > max_size:
                raise HTTPException(
                    status_code=413,
                    detail=f"Request size {size} exceeds maximum allowed {max_size} bytes"
                )

    async def _detect_suspicious_patterns(self, request: Request) -> None:
        """Detect suspicious request patterns."""
        suspicious_indicators = []

        # Check for SQL injection patterns
        payload = await self._get_request_payload(request)
        if payload and self._contains_sql_injection_patterns(payload):
            suspicious_indicators.append("potential_sql_injection")

        # Check for excessive special characters
        if self._excessive_special_characters(payload):
            suspicious_indicators.append("excessive_special_characters")

        # Check for known attack vectors
        user_agent = request.headers.get("user-agent", "")
        if self._suspicious_user_agent(user_agent):
            suspicious_indicators.append("suspicious_user_agent")

        # Check request frequency from same IP
        if await self._rapid_fire_detection(request):
            suspicious_indicators.append("rapid_fire_requests")

        if suspicious_indicators:
            await self._log_suspicious_activity(request, suspicious_indicators)

            # Implement progressive response based on suspicion level
            if len(suspicious_indicators) >= 3:
                raise HTTPException(status_code=429, detail="Suspicious activity detected")
            elif len(suspicious_indicators) >= 2:
                await asyncio.sleep(2)  # Slow down suspicious requests
```

#### 2. Security Headers and CORS Configuration
```python
def add_comprehensive_security_headers(response: Response) -> None:
    """Add comprehensive security headers to all responses."""

    # Content Security Policy
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "connect-src 'self' https:; "
        "font-src 'self'; "
        "object-src 'none'; "
        "media-src 'self'; "
        "frame-src 'none';"
    )

    # XSS Protection
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"

    # HSTS for HTTPS enforcement
    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains; preload"
        )

    # Additional security headers
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = (
        "accelerometer=(), camera=(), geolocation=(), "
        "gyroscope=(), magnetometer=(), microphone=(), "
        "payment=(), usb=()"
    )

    # API-specific headers
    response.headers["X-API-Version"] = "v1"
    response.headers["X-Rate-Limit-Remaining"] = str(get_remaining_requests())
    response.headers["X-Request-ID"] = str(uuid.uuid4())
```

### Monitoring and Alerting

#### 1. Rate Limit Monitoring
```python
class RateLimitMonitor:
    """Monitor rate limiting effectiveness and suspicious activity."""

    async def record_rate_limit_event(
        self,
        event_type: str,
        client_identifier: str,
        endpoint: str,
        limit_type: str,
        additional_data: dict = None
    ) -> None:
        """Record rate limiting events for monitoring."""

        event_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,  # hit_limit, exceeded_limit, suspicious_activity
            "client_identifier": client_identifier,
            "endpoint": endpoint,
            "limit_type": limit_type,  # per_ip, per_api_key, per_endpoint
            "additional_data": additional_data or {}
        }

        # Store in database for analysis
        await self.postgres_client.store_rate_limit_event(event_data)

        # Real-time alerting for critical events
        if event_type in ["exceeded_limit", "suspicious_activity"]:
            await self.send_security_alert(event_data)

    async def analyze_rate_limit_patterns(self) -> dict:
        """Analyze rate limiting patterns for optimization."""

        # Query recent rate limit events
        recent_events = await self.postgres_client.get_rate_limit_events(
            hours_back=24
        )

        analysis = {
            "top_limited_endpoints": self.get_top_limited_endpoints(recent_events),
            "most_active_clients": self.get_most_active_clients(recent_events),
            "suspicious_patterns": self.detect_suspicious_patterns(recent_events),
            "limit_effectiveness": self.calculate_limit_effectiveness(recent_events),
            "recommended_adjustments": self.recommend_limit_adjustments(recent_events)
        }

        return analysis
```

#### 2. Security Event Alerting
```python
class SecurityAlertManager:
    """Manage security alerts and incident response."""

    async def send_security_alert(
        self,
        alert_type: str,
        severity: str,
        event_data: dict,
        immediate_action: bool = False
    ) -> None:
        """Send security alerts through multiple channels."""

        alert_message = {
            "alert_type": alert_type,
            "severity": severity,  # low, medium, high, critical
            "timestamp": datetime.utcnow().isoformat(),
            "service": event_data.get("service"),
            "client_info": event_data.get("client_info", {}),
            "event_details": event_data,
            "immediate_action_required": immediate_action
        }

        # Send to monitoring system
        await self.send_to_monitoring_system(alert_message)

        # Send notifications based on severity
        if severity in ["high", "critical"]:
            await self.send_immediate_notification(alert_message)

        # Log to audit system
        await self.audit_logger.log_security_event(alert_message)

        # Trigger automated responses for critical alerts
        if immediate_action and severity == "critical":
            await self.trigger_automated_response(alert_message)
```

### Implementation Integration

#### 1. Service Integration Pattern
```python
def create_secure_service_app(
    service_name: str,
    rate_limits: dict,
    disable_auth: bool = False
) -> FastAPI:
    """Create FastAPI app with comprehensive rate limiting and security."""

    app = FastAPI(title=f"OmniNode Bridge - {service_name}")

    # Initialize rate limiter with adaptive features
    limiter = AdaptiveRateLimiter(
        key_func=get_client_identifier,
        default_limits=rate_limits["default"]
    )
    app.state.limiter = limiter

    # Add security middleware
    app.add_middleware(
        APISecurityMiddleware,
        service_name=service_name,
        max_request_size=rate_limits.get("max_request_size", 1024*1024),
        enable_suspicious_detection=True
    )

    # Add rate limiting middleware
    app.add_middleware(
        RateLimitMiddleware,
        limiter=limiter,
        adaptive_limits=True,
        reputation_based=True
    )

    # Exception handlers
    app.add_exception_handler(RateLimitExceeded, custom_rate_limit_handler)
    app.add_exception_handler(SecurityViolation, security_violation_handler)

    return app
```

#### 2. Custom Rate Limit Handler
```python
async def custom_rate_limit_handler(
    request: Request,
    exc: RateLimitExceeded
) -> JSONResponse:
    """Custom rate limit exceeded handler with detailed information."""

    # Log rate limit hit
    await rate_limit_monitor.record_rate_limit_event(
        event_type="exceeded_limit",
        client_identifier=get_client_identifier(request),
        endpoint=request.url.path,
        limit_type="endpoint_specific"
    )

    # Calculate retry-after based on adaptive limits
    retry_after = limiter.get_retry_after(request)

    # Provide helpful error response
    response_data = {
        "error": "Rate limit exceeded",
        "message": f"Too many requests to {request.url.path}",
        "retry_after": retry_after,
        "limit_info": {
            "current_limit": limiter.get_current_limit(request),
            "window": limiter.get_window_info(request),
            "reset_time": limiter.get_reset_time(request).isoformat()
        },
        "suggestions": [
            "Implement exponential backoff in your client",
            "Consider upgrading to a higher tier if available",
            "Batch multiple operations into single requests where possible"
        ]
    }

    return JSONResponse(
        status_code=429,
        content=response_data,
        headers={"Retry-After": str(retry_after)}
    )
```

## Consequences

### Positive Consequences

- **DoS/DDoS Protection**: Multi-tier rate limiting prevents service overload and abuse
- **Fair Resource Allocation**: Adaptive limits ensure legitimate users get appropriate access
- **Reputation-Based Intelligence**: System learns to identify and appropriately limit problematic clients
- **Flexible Security Controls**: Comprehensive security middleware protects against multiple attack vectors
- **Detailed Monitoring**: Complete visibility into rate limiting effectiveness and security events
- **Automatic Adaptation**: System automatically adjusts to traffic patterns and threats
- **Progressive Response**: Graceful handling of suspicious activity with appropriate escalation
- **Performance Optimization**: Time-based and load-based adjustments optimize resource utilization

### Negative Consequences

- **Implementation Complexity**: Multiple rate limiting tiers and adaptive features increase system complexity
- **Configuration Management**: Many parameters require careful tuning and ongoing adjustment
- **False Positives**: Legitimate high-volume users may be incorrectly flagged as suspicious
- **Resource Overhead**: Reputation tracking and pattern detection consume additional system resources
- **Debugging Challenges**: Complex rate limiting logic can make troubleshooting more difficult
- **Client Integration**: Clients need to implement proper backoff and retry logic

### Mitigation Strategies

- **Comprehensive Testing**: Extensive testing of rate limiting scenarios and edge cases
- **Monitoring Dashboards**: Real-time visibility into rate limiting effectiveness and issues
- **Allowlist Mechanisms**: Ability to exempt specific clients from certain restrictions
- **Gradual Rollout**: Phased deployment of adaptive features with careful monitoring
- **Documentation**: Clear API documentation explaining rate limits and best practices
- **Client SDKs**: Provide SDKs with built-in rate limiting awareness and retry logic

## Implementation Details

### Rate Limit Storage
```python
# Redis-backed rate limiting for distributed services
RATE_LIMIT_STORAGE = {
    "backend": "redis",
    "host": os.getenv("REDIS_HOST", "localhost"),
    "port": int(os.getenv("REDIS_PORT", 6379)),
    "db": int(os.getenv("REDIS_DB", 0)),
    "key_prefix": "omninode_rate_limit:",
    "ttl": 3600  # 1 hour TTL for rate limit counters
}
```

### Environment Configuration
```yaml
# Production - Strict rate limiting
environment: production
rate_limiting:
  enable_adaptive_limits: true
  enable_reputation_tracking: true
  strict_security_headers: true
  suspicious_activity_threshold: 3
  max_request_size: 1048576  # 1MB default

# Development - Relaxed for testing
environment: development
rate_limiting:
  enable_adaptive_limits: false
  enable_reputation_tracking: false
  strict_security_headers: false
  rate_limit_multiplier: 10.0
  disable_rate_limiting: true  # Optional for testing
```

## Compliance

This rate limiting strategy aligns with ONEX standards by:

- **Security First**: Multiple layers of protection against abuse and attacks
- **Observability**: Comprehensive monitoring and alerting for all security events
- **Resilience**: Adaptive limits that respond to system conditions and threats
- **Performance**: Intelligent rate limiting that optimizes resource utilization
- **User Experience**: Progressive responses that don't unnecessarily block legitimate users
- **Operational Excellence**: Automated threat response and system adaptation

## Related Decisions

- ADR-017: Authentication and Authorization Strategy
- ADR-015: Circuit Breaker Pattern Implementation
- ADR-013: Multi-Service Architecture Pattern
- ADR-019: Monitoring and Observability Strategy

## References

- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [Rate Limiting Best Practices](https://docs.github.com/en/rest/guides/best-practices-for-integrators#dealing-with-rate-limits)
- [FastAPI Rate Limiting](https://slowapi.readthedocs.io/)
- [Redis Rate Limiting Patterns](https://redis.io/commands/incr#pattern-rate-limiter)
- [HTTP Security Headers](https://owasp.org/www-project-secure-headers/)
