# ADR-017: Authentication and Authorization Strategy

**Status**: Accepted
**Date**: 2024-09-25
**Deciders**: OmniNode Bridge Architecture Team
**Technical Story**: Implementation of comprehensive authentication and authorization system for multi-service architecture

## Context

The multi-service architecture requires a robust authentication and authorization system that can handle:

- Service-to-service authentication across distributed components
- External API access with proper security controls
- Multiple authentication methods for different use cases
- API key management with rotation and lifecycle control
- Comprehensive audit logging for security compliance
- Rate limiting and abuse prevention
- JWT-based authentication for stateless operations
- Environment-specific security configurations

Traditional authentication approaches often require complex token servers or centralized authentication services that add operational overhead and single points of failure to distributed systems.

## Decision

We adopt a **Hybrid Authentication Strategy** with multiple authentication methods optimized for different use cases:

### Authentication Architecture

#### 1. Primary Authentication Methods

**API Key Authentication** (Primary for service-to-service)
```python
# Dual-header support for flexibility
Authorization: Bearer <api_key>
X-API-Key: <api_key>

# Automatic key validation
async def verify_api_key(
    request: Request,
    authorization: HTTPAuthorizationCredentials | None = Depends(HTTPBearer(auto_error=False)),
    x_api_key: str | None = Header(None),
) -> bool:
    # Validate from either header with comprehensive audit logging
```

**JWT Authentication** (For complex authorization scenarios)
```python
# Comprehensive JWT claims with security extensions
JWTClaims(
    sub="user_id",          # Subject
    iss="omninode-bridge",  # Issuer
    aud="omninode-api",     # Audience
    scope=["read", "write"], # Permissions
    role="admin",           # User role
    security_level="high",  # Security classification
    session_id="uuid",      # Session tracking
    ip_address="bound_ip"   # IP binding (optional)
)
```

#### 2. Security Middleware Architecture
```python
class SecurityMiddleware(BaseHTTPMiddleware):
    """Comprehensive security processing pipeline."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 1. Request size validation
        await self._validate_request_size(request)

        # 2. Suspicious activity detection
        await self._detect_suspicious_activity(request)

        # 3. Request content validation
        await self._validate_request_content(request)

        # 4. Process authenticated request
        response = await call_next(request)

        # 5. Comprehensive audit logging
        await self._audit_log_request(request, response)

        return response
```

### API Key Management Strategy

#### 1. Secure Key Generation and Storage
```python
class ApiKeyManager:
    """Enterprise-grade API key management with rotation."""

    def __init__(self,
                 rotation_interval_hours: int = 168,  # 7 days
                 max_keys_retained: int = 3):
        # Encryption key derived from multiple environment variables
        self._encryption_key = self._derive_encryption_key()

    def _derive_encryption_key(self) -> bytes:
        """PBKDF2-based key derivation from environment."""
        seed = (
            os.getenv("API_KEY_ENCRYPTION_SEED") +
            os.getenv("SERVICE_INSTANCE_ID") +
            self.service_name
        )
        return hashlib.pbkdf2_hmac("sha256", seed.encode(), salt, 100000, 32)
```

#### 2. Automatic Key Rotation
- **Rotation Interval**: Configurable (default: 7 days)
- **Grace Period**: Old keys remain valid during rotation window
- **Zero-Downtime**: Multiple active keys during transition
- **Audit Trail**: Complete rotation history with timestamps

#### 3. Key Metadata Tracking
```python
class ApiKeyMetadata(BaseModel):
    key_id: str
    created_at: datetime
    expires_at: datetime | None
    last_used: datetime | None
    usage_count: int
    is_active: bool
    created_by: str
    description: str
```

### Authorization Patterns

#### 1. Role-Based Access Control (RBAC)
```python
# JWT-based role authorization
@require_role("admin")
async def admin_endpoint():
    pass

@require_scope(["read", "write"])
async def data_endpoint():
    pass
```

#### 2. API Key Scoping
```python
# Environment-based API key restrictions
RateLimitScope.PER_API_KEY: {
    "requests_per_minute": 100,
    "burst_capacity": 20,
    "scope_restrictions": ["read_only", "write_limited"]
}
```

#### 3. Environment-Specific Security Levels
```python
# Production: Strict security
if environment == "production":
    security_config = SecurityConfig(
        require_https=True,
        jwt_algorithm="RS256",
        key_rotation_hours=168,  # 7 days
        audit_level="comprehensive",
        rate_limit_strict=True
    )

# Development: Relaxed for testing
elif environment == "development":
    security_config = SecurityConfig(
        require_https=False,
        jwt_algorithm="HS256",
        key_rotation_hours=720,  # 30 days
        audit_level="basic",
        disable_auth=True  # Optional for testing
    )
```

### Comprehensive Audit System

#### 1. Security Event Logging
```python
# Automated security event tracking
AuditEventType = {
    AUTHENTICATION_SUCCESS: "auth_success",
    AUTHENTICATION_FAILURE: "auth_failure",
    INVALID_API_KEY: "invalid_api_key",  # pragma: allowlist secret
    MISSING_API_KEY: "missing_api_key",  # pragma: allowlist secret
    SUSPICIOUS_ACTIVITY: "suspicious_activity",
    RATE_LIMIT_EXCEEDED: "rate_limit_exceeded",
    SECURITY_VIOLATION: "security_violation"
}
```

#### 2. Request Tracking
```python
# Every authenticated request logged with:
{
    "event_type": "api_request",
    "timestamp": "2024-09-25T12:00:00Z",
    "request_id": "uuid",
    "client_info": {
        "ip_address": "10.0.1.100",
        "user_agent": "client/1.0",
        "api_key_id": "key_abc123"
    },
    "request_info": {
        "method": "POST",
        "path": "/api/v1/workflows",
        "response_status": 200,
        "response_time_ms": 45.2
    },
    "security_context": {
        "auth_method": "api_key",
        "rate_limit_remaining": 95,
        "suspicious_score": 0.1
    }
}
```

### Rate Limiting Strategy

#### 1. Multi-Tier Rate Limiting
```python
# Per-IP rate limiting
@limiter.limit("100/minute")
async def endpoint():
    pass

# Per-API-Key rate limiting
@limiter.limit("1000/minute", key_func=lambda request: get_api_key(request))
async def authenticated_endpoint():
    pass

# Endpoint-specific limits
@limiter.limit("10/minute")  # Expensive operations
async def heavy_computation_endpoint():
    pass
```

#### 2. Adaptive Rate Limiting
```python
# Dynamic limits based on user behavior
class AdaptiveRateLimiter:
    def calculate_limit(self, api_key: str, endpoint: str) -> int:
        user_reputation = self.get_user_reputation(api_key)
        base_limit = self.endpoint_limits[endpoint]

        if user_reputation > 0.8:
            return int(base_limit * 1.5)  # Trusted users get more
        elif user_reputation < 0.3:
            return int(base_limit * 0.5)  # Suspicious users get less
        return base_limit
```

### JWT Configuration Best Practices

#### 1. Security-First JWT Configuration
```python
JWTConfig(
    algorithm="RS256",              # Asymmetric for production
    access_token_expire_minutes=30, # Short-lived access tokens
    refresh_token_expire_days=7,    # Longer-lived refresh tokens
    require_exp=True,               # Always require expiration
    verify_signature=True,          # Always verify signatures
    leeway=30,                     # 30s clock skew tolerance
    issuer="omninode-bridge",      # Consistent issuer
    audience="omninode-api"        # Specific audience validation
)
```

#### 2. Advanced JWT Claims
```python
# Security-enhanced JWT payload
{
    "sub": "user_123",
    "iss": "omninode-bridge",
    "aud": "omninode-api",
    "exp": 1234567890,
    "iat": 1234567800,
    "nbf": 1234567800,
    "jti": "unique-token-id",

    # Custom security claims
    "scope": ["workflows:read", "workflows:write"],
    "role": "service_admin",
    "session_id": "sess_abc123",
    "client_id": "app_xyz789",
    "security_level": "high",
    "ip_address": "10.0.1.100",  # IP binding for high-security scenarios
    "device_id": "device_456"    # Device binding
}
```

## Implementation Details

### Service Integration Pattern
```python
# Standard service authentication setup
def create_service_app(service_name: str, disable_auth: bool = False):
    app = FastAPI(title=f"OmniNode Bridge - {service_name}")

    # Security middleware
    app.add_middleware(
        SecurityMiddleware,
        service_name=service_name,
        api_key=os.getenv("API_KEY"),
        enable_audit_logging=True,
        enable_request_validation=True
    )

    # Rate limiting
    app.state.limiter = Limiter(key_func=get_remote_address)
    app.add_exception_handler(RateLimitExceeded, rate_limit_handler)

    # Authentication dependency
    auth_dependency = get_auth_dependency(disable_auth)

    return app, auth_dependency
```

### Environment Configuration
```yaml
# Production
environment: production
auth:
  api_key_rotation_hours: 168    # 7 days
  jwt_algorithm: "RS256"
  require_https: true
  audit_level: "comprehensive"
  rate_limit_multiplier: 1.0

# Staging
environment: staging
auth:
  api_key_rotation_hours: 336    # 14 days
  jwt_algorithm: "RS256"
  require_https: true
  audit_level: "standard"
  rate_limit_multiplier: 2.0

# Development
environment: development
auth:
  api_key_rotation_hours: 720    # 30 days
  jwt_algorithm: "HS256"
  require_https: false
  audit_level: "basic"
  rate_limit_multiplier: 10.0
  disable_auth: true             # Optional for testing
```

### Security Hardening
```python
# HTTPS enforcement in production
@app.middleware("http")
async def force_https(request: Request, call_next):
    if os.getenv("ENVIRONMENT") == "production":
        if not request.url.scheme == "https":
            return RedirectResponse(
                url=request.url.replace(scheme="https"),
                status_code=301
            )
    response = await call_next(request)
    return response

# Security headers
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000"
    return response
```

## Consequences

### Positive Consequences

- **Multiple Authentication Options**: Flexible authentication for different use cases and clients
- **Zero-Downtime Key Rotation**: Automatic key rotation without service interruption
- **Comprehensive Audit Trail**: Complete security event logging for compliance and monitoring
- **Environment-Specific Security**: Appropriate security levels for development, staging, and production
- **Rate Limiting Protection**: Multi-tier protection against abuse and DOS attacks
- **Stateless Architecture**: JWT support enables stateless authentication for scalability
- **Security Middleware**: Centralized security processing across all services
- **Advanced JWT Features**: IP binding, device tracking, and custom security claims

### Negative Consequences

- **Complexity**: Multiple authentication methods increase implementation and maintenance complexity
- **Key Management Overhead**: API key rotation and lifecycle management requires careful coordination
- **Performance Impact**: Security middleware and audit logging add processing overhead
- **Configuration Complexity**: Environment-specific configurations require careful management
- **Storage Requirements**: Comprehensive audit logging requires significant storage capacity
- **Debugging Complexity**: Multiple authentication paths can complicate troubleshooting

### Security Considerations

- **Key Storage Security**: Encrypted key storage with PBKDF2 key derivation
- **Audit Log Protection**: Secure audit log storage and integrity verification
- **Rate Limit Bypass**: Advanced users may try to bypass rate limiting through key rotation
- **JWT Token Security**: Proper token lifecycle management and revocation strategies needed
- **Environment Isolation**: Different security configurations must be properly isolated

## Compliance

This authentication strategy aligns with ONEX standards by:

- **Security First**: Multiple layers of security with comprehensive audit logging
- **Zero Trust**: Every request authenticated and authorized regardless of source
- **Observability**: Complete audit trail for all authentication and authorization events
- **Resilience**: Graceful degradation when authentication services are degraded
- **Performance**: Optimized authentication path with minimal latency impact
- **Operational Excellence**: Automated key rotation and lifecycle management

## Migration Strategy

### Phase 1: Foundation Authentication
- Deploy API key authentication for all services
- Implement basic audit logging and rate limiting
- Configure environment-specific security settings

### Phase 2: Enhanced Security Features
- Deploy comprehensive security middleware
- Implement automatic API key rotation
- Enable advanced audit logging and monitoring

### Phase 3: JWT Integration
- Deploy JWT authentication for complex scenarios
- Implement role-based access control (RBAC)
- Enable advanced authorization patterns

### Phase 4: Advanced Security Features
- Deploy adaptive rate limiting based on user behavior
- Implement IP and device binding for high-security scenarios
- Enable comprehensive threat detection and response

## Related Decisions

- ADR-013: Multi-Service Architecture Pattern
- ADR-015: Circuit Breaker Pattern Implementation
- ADR-016: Database Strategy for Multi-Service Architecture
- ADR-018: Rate Limiting and API Security Strategy

## References

- [JWT Security Best Practices](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-jwt-bcp)
- [API Security Best Practices](https://owasp.org/www-project-api-security/)
- [FastAPI Security Documentation](https://fastapi.tiangolo.com/tutorial/security/)
- [OAuth 2.0 Security Considerations](https://datatracker.ietf.org/doc/html/rfc6819)
- [NIST Authentication Guidelines](https://pages.nist.gov/800-63-3/)
