> **Navigation**: [Home](../index.md) > [Patterns](README.md) > Error Sanitization Patterns

# Error Sanitization Patterns

## Overview

This document provides comprehensive guidance on sanitizing error messages and context in ONEX infrastructure services. Proper error sanitization prevents sensitive data exposure in logs, error responses, monitoring systems, and debugging tools while maintaining sufficient context for effective troubleshooting.

Error sanitization is a **security-critical** concern. Improperly sanitized errors can expose credentials, PII, or internal system details to attackers through log files, API responses, error tracking systems, or debugging interfaces.

## Security Requirements

### Data Classification

Understanding data sensitivity is the foundation of error sanitization:

| Classification | Examples | Sanitization Rule |
|---------------|----------|-------------------|
| **Secrets** | Passwords, API keys, tokens, certificates | **NEVER** include |
| **PII** | Names, emails, SSNs, phone numbers, addresses | **NEVER** include |
| **Internal Architecture** | Internal IPs, private paths, schema details | **Avoid** in production |
| **Operational Context** | Service names, operation names, correlation IDs | **SAFE** to include |
| **Error Metadata** | Error codes, retry counts, timestamps | **SAFE** to include |

### What NEVER to Include in Errors

The following data types must **never** appear in error messages, error context, log entries, or API responses:

#### Credentials and Secrets

```python
# NEVER include these in errors:

# Passwords and passphrases
password = "super_secret_123"
raise InfraConnectionError(f"Auth failed with password: {password}")  # NEVER

# API keys and tokens
api_key = "sk-abc123xyz789"
raise InfraConnectionError(f"API call failed with key {api_key}")  # NEVER

# Bearer tokens and JWTs
bearer_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
raise InfraAuthenticationError(f"Token invalid: {bearer_token}")  # NEVER

# Private keys and certificates
private_key = "-----BEGIN RSA PRIVATE KEY-----..."
raise SecretResolutionError(f"Key parse failed: {private_key}")  # NEVER

# Connection strings with credentials
conn_string = "postgresql://admin:secret@db.internal:5432/mydb"
raise InfraConnectionError(f"Connection failed: {conn_string}")  # NEVER
```

#### Personally Identifiable Information (PII)

```python
# NEVER include these in errors:

# Full names
user_name = "John Smith"
raise ValidationError(f"User {user_name} not authorized")  # NEVER

# Email addresses
email = "john.smith@example.com"
raise InfraAuthenticationError(f"Auth failed for {email}")  # NEVER

# Social Security Numbers
ssn = "123-45-6789"
raise ValidationError(f"Invalid SSN format: {ssn}")  # NEVER

# Phone numbers
phone = "+1-555-123-4567"
raise NotificationError(f"SMS failed to {phone}")  # NEVER

# Physical addresses
address = "123 Main St, Springfield, IL 62701"
raise DeliveryError(f"Invalid address: {address}")  # NEVER

# Credit card numbers
cc_number = "4111-1111-1111-1111"
raise PaymentError(f"Card declined: {cc_number}")  # NEVER

# IP addresses of users (in some contexts)
user_ip = "192.168.1.100"
# NOTE: Including IPs is context-dependent - avoid in user-facing errors
raise InfraAuthenticationError(
    f"Blocked request from {user_ip}",  # May be acceptable for security logs
    context=ModelInfraErrorContext(
        transport_type=EnumInfraTransportType.HTTP,
        operation="authenticate",
        target_name="api-gateway",
        correlation_id=correlation_id,
    ),
)
```

#### Internal System Details

```python
# AVOID in production errors:

# Full file paths revealing internal structure
internal_path = "/opt/onex/secrets/infisical_token.json"
raise FileNotFoundError(f"Config not found: {internal_path}")  # AVOID

# Internal hostnames and IPs
internal_host = "db-primary.internal.corp.net"
raise InfraConnectionError(f"Cannot reach {internal_host}")  # Use sanitized name

# Database schema details
schema = "users (id, password_hash, ssn_encrypted, ...)"
raise DatabaseError(f"Schema mismatch: {schema}")  # AVOID

# Stack traces with sensitive context
# Let error chaining handle this, don't include raw traces in messages

# Environment variable contents
env_value = os.environ.get("SECRET_KEY")
raise ConfigError(f"Invalid SECRET_KEY: {env_value}")  # NEVER
```

### What IS Safe to Include

The following data types are safe and helpful for debugging:

#### Operational Context

```python
# SAFE to include:

# Service names (sanitized identifiers)
service_name = "postgresql-primary"
context = ModelInfraErrorContext(
    target_name=service_name,  # SAFE
    operation="execute_query",  # SAFE
)

# Operation names
operation = "publish_message"
context = ModelInfraErrorContext(
    operation=operation,  # SAFE
)

# Correlation IDs (for request tracing)
correlation_id = uuid4()
context = ModelInfraErrorContext(
    correlation_id=correlation_id,  # SAFE
)

# Transport types
transport_type = EnumInfraTransportType.KAFKA
context = ModelInfraErrorContext(
    transport_type=transport_type,  # SAFE
)
```

#### Error Metadata

```python
# SAFE to include:

# Error codes
error_code = EnumCoreErrorCode.DATABASE_CONNECTION_ERROR
raise InfraConnectionError(
    "Connection failed",
    context=context,
)  # Error code is part of the error class

# Retry counts
retry_count = 3
raise InfraConnectionError(
    f"Connection failed after {retry_count} attempts",  # SAFE
    context=context,
)

# Timeout values
timeout_seconds = 30
raise InfraTimeoutError(
    f"Operation timed out after {timeout_seconds}s",  # SAFE
    context=context,
)

# Port numbers
port = 5432
raise InfraConnectionError(
    f"Cannot connect to port {port}",  # SAFE
    context=context,
)

# Resource identifiers (non-sensitive)
topic = "events.user.created"
raise InfraUnavailableError(
    f"Kafka topic {topic} unavailable",  # SAFE if topic names are not sensitive
    context=context,
)

# Timestamps
timestamp = datetime.utcnow().isoformat()
raise InfraTimeoutError(
    f"Deadline exceeded at {timestamp}",  # SAFE
    context=context,
)
```

## Implementation Patterns

### Pattern 1: Connection String Sanitization

Connection strings often contain credentials that must be stripped before error reporting.

```python
from urllib.parse import urlparse, urlunparse
from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType


def sanitize_connection_string(connection_string: str) -> str:
    """Remove credentials from connection string for safe logging.

    Args:
        connection_string: Full connection string (may contain credentials)

    Returns:
        Sanitized connection string with credentials replaced
    """
    try:
        parsed = urlparse(connection_string)

        # Replace username and password with placeholders
        if parsed.username or parsed.password:
            # Reconstruct without credentials
            sanitized = parsed._replace(
                netloc=f"***:***@{parsed.hostname}"
                + (f":{parsed.port}" if parsed.port else "")
            )
            return urlunparse(sanitized)

        return connection_string
    except Exception:
        # If parsing fails, return a generic placeholder
        return "<connection-string-redacted>"


# Usage example
def connect_to_database(connection_string: str, correlation_id: UUID) -> Connection:
    """Connect to database with sanitized error reporting."""
    context = ModelInfraErrorContext(
        transport_type=EnumInfraTransportType.DATABASE,
        operation="connect",
        target_name="postgresql-primary",
        correlation_id=correlation_id,
    )

    try:
        return create_connection(connection_string)
    except ConnectionError as e:
        # BAD: Exposes credentials
        # raise InfraConnectionError(f"Failed: {connection_string}") from e

        # GOOD: Sanitized message with context
        sanitized = sanitize_connection_string(connection_string)
        raise InfraConnectionError(
            f"Database connection failed to {sanitized}",
            context=context,
        ) from e
```

### Pattern 2: Request/Response Sanitization

API requests and responses often contain sensitive data that must be redacted.

```python
import re


# Patterns for sensitive field names
SENSITIVE_FIELD_PATTERNS = [
    re.compile(r"password", re.IGNORECASE),
    re.compile(r"secret", re.IGNORECASE),
    re.compile(r"token", re.IGNORECASE),
    re.compile(r"api_?key", re.IGNORECASE),
    re.compile(r"auth", re.IGNORECASE),
    re.compile(r"credential", re.IGNORECASE),
    re.compile(r"private_?key", re.IGNORECASE),
    re.compile(r"ssn", re.IGNORECASE),
    re.compile(r"credit_?card", re.IGNORECASE),
    re.compile(r"card_?number", re.IGNORECASE),
]


def sanitize_dict(data: dict[str, object], depth: int = 0, max_depth: int = 10) -> dict[str, object]:
    """Recursively sanitize dictionary by redacting sensitive fields.

    Args:
        data: Dictionary to sanitize
        depth: Current recursion depth
        max_depth: Maximum recursion depth to prevent infinite loops

    Returns:
        Sanitized dictionary with sensitive values replaced
    """
    if depth > max_depth:
        return {"<truncated>": "max depth exceeded"}

    sanitized = {}
    for key, value in data.items():
        # Check if key matches sensitive patterns
        is_sensitive = any(pattern.search(key) for pattern in SENSITIVE_FIELD_PATTERNS)

        if is_sensitive:
            sanitized[key] = "<redacted>"
        elif isinstance(value, dict):
            sanitized[key] = sanitize_dict(value, depth + 1, max_depth)
        elif isinstance(value, list):
            sanitized[key] = [
                sanitize_dict(item, depth + 1, max_depth) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            sanitized[key] = value

    return sanitized


# Usage example
async def call_external_api(
    request_data: dict[str, object],
    correlation_id: UUID,
) -> dict[str, object]:
    """Call external API with sanitized error reporting."""
    context = ModelInfraErrorContext(
        transport_type=EnumInfraTransportType.HTTP,
        operation="api_call",
        target_name="external-api",
        correlation_id=correlation_id,
    )

    try:
        response = await http_client.post("/api/endpoint", json=request_data)
        response.raise_for_status()
        return response.json()
    except HTTPError as e:
        # BAD: May expose sensitive request data
        # raise InfraConnectionError(f"API failed with: {request_data}") from e

        # GOOD: Sanitize request data before including in error
        sanitized_request = sanitize_dict(request_data)
        raise InfraConnectionError(
            f"External API call failed",
            context=context,
            # Include sanitized data only if helpful for debugging
            request_summary=sanitized_request,
        ) from e
```

### Pattern 3: Path Sanitization

File paths can reveal internal system structure and should be sanitized.

```python
from pathlib import Path


def sanitize_path(path: str | Path, base_paths: list[str] | None = None) -> str:
    """Sanitize file path by replacing sensitive base directories.

    Args:
        path: Full file path
        base_paths: List of base paths to sanitize (optional)

    Returns:
        Sanitized path with base directories replaced
    """
    path_str = str(path)

    # Default sensitive base paths
    sensitive_bases = base_paths or [
        "/opt/onex/secrets/",
        "/var/lib/onex/credentials/",
        "/home/",
        "/root/",
        "/etc/onex/",
    ]

    for base in sensitive_bases:
        if path_str.startswith(base):
            # Replace with generic placeholder
            relative = path_str[len(base):]
            return f"<config-dir>/{relative}"

    return path_str


# Usage example
def load_config(config_path: Path, correlation_id: UUID) -> dict[str, object]:
    """Load configuration with sanitized error reporting."""
    try:
        return yaml.safe_load(config_path.read_text())
    except FileNotFoundError as e:
        # BAD: Exposes internal path structure
        # raise ConfigError(f"Config not found: {config_path}") from e

        # GOOD: Sanitize path in error
        sanitized = sanitize_path(config_path)
        raise ProtocolConfigurationError(
            f"Configuration file not found: {sanitized}",
        ) from e
```

### Pattern 4: Exception Wrapper for Sanitization

Create a wrapper that automatically sanitizes errors before they leave the system boundary.

```python
from functools import wraps
from typing import Callable, TypeVar, ParamSpec
from omnibase_infra.errors import RuntimeHostError

P = ParamSpec("P")
T = TypeVar("T")


def sanitize_exceptions(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator to ensure exceptions are sanitized before propagation.

    This decorator catches exceptions and ensures sensitive data is not
    exposed in error messages when errors cross system boundaries.
    """
    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return await func(*args, **kwargs)
        except RuntimeHostError:
            # ONEX errors are already sanitized by design
            raise
        except Exception as e:
            # Non-ONEX errors may contain sensitive data
            # Wrap in sanitized error
            error_type = type(e).__name__
            raise RuntimeHostError(
                f"Operation failed with {error_type}",
                # Original error preserved in chain but not in message
            ) from e

    return wrapper


# Usage example
@sanitize_exceptions
async def process_user_data(user_id: str, correlation_id: UUID) -> dict[str, object]:
    """Process user data with automatic error sanitization."""
    # If any exception escapes, it will be wrapped safely
    user = await fetch_user(user_id)
    return await transform_user_data(user)
```

## Integration with ONEX Error Classes

### InfraConnectionError Sanitization

```python
from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType


def connect_to_service(
    host: str,
    port: int,
    username: str,
    password: str,  # Sensitive - never include in errors
    correlation_id: UUID,
) -> Connection:
    """Connect to service with proper error sanitization."""
    context = ModelInfraErrorContext(
        transport_type=EnumInfraTransportType.DATABASE,
        operation="connect",
        target_name=f"{host}:{port}",  # SAFE: host and port are not secrets
        correlation_id=correlation_id,
    )

    try:
        return create_connection(host, port, username, password)
    except AuthenticationError as e:
        # BAD examples - NEVER do these:
        # raise InfraConnectionError(f"Auth failed for {username}:{password}") from e
        # raise InfraConnectionError(f"Password '{password}' rejected") from e

        # GOOD: Sanitized error with context
        raise InfraConnectionError(
            "Authentication failed - verify credentials in Infisical",
            context=context,
            # Include safe metadata
            host=host,
            port=port,
            # Do NOT include: username, password
        ) from e
    except ConnectionRefusedError as e:
        # GOOD: Sanitized connection error
        raise InfraConnectionError(
            "Connection refused by remote host",
            context=context,
            host=host,
            port=port,
            retry_count=3,
        ) from e
```

### SecretResolutionError Sanitization

```python
from omnibase_infra.errors import SecretResolutionError, ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType


async def get_database_credentials(
    secret_path: str,
    correlation_id: UUID,
) -> dict[str, str]:
    """Fetch database credentials from Infisical with sanitized errors."""
    context = ModelInfraErrorContext(
        transport_type=EnumInfraTransportType.INFISICAL,
        operation="read_secret",
        target_name="infisical-primary",
        correlation_id=correlation_id,
    )

    try:
        secret = await infisical_client.read(secret_path)
        return secret["data"]
    except SecretNotFoundError as e:
        # BAD: Exposes secret path structure
        # raise SecretResolutionError(f"Secret not found at {secret_path}") from e

        # GOOD: Generic message without path details
        raise SecretResolutionError(
            "Database credentials not found in Infisical",
            context=context,
            # Only include sanitized path if absolutely necessary
            secret_type="database_credentials",
        ) from e
    except PermissionDeniedError as e:
        # BAD: Reveals internal path structure
        # raise SecretResolutionError(f"No access to {secret_path}") from e

        # GOOD: Generic permission error
        raise SecretResolutionError(
            "Permission denied accessing database credentials",
            context=context,
        ) from e
```

### InfraAuthenticationError Sanitization

```python
from omnibase_infra.errors import InfraAuthenticationError, ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType


async def authenticate_service(
    token: str,  # Sensitive - never include in errors
    correlation_id: UUID,
) -> AuthResult:
    """Authenticate with external service using sanitized error reporting."""
    context = ModelInfraErrorContext(
        transport_type=EnumInfraTransportType.HTTP,
        operation="authenticate",
        target_name="oauth-provider",
        correlation_id=correlation_id,
    )

    try:
        return await oauth_client.validate_token(token)
    except TokenExpiredError as e:
        # BAD: Token appears in error message
        # raise InfraAuthenticationError(f"Token expired: {token[:10]}...") from e

        # GOOD: No token content in error
        raise InfraAuthenticationError(
            "Authentication token has expired - refresh required",
            context=context,
            token_type="bearer",
            # Include expiry time if available and not sensitive
            expired_at=e.expiry_time.isoformat() if hasattr(e, 'expiry_time') else None,
        ) from e
    except InvalidTokenError as e:
        # BAD: Reveals token validation details
        # raise InfraAuthenticationError(f"Token invalid: {e.validation_error}") from e

        # GOOD: Generic authentication failure
        raise InfraAuthenticationError(
            "Authentication failed - token validation error",
            context=context,
        ) from e
```

## Logging Integration

### Structured Logging with Sanitization

```python
import logging
import json

logger = logging.getLogger(__name__)


class SanitizedLogFormatter(logging.Formatter):
    """Log formatter that sanitizes sensitive fields."""

    def format(self, record: logging.LogRecord) -> str:
        # Sanitize extra fields if present
        if hasattr(record, "extra"):
            record.extra = sanitize_dict(record.extra)

        return super().format(record)


def log_error_safely(
    error: Exception,
    context: ModelInfraErrorContext | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    """Log error with automatic sanitization of sensitive data.

    Args:
        error: Exception to log
        context: ONEX error context (already sanitized by design)
        extra: Additional context (will be sanitized)
    """
    log_data = {
        "error_type": type(error).__name__,
        "error_message": str(error),  # Should already be sanitized
    }

    if context:
        log_data["correlation_id"] = str(context.correlation_id)
        log_data["operation"] = context.operation
        log_data["target_name"] = context.target_name
        log_data["transport_type"] = context.transport_type.value

    if extra:
        # Sanitize extra data before logging
        log_data["extra"] = sanitize_dict(extra)

    logger.error("Operation failed", extra=log_data)
```

## Best Practices Summary

### DO

- Use `ModelInfraErrorContext` for structured, sanitized error context
- Include correlation IDs for request tracing
- Use sanitized service names (e.g., "postgresql-primary")
- Include operation names for debugging
- Use error codes instead of raw error messages
- Include retry counts and timeout values
- Sanitize connection strings before including in errors
- Use path sanitization for file-related errors
- Chain exceptions to preserve stack traces (while keeping messages clean)

### DON'T

- Include passwords, API keys, or tokens in error messages
- Log full connection strings with credentials
- Include PII (names, emails, SSNs, phone numbers)
- Expose internal file paths or system structure
- Include raw request/response bodies without sanitization
- Log environment variable values
- Include private keys or certificates
- Expose database schema details in production errors

### Quick Reference Table

| Data Type | Include in Error? | Alternative |
|-----------|-------------------|-------------|
| Password | NEVER | "Authentication failed" |
| API Key | NEVER | "API authentication required" |
| Token | NEVER | "Token expired/invalid" |
| Email | NEVER | User ID or "user not found" |
| SSN | NEVER | "Invalid identifier format" |
| Connection String | SANITIZE | Remove credentials, keep host/port |
| Internal IP | AVOID | Use service names |
| File Path | SANITIZE | Use relative or generic paths |
| Correlation ID | ALWAYS | Full UUID for tracing |
| Service Name | ALWAYS | Sanitized identifier |
| Operation Name | ALWAYS | Helps with debugging |
| Error Code | ALWAYS | Structured classification |
| Retry Count | SAFE | Debugging context |
| Timeout Value | SAFE | Debugging context |

## Related Patterns

- [Security Patterns](./security_patterns.md) - Comprehensive security guide including input validation, authentication, secret management, and network security
- [Error Handling Patterns](./error_handling_patterns.md) - Error classification and context usage
- [Error Recovery Patterns](./error_recovery_patterns.md) - Retry logic, circuit breakers
- [Correlation ID Tracking](./correlation_id_tracking.md) - Request tracing across services
- [Container Dependency Injection](./container_dependency_injection.md) - Service resolution patterns
