> **Navigation**: [Home](../index.md) > [Patterns](README.md) > Security Patterns

# Security Patterns

## Overview

This document provides comprehensive security patterns for ONEX infrastructure development. Security is not an afterthought but a fundamental design principle embedded throughout the ONEX architecture.

**Core Security Principles**:
1. **Defense in Depth** - Multiple layers of security controls
2. **Least Privilege** - Minimum necessary access for all components
3. **Fail Secure** - Errors must not compromise security
4. **Secure by Default** - Safe configurations out of the box
5. **Zero Trust** - Verify explicitly, assume breach

## Table of Contents

- [Error Sanitization](#error-sanitization)
- [Input Validation](#input-validation)
- [Authentication and Authorization](#authentication-and-authorization)
- [Secret Management](#secret-management)
  - [YAML Contract Security](#yaml-contract-security-no-secrets-in-contracts)
- [Network Security](#network-security)
- [Policy Security](#policy-security)
- [Introspection Security](#introspection-security)
- [Logging and Audit](#logging-and-audit)
- [Production Security Checklist](#production-security-checklist)

---

## Error Sanitization

### Security Requirements

Error messages and contexts can inadvertently expose sensitive information. All infrastructure errors must be sanitized before propagation.

#### NEVER Include in Error Messages

| Category | Examples | Risk |
|----------|----------|------|
| Credentials | Passwords, API keys, tokens, secrets | Credential theft |
| Connection Strings | `postgresql://user:pass@host` | Full system access |
| PII | Names, emails, SSNs, phone numbers | Privacy violation, GDPR/CCPA |
| Internal Paths | `/home/user/.ssh/id_rsa`, `/var/secrets/` | Path traversal, information disclosure |
| Private Keys | SSL certificates, encryption keys | Cryptographic compromise |
| Session Data | Cookies, JWT tokens, session IDs | Session hijacking |
| Internal IPs | `192.168.x.x`, `10.x.x.x` in production logs | Network reconnaissance |

#### SAFE to Include

| Category | Examples | Rationale |
|----------|----------|-----------|
| Service Names | `postgresql`, `kafka`, `infisical` | Generic identifiers |
| Operation Names | `connect`, `query`, `authenticate` | No sensitive data |
| Correlation IDs | UUID format | Request tracing, no PII |
| Error Codes | `DATABASE_CONNECTION_ERROR` | Standardized categories |
| Sanitized Hostnames | `db.example.com` | Public DNS, no credentials |
| Port Numbers | `5432`, `9092` | Non-sensitive configuration |
| Retry Counts | `3`, `5` | Operational metadata |
| Timeout Values | `30s`, `60s` | Configuration metadata |

### Sanitization Patterns

```python
from omnibase_infra.errors import InfraConnectionError, ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType
from uuid import uuid4

# UNSAFE - Exposes connection string with credentials
def bad_example(connection_string: str) -> None:
    try:
        connect(connection_string)
    except Exception as e:
        # CRITICAL VULNERABILITY - Credentials in error message
        raise InfraConnectionError(
            f"Failed to connect: {connection_string}"  # NEVER DO THIS
        ) from e


# SAFE - Sanitized error with proper context
def good_example(host: str, port: int, database: str) -> None:
    correlation_id = uuid4()
    context = ModelInfraErrorContext(
        transport_type=EnumInfraTransportType.DATABASE,
        operation="connect",
        target_name=f"{database}@{host}",  # No credentials
        correlation_id=correlation_id,
    )

    try:
        connect(host, port, database)
    except ConnectionError as e:
        raise InfraConnectionError(
            "Database connection failed",
            context=context,
            host=host,          # Hostname only
            port=port,          # Port only
            database=database,  # Database name only
            # NO: username, password, connection_string
        ) from e
```

### URL and Connection String Sanitization

```python
from urllib.parse import urlparse
from dataclasses import dataclass


@dataclass
class SanitizedConnectionInfo:
    """Connection information with credentials removed."""

    host: str
    port: int | None
    database: str | None
    scheme: str

    @classmethod
    def from_connection_string(cls, conn_str: str) -> "SanitizedConnectionInfo":
        """Parse connection string and extract safe components only.

        Args:
            conn_str: Full connection string (may contain credentials)

        Returns:
            SanitizedConnectionInfo with credentials stripped
        """
        parsed = urlparse(conn_str)
        return cls(
            host=parsed.hostname or "unknown",
            port=parsed.port,
            database=parsed.path.lstrip("/") or None,
            scheme=parsed.scheme,
            # NOTE: parsed.username and parsed.password are intentionally NOT included
        )


# Usage
conn_str = "postgresql://admin:secret123@db.internal:5432/mydb"
safe_info = SanitizedConnectionInfo.from_connection_string(conn_str)
# safe_info.host = "db.internal"
# safe_info.port = 5432
# safe_info.database = "mydb"
# safe_info.scheme = "postgresql"
# Credentials are NOT accessible
```

### Exception Wrapping with Sanitization

```python
from omnibase_infra.errors import InfraAuthenticationError, ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType


def wrap_auth_error(original: Exception, correlation_id: UUID) -> InfraAuthenticationError:
    """Wrap external auth error with sanitized context.

    IMPORTANT: The original exception message may contain sensitive data.
    Extract only safe metadata.
    """
    context = ModelInfraErrorContext(
        transport_type=EnumInfraTransportType.INFISICAL,
        operation="authenticate",
        target_name="infisical-primary",
        correlation_id=correlation_id,
    )

    # SAFE: Generic message, no original exception details
    return InfraAuthenticationError(
        "Authentication failed",  # Generic, no details from original
        context=context,
    )
```

**Related**: [Error Handling Patterns](./error_handling_patterns.md#error-sanitization)

---

## Input Validation

### Validation at System Boundaries

All data entering the system must be validated at the boundary before processing. This prevents injection attacks, data corruption, and unexpected behavior.

### Validation Layers

```
External Input → API Gateway → Service Boundary → Domain Logic
                     ↓               ↓                ↓
                 Schema         Type Checking     Business Rules
                Validation       (Pydantic)       Validation
```

### Pydantic Model Validation

ONEX uses Pydantic for all data models, providing automatic validation:

```python
from pydantic import BaseModel, Field, field_validator
from typing import Annotated
import re


class ModelUserInput(BaseModel):
    """User input with validation constraints.

    All user-facing inputs must use Pydantic models with explicit
    validation rules.
    """

    # Length constraints prevent buffer overflow and storage issues
    username: Annotated[str, Field(min_length=3, max_length=64)]

    # Email pattern validation
    email: str

    # Numeric bounds prevent integer overflow
    age: Annotated[int, Field(ge=0, le=150)]

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Validate email format.

        Note: For production, use email-validator library.
        """
        pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        if not re.match(pattern, v):
            raise ValueError("Invalid email format")
        return v.lower()  # Normalize to lowercase

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Validate username contains only safe characters.

        Prevents injection attacks via username field.
        """
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "Username can only contain letters, numbers, underscores, and hyphens"
            )
        return v
```

### SQL Injection Prevention

```python
from uuid import UUID


async def execute_query_safely(
    db_connection: object,
    user_id: UUID,
    correlation_id: UUID,
) -> dict[str, object]:
    """Execute parameterized query to prevent SQL injection.

    CRITICAL: Never use string formatting for SQL queries.
    Always use parameterized queries with proper escaping.
    """
    # SAFE - Parameterized query
    query = "SELECT * FROM users WHERE id = $1"
    result = await db_connection.fetchrow(query, user_id)

    # UNSAFE - String formatting (NEVER DO THIS)
    # query = f"SELECT * FROM users WHERE id = '{user_id}'"  # SQL INJECTION RISK

    return dict(result) if result else {}


async def execute_dynamic_query_safely(
    db_connection: object,
    table_name: str,
    allowed_tables: frozenset[str],
    correlation_id: UUID,
) -> list[dict[str, object]]:
    """Execute query with validated table name.

    For dynamic table names, use allowlist validation.
    """
    # Validate table name against allowlist
    if table_name not in allowed_tables:
        raise ValueError(f"Invalid table: {table_name}")

    # Safe because table_name is from allowlist, not user input
    query = f"SELECT * FROM {table_name} LIMIT 100"
    results = await db_connection.fetch(query)

    return [dict(r) for r in results]
```

### Path Traversal Prevention

```python
from pathlib import Path
from typing import Final

# Allowlist of safe base directories
ALLOWED_UPLOAD_DIRS: Final[frozenset[Path]] = frozenset({
    Path("/var/uploads"),
    Path("/tmp/processing"),
})


def resolve_safe_path(base_dir: Path, user_path: str) -> Path:
    """Resolve user-provided path safely within base directory.

    Prevents path traversal attacks (e.g., ../../../etc/passwd).

    Args:
        base_dir: Allowed base directory (must be in ALLOWED_UPLOAD_DIRS)
        user_path: User-provided relative path

    Returns:
        Resolved absolute path within base_dir

    Raises:
        ValueError: If path escapes base_dir or base_dir not allowed
    """
    if base_dir not in ALLOWED_UPLOAD_DIRS:
        raise ValueError(f"Base directory not in allowlist: {base_dir}")

    # Resolve to absolute path
    resolved = (base_dir / user_path).resolve()

    # Verify resolved path is still within base_dir
    if not resolved.is_relative_to(base_dir):
        raise ValueError(
            f"Path traversal detected: {user_path} escapes {base_dir}"
        )

    return resolved


# Usage
base = Path("/var/uploads")
safe_path = resolve_safe_path(base, "images/photo.jpg")  # OK
# resolve_safe_path(base, "../etc/passwd")  # Raises ValueError
```

### Command Injection Prevention

```python
import subprocess
import shlex
from typing import Final

# Allowlist of executable commands
ALLOWED_COMMANDS: Final[frozenset[str]] = frozenset({
    "git", "docker", "kubectl"
})


def execute_command_safely(
    command: str,
    args: list[str],
) -> subprocess.CompletedProcess:
    """Execute external command with validation.

    CRITICAL: Never use shell=True with user input.
    Always use allowlisted commands and explicit argument lists.
    """
    # Validate command against allowlist
    if command not in ALLOWED_COMMANDS:
        raise ValueError(f"Command not allowed: {command}")

    # Validate arguments don't contain shell metacharacters
    for arg in args:
        if not arg or any(c in arg for c in ";|&$`"):
            raise ValueError(f"Invalid argument: {arg}")

    # SAFE - No shell, explicit argument list
    return subprocess.run(
        [command, *args],
        shell=False,  # CRITICAL: Never shell=True with user input
        capture_output=True,
        text=True,
        timeout=30,
    )


# UNSAFE patterns to avoid:
# subprocess.run(f"git clone {url}", shell=True)  # COMMAND INJECTION
# os.system(user_command)  # COMMAND INJECTION
```

---

## Authentication and Authorization

### Authentication Patterns

#### Token-Based Authentication

```python
from datetime import datetime, timedelta
from typing import Protocol
from uuid import UUID

from pydantic import BaseModel


class ModelAuthToken(BaseModel):
    """Authentication token model."""

    token_id: UUID
    user_id: UUID
    issued_at: datetime
    expires_at: datetime
    scopes: frozenset[str]

    @property
    def is_expired(self) -> bool:
        """Check if token is expired."""
        return datetime.utcnow() >= self.expires_at

    def has_scope(self, required_scope: str) -> bool:
        """Check if token has required scope."""
        return required_scope in self.scopes


class ProtocolAuthenticator(Protocol):
    """Authentication protocol for ONEX services."""

    async def validate_token(self, token: str) -> ModelAuthToken | None:
        """Validate token and return claims if valid.

        Returns:
            ModelAuthToken if valid, None if invalid/expired
        """
        ...

    async def refresh_token(self, token: ModelAuthToken) -> ModelAuthToken:
        """Refresh token before expiry.

        Raises:
            InfraAuthenticationError: If refresh fails
        """
        ...
```

#### Service-to-Service Authentication

```python
from omnibase_infra.errors import InfraAuthenticationError, ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType


class ServiceAuthenticator:
    """Mutual TLS and token-based service authentication."""

    def __init__(
        self,
        service_id: str,
        infisical_client: object,
    ) -> None:
        self._service_id = service_id
        self._infisical = infisical_client

    async def get_service_token(
        self,
        target_service: str,
        correlation_id: UUID,
    ) -> str:
        """Get service token for calling another service.

        Uses Infisical machine identity authentication for service identity.

        Args:
            target_service: Service to authenticate to
            correlation_id: Request tracking ID

        Returns:
            JWT token for service authentication

        Raises:
            InfraAuthenticationError: If token generation fails
        """
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.INFISICAL,
            operation="get_service_token",
            target_name="infisical-primary",
            correlation_id=correlation_id,
        )

        try:
            response = await self._infisical.auth.universal_auth.login(
                client_id=self._service_id,
                client_secret=await self._get_client_secret(),
            )
            return response["access_token"]
        except Exception as e:
            raise InfraAuthenticationError(
                f"Failed to authenticate service {self._service_id}",
                context=context,
            ) from e

    async def _get_client_secret(self) -> str:
        """Retrieve client secret from secure storage.

        IMPORTANT: Client secret should be injected via environment
        or Kubernetes secret, not stored in code.
        """
        import os
        client_secret = os.environ.get("INFISICAL_CLIENT_SECRET")
        if not client_secret:
            raise ValueError("INFISICAL_CLIENT_SECRET not configured")
        return client_secret
```

### Authorization Patterns

#### Role-Based Access Control (RBAC)

```python
from enum import Enum
from typing import Final

from pydantic import BaseModel


class EnumRole(str, Enum):
    """System roles with hierarchical permissions."""

    ADMIN = "admin"
    OPERATOR = "operator"
    DEVELOPER = "developer"
    VIEWER = "viewer"


class EnumPermission(str, Enum):
    """Granular permissions for operations."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"


# Role to permission mapping
ROLE_PERMISSIONS: Final[dict[EnumRole, frozenset[EnumPermission]]] = {
    EnumRole.ADMIN: frozenset({
        EnumPermission.READ,
        EnumPermission.WRITE,
        EnumPermission.DELETE,
        EnumPermission.ADMIN,
    }),
    EnumRole.OPERATOR: frozenset({
        EnumPermission.READ,
        EnumPermission.WRITE,
    }),
    EnumRole.DEVELOPER: frozenset({
        EnumPermission.READ,
        EnumPermission.WRITE,
    }),
    EnumRole.VIEWER: frozenset({
        EnumPermission.READ,
    }),
}


class AuthorizationContext(BaseModel):
    """Context for authorization decisions.

    All authorization decisions include a correlation_id for traceability
    and integration with ONEX observability patterns.
    """

    user_id: UUID
    roles: frozenset[EnumRole]
    resource_id: str
    action: EnumPermission
    correlation_id: UUID  # Required for traceability per ONEX error patterns

    def is_authorized(self) -> bool:
        """Check if action is authorized for user's roles."""
        for role in self.roles:
            if self.action in ROLE_PERMISSIONS.get(role, frozenset()):
                return True
        return False


def require_permission(permission: EnumPermission):
    """Decorator for permission-protected operations.

    Usage:
        @require_permission(EnumPermission.WRITE)
        async def create_resource(ctx: AuthorizationContext, data: dict):
            ...
    """
    def decorator(func):
        async def wrapper(ctx: AuthorizationContext, *args, **kwargs):
            if permission not in get_user_permissions(ctx.roles):
                from omnibase_infra.errors import InfraAuthenticationError, ModelInfraErrorContext
                from omnibase_infra.enums import EnumInfraTransportType

                context = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.HTTP,
                    operation="authorization_check",
                    target_name="permission_guard",
                    correlation_id=ctx.correlation_id,  # Required field, always present
                )
                raise InfraAuthenticationError(
                    f"Authorization denied: permission {permission.value} required",
                    context=context,
                )
            return await func(ctx, *args, **kwargs)
        return wrapper
    return decorator


def get_user_permissions(roles: frozenset[EnumRole]) -> frozenset[EnumPermission]:
    """Aggregate permissions from all roles."""
    permissions: set[EnumPermission] = set()
    for role in roles:
        permissions.update(ROLE_PERMISSIONS.get(role, frozenset()))
    return frozenset(permissions)
```

---

## Secret Management

**See Also**: [Secret Resolver Pattern](./secret_resolver.md) - Centralized secret resolution with caching, multiple sources, and migration guide.

### Infisical Integration

ONEX uses Infisical for centralized secret management (migrated from HashiCorp Vault in OMN-2288).

#### Secret Retrieval Pattern

```python
from omnibase_infra.errors import (
    SecretResolutionError,
    InfraAuthenticationError,
    ModelInfraErrorContext,
)
from omnibase_infra.enums import EnumInfraTransportType
from uuid import UUID


class InfisicalSecretProvider:
    """Secure secret retrieval from Infisical.

    Features:
        - Machine identity authentication
        - Secret caching with TTL
        - Error sanitization
    """

    def __init__(self, infisical_client: object) -> None:
        self._infisical = infisical_client
        self._cache: dict[str, tuple[dict, float]] = {}
        self._cache_ttl = 300.0  # 5 minutes

    async def get_secret(
        self,
        path: str,
        correlation_id: UUID,
    ) -> str:
        """Retrieve secret from Infisical.

        Args:
            path: Secret path (e.g., "/shared/database/POSTGRES_PASSWORD")
            correlation_id: Request tracking ID

        Returns:
            Secret value as string

        Raises:
            SecretResolutionError: Secret not found or access denied
            InfraAuthenticationError: Infisical authentication failed

        SECURITY NOTE:
            - Never log the secret values
            - Never include path in error messages (may reveal structure)
            - Use generic error messages
        """
        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.INFISICAL,
            operation="get_secret",
            target_name="infisical-primary",
            correlation_id=correlation_id,
        )

        try:
            response = await self._infisical.get_secret(path=path)
            return response.secret_value

        except SecretNotFoundError:
            # SAFE: Generic message, no path details
            raise SecretResolutionError(
                "Secret not found",
                context=context,
            )

        except UnauthorizedError:
            # SAFE: Generic message, no auth details
            raise InfraAuthenticationError(
                "Infisical access denied",
                context=context,
            )

        except Exception as e:
            # SAFE: Generic message for unexpected errors
            raise SecretResolutionError(
                "Failed to retrieve secret",
                context=context,
            ) from e
```

#### Secret Rotation

Infisical supports secret rotation via the web UI and API. Secrets are versioned and the runtime automatically fetches the latest value on cache expiry (default 5 minutes for `infisical` source type). No application-side rotation logic is required.

```python
from omnibase_infra.runtime.secret_resolver import SecretResolver

# Cache expires automatically; on expiry, latest value is fetched from Infisical
resolver = SecretResolver(config=config, infisical_handler=infisical_handler)

# After rotation in Infisical UI/API, the resolver will pick up the new
# value on next cache miss (within 5 minutes by default)
db_password = await resolver.get_secret_async("database.postgres.password")
```

### YAML Contract Security: No Secrets in Contracts

**CRITICAL**: ONEX handler contracts (YAML files) must NEVER contain secrets.

Handler contracts are:
- Version-controlled in Git (secrets would be in repository history forever)
- Potentially logged during deployment or debugging
- Shared across environments (dev/staging/prod)
- Readable by anyone with repository access

#### Prohibited Content in Contracts

```yaml
# NEVER include these in handler_contract.yaml or contract.yaml:
config:
  password: "secret123"           # VIOLATION
  api_key: "sk-abc123"            # VIOLATION
  connection_string: "...pass..." # VIOLATION
  private_key: "-----BEGIN..."    # VIOLATION
  token: "eyJhbGc..."             # VIOLATION
```

#### Where to Store Secrets Instead

| Secret Type | Recommended Storage | Access Pattern |
|-------------|-------------------|----------------|
| Database credentials | Infisical (`/shared/database/`) | `await resolver.get_secret_async("database.postgres.password")` |
| API keys | Infisical (`/shared/llm/`) | `await resolver.get_secret_async("llm.openai.api_key")` |
| Service tokens | Infisical machine identity | `await infisical_handler.authenticate()` |
| Static secrets | Kubernetes secrets | Environment variable injection |

**See Also**: [Handler Plugin Loader - Contract Content Security](./handler_plugin_loader.md#contract-content-security-no-secrets-in-contracts) for detailed contract security guidelines.

---

### Environment Variable Security

```python
import os
from typing import Final


# Required secret environment variables
REQUIRED_SECRETS: Final[frozenset[str]] = frozenset({
    "INFISICAL_CLIENT_ID",
    "INFISICAL_CLIENT_SECRET",
    "KAFKA_SASL_PASSWORD",
    "DATABASE_PASSWORD",
})


def validate_secret_environment() -> None:
    """Validate required secrets are configured.

    Call during application startup to fail fast if secrets missing.

    SECURITY NOTES:
        - Never log secret values
        - Never include secret values in error messages
        - Use generic "not configured" messages
    """
    missing = []
    for secret in REQUIRED_SECRETS:
        if not os.environ.get(secret):
            missing.append(secret)

    if missing:
        # SAFE: Only report variable names, not values
        raise EnvironmentError(
            f"Required secrets not configured: {', '.join(missing)}"
        )


def get_secret_from_env(name: str) -> str:
    """Get secret from environment variable.

    Args:
        name: Environment variable name

    Returns:
        Secret value

    Raises:
        ValueError: If secret not configured
    """
    value = os.environ.get(name)
    if not value:
        # SAFE: Only report variable name
        raise ValueError(f"Secret not configured: {name}")
    return value
```

---

## Network Security

### TLS/SSL Configuration

```python
import ssl
from pathlib import Path
from typing import Final


# Minimum TLS version for all connections
MIN_TLS_VERSION: Final[ssl.TLSVersion] = ssl.TLSVersion.TLSv1_3


def create_secure_ssl_context(
    cert_file: Path | None = None,
    key_file: Path | None = None,
    ca_file: Path | None = None,
    verify_mode: ssl.VerifyMode = ssl.CERT_REQUIRED,
) -> ssl.SSLContext:
    """Create secure SSL context for connections.

    SECURITY REQUIREMENTS:
        - TLS 1.3 minimum (TLS 1.2 for legacy compatibility only)
        - Certificate verification enabled by default
        - Strong cipher suites only

    Args:
        cert_file: Path to client certificate (for mTLS)
        key_file: Path to client private key (for mTLS)
        ca_file: Path to CA certificate for verification
        verify_mode: Certificate verification mode

    Returns:
        Configured SSLContext
    """
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = MIN_TLS_VERSION
    context.verify_mode = verify_mode

    # Load CA certificates
    if ca_file:
        context.load_verify_locations(ca_file)
    else:
        context.load_default_certs()

    # Load client certificate for mTLS
    if cert_file and key_file:
        context.load_cert_chain(cert_file, key_file)

    # Disable weak ciphers
    context.set_ciphers("ECDHE+AESGCM:DHE+AESGCM:ECDHE+CHACHA20:DHE+CHACHA20")

    return context
```

### Kafka Security Configuration

```python
from pydantic import BaseModel, SecretStr
from typing import Literal


class ModelKafkaSecurityConfig(BaseModel):
    """Kafka security configuration.

    Supports SASL_SSL for authentication and encryption.
    """

    security_protocol: Literal["SASL_SSL", "SSL", "PLAINTEXT"] = "SASL_SSL"
    sasl_mechanism: Literal["SCRAM-SHA-512", "SCRAM-SHA-256", "PLAIN"] = "SCRAM-SHA-512"
    sasl_username: str
    sasl_password: SecretStr  # Pydantic SecretStr prevents accidental logging
    ssl_cafile: Path | None = None
    ssl_certfile: Path | None = None
    ssl_keyfile: Path | None = None

    def get_producer_config(self) -> dict[str, str | int]:
        """Get Kafka producer configuration.

        Returns:
            Configuration dictionary for KafkaProducer
        """
        config = {
            "security_protocol": self.security_protocol,
            "sasl_mechanism": self.sasl_mechanism,
            "sasl_plain_username": self.sasl_username,
            "sasl_plain_password": self.sasl_password.get_secret_value(),
        }

        if self.ssl_cafile:
            config["ssl_cafile"] = str(self.ssl_cafile)
        if self.ssl_certfile:
            config["ssl_certfile"] = str(self.ssl_certfile)
        if self.ssl_keyfile:
            config["ssl_keyfile"] = str(self.ssl_keyfile)

        return config


# Production configuration example
KAFKA_SECURITY_CONFIG = ModelKafkaSecurityConfig(
    security_protocol="SASL_SSL",
    sasl_mechanism="SCRAM-SHA-512",
    sasl_username="service-user",
    sasl_password=SecretStr(get_secret_from_env("KAFKA_SASL_PASSWORD")),
    ssl_cafile=Path("/etc/kafka/ca.crt"),
)
```

### Database Connection Security

```python
import ssl
from pathlib import Path

from pydantic import BaseModel, SecretStr


class ModelDatabaseSecurityConfig(BaseModel):
    """Database connection security configuration."""

    host: str
    port: int = 5432
    database: str
    username: str
    password: SecretStr
    ssl_mode: Literal["require", "verify-ca", "verify-full"] = "verify-full"
    ssl_ca_cert: Path | None = None
    ssl_client_cert: Path | None = None
    ssl_client_key: Path | None = None

    def get_connection_params(self) -> dict[str, str | ssl.SSLContext]:
        """Get connection parameters for asyncpg/psycopg2.

        Returns:
            Connection parameters with SSL configuration
        """
        params = {
            "host": self.host,
            "port": str(self.port),
            "database": self.database,
            "user": self.username,
            "password": self.password.get_secret_value(),
        }

        if self.ssl_mode == "verify-full":
            ssl_context = create_secure_ssl_context(
                cert_file=self.ssl_client_cert,
                key_file=self.ssl_client_key,
                ca_file=self.ssl_ca_cert,
            )
            params["ssl"] = ssl_context

        return params
```

### Network Segmentation

```python
from typing import Final
from ipaddress import IPv4Network


# Define network zones for security segmentation
class NetworkZones:
    """Network zone definitions for security policies.

    All production services should be deployed in appropriate zones
    with network policies enforcing zone boundaries.
    """

    # Management zone - admin access only
    MANAGEMENT: Final[IPv4Network] = IPv4Network("10.0.0.0/24")

    # Application zone - service-to-service communication
    APPLICATION: Final[IPv4Network] = IPv4Network("10.0.1.0/24")

    # Data zone - database and storage access
    DATA: Final[IPv4Network] = IPv4Network("10.0.2.0/24")

    # DMZ - external-facing services
    DMZ: Final[IPv4Network] = IPv4Network("10.0.3.0/24")


# Zone communication policy
ZONE_POLICY: Final[dict[str, frozenset[str]]] = {
    "DMZ": frozenset({"APPLICATION"}),           # DMZ can call APPLICATION
    "APPLICATION": frozenset({"DATA", "APPLICATION"}),  # APP can call DATA and other APP
    "DATA": frozenset(),                         # DATA zone is isolated
    "MANAGEMENT": frozenset({"APPLICATION", "DATA", "DMZ"}),  # MGMT can call all
}
```

---

## Policy Security

Policy registration and execution require careful security considerations as policies can execute arbitrary code.

### Trust Model

**CRITICAL**: Policies MUST come from trusted sources only.

```python
from typing import Final
from uuid import uuid4

from omnibase_infra.errors import ProtocolConfigurationError, ModelInfraErrorContext
from omnibase_infra.enums import EnumInfraTransportType


# Allowlisted policy packages
TRUSTED_POLICY_SOURCES: Final[frozenset[str]] = frozenset({
    "omnibase_policies",      # First-party policies
    "myapp.policies",         # Internal application policies
    # Add approved third-party packages after security review
})


class PolicySecurityError(ProtocolConfigurationError):
    """Raised when a policy fails security validation.

    Extends ProtocolConfigurationError to integrate with ONEX error handling.

    Error Codes:
        POLICY_SECURITY_001: Policy from untrusted source/namespace
        POLICY_SECURITY_002: Policy code hash not in approved list
        POLICY_SECURITY_003: Policy failed static analysis
    """

    def __init__(
        self,
        message: str,
        error_code: str,
        policy_class_name: str | None = None,
        context: ModelInfraErrorContext | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize PolicySecurityError with required error code.

        Args:
            message: Human-readable error message
            error_code: Error code for categorization (e.g., "POLICY_SECURITY_001")
            policy_class_name: Name of the policy class that failed validation
            context: Optional infrastructure error context
            **kwargs: Additional context fields
        """
        super().__init__(message, context=context, **kwargs)
        self.error_code = error_code
        self.policy_class_name = policy_class_name


def validate_policy_source(
    policy_class: type,
    correlation_id: UUID | None = None,
) -> bool:
    """Validate policy comes from trusted source.

    Args:
        policy_class: Policy class to validate
        correlation_id: Optional correlation ID for tracing

    Returns:
        True if from trusted source

    Raises:
        PolicySecurityError: If from untrusted source (POLICY_SECURITY_001)
    """
    module = policy_class.__module__

    for trusted in TRUSTED_POLICY_SOURCES:
        if module.startswith(trusted):
            return True

    context = ModelInfraErrorContext(
        transport_type=EnumInfraTransportType.RUNTIME,
        operation="validate_policy_source",
        correlation_id=correlation_id or uuid4(),
    )
    raise PolicySecurityError(
        f"Policy from untrusted source namespace: {module}",
        error_code="POLICY_SECURITY_001",
        policy_class_name=policy_class.__name__,
        context=context,
    )
```

### Policy Execution Sandboxing

```python
import asyncio
import resource
from typing import TypeVar
from contextlib import contextmanager

T = TypeVar("T")


@contextmanager
def resource_limits(
    max_memory_mb: int = 100,
    max_cpu_seconds: float = 1.0,
):
    """Apply resource limits for policy execution.

    SECURITY: Prevents resource exhaustion attacks from policies.

    Args:
        max_memory_mb: Maximum memory in MB
        max_cpu_seconds: Maximum CPU time in seconds
    """
    # Store original limits
    orig_mem = resource.getrlimit(resource.RLIMIT_AS)
    orig_cpu = resource.getrlimit(resource.RLIMIT_CPU)

    try:
        # Set memory limit
        resource.setrlimit(
            resource.RLIMIT_AS,
            (max_memory_mb * 1024 * 1024, orig_mem[1]),
        )

        # Set CPU limit
        resource.setrlimit(
            resource.RLIMIT_CPU,
            (int(max_cpu_seconds), int(max_cpu_seconds) + 1),
        )

        yield

    finally:
        # Restore original limits
        resource.setrlimit(resource.RLIMIT_AS, orig_mem)
        resource.setrlimit(resource.RLIMIT_CPU, orig_cpu)


async def execute_policy_safely(
    policy: object,
    context: dict[str, object],
    timeout_seconds: float = 1.0,
    max_memory_mb: int = 100,
) -> object:
    """Execute policy with timeout and resource limits.

    Args:
        policy: Policy instance
        context: Evaluation context
        timeout_seconds: Maximum execution time
        max_memory_mb: Maximum memory allocation

    Returns:
        Policy result

    Raises:
        asyncio.TimeoutError: Policy exceeded timeout
        MemoryError: Policy exceeded memory limit
    """
    async def _execute():
        with resource_limits(max_memory_mb=max_memory_mb):
            return policy.evaluate(context)

    return await asyncio.wait_for(_execute(), timeout=timeout_seconds)
```

**Related**: [Policy Registry Trust Model](./policy_registry_trust_model.md)

---

## Introspection Security

Node introspection uses reflection to discover capabilities. This has security implications.

### Threat Model

| Threat | Risk | Mitigation |
|--------|------|------------|
| Reconnaissance | Attacker learns available operations | Filter sensitive method names |
| Version fingerprinting | Identify vulnerable versions | Keep software updated |
| Architecture mapping | Understand system topology | Network segmentation |
| State inference | Deduce system health/status | Limit FSM state exposure |

### Exposure Control

```python
from typing import Final

# Prefixes to exclude from capability discovery
EXCLUDED_METHOD_PREFIXES: Final[tuple[str, ...]] = (
    "_",           # Private methods
    "get_",        # Getters
    "set_",        # Setters
    "initialize",  # Lifecycle
    "start_",      # Lifecycle
    "stop_",       # Lifecycle
    "shutdown",    # Lifecycle
    # Add additional prefixes for sensitive operations
    "decrypt_",    # Cryptographic operations
    "auth_",       # Authentication internals
    "admin_",      # Administrative operations
)


# Keywords that indicate operational methods (to include)
OPERATION_KEYWORDS: Final[tuple[str, ...]] = (
    "execute",
    "handle",
    "process",
    "run",
    "invoke",
    "call",
)


def filter_capabilities(methods: list[str]) -> list[str]:
    """Filter method list to safe capabilities.

    Args:
        methods: All public method names

    Returns:
        Filtered list of safe capability names
    """
    capabilities = []

    for method in methods:
        # Skip excluded prefixes
        if any(method.startswith(prefix) for prefix in EXCLUDED_METHOD_PREFIXES):
            continue

        # Only include methods with operation keywords
        if any(keyword in method for keyword in OPERATION_KEYWORDS):
            capabilities.append(method)

    return capabilities
```

### Network Security for Introspection

```python
# Kafka topic ACLs for introspection events
INTROSPECTION_TOPIC_ACLS = """
# Allow service accounts to publish introspection events
kafka-acls.sh --add --allow-principal User:service-account \\
    --operation Write --topic node.introspection

# Allow registry to consume introspection events
kafka-acls.sh --add --allow-principal User:registry-service \\
    --operation Read --topic node.introspection

# DENY external access to introspection topics
kafka-acls.sh --add --deny-principal User:* \\
    --operation Read --topic node.introspection \\
    --resource-pattern-type prefixed
"""
```

**Related**: [Handler Plugin Loader](./handler_plugin_loader.md#security-considerations)

### Structured Security Error Reporting

As of OMN-1091, ONEX uses structured error models for security validation failures. This enables consistent error tracking, remediation guidance, and CI integration.

#### Using ModelHandlerValidationError for Security Violations

```python
from omnibase_infra.enums import EnumHandlerType
from omnibase_infra.models.errors import ModelHandlerValidationError
from omnibase_infra.models.handlers import ModelHandlerIdentifier
from omnibase_infra.validation.security_validator import (
    SecurityRuleId,
    validate_method_exposure,
    validate_handler_security,
)


# Example: Validate method exposure during introspection
def validate_node_security(
    node_path: str,
    method_names: list[str],
    method_signatures: dict[str, str],
) -> list[ModelHandlerValidationError]:
    """Validate node for security concerns.

    Returns list of structured validation errors with remediation hints.
    """
    handler_identity = ModelHandlerIdentifier.from_node(
        node_path=node_path,
        handler_type=EnumHandlerType.INFRA_HANDLER,
    )

    # Validate method exposure
    errors = validate_method_exposure(
        method_names=method_names,
        handler_identity=handler_identity,
        method_signatures=method_signatures,
        file_path=node_path,
    )

    return errors


# Example: Integration with introspection
from omnibase_infra.mixins import MixinNodeIntrospection


class SecureNode(MixinNodeIntrospection):
    """Node with security validation."""

    async def validate_security(self) -> list[ModelHandlerValidationError]:
        """Run security validation on this node's introspection data."""
        # Get capabilities via introspection
        capabilities = await self.get_capabilities()

        handler_identity = ModelHandlerIdentifier.from_handler_id(
            str(self._introspection_node_id)
        )

        # Validate capabilities for security concerns
        from omnibase_infra.validation import validate_handler_security

        errors = validate_handler_security(
            handler_identity=handler_identity,
            capabilities=capabilities,
            file_path=__file__,
        )

        return errors
```

#### Security Rule IDs

Security validation uses structured rule IDs for consistent error tracking:

| Rule ID | Violation | Remediation |
|---------|-----------|-------------|
| `SECURITY-001` | Sensitive method exposed | Prefix with underscore |
| `SECURITY-002` | Credential in signature | Use generic parameter names |
| `SECURITY-003` | Admin method public | Move to admin module or prefix with _ |
| `SECURITY-004` | Decrypt method public | Make private or move to crypto module |
| `SECURITY-100` | Credential in config | Move to Infisical/environment |
| `SECURITY-101` | Hardcoded secret | Use secret management |
| `SECURITY-102` | Insecure connection | Enable TLS/SSL |
| `SECURITY-200` | Insecure pattern | Follow security best practices |
| `SECURITY-201` | Missing auth check | Add authentication decorator |
| `SECURITY-202` | Missing input validation | Add Pydantic validation |

#### Error Output Formats

```python
# Example error for CI integration
error = ModelHandlerValidationError.from_security_violation(
    rule_id=SecurityRuleId.SENSITIVE_METHOD_EXPOSED,
    message="Handler exposes 'get_api_key' method",
    remediation_hint="Prefix with underscore: '_get_api_key'",
    handler_identity=ModelHandlerIdentifier.from_handler_id("auth-handler"),
    file_path="nodes/auth/handlers/handler_authenticate.py",
    line_number=42,
)

# Format for logging
print(error.format_for_logging())
# Output:
# Handler Validation Error [SECURITY-001]
# Type: security_validation_error
# Source: static_analysis
# Handler: auth-handler
# File: nodes/auth/handlers/handler_authenticate.py:42
# Message: Handler exposes 'get_api_key' method
# Remediation: Prefix with underscore: '_get_api_key'

# Format for CI (GitHub Actions annotation)
print(error.format_for_ci())
# Output:
# ::error file=nodes/auth/handlers/handler_authenticate.py,line=42::[SECURITY-001] Handler exposes 'get_api_key' method. Remediation: Prefix with underscore: '_get_api_key'

# Structured JSON for APIs
import json
print(json.dumps(error.to_structured_dict(), indent=2))
# Output: Full JSON structure with all error fields
```

#### Integration with Validation Pipeline

```python
from omnibase_infra.validation import (
    validate_handler_security,
    validate_method_exposure,
)


def validate_handlers_in_directory(directory: Path) -> list[ModelHandlerValidationError]:
    """Scan directory for security violations.

    This can be integrated into CI/CD pipelines as a pre-merge check.
    """
    errors: list[ModelHandlerValidationError] = []

    # Discover all handler files
    handler_files = directory.glob("**/node.py")

    for handler_file in handler_files:
        # Load handler and extract capabilities
        # (implementation depends on your introspection setup)
        capabilities = extract_capabilities(handler_file)

        handler_identity = ModelHandlerIdentifier.from_node(
            node_path=str(handler_file),
            handler_type=infer_handler_type(handler_file),
        )

        # Validate security
        handler_errors = validate_handler_security(
            handler_identity=handler_identity,
            capabilities=capabilities,
            file_path=str(handler_file),
        )

        errors.extend(handler_errors)

    return errors


# CI script example
def main() -> int:
    """CI gate for security validation."""
    errors = validate_handlers_in_directory(Path("src/handlers"))

    if not errors:
        print("✓ Security validation passed")
        return 0

    # Print all errors
    print(f"✗ Found {len(errors)} security violations:\n")
    for error in errors:
        print(error.format_for_ci())

    # Return exit code for CI
    blocking_errors = [e for e in errors if e.is_blocking()]
    return 1 if blocking_errors else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

**Related**:
- [ModelHandlerValidationError](../../src/omnibase_infra/models/errors/model_handler_validation_error.py) - Error model implementation
- [SecurityRuleId](../../src/omnibase_infra/validation/validator_security.py) - Rule ID definitions
- [Contract Linter](../../src/omnibase_infra/validation/linter_contract.py) - Similar integration pattern

---

## Logging and Audit

### Secure Logging Patterns

```python
import structlog
from uuid import UUID


def sanitize_log_data(data: dict[str, object]) -> dict[str, object]:
    """Remove sensitive data from log context.

    CRITICAL: Call before logging any user-provided data.
    """
    sensitive_keys = {
        "password", "secret", "token", "key", "credential",
        "authorization", "api_key", "access_token", "refresh_token",
        "ssn", "credit_card", "cvv", "pin",
    }

    sanitized = {}
    for key, value in data.items():
        key_lower = key.lower()

        # Check if key contains sensitive keywords
        if any(s in key_lower for s in sensitive_keys):
            sanitized[key] = "[REDACTED]"
        elif isinstance(value, dict):
            # Recursively sanitize nested dicts
            sanitized[key] = sanitize_log_data(value)
        else:
            sanitized[key] = value

    return sanitized


# Configure structlog with sanitization
structlog.configure(
    processors=[
        # Add timestamp
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),

        # Custom sanitization processor
        lambda _, __, event_dict: sanitize_log_data(event_dict),

        # JSON output for log aggregation
        structlog.processors.JSONRenderer(),
    ],
)


# Usage
logger = structlog.get_logger()

def process_login(username: str, password: str, correlation_id: UUID) -> None:
    # SAFE: password will be redacted by sanitization processor
    logger.info(
        "Login attempt",
        username=username,
        password=password,  # Will become "[REDACTED]" in logs
        correlation_id=str(correlation_id),
    )
```

### Audit Trail Pattern

```python
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel


class EnumAuditAction(str, Enum):
    """Auditable actions."""

    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    PERMISSION_CHANGE = "permission_change"
    SECRET_ACCESS = "secret_access"


class ModelAuditEvent(BaseModel):
    """Immutable audit event record.

    SECURITY: Audit logs must be append-only and tamper-evident.
    Consider using blockchain or write-once storage.
    """

    event_id: UUID
    timestamp: datetime
    action: EnumAuditAction
    actor_id: UUID
    actor_type: str  # "user", "service", "system"
    resource_type: str
    resource_id: str
    correlation_id: UUID
    outcome: str  # "success", "failure", "denied"
    ip_address: str | None = None
    user_agent: str | None = None
    metadata: dict[str, str] = {}  # Additional context (sanitized)


class AuditLogger:
    """Audit logging service.

    All security-relevant events must be logged to the audit trail.
    """

    def __init__(self, storage_backend: object) -> None:
        self._storage = storage_backend

    async def log_event(
        self,
        action: EnumAuditAction,
        actor_id: UUID,
        actor_type: str,
        resource_type: str,
        resource_id: str,
        correlation_id: UUID,
        outcome: str,
        **metadata: str,
    ) -> None:
        """Log audit event.

        Args:
            action: Action performed
            actor_id: Who performed the action
            actor_type: Type of actor (user/service/system)
            resource_type: Type of resource affected
            resource_id: ID of resource affected
            correlation_id: Request correlation ID
            outcome: Action outcome (success/failure/denied)
            **metadata: Additional context (will be sanitized)
        """
        event = ModelAuditEvent(
            event_id=uuid4(),
            timestamp=datetime.utcnow(),
            action=action,
            actor_id=actor_id,
            actor_type=actor_type,
            resource_type=resource_type,
            resource_id=resource_id,
            correlation_id=correlation_id,
            outcome=outcome,
            metadata=sanitize_log_data(metadata),
        )

        await self._storage.append(event)

    async def log_secret_access(
        self,
        actor_id: UUID,
        secret_path: str,
        correlation_id: UUID,
        outcome: str,
    ) -> None:
        """Log secret access for compliance.

        IMPORTANT: Log the path pattern, not the actual secret value.
        """
        # Sanitize path to not reveal structure
        sanitized_path = self._sanitize_secret_path(secret_path)

        await self.log_event(
            action=EnumAuditAction.SECRET_ACCESS,
            actor_id=actor_id,
            actor_type="service",
            resource_type="secret",
            resource_id=sanitized_path,
            correlation_id=correlation_id,
            outcome=outcome,
        )

    def _sanitize_secret_path(self, path: str) -> str:
        """Sanitize secret path for logging.

        Example: secret/data/db/prod/password -> secret/data/db/*/password
        """
        parts = path.split("/")
        if len(parts) > 3:
            # Mask environment-specific parts
            parts[3] = "*"
        return "/".join(parts)
```

---

## Production Security Checklist

Use this checklist before deploying to production:

### Pre-Deployment

- [ ] All secrets stored in Infisical, not in code or environment files
- [ ] **Handler contracts contain NO secrets** (passwords, API keys, tokens)
- [ ] CI pipeline includes secret scanning for contracts (gitleaks, detect-secrets)
- [ ] TLS 1.3 configured for all network connections
- [ ] Certificate verification enabled (`verify_mode=CERT_REQUIRED`)
- [ ] Database connections use SSL with certificate verification
- [ ] Kafka uses SASL_SSL authentication
- [ ] All Pydantic models have validation constraints
- [ ] Input validation at all API boundaries
- [ ] Error sanitization reviewed for all error paths
- [ ] No credentials in log output
- [ ] Audit logging enabled for security events

### Network Security

- [ ] Network zones defined and enforced
- [ ] Firewall rules reviewed and applied
- [ ] Kafka topic ACLs configured
- [ ] Service-to-service mTLS enabled
- [ ] Introspection topics access-controlled
- [ ] No internal services exposed to public network

### Authentication & Authorization

- [ ] All endpoints require authentication
- [ ] RBAC permissions reviewed and minimal
- [ ] Service accounts use short-lived tokens
- [ ] Token refresh implemented
- [ ] Session timeouts configured

### Monitoring & Response

- [ ] Security event alerting configured
- [ ] Audit logs shipped to secure storage
- [ ] Incident response plan documented
- [ ] Secret rotation schedule defined
- [ ] Vulnerability scanning enabled

---

## Related Patterns

- [Error Handling Patterns](./error_handling_patterns.md) - Error context and sanitization
- [Error Recovery Patterns](./error_recovery_patterns.md) - Credential refresh, circuit breakers
- [Correlation ID Tracking](./correlation_id_tracking.md) - Request tracing
- [Secret Resolver Pattern](./secret_resolver.md) - Centralized secret resolution with caching and migration guide
- [Policy Registry Trust Model](./policy_registry_trust_model.md) - Policy security model
- [Container Dependency Injection](./container_dependency_injection.md) - Service resolution
- [Handler Plugin Loader](./handler_plugin_loader.md#security-considerations) - Dynamic import security, threat model, deployment checklist

## See Also

- [CLAUDE.md - Error Sanitization](../../CLAUDE.md#error-sanitization)
- [CLAUDE.md - Handler Plugin Loader](../../CLAUDE.md#handler-plugin-loader) - Security model and error codes
- [ADR: Handler Plugin Loader Security Model](../decisions/adr-handler-plugin-loader-security.md) - Security architecture decisions
- [MixinNodeIntrospection](../../src/omnibase_infra/mixins/mixin_node_introspection.py) - Security documentation in docstring
