"""TLS Configuration Models.

Strongly-typed models for TLS configuration to replace Dict[str, Any] usage.
Maintains ONEX compliance with proper field validation.
"""


from pydantic import BaseModel, Field


class ModelKafkaProducerConfig(BaseModel):
    """Model for Kafka producer configuration."""

    # TLS/SSL Configuration
    security_protocol: str = Field(
        default="SSL",
        description="Security protocol for Kafka connection",
    )

    ssl_ca_location: str | None = Field(
        default=None,
        max_length=500,
        description="Path to CA certificate file",
    )

    ssl_certificate_location: str | None = Field(
        default=None,
        max_length=500,
        description="Path to client certificate file",
    )

    ssl_key_location: str | None = Field(
        default=None,
        max_length=500,
        description="Path to client private key file",
    )

    ssl_key_password: str | None = Field(
        default=None,
        max_length=200,
        description="Private key password",
    )

    ssl_verify_hostname: bool = Field(
        default=True,
        description="Whether to verify hostname in SSL certificates",
    )

    ssl_check_hostname: bool = Field(
        default=True,
        description="Whether to check hostname in SSL certificates",
    )

    # Connection Configuration
    bootstrap_servers: list[str] = Field(
        min_items=1,
        max_items=10,
        description="List of Kafka bootstrap servers",
    )

    client_id: str | None = Field(
        default=None,
        max_length=100,
        description="Client identifier",
    )

    # Performance Configuration
    acks: str = Field(
        default="all",
        pattern="^(0|1|all)$",
        description="Number of acknowledgments required",
    )

    retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of retries for failed sends",
    )

    batch_size: int = Field(
        default=16384,
        ge=1,
        le=1048576,
        description="Batch size in bytes",
    )

    linger_ms: int = Field(
        default=5,
        ge=0,
        le=1000,
        description="Time to wait for additional records in ms",
    )

    buffer_memory: int = Field(
        default=33554432,
        ge=1048576,
        le=134217728,
        description="Total memory available for buffering",
    )

    # Timeout Configuration
    request_timeout_ms: int = Field(
        default=30000,
        ge=1000,
        le=300000,
        description="Request timeout in milliseconds",
    )

    delivery_timeout_ms: int = Field(
        default=120000,
        ge=5000,
        le=600000,
        description="Delivery timeout in milliseconds",
    )

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        extra = "forbid"


class ModelSecurityPolicy(BaseModel):
    """Model for security policy configuration."""

    # TLS Requirements
    tls_version_min: str = Field(
        default="1.2",
        pattern="^(1\\.2|1\\.3)$",
        description="Minimum required TLS version",
    )

    tls_version_max: str = Field(
        default="1.3",
        pattern="^(1\\.2|1\\.3)$",
        description="Maximum allowed TLS version",
    )

    cipher_suites: list[str] = Field(
        min_items=1,
        max_items=20,
        description="Allowed cipher suites",
    )

    # Certificate Requirements
    certificate_validation_required: bool = Field(
        default=True,
        description="Whether certificate validation is required",
    )

    hostname_verification_required: bool = Field(
        default=True,
        description="Whether hostname verification is required",
    )

    certificate_chain_validation: bool = Field(
        default=True,
        description="Whether certificate chain validation is required",
    )

    # Security Features
    perfect_forward_secrecy_required: bool = Field(
        default=True,
        description="Whether perfect forward secrecy is required",
    )

    ocsp_stapling_required: bool = Field(
        default=False,
        description="Whether OCSP stapling is required",
    )

    sni_required: bool = Field(
        default=True,
        description="Whether Server Name Indication is required",
    )

    # Compliance
    fips_mode_enabled: bool = Field(
        default=False,
        description="Whether FIPS mode is enabled",
    )

    compliance_level: str = Field(
        default="standard",
        pattern="^(minimal|standard|strict|maximum)$",
        description="Security compliance level",
    )

    audit_all_connections: bool = Field(
        default=True,
        description="Whether to audit all TLS connections",
    )

    # Timeouts and Limits
    handshake_timeout_ms: int = Field(
        default=30000,
        ge=5000,
        le=120000,
        description="TLS handshake timeout in milliseconds",
    )

    session_timeout_ms: int = Field(
        default=300000,
        ge=60000,
        le=3600000,
        description="TLS session timeout in milliseconds",
    )

    max_connections_per_host: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum connections per host",
    )

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        extra = "forbid"


class ModelCredentialCacheEntry(BaseModel):
    """Model for credential cache entry."""

    # Credential Information
    credential_type: str = Field(
        max_length=50,
        description="Type of credential (api_key, certificate, token)",
    )

    credential_id: str = Field(
        max_length=200,
        description="Identifier for the credential",
    )

    environment: str = Field(
        max_length=50,
        description="Environment the credential is for",
    )

    # Cache Metadata
    cached_at: str = Field(
        description="ISO timestamp when credential was cached",
    )

    expires_at: str | None = Field(
        default=None,
        description="ISO timestamp when credential expires",
    )

    last_validated: str | None = Field(
        default=None,
        description="ISO timestamp of last validation",
    )

    # Usage Tracking
    access_count: int = Field(
        default=0,
        ge=0,
        description="Number of times credential was accessed",
    )

    last_accessed: str | None = Field(
        default=None,
        description="ISO timestamp of last access",
    )

    # Security Status
    is_valid: bool = Field(
        default=True,
        description="Whether credential is currently valid",
    )

    validation_failures: int = Field(
        default=0,
        ge=0,
        description="Number of validation failures",
    )

    is_revoked: bool = Field(
        default=False,
        description="Whether credential has been revoked",
    )

    # Additional Metadata
    source: str | None = Field(
        default=None,
        max_length=100,
        description="Source of the credential",
    )

    scope: list[str] | None = Field(
        default=None,
        max_items=20,
        description="Scopes associated with the credential",
    )

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        extra = "forbid"
