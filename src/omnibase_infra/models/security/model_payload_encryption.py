"""Payload Encryption Models.

Strongly-typed models for payload encryption to replace Dict[str, Any] usage.
Maintains ONEX compliance with proper field validation and security measures.
"""


from pydantic import BaseModel, Field


class ModelEncryptedPayload(BaseModel):
    """Model for encrypted payload data."""

    # Encryption Metadata
    algorithm: str = Field(
        max_length=50,
        description="Encryption algorithm used",
    )

    key_id: str = Field(
        max_length=100,
        description="Identifier of the encryption key",
    )

    iv: str = Field(
        max_length=200,
        description="Initialization vector (base64 encoded)",
    )

    # Encrypted Data
    encrypted_data: str = Field(
        description="Base64 encoded encrypted data",
    )

    # Integrity
    hmac: str = Field(
        max_length=500,
        description="HMAC for data integrity verification",
    )

    checksum: str | None = Field(
        default=None,
        max_length=100,
        description="Additional checksum for verification",
    )

    # Metadata
    encrypted_at: str = Field(
        description="ISO timestamp when data was encrypted",
    )

    encryption_version: str = Field(
        default="1.0",
        max_length=20,
        description="Version of encryption scheme used",
    )

    content_type: str | None = Field(
        default=None,
        max_length=100,
        description="Original content type before encryption",
    )

    original_size_bytes: int | None = Field(
        default=None,
        ge=0,
        description="Size of original data before encryption",
    )

    compressed: bool = Field(
        default=False,
        description="Whether data was compressed before encryption",
    )

    # Security Context
    security_level: str = Field(
        default="standard",
        pattern="^(minimal|standard|high|maximum)$",
        description="Security level used for encryption",
    )

    key_derivation_rounds: int | None = Field(
        default=None,
        ge=1000,
        le=1000000,
        description="Number of key derivation rounds",
    )

    # Expiration
    expires_at: str | None = Field(
        default=None,
        description="ISO timestamp when encrypted data expires",
    )

    # Additional Context
    context_info: list[str] | None = Field(
        default=None,
        max_items=10,
        description="Additional context information (non-sensitive)",
    )

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        extra = "forbid"


class ModelDecryptionRequest(BaseModel):
    """Model for decryption request parameters."""

    # Encrypted payload reference
    encrypted_payload: ModelEncryptedPayload = Field(
        description="Encrypted payload to decrypt",
    )

    # Decryption context
    key_id: str | None = Field(
        default=None,
        max_length=100,
        description="Override key ID for decryption",
    )

    verify_integrity: bool = Field(
        default=True,
        description="Whether to verify data integrity",
    )

    verify_expiration: bool = Field(
        default=True,
        description="Whether to check expiration",
    )

    # Output preferences
    return_format: str = Field(
        default="string",
        pattern="^(string|bytes|dict|auto)$",
        description="Format for decrypted data",
    )

    decompress: bool = Field(
        default=True,
        description="Whether to decompress after decryption",
    )

    # Security validation
    required_security_level: str | None = Field(
        default=None,
        pattern="^(minimal|standard|high|maximum)$",
        description="Minimum required security level",
    )

    allowed_algorithms: list[str] | None = Field(
        default=None,
        max_items=10,
        description="List of allowed encryption algorithms",
    )

    max_age_seconds: int | None = Field(
        default=None,
        ge=0,
        description="Maximum age of encrypted data in seconds",
    )

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        extra = "forbid"


class ModelEncryptionRequest(BaseModel):
    """Model for encryption request parameters."""

    # Data to encrypt
    data: str | bytes = Field(
        description="Data to encrypt",
    )

    # Encryption parameters
    algorithm: str | None = Field(
        default=None,
        max_length=50,
        description="Encryption algorithm to use",
    )

    key_id: str | None = Field(
        default=None,
        max_length=100,
        description="Key ID to use for encryption",
    )

    security_level: str = Field(
        default="standard",
        pattern="^(minimal|standard|high|maximum)$",
        description="Security level for encryption",
    )

    # Processing options
    compress_before_encrypt: bool = Field(
        default=False,
        description="Whether to compress data before encryption",
    )

    include_metadata: bool = Field(
        default=True,
        description="Whether to include metadata in result",
    )

    # Expiration
    ttl_seconds: int | None = Field(
        default=None,
        ge=60,
        le=31536000,  # 1 year
        description="Time-to-live in seconds",
    )

    # Context
    content_type: str | None = Field(
        default=None,
        max_length=100,
        description="Content type of original data",
    )

    context_tags: list[str] | None = Field(
        default=None,
        max_items=10,
        description="Context tags for encryption",
    )

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        extra = "forbid"


class ModelEncryptionStats(BaseModel):
    """Model for encryption service statistics."""

    # Operation counts
    total_encryptions: int = Field(
        ge=0,
        description="Total number of encryption operations",
    )

    total_decryptions: int = Field(
        ge=0,
        description="Total number of decryption operations",
    )

    successful_operations: int = Field(
        ge=0,
        description="Number of successful operations",
    )

    failed_operations: int = Field(
        ge=0,
        description="Number of failed operations",
    )

    # Performance metrics
    average_encryption_time_ms: float = Field(
        ge=0.0,
        description="Average encryption time in milliseconds",
    )

    average_decryption_time_ms: float = Field(
        ge=0.0,
        description="Average decryption time in milliseconds",
    )

    total_data_encrypted_bytes: int = Field(
        ge=0,
        description="Total bytes encrypted",
    )

    total_data_decrypted_bytes: int = Field(
        ge=0,
        description="Total bytes decrypted",
    )

    # Key management
    active_keys_count: int = Field(
        ge=0,
        description="Number of active encryption keys",
    )

    key_rotations_count: int = Field(
        ge=0,
        description="Number of key rotations performed",
    )

    # Cache statistics
    cache_hit_rate: float = Field(
        ge=0.0,
        le=100.0,
        description="Cache hit rate percentage",
    )

    cache_size_bytes: int = Field(
        ge=0,
        description="Current cache size in bytes",
    )

    # Error tracking
    integrity_failures: int = Field(
        ge=0,
        description="Number of integrity verification failures",
    )

    expired_data_requests: int = Field(
        ge=0,
        description="Number of requests for expired data",
    )

    invalid_key_attempts: int = Field(
        ge=0,
        description="Number of invalid key attempts",
    )

    # Time information
    statistics_period_start: str = Field(
        description="ISO timestamp of statistics period start",
    )

    statistics_period_end: str = Field(
        description="ISO timestamp of statistics period end",
    )

    uptime_seconds: int = Field(
        ge=0,
        description="Service uptime in seconds",
    )

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        extra = "forbid"
