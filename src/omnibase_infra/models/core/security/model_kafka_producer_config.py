"""Kafka Producer Configuration Model.

Strongly-typed model for Kafka producer configuration to replace Dict[str, Any] usage.
Maintains ONEX compliance with proper field validation.
"""

from pydantic import BaseModel, Field, SecretStr, field_serializer, field_validator

from omnibase_infra.enums.enum_kafka_acks import EnumKafkaAcks
from omnibase_infra.enums.enum_security_protocol import EnumSecurityProtocol


class ModelKafkaProducerConfig(BaseModel):
    """Model for Kafka producer configuration."""

    # TLS/SSL Configuration
    security_protocol: EnumSecurityProtocol = Field(
        default=EnumSecurityProtocol.SSL,
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

    ssl_key_password: SecretStr | None = Field(
        default=None,
        description="Private key password (securely stored)",
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
    acks: EnumKafkaAcks = Field(
        default=EnumKafkaAcks.ALL,
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

    @field_validator("ssl_key_password", mode="before")
    @classmethod
    def validate_ssl_key_password(cls, v):
        """Validate SSL key password without logging sensitive data."""
        if v is None:
            return v
        if isinstance(v, str):
            # Convert string to SecretStr without logging the value
            if len(v) > 200:
                raise ValueError("SSL key password too long (max 200 characters)")
            return SecretStr(v)
        return v

    @field_serializer("ssl_key_password", when_used="unless-none")
    def serialize_ssl_key_password(self, value: SecretStr | None) -> str:
        """Serialize SSL key password securely - excludes actual value."""
        if value is None:
            return None
        return "***REDACTED***"

    class Config:
        """Pydantic model configuration."""

        validate_assignment = True
        extra = "forbid"
        # Security: Never include sensitive fields in string representation
        json_encoders = {
            SecretStr: lambda v: "***REDACTED***" if v else None,
        }
