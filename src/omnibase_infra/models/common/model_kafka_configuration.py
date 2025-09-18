"""Strongly typed Kafka configuration models."""

from pydantic import BaseModel, Field, SecretStr, field_validator, field_serializer

from ..security.enum_security_protocol import EnumSecurityProtocol
from .enum_sasl_mechanism import EnumSaslMechanism
from .enum_kafka_acks import EnumKafkaAcks
from .enum_kafka_offset_reset import EnumKafkaOffsetReset


class ModelKafkaConfiguration(BaseModel):
    """Strongly typed Kafka configuration model."""

    bootstrap_servers: list[str] = Field(
        description="List of Kafka bootstrap server addresses",
    )

    security_protocol: EnumSecurityProtocol = Field(
        default=EnumSecurityProtocol.PLAINTEXT,
        description="Security protocol for Kafka connection",
    )

    sasl_mechanism: EnumSaslMechanism | None = Field(
        default=None,
        description="SASL mechanism for authentication",
    )

    sasl_username: str | None = Field(
        default=None,
        description="SASL username for authentication",
    )

    sasl_password: SecretStr | None = Field(
        default=None,
        description="SASL password for authentication (securely stored)",
    )

    ssl_ca_location: str | None = Field(
        default=None,
        description="Path to SSL CA certificate file",
    )

    ssl_certificate_location: str | None = Field(
        default=None,
        description="Path to SSL certificate file",
    )

    ssl_key_location: str | None = Field(
        default=None,
        description="Path to SSL private key file",
    )

    ssl_key_password: SecretStr | None = Field(
        default=None,
        description="SSL private key password (securely stored)",
    )

    acks: EnumKafkaAcks = Field(
        default=EnumKafkaAcks.LEADER,
        description="Producer acknowledgment setting",
    )

    retries: int = Field(
        default=3,
        description="Number of retries for failed requests",
        ge=0,
    )

    max_in_flight_requests_per_connection: int = Field(
        default=5,
        description="Maximum unacknowledged requests per connection",
        ge=1,
    )

    batch_size: int = Field(
        default=16384,
        description="Producer batch size in bytes",
        ge=0,
    )

    linger_ms: int = Field(
        default=5,
        description="Producer linger time in milliseconds",
        ge=0,
    )

    connections_max_idle_ms: int = Field(
        default=300000,
        description="Connection idle timeout in milliseconds",
        ge=0,
    )

    request_timeout_ms: int = Field(
        default=30000,
        description="Request timeout in milliseconds",
        ge=1000,
    )

    session_timeout_ms: int = Field(
        default=10000,
        description="Consumer session timeout in milliseconds",
        ge=1000,
    )

    heartbeat_interval_ms: int = Field(
        default=3000,
        description="Consumer heartbeat interval in milliseconds",
        ge=1000,
    )

    enable_auto_commit: bool = Field(
        default=True,
        description="Enable automatic offset commits for consumers",
    )

    auto_offset_reset: EnumKafkaOffsetReset = Field(
        default=EnumKafkaOffsetReset.LATEST,
        description="Consumer offset reset policy",
    )

    @field_validator("sasl_password", "ssl_key_password", mode="before")
    @classmethod
    def validate_passwords(cls, v):
        """Validate passwords without logging sensitive data."""
        if v is None:
            return v
        if isinstance(v, str):
            if len(v) > 200:
                raise ValueError("Password too long (max 200 characters)")
            return SecretStr(v)
        return v

    @field_serializer("sasl_password", "ssl_key_password", when_used="unless-none")
    def serialize_passwords(self, value: SecretStr | None) -> str:
        """Serialize passwords securely - excludes actual values."""
        if value is None:
            return None
        return "***REDACTED***"

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        extra = "forbid"
        # Security: Never include sensitive fields in string representation
        json_encoders = {
            SecretStr: lambda v: "***REDACTED***" if v else None
        }
