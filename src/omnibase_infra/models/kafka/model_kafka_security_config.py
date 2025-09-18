"""Kafka security configuration model."""


from pydantic import BaseModel, Field, SecretStr, field_validator, field_serializer


class ModelKafkaSSLConfig(BaseModel):
    """SSL/TLS configuration for Kafka connections."""

    ssl_check_hostname: bool = Field(default=True, description="Whether to check hostname in SSL certificate")
    ssl_cafile: str | None = Field(default=None, description="Path to CA certificate file")
    ssl_certfile: str | None = Field(default=None, description="Path to client certificate file")
    ssl_keyfile: str | None = Field(default=None, description="Path to client private key file")
    ssl_password: SecretStr | None = Field(default=None, description="Password for client private key (securely stored)")
    ssl_crlfile: str | None = Field(default=None, description="Path to certificate revocation list file")
    ssl_ciphers: str | None = Field(default=None, description="SSL cipher suites to use")
    ssl_protocol: str | None = Field(default="TLSv1_2", description="SSL protocol version")
    ssl_context: str | None = Field(default=None, description="SSL context configuration")

    @field_validator("ssl_password", mode="before")
    @classmethod
    def validate_ssl_password(cls, v):
        """Validate SSL password without logging sensitive data."""
        if v is None:
            return v
        if isinstance(v, str):
            if len(v) > 200:
                raise ValueError("SSL password too long (max 200 characters)")
            return SecretStr(v)
        return v

    @field_serializer("ssl_password", when_used="unless-none")
    def serialize_ssl_password(self, value: SecretStr | None) -> str:
        """Serialize SSL password securely - excludes actual value."""
        if value is None:
            return None
        return "***REDACTED***"


class ModelKafkaSASLConfig(BaseModel):
    """SASL authentication configuration for Kafka connections."""

    sasl_mechanism: str | None = Field(default="PLAIN", description="SASL mechanism (PLAIN, SCRAM-SHA-256, SCRAM-SHA-512, GSSAPI)")
    sasl_plain_username: str | None = Field(default=None, description="Username for PLAIN SASL")
    sasl_plain_password: SecretStr | None = Field(default=None, description="Password for PLAIN SASL (securely stored)")
    sasl_kerberos_service_name: str | None = Field(default="kafka", description="Kerberos service name")
    sasl_kerberos_domain_name: str | None = Field(default=None, description="Kerberos domain name")
    sasl_oauth_token_provider: str | None = Field(default=None, description="OAuth token provider")

    @field_validator("sasl_plain_password", mode="before")
    @classmethod
    def validate_sasl_plain_password(cls, v):
        """Validate SASL password without logging sensitive data."""
        if v is None:
            return v
        if isinstance(v, str):
            if len(v) > 200:
                raise ValueError("SASL password too long (max 200 characters)")
            return SecretStr(v)
        return v

    @field_serializer("sasl_plain_password", when_used="unless-none")
    def serialize_sasl_plain_password(self, value: SecretStr | None) -> str:
        """Serialize SASL password securely - excludes actual value."""
        if value is None:
            return None
        return "***REDACTED***"


class ModelKafkaSecurityConfig(BaseModel):
    """Kafka security configuration model."""

    security_protocol: str = Field(default="PLAINTEXT", description="Security protocol (PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL)")
    ssl_config: ModelKafkaSSLConfig | None = Field(default=None, description="SSL/TLS configuration")
    sasl_config: ModelKafkaSASLConfig | None = Field(default=None, description="SASL authentication configuration")
    enable_auto_commit: bool = Field(default=True, description="Enable automatic offset commits")
    auto_commit_interval_ms: int = Field(default=5000, description="Auto commit interval in milliseconds")
    session_timeout_ms: int = Field(default=10000, description="Session timeout in milliseconds")
    heartbeat_interval_ms: int = Field(default=3000, description="Heartbeat interval in milliseconds")
    max_poll_interval_ms: int = Field(default=300000, description="Maximum poll interval in milliseconds")
    connections_max_idle_ms: int = Field(default=540000, description="Connection max idle time in milliseconds")
    request_timeout_ms: int = Field(default=30000, description="Request timeout in milliseconds")
    retry_backoff_ms: int = Field(default=100, description="Retry backoff time in milliseconds")
    reconnect_backoff_ms: int = Field(default=50, description="Reconnect backoff time in milliseconds")
    reconnect_backoff_max_ms: int = Field(default=1000, description="Maximum reconnect backoff time in milliseconds")
