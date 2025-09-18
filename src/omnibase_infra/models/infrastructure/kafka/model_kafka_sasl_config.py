"""Kafka SASL configuration model."""

from pydantic import BaseModel, Field


class ModelKafkaSASLConfig(BaseModel):
    """SASL authentication configuration for Kafka connections."""

    sasl_mechanism: str | None = Field(
        default="PLAIN",
        description="SASL mechanism (PLAIN, SCRAM-SHA-256, SCRAM-SHA-512, GSSAPI)",
    )
    sasl_plain_username: str | None = Field(
        default=None, description="Username for PLAIN SASL",
    )
    sasl_plain_password: str | None = Field(
        default=None, description="Password for PLAIN SASL",
    )
    sasl_kerberos_service_name: str | None = Field(
        default="kafka", description="Kerberos service name",
    )
    sasl_kerberos_domain_name: str | None = Field(
        default=None, description="Kerberos domain name",
    )
    sasl_oauth_token_provider: str | None = Field(
        default=None, description="OAuth token provider",
    )