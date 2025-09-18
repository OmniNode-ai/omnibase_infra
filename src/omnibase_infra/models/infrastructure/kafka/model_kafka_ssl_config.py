"""Kafka SSL configuration model."""

from pydantic import BaseModel, Field


class ModelKafkaSSLConfig(BaseModel):
    """SSL/TLS configuration for Kafka connections."""

    ssl_check_hostname: bool = Field(
        default=True, description="Whether to check hostname in SSL certificate",
    )
    ssl_cafile: str | None = Field(
        default=None, description="Path to CA certificate file",
    )
    ssl_certfile: str | None = Field(
        default=None, description="Path to client certificate file",
    )
    ssl_keyfile: str | None = Field(
        default=None, description="Path to client private key file",
    )
    ssl_password: str | None = Field(
        default=None, description="Password for client private key",
    )
    ssl_crlfile: str | None = Field(
        default=None, description="Path to certificate revocation list file",
    )
    ssl_ciphers: str | None = Field(
        default=None, description="SSL cipher suites to use",
    )
    ssl_protocol: str | None = Field(
        default="TLSv1_2", description="SSL protocol version",
    )
    ssl_context: str | None = Field(
        default=None, description="SSL context configuration",
    )