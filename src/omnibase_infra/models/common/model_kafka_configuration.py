"""Strongly typed Kafka configuration models."""

from typing import List, Optional
from pydantic import BaseModel, Field


class ModelKafkaConfiguration(BaseModel):
    """Strongly typed Kafka configuration model."""
    
    bootstrap_servers: List[str] = Field(
        description="List of Kafka bootstrap server addresses"
    )
    
    security_protocol: str = Field(
        default="PLAINTEXT",
        description="Security protocol (PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL)"
    )
    
    sasl_mechanism: Optional[str] = Field(
        default=None,
        description="SASL mechanism (PLAIN, SCRAM-SHA-256, SCRAM-SHA-512, GSSAPI)"
    )
    
    sasl_username: Optional[str] = Field(
        default=None,
        description="SASL username for authentication"
    )
    
    sasl_password: Optional[str] = Field(
        default=None,
        description="SASL password for authentication"
    )
    
    ssl_ca_location: Optional[str] = Field(
        default=None,
        description="Path to SSL CA certificate file"
    )
    
    ssl_certificate_location: Optional[str] = Field(
        default=None,
        description="Path to SSL certificate file"
    )
    
    ssl_key_location: Optional[str] = Field(
        default=None,
        description="Path to SSL private key file"
    )
    
    ssl_key_password: Optional[str] = Field(
        default=None,
        description="SSL private key password"
    )
    
    acks: str = Field(
        default="1",
        description="Producer acknowledgment setting (0, 1, all)"
    )
    
    retries: int = Field(
        default=3,
        description="Number of retries for failed requests",
        ge=0
    )
    
    max_in_flight_requests_per_connection: int = Field(
        default=5,
        description="Maximum unacknowledged requests per connection",
        ge=1
    )
    
    batch_size: int = Field(
        default=16384,
        description="Producer batch size in bytes",
        ge=0
    )
    
    linger_ms: int = Field(
        default=5,
        description="Producer linger time in milliseconds",
        ge=0
    )
    
    connections_max_idle_ms: int = Field(
        default=300000,
        description="Connection idle timeout in milliseconds",
        ge=0
    )
    
    request_timeout_ms: int = Field(
        default=30000,
        description="Request timeout in milliseconds",
        ge=1000
    )
    
    session_timeout_ms: int = Field(
        default=10000,
        description="Consumer session timeout in milliseconds",
        ge=1000
    )
    
    heartbeat_interval_ms: int = Field(
        default=3000,
        description="Consumer heartbeat interval in milliseconds",
        ge=1000
    )
    
    enable_auto_commit: bool = Field(
        default=True,
        description="Enable automatic offset commits for consumers"
    )
    
    auto_offset_reset: str = Field(
        default="latest",
        description="Consumer offset reset policy (earliest, latest, none)"
    )