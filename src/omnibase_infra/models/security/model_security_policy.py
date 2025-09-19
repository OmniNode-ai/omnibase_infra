"""Security Policy Configuration Model.

Strongly-typed model for security policy configuration.
Maintains ONEX compliance with proper field validation.
"""

from pydantic import BaseModel, Field

from .enum_compliance_level import EnumComplianceLevel


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

    compliance_level: EnumComplianceLevel = Field(
        default=EnumComplianceLevel.STANDARD,
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
