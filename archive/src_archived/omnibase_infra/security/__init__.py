"""
ONEX Infrastructure Security Module

Comprehensive security components for ONEX infrastructure including:
- Credential management with vault integration
- TLS/SSL configuration for secure communications
- Rate limiting for DoS protection
- Audit logging for compliance and monitoring
- Payload encryption for sensitive data protection

All security modules follow ONEX patterns and provide singleton access
for consistent security policy enforcement across the infrastructure.
"""

from .audit_logger import (
    AuditEvent,
    AuditEventType,
    AuditSeverity,
    ONEXAuditLogger,
    get_audit_logger,
)
from .credential_manager import (
    DatabaseCredentials,
    EventBusCredentials,
    ONEXCredentialManager,
    get_credential_manager,
)
from .payload_encryption import (
    EncryptedPayload,
    EncryptionMetadata,
    ONEXPayloadEncryption,
    get_payload_encryption,
)
from .rate_limiter import (
    ClientRateLimitState,
    ONEXRateLimiter,
    RateLimitDecorator,
    RateLimitRule,
    get_rate_limiter,
)
from .tls_config import (
    KafkaTLSConfig,
    ONEXTLSConfigManager,
    PostgreSQLTLSConfig,
    TLSCertificateConfig,
    get_tls_manager,
)

__all__ = [
    # Credential Management
    "ONEXCredentialManager",
    "DatabaseCredentials",
    "EventBusCredentials",
    "get_credential_manager",

    # TLS Configuration
    "ONEXTLSConfigManager",
    "TLSCertificateConfig",
    "PostgreSQLTLSConfig",
    "KafkaTLSConfig",
    "get_tls_manager",

    # Rate Limiting
    "ONEXRateLimiter",
    "RateLimitRule",
    "ClientRateLimitState",
    "RateLimitDecorator",
    "get_rate_limiter",

    # Audit Logging
    "ONEXAuditLogger",
    "AuditEvent",
    "AuditEventType",
    "AuditSeverity",
    "get_audit_logger",

    # Payload Encryption
    "ONEXPayloadEncryption",
    "EncryptedPayload",
    "EncryptionMetadata",
    "get_payload_encryption",
]
