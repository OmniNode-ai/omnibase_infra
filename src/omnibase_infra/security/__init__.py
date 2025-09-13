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

from .credential_manager import (
    ONEXCredentialManager,
    DatabaseCredentials,
    EventBusCredentials,
    get_credential_manager
)

from .tls_config import (
    ONEXTLSConfigManager,
    TLSCertificateConfig,
    PostgreSQLTLSConfig,
    KafkaTLSConfig,
    get_tls_manager
)

from .rate_limiter import (
    ONEXRateLimiter,
    RateLimitRule,
    ClientRateLimitState,
    RateLimitDecorator,
    get_rate_limiter
)

from .audit_logger import (
    ONEXAuditLogger,
    AuditEvent,
    AuditEventType,
    AuditSeverity,
    get_audit_logger
)

from .payload_encryption import (
    ONEXPayloadEncryption,
    EncryptedPayload,
    EncryptionMetadata,
    get_payload_encryption
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