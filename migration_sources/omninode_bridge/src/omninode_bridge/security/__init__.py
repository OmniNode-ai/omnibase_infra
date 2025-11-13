"""Security utilities and middleware for OmniNode Bridge."""

from .api_key_manager import (
    ApiKeyManager,
    ApiKeyMetadata,
    get_api_key_manager,
    setup_automatic_rotation,
)
from .audit_logger import AuditEventType, AuditLogger, AuditSeverity, get_audit_logger
from .config_validator import (
    ProductionSecurityValidator,
    SecurityConfiguration,
    SecurityValidationResult,
    create_security_validator,
    export_security_report,
    validate_production_security,
)
from .cors import CORSSecurityConfig, get_environment_cors_config, setup_secure_cors
from .exceptions import (
    AuthenticationError,
    BlockedPathAccessError,
    CommandInjectionError,
    CredentialValidationError,
    DangerousFunctionCallError,
    DangerousImportError,
    DynamicCodeExecutionError,
    HardcodedCredentialsError,
    InputValidationError,
    InvalidAPIKeyError,
    InvalidTokenError,
    MaliciousCodeError,
    MultipleSecurityViolationsError,
    NullByteInPathError,
    OutputValidationError,
    PathNotAllowedError,
    PathTraversalError,
    PathValidationError,
    RateLimitExceededError,
    ReDoSError,
    SecurityValidationError,
    SensitivePathAccessError,
    SQLInjectionError,
    UnauthorizedAccessError,
    WeakCredentialError,
    XSSAttemptError,
    create_security_exception,
)
from .headers import (
    SecurityHeadersMiddleware,
    get_security_headers_config,
    setup_security_headers,
    validate_security_headers,
)
from .input_validator import InputValidator, ValidationResult
from .jwt_auth import JWTAuthenticator, JWTClaims, JWTConfig, create_jwt_authenticator
from .middleware import (
    AuthenticationHandler,
    SecurityMiddleware,
    create_auth_handler,
    setup_security_middleware,
)
from .output_validator import OutputValidator, SecurityIssue, ValidationReport
from .path_validator import PathValidator
from .rate_limiting import (
    EndpointSecurity,
    EnhancedRateLimiter,
    SecurityRateLimits,
    get_rate_limiter,
    security_rate_limit,
)
from .regex_validator import (
    RegexValidationResult,
    RegexValidator,
    get_regex_validator,
    safe_compile,
    safe_findall,
    safe_match,
    safe_search,
)
from .request_signing import (
    RequestSigner,
    RequestVerifier,
    SignatureConfig,
    WebhookSecurity,
    create_request_signer,
    create_request_verifier,
    create_webhook_security,
)
from .validation import (
    InputSanitizer,
    SecureHookPayload,
    SecureTaskRequest,
    SecureWorkflowDefinition,
    SecurityValidator,
    get_security_validator,
)

__all__ = [
    # Security exceptions
    "SecurityValidationError",
    "InputValidationError",
    "CommandInjectionError",
    "PathTraversalError",
    "SQLInjectionError",
    "XSSAttemptError",
    "DynamicCodeExecutionError",
    "ReDoSError",
    "OutputValidationError",
    "MaliciousCodeError",
    "DangerousImportError",
    "DangerousFunctionCallError",
    "HardcodedCredentialsError",
    "SensitivePathAccessError",
    "PathValidationError",
    "PathNotAllowedError",
    "BlockedPathAccessError",
    "NullByteInPathError",
    "CredentialValidationError",
    "InvalidAPIKeyError",
    "WeakCredentialError",
    "RateLimitExceededError",
    "AuthenticationError",
    "InvalidTokenError",
    "UnauthorizedAccessError",
    "MultipleSecurityViolationsError",
    "create_security_exception",
    # Input/Output/Path validators
    "InputValidator",
    "ValidationResult",
    "OutputValidator",
    "SecurityIssue",
    "ValidationReport",
    "PathValidator",
    # Regex validation
    "RegexValidator",
    "RegexValidationResult",
    "get_regex_validator",
    "safe_compile",
    "safe_match",
    "safe_search",
    "safe_findall",
    # Audit logging
    "AuditLogger",
    "get_audit_logger",
    "AuditEventType",
    "AuditSeverity",
    # Rate limiting
    "EnhancedRateLimiter",
    "SecurityRateLimits",
    "get_rate_limiter",
    "security_rate_limit",
    "EndpointSecurity",
    # Input validation
    "SecurityValidator",
    "InputSanitizer",
    "get_security_validator",
    "SecureWorkflowDefinition",
    "SecureHookPayload",
    "SecureTaskRequest",
    # Middleware
    "SecurityMiddleware",
    "setup_security_middleware",
    "AuthenticationHandler",
    "create_auth_handler",
    # Security headers
    "SecurityHeadersMiddleware",
    "setup_security_headers",
    "get_security_headers_config",
    "validate_security_headers",
    # API key management
    "ApiKeyManager",
    "ApiKeyMetadata",
    "get_api_key_manager",
    "setup_automatic_rotation",
    # CORS security
    "CORSSecurityConfig",
    "setup_secure_cors",
    "get_environment_cors_config",
    # JWT authentication
    "JWTAuthenticator",
    "JWTConfig",
    "JWTClaims",
    "create_jwt_authenticator",
    # Request signing
    "RequestSigner",
    "RequestVerifier",
    "WebhookSecurity",
    "SignatureConfig",
    "create_webhook_security",
    "create_request_signer",
    "create_request_verifier",
    # Configuration validation
    "ProductionSecurityValidator",
    "SecurityConfiguration",
    "SecurityValidationResult",
    "create_security_validator",
    "validate_production_security",
    "export_security_report",
]
