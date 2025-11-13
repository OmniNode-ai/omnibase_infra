"""JWT authentication middleware with comprehensive security features."""

import logging
import os
import time
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from .audit_logger import AuditEventType, AuditSeverity, get_audit_logger

logger = logging.getLogger(__name__)


class JWTClaims(BaseModel):
    """Standard JWT claims with security extensions."""

    # Standard claims
    sub: str  # Subject (user ID)
    iss: str  # Issuer
    aud: str  # Audience
    exp: int  # Expiration time
    iat: int  # Issued at
    nbf: int  # Not before
    jti: str  # JWT ID

    # Custom security claims
    scope: list[str] = []  # Permissions/scopes
    role: str = "user"  # User role
    session_id: str = ""  # Session identifier
    client_id: str = ""  # Client application ID
    ip_address: str = ""  # Bound IP address (optional)
    device_id: str = ""  # Device identifier (optional)
    security_level: str = "standard"  # Security level: basic, standard, high, critical


class JWTConfig:
    """JWT configuration with security best practices."""

    def __init__(
        self,
        secret_key: str = None,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7,
        issuer: str = "omninode-bridge",
        audience: str = "omninode-api",
        require_exp: bool = True,
        require_iat: bool = True,
        require_nbf: bool = True,
        verify_signature: bool = True,
        verify_exp: bool = True,
        verify_iat: bool = True,
        verify_nbf: bool = True,
        leeway: int = 30,  # seconds
        environment: str = None,
    ):
        """Initialize JWT configuration with security best practices.

        Args:
            secret_key: JWT signing secret (should be strong and unique)
            algorithm: JWT signing algorithm (HS256, RS256, etc.)
            access_token_expire_minutes: Access token expiration in minutes
            refresh_token_expire_days: Refresh token expiration in days
            issuer: JWT issuer claim
            audience: JWT audience claim
            require_exp: Require expiration claim
            require_iat: Require issued at claim
            require_nbf: Require not before claim
            verify_signature: Verify JWT signature
            verify_exp: Verify expiration
            verify_iat: Verify issued at
            verify_nbf: Verify not before
            leeway: Time leeway for clock skew (seconds)
            environment: Deployment environment
        """
        self.environment = (
            environment or os.getenv("ENVIRONMENT", "development").lower()
        )

        # Use environment variable for secret key if not provided
        self.secret_key = secret_key or os.getenv("JWT_SECRET_KEY")
        if not self.secret_key:
            if self.environment == "production":
                raise ValueError(
                    "JWT_SECRET_KEY environment variable must be set in production",
                )
            else:
                # Generate a development key (warn user)
                self.secret_key = f"development-key-{uuid4()}"
                print("WARNING: Using generated JWT secret key for development")

        # Validate secret key strength in production
        if self.environment == "production" and len(self.secret_key) < 32:
            raise ValueError(
                "JWT secret key must be at least 32 characters in production",
            )

        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        self.issuer = issuer
        self.audience = audience

        # Verification options
        self.verify_options = {
            "require_exp": require_exp,
            "require_iat": require_iat,
            "require_nbf": require_nbf,
            "verify_signature": verify_signature,
            "verify_exp": verify_exp,
            "verify_iat": verify_iat,
            "verify_nbf": verify_nbf,
        }

        self.leeway = leeway


class JWTAuthenticator:
    """JWT authentication with security features and audit logging."""

    def __init__(
        self,
        config: JWTConfig,
        service_name: str = "omninode_bridge",
        enable_ip_binding: bool = False,
        enable_device_binding: bool = False,
        max_token_age_seconds: int = None,
        blacklist_enabled: bool = True,
    ):
        """Initialize JWT authenticator.

        Args:
            config: JWT configuration
            service_name: Service name for audit logging
            enable_ip_binding: Bind tokens to IP addresses
            enable_device_binding: Bind tokens to device IDs
            max_token_age_seconds: Maximum token age regardless of exp claim
            blacklist_enabled: Enable token blacklisting
        """
        self.config = config
        self.service_name = service_name
        self.enable_ip_binding = enable_ip_binding
        self.enable_device_binding = enable_device_binding
        self.max_token_age_seconds = max_token_age_seconds
        self.blacklist_enabled = blacklist_enabled

        # Initialize audit logger
        self.audit_logger = get_audit_logger(service_name)

        # Token blacklist (in production, use Redis or database)
        self._blacklist: set = set()

        # HTTP Bearer security scheme
        self.security_scheme = HTTPBearer(auto_error=False)

        # Log initialization
        self.audit_logger.log_event(
            event_type=AuditEventType.SERVICE_STARTUP,
            severity=AuditSeverity.LOW,
            additional_data={
                "component": "jwt_authenticator",
                "algorithm": config.algorithm,
                "access_token_expire_minutes": config.access_token_expire_minutes,
                "ip_binding": enable_ip_binding,
                "device_binding": enable_device_binding,
                "blacklist_enabled": blacklist_enabled,
            },
            message="JWT authenticator initialized",
        )

    def create_access_token(
        self,
        user_id: str,
        scopes: list[str] = None,
        role: str = "user",
        session_id: str = None,
        client_id: str = None,
        ip_address: str = None,
        device_id: str = None,
        security_level: str = "standard",
        custom_claims: dict[str, Any] = None,
    ) -> str:
        """Create an access token with security claims.

        Args:
            user_id: User identifier
            scopes: List of permission scopes
            role: User role
            session_id: Session identifier
            client_id: Client application ID
            ip_address: Client IP address (for binding)
            device_id: Device identifier (for binding)
            security_level: Security level
            custom_claims: Additional custom claims

        Returns:
            Encoded JWT token
        """
        now = datetime.now(UTC)
        expires = now + timedelta(minutes=self.config.access_token_expire_minutes)

        claims = {
            # Standard claims
            "sub": user_id,
            "iss": self.config.issuer,
            "aud": self.config.audience,
            "exp": int(expires.timestamp()),
            "iat": int(now.timestamp()),
            "nbf": int(now.timestamp()),
            "jti": str(uuid4()),
            # Custom security claims
            "scope": scopes or [],
            "role": role,
            "session_id": session_id or str(uuid4()),
            "client_id": client_id or "unknown",
            "security_level": security_level,
            "token_type": "access",
        }

        # Add binding claims if enabled
        if self.enable_ip_binding and ip_address:
            claims["ip_address"] = ip_address

        if self.enable_device_binding and device_id:
            claims["device_id"] = device_id

        # Add custom claims
        if custom_claims:
            claims.update(custom_claims)

        # Create token
        token = jwt.encode(
            claims,
            self.config.secret_key,
            algorithm=self.config.algorithm,
        )

        # Log token creation
        self.audit_logger.log_event(
            event_type=AuditEventType.AUTHENTICATION_SUCCESS,
            severity=AuditSeverity.LOW,
            additional_data={
                "component": "jwt_token_creation",
                "user_id": user_id,
                "token_id": claims["jti"],
                "role": role,
                "scopes": scopes or [],
                "security_level": security_level,
                "expires_at": expires.isoformat(),
            },
            message=f"JWT access token created for user: {user_id}",
        )

        return token

    def create_refresh_token(
        self,
        user_id: str,
        session_id: str,
        access_token_jti: str,
    ) -> str:
        """Create a refresh token linked to an access token.

        Args:
            user_id: User identifier
            session_id: Session identifier
            access_token_jti: JTI of the associated access token

        Returns:
            Encoded refresh token
        """
        now = datetime.now(UTC)
        expires = now + timedelta(days=self.config.refresh_token_expire_days)

        claims = {
            "sub": user_id,
            "iss": self.config.issuer,
            "aud": self.config.audience,
            "exp": int(expires.timestamp()),
            "iat": int(now.timestamp()),
            "nbf": int(now.timestamp()),
            "jti": str(uuid4()),
            "session_id": session_id,
            "access_token_jti": access_token_jti,
            "token_type": "refresh",
        }

        token = jwt.encode(
            claims,
            self.config.secret_key,
            algorithm=self.config.algorithm,
        )

        # Log refresh token creation
        self.audit_logger.log_event(
            event_type=AuditEventType.AUTHENTICATION_SUCCESS,
            severity=AuditSeverity.LOW,
            additional_data={
                "component": "jwt_refresh_token_creation",
                "user_id": user_id,
                "token_id": claims["jti"],
                "session_id": session_id,
                "expires_at": expires.isoformat(),
            },
            message=f"JWT refresh token created for user: {user_id}",
        )

        return token

    def verify_token(
        self,
        token: str,
        request: Request = None,
        required_scopes: list[str] = None,
        required_role: str = None,
        min_security_level: str = None,
    ) -> JWTClaims:
        """Verify and validate a JWT token with security checks.

        Args:
            token: JWT token to verify
            request: FastAPI request object (for IP binding)
            required_scopes: Required scopes for authorization
            required_role: Required role for authorization
            min_security_level: Minimum required security level

        Returns:
            Validated JWT claims

        Raises:
            HTTPException: If token is invalid or authorization fails
        """
        try:
            # Check blacklist first
            if self.blacklist_enabled and token in self._blacklist:
                self.audit_logger.log_event(
                    event_type=AuditEventType.AUTHENTICATION_FAILURE,
                    severity=AuditSeverity.HIGH,
                    request=request,
                    additional_data={
                        "component": "jwt_verification",
                        "reason": "token_blacklisted",
                    },
                    message="Blacklisted JWT token rejected",
                )
                raise HTTPException(status_code=401, detail="Token has been revoked")

            # Decode and verify token
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                options=self.config.verify_options,
                audience=self.config.audience,
                issuer=self.config.issuer,
                leeway=self.config.leeway,
            )

            # Create claims object
            claims = JWTClaims(**payload)

            # Additional security validations
            self._validate_token_security(claims, request)

            # Authorization checks
            if required_scopes:
                self._check_scopes(claims, required_scopes)

            if required_role:
                self._check_role(claims, required_role)

            if min_security_level:
                self._check_security_level(claims, min_security_level)

            # Log successful verification
            self.audit_logger.log_event(
                event_type=AuditEventType.AUTHENTICATION_SUCCESS,
                severity=AuditSeverity.LOW,
                request=request,
                additional_data={
                    "component": "jwt_verification",
                    "user_id": claims.sub,
                    "token_id": claims.jti,
                    "role": claims.role,
                    "scopes": claims.scope,
                    "security_level": claims.security_level,
                },
                message=f"JWT token verified for user: {claims.sub}",
            )

            return claims

        except jwt.ExpiredSignatureError:
            self.audit_logger.log_authentication_failure(
                reason="token_expired",
                auth_method="jwt",
                request=request,
            )
            raise HTTPException(status_code=401, detail="Token has expired")

        except jwt.InvalidTokenError as e:
            self.audit_logger.log_authentication_failure(
                reason=f"invalid_token: {e!s}",
                auth_method="jwt",
                request=request,
            )
            raise HTTPException(status_code=401, detail="Invalid token")

        except (ValueError, KeyError, TypeError) as e:
            # Expected validation errors during token processing
            self.audit_logger.log_event(
                event_type=AuditEventType.AUTHENTICATION_FAILURE,
                severity=AuditSeverity.MEDIUM,
                request=request,
                additional_data={
                    "component": "jwt_verification",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                message=f"JWT token validation failed: {e}",
            )
            raise HTTPException(status_code=401, detail="Token validation failed")

        except Exception as e:
            # Unexpected errors - log as security violation
            self.audit_logger.log_event(
                event_type=AuditEventType.SECURITY_VIOLATION,
                severity=AuditSeverity.HIGH,
                request=request,
                additional_data={
                    "component": "jwt_verification",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                message=f"Unexpected JWT verification error: {e}",
            )
            raise HTTPException(status_code=401, detail="Token verification failed")

    def _validate_token_security(
        self,
        claims: JWTClaims,
        request: Request = None,
    ) -> None:
        """Validate token security constraints."""
        # Check maximum token age
        if self.max_token_age_seconds:
            token_age = time.time() - claims.iat
            if token_age > self.max_token_age_seconds:
                raise HTTPException(status_code=401, detail="Token exceeds maximum age")

        # IP binding validation
        if self.enable_ip_binding and hasattr(claims, "ip_address") and request:
            client_ip = request.client.host if request.client else None
            if client_ip and claims.ip_address and client_ip != claims.ip_address:
                self.audit_logger.log_event(
                    event_type=AuditEventType.SECURITY_VIOLATION,
                    severity=AuditSeverity.HIGH,
                    request=request,
                    additional_data={
                        "component": "jwt_ip_binding",
                        "token_ip": claims.ip_address,
                        "request_ip": client_ip,
                        "user_id": claims.sub,
                    },
                    message="JWT token IP binding violation",
                )
                raise HTTPException(
                    status_code=401,
                    detail="Token IP binding violation",
                )

        # Device binding validation
        if self.enable_device_binding and hasattr(claims, "device_id"):
            # Implementation would check device ID from request headers
            # This is a placeholder for custom device binding logic
            pass

    def _check_scopes(self, claims: JWTClaims, required_scopes: list[str]) -> None:
        """Check if token has required scopes."""
        missing_scopes = set(required_scopes) - set(claims.scope)
        if missing_scopes:
            self.audit_logger.log_event(
                event_type=AuditEventType.AUTHORIZATION_FAILURE,
                severity=AuditSeverity.MEDIUM,
                additional_data={
                    "component": "jwt_scope_check",
                    "user_id": claims.sub,
                    "required_scopes": required_scopes,
                    "token_scopes": claims.scope,
                    "missing_scopes": list(missing_scopes),
                },
                message="JWT token insufficient scopes",
            )
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Missing scopes: {', '.join(missing_scopes)}",
            )

    def _check_role(self, claims: JWTClaims, required_role: str) -> None:
        """Check if token has required role."""
        role_hierarchy = {
            "user": 1,
            "moderator": 2,
            "admin": 3,
            "superadmin": 4,
        }

        user_level = role_hierarchy.get(claims.role, 0)
        required_level = role_hierarchy.get(required_role, 0)

        if user_level < required_level:
            self.audit_logger.log_event(
                event_type=AuditEventType.AUTHORIZATION_FAILURE,
                severity=AuditSeverity.MEDIUM,
                additional_data={
                    "component": "jwt_role_check",
                    "user_id": claims.sub,
                    "user_role": claims.role,
                    "required_role": required_role,
                },
                message="JWT token insufficient role",
            )
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient role. Required: {required_role}, have: {claims.role}",
            )

    def _check_security_level(self, claims: JWTClaims, min_security_level: str) -> None:
        """Check if token meets minimum security level."""
        security_levels = {
            "basic": 1,
            "standard": 2,
            "high": 3,
            "critical": 4,
        }

        token_level = security_levels.get(claims.security_level, 0)
        required_level = security_levels.get(min_security_level, 0)

        if token_level < required_level:
            self.audit_logger.log_event(
                event_type=AuditEventType.AUTHORIZATION_FAILURE,
                severity=AuditSeverity.HIGH,
                additional_data={
                    "component": "jwt_security_level_check",
                    "user_id": claims.sub,
                    "token_security_level": claims.security_level,
                    "required_security_level": min_security_level,
                },
                message="JWT token insufficient security level",
            )
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient security level. Required: {min_security_level}",
            )

    def revoke_token(self, token: str, reason: str = "manual_revocation") -> None:
        """Add token to blacklist."""
        if self.blacklist_enabled:
            self._blacklist.add(token)

            # Extract token ID for logging
            try:
                payload = jwt.decode(
                    token,
                    self.config.secret_key,
                    algorithms=[self.config.algorithm],
                    options={"verify_signature": False},
                )
                token_id = payload.get("jti", "unknown")
                user_id = payload.get("sub", "unknown")
            except (jwt.DecodeError, jwt.InvalidTokenError, ValueError, KeyError) as e:
                # Token decoding failed - use unknown identifiers
                logger.debug(f"Failed to decode token for revocation logging: {e}")
                token_id = "unknown"
                user_id = "unknown"

            self.audit_logger.log_event(
                event_type=AuditEventType.SESSION_TERMINATED,
                severity=AuditSeverity.MEDIUM,
                additional_data={
                    "component": "jwt_revocation",
                    "token_id": token_id,
                    "user_id": user_id,
                    "reason": reason,
                },
                message=f"JWT token revoked: {reason}",
            )

    def create_auth_dependency(
        self,
        required_scopes: list[str] = None,
        required_role: str = None,
        min_security_level: str = None,
    ):
        """Create FastAPI dependency for JWT authentication.

        Args:
            required_scopes: Required scopes for the endpoint
            required_role: Required role for the endpoint
            min_security_level: Minimum security level for the endpoint

        Returns:
            FastAPI dependency function
        """

        async def verify_jwt_token(
            request: Request,
            credentials: HTTPAuthorizationCredentials = Depends(self.security_scheme),
        ) -> JWTClaims:
            if not credentials or not credentials.credentials:
                raise HTTPException(status_code=401, detail="Missing JWT token")

            return self.verify_token(
                token=credentials.credentials,
                request=request,
                required_scopes=required_scopes,
                required_role=required_role,
                min_security_level=min_security_level,
            )

        return verify_jwt_token


def create_jwt_authenticator(
    environment: str = None,
    service_name: str = "omninode_bridge",
    **config_kwargs,
) -> JWTAuthenticator:
    """Create JWT authenticator with environment-specific configuration.

    Args:
        environment: Deployment environment
        service_name: Service name for audit logging
        **config_kwargs: Additional JWT configuration options

    Returns:
        JWTAuthenticator instance
    """
    environment = environment or os.getenv("ENVIRONMENT", "development").lower()

    # Environment-specific defaults
    if environment == "production":
        defaults = {
            "access_token_expire_minutes": 15,  # Shorter in production
            "enable_ip_binding": True,
            "blacklist_enabled": True,
            "max_token_age_seconds": 3600,  # 1 hour max regardless of exp
        }
    elif environment == "staging":
        defaults = {
            "access_token_expire_minutes": 30,
            "enable_ip_binding": False,
            "blacklist_enabled": True,
        }
    else:  # development
        defaults = {
            "access_token_expire_minutes": 60,  # Longer for development
            "enable_ip_binding": False,
            "blacklist_enabled": False,
        }

    # Merge with provided config
    defaults.update(config_kwargs)

    # Create JWT config
    jwt_config = JWTConfig(environment=environment)

    # Create authenticator
    return JWTAuthenticator(config=jwt_config, service_name=service_name, **defaults)
