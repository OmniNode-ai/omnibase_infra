"""Enhanced CORS configuration with security-focused environment-specific restrictions."""

import os
import re
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .audit_logger import AuditEventType, AuditSeverity, get_audit_logger


class CORSSecurityConfig:
    """Security-focused CORS configuration with environment-specific restrictions."""

    def __init__(
        self,
        environment: str = None,
        service_name: str = "omninode_bridge",
        custom_origins: list[str] | None = None,
        custom_headers: list[str] | None = None,
        custom_methods: list[str] | None = None,
        enable_credentials: bool = True,
        max_age: int = 86400,  # 24 hours
        strict_mode: bool = None,
    ):
        """Initialize CORS security configuration.

        Args:
            environment: Deployment environment (production, staging, development)
            service_name: Service name for audit logging
            custom_origins: Additional allowed origins beyond defaults
            custom_headers: Additional allowed headers beyond defaults
            custom_methods: Custom allowed methods (overrides defaults)
            enable_credentials: Allow credentials in CORS requests
            max_age: Preflight cache duration in seconds
            strict_mode: Enable strict CORS validation (auto-detected if None)
        """
        self.environment = (
            environment or os.getenv("ENVIRONMENT", "development").lower()
        )
        self.service_name = service_name
        self.custom_origins = custom_origins or []
        self.custom_headers = custom_headers or []
        self.custom_methods = custom_methods
        self.enable_credentials = enable_credentials
        self.max_age = max_age

        # Auto-detect strict mode based on environment
        if strict_mode is None:
            self.strict_mode = self.environment in ["production", "staging"]
        else:
            self.strict_mode = strict_mode

        # Initialize audit logger
        self.audit_logger = get_audit_logger(service_name)

        # Build CORS configuration
        self.cors_config = self._build_cors_config()

        # Log CORS initialization
        self.audit_logger.log_event(
            event_type=AuditEventType.SERVICE_STARTUP,
            severity=AuditSeverity.LOW,
            additional_data={
                "component": "cors_security_config",
                "environment": self.environment,
                "strict_mode": self.strict_mode,
                "allowed_origins_count": len(self.cors_config["allow_origins"]),
                "credentials_enabled": self.enable_credentials,
            },
            message="CORS security configuration initialized",
        )

    def _get_default_origins(self) -> list[str]:
        """Get default allowed origins based on environment."""
        if self.environment == "production":
            # Production: Only explicitly allowed domains
            production_origins = os.getenv("CORS_PRODUCTION_ORIGINS", "").split(",")
            return [origin.strip() for origin in production_origins if origin.strip()]

        elif self.environment == "staging":
            # Staging: Known staging domains plus some flexibility
            staging_origins = os.getenv(
                "CORS_STAGING_ORIGINS",
                "https://staging.omninode.com,https://staging-api.omninode.com",
            ).split(",")
            return [origin.strip() for origin in staging_origins if origin.strip()]

        else:  # development
            # Development: Local development servers
            return [
                "http://localhost:3000",
                "http://localhost:3001",
                "http://localhost:8000",
                "http://localhost:8080",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8000",
                "http://0.0.0.0:3000",
                "http://0.0.0.0:8000",
            ]

    def _get_allowed_methods(self) -> list[str]:
        """Get allowed HTTP methods based on environment."""
        if self.custom_methods:
            return self.custom_methods

        if self.environment == "production":
            # Production: Minimal required methods
            return ["GET", "POST", "OPTIONS"]
        else:
            # Development/Staging: More permissive for testing
            return ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]

    def _get_allowed_headers(self) -> list[str]:
        """Get allowed headers based on environment and security requirements."""
        # Base security headers always allowed
        base_headers = [
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-API-Key",
            "X-Request-ID",
            "X-Trace-ID",
            "Accept",
            "Accept-Language",
            "Accept-Encoding",
        ]

        # Environment-specific headers
        if self.environment == "production":
            # Production: Minimal required headers
            allowed_headers = base_headers
        else:
            # Development/Staging: Additional headers for development tools
            allowed_headers = base_headers + [
                "X-Development-Mode",
                "X-Debug-Level",
                "X-Source-Map",
                "X-Hot-Reload",
            ]

        # Add custom headers
        allowed_headers.extend(self.custom_headers)

        return list(set(allowed_headers))  # Remove duplicates

    def _validate_origin_security(self, origin: str) -> bool:
        """Validate that an origin meets security requirements.

        Args:
            origin: Origin URL to validate

        Returns:
            True if origin is secure, False otherwise
        """
        try:
            parsed = urlparse(origin)

            # Basic URL validation
            if not parsed.scheme or not parsed.netloc:
                return False

            # HTTPS requirement in production
            if self.environment == "production" and parsed.scheme != "https":
                self.audit_logger.log_event(
                    event_type=AuditEventType.SECURITY_VIOLATION,
                    severity=AuditSeverity.HIGH,
                    additional_data={
                        "component": "cors_validation",
                        "origin": origin,
                        "violation": "non_https_in_production",
                    },
                    message=f"Non-HTTPS origin rejected in production: {origin}",
                )
                return False

            # Block localhost in production
            if self.environment == "production" and parsed.hostname in [
                "localhost",
                "127.0.0.1",
                "0.0.0.0",
            ]:
                self.audit_logger.log_event(
                    event_type=AuditEventType.SECURITY_VIOLATION,
                    severity=AuditSeverity.HIGH,
                    additional_data={
                        "component": "cors_validation",
                        "origin": origin,
                        "violation": "localhost_in_production",
                    },
                    message=f"Localhost origin rejected in production: {origin}",
                )
                return False

            # Block suspicious domains
            suspicious_patterns = [
                r".*\.ngrok\.io$",  # Tunneling services
                r".*\.localtunnel\.me$",
                r".*\.herokuapp\.com$",  # Unless specifically allowed
                r".*\.repl\.it$",
                r".*\.glitch\.me$",
            ]

            for pattern in suspicious_patterns:
                if re.match(pattern, parsed.netloc, re.IGNORECASE):
                    if self.strict_mode:
                        self.audit_logger.log_event(
                            event_type=AuditEventType.SECURITY_VIOLATION,
                            severity=AuditSeverity.MEDIUM,
                            additional_data={
                                "component": "cors_validation",
                                "origin": origin,
                                "violation": "suspicious_domain",
                                "pattern": pattern,
                            },
                            message=f"Suspicious domain origin rejected: {origin}",
                        )
                        return False

            return True

        except Exception as e:
            self.audit_logger.log_event(
                event_type=AuditEventType.SECURITY_VIOLATION,
                severity=AuditSeverity.MEDIUM,
                additional_data={
                    "component": "cors_validation",
                    "origin": origin,
                    "error": str(e),
                },
                message=f"Origin validation error: {origin}",
            )
            return False

    def _build_cors_config(self) -> dict[str, Any]:
        """Build CORS configuration dictionary."""
        # Get default origins and add custom ones
        origins = self._get_default_origins()
        all_origins = origins + self.custom_origins

        # Validate origins
        validated_origins = []
        for origin in all_origins:
            if self._validate_origin_security(origin):
                validated_origins.append(origin)
            else:
                self.audit_logger.log_event(
                    event_type=AuditEventType.SECURITY_VIOLATION,
                    severity=AuditSeverity.MEDIUM,
                    additional_data={
                        "component": "cors_config",
                        "rejected_origin": origin,
                    },
                    message=f"CORS origin rejected during validation: {origin}",
                )

        # Log if no valid origins found
        if not validated_origins:
            self.audit_logger.log_event(
                event_type=AuditEventType.SECURITY_VIOLATION,
                severity=AuditSeverity.CRITICAL,
                additional_data={
                    "component": "cors_config",
                    "environment": self.environment,
                    "total_origins_attempted": len(all_origins),
                },
                message="No valid CORS origins found - service may be inaccessible",
            )

        return {
            "allow_origins": validated_origins,
            "allow_credentials": self.enable_credentials,
            "allow_methods": self._get_allowed_methods(),
            "allow_headers": self._get_allowed_headers(),
            "max_age": self.max_age,
        }

    def apply_to_app(self, app: FastAPI) -> None:
        """Apply CORS configuration to FastAPI application.

        Args:
            app: FastAPI application instance
        """
        # Log CORS application
        self.audit_logger.log_event(
            event_type=AuditEventType.SERVICE_STARTUP,
            severity=AuditSeverity.LOW,
            additional_data={
                "component": "cors_application",
                "config": {
                    "origins_count": len(self.cors_config["allow_origins"]),
                    "methods": self.cors_config["allow_methods"],
                    "credentials": self.cors_config["allow_credentials"],
                    "environment": self.environment,
                },
            },
            message="Applying CORS configuration to FastAPI application",
        )

        # Add CORS middleware
        app.add_middleware(CORSMiddleware, **self.cors_config)

    def get_config(self) -> dict[str, Any]:
        """Get the current CORS configuration.

        Returns:
            Dictionary with CORS configuration
        """
        return self.cors_config.copy()

    def validate_request_origin(self, origin: str) -> bool:
        """Validate if a request origin is allowed.

        Args:
            origin: Origin header value from request

        Returns:
            True if origin is allowed, False otherwise
        """
        if not origin:
            return True  # No origin header is generally allowed

        # Check against allowed origins
        if origin in self.cors_config["allow_origins"]:
            return True

        # Check wildcard patterns (if any)
        for allowed_origin in self.cors_config["allow_origins"]:
            if allowed_origin == "*":
                return True  # Wildcard allows all (should be avoided in production)

        # Log blocked origin
        self.audit_logger.log_event(
            event_type=AuditEventType.SECURITY_VIOLATION,
            severity=AuditSeverity.MEDIUM,
            additional_data={
                "component": "cors_request_validation",
                "blocked_origin": origin,
                "allowed_origins": self.cors_config["allow_origins"],
            },
            message=f"CORS request blocked: origin not allowed: {origin}",
        )

        return False

    def get_security_report(self) -> dict[str, Any]:
        """Generate a security report for current CORS configuration.

        Returns:
            Dictionary with security analysis
        """
        config = self.cors_config
        security_issues = []
        security_score = 100

        # Check for wildcard origins
        if "*" in config["allow_origins"]:
            security_issues.append("Wildcard origin '*' allows any domain")
            security_score -= 50

        # Check for non-HTTPS origins in production
        if self.environment == "production":
            for origin in config["allow_origins"]:
                if origin.startswith("http://"):
                    security_issues.append(f"Non-HTTPS origin in production: {origin}")
                    security_score -= 20

        # Check for dangerous methods
        dangerous_methods = ["DELETE", "PUT", "PATCH"]
        if self.environment == "production":
            for method in dangerous_methods:
                if method in config["allow_methods"]:
                    security_issues.append(
                        f"Potentially dangerous method in production: {method}",
                    )
                    security_score -= 10

        # Check credentials with wildcard
        if config["allow_credentials"] and "*" in config["allow_origins"]:
            security_issues.append(
                "Credentials enabled with wildcard origin (dangerous)",
            )
            security_score -= 30

        return {
            "environment": self.environment,
            "security_score": max(0, security_score),
            "security_level": (
                "high"
                if security_score >= 80
                else "medium" if security_score >= 60 else "low"
            ),
            "issues": security_issues,
            "config_summary": {
                "origins_count": len(config["allow_origins"]),
                "methods_count": len(config["allow_methods"]),
                "headers_count": len(config["allow_headers"]),
                "credentials_enabled": config["allow_credentials"],
                "max_age": config["max_age"],
            },
            "recommendations": self._get_security_recommendations(),
        }

    def _get_security_recommendations(self) -> list[str]:
        """Get security recommendations for current configuration."""
        recommendations = []

        if self.environment == "production":
            if "*" in self.cors_config["allow_origins"]:
                recommendations.append("Remove wildcard origins in production")

            for origin in self.cors_config["allow_origins"]:
                if origin.startswith("http://"):
                    recommendations.append(f"Change {origin} to HTTPS")

            dangerous_methods = set(self.cors_config["allow_methods"]) & {
                "DELETE",
                "PUT",
                "PATCH",
            }
            if dangerous_methods:
                recommendations.append(
                    f"Consider removing methods: {', '.join(dangerous_methods)}",
                )

        # General recommendations
        if (
            self.cors_config["allow_credentials"]
            and len(self.cors_config["allow_origins"]) > 5
        ):
            recommendations.append(
                "Consider reducing allowed origins when credentials are enabled",
            )

        if self.max_age > 86400:  # More than 24 hours
            recommendations.append("Consider reducing preflight cache time (max_age)")

        return recommendations


def setup_secure_cors(
    app: FastAPI,
    environment: str = None,
    service_name: str = "omninode_bridge",
    **kwargs,
) -> CORSSecurityConfig:
    """Setup secure CORS configuration for FastAPI application.

    Args:
        app: FastAPI application instance
        environment: Deployment environment
        service_name: Service name for audit logging
        **kwargs: Additional CORS configuration options

    Returns:
        CORSSecurityConfig instance
    """
    cors_config = CORSSecurityConfig(
        environment=environment,
        service_name=service_name,
        **kwargs,
    )

    cors_config.apply_to_app(app)
    return cors_config


def get_environment_cors_config(environment: str) -> dict[str, Any]:
    """Get recommended CORS configuration for specific environment.

    Args:
        environment: Target environment

    Returns:
        Dictionary with recommended CORS settings
    """
    if environment.lower() == "production":
        return {
            "strict_mode": True,
            "enable_credentials": True,
            "max_age": 3600,  # 1 hour
            "custom_methods": ["GET", "POST", "OPTIONS"],  # Minimal methods
        }
    elif environment.lower() == "staging":
        return {
            "strict_mode": True,
            "enable_credentials": True,
            "max_age": 1800,  # 30 minutes
            "custom_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        }
    else:  # development
        return {
            "strict_mode": False,
            "enable_credentials": True,
            "max_age": 300,  # 5 minutes
            # Use default methods (more permissive)
        }
