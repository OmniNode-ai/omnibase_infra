"""Security headers middleware for comprehensive HTTP security."""

import os
import time
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .audit_logger import AuditEventType, AuditSeverity, get_audit_logger


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Comprehensive security headers middleware with environment-specific configuration."""

    def __init__(
        self,
        app: FastAPI,
        service_name: str,
        environment: str = None,
        enable_hsts: bool = True,
        enable_csp: bool = True,
        enable_frame_options: bool = True,
        enable_content_type_options: bool = True,
        enable_referrer_policy: bool = True,
        enable_permissions_policy: bool = True,
        custom_headers: dict[str, str] | None = None,
    ):
        """Initialize security headers middleware.

        Args:
            app: FastAPI application instance
            service_name: Service name for audit logging
            environment: Deployment environment (production, staging, development)
            enable_hsts: Enable HTTP Strict Transport Security
            enable_csp: Enable Content Security Policy
            enable_frame_options: Enable X-Frame-Options
            enable_content_type_options: Enable X-Content-Type-Options
            enable_referrer_policy: Enable Referrer-Policy
            enable_permissions_policy: Enable Permissions-Policy
            custom_headers: Additional custom security headers
        """
        super().__init__(app)
        self.service_name = service_name
        self.environment = (
            environment or os.getenv("ENVIRONMENT", "development").lower()
        )

        # Security features configuration
        self.enable_hsts = enable_hsts
        self.enable_csp = enable_csp
        self.enable_frame_options = enable_frame_options
        self.enable_content_type_options = enable_content_type_options
        self.enable_referrer_policy = enable_referrer_policy
        self.enable_permissions_policy = enable_permissions_policy
        self.custom_headers = custom_headers or {}

        # Initialize audit logger
        self.audit_logger = get_audit_logger(service_name)

        # Build security headers based on environment
        self.security_headers = self._build_security_headers()

        # Log security headers initialization
        self.audit_logger.log_event(
            event_type=AuditEventType.SERVICE_STARTUP,
            severity=AuditSeverity.LOW,
            additional_data={
                "component": "security_headers_middleware",
                "environment": self.environment,
                "headers_count": len(self.security_headers),
                "enabled_features": {
                    "hsts": self.enable_hsts,
                    "csp": self.enable_csp,
                    "frame_options": self.enable_frame_options,
                    "content_type_options": self.enable_content_type_options,
                    "referrer_policy": self.enable_referrer_policy,
                    "permissions_policy": self.enable_permissions_policy,
                },
            },
            message="Security headers middleware initialized",
        )

    def _build_security_headers(self) -> dict[str, str]:
        """Build security headers based on environment and configuration."""
        headers = {}

        # X-Content-Type-Options: Prevent MIME type sniffing
        if self.enable_content_type_options:
            headers["X-Content-Type-Options"] = "nosniff"

        # X-Frame-Options: Prevent clickjacking
        if self.enable_frame_options:
            headers["X-Frame-Options"] = "DENY"

        # X-XSS-Protection: Enable XSS filtering (legacy, but still useful)
        headers["X-XSS-Protection"] = "1; mode=block"

        # Referrer-Policy: Control referrer information
        if self.enable_referrer_policy:
            if self.environment == "production":
                headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            else:
                headers["Referrer-Policy"] = "no-referrer-when-downgrade"

        # Content Security Policy
        if self.enable_csp:
            csp_directives = self._build_csp_directives()
            headers["Content-Security-Policy"] = "; ".join(csp_directives)

        # HTTP Strict Transport Security (HSTS)
        if self.enable_hsts and self.environment == "production":
            # Only enable HSTS in production with HTTPS
            headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )

        # Permissions Policy (Feature Policy successor)
        if self.enable_permissions_policy:
            permissions = self._build_permissions_policy()
            headers["Permissions-Policy"] = ", ".join(permissions)

        # Additional security headers
        headers.update(
            {
                # Cache control for sensitive responses
                "Cache-Control": "no-store, no-cache, must-revalidate, private",
                "Pragma": "no-cache",
                "Expires": "0",
                # Cross-origin policies
                "Cross-Origin-Embedder-Policy": "require-corp",
                "Cross-Origin-Opener-Policy": "same-origin",
                "Cross-Origin-Resource-Policy": "same-origin",
                # Server information hiding
                "Server": f"OmniNode-Bridge/{self.service_name}",
                # Security-related headers
                "X-Permitted-Cross-Domain-Policies": "none",
                "X-Download-Options": "noopen",
            },
        )

        # Environment-specific headers
        if self.environment == "development":
            headers.update(
                {
                    "X-Development-Mode": "true",
                    "Access-Control-Allow-Private-Network": "true",  # For local development
                },
            )
        elif self.environment == "production":
            headers.update(
                {
                    "X-Production-Security": "enabled",
                    # More restrictive cache control in production
                    "Cache-Control": "no-store, no-cache, must-revalidate, private, max-age=0",
                },
            )

        # Add custom headers
        headers.update(self.custom_headers)

        return headers

    def _build_csp_directives(self) -> list[str]:
        """Build Content Security Policy directives based on environment."""
        if self.environment == "production":
            # Strict CSP for production
            return [
                "default-src 'self'",
                "script-src 'self' 'unsafe-inline'",  # FastAPI docs need inline scripts
                "style-src 'self' 'unsafe-inline'",  # FastAPI docs need inline styles
                "img-src 'self' data: https:",
                "font-src 'self' https:",
                "connect-src 'self'",
                "media-src 'none'",
                "object-src 'none'",
                "child-src 'none'",
                "worker-src 'none'",
                "frame-ancestors 'none'",
                "form-action 'self'",
                "base-uri 'self'",
                "manifest-src 'self'",
                "upgrade-insecure-requests",
            ]
        elif self.environment == "staging":
            # Slightly more permissive for staging
            return [
                "default-src 'self'",
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
                "style-src 'self' 'unsafe-inline'",
                "img-src 'self' data: https:",
                "font-src 'self' https:",
                "connect-src 'self' https:",
                "media-src 'self'",
                "object-src 'none'",
                "child-src 'self'",
                "worker-src 'self'",
                "frame-ancestors 'none'",
                "form-action 'self'",
                "base-uri 'self'",
                "manifest-src 'self'",
            ]
        else:  # development
            # More permissive for development
            return [
                "default-src 'self'",
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
                "style-src 'self' 'unsafe-inline'",
                "img-src 'self' data: https: http:",
                "font-src 'self' https: http:",
                "connect-src 'self' https: http: ws: wss:",  # Allow WebSocket for dev tools
                "media-src 'self'",
                "object-src 'self'",
                "child-src 'self'",
                "worker-src 'self'",
                "frame-ancestors 'self'",
                "form-action 'self'",
                "base-uri 'self'",
                "manifest-src 'self'",
            ]

    def _build_permissions_policy(self) -> list[str]:
        """Build Permissions Policy directives."""
        # Disable potentially dangerous features
        return [
            "accelerometer=()",
            "ambient-light-sensor=()",
            "autoplay=()",
            "battery=()",
            "camera=()",
            "display-capture=()",
            "document-domain=()",
            "encrypted-media=()",
            "execution-while-not-rendered=()",
            "execution-while-out-of-viewport=()",
            "fullscreen=()",
            "geolocation=()",
            "gyroscope=()",
            "magnetometer=()",
            "microphone=()",
            "midi=()",
            "navigation-override=()",
            "payment=()",
            "picture-in-picture=()",
            "publickey-credentials-get=()",
            "screen-wake-lock=()",
            "sync-xhr=()",
            "usb=()",
            "web-share=()",
            "xr-spatial-tracking=()",
        ]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to all responses."""
        start_time = time.time()

        try:
            # Process the request
            response = await call_next(request)

            # Add security headers to response
            for header_name, header_value in self.security_headers.items():
                response.headers[header_name] = header_value

            # Add request ID for tracking
            request_id = getattr(request.state, "request_id", str(uuid4()))
            response.headers["X-Request-ID"] = request_id

            # Add processing time header for monitoring
            processing_time = (time.time() - start_time) * 1000
            response.headers["X-Processing-Time-MS"] = f"{processing_time:.2f}"

            # Security-specific headers based on response
            if response.status_code >= 400:
                # Add additional security headers for error responses
                response.headers["X-Error-Secure"] = "true"

                # Log security-relevant error responses
                if response.status_code in [401, 403, 429]:
                    self.audit_logger.log_event(
                        event_type=(
                            AuditEventType.SECURITY_VIOLATION
                            if response.status_code == 403
                            else AuditEventType.AUTHENTICATION_FAILURE
                        ),
                        severity=AuditSeverity.MEDIUM,
                        request=request,
                        additional_data={
                            "response_status": response.status_code,
                            "processing_time_ms": processing_time,
                            "request_id": request_id,
                        },
                        message=f"Security-relevant response: {response.status_code}",
                    )

            return response

        except Exception as e:
            # Log error in security headers middleware
            self.audit_logger.log_event(
                event_type=AuditEventType.SECURITY_VIOLATION,
                severity=AuditSeverity.HIGH,
                request=request,
                additional_data={
                    "component": "security_headers_middleware",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                message=f"Error in security headers middleware: {e}",
            )

            # Re-raise the exception to be handled by the application
            raise


def setup_security_headers(
    app: FastAPI,
    service_name: str,
    environment: str = None,
    **kwargs,
) -> SecurityHeadersMiddleware:
    """Setup security headers middleware for FastAPI application.

    Args:
        app: FastAPI application instance
        service_name: Service name for audit logging
        environment: Deployment environment
        **kwargs: Additional configuration options

    Returns:
        SecurityHeadersMiddleware instance
    """
    middleware = SecurityHeadersMiddleware(
        app=app,
        service_name=service_name,
        environment=environment,
        **kwargs,
    )

    # Add middleware to the application
    app.add_middleware(
        SecurityHeadersMiddleware,
        **{"service_name": service_name, "environment": environment, **kwargs},
    )

    return middleware


def get_security_headers_config(environment: str = None) -> dict[str, Any]:
    """Get recommended security headers configuration for environment.

    Args:
        environment: Target environment (production, staging, development)

    Returns:
        Dictionary with recommended configuration
    """
    environment = environment or os.getenv("ENVIRONMENT", "development").lower()

    base_config = {
        "enable_hsts": True,
        "enable_csp": True,
        "enable_frame_options": True,
        "enable_content_type_options": True,
        "enable_referrer_policy": True,
        "enable_permissions_policy": True,
    }

    if environment == "production":
        return {
            **base_config,
            "custom_headers": {
                "X-Production-Security": "strict",
                "X-Security-Level": "maximum",
            },
        }
    elif environment == "staging":
        return {
            **base_config,
            "custom_headers": {
                "X-Staging-Environment": "true",
                "X-Security-Level": "high",
            },
        }
    else:  # development
        return {
            **base_config,
            "enable_hsts": False,  # Don't enable HSTS in development
            "custom_headers": {
                "X-Development-Environment": "true",
                "X-Security-Level": "development",
            },
        }


def validate_security_headers(response: Response) -> dict[str, Any]:
    """Validate that security headers are properly set on a response.

    Args:
        response: FastAPI Response object

    Returns:
        Dictionary with validation results
    """
    required_headers = [
        "X-Content-Type-Options",
        "X-Frame-Options",
        "X-XSS-Protection",
        "Referrer-Policy",
        "Content-Security-Policy",
        "Cache-Control",
        "X-Request-ID",
    ]

    missing_headers = []
    present_headers = {}

    for header in required_headers:
        if header in response.headers:
            present_headers[header] = response.headers[header]
        else:
            missing_headers.append(header)

    return {
        "validation_passed": len(missing_headers) == 0,
        "missing_headers": missing_headers,
        "present_headers": present_headers,
        "total_security_headers": len(present_headers),
        "security_score": (len(present_headers) / len(required_headers)) * 100,
    }
