"""Enhanced rate limiting with security-focused features."""

import time
from dataclasses import dataclass
from enum import Enum

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


class EndpointSecurity(Enum):
    """Security levels for different endpoint types."""

    PUBLIC = "public"  # Low security, higher limits
    AUTHENTICATED = "authenticated"  # Medium security, moderate limits
    SENSITIVE = "sensitive"  # High security, strict limits
    ADMIN = "admin"  # Critical security, very strict limits


@dataclass
class SecurityRateLimits:
    """Rate limit configurations based on security level."""

    # Requests per minute by security level
    PUBLIC = "240/minute"  # 4 requests per second
    AUTHENTICATED = "120/minute"  # 2 requests per second
    SENSITIVE = "60/minute"  # 1 request per second
    ADMIN = "30/minute"  # 0.5 requests per second

    # Special endpoint limits
    HEALTH_CHECK = "120/minute"  # Health checks
    METRICS = "60/minute"  # Metrics endpoint
    DOCUMENTATION = "300/minute"  # API documentation

    # Authentication specific
    AUTH_SUCCESS = "600/minute"  # High limit for successful auth
    AUTH_FAILURE = "10/minute"  # Very low limit for failed auth

    # Workflow specific
    WORKFLOW_SUBMISSION = "30/minute"  # Workflow submissions
    WORKFLOW_STATUS = "180/minute"  # Status queries
    WORKFLOW_CONTROL = "20/minute"  # Pause/resume/cancel

    # Session management
    SESSION_QUERY = "60/minute"  # Session queries
    SESSION_TERMINATE = "20/minute"  # Session termination

    # Hook processing
    HOOK_PROCESSING = "200/minute"  # Webhook processing

    # Model metrics
    MODEL_EXECUTION = "40/minute"  # Model task execution
    MODEL_COMPARISON = "10/minute"  # Resource-intensive comparisons


class EnhancedRateLimiter:
    """Enhanced rate limiter with security-focused features."""

    def __init__(self, key_func=get_remote_address):
        """Initialize enhanced rate limiter."""
        self.limiter = Limiter(key_func=key_func)
        self.failed_auth_attempts: dict[str, list[float]] = {}
        self.suspicious_ips: dict[str, float] = {}

    def get_limit_for_endpoint(
        self,
        endpoint_type: str,
        security_level: EndpointSecurity = EndpointSecurity.AUTHENTICATED,
    ) -> str:
        """Get appropriate rate limit for endpoint type and security level."""

        # Special endpoint types
        endpoint_limits = {
            "health": SecurityRateLimits.HEALTH_CHECK,
            "metrics": SecurityRateLimits.METRICS,
            "docs": SecurityRateLimits.DOCUMENTATION,
            "workflow_submission": SecurityRateLimits.WORKFLOW_SUBMISSION,
            "workflow_status": SecurityRateLimits.WORKFLOW_STATUS,
            "workflow_control": SecurityRateLimits.WORKFLOW_CONTROL,
            "session_query": SecurityRateLimits.SESSION_QUERY,
            "session_terminate": SecurityRateLimits.SESSION_TERMINATE,
            "hook_processing": SecurityRateLimits.HOOK_PROCESSING,
            "model_execution": SecurityRateLimits.MODEL_EXECUTION,
            "model_comparison": SecurityRateLimits.MODEL_COMPARISON,
        }

        if endpoint_type in endpoint_limits:
            return endpoint_limits[endpoint_type]

        # Default to security level limits
        security_limits = {
            EndpointSecurity.PUBLIC: SecurityRateLimits.PUBLIC,
            EndpointSecurity.AUTHENTICATED: SecurityRateLimits.AUTHENTICATED,
            EndpointSecurity.SENSITIVE: SecurityRateLimits.SENSITIVE,
            EndpointSecurity.ADMIN: SecurityRateLimits.ADMIN,
        }

        return security_limits.get(security_level, SecurityRateLimits.AUTHENTICATED)

    def record_auth_failure(self, client_ip: str) -> None:
        """Record authentication failure for adaptive rate limiting."""
        current_time = time.time()

        if client_ip not in self.failed_auth_attempts:
            self.failed_auth_attempts[client_ip] = []

        # Add current failure
        self.failed_auth_attempts[client_ip].append(current_time)

        # Clean up old failures (older than 1 hour)
        hour_ago = current_time - 3600
        self.failed_auth_attempts[client_ip] = [
            t for t in self.failed_auth_attempts[client_ip] if t > hour_ago
        ]

        # Mark as suspicious if too many failures
        recent_failures = len(
            [
                t
                for t in self.failed_auth_attempts[client_ip]
                if t > current_time - 900  # Last 15 minutes
            ],
        )

        if recent_failures >= 5:
            self.suspicious_ips[client_ip] = current_time

    def is_suspicious_ip(self, client_ip: str) -> bool:
        """Check if IP is marked as suspicious."""
        if client_ip not in self.suspicious_ips:
            return False

        # Remove old suspicious markings (24 hours)
        if time.time() - self.suspicious_ips[client_ip] > 86400:
            del self.suspicious_ips[client_ip]
            return False

        return True

    def get_adaptive_limit(self, client_ip: str, base_limit: str) -> str:
        """Get adaptive rate limit based on client behavior."""
        if self.is_suspicious_ip(client_ip):
            # Reduce limit by 75% for suspicious IPs
            base_number = int(base_limit.split("/")[0])
            period = base_limit.split("/")[1]
            return f"{max(1, base_number // 4)}/{period}"

        # Check recent auth failures
        if client_ip in self.failed_auth_attempts:
            recent_failures = len(
                [
                    t
                    for t in self.failed_auth_attempts[client_ip]
                    if t > time.time() - 300  # Last 5 minutes
                ],
            )

            if recent_failures >= 3:
                # Reduce limit by 50% for clients with recent auth failures
                base_number = int(base_limit.split("/")[0])
                period = base_limit.split("/")[1]
                return f"{max(1, base_number // 2)}/{period}"

        return base_limit

    def create_security_limiter(
        self,
        endpoint_type: str,
        security_level: EndpointSecurity = EndpointSecurity.AUTHENTICATED,
    ):
        """Create a rate limiter decorator for specific endpoint type."""
        base_limit = self.get_limit_for_endpoint(endpoint_type, security_level)

        def limiter_decorator(func):
            def wrapper(*args, **kwargs):
                # Get request from kwargs or args
                request = None
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

                if request:
                    client_ip = request.client.host if request.client else "unknown"
                    adaptive_limit = self.get_adaptive_limit(client_ip, base_limit)
                    return self.limiter.limit(adaptive_limit)(func)(*args, **kwargs)
                else:
                    return self.limiter.limit(base_limit)(func)(*args, **kwargs)

            return wrapper

        return limiter_decorator

    def create_auth_failure_limiter(self):
        """Create special rate limiter for authentication failures."""
        return self.limiter.limit(SecurityRateLimits.AUTH_FAILURE)

    def create_auth_success_limiter(self):
        """Create rate limiter for successful authentication."""
        return self.limiter.limit(SecurityRateLimits.AUTH_SUCCESS)


# Global enhanced rate limiter instance
enhanced_limiter = EnhancedRateLimiter()


def get_rate_limiter() -> EnhancedRateLimiter:
    """Get the global enhanced rate limiter instance."""
    return enhanced_limiter


def security_rate_limit(
    endpoint_type: str,
    security_level: EndpointSecurity = EndpointSecurity.AUTHENTICATED,
):
    """Decorator for applying security-focused rate limiting."""
    limit = enhanced_limiter.get_limit_for_endpoint(endpoint_type, security_level)
    return enhanced_limiter.limiter.limit(limit)
