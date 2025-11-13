"""Security middleware for FastAPI applications."""

import time
from collections.abc import Callable
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .audit_logger import AuditEventType, AuditSeverity, get_audit_logger
from .rate_limiting import get_rate_limiter
from .validation import get_security_validator


class SecurityMiddleware(BaseHTTPMiddleware):
    """Comprehensive security middleware."""

    def __init__(
        self,
        app: FastAPI,
        service_name: str,
        api_key: str,
        enable_audit_logging: bool = True,
        enable_request_validation: bool = True,
        max_request_size: int = 10 * 1024 * 1024,  # 10MB
    ):
        """Initialize security middleware."""
        super().__init__(app)
        self.service_name = service_name
        self.api_key = api_key
        self.enable_audit_logging = enable_audit_logging
        self.enable_request_validation = enable_request_validation
        self.max_request_size = max_request_size

        # Initialize security components
        self.audit_logger = get_audit_logger(service_name)
        self.rate_limiter = get_rate_limiter()
        self.security_validator = get_security_validator()

        # Track suspicious activity
        self.suspicious_requests: dict[str, list] = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through security middleware."""
        start_time = time.time()
        request_id = str(uuid4())

        # Add request ID to request state
        request.state.request_id = request_id
        request.state.start_time = start_time

        try:
            # 1. Request size validation
            await self._validate_request_size(request)

            # 2. Suspicious activity detection
            await self._detect_suspicious_activity(request)

            # 3. Request validation (if enabled)
            if self.enable_request_validation:
                await self._validate_request_content(request)

            # 4. Process request
            response = await call_next(request)

            # 5. Audit logging (if enabled)
            if self.enable_audit_logging:
                await self._audit_log_request(request, response, start_time)

            return response

        except HTTPException as e:
            # Handle HTTP exceptions with audit logging
            if self.enable_audit_logging:
                await self._audit_log_error(request, e, start_time)

            return JSONResponse(
                status_code=e.status_code,
                content={"detail": e.detail, "request_id": request_id},
            )

        except Exception as e:
            # Handle unexpected errors
            if self.enable_audit_logging:
                self.audit_logger.log_event(
                    event_type=AuditEventType.SECURITY_VIOLATION,
                    severity=AuditSeverity.CRITICAL,
                    request=request,
                    additional_data={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "request_id": request_id,
                    },
                    message=f"Unexpected error in security middleware: {e}",
                )

            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "request_id": request_id},
            )

    async def _validate_request_size(self, request: Request) -> None:
        """Validate request size."""
        content_length = request.headers.get("content-length")
        if content_length:
            size = int(content_length)
            if size > self.max_request_size:
                if self.enable_audit_logging:
                    self.audit_logger.log_event(
                        event_type=AuditEventType.PAYLOAD_SIZE_EXCEEDED,
                        severity=AuditSeverity.MEDIUM,
                        request=request,
                        additional_data={
                            "content_length": size,
                            "max_allowed": self.max_request_size,
                        },
                        message=f"Request size {size} exceeds maximum {self.max_request_size}",
                    )

                raise HTTPException(
                    status_code=413,
                    detail=f"Request too large. Maximum size: {self.max_request_size} bytes",
                )

    async def _detect_suspicious_activity(self, request: Request) -> None:
        """Detect potentially suspicious activity."""
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()

        # Initialize tracking for new IPs
        if client_ip not in self.suspicious_requests:
            self.suspicious_requests[client_ip] = []

        # Add current request
        self.suspicious_requests[client_ip].append(current_time)

        # Clean up old requests (older than 5 minutes)
        five_minutes_ago = current_time - 300
        self.suspicious_requests[client_ip] = [
            t for t in self.suspicious_requests[client_ip] if t > five_minutes_ago
        ]

        # Check for suspicious patterns
        recent_requests = len(self.suspicious_requests[client_ip])

        # Suspicious if too many requests in short time
        if recent_requests > 500:  # More than 500 requests in 5 minutes
            risk_score = min(1.0, recent_requests / 1000)

            if self.enable_audit_logging:
                self.audit_logger.log_suspicious_activity(
                    activity_type="high_request_volume",
                    risk_score=risk_score,
                    indicators=[f"requests_5min:{recent_requests}"],
                    request=request,
                )

            # Block highly suspicious activity
            if risk_score > 0.8:
                raise HTTPException(
                    status_code=429,
                    detail="Suspicious activity detected. Please try again later.",
                )

    async def _validate_request_content(self, request: Request) -> None:
        """Validate request content for security."""
        # Skip validation for certain paths
        skip_paths = ["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]
        if request.url.path in skip_paths:
            return

        # Validate query parameters
        for key, value in request.query_params.items():
            try:
                self.security_validator.validate_input_safety(
                    value,
                    f"query_param.{key}",
                )
            except HTTPException as e:
                if self.enable_audit_logging:
                    self.audit_logger.log_malicious_input_detected(
                        input_type=f"query_param.{key}",
                        pattern_matched=str(e.detail),
                        request=request,
                    )
                raise

        # Validate headers (basic check)
        suspicious_headers = ["x-forwarded-for", "x-real-ip", "user-agent"]
        for header in suspicious_headers:
            value = request.headers.get(header)
            if value:
                try:
                    # Basic validation for headers
                    if len(value) > 1000:  # Reasonable header length limit
                        raise HTTPException(
                            status_code=400,
                            detail=f"Header {header} too long",
                        )

                    # Check for null bytes and control characters
                    if "\x00" in value or any(
                        ord(c) < 32 and c not in "\n\t" for c in value
                    ):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid characters in header {header}",
                        )

                except HTTPException as e:
                    if self.enable_audit_logging:
                        self.audit_logger.log_malicious_input_detected(
                            input_type=f"header.{header}",
                            pattern_matched=str(e.detail),
                            request=request,
                        )
                    raise

    async def _audit_log_request(
        self,
        request: Request,
        response: Response,
        start_time: float,
    ) -> None:
        """Log successful request for audit."""
        processing_time = (time.time() - start_time) * 1000

        # Log based on endpoint type
        if request.url.path.startswith("/hooks"):
            event_type = AuditEventType.WORKFLOW_SUBMISSION
        elif request.url.path.startswith("/sessions"):
            event_type = (
                AuditEventType.SESSION_CREATED
                if request.method == "POST"
                else AuditEventType.SESSION_TERMINATED
            )
        else:
            event_type = AuditEventType.AUTHORIZATION_SUCCESS

        self.audit_logger.log_event(
            event_type=event_type,
            severity=AuditSeverity.LOW,
            request=request,
            additional_data={
                "response_status": response.status_code,
                "processing_time_ms": processing_time,
                "request_id": request.state.request_id,
            },
            message=f"Request processed: {request.method} {request.url.path}",
        )

    async def _audit_log_error(
        self,
        request: Request,
        error: HTTPException,
        start_time: float,
    ) -> None:
        """Log error for audit."""
        processing_time = (time.time() - start_time) * 1000

        # Determine event type based on error
        if error.status_code == 401:
            event_type = AuditEventType.AUTHENTICATION_FAILURE
            severity = AuditSeverity.HIGH
        elif error.status_code == 403:
            event_type = AuditEventType.AUTHORIZATION_FAILURE
            severity = AuditSeverity.HIGH
        elif error.status_code == 429:
            event_type = AuditEventType.RATE_LIMIT_EXCEEDED
            severity = AuditSeverity.MEDIUM
        elif error.status_code == 400:
            event_type = AuditEventType.INPUT_VALIDATION_FAILURE
            severity = AuditSeverity.MEDIUM
        else:
            event_type = AuditEventType.SECURITY_VIOLATION
            severity = AuditSeverity.HIGH

        self.audit_logger.log_event(
            event_type=event_type,
            severity=severity,
            request=request,
            additional_data={
                "error_status": error.status_code,
                "error_detail": error.detail,
                "processing_time_ms": processing_time,
                "request_id": request.state.request_id,
            },
            message=f"Request failed: {error.status_code} - {error.detail}",
        )


def setup_security_middleware(
    app: FastAPI,
    service_name: str,
    api_key: str,
    enable_audit_logging: bool = True,
    enable_request_validation: bool = True,
    max_request_size: int = 10 * 1024 * 1024,
) -> SecurityMiddleware:
    """Setup security middleware for FastAPI app."""

    middleware = SecurityMiddleware(
        app=app,
        service_name=service_name,
        api_key=api_key,
        enable_audit_logging=enable_audit_logging,
        enable_request_validation=enable_request_validation,
        max_request_size=max_request_size,
    )

    # Add middleware to app
    app.add_middleware(
        SecurityMiddleware,
        service_name=service_name,
        api_key=api_key,
        enable_audit_logging=enable_audit_logging,
        enable_request_validation=enable_request_validation,
        max_request_size=max_request_size,
    )

    return middleware


class AuthenticationHandler:
    """Handle API key authentication with security features."""

    def __init__(self, api_key: str, service_name: str):
        """Initialize authentication handler."""
        self.api_key = api_key
        self.service_name = service_name
        self.audit_logger = get_audit_logger(service_name)
        self.rate_limiter = get_rate_limiter()
        self.security = HTTPBearer(auto_error=False)

    async def verify_api_key(
        self,
        request: Request,
        credentials: HTTPAuthorizationCredentials = None,
        x_api_key: str = None,
    ) -> bool:
        """Verify API key with security logging."""
        provided_key = None
        auth_method = "unknown"

        # Extract API key from different sources
        if credentials and credentials.scheme.lower() == "bearer":
            provided_key = credentials.credentials
            auth_method = "bearer_token"
        elif x_api_key:
            provided_key = x_api_key
            auth_method = "x_api_key_header"

        client_ip = request.client.host if request.client else "unknown"

        # Check if key is provided and valid
        if not provided_key or provided_key != self.api_key:
            # Record authentication failure
            self.rate_limiter.record_auth_failure(client_ip)

            # Log authentication failure
            self.audit_logger.log_authentication_failure(
                reason="invalid_or_missing_api_key",
                auth_method=auth_method,
                request=request,
            )

            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key. Provide via Authorization: Bearer <key> or X-API-Key header",
            )

        # Log successful authentication
        self.audit_logger.log_authentication_success(
            auth_method=auth_method,
            request=request,
        )

        return True


def create_auth_handler(api_key: str, service_name: str) -> AuthenticationHandler:
    """Create authentication handler."""
    return AuthenticationHandler(api_key, service_name)
