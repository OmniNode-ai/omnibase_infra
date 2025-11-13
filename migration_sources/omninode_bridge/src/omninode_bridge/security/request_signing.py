"""Request signing and verification for webhook security."""

import hashlib
import hmac
import time

from fastapi import HTTPException, Request
from pydantic import BaseModel

from .audit_logger import AuditEventType, AuditSeverity, get_audit_logger


class SignatureConfig(BaseModel):
    """Configuration for request signing."""

    algorithm: str = "sha256"
    header_name: str = "X-Signature"
    timestamp_header: str = "X-Timestamp"
    timestamp_tolerance: int = 300  # 5 minutes
    require_timestamp: bool = True


class RequestSigner:
    """Sign outgoing requests with HMAC signatures."""

    def __init__(self, secret_key: str, config: SignatureConfig = None):
        """Initialize request signer.

        Args:
            secret_key: Secret key for signing
            config: Signature configuration
        """
        self.secret_key = secret_key.encode("utf-8")
        self.config = config or SignatureConfig()

    def sign_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str] = None,
        body: bytes = None,
        timestamp: int | None = None,
    ) -> dict[str, str]:
        """Sign a request and return headers to add.

        Args:
            method: HTTP method
            url: Request URL
            headers: Request headers
            body: Request body
            timestamp: Unix timestamp (current time if None)

        Returns:
            Dictionary of headers to add to request
        """
        headers = headers or {}
        body = body or b""
        timestamp = timestamp or int(time.time())

        # Create canonical string
        canonical_string = self._create_canonical_string(
            method,
            url,
            headers,
            body,
            timestamp,
        )

        # Create signature
        signature = hmac.new(
            self.secret_key,
            canonical_string.encode("utf-8"),
            getattr(hashlib, self.config.algorithm),
        ).hexdigest()

        # Return headers
        signature_headers = {
            self.config.header_name: f"{self.config.algorithm}={signature}",
        }

        if self.config.require_timestamp:
            signature_headers[self.config.timestamp_header] = str(timestamp)

        return signature_headers

    def _create_canonical_string(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes,
        timestamp: int,
    ) -> str:
        """Create canonical string for signing."""
        # Parse URL components
        from urllib.parse import parse_qs, urlparse

        parsed_url = urlparse(url)

        # Canonical request components
        canonical_method = method.upper()
        canonical_path = parsed_url.path or "/"

        # Sort query parameters
        query_params = parse_qs(parsed_url.query)
        sorted_params = []
        for key in sorted(query_params.keys()):
            for value in sorted(query_params[key]):
                sorted_params.append(f"{key}={value}")
        canonical_query = "&".join(sorted_params)

        # Select and sort headers for signing
        headers_to_sign = self._get_headers_to_sign(headers)
        canonical_headers = []
        for key in sorted(headers_to_sign.keys()):
            canonical_headers.append(f"{key.lower()}:{headers_to_sign[key].strip()}")
        canonical_headers_str = "\n".join(canonical_headers)

        # Body hash
        body_hash = hashlib.sha256(body).hexdigest()

        # Combine all components
        canonical_string = "\n".join(
            [
                canonical_method,
                canonical_path,
                canonical_query,
                canonical_headers_str,
                str(timestamp),
                body_hash,
            ],
        )

        return canonical_string

    def _get_headers_to_sign(self, headers: dict[str, str]) -> dict[str, str]:
        """Get headers that should be included in signature."""
        # Headers to include in signature (case-insensitive)
        important_headers = {
            "content-type",
            "content-length",
            "host",
            "user-agent",
            "authorization",
        }

        headers_to_sign = {}
        for key, value in headers.items():
            if key.lower() in important_headers:
                headers_to_sign[key] = value

        return headers_to_sign


class RequestVerifier:
    """Verify incoming signed requests."""

    def __init__(
        self,
        secret_key: str,
        service_name: str,
        config: SignatureConfig = None,
    ):
        """Initialize request verifier.

        Args:
            secret_key: Secret key for verification
            service_name: Service name for audit logging
            config: Signature configuration
        """
        self.secret_key = secret_key.encode("utf-8")
        self.service_name = service_name
        self.config = config or SignatureConfig()
        self.audit_logger = get_audit_logger(service_name)

    async def verify_request(self, request: Request) -> bool:
        """Verify request signature.

        Args:
            request: FastAPI request object

        Returns:
            True if signature is valid

        Raises:
            HTTPException: If verification fails
        """
        try:
            # Extract signature from headers
            signature_header = request.headers.get(self.config.header_name)
            if not signature_header:
                self._log_verification_failure(
                    request,
                    "missing_signature",
                    "No signature header found",
                )
                raise HTTPException(
                    status_code=401,
                    detail=f"Missing signature header: {self.config.header_name}",
                )

            # Parse signature
            if "=" not in signature_header:
                self._log_verification_failure(
                    request,
                    "invalid_signature_format",
                    f"Invalid signature format: {signature_header}",
                )
                raise HTTPException(status_code=401, detail="Invalid signature format")

            algorithm, provided_signature = signature_header.split("=", 1)

            if algorithm != self.config.algorithm:
                self._log_verification_failure(
                    request,
                    "unsupported_algorithm",
                    f"Unsupported algorithm: {algorithm}",
                )
                raise HTTPException(
                    status_code=401,
                    detail=f"Unsupported signature algorithm: {algorithm}",
                )

            # Verify timestamp if required
            if self.config.require_timestamp:
                timestamp_header = request.headers.get(self.config.timestamp_header)
                if not timestamp_header:
                    self._log_verification_failure(
                        request,
                        "missing_timestamp",
                        "No timestamp header found",
                    )
                    raise HTTPException(
                        status_code=401,
                        detail=f"Missing timestamp header: {self.config.timestamp_header}",
                    )

                try:
                    request_timestamp = int(timestamp_header)
                except ValueError:
                    self._log_verification_failure(
                        request,
                        "invalid_timestamp",
                        f"Invalid timestamp: {timestamp_header}",
                    )
                    raise HTTPException(
                        status_code=401,
                        detail="Invalid timestamp format",
                    )

                # Check timestamp tolerance
                current_time = int(time.time())
                time_diff = abs(current_time - request_timestamp)

                if time_diff > self.config.timestamp_tolerance:
                    self._log_verification_failure(
                        request,
                        "timestamp_expired",
                        f"Timestamp difference: {time_diff}s",
                    )
                    raise HTTPException(
                        status_code=401,
                        detail="Request timestamp outside acceptable range",
                    )
            else:
                request_timestamp = int(time.time())

            # Read request body
            body = await request.body()

            # Recreate signature
            expected_signature = self._calculate_signature(
                request,
                body,
                request_timestamp,
            )

            # Verify signature
            if not hmac.compare_digest(provided_signature, expected_signature):
                self._log_verification_failure(
                    request,
                    "signature_mismatch",
                    "Signature verification failed",
                )
                raise HTTPException(status_code=401, detail="Invalid signature")

            # Log successful verification
            self.audit_logger.log_event(
                event_type=AuditEventType.AUTHENTICATION_SUCCESS,
                severity=AuditSeverity.LOW,
                request=request,
                additional_data={
                    "component": "request_verifier",
                    "verification_method": "hmac_signature",
                    "algorithm": algorithm,
                    "timestamp_diff": abs(int(time.time()) - request_timestamp),
                },
                message="Request signature verified successfully",
            )

            return True

        except HTTPException:
            raise
        except Exception as e:
            self._log_verification_failure(request, "verification_error", str(e))
            raise HTTPException(status_code=500, detail="Signature verification failed")

    def _calculate_signature(
        self,
        request: Request,
        body: bytes,
        timestamp: int,
    ) -> str:
        """Calculate expected signature for request."""
        # Create signer with same config
        signer = RequestSigner(self.secret_key.decode("utf-8"), self.config)

        # Get headers
        headers = dict(request.headers)

        # Create canonical string
        canonical_string = signer._create_canonical_string(
            request.method,
            str(request.url),
            headers,
            body,
            timestamp,
        )

        # Calculate signature
        return hmac.new(
            self.secret_key,
            canonical_string.encode("utf-8"),
            getattr(hashlib, self.config.algorithm),
        ).hexdigest()

    def _log_verification_failure(
        self,
        request: Request,
        failure_type: str,
        details: str,
    ) -> None:
        """Log signature verification failure."""
        self.audit_logger.log_event(
            event_type=AuditEventType.AUTHENTICATION_FAILURE,
            severity=AuditSeverity.HIGH,
            request=request,
            additional_data={
                "component": "request_verifier",
                "failure_type": failure_type,
                "details": details,
                "provided_headers": {
                    key: value
                    for key, value in request.headers.items()
                    if key.lower().startswith("x-")
                },
            },
            message=f"Request signature verification failed: {failure_type}",
        )


class WebhookSecurity:
    """Combined webhook security with signing and additional protections."""

    def __init__(
        self,
        signing_secret: str,
        service_name: str,
        config: SignatureConfig = None,
        allowed_ips: list[str] | None = None,
        rate_limit_per_ip: int = 100,
    ):
        """Initialize webhook security.

        Args:
            signing_secret: Secret for request signing
            service_name: Service name for audit logging
            config: Signature configuration
            allowed_ips: List of allowed IP addresses (None = allow all)
            rate_limit_per_ip: Rate limit per IP per hour
        """
        self.verifier = RequestVerifier(signing_secret, service_name, config)
        self.service_name = service_name
        self.allowed_ips = set(allowed_ips) if allowed_ips else None
        self.rate_limit_per_ip = rate_limit_per_ip
        self.audit_logger = get_audit_logger(service_name)

        # Simple in-memory rate limiting (consider Redis for production)
        self._ip_requests: dict[str, list[float]] = {}

    async def verify_webhook_request(self, request: Request) -> bool:
        """Comprehensive webhook request verification.

        Args:
            request: FastAPI request object

        Returns:
            True if all checks pass
        """
        client_ip = self._get_client_ip(request)

        # 1. IP allowlist check
        if self.allowed_ips and client_ip not in self.allowed_ips:
            self.audit_logger.log_event(
                event_type=AuditEventType.AUTHORIZATION_FAILURE,
                severity=AuditSeverity.HIGH,
                request=request,
                additional_data={
                    "component": "webhook_security",
                    "client_ip": client_ip,
                    "check_failed": "ip_allowlist",
                    "allowed_ips": list(self.allowed_ips),
                },
                message=f"Webhook request from unauthorized IP: {client_ip}",
            )
            raise HTTPException(status_code=403, detail="IP address not authorized")

        # 2. Rate limiting check
        if not self._check_rate_limit(client_ip):
            self.audit_logger.log_event(
                event_type=AuditEventType.RATE_LIMIT_EXCEEDED,
                severity=AuditSeverity.MEDIUM,
                request=request,
                additional_data={
                    "component": "webhook_security",
                    "client_ip": client_ip,
                    "rate_limit": self.rate_limit_per_ip,
                },
                message=f"Rate limit exceeded for IP: {client_ip}",
            )
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # 3. Request signature verification
        await self.verifier.verify_request(request)

        # 4. Content-Type validation for webhooks
        content_type = request.headers.get("content-type", "").lower()
        if not content_type.startswith("application/json"):
            self.audit_logger.log_event(
                event_type=AuditEventType.INPUT_VALIDATION_FAILURE,
                severity=AuditSeverity.MEDIUM,
                request=request,
                additional_data={
                    "component": "webhook_security",
                    "content_type": content_type,
                },
                message=f"Invalid content type for webhook: {content_type}",
            )
            raise HTTPException(
                status_code=400,
                detail="Invalid content type. Expected application/json",
            )

        # Log successful verification
        self.audit_logger.log_event(
            event_type=AuditEventType.AUTHORIZATION_SUCCESS,
            severity=AuditSeverity.LOW,
            request=request,
            additional_data={
                "component": "webhook_security",
                "client_ip": client_ip,
                "content_type": content_type,
            },
            message="Webhook request verified successfully",
        )

        return True

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address considering proxies."""
        # Check X-Forwarded-For header first
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()

        # Check X-Real-IP header
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()

        # Fall back to direct client IP
        return request.client.host if request.client else "unknown"

    def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if client IP is within rate limit."""
        current_time = time.time()
        hour_ago = current_time - 3600  # 1 hour

        # Initialize tracking for new IPs
        if client_ip not in self._ip_requests:
            self._ip_requests[client_ip] = []

        # Clean old requests
        self._ip_requests[client_ip] = [
            req_time for req_time in self._ip_requests[client_ip] if req_time > hour_ago
        ]

        # Check limit
        if len(self._ip_requests[client_ip]) >= self.rate_limit_per_ip:
            return False

        # Add current request
        self._ip_requests[client_ip].append(current_time)
        return True


# Utility functions for easy integration


def create_webhook_security(
    signing_secret: str,
    service_name: str,
    **kwargs,
) -> WebhookSecurity:
    """Create webhook security instance with default configuration."""
    return WebhookSecurity(signing_secret, service_name, **kwargs)


def create_request_signer(secret_key: str, **kwargs) -> RequestSigner:
    """Create request signer with default configuration."""
    config = SignatureConfig(**kwargs)
    return RequestSigner(secret_key, config)


def create_request_verifier(
    secret_key: str,
    service_name: str,
    **kwargs,
) -> RequestVerifier:
    """Create request verifier with default configuration."""
    config_kwargs = {k: v for k, v in kwargs.items() if k in SignatureConfig.__fields__}
    config = SignatureConfig(**config_kwargs)
    return RequestVerifier(secret_key, service_name, config)
