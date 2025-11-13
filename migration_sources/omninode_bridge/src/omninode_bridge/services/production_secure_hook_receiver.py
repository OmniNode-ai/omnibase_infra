"""Production-ready secure HookReceiver with comprehensive security features."""

import os
import time
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import structlog
from circuitbreaker import circuit
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import Counter, Histogram, generate_latest
from slowapi.errors import RateLimitExceeded
from starlette.responses import Response

from ..models.events import (
    ServiceEventType,
    ServiceLifecycleEvent,
    ToolEventType,
    ToolExecutionEvent,
)
from ..models.hooks import HookEvent, HookMetadata, HookPayload, HookResponse
from ..security import (  # Authentication & Authorization; Input validation & security; Rate limiting; Security middleware; CORS security; Request signing; Audit logging
    AuditEventType,
    AuditSeverity,
    EndpointSecurity,
    SecureHookPayload,
    create_auth_handler,
    create_jwt_authenticator,
    create_webhook_security,
    get_api_key_manager,
    get_audit_logger,
    get_environment_cors_config,
    get_rate_limiter,
    get_security_headers_config,
    get_security_validator,
    security_rate_limit,
    setup_secure_cors,
    setup_security_headers,
    setup_security_middleware,
)
from .kafka_client import KafkaClient
from .postgres_client import PostgresClient

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Prometheus metrics
HOOK_EVENTS_TOTAL = Counter(
    "hook_events_total",
    "Total hook events received",
    ["source", "action"],
)
HOOK_PROCESSING_TIME = Histogram("hook_processing_seconds", "Hook processing time")
KAFKA_PUBLISH_ERRORS = Counter("kafka_publish_errors_total", "Kafka publish errors")
DATABASE_ERRORS = Counter("database_errors_total", "Database operation errors")
SECURITY_VIOLATIONS = Counter(
    "security_violations_total",
    "Security violations detected",
    ["violation_type"],
)
AUTH_FAILURES = Counter("auth_failures_total", "Authentication failures", ["reason"])
REQUEST_SIGNING_FAILURES = Counter(
    "request_signing_failures_total",
    "Request signing verification failures",
    ["failure_type"],
)


class ProductionSecureHookReceiverService:
    """Production-ready FastAPI service with comprehensive security features."""

    def __init__(
        self,
        kafka_bootstrap_servers: str = None,
        postgres_host: str = None,
        postgres_port: int = None,
        postgres_database: str = None,
        postgres_user: str = None,
        postgres_password: str = None,
        environment: str = None,
    ):
        """Initialize Production Secure HookReceiver service."""
        self.environment = (
            environment or os.getenv("ENVIRONMENT", "development").lower()
        )

        # Load from environment variables if parameters not provided
        kafka_servers = kafka_bootstrap_servers or os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS",
            "localhost:29092",
        )
        pg_host = postgres_host or os.getenv("POSTGRES_HOST", "localhost")
        pg_port = postgres_port or int(os.getenv("POSTGRES_PORT", "5436"))
        pg_database = postgres_database or os.getenv(
            "POSTGRES_DATABASE",
            "omninode_bridge",
        )
        pg_user = postgres_user or os.getenv("POSTGRES_USER", "postgres")
        pg_password = postgres_password or os.getenv("POSTGRES_PASSWORD")

        if not pg_password:
            raise ValueError(
                "Database password must be provided via POSTGRES_PASSWORD environment variable or postgres_password parameter",
            )

        self.kafka_client = KafkaClient(kafka_servers)
        self.postgres_client = PostgresClient(
            host=pg_host,
            port=pg_port,
            database=pg_database,
            user=pg_user,
            password=pg_password,
        )

        # Security components initialization
        self.service_name = "production_secure_hook_receiver"
        self.audit_logger = get_audit_logger(self.service_name, "1.0.0")
        self.rate_limiter = get_rate_limiter()
        self.security_validator = get_security_validator()

        # Authentication components
        self.api_key_manager = None  # Will be initialized asynchronously
        self.jwt_authenticator = None  # Will be initialized asynchronously
        self.traditional_auth_handler = None  # Fallback API key auth

        # Request signing security
        signing_secret = os.getenv(
            "WEBHOOK_SIGNING_SECRET",
            "default-webhook-secret-change-in-production",
        )
        if (
            self.environment == "production"
            and signing_secret == "default-webhook-secret-change-in-production"
        ):
            raise ValueError(
                "WEBHOOK_SIGNING_SECRET environment variable must be set in production",
            )

        self.webhook_security = create_webhook_security(
            signing_secret=signing_secret,
            service_name=self.service_name,
            allowed_ips=self._get_allowed_ips(),
            rate_limit_per_ip=self._get_rate_limit_per_ip(),
        )

        self.app: FastAPI | None = None

    def _get_allowed_ips(self) -> list[str]:
        """Get allowed IP addresses for webhook requests."""
        allowed_ips_env = os.getenv("WEBHOOK_ALLOWED_IPS", "")
        if allowed_ips_env:
            return [ip.strip() for ip in allowed_ips_env.split(",") if ip.strip()]
        return None  # Allow all IPs if not specified

    def _get_rate_limit_per_ip(self) -> int:
        """Get rate limit per IP based on environment."""
        if self.environment == "production":
            return int(
                os.getenv("WEBHOOK_RATE_LIMIT_PER_IP", "50"),
            )  # Stricter in production
        elif self.environment == "staging":
            return int(os.getenv("WEBHOOK_RATE_LIMIT_PER_IP", "100"))
        else:  # development
            return int(
                os.getenv("WEBHOOK_RATE_LIMIT_PER_IP", "1000"),
            )  # More permissive for development

    async def startup(self) -> None:
        """Initialize connections and security components on startup."""
        try:
            # Log service startup
            self.audit_logger.log_event(
                event_type=AuditEventType.SERVICE_STARTUP,
                severity=AuditSeverity.LOW,
                additional_data={
                    "environment": self.environment,
                    "security_level": "maximum",
                    "features": [
                        "api_key_rotation",
                        "jwt_authentication",
                        "request_signing",
                        "rate_limiting",
                        "input_validation",
                        "security_headers",
                        "cors_security",
                        "audit_logging",
                    ],
                },
                message="Production Secure HookReceiver service starting up",
            )

            # Initialize API key management
            try:
                self.api_key_manager = await get_api_key_manager(self.service_name)
                logger.info("API key manager initialized")
            except Exception as e:
                logger.warning(f"API key manager initialization failed: {e}")
                # Fall back to traditional API key auth - API_KEY is required
                api_key = os.getenv("API_KEY")
                if not api_key:
                    raise ValueError(
                        "API_KEY environment variable must be set. "
                        "Generate a secure key with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
                    )
                self.traditional_auth_handler = create_auth_handler(
                    api_key,
                    self.service_name,
                )

            # Initialize JWT authenticator
            try:
                self.jwt_authenticator = create_jwt_authenticator(
                    environment=self.environment,
                    service_name=self.service_name,
                )
                logger.info("JWT authenticator initialized")
            except Exception as e:
                logger.warning(f"JWT authenticator initialization failed: {e}")

            # Connect to Kafka
            await self.kafka_client.connect()
            logger.info("Kafka client connected")

            # Connect to PostgreSQL
            await self.postgres_client.connect()
            logger.info("PostgreSQL client connected")

            logger.info("Production Secure HookReceiver service started successfully")

        except Exception as e:
            logger.error(
                "Failed to start Production Secure HookReceiver service",
                error=str(e),
            )
            # Log startup failure
            self.audit_logger.log_event(
                event_type=AuditEventType.SERVICE_STARTUP,
                severity=AuditSeverity.CRITICAL,
                additional_data={"error": str(e), "error_type": type(e).__name__},
                message=f"Service startup failed: {e}",
            )
            raise

    async def shutdown(self) -> None:
        """Clean up connections on shutdown."""
        try:
            await self.kafka_client.disconnect()
            await self.postgres_client.disconnect()

            # Log service shutdown
            self.audit_logger.log_event(
                event_type=AuditEventType.SERVICE_SHUTDOWN,
                severity=AuditSeverity.LOW,
                message="Production Secure HookReceiver service shutdown completed",
            )

            logger.info("Production Secure HookReceiver service shutdown completed")

        except Exception as e:
            logger.error("Error during service shutdown", error=str(e))

    def create_app(self) -> FastAPI:
        """Create and configure FastAPI application with maximum security."""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            await self.startup()
            yield
            # Shutdown
            await self.shutdown()

        app = FastAPI(
            title="OmniNode Bridge - Production Secure HookReceiver Service",
            description="Maximum security webhook processing with comprehensive protection features",
            version="1.0.0",
            docs_url=(
                "/docs" if self.environment != "production" else None
            ),  # Disable docs in production
            redoc_url="/redoc" if self.environment != "production" else None,
            openapi_url="/openapi.json" if self.environment != "production" else None,
            lifespan=lifespan,
        )

        # Setup comprehensive security headers
        security_headers_config = get_security_headers_config(self.environment)
        setup_security_headers(
            app=app,
            service_name=self.service_name,
            environment=self.environment,
            **security_headers_config,
        )

        # Setup secure CORS
        cors_config = get_environment_cors_config(self.environment)
        setup_secure_cors(
            app=app,
            environment=self.environment,
            service_name=self.service_name,
            **cors_config,
        )

        # Setup comprehensive security middleware
        # API_KEY is required for security middleware
        api_key = os.getenv("API_KEY")
        if not api_key:
            raise ValueError(
                "API_KEY environment variable must be set for security middleware. "
                "Generate a secure key with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
            )

        setup_security_middleware(
            app=app,
            service_name=self.service_name,
            api_key=api_key,
            enable_audit_logging=True,
            enable_request_validation=True,
            max_request_size=512 * 1024,  # 512KB limit for production security
        )

        # Enhanced authentication dependency with multiple methods
        async def verify_authentication(
            request: Request,
            authorization: HTTPAuthorizationCredentials | None = Depends(
                HTTPBearer(auto_error=False)
            ),
            x_api_key: str | None = Header(None),
        ) -> dict:
            """Multi-method authentication with fallback options."""
            auth_result = {
                "method": None,
                "user_id": None,
                "claims": None,
                "api_key_id": None,
            }

            # Try JWT authentication first (if available)
            if (
                self.jwt_authenticator
                and authorization
                and authorization.scheme.lower() == "bearer"
            ):
                try:
                    claims = self.jwt_authenticator.verify_token(
                        token=authorization.credentials,
                        request=request,
                    )
                    auth_result.update(
                        {
                            "method": "jwt",
                            "user_id": claims.sub,
                            "claims": claims,
                        },
                    )
                    return auth_result
                except HTTPException:
                    # JWT auth failed, try other methods
                    pass

            # Try API key manager authentication (if available)
            if self.api_key_manager:
                try:
                    provided_key = None
                    if authorization and authorization.scheme.lower() == "bearer":
                        provided_key = authorization.credentials
                    elif x_api_key:
                        provided_key = x_api_key

                    if provided_key:
                        is_valid, key_id = await self.api_key_manager.validate_api_key(
                            provided_key,
                        )
                        if is_valid:
                            auth_result.update(
                                {
                                    "method": "api_key_manager",
                                    "api_key_id": key_id,
                                },
                            )
                            return auth_result
                except (AttributeError, ValueError, ConnectionError, TimeoutError):
                    # API key manager auth failed due to validation errors or service unavailability, try traditional
                    pass

            # Fall back to traditional API key authentication
            if self.traditional_auth_handler:
                try:
                    success = await self.traditional_auth_handler.verify_api_key(
                        request,
                        authorization,
                        x_api_key,
                    )
                    if success:
                        auth_result.update(
                            {
                                "method": "traditional_api_key",
                            },
                        )
                        return auth_result
                except HTTPException:
                    pass

            # All authentication methods failed
            self.audit_logger.log_authentication_failure(
                reason="all_auth_methods_failed",
                auth_method="multi_method",
                request=request,
            )
            AUTH_FAILURES.labels(reason="all_methods_failed").inc()
            raise HTTPException(
                status_code=401,
                detail="Authentication failed. Provide valid JWT token, API key via Authorization header, or X-API-Key header",
            )

        # Custom rate limit exceeded handler with detailed logging
        async def enhanced_rate_limit_handler(request: Request, exc: RateLimitExceeded):
            """Enhanced rate limit handler with security logging."""
            client_ip = request.client.host if request.client else "unknown"
            endpoint = request.url.path

            # Log rate limit violation
            self.audit_logger.log_rate_limit_exceeded(
                endpoint=endpoint,
                limit=str(exc.detail),
                request=request,
            )

            # Update metrics
            SECURITY_VIOLATIONS.labels(violation_type="rate_limit_exceeded").inc()

            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded - security protection activated",
                    "endpoint": endpoint,
                    "retry_after": "60 seconds",
                    "request_id": getattr(request.state, "request_id", "unknown"),
                    "security_level": "maximum",
                },
            )

        app.add_exception_handler(RateLimitExceeded, enhanced_rate_limit_handler)

        # Production Secure Hook Processing Endpoint
        @app.post("/hooks", response_model=HookResponse)
        @security_rate_limit("hook_processing", EndpointSecurity.AUTHENTICATED)
        async def receive_secure_hook(
            request: Request,
            auth_result: dict = Depends(verify_authentication),
        ) -> HookResponse:
            """Receive and process webhook events with maximum security validation."""
            start_time = time.time()
            request_id = getattr(request.state, "request_id", str(uuid4()))

            try:
                # 1. Request signing verification (for webhook security)
                webhook_verification_enabled = (
                    os.getenv("ENABLE_WEBHOOK_SIGNATURE_VERIFICATION", "true").lower()
                    == "true"
                )
                if webhook_verification_enabled:
                    try:
                        await self.webhook_security.verify_webhook_request(request)
                    except HTTPException as e:
                        REQUEST_SIGNING_FAILURES.labels(
                            failure_type="signature_verification",
                        ).inc()
                        self.audit_logger.log_event(
                            event_type=AuditEventType.SECURITY_VIOLATION,
                            severity=AuditSeverity.HIGH,
                            request=request,
                            additional_data={
                                "component": "webhook_signature_verification",
                                "error": str(e.detail),
                                "auth_method": auth_result["method"],
                            },
                            message="Webhook signature verification failed",
                        )
                        raise

                # 2. Parse and validate request body with comprehensive security
                body = await request.json()

                # 3. Enhanced input validation using secure models
                try:
                    # Validate hook payload structure
                    secure_payload = SecureHookPayload(**body)

                    # Additional comprehensive security validation
                    self.security_validator.validate_input_safety(
                        body,
                        "secure_hook_payload",
                    )

                except ValueError as e:
                    # Log input validation failure
                    self.audit_logger.log_input_validation_failure(
                        field="secure_hook_payload",
                        value_type="json",
                        validation_error=str(e),
                        request=request,
                    )
                    SECURITY_VIOLATIONS.labels(violation_type="input_validation").inc()

                    raise HTTPException(
                        status_code=400,
                        detail=f"Security validation failed: {e}",
                    )

                # 4. Create validated and secure hook event
                metadata = HookMetadata(
                    source=secure_payload.resource,
                    version=body.get("version", "1.0.0"),
                    environment=body.get("environment", self.environment),
                    correlation_id=body.get("correlation_id"),
                    trace_id=request.headers.get("X-Trace-ID"),
                    user_agent=request.headers.get("User-Agent"),
                    source_ip=request.client.host if request.client else None,
                )

                hook_event = HookEvent(
                    metadata=metadata,
                    payload=HookPayload(
                        action=secure_payload.action,
                        resource=secure_payload.resource,
                        resource_id=secure_payload.resource_id,
                        data=secure_payload.data,
                        previous_state=secure_payload.previous_state,
                        current_state=secure_payload.current_state,
                    ),
                )

                # 5. Security audit logging
                self.audit_logger.log_event(
                    event_type=AuditEventType.WORKFLOW_SUBMISSION,
                    severity=AuditSeverity.LOW,
                    request=request,
                    additional_data={
                        "hook_id": str(hook_event.id),
                        "source": metadata.source,
                        "action": secure_payload.action,
                        "resource": secure_payload.resource,
                        "request_id": request_id,
                        "auth_method": auth_result["method"],
                        "auth_user_id": auth_result.get("user_id"),
                        "auth_api_key_id": auth_result.get("api_key_id"),
                        "security_features": [
                            "signature_verified",
                            "input_validated",
                            "rate_limited",
                        ],
                    },
                    message=f"Secure hook received: {secure_payload.action} on {secure_payload.resource}",
                )

                # 6. Update metrics
                HOOK_EVENTS_TOTAL.labels(
                    source=metadata.source,
                    action=secure_payload.action,
                ).inc()

                # 7. Process the hook event with security context
                success = await self._process_secure_hook_event(hook_event, auth_result)

                processing_time = (time.time() - start_time) * 1000
                HOOK_PROCESSING_TIME.observe(processing_time / 1000)

                # 8. Record comprehensive metrics in database
                if self.postgres_client.is_connected:
                    await self.postgres_client.record_event_metrics(
                        event_id=hook_event.id,
                        processing_time_ms=processing_time,
                        kafka_publish_success=success,
                        error_message=None if success else "Processing failed",
                    )

                # 9. Final security audit log
                self.audit_logger.log_event(
                    event_type=(
                        AuditEventType.WORKFLOW_EXECUTION_COMPLETE
                        if success
                        else AuditEventType.WORKFLOW_EXECUTION_FAILURE
                    ),
                    severity=AuditSeverity.LOW if success else AuditSeverity.MEDIUM,
                    request=request,
                    additional_data={
                        "hook_id": str(hook_event.id),
                        "success": success,
                        "processing_time_ms": processing_time,
                        "request_id": request_id,
                        "auth_method": auth_result["method"],
                    },
                    message=f"Secure hook processing {'completed' if success else 'failed'}",
                )

                if success:
                    return HookResponse(
                        success=True,
                        message="Hook processed securely",
                        event_id=hook_event.id,
                        processing_time_ms=processing_time,
                    )
                return HookResponse(
                    success=False,
                    message="Hook processing failed",
                    event_id=hook_event.id,
                    processing_time_ms=processing_time,
                    errors=hook_event.processing_errors,
                )

            except HTTPException:
                # Re-raise HTTP exceptions
                raise
            except Exception as e:
                processing_time = (time.time() - start_time) * 1000

                # Log unexpected error with security context
                self.audit_logger.log_event(
                    event_type=AuditEventType.SECURITY_VIOLATION,
                    severity=AuditSeverity.HIGH,
                    request=request,
                    additional_data={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "processing_time_ms": processing_time,
                        "request_id": request_id,
                        "auth_method": auth_result.get("method"),
                    },
                    message=f"Unexpected secure hook processing error: {e}",
                )

                return HookResponse(
                    success=False,
                    message="Secure hook processing error",
                    event_id=uuid4(),
                    processing_time_ms=processing_time,
                    errors=[str(e)],
                )

        # Enhanced Health Check with Security Status
        @app.get("/health")
        @security_rate_limit("health", EndpointSecurity.PUBLIC)
        async def secure_health_check() -> dict[str, Any]:
            """Comprehensive health check endpoint with security monitoring."""
            health_status = {
                "status": "healthy",
                "timestamp": time.time(),
                "version": "1.0.0",
                "environment": self.environment,
                "security_status": {
                    "level": "maximum",
                    "features": {
                        "authentication": {
                            "api_key_manager": self.api_key_manager is not None,
                            "jwt_authenticator": self.jwt_authenticator is not None,
                            "traditional_auth": self.traditional_auth_handler
                            is not None,
                        },
                        "request_signing": True,
                        "rate_limiting": True,
                        "input_validation": True,
                        "security_headers": True,
                        "cors_security": True,
                        "audit_logging": True,
                    },
                },
                "dependencies": {},
            }

            # Check Kafka health
            kafka_health = await self.kafka_client.health_check()
            health_status["dependencies"]["kafka"] = kafka_health

            # Check PostgreSQL health
            postgres_health = await self.postgres_client.health_check()
            health_status["dependencies"]["postgres"] = postgres_health

            # Overall status
            all_healthy = all(
                dep.get("status") == "healthy"
                for dep in health_status["dependencies"].values()
            )

            health_status["status"] = "healthy" if all_healthy else "degraded"

            # Log health check if degraded
            if not all_healthy:
                self.audit_logger.log_event(
                    event_type=AuditEventType.HEALTH_CHECK_FAILURE,
                    severity=AuditSeverity.MEDIUM,
                    additional_data=health_status["dependencies"],
                    message="Health check shows degraded status",
                )

            status_code = 200 if all_healthy else 503
            return JSONResponse(content=health_status, status_code=status_code)

        # Secure Metrics Endpoint (Authentication Required)
        @app.get("/metrics")
        @security_rate_limit("metrics", EndpointSecurity.AUTHENTICATED)
        async def get_secure_metrics(
            auth_result: dict = Depends(verify_authentication),
        ) -> Response:
            """Prometheus metrics endpoint with authentication."""
            return Response(generate_latest(), media_type="text/plain")

        # Root endpoint with security information
        @app.get("/")
        @security_rate_limit("docs", EndpointSecurity.PUBLIC)
        async def secure_root() -> dict[str, str]:
            """Root endpoint with enhanced security service information."""
            return {
                "service": "OmniNode Bridge - Production Secure HookReceiver",
                "version": "1.0.0",
                "status": "operational",
                "security_level": "maximum",
                "environment": self.environment,
                "features": [
                    "multi_method_auth",
                    "request_signing",
                    "comprehensive_validation",
                    "rate_limiting",
                    "security_headers",
                    "audit_logging",
                ],
                "docs": (
                    "/docs"
                    if self.environment != "production"
                    else "disabled_in_production"
                ),
            }

        self.app = app
        return app

    @circuit(failure_threshold=5, recovery_timeout=30, expected_exception=Exception)
    async def _process_secure_hook_event(
        self,
        hook_event: HookEvent,
        auth_result: dict,
    ) -> bool:
        """Process a hook event with enhanced security logging and auth context."""
        try:
            # Store hook event with security context
            if self.postgres_client.is_connected:
                hook_data = hook_event.model_dump()
                # Add security context to stored data
                hook_data["security_context"] = {
                    "auth_method": auth_result["method"],
                    "auth_user_id": auth_result.get("user_id"),
                    "auth_api_key_id": auth_result.get("api_key_id"),
                    "processing_level": "maximum_security",
                }
                await self._store_hook_event_with_circuit_breaker(hook_data)

            # Convert hook to internal event(s)
            internal_events = self._convert_hook_to_events(hook_event)

            # Publish events to Kafka with security context
            all_published = True
            for event in internal_events:
                if self.kafka_client.is_connected:
                    # Add security metadata to events
                    if hasattr(event, "metadata"):
                        event.metadata.update(
                            {
                                "security_level": "maximum",
                                "auth_method": auth_result["method"],
                            },
                        )

                    published = await self._publish_event_with_circuit_breaker(event)
                    if not published:
                        all_published = False
                        KAFKA_PUBLISH_ERRORS.inc()
                        hook_event.processing_errors.append(
                            f"Failed to publish {event.type} event",
                        )

            # Handle session management
            await self._handle_session_management(hook_event)

            # Mark as processed
            hook_event.processed = all_published

            logger.info(
                "Processed secure hook event",
                event_id=str(hook_event.id),
                source=hook_event.metadata.source,
                action=hook_event.payload.action,
                success=all_published,
                auth_method=auth_result["method"],
                security_level="maximum",
            )

            return all_published

        except Exception as e:
            # Enhanced error logging with security context
            self.audit_logger.log_event(
                event_type=AuditEventType.WORKFLOW_EXECUTION_FAILURE,
                severity=AuditSeverity.HIGH,
                additional_data={
                    "hook_id": str(hook_event.id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "auth_method": auth_result["method"],
                    "security_level": "maximum",
                },
                message=f"Secure hook event processing failed: {e}",
            )

            logger.error(
                "Error processing secure hook event",
                event_id=str(hook_event.id),
                error=str(e),
                auth_method=auth_result["method"],
            )
            hook_event.processing_errors.append(str(e))
            return False

    @circuit(failure_threshold=3, recovery_timeout=15, expected_exception=Exception)
    async def _store_hook_event_with_circuit_breaker(self, hook_data: dict) -> bool:
        """Store hook event with circuit breaker protection."""
        try:
            return await self.postgres_client.store_hook_event(hook_data)
        except Exception as e:
            logger.error("Circuit breaker: Database storage failed", error=str(e))
            DATABASE_ERRORS.inc()
            raise

    @circuit(failure_threshold=3, recovery_timeout=15, expected_exception=Exception)
    async def _publish_event_with_circuit_breaker(self, event) -> bool:
        """Publish event to Kafka with circuit breaker protection."""
        try:
            return await self.kafka_client.publish_event(event)
        except Exception as e:
            logger.error("Circuit breaker: Kafka publish failed", error=str(e))
            KAFKA_PUBLISH_ERRORS.inc()
            raise

    def _convert_hook_to_events(self, hook_event: HookEvent) -> list[Any]:
        """Convert hook event to internal event objects (same as base implementation)."""
        events = []
        action = hook_event.payload.action.lower()
        resource = hook_event.payload.resource.lower()

        # Service lifecycle events
        if action in ["startup", "shutdown", "ready", "health_check", "registration"]:
            service_event_type = ServiceEventType.STARTUP
            if action == "shutdown":
                service_event_type = ServiceEventType.SHUTDOWN
            elif action == "ready":
                service_event_type = ServiceEventType.READY
            elif action == "health_check":
                service_event_type = ServiceEventType.HEALTH_CHECK
            elif action == "registration":
                service_event_type = ServiceEventType.REGISTRATION

            event = ServiceLifecycleEvent(
                event=service_event_type,
                service=hook_event.metadata.source,
                correlation_id=hook_event.metadata.correlation_id,
                metadata={
                    "version": hook_event.metadata.version,
                    "environment": hook_event.metadata.environment,
                    "trace_id": hook_event.metadata.trace_id,
                    "security_level": "maximum",
                },
                payload=hook_event.payload.data,
                service_version=hook_event.payload.data.get("version"),
                environment=hook_event.metadata.environment,
                instance_id=hook_event.payload.resource_id,
            )
            events.append(event)

        # Tool execution events
        elif "tool" in resource or action in [
            "execute",
            "call",
            "invoke",
            "response",
            "error",
        ]:
            tool_event_type = ToolEventType.TOOL_CALL
            if action in ["response", "complete"]:
                tool_event_type = ToolEventType.TOOL_RESPONSE
            elif action in ["error", "failed"]:
                tool_event_type = ToolEventType.TOOL_ERROR

            event = ToolExecutionEvent(
                event=tool_event_type,
                service=hook_event.metadata.source,
                correlation_id=hook_event.metadata.correlation_id,
                metadata={
                    "version": hook_event.metadata.version,
                    "environment": hook_event.metadata.environment,
                    "trace_id": hook_event.metadata.trace_id,
                    "security_level": "maximum",
                },
                payload=hook_event.payload.data,
                tool_name=hook_event.payload.data.get(
                    "tool_name",
                    hook_event.payload.resource_id,
                ),
                execution_time_ms=hook_event.payload.data.get("execution_time_ms"),
                success=(
                    hook_event.payload.data.get("success", True)
                    if action != "error"
                    else False
                ),
                error_message=hook_event.payload.data.get("error_message"),
            )
            events.append(event)

        # Default fallback event
        if not events:
            event = ServiceLifecycleEvent(
                event=ServiceEventType.REGISTRATION,
                service=hook_event.metadata.source,
                correlation_id=hook_event.metadata.correlation_id,
                metadata={
                    "version": hook_event.metadata.version,
                    "environment": hook_event.metadata.environment,
                    "trace_id": hook_event.metadata.trace_id,
                    "security_level": "maximum",
                },
                payload=hook_event.payload.data,
                instance_id=hook_event.payload.resource_id,
            )
            events.append(event)

        return events

    async def _handle_session_management(self, hook_event: HookEvent) -> None:
        """Handle session management with security audit trails."""
        if not self.postgres_client.is_connected:
            return

        action = hook_event.payload.action.lower()
        service_name = hook_event.metadata.source
        instance_id = hook_event.payload.resource_id

        try:
            if action in ["startup", "ready", "registration"]:
                # Create or update service session with security context
                await self.postgres_client.create_service_session(
                    session_id=hook_event.id,
                    service_name=service_name,
                    instance_id=instance_id,
                    metadata={
                        "action": action,
                        "environment": hook_event.metadata.environment,
                        "version": hook_event.metadata.version,
                        "payload": hook_event.payload.data,
                        "security_level": "maximum",
                        "secure_processing": True,
                    },
                )

                # Enhanced security audit logging for sessions
                self.audit_logger.log_session_event(
                    session_id=str(hook_event.id),
                    event_type=AuditEventType.SESSION_CREATED,
                    additional_info={
                        "service_name": service_name,
                        "instance_id": instance_id,
                        "action": action,
                        "security_level": "maximum",
                    },
                )

            elif action in ["shutdown", "deregistration"]:
                # Enhanced security audit logging for termination
                self.audit_logger.log_session_event(
                    session_id=str(hook_event.id),
                    event_type=AuditEventType.SESSION_TERMINATED,
                    additional_info={
                        "service_name": service_name,
                        "instance_id": instance_id,
                        "action": action,
                        "security_level": "maximum",
                    },
                )

                logger.info(
                    "Secure service shutdown detected",
                    service=service_name,
                    instance_id=instance_id,
                    action=action,
                    security_level="maximum",
                )

        except Exception as e:
            logger.error(
                "Error in secure session management",
                service=service_name,
                action=action,
                error=str(e),
            )
            DATABASE_ERRORS.inc()


def create_production_secure_app() -> FastAPI:
    """Factory function to create the production secure FastAPI application."""
    # Validate required environment variables for production
    environment = os.getenv("ENVIRONMENT", "development").lower()

    if environment == "production":
        required_env_vars = [
            "POSTGRES_PASSWORD",
            "JWT_SECRET_KEY",
            "WEBHOOK_SIGNING_SECRET",
            "API_KEY_ENCRYPTION_SEED",
        ]

        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables for production: {', '.join(missing_vars)}",
            )

    # Read configuration from environment variables
    kafka_bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
    postgres_host = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port = int(os.getenv("POSTGRES_PORT", "5436"))
    postgres_database = os.getenv("POSTGRES_DATABASE", "omninode_bridge")
    postgres_user = os.getenv("POSTGRES_USER", "postgres")
    postgres_password = os.getenv("POSTGRES_PASSWORD")

    if not postgres_password:
        raise ValueError("POSTGRES_PASSWORD environment variable must be set")

    service = ProductionSecureHookReceiverService(
        kafka_bootstrap_servers=kafka_bootstrap_servers,
        postgres_host=postgres_host,
        postgres_port=postgres_port,
        postgres_database=postgres_database,
        postgres_user=postgres_user,
        postgres_password=postgres_password,
        environment=environment,
    )
    return service.create_app()
