"""Enhanced HookReceiver FastAPI service with comprehensive security."""

import os
import time
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID, uuid4

import structlog
from circuitbreaker import circuit
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
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
from ..security import (
    AuditEventType,
    AuditSeverity,
    EndpointSecurity,
    SecureHookPayload,
    create_auth_handler,
    get_audit_logger,
    get_rate_limiter,
    get_security_validator,
    security_rate_limit,
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


class EnhancedHookReceiverService:
    """Enhanced FastAPI service for receiving and processing webhook events with comprehensive security."""

    def __init__(
        self,
        kafka_bootstrap_servers: str = None,
        postgres_host: str = None,
        postgres_port: int = None,
        postgres_database: str = None,
        postgres_user: str = None,
        postgres_password: str = None,
    ):
        """Initialize Enhanced HookReceiver service."""
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

        # Security components
        self.api_key = os.getenv("API_KEY")
        if not self.api_key:
            raise ValueError(
                "API_KEY environment variable is required for authentication"
            )
        self.audit_logger = get_audit_logger("hook_receiver", "0.2.0")
        self.rate_limiter = get_rate_limiter()
        self.security_validator = get_security_validator()
        self.auth_handler = create_auth_handler(self.api_key, "hook_receiver")

        self.app: FastAPI | None = None

    async def startup(self) -> None:
        """Initialize connections on startup."""
        try:
            # Log service startup
            self.audit_logger.log_event(
                event_type=AuditEventType.SERVICE_STARTUP,
                severity=AuditSeverity.LOW,
                message="Enhanced HookReceiver service starting up",
            )

            # Connect to Kafka
            await self.kafka_client.connect()
            logger.info("Kafka client connected")

            # Connect to PostgreSQL
            await self.postgres_client.connect()
            logger.info("PostgreSQL client connected")

            logger.info("Enhanced HookReceiver service started successfully")

        except Exception as e:
            logger.error("Failed to start Enhanced HookReceiver service", error=str(e))
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
                message="Enhanced HookReceiver service shutdown completed",
            )

            logger.info("Enhanced HookReceiver service shutdown completed")

        except Exception as e:
            logger.error("Error during service shutdown", error=str(e))

    def create_app(self) -> FastAPI:
        """Create and configure FastAPI application with enhanced security."""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            await self.startup()
            yield
            # Shutdown
            await self.shutdown()

        app = FastAPI(
            title="OmniNode Bridge - Enhanced HookReceiver Service",
            description="Intelligent service lifecycle management through webhook processing with comprehensive security",
            version="0.2.0",
            docs_url="/docs",
            redoc_url="/redoc",
            lifespan=lifespan,
        )

        # Setup comprehensive security middleware
        setup_security_middleware(
            app=app,
            service_name="hook_receiver",
            api_key=self.api_key,
            enable_audit_logging=True,
            enable_request_validation=True,
            max_request_size=1024 * 1024,  # 1MB limit for hooks
        )

        # Add CORS middleware
        allowed_origins = os.getenv(
            "CORS_ALLOWED_ORIGINS",
            "http://localhost:3000,http://localhost:8000",
        ).split(",")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[origin.strip() for origin in allowed_origins],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
        )

        # Enhanced authentication dependency
        async def verify_api_key(
            request: Request,
            authorization: HTTPAuthorizationCredentials | None = Depends(
                HTTPBearer(auto_error=False)
            ),
            x_api_key: str | None = Header(None),
        ) -> bool:
            """Enhanced API key verification with audit logging."""
            return await self.auth_handler.verify_api_key(
                request,
                authorization,
                x_api_key,
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
                    "detail": "Rate limit exceeded",
                    "endpoint": endpoint,
                    "retry_after": "60 seconds",
                    "request_id": getattr(request.state, "request_id", "unknown"),
                },
            )

        app.add_exception_handler(RateLimitExceeded, enhanced_rate_limit_handler)

        # Enhanced Hook Processing Endpoint
        @app.post("/hooks", response_model=HookResponse)
        @security_rate_limit("hook_processing", EndpointSecurity.AUTHENTICATED)
        async def receive_hook(
            request: Request,
            _: bool = Depends(verify_api_key),
        ) -> HookResponse:
            """Receive and process webhook events with enhanced security validation."""
            start_time = time.time()
            request_id = getattr(request.state, "request_id", str(uuid4()))

            try:
                # Parse and validate request body
                body = await request.json()

                # Enhanced input validation using secure models
                try:
                    # Validate hook payload structure
                    secure_payload = SecureHookPayload(**body)

                    # Additional security validation
                    self.security_validator.validate_input_safety(body, "hook_payload")

                except ValueError as e:
                    # Log input validation failure
                    self.audit_logger.log_input_validation_failure(
                        field="hook_payload",
                        value_type="json",
                        validation_error=str(e),
                        request=request,
                    )
                    SECURITY_VIOLATIONS.labels(violation_type="input_validation").inc()

                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid hook payload: {e}",
                    )

                # Extract metadata from request
                metadata = HookMetadata(
                    source=secure_payload.resource,
                    version=body.get("version", "1.0.0"),
                    environment=body.get("environment", "development"),
                    correlation_id=body.get("correlation_id"),
                    trace_id=request.headers.get("X-Trace-ID"),
                    user_agent=request.headers.get("User-Agent"),
                    source_ip=request.client.host if request.client else None,
                )

                # Create validated hook event
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

                # Log hook received
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
                    },
                    message=f"Hook received: {secure_payload.action} on {secure_payload.resource}",
                )

                # Update metrics
                HOOK_EVENTS_TOTAL.labels(
                    source=metadata.source,
                    action=secure_payload.action,
                ).inc()

                # Process the hook event
                success = await self._process_hook_event(hook_event)

                processing_time = (time.time() - start_time) * 1000
                HOOK_PROCESSING_TIME.observe(processing_time / 1000)

                # Record metrics in database
                if self.postgres_client.is_connected:
                    await self.postgres_client.record_event_metrics(
                        event_id=hook_event.id,
                        processing_time_ms=processing_time,
                        kafka_publish_success=success,
                        error_message=None if success else "Processing failed",
                    )

                # Log processing result
                self.audit_logger.log_event(
                    event_type=(
                        AuditEventType.WORKFLOW_EXECUTION_COMPLETE
                        if success
                        else AuditEventType.WORKFLOW_EXECUTION_FAILURE
                    ),
                    severity=(AuditSeverity.LOW if success else AuditSeverity.MEDIUM),
                    request=request,
                    additional_data={
                        "hook_id": str(hook_event.id),
                        "success": success,
                        "processing_time_ms": processing_time,
                        "request_id": request_id,
                    },
                    message=f"Hook processing {'completed' if success else 'failed'}",
                )

                if success:
                    return HookResponse(
                        success=True,
                        message="Hook processed successfully",
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

                # Log unexpected error
                self.audit_logger.log_event(
                    event_type=AuditEventType.SECURITY_VIOLATION,
                    severity=AuditSeverity.HIGH,
                    request=request,
                    additional_data={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "processing_time_ms": processing_time,
                        "request_id": request_id,
                    },
                    message=f"Unexpected hook processing error: {e}",
                )

                return HookResponse(
                    success=False,
                    message="Hook processing error",
                    event_id=uuid4(),
                    processing_time_ms=processing_time,
                    errors=[str(e)],
                )

        # Enhanced Health Check
        @app.get("/health")
        @security_rate_limit("health", EndpointSecurity.PUBLIC)
        async def health_check() -> dict[str, Any]:
            """Comprehensive health check endpoint with security monitoring."""
            health_status = {
                "status": "healthy",
                "timestamp": time.time(),
                "version": "0.2.0",
                "security_features": {
                    "audit_logging": True,
                    "rate_limiting": True,
                    "input_validation": True,
                    "authentication": True,
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

        # Secure Metrics Endpoint
        @app.get("/metrics")
        @security_rate_limit("metrics", EndpointSecurity.AUTHENTICATED)
        async def get_metrics(_: bool = Depends(verify_api_key)) -> Response:
            """Prometheus metrics endpoint with authentication."""
            return Response(generate_latest(), media_type="text/plain")

        # Enhanced Session Management
        @app.get("/sessions")
        @security_rate_limit("session_query", EndpointSecurity.SENSITIVE)
        async def get_active_sessions(
            request: Request,
            _: bool = Depends(verify_api_key),
        ) -> dict[str, Any]:
            """Get active service sessions with enhanced security."""
            if not self.postgres_client.is_connected:
                raise HTTPException(status_code=503, detail="Database not available")

            # Log session query
            self.audit_logger.log_session_event(
                session_id="query_all",
                event_type=AuditEventType.SESSION_CREATED,
                request=request,
                additional_info={"action": "query_active_sessions"},
            )

            sessions = await self.postgres_client.get_active_sessions()
            return {
                "active_sessions": sessions,
                "count": len(sessions),
                "query_timestamp": time.time(),
            }

        @app.post("/sessions/{session_id}/end")
        @security_rate_limit("session_terminate", EndpointSecurity.SENSITIVE)
        async def end_session(
            session_id: UUID,
            request: Request,
            _: bool = Depends(verify_api_key),
        ) -> dict[str, Any]:
            """End a service session with enhanced security and logging."""
            if not self.postgres_client.is_connected:
                raise HTTPException(status_code=503, detail="Database not available")

            # Log session termination attempt
            self.audit_logger.log_session_event(
                session_id=str(session_id),
                event_type=AuditEventType.SESSION_TERMINATED,
                request=request,
                additional_info={"action": "terminate_session"},
            )

            success = await self.postgres_client.end_service_session(session_id)

            if success:
                return {
                    "message": f"Session {session_id} ended successfully",
                    "timestamp": time.time(),
                }
            raise HTTPException(status_code=404, detail="Session not found")

        # Service Information
        @app.get("/")
        @security_rate_limit("docs", EndpointSecurity.PUBLIC)
        async def root() -> dict[str, str]:
            """Root endpoint with enhanced service information."""
            return {
                "service": "OmniNode Bridge - Enhanced HookReceiver",
                "version": "0.2.0",
                "status": "operational",
                "security_features": "comprehensive",
                "docs": "/docs",
            }

        self.app = app
        return app

    @circuit(failure_threshold=5, recovery_timeout=30, expected_exception=Exception)
    async def _process_hook_event(self, hook_event: HookEvent) -> bool:
        """Process a hook event with enhanced error handling and security logging."""
        try:
            # Store hook event in database for persistence and audit trail
            if self.postgres_client.is_connected:
                hook_data = hook_event.model_dump()
                await self._store_hook_event_with_circuit_breaker(hook_data)

            # Convert hook to internal event(s)
            internal_events = self._convert_hook_to_events(hook_event)

            # Publish events to Kafka
            all_published = True
            for event in internal_events:
                if self.kafka_client.is_connected:
                    published = await self._publish_event_with_circuit_breaker(event)
                    if not published:
                        all_published = False
                        KAFKA_PUBLISH_ERRORS.inc()
                        hook_event.processing_errors.append(
                            f"Failed to publish {event.type} event",
                        )

            # Handle session management for service lifecycle events
            await self._handle_session_management(hook_event)

            # Mark as processed if everything succeeded
            hook_event.processed = all_published

            logger.info(
                "Processed hook event",
                event_id=str(hook_event.id),
                source=hook_event.metadata.source,
                action=hook_event.payload.action,
                success=all_published,
            )

            return all_published

        except Exception as e:
            # Log processing error
            self.audit_logger.log_event(
                event_type=AuditEventType.WORKFLOW_EXECUTION_FAILURE,
                severity=AuditSeverity.HIGH,
                additional_data={
                    "hook_id": str(hook_event.id),
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                message=f"Hook event processing failed: {e}",
            )

            logger.error(
                "Error processing hook event",
                event_id=str(hook_event.id),
                error=str(e),
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
        """Convert hook event to internal event objects."""
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

        # If no specific event type matched, create a generic service lifecycle event
        if not events:
            event = ServiceLifecycleEvent(
                event=ServiceEventType.REGISTRATION,
                service=hook_event.metadata.source,
                correlation_id=hook_event.metadata.correlation_id,
                metadata={
                    "version": hook_event.metadata.version,
                    "environment": hook_event.metadata.environment,
                    "trace_id": hook_event.metadata.trace_id,
                },
                payload=hook_event.payload.data,
                instance_id=hook_event.payload.resource_id,
            )
            events.append(event)

        return events

    async def _handle_session_management(self, hook_event: HookEvent) -> None:
        """Handle session management based on hook events."""
        if not self.postgres_client.is_connected:
            return

        action = hook_event.payload.action.lower()
        service_name = hook_event.metadata.source
        instance_id = hook_event.payload.resource_id

        try:
            if action in ["startup", "ready", "registration"]:
                # Create or update service session
                await self.postgres_client.create_service_session(
                    session_id=hook_event.id,
                    service_name=service_name,
                    instance_id=instance_id,
                    metadata={
                        "action": action,
                        "environment": hook_event.metadata.environment,
                        "version": hook_event.metadata.version,
                        "payload": hook_event.payload.data,
                    },
                )

                # Log session creation
                self.audit_logger.log_session_event(
                    session_id=str(hook_event.id),
                    event_type=AuditEventType.SESSION_CREATED,
                    additional_info={
                        "service_name": service_name,
                        "instance_id": instance_id,
                        "action": action,
                    },
                )

            elif action in ["shutdown", "deregistration"]:
                # Log session termination
                self.audit_logger.log_session_event(
                    session_id=str(hook_event.id),
                    event_type=AuditEventType.SESSION_TERMINATED,
                    additional_info={
                        "service_name": service_name,
                        "instance_id": instance_id,
                        "action": action,
                    },
                )

                logger.info(
                    "Service shutdown detected",
                    service=service_name,
                    instance_id=instance_id,
                    action=action,
                )

        except Exception as e:
            logger.error(
                "Error in session management",
                service=service_name,
                action=action,
                error=str(e),
            )
            DATABASE_ERRORS.inc()


def create_enhanced_app() -> FastAPI:
    """Factory function to create the enhanced FastAPI application."""
    # Read configuration from environment variables
    kafka_bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
    postgres_host = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port = int(os.getenv("POSTGRES_PORT", "5436"))
    postgres_database = os.getenv("POSTGRES_DATABASE", "omninode_bridge")
    postgres_user = os.getenv("POSTGRES_USER", "postgres")
    postgres_password = os.getenv("POSTGRES_PASSWORD")

    if not postgres_password:
        raise ValueError("POSTGRES_PASSWORD environment variable must be set")

    service = EnhancedHookReceiverService(
        kafka_bootstrap_servers=kafka_bootstrap_servers,
        postgres_host=postgres_host,
        postgres_port=postgres_port,
        postgres_database=postgres_database,
        postgres_user=postgres_user,
        postgres_password=postgres_password,
    )
    return service.create_app()
