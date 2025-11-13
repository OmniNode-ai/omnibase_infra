"""HookReceiver FastAPI service for OmniNode Bridge."""

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
from starlette.responses import Response

from ..models.events import (
    ServiceEventType,
    ServiceLifecycleEvent,
    ToolEventType,
    ToolExecutionEvent,
)
from ..models.hooks import HookEvent, HookMetadata, HookPayload, HookResponse
from ..security.audit_logger import AuditEventType, AuditSeverity, get_audit_logger
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

# Initialize audit logger
audit_logger = get_audit_logger("hook_receiver", "0.1.0")

# Rate limiting configuration - will be initialized per service instance

# Prometheus metrics - with collision prevention
try:
    from prometheus_client import REGISTRY

    # Check if metrics already exist to prevent collisions
    def get_or_create_counter(name, description, labels=None):
        """Get existing counter or create new one if it doesn't exist."""
        try:
            for collector in REGISTRY._collector_to_names:
                if hasattr(collector, "_name") and collector._name == name:
                    return collector
        except (AttributeError, TypeError):
            # Registry access failed due to missing attributes or type errors
            pass
        return Counter(name, description, labels or [])

    def get_or_create_histogram(name, description):
        """Get existing histogram or create new one if it doesn't exist."""
        try:
            for collector in REGISTRY._collector_to_names:
                if hasattr(collector, "_name") and collector._name == name:
                    return collector
        except (AttributeError, TypeError):
            # Registry access failed due to missing attributes or type errors
            pass
        return Histogram(name, description)

    HOOK_EVENTS_TOTAL = get_or_create_counter(
        "hook_events_basic_total",  # Made unique
        "Total hook events received (basic receiver)",
        ["source", "action"],
    )
    HOOK_PROCESSING_TIME = get_or_create_histogram(
        "hook_processing_basic_seconds",  # Made unique
        "Hook processing time (basic receiver)",
    )
    KAFKA_PUBLISH_ERRORS = get_or_create_counter(
        "kafka_publish_basic_errors_total",  # Made unique
        "Kafka publish errors (basic receiver)",
    )
    DATABASE_ERRORS = get_or_create_counter(
        "database_basic_errors_total",  # Made unique
        "Database operation errors (basic receiver)",
    )
    RATE_LIMIT_EXCEEDED = get_or_create_counter(
        "rate_limit_basic_exceeded_total",  # Made unique
        "Rate limit exceeded count (basic receiver)",
        ["endpoint"],
    )
except Exception as e:
    # Fallback if Prometheus setup fails
    logger.warning(f"Failed to setup Prometheus metrics: {e}")

    class MockMetric:
        def inc(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

    HOOK_EVENTS_TOTAL = MockMetric()
    HOOK_PROCESSING_TIME = MockMetric()
    KAFKA_PUBLISH_ERRORS = MockMetric()
    DATABASE_ERRORS = MockMetric()
    RATE_LIMIT_EXCEEDED = MockMetric()


class HookReceiverService:
    """FastAPI service for receiving and processing webhook events."""

    def __init__(
        self,
        kafka_bootstrap_servers: str = None,
        postgres_host: str = None,
        postgres_port: int = None,
        postgres_database: str = None,
        postgres_user: str = None,
        postgres_password: str = None,
    ):
        """Initialize HookReceiver service.

        Args:
            kafka_bootstrap_servers: Kafka bootstrap servers
            postgres_host: PostgreSQL host
            postgres_port: PostgreSQL port
            postgres_database: PostgreSQL database
            postgres_user: PostgreSQL user
            postgres_password: PostgreSQL password
        """
        # Load configuration from environment variables
        from omninode_bridge.config.environment_config import (
            DatabaseConfig,
            KafkaConfig,
        )

        # Get database configuration from environment
        db_config = DatabaseConfig()
        pg_host = postgres_host or db_config.host
        pg_port = postgres_port or db_config.port
        pg_database = postgres_database or db_config.database
        pg_user = postgres_user or db_config.user
        pg_password = postgres_password or db_config.password

        # Get Kafka configuration from environment
        kafka_config = KafkaConfig()
        kafka_servers = kafka_bootstrap_servers or kafka_config.bootstrap_servers

        if not pg_password:
            raise ValueError(
                "Database password must be provided via secure configuration or postgres_password parameter",
            )

        self.kafka_client = KafkaClient(kafka_servers)
        self.postgres_client = PostgresClient(
            host=pg_host,
            port=pg_port,
            database=pg_database,
            user=pg_user,
            password=pg_password,
        )

        # Initialize comprehensive rate limiting service
        from omninode_bridge.services.rate_limiting_service import RateLimitingService

        self.rate_limiting_service = RateLimitingService(
            # Uses default production-ready configuration
        )
        self.app: FastAPI | None = None

    async def _on_rate_limit_exceeded(
        self, request: Request, limit_type: str, limit_value: str
    ) -> None:
        """Handle rate limit exceeded events with audit logging and metrics.

        Args:
            request: FastAPI request object
            limit_type: Type of rate limit exceeded (e.g., 'per_ip', 'per_user', 'global')
            limit_value: The limit that was exceeded (e.g., '100/minute')
        """
        try:
            # Log rate limit exceeded event for audit
            audit_logger.log_event(
                event_type=AuditEventType.RATE_LIMIT_EXCEEDED,
                severity=AuditSeverity.MEDIUM,
                request=request,
                additional_data={
                    "endpoint": request.url.path,
                    "limit_type": limit_type,
                    "limit_value": limit_value,
                    "user_agent": request.headers.get("User-Agent", "unknown"),
                    "source_ip": request.client.host if request.client else "unknown",
                },
                message=f"Rate limit exceeded: {limit_type} limit of {limit_value} for {request.url.path}",
            )

            # Update Prometheus counter
            RATE_LIMIT_EXCEEDED.labels(endpoint=request.url.path).inc()

            logger.warning(
                f"Rate limit exceeded for {request.url.path}",
                extra={
                    "endpoint": request.url.path,
                    "limit_type": limit_type,
                    "limit_value": limit_value,
                    "source_ip": request.client.host if request.client else "unknown",
                },
            )

        except (AttributeError, KeyError, ValueError) as e:
            # Expected errors during rate limit event logging
            logger.warning(f"Error handling rate limit exceeded event: {e}")
        except Exception as e:
            # Unexpected errors - log with full context
            logger.error(
                f"Unexpected error handling rate limit exceeded event: {e}",
                exc_info=True,
            )

    async def startup(self) -> None:
        """Initialize connections on startup."""
        try:
            # Connect to Kafka
            await self.kafka_client.connect()
            logger.info("Kafka client connected")

            # Connect to PostgreSQL
            await self.postgres_client.connect()
            logger.info("PostgreSQL client connected")

            logger.info("HookReceiver service started successfully")

        except Exception as e:
            logger.error("Failed to start HookReceiver service", error=str(e))
            raise

    async def shutdown(self) -> None:
        """Clean up connections on shutdown."""
        shutdown_errors = []

        # Attempt to disconnect Kafka client
        try:
            await self.kafka_client.disconnect()
        except Exception as e:
            shutdown_errors.append(f"Kafka shutdown error: {e}")
            logger.error("Error during service shutdown", error=str(e))

        # Attempt to disconnect PostgreSQL client
        try:
            await self.postgres_client.disconnect()
        except Exception as e:
            shutdown_errors.append(f"PostgreSQL shutdown error: {e}")
            logger.error("Error during service shutdown", error=str(e))

        if shutdown_errors:
            logger.error("Shutdown completed with errors", errors=shutdown_errors)
        else:
            logger.info("HookReceiver service shutdown completed")

    def create_app(self) -> FastAPI:
        """Create and configure FastAPI application."""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            await self.startup()
            yield
            # Shutdown
            await self.shutdown()

        app = FastAPI(
            title="OmniNode Bridge - HookReceiver Service",
            description="Intelligent service lifecycle management through webhook processing",
            version="0.1.0",
            docs_url="/docs",
            redoc_url="/redoc",
            lifespan=lifespan,
        )

        # Store service instance in app state
        app.state.hook_receiver_service = self

        # Add comprehensive rate limiting middleware
        rate_limit_middleware = self.rate_limiting_service.create_fastapi_middleware()
        app.middleware("http")(rate_limit_middleware)

        # Add CORS middleware with environment-based configuration
        from omninode_bridge.config.environment_config import SecurityConfig

        security_config = SecurityConfig()

        app.add_middleware(
            CORSMiddleware,
            allow_origins=security_config.cors_allowed_origins,
            allow_credentials=security_config.cors_allow_credentials,
            allow_methods=security_config.cors_allow_methods,
            allow_headers=security_config.cors_allow_headers,
        )

        # Authentication setup
        security = HTTPBearer(auto_error=False)
        api_key = os.getenv("API_KEY")
        if not api_key:
            logger.error(
                "API_KEY environment variable not set - authentication will fail",
            )
            api_key = "MISSING_API_KEY"  # Will cause all authentication to fail

        async def verify_api_key(
            request: Request,
            authorization: HTTPAuthorizationCredentials | None = Depends(security),
            x_api_key: str | None = Header(None),
        ) -> bool:
            """Verify API key from Authorization header or X-API-Key header."""
            provided_key = None
            auth_method = None

            if authorization and authorization.scheme.lower() == "bearer":
                provided_key = authorization.credentials
                auth_method = "bearer_token"
            elif x_api_key:
                provided_key = x_api_key
                auth_method = "api_key_header"

            if not provided_key:
                # Log missing API key
                audit_logger.log_authentication_failure(
                    reason="Missing API key",
                    auth_method="api_key",
                    request=request,
                )
                raise HTTPException(
                    status_code=401,
                    detail="Invalid or missing API key. Provide via Authorization: Bearer <key> or X-API-Key header",
                )
            elif provided_key != api_key:
                # Log invalid API key
                audit_logger.log_authentication_failure(
                    reason="Invalid API key",
                    auth_method=auth_method,
                    request=request,
                )
                raise HTTPException(
                    status_code=401,
                    detail="Invalid or missing API key. Provide via Authorization: Bearer <key> or X-API-Key header",
                )

            # Log successful authentication
            audit_logger.log_authentication_success(
                auth_method=auth_method,
                request=request,
            )
            return True

        # Hook endpoints
        @app.post("/hooks", response_model=HookResponse)
        async def receive_hook(
            request: Request,
            _: bool = Depends(verify_api_key),
        ) -> HookResponse:
            """Receive and process webhook events."""
            # Get service instance from app state
            service = request.app.state.hook_receiver_service

            start_time = time.time()

            # Parse request body - handle JSON parsing errors specifically
            try:
                body = await request.json()
            except Exception as json_error:
                # JSON parsing failed - return 422 for invalid JSON
                audit_logger.log_input_validation_failure(
                    field="request_body",
                    value_type="json",
                    validation_error=f"Invalid JSON: {json_error!s}",
                    request=request,
                )
                raise HTTPException(status_code=422, detail="Invalid JSON payload")

            try:

                # Extract metadata from request
                metadata = HookMetadata(
                    source=body.get("source", "unknown"),
                    version=body.get("version", "1.0.0"),
                    environment=body.get("environment", "development"),
                    correlation_id=body.get("correlation_id"),
                    trace_id=request.headers.get("X-Trace-ID"),
                    user_agent=request.headers.get("User-Agent"),
                    source_ip=request.client.host if request.client else None,
                )

                # Extract payload
                payload = HookPayload(
                    action=body.get("action", "unknown"),
                    resource=body.get("resource", "unknown"),
                    resource_id=body.get("resource_id", "unknown"),
                    data=body.get("data", {}),
                    previous_state=body.get("previous_state"),
                    current_state=body.get("current_state"),
                )

                # Create hook event
                hook_event = HookEvent(
                    metadata=metadata,
                    payload=payload,
                )

                # Log hook event submission for audit
                audit_logger.log_event(
                    event_type=AuditEventType.WORKFLOW_SUBMISSION,
                    severity=AuditSeverity.LOW,
                    request=request,
                    additional_data={
                        "hook_id": str(hook_event.id),
                        "source": metadata.source,
                        "action": payload.action,
                        "resource": payload.resource,
                        "resource_id": payload.resource_id,
                        "environment": metadata.environment,
                    },
                    message=f"Hook event received: {payload.action} on {payload.resource}",
                )

                # Update metrics
                HOOK_EVENTS_TOTAL.labels(
                    source=metadata.source,
                    action=payload.action,
                ).inc()

                # Process the hook event
                success = await service._process_hook_event(hook_event)

                processing_time = (
                    time.time() - start_time
                ) * 1000  # Convert to milliseconds
                HOOK_PROCESSING_TIME.observe(processing_time / 1000)

                # Record metrics in database
                if service.postgres_client.is_connected:
                    await service.postgres_client.record_event_metrics(
                        event_id=hook_event.id,
                        processing_time_ms=processing_time,
                        kafka_publish_success=success,
                        error_message=None if success else "Processing failed",
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

            except Exception as e:
                processing_time = (time.time() - start_time) * 1000
                logger.error("Error processing hook", error=str(e))

                return HookResponse(
                    success=False,
                    message="Hook processing error",
                    event_id=uuid4(),
                    processing_time_ms=processing_time,
                    errors=[str(e)],
                )

        @app.get("/health")
        async def health_check(request: Request) -> dict[str, Any]:
            """Comprehensive health check endpoint."""
            # Get service instance from app state
            service = request.app.state.hook_receiver_service

            health_status = {
                "status": "healthy",
                "timestamp": time.time(),
                "version": "0.1.0",
                "uptime": time.time(),  # Simplified uptime
                "dependencies": {},
            }

            # Check Kafka health
            kafka_health = await service.kafka_client.health_check()
            health_status["dependencies"]["kafka"] = kafka_health

            # Check PostgreSQL health
            postgres_health = await service.postgres_client.health_check()
            health_status["dependencies"]["postgres"] = postgres_health

            # Overall status
            all_healthy = all(
                dep.get("status") == "healthy"
                for dep in health_status["dependencies"].values()
            )

            health_status["status"] = "healthy" if all_healthy else "degraded"

            status_code = 200 if all_healthy else 503
            return JSONResponse(content=health_status, status_code=status_code)

        @app.get("/metrics")
        async def get_metrics() -> Response:
            """Prometheus metrics endpoint."""
            return Response(generate_latest(), media_type="text/plain")

        @app.get("/sessions")
        async def get_active_sessions(
            request: Request,
            _: bool = Depends(verify_api_key),
        ) -> dict[str, Any]:
            """Get active service sessions."""
            # Get service instance from app state
            service = request.app.state.hook_receiver_service

            if not service.postgres_client.is_connected:
                raise HTTPException(status_code=503, detail="Database not available")

            sessions = await service.postgres_client.get_active_sessions()

            # Log session access for audit
            audit_logger.log_event(
                event_type=AuditEventType.SESSION_CREATED,  # Using as session access event
                severity=AuditSeverity.LOW,
                request=request,
                additional_data={
                    "action": "list_sessions",
                    "session_count": len(sessions),
                },
                message=f"Session list accessed: {len(sessions)} active sessions",
            )

            return {
                "active_sessions": sessions,
                "count": len(sessions),
            }

        @app.post("/sessions/{session_id}/end")
        async def end_session(
            session_id: UUID,
            request: Request,
            _: bool = Depends(verify_api_key),
        ) -> dict[str, Any]:
            """End a service session."""
            # Get service instance from app state
            service = request.app.state.hook_receiver_service

            if not service.postgres_client.is_connected:
                raise HTTPException(status_code=503, detail="Database not available")

            success = await service.postgres_client.end_service_session(session_id)

            # Log session termination for audit
            audit_logger.log_session_event(
                session_id=str(session_id),
                event_type=AuditEventType.SESSION_TERMINATED,
                request=request,
                additional_info={"success": success, "action": "manual_termination"},
            )

            if success:
                return {"message": f"Session {session_id} ended successfully"}
            raise HTTPException(status_code=404, detail="Session not found")

        @app.get("/")
        async def root() -> dict[str, str]:
            """Root endpoint with service information."""
            return {
                "service": "OmniNode Bridge - HookReceiver",
                "version": "0.1.0",
                "status": "operational",
                "docs": "/docs",
            }

        self.app = app
        return app

    @circuit(failure_threshold=5, recovery_timeout=30, expected_exception=Exception)
    async def _process_hook_event(self, hook_event: HookEvent) -> bool:
        """Process a hook event by converting it to internal events and publishing to Kafka.

        Args:
            hook_event: The hook event to process

        Returns:
            True if processing succeeded, False otherwise
        """
        try:
            # Store hook event in database for persistence (non-fatal if fails)
            if self.postgres_client.is_connected:
                try:
                    hook_data = hook_event.model_dump()
                    await self._store_hook_event_with_circuit_breaker(hook_data)
                except Exception as db_error:
                    # Database storage failure is non-fatal - log and continue
                    logger.warning(
                        "Database storage failed, continuing with event processing",
                        event_id=str(hook_event.id),
                        error=str(db_error),
                    )

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
        """Convert hook event to internal event objects.

        Args:
            hook_event: Hook event to convert

        Returns:
            List of internal event objects
        """
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
                event=ServiceEventType.REGISTRATION,  # Generic event type
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
        """Handle session management based on hook events.

        Args:
            hook_event: Hook event to process for session management
        """
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

            elif action in ["shutdown", "deregistration"]:
                # End service session if we can find it by service name and instance
                # For now, just log this - we'd need more sophisticated session tracking
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


def create_app() -> FastAPI:
    """Factory function to create the FastAPI application with secure configuration."""
    # Create service with secure configuration - parameters will be loaded from Vault/secure config
    service = HookReceiverService()
    return service.create_app()
