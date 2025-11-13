"""Request correlation middleware for tracking requests across services."""

import uuid
from collections.abc import Callable
from contextvars import ContextVar

import structlog
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Context variables for request correlation
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")
parent_span_id_var: ContextVar[str] = ContextVar("parent_span_id", default="")

# Standard headers for request correlation
REQUEST_ID_HEADER = "X-Request-ID"
TRACE_ID_HEADER = "X-Trace-ID"
PARENT_SPAN_ID_HEADER = "X-Parent-Span-ID"
CORRELATION_ID_HEADER = "X-Correlation-ID"

logger = structlog.get_logger(__name__)


class RequestCorrelationMiddleware(BaseHTTPMiddleware):
    """Middleware to track request correlation across services."""

    def __init__(
        self,
        app: FastAPI,
        service_name: str,
        generate_request_id: bool = True,
        log_requests: bool = True,
        propagate_headers: bool = True,
    ):
        """Initialize request correlation middleware.

        Args:
            app: FastAPI application instance
            service_name: Name of the service for correlation context
            generate_request_id: Generate new request ID if not present
            log_requests: Log incoming requests with correlation info
            propagate_headers: Add correlation headers to responses
        """
        super().__init__(app)
        self.service_name = service_name
        self.generate_request_id = generate_request_id
        self.log_requests = log_requests
        self.propagate_headers = propagate_headers

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with correlation tracking."""
        # Extract or generate correlation IDs
        request_id = self._extract_or_generate_request_id(request)
        trace_id = self._extract_or_generate_trace_id(request)
        parent_span_id = request.headers.get(PARENT_SPAN_ID_HEADER, "")

        # Set context variables
        request_id_var.set(request_id)
        trace_id_var.set(trace_id)
        parent_span_id_var.set(parent_span_id)

        # Add correlation info to request state
        request.state.request_id = request_id
        request.state.trace_id = trace_id
        request.state.parent_span_id = parent_span_id
        request.state.service_name = self.service_name

        # Log incoming request
        if self.log_requests:
            logger.info(
                "Incoming request",
                request_id=request_id,
                trace_id=trace_id,
                parent_span_id=parent_span_id,
                method=request.method,
                path=request.url.path,
                client_host=request.client.host if request.client else None,
                user_agent=request.headers.get("User-Agent"),
                service_name=self.service_name,
            )

        try:
            # Process request
            response = await call_next(request)

            # Add correlation headers to response
            if self.propagate_headers:
                response.headers[REQUEST_ID_HEADER] = request_id
                response.headers[TRACE_ID_HEADER] = trace_id
                if parent_span_id:
                    response.headers[PARENT_SPAN_ID_HEADER] = parent_span_id
                response.headers[CORRELATION_ID_HEADER] = f"{trace_id}:{request_id}"

            # Log response
            if self.log_requests:
                logger.info(
                    "Request completed",
                    request_id=request_id,
                    trace_id=trace_id,
                    status_code=response.status_code,
                    method=request.method,
                    path=request.url.path,
                    service_name=self.service_name,
                )

            return response

        except Exception as e:
            # Log error with correlation context
            logger.error(
                "Request failed",
                request_id=request_id,
                trace_id=trace_id,
                method=request.method,
                path=request.url.path,
                error=str(e),
                error_type=type(e).__name__,
                service_name=self.service_name,
            )
            raise

    def _extract_or_generate_request_id(self, request: Request) -> str:
        """Extract request ID from headers or generate new one."""
        request_id = request.headers.get(REQUEST_ID_HEADER)
        if not request_id and self.generate_request_id:
            request_id = str(uuid.uuid4())
        return request_id or ""

    def _extract_or_generate_trace_id(self, request: Request) -> str:
        """Extract trace ID from headers or generate new one."""
        trace_id = request.headers.get(TRACE_ID_HEADER)
        if not trace_id:
            # Try to extract from correlation header
            correlation_header = request.headers.get(CORRELATION_ID_HEADER)
            if correlation_header and ":" in correlation_header:
                trace_id = correlation_header.split(":")[0]
            elif self.generate_request_id:
                trace_id = str(uuid.uuid4())
        return trace_id or ""


def get_request_id() -> str:
    """Get current request ID from context."""
    return request_id_var.get("")


def get_trace_id() -> str:
    """Get current trace ID from context."""
    return trace_id_var.get("")


def get_parent_span_id() -> str:
    """Get current parent span ID from context."""
    return parent_span_id_var.get("")


def get_correlation_context() -> dict[str, str]:
    """Get all correlation context as dictionary."""
    return {
        "request_id": get_request_id(),
        "trace_id": get_trace_id(),
        "parent_span_id": get_parent_span_id(),
    }


def configure_request_correlation_logging() -> None:
    """Configure structlog to include correlation IDs in all log messages."""

    def add_correlation_ids(logger, method_name, event_dict):
        """Add correlation IDs to log messages."""
        correlation_context = get_correlation_context()
        for key, value in correlation_context.items():
            if value:
                event_dict[key] = value
        return event_dict

    # Add the processor to structlog configuration
    structlog.configure(
        processors=[
            add_correlation_ids,
            *structlog.get_config()["processors"],
        ]
    )


def add_request_correlation_middleware(
    app: FastAPI,
    service_name: str,
    generate_request_id: bool = True,
    log_requests: bool = True,
    propagate_headers: bool = True,
) -> None:
    """Add request correlation middleware to FastAPI app.

    Args:
        app: FastAPI application instance
        service_name: Name of the service for correlation context
        generate_request_id: Generate new request ID if not present
        log_requests: Log incoming requests with correlation info
        propagate_headers: Add correlation headers to responses
    """
    app.add_middleware(
        RequestCorrelationMiddleware,
        service_name=service_name,
        generate_request_id=generate_request_id,
        log_requests=log_requests,
        propagate_headers=propagate_headers,
    )

    # Configure logging to include correlation IDs
    configure_request_correlation_logging()


# HTTP client utilities for outbound requests
async def add_correlation_headers(headers: dict[str, str]) -> dict[str, str]:
    """Add correlation headers to outbound HTTP requests.

    Args:
        headers: Existing headers dictionary

    Returns:
        Updated headers with correlation IDs
    """
    correlation_context = get_correlation_context()

    if correlation_context["request_id"]:
        headers[REQUEST_ID_HEADER] = correlation_context["request_id"]
    if correlation_context["trace_id"]:
        headers[TRACE_ID_HEADER] = correlation_context["trace_id"]
    if correlation_context["request_id"] and correlation_context["trace_id"]:
        headers[CORRELATION_ID_HEADER] = (
            f"{correlation_context['trace_id']}:{correlation_context['request_id']}"
        )

    return headers
