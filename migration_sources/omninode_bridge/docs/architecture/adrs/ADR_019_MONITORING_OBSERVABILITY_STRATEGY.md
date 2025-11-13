# ADR-019: Monitoring and Observability Strategy

**Status**: Accepted
**Date**: 2024-09-25
**Deciders**: OmniNode Bridge Architecture Team
**Technical Story**: Implementation of comprehensive monitoring and observability for multi-service production architecture

## Context

The multi-service architecture (HookReceiver, ModelMetrics API, WorkflowCoordinator) requires comprehensive monitoring and observability capabilities that can handle:

- **Multi-Service Visibility**: Monitor health, performance, and behavior across all three services
- **End-to-End Tracing**: Track requests and workflows as they flow through the entire system
- **Performance Monitoring**: Real-time metrics collection and performance trend analysis
- **Error Tracking**: Comprehensive error detection, correlation, and root cause analysis
- **Business Metrics**: Track workflow execution success rates, processing times, and throughput
- **Infrastructure Monitoring**: System resources, database performance, and Kafka health
- **Security Monitoring**: Authentication failures, rate limiting, and suspicious activity detection
- **Alerting and Incident Response**: Proactive alerting with intelligent noise reduction
- **Operational Dashboards**: Real-time visibility for operations and development teams

Traditional monitoring approaches often create silos between different types of telemetry data (logs, metrics, traces), making it difficult to correlate issues across services and understand system behavior holistically.

## Decision

We adopt a **Unified Observability Platform** strategy using the three pillars of observability (metrics, logs, traces) with comprehensive correlation and analysis capabilities:

### Observability Architecture

#### 1. Three Pillars Implementation
```python
# Structured logging with correlation
import structlog

logger = structlog.get_logger().bind(
    service="hook_receiver",
    version="0.2.0",
    correlation_id=get_correlation_id(),
    trace_id=get_trace_id()
)

# Metrics collection
from prometheus_client import Counter, Histogram, Gauge, Info

# Service-specific metrics
request_count = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['service', 'method', 'endpoint', 'status']
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['service', 'endpoint']
)

active_workflows = Gauge(
    'active_workflows_total',
    'Number of active workflows',
    ['service', 'status']
)

# Distributed tracing
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

tracer = trace.get_tracer(__name__)
```

#### 2. Service-Specific Monitoring Configuration
```python
# HookReceiver Service - Event processing monitoring
HOOK_RECEIVER_METRICS = {
    "webhook_events_processed": Counter,
    "webhook_processing_duration": Histogram,
    "kafka_publish_success_rate": Gauge,
    "webhook_validation_errors": Counter,
    "concurrent_webhook_processing": Gauge,
}

# ModelMetrics API Service - Analytics monitoring
MODEL_METRICS_MONITORING = {
    "model_inference_requests": Counter,
    "model_response_time": Histogram,
    "model_accuracy_scores": Gauge,
    "batch_processing_queue_size": Gauge,
    "model_error_rate": Counter,
}

# WorkflowCoordinator Service - Orchestration monitoring
WORKFLOW_COORDINATOR_METRICS = {
    "workflows_executed": Counter,
    "workflow_execution_duration": Histogram,
    "workflow_success_rate": Gauge,
    "parallel_workflow_capacity": Gauge,
    "workflow_queue_depth": Gauge,
}
```

### Comprehensive Metrics Collection

#### 1. Application-Level Metrics
```python
class ApplicationMetrics:
    """Comprehensive application metrics collection."""

    def __init__(self, service_name: str):
        self.service_name = service_name

        # Business metrics
        self.business_operations = Counter(
            'business_operations_total',
            'Business operations by type and outcome',
            ['service', 'operation_type', 'outcome']
        )

        # Performance metrics
        self.operation_duration = Histogram(
            'operation_duration_seconds',
            'Operation execution time',
            ['service', 'operation_type']
        )

        # Error tracking
        self.error_count = Counter(
            'errors_total',
            'Total errors by type and severity',
            ['service', 'error_type', 'severity']
        )

        # Resource utilization
        self.resource_utilization = Gauge(
            'resource_utilization_percent',
            'Resource utilization percentage',
            ['service', 'resource_type']
        )

    async def record_operation(
        self,
        operation_type: str,
        duration: float,
        outcome: str,
        additional_labels: dict = None
    ) -> None:
        """Record a business operation with comprehensive metrics."""

        labels = {
            'service': self.service_name,
            'operation_type': operation_type,
            'outcome': outcome
        }
        if additional_labels:
            labels.update(additional_labels)

        # Record business metrics
        self.business_operations.labels(**labels).inc()

        # Record performance metrics
        self.operation_duration.labels(
            service=self.service_name,
            operation_type=operation_type
        ).observe(duration)

        # Log structured event
        logger.info(
            "Operation completed",
            operation_type=operation_type,
            duration_seconds=duration,
            outcome=outcome,
            **additional_labels or {}
        )
```

#### 2. Infrastructure Metrics
```python
class InfrastructureMetrics:
    """Monitor infrastructure components health and performance."""

    def __init__(self):
        # Database metrics
        self.db_connection_pool_size = Gauge(
            'db_connection_pool_size',
            'Database connection pool metrics',
            ['service', 'pool_state']
        )

        self.db_query_duration = Histogram(
            'db_query_duration_seconds',
            'Database query execution time',
            ['service', 'query_type']
        )

        # Kafka metrics
        self.kafka_message_throughput = Counter(
            'kafka_messages_total',
            'Kafka message processing',
            ['service', 'topic', 'outcome']
        )

        self.kafka_consumer_lag = Gauge(
            'kafka_consumer_lag_messages',
            'Kafka consumer lag',
            ['service', 'topic', 'partition']
        )

        # Circuit breaker metrics
        self.circuit_breaker_state = Gauge(
            'circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open, 2=half-open)',
            ['service', 'circuit_name']
        )

    async def update_infrastructure_metrics(self) -> None:
        """Update infrastructure metrics from system state."""

        # Database connection pool monitoring
        if hasattr(self, 'postgres_client') and self.postgres_client:
            pool_stats = await self.postgres_client.get_pool_statistics()

            self.db_connection_pool_size.labels(
                service=self.service_name,
                pool_state='active'
            ).set(pool_stats['active_connections'])

            self.db_connection_pool_size.labels(
                service=self.service_name,
                pool_state='idle'
            ).set(pool_stats['idle_connections'])

        # Kafka health monitoring
        if hasattr(self, 'kafka_client') and self.kafka_client:
            kafka_metrics = await self.kafka_client.get_metrics()

            for topic, lag in kafka_metrics['consumer_lag'].items():
                self.kafka_consumer_lag.labels(
                    service=self.service_name,
                    topic=topic,
                    partition='0'  # Simplified for single partition
                ).set(lag)
```

### Distributed Tracing Implementation

#### 1. Request Tracing Across Services
```python
class DistributedTracing:
    """Implement distributed tracing across all services."""

    @staticmethod
    def trace_request(operation_name: str):
        """Decorator to trace requests across service boundaries."""

        def decorator(func):
            async def wrapper(*args, **kwargs):
                with tracer.start_as_current_span(operation_name) as span:
                    # Add common attributes
                    span.set_attribute("service.name", get_service_name())
                    span.set_attribute("service.version", get_service_version())

                    # Add request-specific attributes
                    if 'request' in kwargs:
                        request = kwargs['request']
                        span.set_attribute("http.method", request.method)
                        span.set_attribute("http.url", str(request.url))
                        span.set_attribute("http.user_agent",
                                         request.headers.get("user-agent", ""))

                    try:
                        result = await func(*args, **kwargs)
                        span.set_status(trace.Status(trace.StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(trace.Status(
                            trace.StatusCode.ERROR,
                            str(e)
                        ))
                        span.record_exception(e)
                        raise

            return wrapper
        return decorator

    @staticmethod
    async def trace_external_call(
        service_name: str,
        operation: str,
        call_func: Callable,
        *args,
        **kwargs
    ):
        """Trace calls to external services."""

        span_name = f"{service_name}.{operation}"

        with tracer.start_as_current_span(span_name) as span:
            span.set_attribute("external.service", service_name)
            span.set_attribute("external.operation", operation)

            start_time = time.time()
            try:
                result = await call_func(*args, **kwargs)
                span.set_attribute("external.success", True)
                return result
            except Exception as e:
                span.set_attribute("external.success", False)
                span.set_attribute("external.error", str(e))
                raise
            finally:
                duration = time.time() - start_time
                span.set_attribute("external.duration_seconds", duration)
```

#### 2. Workflow-Specific Tracing
```python
class WorkflowTracing:
    """Enhanced tracing for workflow execution."""

    @staticmethod
    async def trace_workflow_execution(
        workflow_id: str,
        workflow_name: str,
        execution_func: Callable
    ):
        """Comprehensive workflow execution tracing."""

        with tracer.start_as_current_span("workflow.execution") as parent_span:
            parent_span.set_attribute("workflow.id", workflow_id)
            parent_span.set_attribute("workflow.name", workflow_name)

            # Create workflow context for child spans
            workflow_context = {
                "workflow_id": workflow_id,
                "workflow_name": workflow_name,
                "trace_id": parent_span.get_span_context().trace_id
            }

            try:
                # Execute workflow with context propagation
                result = await execution_func(workflow_context)

                # Record workflow success metrics
                parent_span.set_attribute("workflow.status", "success")
                parent_span.set_attribute("workflow.tasks_completed",
                                        result.get('tasks_completed', 0))

                return result

            except Exception as e:
                parent_span.set_attribute("workflow.status", "failed")
                parent_span.set_attribute("workflow.error", str(e))
                raise
```

### Structured Logging Strategy

#### 1. Unified Log Format
```python
# Structured logging configuration
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(min_level=logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False,
)

class StructuredLogger:
    """Centralized structured logging for all services."""

    def __init__(self, service_name: str, version: str):
        self.logger = structlog.get_logger().bind(
            service=service_name,
            version=version,
            environment=os.getenv("ENVIRONMENT", "development")
        )

    async def log_request(
        self,
        request: Request,
        response_status: int,
        duration: float,
        additional_data: dict = None
    ) -> None:
        """Log HTTP request with comprehensive context."""

        log_data = {
            "event": "http_request",
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "status_code": response_status,
            "duration_ms": duration * 1000,
            "client_ip": request.client.host,
            "user_agent": request.headers.get("user-agent", ""),
            "correlation_id": get_correlation_id(request),
            "trace_id": get_trace_id(),
        }

        if additional_data:
            log_data.update(additional_data)

        if response_status >= 400:
            self.logger.warning("HTTP request failed", **log_data)
        else:
            self.logger.info("HTTP request processed", **log_data)

    async def log_business_event(
        self,
        event_type: str,
        event_data: dict,
        correlation_context: dict = None
    ) -> None:
        """Log business events with full context."""

        log_entry = {
            "event": "business_event",
            "event_type": event_type,
            "event_data": event_data,
            "correlation_id": get_correlation_id(),
            "trace_id": get_trace_id(),
        }

        if correlation_context:
            log_entry["correlation_context"] = correlation_context

        self.logger.info("Business event occurred", **log_entry)
```

### Alerting and Incident Response

#### 1. Intelligent Alerting Rules
```python
class AlertingManager:
    """Manage alerts with intelligent noise reduction."""

    def __init__(self):
        self.alert_rules = {
            # Critical system alerts
            "high_error_rate": {
                "condition": "error_rate > 5%",
                "window": "5m",
                "severity": "critical",
                "description": "Error rate exceeds 5% over 5 minutes",
                "runbook": "https://docs.company.com/runbooks/high-error-rate"
            },

            # Performance alerts
            "slow_response_time": {
                "condition": "p95_response_time > 2s",
                "window": "10m",
                "severity": "warning",
                "description": "95th percentile response time exceeds 2 seconds",
                "runbook": "https://docs.company.com/runbooks/slow-response"
            },

            # Business metric alerts
            "workflow_failure_rate": {
                "condition": "workflow_success_rate < 95%",
                "window": "15m",
                "severity": "warning",
                "description": "Workflow success rate below 95%",
                "runbook": "https://docs.company.com/runbooks/workflow-failures"
            },

            # Infrastructure alerts
            "database_connection_exhaustion": {
                "condition": "db_pool_utilization > 90%",
                "window": "5m",
                "severity": "high",
                "description": "Database connection pool utilization over 90%",
                "runbook": "https://docs.company.com/runbooks/db-connections"
            }
        }

    async def evaluate_alerts(self, metrics_data: dict) -> list:
        """Evaluate alert conditions and generate notifications."""

        triggered_alerts = []

        for alert_name, rule in self.alert_rules.items():
            if await self.evaluate_condition(rule["condition"], metrics_data):
                alert = {
                    "name": alert_name,
                    "severity": rule["severity"],
                    "description": rule["description"],
                    "timestamp": datetime.utcnow().isoformat(),
                    "runbook_url": rule.get("runbook"),
                    "metrics_snapshot": self.get_relevant_metrics(
                        alert_name, metrics_data
                    )
                }

                # Apply intelligent noise reduction
                if not await self.is_duplicate_alert(alert):
                    triggered_alerts.append(alert)
                    await self.send_alert_notification(alert)

        return triggered_alerts
```

#### 2. Escalation and Response
```python
class IncidentResponse:
    """Automated incident response and escalation."""

    async def handle_alert(self, alert: dict) -> None:
        """Handle alert with appropriate response actions."""

        severity = alert["severity"]

        # Immediate automated responses
        if severity == "critical":
            await self.trigger_immediate_response(alert)
            await self.notify_oncall_engineer(alert)
            await self.create_incident_ticket(alert)

        elif severity == "high":
            await self.notify_team_channel(alert)
            await self.create_incident_ticket(alert)

        elif severity == "warning":
            await self.notify_team_channel(alert)

        # Record alert for analysis
        await self.record_alert_event(alert)

    async def trigger_immediate_response(self, alert: dict) -> None:
        """Trigger immediate automated response actions."""

        alert_type = alert["name"]

        response_actions = {
            "high_error_rate": self.enable_circuit_breaker,
            "database_connection_exhaustion": self.scale_connection_pool,
            "memory_exhaustion": self.trigger_garbage_collection,
            "disk_space_critical": self.cleanup_old_logs,
        }

        if alert_type in response_actions:
            await response_actions[alert_type](alert)
```

### Dashboard and Visualization

#### 1. Service Health Dashboard
```python
class HealthDashboard:
    """Generate service health dashboard data."""

    async def get_service_overview(self) -> dict:
        """Get comprehensive service health overview."""

        return {
            "services": {
                "hook_receiver": await self.get_service_health("hook_receiver"),
                "model_metrics": await self.get_service_health("model_metrics"),
                "workflow_coordinator": await self.get_service_health("workflow_coordinator")
            },
            "infrastructure": {
                "database": await self.get_database_health(),
                "kafka": await self.get_kafka_health(),
                "redis": await self.get_redis_health()
            },
            "business_metrics": {
                "total_workflows_today": await self.get_daily_workflow_count(),
                "average_workflow_duration": await self.get_avg_workflow_duration(),
                "success_rate_24h": await self.get_success_rate_24h()
            },
            "alerts": {
                "active_alerts": await self.get_active_alerts(),
                "alert_trends": await self.get_alert_trends()
            }
        }

    async def get_service_health(self, service_name: str) -> dict:
        """Get detailed health metrics for a specific service."""

        return {
            "status": await self.determine_service_status(service_name),
            "uptime": await self.get_service_uptime(service_name),
            "request_rate": await self.get_request_rate(service_name),
            "error_rate": await self.get_error_rate(service_name),
            "response_time_p95": await self.get_response_time_p95(service_name),
            "active_connections": await self.get_active_connections(service_name),
            "recent_errors": await self.get_recent_errors(service_name, limit=5)
        }
```

## Implementation Details

### Monitoring Stack Configuration
```yaml
# Prometheus configuration
prometheus:
  global:
    scrape_interval: 15s
    evaluation_interval: 15s
  rule_files:
    - "/etc/prometheus/alert.rules.yml"
  scrape_configs:
    - job_name: 'hook-receiver'
      static_configs:
        - targets: ['hook-receiver:8001']
      metrics_path: '/metrics'
    - job_name: 'model-metrics'
      static_configs:
        - targets: ['model-metrics:8002']
    - job_name: 'workflow-coordinator'
      static_configs:
        - targets: ['workflow-coordinator:8003']

# Grafana dashboard configuration
grafana:
  dashboards:
    - name: "Service Overview"
      uid: "service-overview"
      panels:
        - title: "Request Rate"
          type: "graph"
          targets:
            - expr: "rate(http_requests_total[5m])"
        - title: "Error Rate"
          type: "stat"
          targets:
            - expr: "rate(http_requests_total{status=~'5..'}[5m]) / rate(http_requests_total[5m])"
```

### Log Aggregation Setup
```yaml
# ELK Stack configuration for log aggregation
elasticsearch:
  cluster.name: "omninode-logs"
  node.name: "omninode-node-1"
  path.data: "/usr/share/elasticsearch/data"
  path.logs: "/usr/share/elasticsearch/logs"
  network.host: "0.0.0.0"
  discovery.seed_hosts: ["elasticsearch"]
  cluster.initial_master_nodes: ["omninode-node-1"]

logstash:
  input:
    beats:
      port: 5044
  filter:
    json:
      source: "message"
    date:
      match: ["timestamp", "ISO8601"]
  output:
    elasticsearch:
      hosts: ["elasticsearch:9200"]
      index: "omninode-logs-%{+YYYY.MM.dd}"

kibana:
  server.host: "0.0.0.0"
  elasticsearch.hosts: ["http://elasticsearch:9200"]
  logging.dest: stdout
```

## Consequences

### Positive Consequences

- **Complete Visibility**: Full observability across all services and infrastructure components
- **Proactive Issue Detection**: Early warning system prevents incidents from escalating
- **Fast Root Cause Analysis**: Correlated telemetry data enables quick problem resolution
- **Performance Optimization**: Continuous performance monitoring identifies optimization opportunities
- **Business Intelligence**: Business metrics provide insights into system usage and efficiency
- **Operational Efficiency**: Automated alerting and response reduces manual intervention needs
- **Compliance Readiness**: Comprehensive audit trails and monitoring support compliance requirements
- **Data-Driven Decisions**: Rich telemetry data supports informed architectural and operational decisions

### Negative Consequences

- **Infrastructure Overhead**: Monitoring stack requires significant additional resources
- **Storage Requirements**: Metrics, logs, and traces consume substantial storage capacity
- **Network Impact**: Telemetry data transmission increases network utilization
- **Complexity Management**: Comprehensive monitoring adds operational complexity
- **Alert Fatigue**: Poorly configured alerts can overwhelm operations teams
- **Performance Impact**: Instrumentation code adds small latency to request processing
- **Cost Implications**: Monitoring infrastructure and storage costs can be significant

### Mitigation Strategies

- **Efficient Sampling**: Implement intelligent sampling for traces to reduce overhead
- **Data Retention Policies**: Automated cleanup of old telemetry data to manage storage
- **Alert Tuning**: Continuous refinement of alert thresholds to minimize false positives
- **Performance Budgets**: Set and monitor performance budgets for instrumentation overhead
- **Monitoring Monitoring**: Monitor the monitoring stack itself for reliability
- **Cost Optimization**: Regular review and optimization of monitoring costs

## Compliance

This monitoring strategy aligns with ONEX standards by:

- **Observability First**: Comprehensive visibility into all system components and behaviors
- **Proactive Operations**: Early detection and automated response to system issues
- **Performance Excellence**: Continuous monitoring and optimization of system performance
- **Security Monitoring**: Real-time detection of security events and anomalies
- **Business Alignment**: Monitoring of business metrics alongside technical metrics
- **Operational Excellence**: Automated monitoring, alerting, and response capabilities

## Related Decisions

- ADR-013: Multi-Service Architecture Pattern
- ADR-015: Circuit Breaker Pattern Implementation
- ADR-017: Authentication and Authorization Strategy
- ADR-018: Rate Limiting and API Security Strategy

## References

- [The Three Pillars of Observability](https://peter.bourgon.org/blog/2017/02/21/metrics-tracing-and-logging.html)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Structured Logging with structlog](https://www.structlog.org/)
- [SRE Book - Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)
