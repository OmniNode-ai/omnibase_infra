"""
Performance metrics data models.

Pydantic v2 models for type-safe metrics, alerts, and configurations.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class MetricType(str, Enum):
    """Metric type enumeration."""

    TIMING = "timing"
    COUNTER = "counter"
    GAUGE = "gauge"
    RATE = "rate"


class Metric(BaseModel):
    """
    Performance metric with metadata.

    Attributes:
        metric_id: Unique metric identifier
        metric_name: Metric name (e.g., "routing_decision_time_ms")
        metric_type: Type of metric
        value: Metric value
        unit: Unit of measurement (ms, count, KB, %)
        tags: Optional tags for filtering/grouping
        timestamp: Metric timestamp
        correlation_id: Optional correlation ID for tracing
        agent_id: Optional agent identifier
    """

    metric_id: str = Field(default_factory=lambda: str(uuid4()))
    metric_name: str = Field(..., min_length=1, max_length=100)
    metric_type: MetricType
    value: float
    unit: str = Field(..., min_length=1, max_length=20)
    tags: dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    agent_id: Optional[str] = None

    class Config:
        """Pydantic configuration."""

        frozen = False
        validate_assignment = True


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


class Alert(BaseModel):
    """
    Performance alert.

    Attributes:
        alert_id: Unique alert identifier
        severity: Alert severity level
        metric_name: Metric that triggered alert
        threshold: Threshold value that was exceeded
        actual_value: Actual metric value
        message: Human-readable alert message
        timestamp: Alert timestamp
        tags: Optional tags from metric
        correlation_id: Optional correlation ID
    """

    alert_id: str = Field(default_factory=lambda: str(uuid4()))
    severity: AlertSeverity
    metric_name: str = Field(..., min_length=1, max_length=100)
    threshold: float
    actual_value: float
    message: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tags: dict[str, str] = Field(default_factory=dict)
    correlation_id: Optional[str] = None

    class Config:
        """Pydantic configuration."""

        frozen = True


class AlertRule(BaseModel):
    """
    Alert rule configuration.

    Attributes:
        rule_id: Unique rule identifier
        metric_name: Metric name to monitor
        threshold: Threshold value
        operator: Comparison operator (gt, lt, gte, lte)
        severity: Alert severity if threshold exceeded
        message_template: Message template (supports {metric_name}, {value}, {threshold})
    """

    rule_id: str = Field(default_factory=lambda: str(uuid4()))
    metric_name: str = Field(..., min_length=1, max_length=100)
    threshold: float
    operator: str = Field(..., pattern="^(gt|lt|gte|lte)$")
    severity: AlertSeverity
    message_template: str = Field(
        default="{metric_name} {operator} {threshold}: actual={value}"
    )

    class Config:
        """Pydantic configuration."""

        frozen = True

    def evaluate(self, metric: Metric) -> Optional[Alert]:
        """
        Evaluate rule against metric.

        Args:
            metric: Metric to evaluate

        Returns:
            Alert if rule triggered, None otherwise
        """
        if metric.metric_name != self.metric_name:
            return None

        triggered = False
        if self.operator == "gt":
            triggered = metric.value > self.threshold
        elif self.operator == "lt":
            triggered = metric.value < self.threshold
        elif self.operator == "gte":
            triggered = metric.value >= self.threshold
        elif self.operator == "lte":
            triggered = metric.value <= self.threshold

        if not triggered:
            return None

        message = self.message_template.format(
            metric_name=metric.metric_name,
            value=metric.value,
            threshold=self.threshold,
            operator=self.operator,
        )

        return Alert(
            severity=self.severity,
            metric_name=metric.metric_name,
            threshold=self.threshold,
            actual_value=metric.value,
            message=message,
            tags=metric.tags,
            correlation_id=metric.correlation_id,
        )


class MetricEventPayload(BaseModel):
    """Kafka metric event payload."""

    metric_id: str
    metric_name: str
    metric_type: str
    value: float
    unit: str
    tags: dict[str, str]
    agent_id: Optional[str] = None
    correlation_id: Optional[str] = None


class MetricEvent(BaseModel):
    """
    Kafka metric event (OnexEnvelopeV1 format).

    Attributes:
        event_type: Event type identifier
        event_version: Event version
        event_id: Unique event ID
        timestamp: Event timestamp
        source_service: Source service name
        correlation_id: Optional correlation ID
        payload: Metric payload
    """

    event_type: str = "metric.recorded"
    event_version: str = "v1"
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_service: str = "omninode-bridge"
    correlation_id: Optional[str] = None
    payload: MetricEventPayload

    class Config:
        """Pydantic configuration."""

        frozen = True


class AlertEventPayload(BaseModel):
    """Kafka alert event payload."""

    alert_id: str
    severity: AlertSeverity
    metric_name: str
    threshold: float
    actual_value: float
    message: str
    tags: dict[str, str]


class AlertEvent(BaseModel):
    """
    Kafka alert event (OnexEnvelopeV1 format).

    Attributes:
        event_type: Event type identifier
        event_version: Event version
        event_id: Unique event ID
        timestamp: Event timestamp
        source_service: Source service name
        correlation_id: Optional correlation ID
        payload: Alert payload
    """

    event_type: str = "alert.triggered"
    event_version: str = "v1"
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_service: str = "omninode-bridge"
    correlation_id: Optional[str] = None
    payload: AlertEventPayload

    class Config:
        """Pydantic configuration."""

        frozen = True
