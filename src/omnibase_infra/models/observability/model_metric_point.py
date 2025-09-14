"""Metric Point Model.

Shared model for individual metric data points.
Used across observability infrastructure for metric collection.
"""

from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, Optional
from datetime import datetime


class MetricTypeEnum(str, Enum):
    """Types of metrics collected by observability system."""
    COUNTER = "counter"           # Monotonically increasing values
    GAUGE = "gauge"              # Point-in-time values
    HISTOGRAM = "histogram"      # Distribution of values
    SUMMARY = "summary"          # Summary statistics


class ModelMetricPoint(BaseModel):
    """Model for single metric data point."""
    
    name: str = Field(
        description="Metric name identifier"
    )
    
    value: float = Field(
        description="Metric value"
    )
    
    timestamp: datetime = Field(
        description="Metric collection timestamp"
    )
    
    metric_type: MetricTypeEnum = Field(
        default=MetricTypeEnum.GAUGE,
        description="Type of metric"
    )
    
    labels: Dict[str, str] = Field(
        default_factory=dict,
        description="Metric labels for categorization"
    )
    
    unit: Optional[str] = Field(
        default=None,
        description="Unit of measurement (e.g., 'bytes', 'seconds', 'percent')"
    )
    
    source: Optional[str] = Field(
        default=None,
        description="Source component that generated the metric"
    )
    
    environment: Optional[str] = Field(
        default=None,
        description="Environment where metric was collected"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }