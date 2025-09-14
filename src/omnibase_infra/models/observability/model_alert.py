"""Alert Model.

Shared model for infrastructure alerts and notifications.
Used across observability infrastructure for alert management.
"""

from enum import Enum
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
from datetime import datetime


class AlertSeverityEnum(str, Enum):
    """Alert severity levels."""
    CRITICAL = "critical"        # Service-affecting issues
    HIGH = "high"               # Performance degradation
    MEDIUM = "medium"           # Potential issues
    LOW = "low"                 # Informational


class ModelAlert(BaseModel):
    """Model for infrastructure alerts."""
    
    id: str = Field(
        description="Unique alert identifier"
    )
    
    name: str = Field(
        description="Alert name/title"
    )
    
    description: str = Field(
        description="Detailed alert description"
    )
    
    severity: AlertSeverityEnum = Field(
        description="Alert severity level"
    )
    
    timestamp: datetime = Field(
        description="Alert creation timestamp"
    )
    
    source: str = Field(
        description="Source component that generated the alert"
    )
    
    resolved: bool = Field(
        default=False,
        description="Whether the alert has been resolved"
    )
    
    resolution_timestamp: Optional[datetime] = Field(
        default=None,
        description="Alert resolution timestamp"
    )
    
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional alert details and context"
    )
    
    environment: Optional[str] = Field(
        default=None,
        description="Environment where alert was generated"
    )
    
    threshold_value: Optional[float] = Field(
        default=None,
        description="Threshold value that triggered the alert"
    )
    
    current_value: Optional[float] = Field(
        default=None,
        description="Current metric value when alert was triggered"
    )
    
    alert_rule: Optional[str] = Field(
        default=None,
        description="Alert rule that triggered this alert"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }