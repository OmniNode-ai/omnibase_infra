"""Data models for agent registry."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class AgentStatus(str, Enum):
    """Agent status enumeration."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    ERROR = "error"


class AgentType(str, Enum):
    """Agent type enumeration."""

    CONTRACT_INFERENCER = "contract_inferencer"
    TEMPLATE_SELECTOR = "template_selector"
    BUSINESS_LOGIC_GENERATOR = "business_logic_generator"
    VALIDATOR = "validator"
    DEBUG = "debug"
    ANALYZER = "analyzer"
    RESEARCHER = "researcher"


class AgentMetadata(BaseModel):
    """Agent metadata."""

    agent_type: AgentType
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    description: str
    author: Optional[str] = None
    max_concurrent_tasks: int = Field(default=10, ge=1, le=100)
    timeout_seconds: int = Field(default=300, ge=1, le=3600)
    priority: int = Field(default=50, ge=0, le=100)

    # Performance characteristics
    avg_execution_time_ms: Optional[float] = Field(default=None, ge=0)
    success_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # Additional metadata
    tags: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic model configuration."""

        frozen = False  # Allow mutation for runtime metrics updates


class AgentInfo(BaseModel):
    """Complete agent information."""

    agent_id: str
    capabilities: list[str] = Field(min_length=1)
    metadata: AgentMetadata
    registered_at: datetime
    last_heartbeat: datetime
    status: AgentStatus = AgentStatus.ACTIVE

    # Runtime metrics
    active_tasks: int = Field(default=0, ge=0)
    total_tasks_completed: int = Field(default=0, ge=0)
    total_tasks_failed: int = Field(default=0, ge=0)

    @field_validator("capabilities")
    @classmethod
    def capabilities_not_empty(cls, v: list[str]) -> list[str]:
        """Validate capabilities list is not empty."""
        if not v:
            raise ValueError("capabilities must contain at least one capability")
        return v

    class Config:
        """Pydantic model configuration."""

        frozen = False  # Allow mutation for runtime metrics updates


class Task(BaseModel):
    """Task to be matched to an agent."""

    task_id: UUID = Field(default_factory=uuid4)
    task_type: str
    requirements: list[str] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)
    priority: int = Field(default=50, ge=0, le=100)
    timeout_seconds: int = Field(default=300, gt=0)

    # Context for matching
    complexity: Optional[str] = Field(default=None)
    domain: Optional[str] = None
    node_type: Optional[str] = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class RegistrationResult(BaseModel):
    """Result of agent registration."""

    success: bool
    agent_id: str
    message: str
    registration_time_ms: float = Field(ge=0)
    errors: list[str] = Field(default_factory=list)


class ConfidenceScore(BaseModel):
    """Confidence score breakdown."""

    total: float = Field(..., ge=0.0, le=1.0)
    capability_score: float = Field(..., ge=0.0, le=1.0)
    load_score: float = Field(..., ge=0.0, le=1.0)
    priority_score: float = Field(..., ge=0.0, le=1.0)
    success_rate_score: float = Field(..., ge=0.0, le=1.0)

    explanation: str
