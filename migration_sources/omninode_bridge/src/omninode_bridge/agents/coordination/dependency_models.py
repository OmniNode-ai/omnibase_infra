"""
Data models for dependency resolution in multi-agent coordination.

This module provides type-safe dependency models supporting:
- agent_completion: Wait for another agent to complete
- resource_availability: Check resource availability
- quality_gate: Wait for quality gate to pass

Performance target: <2s total resolution time
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class DependencyType(str, Enum):
    """Types of dependencies that can be resolved."""

    AGENT_COMPLETION = "agent_completion"
    RESOURCE_AVAILABILITY = "resource_availability"
    QUALITY_GATE = "quality_gate"


class DependencyStatus(str, Enum):
    """Status of dependency resolution."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class Dependency:
    """
    Represents a single dependency that must be resolved.

    Attributes:
        dependency_id: Unique dependency identifier
        dependency_type: Type of dependency (agent_completion, resource_availability, quality_gate)
        target: Target identifier (agent_id, resource_id, gate_id)
        timeout: Maximum wait time in seconds (default: 120)
        retry_count: Current retry count (default: 0)
        max_retries: Maximum retry attempts (default: 3)
        metadata: Additional dependency-specific metadata
        status: Current dependency status
        resolved_at: Timestamp when dependency was resolved
        error_message: Error message if resolution failed
    """

    dependency_id: str
    dependency_type: DependencyType
    target: str
    timeout: int = 120
    retry_count: int = 0
    max_retries: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)
    status: DependencyStatus = DependencyStatus.PENDING
    resolved_at: Optional[datetime] = None
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate dependency configuration."""
        if self.timeout <= 0:
            raise ValueError(f"Timeout must be positive, got {self.timeout}")

        if self.max_retries < 0:
            raise ValueError(
                f"max_retries must be non-negative, got {self.max_retries}"
            )

        # Convert string to enum if needed
        if isinstance(self.dependency_type, str):
            self.dependency_type = DependencyType(self.dependency_type)

        if isinstance(self.status, str):
            self.status = DependencyStatus(self.status)

    def mark_resolved(self) -> None:
        """Mark dependency as resolved with timestamp."""
        self.status = DependencyStatus.RESOLVED
        self.resolved_at = datetime.utcnow()

    def mark_failed(self, error_message: str) -> None:
        """Mark dependency as failed with error message."""
        self.status = DependencyStatus.FAILED
        self.error_message = error_message

    def mark_timeout(self) -> None:
        """Mark dependency as timed out."""
        self.status = DependencyStatus.TIMEOUT
        self.error_message = (
            f"Dependency '{self.dependency_id}' timed out after {self.timeout}s"
        )

    def increment_retry(self) -> bool:
        """
        Increment retry count.

        Returns:
            True if retries remain, False if max retries exceeded
        """
        self.retry_count += 1
        return self.retry_count <= self.max_retries

    def to_dict(self) -> dict[str, Any]:
        """Convert dependency to dictionary for serialization."""
        return {
            "dependency_id": self.dependency_id,
            "dependency_type": self.dependency_type.value,
            "target": self.target,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "metadata": self.metadata,
            "status": self.status.value,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Dependency":
        """Create dependency from dictionary."""
        if data.get("resolved_at"):
            data["resolved_at"] = datetime.fromisoformat(data["resolved_at"])

        return cls(**data)


@dataclass
class DependencyResolutionResult:
    """
    Result of dependency resolution attempt.

    Attributes:
        coordination_id: Coordination session ID
        dependency_id: Dependency that was resolved
        success: Whether resolution succeeded
        status: Final status of the dependency
        duration_ms: Resolution duration in milliseconds
        attempts: Number of resolution attempts
        error_message: Error message if resolution failed
        metadata: Additional result metadata
    """

    coordination_id: str
    dependency_id: str
    success: bool
    status: DependencyStatus
    duration_ms: float
    attempts: int = 1
    error_message: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "coordination_id": self.coordination_id,
            "dependency_id": self.dependency_id,
            "success": self.success,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "attempts": self.attempts,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class AgentCompletionConfig:
    """Configuration for agent_completion dependency type."""

    agent_id: str
    completion_event: str = "completion"
    require_success: bool = True

    def to_metadata(self) -> dict[str, Any]:
        """Convert to metadata dictionary."""
        return {
            "agent_id": self.agent_id,
            "completion_event": self.completion_event,
            "require_success": self.require_success,
        }


@dataclass
class ResourceAvailabilityConfig:
    """Configuration for resource_availability dependency type."""

    resource_id: str
    resource_type: str  # "database", "api", "file", "service"
    check_interval_ms: int = 100
    availability_threshold: float = 1.0  # 0.0-1.0 (100% by default)

    def to_metadata(self) -> dict[str, Any]:
        """Convert to metadata dictionary."""
        return {
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "check_interval_ms": self.check_interval_ms,
            "availability_threshold": self.availability_threshold,
        }


@dataclass
class QualityGateConfig:
    """Configuration for quality_gate dependency type."""

    gate_id: str
    gate_type: str  # "coverage", "linting", "type_checking", "security"
    threshold: float  # Minimum score required (0.0-1.0)
    check_interval_ms: int = 500

    def to_metadata(self) -> dict[str, Any]:
        """Convert to metadata dictionary."""
        return {
            "gate_id": self.gate_id,
            "gate_type": self.gate_type,
            "threshold": self.threshold,
            "check_interval_ms": self.check_interval_ms,
        }
