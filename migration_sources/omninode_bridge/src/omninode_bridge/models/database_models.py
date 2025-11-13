"""SQLAlchemy models for OmniNode Bridge database schema."""

import uuid
from enum import Enum

from sqlalchemy import ARRAY, Boolean, CheckConstraint, Column, DateTime
from sqlalchemy import Enum as SQLEnum
from sqlalchemy import Float, ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from . import Base


class EnumNodeType(str, Enum):
    """ONEX-compliant node type enumeration following naming conventions."""

    EFFECT = "effect"
    COMPUTE = "compute"
    REDUCER = "reducer"
    ORCHESTRATOR = "orchestrator"


class ServiceSession(Base):
    """Service sessions table with enhanced security."""

    __tablename__ = "service_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    service_name = Column(String(255), nullable=False)
    instance_id = Column(String(255))
    session_start = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    session_end = Column(DateTime(timezone=True))
    status = Column(
        String(50),
        nullable=False,
        default="active",
        server_default=text("'active'"),
    )
    session_metadata = Column(JSONB, default=dict, server_default=text("'{}'"))
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'ended', 'terminated', 'failed')",
            name="service_sessions_status_check",
        ),
        CheckConstraint("session_start <= NOW()", name="session_start_valid"),
        CheckConstraint(
            "session_end IS NULL OR session_end >= session_start",
            name="session_end_valid",
        ),
        Index(
            "idx_service_sessions_service_status",
            "service_name",
            "status",
            postgresql_where=text("status = 'active'"),
        ),
        Index(
            "idx_service_sessions_created_at",
            "created_at",
            postgresql_ops={"created_at": "DESC"},
        ),
        Index(
            "idx_service_sessions_cleanup",
            "session_end",
            postgresql_where=text("session_end IS NOT NULL"),
        ),
    )


class HookEvent(Base):
    """Hook events table for persistence and debugging with security enhancements."""

    __tablename__ = "hook_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source = Column(String(255), nullable=False)
    action = Column(String(255), nullable=False)
    resource = Column(String(255), nullable=False)
    resource_id = Column(String(255), nullable=False)
    payload = Column(JSONB, nullable=False, default=dict, server_default=text("'{}'"))
    event_metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'"),
    )
    processed = Column(Boolean, nullable=False, default=False)
    processing_errors = Column(ARRAY(Text))
    retry_count = Column(Integer, nullable=False, default=0)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    processed_at = Column(DateTime(timezone=True))

    # Relationship to event metrics
    event_metrics = relationship("EventMetric", back_populates="hook_event")

    __table_args__ = (
        CheckConstraint(
            "retry_count >= 0 AND retry_count <= 10",
            name="retry_count_valid",
        ),
        CheckConstraint(
            "processed_at IS NULL OR processed_at >= created_at",
            name="processing_time_valid",
        ),
        CheckConstraint("retry_count <= 10", name="retry_count_limit"),
        Index("idx_hook_events_source_action", "source", "action"),
        Index(
            "idx_hook_events_processed",
            "processed",
            "created_at",
            postgresql_where=text("NOT processed"),
            postgresql_ops={"created_at": "DESC"},
        ),
        Index(
            "idx_hook_events_created_at",
            "created_at",
            postgresql_ops={"created_at": "DESC"},
        ),
        Index(
            "idx_hook_events_retry",
            "retry_count",
            "created_at",
            postgresql_where=text("retry_count > 0"),
        ),
    )


class EventMetric(Base):
    """Event processing metrics with enhanced constraints."""

    __tablename__ = "event_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(UUID(as_uuid=True), ForeignKey("hook_events.id"), nullable=False)
    processing_time_ms = Column(Float, nullable=False)
    kafka_publish_success = Column(Boolean, nullable=False)
    error_message = Column(Text)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationship to hook events
    hook_event = relationship("HookEvent", back_populates="event_metrics")

    __table_args__ = (
        CheckConstraint("processing_time_ms >= 0", name="processing_time_non_negative"),
        CheckConstraint(
            "processing_time_ms < 300000",
            name="processing_time_reasonable",
        ),  # 5 minutes max
        Index("idx_event_metrics_event_id", "event_id"),
        Index(
            "idx_event_metrics_created_at",
            "created_at",
            postgresql_ops={"created_at": "DESC"},
        ),
        Index(
            "idx_event_metrics_performance",
            "processing_time_ms",
            "kafka_publish_success",
        ),
    )


class SecurityAuditLog(Base):
    """Security audit table for connection and authentication events."""

    __tablename__ = "security_audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String(100), nullable=False)
    client_info = Column(JSONB)
    success = Column(Boolean, nullable=False)
    error_details = Column(Text)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    __table_args__ = (
        Index(
            "idx_security_audit_event_type",
            "event_type",
            "created_at",
            postgresql_ops={"created_at": "DESC"},
        ),
        Index(
            "idx_security_audit_failed",
            "created_at",
            postgresql_where=text("NOT success"),
            postgresql_ops={"created_at": "DESC"},
        ),
    )


class ConnectionMetric(Base):
    """Performance monitoring table."""

    __tablename__ = "connection_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pool_size = Column(Integer, nullable=False)
    active_connections = Column(Integer, nullable=False)
    idle_connections = Column(Integer, nullable=False)
    total_queries = Column(Integer, nullable=False, default=0)
    failed_queries = Column(Integer, nullable=False, default=0)
    avg_query_time_ms = Column(Float)
    recorded_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    __table_args__ = (
        Index(
            "idx_connection_metrics_recorded_at",
            "recorded_at",
            postgresql_ops={"recorded_at": "DESC"},
        ),
    )


class Workflow(Base):
    """Workflows table for workflow coordinator."""

    __tablename__ = "workflows"

    workflow_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    definition = Column(JSONB, nullable=False)
    status = Column(String(50), nullable=False)
    input_data = Column(JSONB)
    output_data = Column(JSONB)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship to workflow tasks
    tasks = relationship("WorkflowTask", back_populates="workflow")

    __table_args__ = (Index("idx_workflows_status", "status"),)


class WorkflowTask(Base):
    """Workflow tasks table for workflow coordinator."""

    __tablename__ = "workflow_tasks"

    task_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_id = Column(
        UUID(as_uuid=True),
        ForeignKey("workflows.workflow_id"),
        nullable=False,
    )
    name = Column(String(255), nullable=False)
    task_type = Column(String(50), nullable=False)
    config = Column(JSONB, nullable=False)
    dependencies = Column(JSONB)
    status = Column(String(50), nullable=False)
    result = Column(JSONB)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship to workflows
    workflow = relationship("Workflow", back_populates="tasks")

    __table_args__ = (
        Index("idx_workflow_tasks_workflow_id", "workflow_id"),
        Index("idx_workflow_tasks_status", "status"),
    )


class NodeRegistration(Base):
    """Node registrations table for dynamic node discovery and orchestration."""

    __tablename__ = "node_registrations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    node_id = Column(String(255), unique=True, nullable=False)
    node_type = Column(
        SQLEnum(EnumNodeType, name="node_type_enum", create_type=False),
        nullable=False,
    )
    capabilities = Column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'")
    )
    endpoints = Column(JSONB, nullable=False, default=dict, server_default=text("'{}'"))
    node_metadata = Column(
        "metadata", JSONB, nullable=False, default=dict, server_default=text("'{}'")
    )
    health_endpoint = Column(String(500))
    last_heartbeat = Column(DateTime(timezone=True))
    registered_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("idx_node_registrations_node_id", "node_id"),
        Index("idx_node_registrations_node_type", "node_type"),
        Index(
            "idx_node_registrations_last_heartbeat",
            "last_heartbeat",
            postgresql_ops={"last_heartbeat": "DESC"},
        ),
        # GIN indexes for JSONB fields (created via raw SQL in migration)
    )
