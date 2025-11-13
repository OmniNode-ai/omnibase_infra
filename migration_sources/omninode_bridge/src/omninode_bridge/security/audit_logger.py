"""Comprehensive audit logging for security events with correlation ID support."""

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import structlog
from fastapi import Request

from ..middleware.request_correlation import get_correlation_context


class AuditEventType(Enum):
    """Types of security events to audit."""

    # Authentication events
    AUTHENTICATION_SUCCESS = "authentication_success"
    AUTHENTICATION_FAILURE = "authentication_failure"
    INVALID_API_KEY = "invalid_api_key"
    MISSING_API_KEY = "missing_api_key"

    # Authorization events
    AUTHORIZATION_SUCCESS = "authorization_success"
    AUTHORIZATION_FAILURE = "authorization_failure"
    INSUFFICIENT_PERMISSIONS = "insufficient_permissions"

    # Rate limiting events
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    RATE_LIMIT_WARNING = "rate_limit_warning"

    # Input validation events
    INPUT_VALIDATION_FAILURE = "input_validation_failure"
    MALICIOUS_INPUT_DETECTED = "malicious_input_detected"
    PAYLOAD_SIZE_EXCEEDED = "payload_size_exceeded"

    # Workflow events
    WORKFLOW_SUBMISSION = "workflow_submission"
    WORKFLOW_EXECUTION_START = "workflow_execution_start"
    WORKFLOW_EXECUTION_COMPLETE = "workflow_execution_complete"
    WORKFLOW_EXECUTION_FAILURE = "workflow_execution_failure"

    # Session events
    SESSION_CREATED = "session_created"
    SESSION_TERMINATED = "session_terminated"
    SESSION_EXPIRED = "session_expired"

    # System events
    SERVICE_STARTUP = "service_startup"
    SERVICE_SHUTDOWN = "service_shutdown"
    HEALTH_CHECK_FAILURE = "health_check_failure"

    # Resource access events
    RESOURCE_ACCESS = "resource_access"
    RESOURCE_CREATION = "resource_creation"
    RESOURCE_DELETION = "resource_deletion"

    # Security events
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    SECURITY_VIOLATION = "security_violation"
    POTENTIAL_ATTACK = "potential_attack"

    # Configuration and sensitive operations
    CONFIG_CHANGE = "config_change"
    SECRET_ACCESS = "secret_access"
    ADMIN_OPERATION = "admin_operation"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXPORT = "data_export"
    BULK_OPERATION = "bulk_operation"

    # Infrastructure events
    INFRASTRUCTURE_ACCESS = "infrastructure_access"
    DEPLOYMENT_EVENT = "deployment_event"
    SCALING_EVENT = "scaling_event"
    BACKUP_OPERATION = "backup_operation"

    # Data privacy events
    PII_ACCESS = "pii_access"
    DATA_MODIFICATION = "data_modification"
    CONSENT_CHANGE = "consent_change"


class AuditSeverity(Enum):
    """Severity levels for audit events."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditLogger:
    """Enhanced audit logging system for security events."""

    def __init__(self, service_name: str, version: str = "1.0.0"):
        """Initialize audit logger."""
        self.service_name = service_name
        self.version = version
        self.logger = structlog.get_logger(f"audit.{service_name}")

    def log_event(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity = AuditSeverity.MEDIUM,
        user_id: str | None = None,
        session_id: str | None = None,
        request: Request | None = None,
        additional_data: dict[str, Any] | None = None,
        message: str | None = None,
        correlation_context: Optional[dict[str, str]] = None,
    ) -> str:
        """Log a security audit event with correlation ID support."""

        audit_id = str(uuid4())
        timestamp = datetime.now(UTC)

        # Get correlation context from middleware or use provided
        if correlation_context is None:
            correlation_context = get_correlation_context()

        # Build audit record
        audit_record = {
            "audit_id": audit_id,
            "timestamp": timestamp.isoformat(),
            "service_name": self.service_name,
            "service_version": self.version,
            "event_type": event_type.value,
            "severity": severity.value,
            "user_id": user_id,
            "session_id": session_id,
        }

        # Add correlation context
        if correlation_context:
            audit_record["correlation_context"] = correlation_context
            # Also add individual correlation IDs at top level for easier querying
            for key, value in correlation_context.items():
                if value:
                    audit_record[f"correlation_{key}"] = value

        # Add request information if available
        if request:
            audit_record.update(
                {
                    "request": {
                        "method": request.method,
                        "url": str(request.url),
                        "path": request.url.path,
                        "query_params": dict(request.query_params),
                        "headers": {
                            "user_agent": request.headers.get("User-Agent"),
                            "content_type": request.headers.get("Content-Type"),
                            "content_length": request.headers.get("Content-Length"),
                            "x_forwarded_for": request.headers.get("X-Forwarded-For"),
                            "x_real_ip": request.headers.get("X-Real-IP"),
                        },
                        "client": {
                            "host": request.client.host if request.client else None,
                            "port": request.client.port if request.client else None,
                        },
                    },
                },
            )

        # Add additional data
        if additional_data:
            audit_record["additional_data"] = additional_data

        # Add message
        if message:
            audit_record["message"] = message

        # Log with appropriate level based on severity
        log_method = self._get_log_method(severity)
        log_method(f"AUDIT: {event_type.value}", extra=audit_record)

        return audit_id

    def log_authentication_success(
        self,
        user_id: str | None = None,
        auth_method: str = "api_key",
        request: Request | None = None,
    ) -> str:
        """Log successful authentication."""
        return self.log_event(
            event_type=AuditEventType.AUTHENTICATION_SUCCESS,
            severity=AuditSeverity.LOW,
            user_id=user_id,
            request=request,
            additional_data={
                "auth_method": auth_method,
            },
            message=f"User authenticated successfully via {auth_method}",
        )

    def log_authentication_failure(
        self,
        reason: str,
        attempted_user: str | None = None,
        auth_method: str = "api_key",
        request: Request | None = None,
    ) -> str:
        """Log failed authentication attempt."""
        return self.log_event(
            event_type=AuditEventType.AUTHENTICATION_FAILURE,
            severity=AuditSeverity.HIGH,
            user_id=attempted_user,
            request=request,
            additional_data={
                "auth_method": auth_method,
                "failure_reason": reason,
            },
            message=f"Authentication failed: {reason}",
        )

    def log_rate_limit_exceeded(
        self,
        endpoint: str,
        limit: str,
        request: Request | None = None,
        user_id: str | None = None,
    ) -> str:
        """Log rate limit exceeded."""
        return self.log_event(
            event_type=AuditEventType.RATE_LIMIT_EXCEEDED,
            severity=AuditSeverity.MEDIUM,
            user_id=user_id,
            request=request,
            additional_data={
                "endpoint": endpoint,
                "rate_limit": limit,
            },
            message=f"Rate limit exceeded for endpoint {endpoint}",
        )

    def log_input_validation_failure(
        self,
        field: str,
        value_type: str,
        validation_error: str,
        request: Request | None = None,
        user_id: str | None = None,
    ) -> str:
        """Log input validation failure."""
        return self.log_event(
            event_type=AuditEventType.INPUT_VALIDATION_FAILURE,
            severity=AuditSeverity.MEDIUM,
            user_id=user_id,
            request=request,
            additional_data={
                "field": field,
                "value_type": value_type,
                "validation_error": validation_error,
            },
            message=f"Input validation failed for field '{field}': {validation_error}",
        )

    def log_malicious_input_detected(
        self,
        input_type: str,
        pattern_matched: str,
        request: Request | None = None,
        user_id: str | None = None,
    ) -> str:
        """Log detection of potentially malicious input."""
        return self.log_event(
            event_type=AuditEventType.MALICIOUS_INPUT_DETECTED,
            severity=AuditSeverity.HIGH,
            user_id=user_id,
            request=request,
            additional_data={
                "input_type": input_type,
                "pattern_matched": pattern_matched,
                "requires_investigation": True,
            },
            message=f"Malicious input detected: {pattern_matched} in {input_type}",
        )

    def log_workflow_submission(
        self,
        workflow_id: str,
        workflow_name: str,
        task_count: int,
        request: Request | None = None,
        user_id: str | None = None,
    ) -> str:
        """Log workflow submission."""
        return self.log_event(
            event_type=AuditEventType.WORKFLOW_SUBMISSION,
            severity=AuditSeverity.LOW,
            user_id=user_id,
            request=request,
            additional_data={
                "workflow_id": workflow_id,
                "workflow_name": workflow_name,
                "task_count": task_count,
            },
            message=f"Workflow submitted: {workflow_name} ({workflow_id})",
        )

    def log_session_event(
        self,
        session_id: str,
        event_type: AuditEventType,
        request: Request | None = None,
        user_id: str | None = None,
        additional_info: dict[str, Any] | None = None,
    ) -> str:
        """Log session-related events."""
        return self.log_event(
            event_type=event_type,
            severity=AuditSeverity.LOW,
            user_id=user_id,
            session_id=session_id,
            request=request,
            additional_data=additional_info,
            message=f"Session event: {event_type.value}",
        )

    def log_suspicious_activity(
        self,
        activity_type: str,
        risk_score: float,
        indicators: list[str],
        request: Request | None = None,
        user_id: str | None = None,
    ) -> str:
        """Log suspicious activity that may indicate an attack."""
        severity = AuditSeverity.CRITICAL if risk_score > 0.8 else AuditSeverity.HIGH

        return self.log_event(
            event_type=AuditEventType.SUSPICIOUS_ACTIVITY,
            severity=severity,
            user_id=user_id,
            request=request,
            additional_data={
                "activity_type": activity_type,
                "risk_score": risk_score,
                "indicators": indicators,
                "requires_immediate_attention": risk_score > 0.8,
            },
            message=f"Suspicious activity detected: {activity_type} (risk: {risk_score})",
        )

    def log_config_change(
        self,
        config_key: str,
        old_value: str | None = None,
        new_value: str | None = None,
        user_id: str | None = None,
        request: Request | None = None,
    ) -> str:
        """Log configuration changes (sensitive operation)."""
        return self.log_event(
            event_type=AuditEventType.CONFIG_CHANGE,
            severity=AuditSeverity.HIGH,
            user_id=user_id,
            request=request,
            additional_data={
                "config_key": config_key,
                "old_value": old_value,
                "new_value": new_value,
                "change_type": "configuration_update",
            },
            message=f"Configuration changed: {config_key}",
        )

    def log_secret_access(
        self,
        secret_name: str,
        access_type: str,  # read, write, delete
        user_id: str | None = None,
        request: Request | None = None,
    ) -> str:
        """Log secret/credential access (sensitive operation)."""
        return self.log_event(
            event_type=AuditEventType.SECRET_ACCESS,
            severity=AuditSeverity.HIGH,
            user_id=user_id,
            request=request,
            additional_data={
                "secret_name": secret_name,
                "access_type": access_type,
                "requires_investigation": True,
            },
            message=f"Secret access: {access_type} operation on {secret_name}",
        )

    def log_admin_operation(
        self,
        operation: str,
        target_resource: str,
        operation_result: str = "success",
        user_id: str | None = None,
        request: Request | None = None,
        additional_context: dict[str, Any] | None = None,
    ) -> str:
        """Log administrative operations (sensitive operation)."""
        return self.log_event(
            event_type=AuditEventType.ADMIN_OPERATION,
            severity=AuditSeverity.HIGH,
            user_id=user_id,
            request=request,
            additional_data={
                "operation": operation,
                "target_resource": target_resource,
                "operation_result": operation_result,
                "admin_context": additional_context or {},
            },
            message=f"Admin operation: {operation} on {target_resource} ({operation_result})",
        )

    def log_privilege_escalation(
        self,
        from_privilege: str,
        to_privilege: str,
        escalation_method: str,
        user_id: str | None = None,
        request: Request | None = None,
    ) -> str:
        """Log privilege escalation events (sensitive operation)."""
        return self.log_event(
            event_type=AuditEventType.PRIVILEGE_ESCALATION,
            severity=AuditSeverity.CRITICAL,
            user_id=user_id,
            request=request,
            additional_data={
                "from_privilege": from_privilege,
                "to_privilege": to_privilege,
                "escalation_method": escalation_method,
                "requires_immediate_review": True,
            },
            message=f"Privilege escalation: {from_privilege} -> {to_privilege} via {escalation_method}",
        )

    def log_data_export(
        self,
        data_type: str,
        record_count: int,
        export_format: str,
        user_id: str | None = None,
        request: Request | None = None,
    ) -> str:
        """Log data export operations (sensitive operation)."""
        severity = AuditSeverity.CRITICAL if record_count > 1000 else AuditSeverity.HIGH

        return self.log_event(
            event_type=AuditEventType.DATA_EXPORT,
            severity=severity,
            user_id=user_id,
            request=request,
            additional_data={
                "data_type": data_type,
                "record_count": record_count,
                "export_format": export_format,
                "large_export": record_count > 1000,
            },
            message=f"Data export: {record_count} {data_type} records in {export_format} format",
        )

    def log_bulk_operation(
        self,
        operation_type: str,
        affected_count: int,
        operation_details: dict[str, Any],
        user_id: str | None = None,
        request: Request | None = None,
    ) -> str:
        """Log bulk operations (sensitive operation)."""
        severity = (
            AuditSeverity.CRITICAL if affected_count > 100 else AuditSeverity.HIGH
        )

        return self.log_event(
            event_type=AuditEventType.BULK_OPERATION,
            severity=severity,
            user_id=user_id,
            request=request,
            additional_data={
                "operation_type": operation_type,
                "affected_count": affected_count,
                "operation_details": operation_details,
                "large_operation": affected_count > 100,
            },
            message=f"Bulk operation: {operation_type} affecting {affected_count} items",
        )

    def log_pii_access(
        self,
        pii_type: str,
        access_reason: str,
        data_subject_id: str | None = None,
        user_id: str | None = None,
        request: Request | None = None,
    ) -> str:
        """Log PII access (sensitive operation for privacy compliance)."""
        return self.log_event(
            event_type=AuditEventType.PII_ACCESS,
            severity=AuditSeverity.HIGH,
            user_id=user_id,
            request=request,
            additional_data={
                "pii_type": pii_type,
                "access_reason": access_reason,
                "data_subject_id": data_subject_id,
                "privacy_compliance_event": True,
            },
            message=f"PII access: {pii_type} for {access_reason}",
        )

    def log_infrastructure_access(
        self,
        infrastructure_component: str,
        access_type: str,
        access_result: str = "success",
        user_id: str | None = None,
        request: Request | None = None,
    ) -> str:
        """Log infrastructure access events."""
        return self.log_event(
            event_type=AuditEventType.INFRASTRUCTURE_ACCESS,
            severity=AuditSeverity.MEDIUM,
            user_id=user_id,
            request=request,
            additional_data={
                "infrastructure_component": infrastructure_component,
                "access_type": access_type,
                "access_result": access_result,
            },
            message=f"Infrastructure access: {access_type} on {infrastructure_component} ({access_result})",
        )

    def log_deployment_event(
        self,
        deployment_type: str,
        service_version: str,
        deployment_result: str = "success",
        user_id: str | None = None,
        request: Request | None = None,
        deployment_details: dict[str, Any] | None = None,
    ) -> str:
        """Log deployment events."""
        return self.log_event(
            event_type=AuditEventType.DEPLOYMENT_EVENT,
            severity=AuditSeverity.MEDIUM,
            user_id=user_id,
            request=request,
            additional_data={
                "deployment_type": deployment_type,
                "service_version": service_version,
                "deployment_result": deployment_result,
                "deployment_details": deployment_details or {},
            },
            message=f"Deployment: {deployment_type} version {service_version} ({deployment_result})",
        )

    def _get_log_method(self, severity: AuditSeverity):
        """Get appropriate logging method based on severity."""
        severity_map = {
            AuditSeverity.LOW: self.logger.info,
            AuditSeverity.MEDIUM: self.logger.warning,
            AuditSeverity.HIGH: self.logger.error,
            AuditSeverity.CRITICAL: self.logger.critical,
        }
        return severity_map.get(severity, self.logger.info)


# Global audit logger instances
_audit_loggers: dict[str, AuditLogger] = {}


def get_audit_logger(service_name: str, version: str = "1.0.0") -> AuditLogger:
    """Get or create audit logger for a service."""
    key = f"{service_name}:{version}"
    if key not in _audit_loggers:
        _audit_loggers[key] = AuditLogger(service_name, version)
    return _audit_loggers[key]
