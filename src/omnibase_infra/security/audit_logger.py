"""
ONEX Audit Logging for Infrastructure Security

Provides comprehensive audit logging for sensitive database operations
and infrastructure activities for compliance and security monitoring.

Per ONEX security requirements:
- Structured audit logging with standardized formats
- Tamper-proof audit trails with integrity verification
- Real-time security event alerting
- Compliance reporting and data retention
"""

import json
import logging
import hashlib
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

from omnibase_core.core.onex_error import OnexError, CoreErrorCode


class AuditEventType(Enum):
    """Types of events that should be audited."""
    DATABASE_QUERY = "database_query"
    DATABASE_MODIFICATION = "database_modification"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_VIOLATION = "security_violation"
    EVENT_PUBLISH = "event_publish"
    ADMIN_OPERATION = "admin_operation"
    DATA_ACCESS = "data_access"
    SYSTEM_ERROR = "system_error"


class AuditSeverity(Enum):
    """Severity levels for audit events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Standardized audit event structure."""
    event_id: str
    timestamp: str
    event_type: AuditEventType
    severity: AuditSeverity
    user_id: Optional[str]
    client_id: Optional[str]
    session_id: Optional[str]
    correlation_id: Optional[str]
    resource: str
    action: str
    outcome: str  # "success", "failure", "denied"
    details: Dict[str, Any]
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and enrich audit event."""
        if not self.event_id:
            self.event_id = self._generate_event_id()
        
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        
        # Sanitize sensitive data in details
        self.details = self._sanitize_details(self.details)
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        timestamp_ms = int(time.time() * 1000)
        content = f"{timestamp_ms}{self.event_type.value}{self.resource}{self.action}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _sanitize_details(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Remove or mask sensitive information from details."""
        sanitized = {}
        
        for key, value in details.items():
            key_lower = key.lower()
            
            # Mask sensitive fields
            if any(sensitive in key_lower for sensitive in ['password', 'token', 'secret', 'key']):
                sanitized[key] = "***REDACTED***"
            elif key_lower in ['ssn', 'credit_card', 'account_number']:
                sanitized[key] = "***REDACTED***"
            elif hasattr(value, '__len__') and hasattr(value, 'strip') and len(value) > 500:
                # Truncate very long strings
                sanitized[key] = value[:497] + "..."
            else:
                sanitized[key] = value
        
        return sanitized
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['event_type'] = self.event_type.value
        result['severity'] = self.severity.value
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)
    
    def get_integrity_hash(self) -> str:
        """Generate integrity hash for tamper detection."""
        content = self.to_json()
        return hashlib.sha256(content.encode()).hexdigest()


class ONEXAuditLogger:
    """
    ONEX audit logger for infrastructure security monitoring.
    
    Features:
    - Structured audit event logging
    - Tamper-proof audit trails
    - Real-time security alerting
    - Compliance reporting
    """
    
    def __init__(self):
        self._logger = logging.getLogger("onex.audit")
        self._setup_audit_logger()
        
        # Audit configuration
        self._enabled = self._get_audit_config("enabled", True)
        self._min_severity = AuditSeverity(self._get_audit_config("min_severity", "low"))
        self._alert_threshold = AuditSeverity(self._get_audit_config("alert_threshold", "high"))
        
        # Audit trail integrity
        self._last_hash = ""
        self._sequence_number = 0
        
        self._logger.info("ONEX audit logger initialized")
    
    def _setup_audit_logger(self):
        """Configure audit-specific logger with secure settings."""
        # Create separate handler for audit logs
        audit_handler = logging.FileHandler(
            "/var/log/onex/audit.log",
            mode='a',
            encoding='utf-8'
        )
        
        # JSON formatter for structured logging
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": %(message)s}'
        )
        audit_handler.setFormatter(formatter)
        
        # Set security-focused configuration
        audit_handler.setLevel(logging.INFO)
        self._logger.addHandler(audit_handler)
        self._logger.setLevel(logging.INFO)
        
        # Prevent audit logs from going to parent loggers
        self._logger.propagate = False
    
    def _get_audit_config(self, key: str, default: Any) -> Any:
        """Get audit configuration value."""
        import os
        env_key = f"ONEX_AUDIT_{key.upper()}"
        return os.getenv(env_key, default)
    
    def log_database_operation(self, 
                              user_id: Optional[str],
                              client_id: Optional[str],
                              correlation_id: Optional[str],
                              operation: str,
                              table_name: str,
                              query_type: str,
                              row_count: Optional[int] = None,
                              outcome: str = "success",
                              error_message: Optional[str] = None,
                              execution_time_ms: Optional[float] = None):
        """
        Log database operation for audit trail.
        
        Args:
            user_id: User performing the operation
            client_id: Client identifier
            correlation_id: Request correlation ID
            operation: Type of operation (SELECT, INSERT, UPDATE, DELETE)
            table_name: Database table affected
            query_type: Query classification (read, write, admin)
            row_count: Number of rows affected
            outcome: Operation outcome (success, failure, denied)
            error_message: Error message if operation failed
            execution_time_ms: Query execution time in milliseconds
        """
        # Determine severity based on operation type and outcome
        severity = AuditSeverity.LOW
        if query_type == "admin" or operation in ["DELETE", "DROP", "ALTER"]:
            severity = AuditSeverity.HIGH
        elif query_type == "write" or operation in ["INSERT", "UPDATE"]:
            severity = AuditSeverity.MEDIUM
        
        if outcome == "failure":
            severity = AuditSeverity.HIGH
        elif outcome == "denied":
            severity = AuditSeverity.CRITICAL
        
        details = {
            "operation": operation,
            "table_name": table_name,
            "query_type": query_type,
            "row_count": row_count,
            "execution_time_ms": execution_time_ms
        }
        
        if error_message:
            details["error_message"] = error_message
        
        event = AuditEvent(
            event_id="",
            timestamp="",
            event_type=AuditEventType.DATABASE_QUERY if query_type == "read" else AuditEventType.DATABASE_MODIFICATION,
            severity=severity,
            user_id=user_id,
            client_id=client_id,
            session_id=None,
            correlation_id=correlation_id,
            resource=f"database.{table_name}",
            action=operation.lower(),
            outcome=outcome,
            details=details
        )
        
        self._log_audit_event(event)
    
    def log_authentication_event(self,
                                user_id: Optional[str],
                                client_id: Optional[str],
                                auth_method: str,
                                outcome: str,
                                source_ip: Optional[str] = None,
                                user_agent: Optional[str] = None,
                                failure_reason: Optional[str] = None):
        """
        Log authentication event.
        
        Args:
            user_id: User attempting authentication
            client_id: Client identifier
            auth_method: Authentication method used
            outcome: Authentication outcome (success, failure, denied)
            source_ip: Source IP address
            user_agent: User agent string
            failure_reason: Reason for authentication failure
        """
        severity = AuditSeverity.MEDIUM if outcome == "success" else AuditSeverity.HIGH
        
        details = {
            "auth_method": auth_method
        }
        
        if failure_reason:
            details["failure_reason"] = failure_reason
        
        event = AuditEvent(
            event_id="",
            timestamp="",
            event_type=AuditEventType.AUTHENTICATION,
            severity=severity,
            user_id=user_id,
            client_id=client_id,
            session_id=None,
            correlation_id=None,
            resource="authentication",
            action="authenticate",
            outcome=outcome,
            details=details,
            source_ip=source_ip,
            user_agent=user_agent
        )
        
        self._log_audit_event(event)
    
    def log_event_publish(self,
                         client_id: Optional[str],
                         correlation_id: Optional[str],
                         event_type: str,
                         topic: str,
                         outcome: str,
                         rate_limited: bool = False,
                         error_message: Optional[str] = None):
        """
        Log event publishing activity.
        
        Args:
            client_id: Client publishing the event
            correlation_id: Event correlation ID
            event_type: Type of event being published
            topic: RedPanda topic
            outcome: Publishing outcome (success, failure, denied)
            rate_limited: Whether request was rate limited
            error_message: Error message if publishing failed
        """
        severity = AuditSeverity.LOW
        if rate_limited:
            severity = AuditSeverity.MEDIUM
        elif outcome != "success":
            severity = AuditSeverity.MEDIUM
        
        details = {
            "event_type": event_type,
            "topic": topic,
            "rate_limited": rate_limited
        }
        
        if error_message:
            details["error_message"] = error_message
        
        event = AuditEvent(
            event_id="",
            timestamp="",
            event_type=AuditEventType.EVENT_PUBLISH,
            severity=severity,
            user_id=None,
            client_id=client_id,
            session_id=None,
            correlation_id=correlation_id,
            resource=f"event_bus.{topic}",
            action="publish",
            outcome=outcome,
            details=details
        )
        
        self._log_audit_event(event)
    
    def log_security_violation(self,
                              client_id: Optional[str],
                              violation_type: str,
                              description: str,
                              source_ip: Optional[str] = None,
                              details: Optional[Dict[str, Any]] = None):
        """
        Log security violation event.
        
        Args:
            client_id: Client involved in violation
            violation_type: Type of security violation
            description: Description of the violation
            source_ip: Source IP address
            details: Additional violation details
        """
        event_details = {
            "violation_type": violation_type,
            "description": description
        }
        
        if details:
            event_details.update(details)
        
        event = AuditEvent(
            event_id="",
            timestamp="",
            event_type=AuditEventType.SECURITY_VIOLATION,
            severity=AuditSeverity.CRITICAL,
            user_id=None,
            client_id=client_id,
            session_id=None,
            correlation_id=None,
            resource="security",
            action="violation",
            outcome="detected",
            details=event_details,
            source_ip=source_ip
        )
        
        self._log_audit_event(event)
    
    def _log_audit_event(self, event: AuditEvent):
        """
        Log audit event with integrity checking.
        
        Args:
            event: Audit event to log
        """
        if not self._enabled:
            return
        
        # Check minimum severity threshold
        severity_levels = {
            AuditSeverity.LOW: 1,
            AuditSeverity.MEDIUM: 2,
            AuditSeverity.HIGH: 3,
            AuditSeverity.CRITICAL: 4
        }
        
        if severity_levels[event.severity] < severity_levels[self._min_severity]:
            return
        
        # Add sequence number and chain hash for integrity
        self._sequence_number += 1
        event.metadata = {
            "sequence_number": self._sequence_number,
            "previous_hash": self._last_hash,
            "integrity_hash": event.get_integrity_hash()
        }
        
        # Update chain hash
        self._last_hash = event.get_integrity_hash()
        
        # Log the event
        self._logger.info(event.to_json())
        
        # Send alerts for high-severity events
        if severity_levels[event.severity] >= severity_levels[self._alert_threshold]:
            self._send_security_alert(event)
    
    def _send_security_alert(self, event: AuditEvent):
        """
        Send real-time security alert for high-severity events.
        
        Args:
            event: High-severity audit event
        """
        # TODO: Implement integration with alerting systems
        # (Slack, PagerDuty, email, etc.)
        self._logger.critical(f"SECURITY ALERT: {event.event_type.value} - {event.action} - {event.outcome}")
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """
        Get audit logging statistics.
        
        Returns:
            Dictionary with audit statistics
        """
        return {
            "enabled": self._enabled,
            "min_severity": self._min_severity.value,
            "alert_threshold": self._alert_threshold.value,
            "sequence_number": self._sequence_number,
            "last_hash": self._last_hash[-8:] if self._last_hash else None  # Show last 8 chars
        }
    
    def verify_integrity(self, events: List[AuditEvent]) -> bool:
        """
        Verify integrity of audit event chain.
        
        Args:
            events: List of audit events to verify
            
        Returns:
            True if chain integrity is valid
        """
        if not events:
            return True
        
        previous_hash = ""
        
        for event in sorted(events, key=lambda e: e.metadata.get("sequence_number", 0)):
            if not event.metadata:
                continue
            
            # Verify hash chain
            if event.metadata.get("previous_hash") != previous_hash:
                return False
            
            # Verify event integrity
            expected_hash = event.get_integrity_hash()
            actual_hash = event.metadata.get("integrity_hash")
            
            if expected_hash != actual_hash:
                return False
            
            previous_hash = actual_hash
        
        return True


# Global audit logger instance
_audit_logger: Optional[ONEXAuditLogger] = None


def get_audit_logger() -> ONEXAuditLogger:
    """
    Get global audit logger instance.
    
    Returns:
        ONEXAuditLogger singleton instance
    """
    global _audit_logger
    
    if _audit_logger is None:
        _audit_logger = ONEXAuditLogger()
    
    return _audit_logger