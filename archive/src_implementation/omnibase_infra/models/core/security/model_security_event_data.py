"""DEPRECATED: This file contained multiple models and has been split.

This file is deprecated. The models have been moved to separate files:
- ModelSecurityEventDetails -> model_security_event_details.py
- ModelSecurityEventMetadata -> model_security_event_metadata.py
- ModelAuditLogEntry -> model_audit_log_entry.py

This file will be removed in a future version.
"""

# Re-export models for backwards compatibility
from .model_audit_log_entry import ModelAuditLogEntry
from .model_security_event_details import ModelSecurityEventDetails
from .model_security_event_metadata import ModelSecurityEventMetadata

__all__ = [
    "ModelAuditLogEntry",
    "ModelSecurityEventDetails",
    "ModelSecurityEventMetadata",
]
