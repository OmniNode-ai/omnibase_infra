"""OmniNode Topic Class enumeration."""

from enum import Enum


class EnumOmniNodeTopicClass(str, Enum):
    """
    OmniNode Topic Classes for proper topic namespace organization.
    
    Following the OmniNode topic design:
    <env>.<tenant>.<context>.<class>.<topic>.<v>
    
    Topic classes define the type of content and usage patterns.
    """
    
    # Core event processing
    EVT = "evt"  # Events - State change notifications
    CMD = "cmd"  # Commands - Action requests
    QRS = "qrs"  # Query-Response - Request/response patterns
    
    # Control and management  
    CTL = "ctl"  # Control - Control plane operations
    RTY = "rty"  # Retry - Retry processing
    DLT = "dlt"  # Dead Letter - Failed messages
    
    # Data and monitoring
    CDC = "cdc"  # Change Data Capture - Database changes
    MET = "met"  # Metrics - Performance and operational metrics
    AUD = "aud"  # Audit - Audit trail and compliance
    LOG = "log"  # Logs - Application logging