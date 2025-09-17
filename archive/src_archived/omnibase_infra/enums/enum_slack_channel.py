"""
Slack Channel Enum for ONEX Infrastructure Notifications.

This enum defines production Slack channels for different notification types
in the ONEX infrastructure system.
"""

from enum import Enum


class EnumSlackChannel(str, Enum):
    """Production Slack channels for different notification types."""

    ALERTS = "#infrastructure-alerts"
    GENERAL = "#dev-general"
    CRITICAL = "#critical-alerts"
    MONITORING = "#infrastructure-monitoring"
    DEPLOYMENTS = "#deployments"
    SECURITY = "#security-alerts"
