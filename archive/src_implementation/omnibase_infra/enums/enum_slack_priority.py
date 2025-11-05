"""
Slack Priority Enum for ONEX Infrastructure Alert Formatting.

This enum defines alert priority levels with corresponding Slack formatting
colors for the ONEX infrastructure system.
"""

from enum import Enum


class EnumSlackPriority(str, Enum):
    """Alert priority levels with corresponding Slack formatting."""

    CRITICAL = "danger"  # Red
    HIGH = "warning"     # Yellow
    MEDIUM = "good"      # Green
    INFO = "#36a64f"     # Custom green
