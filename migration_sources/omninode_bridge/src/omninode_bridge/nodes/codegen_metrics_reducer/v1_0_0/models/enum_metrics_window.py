"""
Time Window Enumeration for Metrics Aggregation.

Defines aggregation windows for code generation metrics:
- HOURLY: 1-hour rolling windows
- DAILY: 24-hour rolling windows
- WEEKLY: 7-day rolling windows
- MONTHLY: 30-day rolling windows
"""

from enum import Enum


class EnumMetricsWindow(str, Enum):
    """Time window types for metrics aggregation."""

    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

    def get_window_seconds(self) -> int:
        """
        Get window size in seconds.

        Returns:
            Window size in seconds
        """
        return {
            self.HOURLY: 3600,  # 1 hour
            self.DAILY: 86400,  # 24 hours
            self.WEEKLY: 604800,  # 7 days
            self.MONTHLY: 2592000,  # 30 days
        }[self]

    def get_window_milliseconds(self) -> int:
        """
        Get window size in milliseconds.

        Returns:
            Window size in milliseconds
        """
        return self.get_window_seconds() * 1000
