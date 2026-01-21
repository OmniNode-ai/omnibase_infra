# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Widget defaults model for dashboard configuration.

Related Tickets:
    - OMN-1278: Contract-Driven Dashboard - Registry Discovery
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelWidgetDefaults(BaseModel):
    """Default configuration for a widget type.

    Attributes vary by widget type and provide sensible defaults.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    # Common fields - all optional
    show_timestamp: bool | None = None
    max_items: int | None = None
    refresh_interval_seconds: int | None = None


__all__ = ["ModelWidgetDefaults"]
