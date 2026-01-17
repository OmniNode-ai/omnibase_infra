# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Observability configuration models.

This module exports configuration models for observability sinks.
"""

from omnibase_infra.observability.models.model_logging_sink_config import (
    ModelLoggingSinkConfig,
)
from omnibase_infra.observability.models.model_metrics_sink_config import (
    ModelMetricsSinkConfig,
)

__all__: list[str] = [
    "ModelLoggingSinkConfig",
    "ModelMetricsSinkConfig",
]
