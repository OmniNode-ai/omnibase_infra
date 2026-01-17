# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Observability configuration models.

This module exports configuration models for observability sinks.

Note:
    ModelMetricsPolicy is a local implementation until omnibase_core
    releases OMN-1367. Once released, imports should switch to:
    - from omnibase_core.models.observability import ModelMetricsPolicy
    - from omnibase_core.enums import EnumMetricsPolicyViolationAction
"""

from omnibase_infra.observability.models.model_logging_sink_config import (
    ModelLoggingSinkConfig,
)
from omnibase_infra.observability.models.model_metrics_policy import (
    EnumMetricsPolicyViolationAction,
    ModelMetricsPolicy,
)
from omnibase_infra.observability.models.model_metrics_sink_config import (
    ModelMetricsSinkConfig,
)

__all__: list[str] = [
    "EnumMetricsPolicyViolationAction",
    "ModelLoggingSinkConfig",
    "ModelMetricsPolicy",
    "ModelMetricsSinkConfig",
]
