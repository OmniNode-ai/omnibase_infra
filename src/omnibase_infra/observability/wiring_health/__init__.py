# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Wiring health monitoring for ONEX event bus.

This module provides infrastructure for detecting silent partial wiring failures
by comparing message emission counts to consumption counts on critical feedback
loop topics.

Architecture:
    - MixinEmissionCounter: Counts messages published to EventBus
    - MixinConsumptionCounter: Counts messages successfully consumed
    - WiringHealthChecker: Computes mismatch ratios and emits alerts

Design Decisions:
    - Prometheus scrape is the scheduler (no background loop)
    - Only monitors WIRING_HEALTH_MONITORED_TOPICS (bounded memory)
    - Checker + health endpoint pattern (ONEX-aligned primitives)
    - 5% mismatch threshold accounts for at-least-once delivery

See Also:
    - OMN-1895: Wiring health monitor implementation
    - topic_constants.py: WIRING_HEALTH_MONITORED_TOPICS
"""

from omnibase_infra.observability.wiring_health.mixin_consumption_counter import (
    MixinConsumptionCounter,
)
from omnibase_infra.observability.wiring_health.mixin_emission_counter import (
    MixinEmissionCounter,
)
from omnibase_infra.observability.wiring_health.model_topic_wiring_health import (
    DEFAULT_MISMATCH_THRESHOLD,
    ModelTopicWiringHealth,
)
from omnibase_infra.observability.wiring_health.model_wiring_health_alert import (
    ModelWiringHealthAlert,
)
from omnibase_infra.observability.wiring_health.model_wiring_health_metrics import (
    ModelWiringHealthMetrics,
)
from omnibase_infra.observability.wiring_health.protocol_consumption_count_source import (
    ProtocolConsumptionCountSource,
)
from omnibase_infra.observability.wiring_health.protocol_emission_count_source import (
    ProtocolEmissionCountSource,
)
from omnibase_infra.observability.wiring_health.wiring_health_checker import (
    WiringHealthChecker,
)

__all__ = [
    # Constants
    "DEFAULT_MISMATCH_THRESHOLD",
    # Checker
    "WiringHealthChecker",
    # Mixins
    "MixinConsumptionCounter",
    "MixinEmissionCounter",
    # Models
    "ModelTopicWiringHealth",
    "ModelWiringHealthAlert",
    "ModelWiringHealthMetrics",
    # Protocols
    "ProtocolConsumptionCountSource",
    "ProtocolEmissionCountSource",
]
