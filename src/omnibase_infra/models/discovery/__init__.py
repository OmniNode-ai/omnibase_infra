# Copyright 2025 OmniNode Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Discovery models for node introspection and capability reporting.

Note:
    ModelNodeIntrospectionEvent has been moved to omnibase_infra.models.registration.
    Import it from there for new code.
"""

from omnibase_infra.models.discovery.model_dependency_spec import ModelDependencySpec
from omnibase_infra.models.discovery.model_discovered_capabilities import (
    ModelDiscoveredCapabilities,
)
from omnibase_infra.models.discovery.model_introspection_config import (
    DEFAULT_HEARTBEAT_TOPIC,
    DEFAULT_INTROSPECTION_TOPIC,
    DEFAULT_REQUEST_INTROSPECTION_TOPIC,
    INVALID_TOPIC_CHARS,
    TOPIC_PATTERN,
    VERSION_SUFFIX_PATTERN,
    ModelIntrospectionConfig,
)
from omnibase_infra.models.discovery.model_introspection_performance_metrics import (
    ModelIntrospectionPerformanceMetrics,
)
from omnibase_infra.models.discovery.model_introspection_task_config import (
    ModelIntrospectionTaskConfig,
)

__all__ = [
    "DEFAULT_HEARTBEAT_TOPIC",
    "DEFAULT_INTROSPECTION_TOPIC",
    "DEFAULT_REQUEST_INTROSPECTION_TOPIC",
    "INVALID_TOPIC_CHARS",
    "TOPIC_PATTERN",
    "VERSION_SUFFIX_PATTERN",
    "ModelDependencySpec",
    "ModelDiscoveredCapabilities",
    "ModelIntrospectionConfig",
    "ModelIntrospectionPerformanceMetrics",
    "ModelIntrospectionTaskConfig",
]
