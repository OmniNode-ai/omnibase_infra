# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""TypedDict definitions for omnibase_infra.

This package contains TypedDict definitions that represent JSON-serialized
forms of Pydantic models, enabling proper type checking for cache operations
and JSON serialization/deserialization without requiring type: ignore comments.

Available TypedDicts:
    - TypedDictEnvelopeBuildParams: Parameters for building ModelEventEnvelope
    - TypedDictIntrospectionCache: JSON-serialized ModelNodeIntrospectionEvent
    - TypedDictPerformanceMetricsCache: JSON-serialized introspection performance metrics
"""

from omnibase_infra.types.typed_dict.typed_dict_envelope_build_params import (
    TypedDictEnvelopeBuildParams,
)
from omnibase_infra.types.typed_dict.typed_dict_introspection_cache import (
    TypedDictIntrospectionCache,
)
from omnibase_infra.types.typed_dict.typed_dict_performance_metrics_cache import (
    TypedDictPerformanceMetricsCache,
)

__all__ = [
    "TypedDictEnvelopeBuildParams",
    "TypedDictIntrospectionCache",
    "TypedDictPerformanceMetricsCache",
]
