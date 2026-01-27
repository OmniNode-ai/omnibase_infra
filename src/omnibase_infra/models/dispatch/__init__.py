# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Dispatch Engine Models.

Core models for the ONEX runtime dispatch engine that routes messages
based on topic category and message type, and publishes dispatcher outputs.

This module provides:
- **ModelDispatchContext**: Dispatch context with time injection control
- **ModelDispatchRoute**: Routing rules that map topic patterns to dispatchers
- **ModelDispatchResult**: Results of dispatch operations with metrics
- **ModelDispatcherRegistration**: Dispatcher registration metadata
- **ModelDispatcherMetrics**: Per-dispatcher metrics
- **ModelDispatchMetrics**: Aggregate dispatch metrics
- **EnumDispatchStatus**: Status values for dispatch outcomes

Design Principles:
    - **Pure Domain Models**: No I/O dependencies, no infrastructure concerns
    - **Immutable**: All models are frozen (thread-safe after creation)
    - **Typed**: Strong typing with validation constraints
    - **Serializable**: Full JSON serialization support

Data Flow:
    ```
    +------------------------------------------------------------------+
    |                     Dispatch Engine Flow                          |
    +------------------------------------------------------------------+
    |                                                                  |
    |   Incoming Message      Route Matching       Dispatcher Execution |
    |        |                     |                      |            |
    |        |  (topic, category)  |                      |            |
    |        |-------------------->|                      |            |
    |        |                     |  ModelDispatchRoute  |            |
    |        |                     |--------------------->|            |
    |        |                     |                      |            |
    |        |                     |                      | execute    |
    |        |                     |                      |-------->   |
    |        |                     |                      |            |
    |        |                     |  ModelDispatchResult |            |
    |        |<--------------------|<---------------------|            |
    |                                                                  |
    +------------------------------------------------------------------+
    ```

Usage:
    >>> from omnibase_infra.models.dispatch import (
    ...     ModelDispatchRoute,
    ...     ModelDispatchResult,
    ...     ModelDispatcherRegistration,
    ...     EnumDispatchStatus,
    ... )
    >>> from omnibase_infra.enums import EnumMessageCategory
    >>> from omnibase_core.enums.enum_node_kind import EnumNodeKind
    >>> from uuid import uuid4
    >>> from datetime import datetime, UTC
    >>>
    >>> # Register a dispatcher
    >>> dispatcher = ModelDispatcherRegistration(
    ...     dispatcher_id="user-dispatcher",
    ...     dispatcher_name="User Event Dispatcher",
    ...     node_kind=EnumNodeKind.REDUCER,
    ...     supported_categories=[EnumMessageCategory.EVENT],
    ...     registered_at=datetime.now(UTC),
    ... )
    >>>
    >>> # Create a route
    >>> route = ModelDispatchRoute(
    ...     route_id="user-route",
    ...     topic_pattern="*.user.events.*",
    ...     message_category=EnumMessageCategory.EVENT,
    ...     dispatcher_id="user-dispatcher",
    ... )
    >>>
    >>> # Check if route matches
    >>> route.matches_topic("dev.user.events.v1")
    True
    >>>
    >>> # Create a dispatch result
    >>> # NOTE: Per ONEX guidelines, always include correlation_id (generate with uuid4() if not propagated)
    >>> result = ModelDispatchResult(
    ...     dispatch_id=uuid4(),
    ...     status=EnumDispatchStatus.SUCCESS,
    ...     topic="dev.user.events.v1",
    ...     route_id="user-route",
    ...     dispatcher_id="user-dispatcher",
    ...     started_at=datetime.now(UTC),
    ...     correlation_id=uuid4(),  # Always generate if not propagating from incoming message
    ... )

See Also:
    omnibase_infra.enums.EnumMessageCategory: Message category classification
    omnibase_core.enums.EnumExecutionShape: Valid execution patterns
    omnibase_core.models.events.ModelEventEnvelope: Event wrapper with routing info
"""

from omnibase_infra.enums import EnumDispatchStatus, EnumTopicStandard
from omnibase_infra.models.dispatch.model_debug_trace_snapshot import (
    ModelDebugTraceSnapshot,
)
from omnibase_infra.models.dispatch.model_dispatch_context import ModelDispatchContext
from omnibase_infra.models.dispatch.model_dispatch_error import ModelDispatchError
from omnibase_infra.models.dispatch.model_dispatch_log_context import (
    ModelDispatchLogContext,
)
from omnibase_infra.models.dispatch.model_dispatch_metadata import ModelDispatchMetadata
from omnibase_infra.models.dispatch.model_dispatch_metrics import ModelDispatchMetrics
from omnibase_infra.models.dispatch.model_dispatch_outcome import ModelDispatchOutcome
from omnibase_infra.models.dispatch.model_dispatch_outputs import ModelDispatchOutputs
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_infra.models.dispatch.model_dispatcher_metrics import (
    ModelDispatcherMetrics,
)
from omnibase_infra.models.dispatch.model_dispatcher_registration import (
    ModelDispatcherRegistration,
)
from omnibase_infra.models.dispatch.model_materialized_dispatch import (
    ModelMaterializedDispatch,
)
from omnibase_infra.models.dispatch.model_parsed_topic import ModelParsedTopic
from omnibase_infra.models.dispatch.model_topic_parser import (
    ModelTopicParser,
    TypeCacheInfo,
    clear_topic_parse_cache,
    get_topic_parse_cache_info,
)
from omnibase_infra.models.dispatch.model_tracing_context import ModelTracingContext

__all__ = [
    # Cache utilities
    "TypeCacheInfo",
    # Enums
    "EnumDispatchStatus",
    "EnumTopicStandard",
    # Models
    "ModelDebugTraceSnapshot",
    "ModelDispatchContext",
    "ModelDispatchError",
    "ModelDispatchLogContext",
    "ModelMaterializedDispatch",
    "ModelDispatchMetadata",
    "ModelDispatchMetrics",
    "ModelDispatchOutcome",
    "ModelDispatchOutputs",
    "ModelDispatchResult",
    "ModelDispatchRoute",
    "ModelDispatcherMetrics",
    "ModelDispatcherRegistration",
    "ModelParsedTopic",
    "ModelTopicParser",
    "ModelTracingContext",
    "clear_topic_parse_cache",
    "get_topic_parse_cache_info",
]
