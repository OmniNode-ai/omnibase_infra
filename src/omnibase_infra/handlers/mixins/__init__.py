# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler-specific mixins for HandlerConsul and other handlers.

These mixins encapsulate specific functionality domains to reduce class
complexity and improve maintainability.

Consul Mixins:
    - MixinConsulInitialization: Configuration parsing and client setup
    - MixinConsulKV: Key-value store operations (get, put)
    - MixinConsulService: Service registration operations (register, deregister)
    - MixinConsulTopicIndex: Topic index management for event bus routing
"""

from omnibase_infra.handlers.mixins.mixin_consul_initialization import (
    MixinConsulInitialization,
)
from omnibase_infra.handlers.mixins.mixin_consul_kv import MixinConsulKV
from omnibase_infra.handlers.mixins.mixin_consul_service import MixinConsulService
from omnibase_infra.handlers.mixins.mixin_consul_topic_index import (
    MixinConsulTopicIndex,
)

__all__: list[str] = [
    # Consul mixins
    "MixinConsulInitialization",
    "MixinConsulKV",
    "MixinConsulService",
    "MixinConsulTopicIndex",
]
