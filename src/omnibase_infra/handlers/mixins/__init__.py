# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler-specific mixins for HandlerConsul, HandlerVault, and other handlers.

These mixins encapsulate specific functionality domains to reduce class
complexity and improve maintainability.

Consul Mixins:
    - MixinConsulInitialization: Configuration parsing and client setup
    - MixinConsulKV: Key-value store operations (get, put)
    - MixinConsulService: Service registration operations (register, deregister)
    - MixinConsulTopicIndex: Topic index management for event bus routing

Vault Mixins:
    - MixinVaultInitialization: Configuration parsing and client setup
    - MixinVaultRetry: Retry logic with exponential backoff
    - MixinVaultSecrets: CRUD operations for secrets
    - MixinVaultToken: Token management and renewal
"""

from omnibase_infra.handlers.mixins.mixin_consul_initialization import (
    MixinConsulInitialization,
)
from omnibase_infra.handlers.mixins.mixin_consul_kv import MixinConsulKV
from omnibase_infra.handlers.mixins.mixin_consul_service import MixinConsulService
from omnibase_infra.handlers.mixins.mixin_consul_topic_index import (
    MixinConsulTopicIndex,
)
from omnibase_infra.handlers.mixins.mixin_vault_initialization import (
    MixinVaultInitialization,
)
from omnibase_infra.handlers.mixins.mixin_vault_retry import MixinVaultRetry
from omnibase_infra.handlers.mixins.mixin_vault_secrets import MixinVaultSecrets
from omnibase_infra.handlers.mixins.mixin_vault_token import MixinVaultToken

__all__: list[str] = [
    # Consul mixins
    "MixinConsulInitialization",
    "MixinConsulKV",
    "MixinConsulService",
    "MixinConsulTopicIndex",
    # Vault mixins
    "MixinVaultInitialization",
    "MixinVaultRetry",
    "MixinVaultSecrets",
    "MixinVaultToken",
]
