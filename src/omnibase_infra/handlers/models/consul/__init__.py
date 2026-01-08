# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul Handler Models Package.

This package provides strongly-typed Pydantic models for Consul handler response payloads.
Each Consul operation has a dedicated payload model with a discriminator field for
type-safe union handling.

Payload Types:
    - ModelConsulKVGetFoundPayload: Single key found in KV store
    - ModelConsulKVGetNotFoundPayload: Key not found in KV store
    - ModelConsulKVGetRecursePayload: Recursive key listing from KV store
    - ModelConsulKVPutPayload: KV put operation result
    - ModelConsulRegisterPayload: Service registration result
    - ModelConsulDeregisterPayload: Service deregistration result

Union Type:
    ConsulPayload: Discriminated union of all payload types using operation_type field

Registry-Based Alternative (OMN-1007):
    - ModelPayloadConsul: Base model for all Consul handler payloads
    - RegistryPayloadConsul: Decorator-based registry for payload type discovery
"""

from omnibase_infra.handlers.models.consul.enum_consul_operation_type import (
    EnumConsulOperationType,
)
from omnibase_infra.handlers.models.consul.model_consul_deregister_payload import (
    ModelConsulDeregisterPayload,
)
from omnibase_infra.handlers.models.consul.model_consul_handler_config import (
    ModelConsulHandlerConfig,
)
from omnibase_infra.handlers.models.consul.model_consul_handler_payload import (
    ConsulPayload,
    ModelConsulHandlerPayload,
)
from omnibase_infra.handlers.models.consul.model_consul_kv_get_found_payload import (
    ModelConsulKVGetFoundPayload,
)
from omnibase_infra.handlers.models.consul.model_consul_kv_get_not_found_payload import (
    ModelConsulKVGetNotFoundPayload,
)
from omnibase_infra.handlers.models.consul.model_consul_kv_get_recurse_payload import (
    ModelConsulKVGetRecursePayload,
)
from omnibase_infra.handlers.models.consul.model_consul_kv_item import (
    ModelConsulKVItem,
)
from omnibase_infra.handlers.models.consul.model_consul_kv_put_payload import (
    ModelConsulKVPutPayload,
)
from omnibase_infra.handlers.models.consul.model_consul_register_payload import (
    ModelConsulRegisterPayload,
)
from omnibase_infra.handlers.models.consul.model_consul_retry_config import (
    ModelConsulRetryConfig,
)
from omnibase_infra.handlers.models.consul.model_payload_consul import (
    ModelPayloadConsul,
)
from omnibase_infra.handlers.models.consul.registry_payload_consul import (
    RegistryPayloadConsul,
)

__all__: list[str] = [
    "ConsulPayload",
    "EnumConsulOperationType",
    "ModelConsulDeregisterPayload",
    "ModelConsulHandlerConfig",
    "ModelConsulHandlerPayload",
    "ModelConsulKVGetFoundPayload",
    "ModelConsulKVGetNotFoundPayload",
    "ModelConsulKVGetRecursePayload",
    "ModelConsulKVItem",
    "ModelConsulKVPutPayload",
    "ModelConsulRegisterPayload",
    "ModelConsulRetryConfig",
    "ModelPayloadConsul",
    "RegistryPayloadConsul",
]
