# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Generic lease: etcd store, pure projection reducer, projection writer (OMN-21424)."""

from __future__ import annotations

from omnibase_infra.handlers.lease.handler_lease_orchestrator import (
    HandlerLeaseAcquire,
    HandlerLeaseRelease,
    HandlerLeaseRenew,
)
from omnibase_infra.handlers.lease.handler_lease_projection import (
    UPSERT_LEASE_PROJECTION_SQL,
    HandlerLeaseProjection,
    lease_projection_params,
)
from omnibase_infra.handlers.lease.handler_lease_reducer import (
    HandlerLeaseReducer,
    LeaseProjectionState,
    reduce_lease_event,
)
from omnibase_infra.handlers.lease.lease_fence import check_fence
from omnibase_infra.handlers.lease.lease_store import (
    LEASE_PREFIX,
    LeaseStore,
    lease_kinds_from_spec,
    lease_store_from_config,
    resource_prefix,
)

__all__ = [
    "LEASE_PREFIX",
    "UPSERT_LEASE_PROJECTION_SQL",
    "HandlerLeaseAcquire",
    "HandlerLeaseProjection",
    "HandlerLeaseReducer",
    "HandlerLeaseRelease",
    "HandlerLeaseRenew",
    "LeaseStore",
    "LeaseProjectionState",
    "check_fence",
    "lease_kinds_from_spec",
    "lease_projection_params",
    "lease_store_from_config",
    "reduce_lease_event",
    "resource_prefix",
]
