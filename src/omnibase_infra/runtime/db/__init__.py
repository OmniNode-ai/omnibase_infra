# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Database runtime module for omnibase_infra.

This module provides the PostgresRepositoryRuntime and associated models for
database operations with safety constraints and deterministic query behavior.

Components:
    - PostgresRepositoryRuntime: Generic Postgres runtime for contract execution
    - ModelRepositoryRuntimeConfig: Configuration for runtime behavior
    - ModelDbRepositoryContract: Repository contract specification
    - ModelDbOperation: Database operation specification
    - ModelDbReturn: Return type specification

The db module enables safe, configurable database operations with:
    - Contract-driven operations (all ops defined in ModelDbRepositoryContract)
    - Operation allowlisting (select, insert, update, upsert)
    - Row limits for multi-row selects
    - Query timeouts with asyncio.wait_for()
    - Deterministic ORDER BY injection
    - Metrics emission

Example:
    >>> import asyncpg
    >>> from omnibase_infra.runtime.db import (
    ...     PostgresRepositoryRuntime,
    ...     ModelDbRepositoryContract,
    ...     ModelDbOperation,
    ...     ModelDbReturn,
    ...     ModelRepositoryRuntimeConfig,
    ... )
    >>>
    >>> # Define contract
    >>> contract = ModelDbRepositoryContract(
    ...     name="users",
    ...     database_ref="primary",
    ...     ops={
    ...         "find_by_id": ModelDbOperation(
    ...             mode="select",
    ...             sql="SELECT * FROM users WHERE id = $1",
    ...             params=["user_id"],
    ...             returns=ModelDbReturn(many=False),
    ...         ),
    ...     },
    ... )
    >>>
    >>> # Create runtime with pool
    >>> pool = await asyncpg.create_pool(...)
    >>> runtime = PostgresRepositoryRuntime(pool, contract)
    >>>
    >>> # Execute operation
    >>> user = await runtime.call("find_by_id", 123)
"""

from __future__ import annotations

from omnibase_infra.runtime.db.models import (
    ModelDbOperation,
    ModelDbParam,
    ModelDbRepositoryContract,
    ModelDbReturn,
    ModelDbSafetyPolicy,
    ModelRepositoryRuntimeConfig,
)
from omnibase_infra.runtime.db.postgres_repository_runtime import (
    PostgresRepositoryRuntime,
)

__all__: list[str] = [
    "ModelDbOperation",
    "ModelDbParam",
    "ModelDbRepositoryContract",
    "ModelDbReturn",
    "ModelDbSafetyPolicy",
    "ModelRepositoryRuntimeConfig",
    "PostgresRepositoryRuntime",
]
