# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Protocol boundary for operator-injected RSD PostgreSQL capabilities."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_infra.testing.rsd_postgres_acceptance_capability import (
        ModelRsdPostgresAcceptanceEvidence,
    )

PostgresLifecycleConnectionFactory = Callable[[], AbstractContextManager[object]]


@dataclass(frozen=True, slots=True)
class RsdPostgresAcceptanceCapability:
    """Typed operator capability returned for an accepted opaque ref."""

    connection_factory: PostgresLifecycleConnectionFactory
    evidence: ModelRsdPostgresAcceptanceEvidence


CapabilityResolver = Callable[[str], RsdPostgresAcceptanceCapability]


__all__ = [
    "CapabilityResolver",
    "PostgresLifecycleConnectionFactory",
    "RsdPostgresAcceptanceCapability",
]
