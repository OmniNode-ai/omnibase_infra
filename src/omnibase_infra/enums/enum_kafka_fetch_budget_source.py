# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Source of the memory limit the aggregate consumer fetch budget divides.

OMN-17888. The aggregate Kafka consumer fetch bound is computed from a memory
limit, not chosen. This enum names *where that limit comes from* so the answer
is a declared decision rather than an inferred one.

There is deliberately no ``UNBOUNDED`` member: an unbounded aggregate fetch is
the defect this budget exists to remove, and adding it back as an enum value
would make the regression a one-word config edit.
"""

from __future__ import annotations

from enum import Enum


class EnumKafkaFetchBudgetSource(str, Enum):
    """Where ``ModelKafkaConsumerFetchBudget`` reads its memory limit from.

    Members:
        CONTAINER_CGROUP_LIMIT: Read the live cgroup v2 memory limit
            (``/sys/fs/cgroup/memory.max``) at event-bus construction. This is
            the deployment source: the number is the same one the container
            runtime enforces, so the budget cannot drift from the limit that
            actually kills the process. Fails fast when the file is absent or
            reads ``max`` (unconstrained) -- an unresolvable limit is never
            replaced with an assumed one.
        DECLARED_BYTES: Use the explicit ``memory_limit_bytes`` on the budget.
            For tests, which must inject the limit rather than read the host's
            (a suite whose verdict depends on where it runs proves nothing),
            and for hosts whose limit is known but not exposed through cgroup
            v2.
    """

    CONTAINER_CGROUP_LIMIT = "container_cgroup_limit"
    DECLARED_BYTES = "declared_bytes"
