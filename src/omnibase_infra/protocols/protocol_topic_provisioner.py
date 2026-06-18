# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Structural protocol for the per-contract boot interleave provisioner (OMN-13237).

The boot interleave (``subscribe_wired_contract_topics``) depends on this shape
rather than the concrete ``TopicProvisioner`` so the no-global-gather regression
test (W7) can substitute a fake provisioner that records call order, and so the
interleave stays decoupled from the admin client at module-parse time.

Related Tickets:
    - OMN-13237: Per-contract scoped topic provisioning at runtime boot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable
from uuid import UUID

from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.event_bus.model_topic_set_readiness import (
    ModelTopicSetReadiness,
)
from omnibase_infra.topics.model_topic_spec import ModelTopicSpec


@runtime_checkable
class ProtocolTopicProvisioner(Protocol):
    """Minimal provisioner shape the boot interleave needs.

    ``ensure_topic_exists`` is idempotent topic creation; ``confirm_topics_ready``
    is the deterministic metadata-readiness confirm (§3.7).
    """

    async def ensure_topic_exists(
        self,
        topic_name: str,
        spec: ModelTopicSpec | None = None,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Idempotently ensure a single topic exists; True if present/created."""
        ...

    async def confirm_topics_ready(
        self,
        topics: Sequence[str],
        *,
        expected_specs: Mapping[str, ModelTopicSpec] | None = None,
        config: ModelTopicReadinessConfig | None = None,
        correlation_id: UUID | None = None,
    ) -> ModelTopicSetReadiness:
        """Confirm broker metadata for ``topics`` converged (§3.7)."""
        ...


__all__: list[str] = ["ProtocolTopicProvisioner"]
