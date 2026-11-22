# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The kernel's one ``TopicProvisioner``, as the auto-create pass uses it (OMN-26359)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol
from uuid import UUID

__all__ = ["ProtocolProvisionedTopicsEnsurer"]


class ProtocolProvisionedTopicsEnsurer(Protocol):
    """The kernel's one ``TopicProvisioner``, as the auto-create pass uses it."""

    async def ensure_provisioned_topics_exist(
        self, correlation_id: UUID | None = None
    ) -> Mapping[str, object]: ...
